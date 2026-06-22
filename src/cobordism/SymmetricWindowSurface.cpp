// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SymmetricWindowSurface.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace tessera::cobordism {

namespace {

using Face = SymmetricWindowSurface::Face;

// The 20 faces of the regular icosahedron (12 vertices) — the genus-0 (S^2) seed.
std::vector<Face> icosahedron() {
  std::vector<Face> faces = {
      {0, 1, 2},   {0, 2, 3},   {0, 3, 4},  {0, 4, 5},  {0, 5, 1},
      {1, 5, 10},  {1, 10, 6},  {1, 6, 2},  {2, 6, 7},  {2, 7, 3},
      {3, 7, 8},   {3, 8, 4},   {4, 8, 9},  {4, 9, 5},  {5, 9, 10},
      {6, 10, 11}, {7, 6, 11},  {8, 7, 11}, {9, 8, 11}, {10, 9, 11}};
  for (auto &f : faces) std::sort(f.begin(), f.end());
  return faces;
}

using EdgeMap =
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::uint64_t>>;

// A frequency-N geodesic icosahedron and its per-edge sub-vertex map.
struct GeodesicSphere {
  std::vector<Face> faces;  // 20*N^2 sub-triangles (sorted)
  int frequency;            // N
  EdgeMap edgePts;          // undirected icosa edge (min,max) -> N-1 ids from min
};

// The step-th sub-vertex (step = 1..N-1) along the icosa edge from u toward v.
std::uint64_t edgePoint(const EdgeMap &edgePts, int N, std::uint64_t u,
                        std::uint64_t v, int step) {
  const std::pair<std::uint64_t, std::uint64_t> e{std::min(u, v), std::max(u, v)};
  const int idx = (u == e.first) ? step - 1 : N - step - 1;
  return edgePts.at(e)[static_cast<std::size_t>(idx)];
}

// Frequency-N geodesic subdivision of the icosahedron: each icosa edge gets N-1
// sub-vertices and each face is split into N^2 sub-triangles, giving a connected
// S^2 with 12 + 30(N-1) + 20*(N-1)(N-2)/2 vertices and 20*N^2 faces. N=2 is the
// 42-vertex/80-face base of #398 (enough to host the 12 vertex-disjoint hole
// triangles); larger N refines the lattice (the tunable granularity, #404), which
// shrinks the discretization residual and drives the singlet overlap -> 1. Uses a
// face-order numbering (edge blocks on first encounter, then the face's interior),
// so N=2 reproduces geodesicTwoSphere exactly (backward-compatible). Deterministic.
// The edge map is returned so the window generator lifts the C3 rotations across the
// subdivision.
GeodesicSphere geodesicSphere(int N) {
  const auto ico = icosahedron();
  EdgeMap edgePts;
  std::uint64_t next = 12;
  auto ensureEdge = [&](std::uint64_t u, std::uint64_t v) {
    const std::pair<std::uint64_t, std::uint64_t> e{std::min(u, v), std::max(u, v)};
    if (edgePts.count(e)) return;
    std::vector<std::uint64_t> ids(static_cast<std::size_t>(N - 1));
    for (auto &id : ids) id = next++;
    edgePts.emplace(e, std::move(ids));
  };
  // Per-face interior points (barycentric i,j,k all > 0), keyed by (j,k); assigned
  // in face order AFTER that face's edges (the first-encounter scheme above).
  std::map<Face, std::map<std::pair<int, int>, std::uint64_t>> interior;
  for (const auto &f : ico) {
    ensureEdge(f[0], f[1]);
    ensureEdge(f[1], f[2]);
    ensureEdge(f[2], f[0]);
    std::map<std::pair<int, int>, std::uint64_t> in;
    for (int j = 1; j < N; ++j)
      for (int k = 1; k < N - j; ++k) in[{j, k}] = next++;
    interior[f] = std::move(in);
  }
  // P(j,k) in face (a,b,c): barycentric i=N-j-k toward a, j toward b, k toward c.
  auto gridId = [&](const Face &f, int j, int k) -> std::uint64_t {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const int i = N - j - k;
    if (i == N) return a;
    if (j == N) return b;
    if (k == N) return c;
    if (k == 0) return edgePoint(edgePts, N, a, b, j);  // edge a-b
    if (j == 0) return edgePoint(edgePts, N, a, c, k);  // edge a-c
    if (i == 0) return edgePoint(edgePts, N, b, c, k);  // edge b-c
    return interior.at(f).at({j, k});
  };
  std::vector<Face> faces;
  faces.reserve(ico.size() * static_cast<std::size_t>(N) * N);
  for (const auto &f : ico)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N - j; ++k) {
        Face up = {gridId(f, j, k), gridId(f, j + 1, k), gridId(f, j, k + 1)};
        std::sort(up.begin(), up.end());
        faces.push_back(std::move(up));
        if (j + k < N - 1) {
          Face dn = {gridId(f, j + 1, k), gridId(f, j, k + 1),
                     gridId(f, j + 1, k + 1)};
          std::sort(dn.begin(), dn.end());
          faces.push_back(std::move(dn));
        }
      }
  return {std::move(faces), N, std::move(edgePts)};
}

// The four A4-tetrahedral, C3-symmetric register windows (#398) — one orbit of a
// tetrahedral subgroup A4 < icosahedral rotation group. Each window is a C3 orbit
// of three vertex-disjoint corner sub-triangles seated at one of the icosahedron's
// four tetrahedral vertex-orbits ({2,8,10},{1,4,7},{0,6,9},{3,5,11}, which
// partition all 12 original vertices); the windows are A4-equivalent, so the
// per-window period-transport blocks are cyclically related and a color-symmetric
// (omega-representation) input transports to the EXACT singlet with manifest S3 —
// unlike a greedy pick whose windows are geometrically inequivalent.
//
// Generated FROM the symmetry (four C3 generators + a seed corner per window + the
// sub-vertex lift), so it is correct at any frequency N and in whatever numbering
// geodesicSphere() produces (hardcoding the triples is fragile against it). The seed
// corner sub-triangle at vertex v is {v, the nearest sub-vertex toward each of two
// neighbours}; the C3 rotation lifts it across the subdivision (a base vertex maps by
// the permutation; a sub-vertex on edge (p,q) at step s maps to step s on edge
// (perm[p],perm[q])). Asserts: each window a genuine C3 orbit, all 12 holes real
// faces, all 36 vertices distinct.
std::vector<std::vector<Face>> symmetricWindows(const EdgeMap &edgePts, int N,
                                                const std::set<Face> &faceSet) {
  // The four C3 rotations as 12-vertex permutations: a[w] cycles window w's
  // tetrahedral vertex-orbit (all order 3, orientation-preserving rotations).
  static const std::array<std::array<int, 12>, 4> a = {{
      {{4, 3, 8, 9, 5, 0, 7, 11, 10, 1, 2, 6}},   // A: 2->8->10
      {{3, 4, 0, 2, 7, 8, 5, 1, 6, 11, 9, 10}},   // B: 1->4->7
      {{6, 10, 11, 7, 2, 1, 9, 8, 3, 0, 5, 4}},   // C: 0->6->9
      {{10, 6, 1, 5, 9, 11, 2, 0, 4, 8, 7, 3}},   // R: 3->5->11
  }};
  // The seed corner sub-triangle per window: (vertex v, two icosa neighbours).
  static const std::array<std::array<std::uint64_t, 3>, 4> seed = {{
      {{2, 0, 1}}, {{1, 6, 10}}, {{0, 3, 4}}, {{3, 2, 7}},
  }};
  // Reverse sub-vertex lookup: id -> (its icosa edge, its step index from the edge's
  // min endpoint), so a vertex permutation lifts onto the subdivision.
  std::map<std::uint64_t, std::pair<std::pair<std::uint64_t, std::uint64_t>, int>> rev;
  for (const auto &kv : edgePts)
    for (std::size_t idx = 0; idx < kv.second.size(); ++idx)
      rev[kv.second[idx]] = {kv.first, static_cast<int>(idx)};
  auto apply = [&](const std::array<int, 12> &p, Face h) {
    for (auto &v : h) {
      if (v < 12) {
        v = static_cast<std::uint64_t>(p[v]);
      } else {
        const auto &er = rev.at(v);  // ((edge min, edge max), step-1)
        v = edgePoint(edgePts, N, static_cast<std::uint64_t>(p[er.first.first]),
                      static_cast<std::uint64_t>(p[er.first.second]),
                      er.second + 1);
      }
    }
    std::sort(h.begin(), h.end());
    return h;
  };

  std::vector<std::vector<Face>> windows(4);
  std::set<std::uint64_t> used;
  for (int w = 0; w < 4; ++w) {
    Face s = {seed[w][0], edgePoint(edgePts, N, seed[w][0], seed[w][1], 1),
              edgePoint(edgePts, N, seed[w][0], seed[w][2], 1)};
    std::sort(s.begin(), s.end());
    const Face h1 = apply(a[w], s);
    const Face h2 = apply(a[w], h1);
    if (apply(a[w], h2) != s)
      throw std::runtime_error(
          "SymmetricWindowSurface: symmetric window is not a C3 orbit");
    for (const Face &h : {s, h1, h2}) {
      if (!faceSet.count(h))
        throw std::runtime_error(
            "SymmetricWindowSurface: symmetric hole is not a base face");
      for (const auto v : h)
        if (!used.insert(v).second)
          throw std::runtime_error(
              "SymmetricWindowSurface: symmetric windows are not "
              "vertex-disjoint");
      windows[w].push_back(h);
    }
  }
  return windows;  // 4 windows x 3 holes; 36 distinct vertices (asserted above)
}

}  // namespace

SymmetricWindowSurface::Surface SymmetricWindowSurface::build(int frequency) {
  if (frequency < 2)
    throw std::invalid_argument(
        "SymmetricWindowSurface: frequency must be >= 2 (N=2 is the base that "
        "hosts the 12 disjoint holes; larger N refines the lattice)");
  const auto sphere = geodesicSphere(frequency);
  const std::set<Face> faceSet(sphere.faces.begin(), sphere.faces.end());
  auto windows = symmetricWindows(sphere.edgePts, frequency, faceSet);
  return Surface{sphere.faces, std::move(windows)};
}

}  // namespace tessera::cobordism
