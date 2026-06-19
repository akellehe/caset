// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/RegisterTopology.h"

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <stdexcept>

#include "cobordism/EigenstateSynthesis.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

namespace {

using Face = std::vector<std::uint64_t>;

// The 20 faces of the regular icosahedron (12 vertices), the genus-0 (S^2)
// register seed (the same seed the #353 register uses).
std::vector<Face> icosahedron() {
  std::vector<Face> faces = {
      {0, 1, 2},   {0, 2, 3},   {0, 3, 4},  {0, 4, 5},  {0, 5, 1},
      {1, 5, 10},  {1, 10, 6},  {1, 6, 2},  {2, 6, 7},  {2, 7, 3},
      {3, 7, 8},   {3, 8, 4},   {4, 8, 9},  {4, 9, 5},  {5, 9, 10},
      {6, 10, 11}, {7, 6, 11},  {8, 7, 11}, {9, 8, 11}, {10, 9, 11}};
  for (auto &f : faces) std::sort(f.begin(), f.end());
  return faces;
}

// One geodesic (1 -> 4) subdivision of a triangulated surface: each triangle
// splits into four on its edge midpoints (fresh midpoint ids after the max).
// Gives the level-1 register seed (42 vertices) — room for three color holes
// plus the connection ports as pairwise vertex-disjoint faces.
std::vector<Face> subdivideFaces(const std::vector<Face> &faces) {
  std::uint64_t nxt = 0;
  for (const auto &f : faces)
    for (const auto v : f) nxt = std::max(nxt, v + 1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> mid;
  auto m = [&](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
    const auto key = std::make_pair(std::min(a, b), std::max(a, b));
    const auto it = mid.find(key);
    if (it != mid.end()) return it->second;
    const std::uint64_t id = nxt++;
    mid[key] = id;
    return id;
  };
  std::vector<Face> out;
  for (const auto &f : faces) {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const std::uint64_t ab = m(a, b), bc = m(b, c), ca = m(c, a);
    std::vector<Face> tris = {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}};
    for (auto &s : tris) {
      std::sort(s.begin(), s.end());
      out.push_back(s);
    }
  }
  return out;
}

// The first k pairwise vertex-disjoint faces of `faces`.
std::vector<Face> disjointHoles(const std::vector<Face> &faces, std::size_t k) {
  std::vector<Face> holes;
  std::set<std::uint64_t> used;
  for (const auto &f : faces) {
    if (holes.size() >= k) break;
    bool overlap = false;
    for (const auto v : f)
      if (used.count(v)) { overlap = true; break; }
    if (!overlap) {
      holes.push_back(f);
      for (const auto v : f) used.insert(v);
    }
  }
  return holes;
}

// Shift every vertex id of a face list by `off` (a distinct block).
std::vector<Face> shift(const std::vector<Face> &faces, std::uint64_t off) {
  std::vector<Face> out;
  out.reserve(faces.size());
  for (const auto &f : faces) {
    Face g;
    g.reserve(f.size());
    for (const auto v : f) g.push_back(v + off);
    std::sort(g.begin(), g.end());
    out.push_back(std::move(g));
  }
  return out;
}

// The six lateral triangles of the triangular prism between sorted triangles
// t0=(a,b,c) and t1=(a',b',c'), corresponded by sorted order. This ADDS
// triangles to join two hole-circles into a tube — it never identifies the two
// blocks' cells, so no weld is created.
std::vector<Face> tube(const Face &t0, const Face &t1) {
  const std::uint64_t a = t0[0], b = t0[1], c = t0[2];
  const std::uint64_t a2 = t1[0], b2 = t1[1], c2 = t1[2];
  const std::uint64_t quads[3][4] = {
      {a, b, b2, a2}, {b, c, c2, b2}, {c, a, a2, c2}};
  std::vector<Face> out;
  for (const auto &q : quads) {
    Face t = {q[0], q[1], q[2]};
    std::sort(t.begin(), t.end());
    out.push_back(t);
    Face u = {q[0], q[2], q[3]};
    std::sort(u.begin(), u.end());
    out.push_back(u);
  }
  return out;
}

}  // namespace

void RegisterTopology::validateStateDim(std::size_t d) const {
  if (d != 3)
    throw std::invalid_argument(
        "MergeCobordism (register topology): state dimension must be 3 (the "
        "color triple on the Sigma=0 hyperplane)");
}

std::shared_ptr<Spacetime> RegisterTopology::build(
    std::size_t /*stateDim*/, std::uint64_t seed,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  // Three register blocks A, B, R (one per state: two inputs + one result), each
  // a holed icosahedron, with disjoint vertex ids; input blocks join the result
  // block by additive tubes (A->R, B->R). The level-1 subdivided icosahedron has
  // room for three color holes + the ports as vertex-disjoint faces.
  const auto seedFaces = subdivideFaces(icosahedron());
  std::uint64_t stride = 0;
  for (const auto &f : seedFaces)
    for (const auto v : f) stride = std::max(stride, v + 1);

  constexpr std::size_t kBlocks = 3;        // A, B, R
  constexpr std::size_t kResult = 2;        // R is the third block
  std::vector<std::vector<Face>> ports(kBlocks);
  std::vector<std::set<Face>> omitted(kBlocks);  // holes + ports (not laid)
  blockHoles_.assign(kBlocks, {});
  for (std::size_t b = 0; b < kBlocks; ++b) {
    const std::size_t nPorts = (b == kResult) ? 2 : 1;  // R joins both inputs
    const std::size_t need = 3 + nPorts;
    const auto blockFaces = shift(seedFaces, static_cast<std::uint64_t>(b) * stride);
    const auto picks = disjointHoles(blockFaces, need);
    if (picks.size() < need)
      throw std::runtime_error(
          "RegisterTopology: register seed lacks enough vertex-disjoint faces "
          "for the color holes and connection ports");
    for (std::size_t i = 0; i < 3; ++i) blockHoles_[b].push_back(picks[i]);
    for (std::size_t i = 3; i < need; ++i) ports[b].push_back(picks[i]);
    for (const auto &f : picks) omitted[b].insert(f);
  }

  // Assemble: each block's faces (minus its holes and ports) + one tube per
  // input->result connection.
  std::set<Face> cellSet;
  for (std::size_t b = 0; b < kBlocks; ++b) {
    const auto blockFaces = shift(seedFaces, static_cast<std::uint64_t>(b) * stride);
    for (const auto &f : blockFaces)
      if (!omitted[b].count(f)) cellSet.insert(f);
  }
  for (const auto &t : tube(ports[0][0], ports[kResult][0])) cellSet.insert(t);
  for (const auto &t : tube(ports[1][0], ports[kResult][1])) cellSet.insert(t);

  // === manifold gate: no weld (every edge in <= 2 triangles) ===
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> edgeCoface;
  for (const auto &c : cellSet)
    for (std::size_t i = 0; i < c.size(); ++i)
      for (std::size_t j = i + 1; j < c.size(); ++j)
        ++edgeCoface[{std::min(c[i], c[j]), std::max(c[i], c[j])}];
  for (const auto &kv : edgeCoface)
    if (kv.second > 2)
      throw std::runtime_error(
          "RegisterTopology: non-manifold seed (an edge has > 2 cofaces) — the "
          "construction welded two blocks");

  std::vector<std::vector<std::uint64_t>> cellList(cellSet.begin(), cellSet.end());
  auto cobordism = Spacetime::fromCells(2, cellList, 1.0, 0.0);

  // Seed the metric off the degenerate uniform point (break the l^2 = 1
  // degeneracy so the first relax step can descend), as the operator seed does.
  std::mt19937 jrng(static_cast<std::uint32_t>(seed) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> jitter(0.7, 1.3);
  for (auto *e : cobordism->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(jitter(jrng), 0.0));

  // dW = the boundary 1-cells: edges in exactly one triangle (the opened color
  // hole-circles; the ports are tube-filled, hence interior).
  boundaryCells.clear();
  for (const auto &kv : edgeCoface)
    if (kv.second == 1)
      boundaryCells.push_back({kv.first.first, kv.first.second});

  // === manifold gate: dualComplexValid (rigorous for n <= 3) ===
  const auto valid = EigenstateSynthesis(cobordism, 1).dualComplexValid();
  if (!valid.first)
    throw std::runtime_error(
        "RegisterTopology: seed failed the dual-complex manifold gate: " +
        valid.second);

  return cobordism;
}

void RegisterTopology::readout(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<EdgeLoop> &loops,
    std::vector<std::complex<double>> &targets) const {
  // Each block's three color hole-circles carry that state's three color
  // amplitudes (no S^1). The hole-circle of a sorted triangle (a<b<c) is the
  // signed walk a -> b -> c -> a.
  loops.clear();
  targets.clear();
  if (blockHoles_.empty() || !cobordism) return;
  auto &vlist = *cobordism->getVertexList();
  auto edge = [&](std::uint64_t u, std::uint64_t v) {
    ::tessera::mesh::Vertex *vu = vlist.get(u), *vv = vlist.get(v);
    if (!vu || !vv)
      throw std::runtime_error(
          "RegisterTopology::readout: loop vertex absent from W");
    return ::tessera::mesh::Edge(vu, vv, std::complex<double>(1.0, 0.0));
  };
  const std::size_t nStates = std::min(states.size(), blockHoles_.size());
  for (std::size_t s = 0; s < nStates; ++s) {
    for (std::size_t k = 0; k < blockHoles_[s].size(); ++k) {
      Face h(blockHoles_[s][k]);
      std::sort(h.begin(), h.end());
      loops.push_back({edge(h[0], h[1]), edge(h[1], h[2]), edge(h[2], h[0])});
      targets.push_back(k < states[s].size() ? states[s][k]
                                             : std::complex<double>(0));
    }
  }
}

}  // namespace tessera::cobordism
