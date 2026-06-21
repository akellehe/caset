// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/TripartiteRegisterTopology.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

namespace {

using Face = std::vector<std::uint64_t>;

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

using MidMap = std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t>;

// The 2-frequency geodesic icosahedron and its edge-midpoint map.
struct GeodesicSphere {
  std::vector<Face> faces;  // 80 sub-triangles (sorted)
  MidMap mid;               // undirected icosa edge (min,max) -> midpoint vertex id
};

// 2-frequency geodesic subdivision of the icosahedron: an edge-midpoint vertex
// per edge (30 new, ids 12..41), each face split into 4. A connected S^2 with 42
// vertices and 80 faces — enough to host 12 vertex-disjoint hole triangles (the
// bare icosahedron's 12 vertices admit only 4 disjoint holes; we need 12 = 4
// windows of 3). Deterministic (no metric/seed dependence). The midpoint map is
// returned so the symmetric-window generator can lift the C3 rotations.
GeodesicSphere geodesicTwoSphere() {
  const auto ico = icosahedron();
  MidMap mid;
  std::uint64_t next = 12;
  auto midpoint = [&](std::uint64_t a, std::uint64_t b) {
    const std::pair<std::uint64_t, std::uint64_t> key{std::min(a, b),
                                                      std::max(a, b)};
    const auto it = mid.find(key);
    if (it != mid.end()) return it->second;
    return mid[key] = next++;
  };
  std::vector<Face> faces;
  faces.reserve(ico.size() * 4);
  for (const auto &f : ico) {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const std::uint64_t ab = midpoint(a, b), bc = midpoint(b, c),
                        ca = midpoint(c, a);
    for (Face sub : {Face{a, ab, ca}, Face{b, bc, ab}, Face{c, ca, bc},
                     Face{ab, bc, ca}}) {
      std::sort(sub.begin(), sub.end());
      faces.push_back(std::move(sub));
    }
  }
  return {std::move(faces), std::move(mid)};
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
// Generated FROM the symmetry (four C3 generators + a seed corner per window +
// the midpoint lift), so it is correct in whatever vertex numbering
// geodesicTwoSphere() produces (hardcoding the 12 triples is fragile against the
// midpoint numbering). Asserts the construction: each window a genuine C3 orbit,
// all 12 holes real faces, all 36 vertices distinct.
std::vector<std::vector<Face>> symmetricWindows(const MidMap &mid,
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
  auto m = [&](std::uint64_t x, std::uint64_t y) {
    return mid.at({std::min(x, y), std::max(x, y)});
  };
  // Reverse midpoint lookup (id -> the icosa edge it bisects), to lift a vertex
  // permutation onto the geodesic vertices: a base vertex maps by the
  // permutation, a midpoint m(p,q) maps to m(perm[p], perm[q]).
  std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> rev;
  for (const auto &kv : mid) rev[kv.second] = kv.first;
  auto apply = [&](const std::array<int, 12> &p, Face h) {
    for (auto &v : h) {
      if (v < 12) {
        v = static_cast<std::uint64_t>(p[v]);
      } else {
        const auto pq = rev.at(v);
        v = m(static_cast<std::uint64_t>(p[pq.first]),
              static_cast<std::uint64_t>(p[pq.second]));
      }
    }
    std::sort(h.begin(), h.end());
    return h;
  };

  std::vector<std::vector<Face>> windows(4);
  std::set<std::uint64_t> used;
  for (int w = 0; w < 4; ++w) {
    Face s = {seed[w][0], m(seed[w][0], seed[w][1]), m(seed[w][0], seed[w][2])};
    std::sort(s.begin(), s.end());
    const Face h1 = apply(a[w], s);
    const Face h2 = apply(a[w], h1);
    if (apply(a[w], h2) != s)
      throw std::runtime_error(
          "TripartiteRegisterTopology: symmetric window is not a C3 orbit");
    for (const Face &h : {s, h1, h2}) {
      if (!faceSet.count(h))
        throw std::runtime_error(
            "TripartiteRegisterTopology: symmetric hole is not a base face");
      for (const auto v : h)
        if (!used.insert(v).second)
          throw std::runtime_error(
              "TripartiteRegisterTopology: symmetric windows are not "
              "vertex-disjoint");
      windows[w].push_back(h);
    }
  }
  return windows;  // 4 windows x 3 holes; 36 distinct vertices (asserted above)
}

}  // namespace

void TripartiteRegisterTopology::setEntangledMetric(double intraMI,
                                                    double crossMI,
                                                    double iMax) {
  vrIntraMI_ = intraMI;
  vrCrossMI_ = crossMI;
  vrIMax_ = iMax;
  vrSeed_ = true;
}

void TripartiteRegisterTopology::setLorentzianWorldlines(double worldlineLsq) {
  lorentzian_ = true;
  lorentzWorldlineLsq_ = worldlineLsq;
}

void TripartiteRegisterTopology::validateStateDim(std::size_t d) const {
  if (d != 3)
    throw std::invalid_argument(
        "MergeCobordism (tripartite register topology): state dimension must be "
        "3 (the color triple on the Sigma=0 hyperplane)");
}

std::shared_ptr<Spacetime> TripartiteRegisterTopology::build(
    std::size_t /*stateDim*/, std::uint64_t /*seed*/,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  // The trivalent W_ABC junction, built NON-WELDED: ONE connected base surface (a
  // 2-frequency geodesic icosahedron, S^2, 42 vertices) minus 12 vertex-disjoint
  // hole triangles grouped into FOUR windows of three — A, B, C (inputs) and R
  // (the emergent result) — extruded x I (prismCells). The four windows are
  // distinct holes = INDEPENDENT cycles (not stacked layers of one shared
  // register), so the three inputs do not average; the result is confined to
  // Sigma=0 by the surface's global Stokes relation (Sigma_R = -Sigma_inputs).
  const auto sphere = geodesicTwoSphere();
  const auto &faces = sphere.faces;
  const std::set<Face> faceSet(faces.begin(), faces.end());

  // The four A4-tetrahedral, C3-symmetric windows (A,B,C inputs, R the emergent
  // result), generated from the icosahedral symmetry: all 12 holes vertex-disjoint
  // and the windows A4-equivalent, so the per-window transport blocks are
  // cyclically related and a color-symmetric input -> the EXACT singlet (manifest
  // S3). Flattened to 12 holes in window order A,B,C,R.
  const auto windows = symmetricWindows(sphere.mid, faceSet);
  std::vector<Face> holes;
  holes.reserve(12);
  for (const auto &w : windows)
    for (const auto &h : w) holes.push_back(h);

  // Cache the windows for readoutHoles(). Windows live on ONE surface, so holes
  // carry absolute vertex ids (no per-block stride offset, unlike
  // RegisterTopology's layered blocks). Set BEFORE the metric seed: the van
  // Raamsdonk seed needs the per-vertex party (window) map.
  blockHoles_.assign(4, {});
  for (std::size_t blk = 0; blk < 4; ++blk)
    for (const auto &h : windows[blk]) blockHoles_[blk].push_back(h);

  // Holed surface -> prism (x I) -> 3-complex.
  const std::set<Face> holeSet(holes.begin(), holes.end());
  std::vector<Face> holed;
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);
  const auto prism = Spacetime::prismCells(holed, /*layers=*/2);
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(prism.size());
  for (auto c : prism) {
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  auto cobordism = Spacetime::fromCells(3, cells, 1.0, 0.0);

  // Per-layer vertex stride (matches Spacetime::prismCells): the base surface
  // vertex count, so a cobordism vertex id's base (window) vertex is id % N.
  std::uint64_t N = 0;
  for (const auto &f : holed)
    for (const auto v : f) N = std::max(N, v + 1);

  // Seed the metric. VAN RAAMSDONK (geometry from the singlet entanglement) when
  // setEntangledMetric() was called: each edge's squared length is
  // l^2 = (-log(I/iMax))^2 with I the mutual information of its endpoints' color
  // parties -- intra-window (bound, short), cross-window (longer), bulk (capped).
  // The metric is REAL (the omega-phases ride on the complex BOUNDARY inputs, not
  // the metric). Otherwise the symmetric UNIFORM seed (l^2 = 1): the symmetric
  // windows need a symmetry-respecting metric for the transport to intertwine the
  // color Z3 -- a random jitter would break the A4 symmetry and scatter the
  // singlet -- and on the (g-invariant) uniform point junction charge
  // conservation is EXACT (Sigma_R -> 0). REGGE: causal type emergent.
  if (vrSeed_) {
    std::vector<int> partyOf(static_cast<std::size_t>(N), -1);
    for (std::size_t blk = 0; blk < blockHoles_.size(); ++blk)
      for (const auto &hole : blockHoles_[blk])
        for (const auto v : hole)
          if (v < N) partyOf[static_cast<std::size_t>(v)] = static_cast<int>(blk);
    for (auto *e : cobordism->getEdgeList()->toVector()) {
      const int pu =
          partyOf[static_cast<std::size_t>(e->getSource()->getId() % N)];
      const int pv =
          partyOf[static_cast<std::size_t>(e->getTarget()->getId() % N)];
      const double I = (pu >= 0 && pu == pv)   ? vrIntraMI_
                       : (pu >= 0 && pv >= 0)  ? vrCrossMI_
                                               : 0.0;
      e->setSquaredLength(std::complex<double>(
          ::tessera::mesh::Edge::vanRaamsdonkSquaredLength(I, vrIMax_, 1e-10),
          0.0));
    }
  } else {
    for (auto *e : cobordism->getEdgeList()->toVector())
      e->setSquaredLength(std::complex<double>(1.0, 0.0));
  }

  // LORENTZIAN worldlines: overwrite the cross-layer (forward-time) edges as
  // timelike (l^2 < 0), so the dual Regge action goes complex (Im S != 0) and its
  // harmonics carry the singlet's omega-phases (a real/spacelike metric cannot).
  // A worldline whose l^2 relaxes through 0 is null = a photon. The base-vertex
  // layer of a cobordism vertex is id / N (the per-layer stride).
  if (lorentzian_)
    for (auto *e : cobordism->getEdgeList()->toVector())
      if ((e->getSource()->getId() / N) != (e->getTarget()->getId() / N))
        e->setSquaredLength(std::complex<double>(lorentzWorldlineLsq_, 0.0));

  // Per-hole induced-orientation signs (the generalization of kColorSign): the
  // endSignCovector of the base surface over the 12 holes, grouped per window.
  // The carried condition is sign . psi = 0, so targets are pre-multiplied by it.
  const std::vector<int> covec = ChainComplex::endSignCovector(faces, holes);
  if (covec.size() != 12)
    throw std::runtime_error(
        "TripartiteRegisterTopology: endSignCovector returned " +
        std::to_string(covec.size()) + " signs, expected 12");
  signTable_.assign(4, std::vector<int>(3, 1));
  for (std::size_t blk = 0; blk < 4; ++blk)
    for (std::size_t k = 0; k < 3; ++k)
      signTable_[blk][k] = covec[blk * 3 + k];

  // dW = the single-coface triangles (caps + hole-tube walls); also the manifold
  // gate: every triangle must sit in <= 2 tets (no weld).
  std::map<Face, int> cofaceCount;
  for (const auto &c : cells)
    for (std::size_t i = 0; i < c.size(); ++i) {
      Face tri;
      tri.reserve(c.size() - 1);
      for (std::size_t j = 0; j < c.size(); ++j)
        if (j != i) tri.push_back(c[j]);
      ++cofaceCount[tri];
    }
  boundaryCells.clear();
  for (const auto &kv : cofaceCount) {
    if (kv.second > 2)
      throw std::runtime_error(
          "TripartiteRegisterTopology: non-manifold seed (a triangle has > 2 "
          "cofaces)");
    if (kv.second == 1) boundaryCells.push_back(kv.first);
  }

  // === manifold gate: dualComplexValid (rigorous for n <= 3) ===
  const auto valid = EigenstateSynthesis(cobordism, 1).dualComplexValid();
  if (!valid.first)
    throw std::runtime_error(
        "TripartiteRegisterTopology: seed failed the dual-complex manifold "
        "gate: " +
        valid.second);

  return cobordism;
}

void TripartiteRegisterTopology::readoutHoles(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<std::vector<std::uint64_t>> &inputHoles,
    std::vector<std::complex<double>> &inputTargets,
    std::vector<std::vector<std::uint64_t>> &resultHoles) const {
  // The EXACT (#353 period) read-out on distinct windows. Pin the supplied input
  // states on windows A,B,C (one window per state) over residualForPeriods, with
  // each window's own induced-orientation covector signTable_; the first unpinned
  // window (R) is returned as the EMERGENT result block, read over cyclePeriods.
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  if (blockHoles_.empty() || !cobordism) return;

  const std::size_t nStates = std::min(states.size(), blockHoles_.size());
  for (std::size_t s = 0; s < nStates; ++s)
    for (std::size_t k = 0; k < blockHoles_[s].size(); ++k) {
      inputHoles.push_back(blockHoles_[s][k]);
      const std::complex<double> a =
          k < states[s].size() ? states[s][k] : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(signTable_[s][k]) * a);
    }
  if (nStates < blockHoles_.size())
    for (const auto &h : blockHoles_[nStates]) resultHoles.push_back(h);
}

}  // namespace tessera::cobordism
