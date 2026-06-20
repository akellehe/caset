// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/TripartiteRegisterTopology.h"

#include <algorithm>
#include <map>
#include <random>
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

// 2-frequency geodesic subdivision of the icosahedron: an edge-midpoint vertex
// per edge (30 new, ids 12..41), each face split into 4. A connected S^2 with 42
// vertices and 80 faces — enough to host 12 vertex-disjoint hole triangles (the
// bare icosahedron's 12 vertices admit only 4 disjoint holes; we need 12 = 4
// windows of 3). Deterministic (no metric/seed dependence).
std::vector<Face> geodesicTwoSphere() {
  const auto ico = icosahedron();
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> mid;
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
  return faces;
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
    std::size_t /*stateDim*/, std::uint64_t seed,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  // The trivalent W_ABC junction, built NON-WELDED: ONE connected base surface (a
  // 2-frequency geodesic icosahedron, S^2, 42 vertices) minus 12 vertex-disjoint
  // hole triangles grouped into FOUR windows of three — A, B, C (inputs) and R
  // (the emergent result) — extruded x I (prismCells). The four windows are
  // distinct holes = INDEPENDENT cycles (not stacked layers of one shared
  // register), so the three inputs do not average; the result is confined to
  // Sigma=0 by the surface's global Stokes relation (Sigma_R = -Sigma_inputs).
  const auto faces = geodesicTwoSphere();

  // Pick 12 vertex-disjoint hole triangles greedily (deterministic face order ->
  // seed-independent holes), grouped into 4 windows of 3 in pick order: A,B,C,R.
  std::set<std::uint64_t> used;
  std::vector<Face> holes;
  for (const auto &f : faces) {
    if (used.count(f[0]) || used.count(f[1]) || used.count(f[2])) continue;
    holes.push_back(f);  // already sorted
    used.insert(f.begin(), f.end());
    if (holes.size() == 12) break;
  }
  if (holes.size() != 12)
    throw std::runtime_error(
        "TripartiteRegisterTopology: could not select 12 vertex-disjoint holes "
        "on the geodesic base surface");

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

  // The four windows (A,B,C,R) of three color holes each. Windows live on ONE
  // surface, so holes carry absolute vertex ids (NO per-block stride offset,
  // unlike RegisterTopology's layered blocks). Computed BEFORE the metric seed:
  // the van Raamsdonk seed needs the per-vertex party (window) map.
  blockHoles_.assign(4, {});
  for (std::size_t blk = 0; blk < 4; ++blk)
    for (std::size_t k = 0; k < 3; ++k)
      blockHoles_[blk].push_back(holes[blk * 3 + k]);

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
  // the metric). Otherwise fall back to the jitter seed off the degenerate
  // uniform point. REGGE: causal type emergent.
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
    std::mt19937 jrng(static_cast<std::uint32_t>(seed) ^ 0x9e3779b9u);
    std::uniform_real_distribution<double> jitter(0.7, 1.3);
    for (auto *e : cobordism->getEdgeList()->toVector())
      e->setSquaredLength(std::complex<double>(jitter(jrng), 0.0));
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
