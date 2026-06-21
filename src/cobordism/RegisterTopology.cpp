// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/RegisterTopology.h"

#include <algorithm>
#include <map>
#include <optional>
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

// The three holonomy-class (color) holes: three pairwise vertex-disjoint
// icosahedron faces (the #353 `_CLASS_HOLES`). Removing them opens the b_1 = 2
// color register on the Sigma = 0 hyperplane (the S_3 standard rep). Order is
// load-bearing: it pairs with the induced-orientation sign (+1, +1, -1) below.
std::vector<Face> classHoles() {
  std::vector<Face> holes = {{0, 1, 2}, {3, 7, 8}, {4, 9, 5}};
  for (auto &h : holes) std::sort(h.begin(), h.end());
  return holes;
}

}  // namespace

// The induced-orientation covector on the three color holes — `endSignCovector`
// of the icosahedron `classHoles()` — equal to the #353 `SIGN_BLOCK`. The carried
// (Sigma = 0) condition is `sign . periods == 0`, so a state's target periods
// must be pre-multiplied by it; feeding raw components mis-floors the color
// singlet [1, omega, omega^2] (whose plain-sum is 0 but raw n.a != 0).
const int RegisterTopology::kColorSign[3] = {1, 1, -1};

void RegisterTopology::validateStateDim(std::size_t d) const {
  if (d != 3)
    throw std::invalid_argument(
        "MergeCobordism (register topology): state dimension must be 3 (the "
        "color triple on the Sigma=0 hyperplane)");
}

void RegisterTopology::setTwist(
    const std::unordered_map<std::uint64_t, std::uint64_t> &twist) {
  twist_ = twist;
}

std::unordered_map<std::uint64_t, std::uint64_t>
RegisterTopology::orientationReversingTwist() {
  // Reverse the induced orientation of each of the three color holes by swapping
  // the two smallest vertices of each ((0,1,2)->swap(0,1); (3,7,8)->swap(3,7);
  // (4,5,9)->swap(4,5)), fixing the rest. A within-hole transposition is an odd
  // permutation, so each hole's carried color period flips sign — the exact
  // antisymmetrizer. It is an involution that preserves each hole triangle setwise,
  // so the read-out blocks are the same removed triangles, only relabeled within.
  // Applied cumulatively by prismCells: B (= phi) enters reversed, R (= phi^2 = id)
  // returns to base, so on the uniform metric M_B = -M_A exactly (pure 3bar).
  std::unordered_map<std::uint64_t, std::uint64_t> twist;
  for (const auto &h : classHoles()) {  // h is sorted (a < b < c)
    twist[h[0]] = h[1];
    twist[h[1]] = h[0];
  }
  return twist;
}

std::shared_ptr<Spacetime> RegisterTopology::build(
    std::size_t /*stateDim*/, std::uint64_t seed,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  // The #353 color-register merge, built NON-WELDED: the holed icosahedron
  // (S^2 - 3 color holes, b_1 = 2 on the Sigma=0 hyperplane) extruded over a
  // 3-layer staircase (Spacetime::prismCells), giving one connected 3-complex
  // with b_1(W) = 2 — ONE shared color register across the three blocks (the
  // confinement: a Sigma != 0 config cannot be carried). The three blocks are
  // the three vertex layers (stride N): block A = layer 0, B = layer 1,
  // R (result) = layer 2; each layer's three color hole-circles are the read-out
  // cycles. b_1 = 2 (not the tube-merge's b_1 = 8) is what reproduces #353's
  // realizability map; the continuous staircase is a valid manifold (every
  // triangle in <= 2 tets, dualComplexValid) — never the welded shared-block
  // construction (which is a P x I transport that buries the result).
  const auto ico = icosahedron();
  const auto holes = classHoles();
  const std::set<Face> holeSet(holes.begin(), holes.end());
  std::vector<Face> holed;  // icosahedron with the 3 color holes removed (b_1=2)
  for (const auto &f : ico)
    if (!holeSet.count(f)) holed.push_back(f);

  std::uint64_t N = 0;  // per-layer vertex stride (matches Spacetime::prismCells)
  for (const auto &f : holed)
    for (const auto v : f) N = std::max(N, v + 1);

  // (holed icosahedron) x [0,2]: a 3-layer staircase of tets (NOT looped — an
  // interval, so the two end caps + the hole-tubes are the boundary). An optional
  // twist (#416) is applied cumulatively up the layers (the mapping-torus twist);
  // empty => identity (the generic, untwisted bipartite merge).
  const std::optional<std::unordered_map<std::uint64_t, std::uint64_t>> twistOpt =
      twist_.empty() ? std::nullopt : std::optional(twist_);
  const auto prism = Spacetime::prismCells(holed, /*layers=*/2, twistOpt);
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(prism.size());
  for (auto c : prism) {
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }

  auto cobordism = Spacetime::fromCells(3, cells, 1.0, 0.0);

  // Seed the metric off the degenerate uniform point (break the l^2 = 1
  // degeneracy so the first relax step can descend), as the operator seed does.
  // REGGE: causal type emergent.
  std::mt19937 jrng(static_cast<std::uint32_t>(seed) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> jitter(0.7, 1.3);
  for (auto *e : cobordism->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(jitter(jrng), 0.0));

  // The per-block (per-layer) color holes, in fixed class-hole order so they pair
  // with kColorSign: block A = layer 0, B = layer 1, R = layer 2. Each block sits
  // at phi^blk of the base (phi the twist, cumulative as in prismCells), so a
  // twisted tube's read-out follows the relabeled holes. phi^0 = identity.
  std::vector<std::uint64_t> phi(N);
  for (std::uint64_t v = 0; v < N; ++v) phi[v] = v;  // phi^0
  blockHoles_.assign(3, {});
  for (std::size_t blk = 0; blk < 3; ++blk) {
    const std::uint64_t off = static_cast<std::uint64_t>(blk) * N;
    for (const auto &h : holes) {
      Face shifted;
      shifted.reserve(h.size());
      for (const auto v : h) shifted.push_back(phi[v] + off);
      std::sort(shifted.begin(), shifted.end());
      blockHoles_[blk].push_back(std::move(shifted));
    }
    // Advance phi -> phi^(blk+1) by composing the base twist once more.
    if (!twist_.empty()) {
      std::vector<std::uint64_t> next(N);
      for (std::uint64_t v = 0; v < N; ++v) {
        const auto it = twist_.find(phi[v]);
        next[v] = (it != twist_.end()) ? it->second : phi[v];
      }
      phi = std::move(next);
    }
  }

  // dW = the single-coface triangles (the two end caps + the hole-tube walls);
  // also the manifold gate: every triangle must sit in <= 2 tets (no weld).
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
          "RegisterTopology: non-manifold seed (a triangle has > 2 cofaces)");
    if (kv.second == 1) boundaryCells.push_back(kv.first);
  }

  // === manifold gate: dualComplexValid (rigorous for n <= 3) ===
  const auto valid = EigenstateSynthesis(cobordism, 1).dualComplexValid();
  if (!valid.first)
    throw std::runtime_error(
        "RegisterTopology: seed failed the dual-complex manifold gate: " +
        valid.second);

  return cobordism;
}

void RegisterTopology::readoutHoles(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<std::vector<std::uint64_t>> &inputHoles,
    std::vector<std::complex<double>> &inputTargets,
    std::vector<std::vector<std::uint64_t>> &resultHoles,
    std::vector<int> &resultSigns) const {
  // The EXACT (#353 period) read-out: each block's three color holes are removed
  // triangles whose dual periods carry that state's three color amplitudes (no
  // S^1). The merge scores the PINNED INPUT blocks over residualForPeriods (the
  // period of a removed triangle (a<b<c), machine-zero on a carried target) and
  // reads the EMERGENT result block over cyclePeriods. The target periods are
  // PRE-MULTIPLIED by the induced-orientation covector kColorSign (the #353
  // SIGN_BLOCK), so the carried condition is sign . psi = 0 (the color singlet
  // [1, omega, omega^2] realizes; raw components would mis-floor it).
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  resultSigns.clear();  // bipartite register: result left unsigned (behavior
                        // unchanged; result-sign symmetry is a #410 follow-up).
  if (blockHoles_.empty() || !cobordism) return;

  // Pin the supplied states (inputs first); for the #353 inputs -> emergent
  // result flow the caller supplies inputs only, so the first UNPINNED block is
  // the result block R (read after the relax, not pinned).
  const std::size_t nStates = std::min(states.size(), blockHoles_.size());
  for (std::size_t s = 0; s < nStates; ++s) {
    for (std::size_t k = 0; k < blockHoles_[s].size(); ++k) {
      Face h(blockHoles_[s][k]);
      std::sort(h.begin(), h.end());
      inputHoles.push_back(std::move(h));
      const std::complex<double> a =
          k < states[s].size() ? states[s][k] : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(kColorSign[k % 3]) * a);
    }
  }
  if (nStates < blockHoles_.size())
    for (const auto &h : blockHoles_[nStates]) {
      Face sorted(h);
      std::sort(sorted.begin(), sorted.end());
      resultHoles.push_back(std::move(sorted));
    }
}

}  // namespace tessera::cobordism
