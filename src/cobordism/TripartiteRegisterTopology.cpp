// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/TripartiteRegisterTopology.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/SymmetricWindowSurface.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

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

void TripartiteRegisterTopology::setFrequency(int frequency) {
  if (frequency < 2)
    throw std::invalid_argument(
        "TripartiteRegisterTopology: frequency must be >= 2 (N=2 is the base "
        "that hosts the 12 disjoint holes; larger N refines the lattice)");
  frequency_ = frequency;
}

void TripartiteRegisterTopology::setSymmetricInterior(bool on) {
  symmetricInterior_ = on;
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
  using Face = std::vector<std::uint64_t>;
  // The trivalent W_ABC junction, built NON-WELDED: ONE connected base surface (a
  // frequency-N geodesic icosahedron, S^2; N=2 -> 42 vertices) minus 12
  // vertex-disjoint hole triangles grouped into FOUR windows of three — A, B, C
  // (inputs) and R (the emergent result) — extruded x I (prismCells). The four
  // windows are distinct holes = INDEPENDENT cycles (not stacked layers of one
  // shared register), so the three inputs do not average; the result is confined to
  // Sigma=0 by the surface's global Stokes relation (Sigma_R = -Sigma_inputs). The
  // frequency is tunable (#404): larger N refines the lattice, shrinking the
  // discretization residual and driving the singlet overlap -> 1. The symmetric
  // window surface (geodesic sphere + the four A4 windows) is shared with
  // BipartiteCreationTopology via SymmetricWindowSurface.
  const auto surface = SymmetricWindowSurface::build(frequency_);
  const auto &faces = surface.faces;

  // The four A4-tetrahedral, C3-symmetric windows (A,B,C inputs, R the emergent
  // result), generated from the icosahedral symmetry: all 12 holes vertex-disjoint
  // and the windows A4-equivalent, so the per-window transport blocks are
  // cyclically related and a color-symmetric input -> the EXACT singlet (manifest
  // S3). Flattened to 12 holes in window order A,B,C,R.
  const auto &windows = surface.windows;
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

  // Holed surface -> 3-complex: the symmetric apex stacking (#413, label-independent,
  // no std::sort diagonal -> exactly equivariant transport) or the prism extrusion.
  const std::set<Face> holeSet(holes.begin(), holes.end());
  std::vector<Face> holed;
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);
  const auto prism = symmetricInterior_
                         ? Spacetime::symmetricStackCells(holed)
                         : Spacetime::prismCells(holed, /*layers=*/2);
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
  if (vrSeed_ && !symmetricInterior_) {
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

  // LORENTZIAN worldlines: overwrite the cross-time (forward-time) edges as timelike
  // (l^2 < 0), so the dual Regge action goes complex (Im S != 0) and its harmonics
  // carry the singlet's omega-phases (a real/spacelike metric cannot). A worldline
  // whose l^2 relaxes through 0 is null = a photon. The time of a cobordism vertex is
  // id / N (the per-layer stride) on the prism; on the symmetric apex interior the
  // bottom (id < N) is t=0, the top (N <= id < 2N) is t=2, and the face-apexes
  // (id >= 2N) are t=1. Both labelings are g-symmetric, so timelike-izing the
  // worldlines preserves the exact equivariance.
  if (lorentzian_) {
    const bool sym = symmetricInterior_;
    const auto timeOf = [N, sym](std::uint64_t id) -> std::uint64_t {
      if (sym) return id < N ? 0u : (id < 2 * N ? 2u : 1u);
      return id / N;
    };
    for (auto *e : cobordism->getEdgeList()->toVector())
      if (timeOf(e->getSource()->getId()) != timeOf(e->getTarget()->getId()))
        e->setSquaredLength(std::complex<double>(lorentzWorldlineLsq_, 0.0));
  }

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
    std::vector<std::vector<std::uint64_t>> &resultHoles,
    std::vector<int> &resultSigns) const {
  // The EXACT (#353 period) read-out on distinct windows. Pin the supplied input
  // states on windows A,B,C (one window per state) over residualForPeriods, with
  // each window's own induced-orientation covector signTable_; the first unpinned
  // window (R) is returned as the EMERGENT result block, read over cyclePeriods.
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  resultSigns.clear();
  if (blockHoles_.empty() || !cobordism) return;

  const std::size_t nStates = std::min(states.size(), blockHoles_.size());
  for (std::size_t s = 0; s < nStates; ++s)
    for (std::size_t k = 0; k < blockHoles_[s].size(); ++k) {
      inputHoles.push_back(blockHoles_[s][k]);
      const std::complex<double> a =
          k < states[s].size() ? states[s][k] : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(signTable_[s][k]) * a);
    }
  if (nStates < blockHoles_.size()) {
    for (const auto &h : blockHoles_[nStates]) resultHoles.push_back(h);
    // The result block's induced-orientation signs (signTable_[nStates], the same
    // endSignCovector that signs the inputs), so the emergent result is read in the
    // SAME global surface orientation as the inputs -- making sigma_R the
    // relabeling-invariant Stokes charge rather than a bare per-hole-sorted sum.
    for (std::size_t k = 0; k < blockHoles_[nStates].size(); ++k)
      resultSigns.push_back(signTable_[nStates][k]);
  }
}

}  // namespace tessera::cobordism
