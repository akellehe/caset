// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/BipartiteCreationTopology.h"

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

void BipartiteCreationTopology::setLorentzianWorldlines(double worldlineLsq) {
  lorentzian_ = true;
  lorentzWorldlineLsq_ = worldlineLsq;
}

void BipartiteCreationTopology::setUTurnTwist(bool on) { uTurnTwist_ = on; }

void BipartiteCreationTopology::setFrequency(int frequency) {
  if (frequency < 2)
    throw std::invalid_argument(
        "BipartiteCreationTopology: frequency must be >= 2 (N=2 is the base that "
        "hosts the disjoint holes; larger N refines the lattice)");
  frequency_ = frequency;
}

void BipartiteCreationTopology::validateStateDim(std::size_t d) const {
  if (d != 3)
    throw std::invalid_argument(
        "TransportCobordism (bipartite creation topology): state dimension must "
        "be 3 (the color triple on the Sigma=0 hyperplane)");
}

std::shared_ptr<Spacetime> BipartiteCreationTopology::build(
    std::size_t /*stateDim*/, std::uint64_t /*seed*/,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  using Face = std::vector<std::uint64_t>;
  // The q/qbar creation node, built NON-WELDED: ONE connected base surface (the
  // shared frequency-N symmetric-window geodesic icosahedron) minus the first THREE
  // A4 windows of three holes each -- window 0 the seed (pinned input), windows 1,2
  // the emergent q,qbar (the fourth A4 window is left filled). The three punched
  // windows are one C3 orbit of A4, so the seed -> q,qbar transport intertwines the
  // color Z3 and a color-symmetric seed yields color-indefinite emergent windows.
  // Pair neutrality (sigma_q + sigma_qbar = -sigma_seed) is the surface's global
  // Stokes relation -- the same conservation that confines color.
  const auto surface = SymmetricWindowSurface::build(frequency_);
  const auto &faces = surface.faces;

  // The first three A4 windows (seed=0, q=1, qbar=2); window 3 is NOT punched. The
  // holes are flattened in window order for endSignCovector.
  std::vector<Face> holes;
  holes.reserve(9);
  for (std::size_t w = 0; w < 3; ++w)
    for (const auto &h : surface.windows[w]) holes.push_back(h);

  // Cache the windows for readoutHoles()/the accessors.
  blockHoles_.assign(3, {});
  for (std::size_t w = 0; w < 3; ++w)
    for (const auto &h : surface.windows[w]) blockHoles_[w].push_back(h);

  // Holed surface -> 3-complex: the symmetric apex stacking (#413/#429,
  // label-independent, exactly equivariant; a single reflect-and-cap layer -> ONE
  // creation vertex, the localized U-turn flip) or the legacy prism extrusion.
  const std::set<Face> holeSet(holes.begin(), holes.end());
  std::vector<Face> holed;
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);
  const auto prism = Spacetime::symmetricStackCells(holed, apexReflections_);
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(prism.size());
  for (auto c : prism) {
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  auto cobordism = Spacetime::fromCells(3, cells, 1.0, 0.0);

  // Per-layer vertex stride: the base surface vertex count, so a cobordism vertex
  // id's base (window) vertex is id % N and its time class follows the apex labeling.
  std::uint64_t N = 0;
  for (const auto &f : holed)
    for (const auto v : f) N = std::max(N, v + 1);

  // The symmetric UNIFORM seed (l^2 = 1): the symmetric windows need a
  // symmetry-respecting metric for the transport to intertwine the color Z3, and on
  // the (g-invariant) uniform point the Stokes pair-neutrality is EXACT. REGGE:
  // causal type emergent.
  for (auto *e : cobordism->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(1.0, 0.0));

  // LORENTZIAN worldlines (the creation-vertex / U-turn edges timelike): the
  // cross-time edges get l^2 < 0, so the dual Regge action is complex (Im S != 0)
  // and the field strength has a non-empty electric (timelike-leg) sector -- the
  // precondition for a non-degenerate emergent Gauss-law charge. On the symmetric
  // apex interior the bottom (id < N) is t=0, the top (N <= id < 2N) is t=2, and the
  // apexes (id >= 2N) are t=1. The labeling is g-symmetric, so timelike-izing the
  // worldlines preserves the exact equivariance.
  if (lorentzian_) {
    const auto timeOf = [N](std::uint64_t id) -> std::uint64_t {
      return id < N ? 0u : (id < 2 * N ? 2u : 1u);
    };
    for (auto *e : cobordism->getEdgeList()->toVector())
      if (timeOf(e->getSource()->getId()) != timeOf(e->getTarget()->getId()))
        e->setSquaredLength(std::complex<double>(lorentzWorldlineLsq_, 0.0));
  }

  // Per-hole induced-orientation signs (endSignCovector over the 9 holes, grouped
  // per window: seed, q, qbar). The carried condition is sign . psi = 0, so targets
  // are pre-multiplied by it; the emergent windows are read in the SAME global
  // orientation, making sigma the relabeling-invariant Stokes charge (#412).
  const std::vector<int> covec = ChainComplex::endSignCovector(faces, holes);
  if (covec.size() != 9)
    throw std::runtime_error(
        "BipartiteCreationTopology: endSignCovector returned " +
        std::to_string(covec.size()) + " signs, expected 9");
  signTable_.assign(3, std::vector<int>(3, 1));
  for (std::size_t w = 0; w < 3; ++w)
    for (std::size_t k = 0; k < 3; ++k)
      signTable_[w][k] = covec[w * 3 + k];

  // The orientation-reversing U-TURN TWIST (#416) on the antiquark window: reverse
  // its induced orientation so each carried qbar period (and, via the bridge, its
  // emergent electric charge) is the sign-flipped image of the quark's -- the
  // geometric realization of qbar = q backward in time. Realized as a sign reversal
  // of the qbar window's covector (the readout-level form of the within-hole
  // transposition that RegisterTopology::orientationReversingTwist applies).
  if (uTurnTwist_)
    for (auto &s : signTable_[2]) s = -s;

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
          "BipartiteCreationTopology: non-manifold seed (a triangle has > 2 "
          "cofaces)");
    if (kv.second == 1) boundaryCells.push_back(kv.first);
  }

  // === manifold gate: dualComplexValid (rigorous for n <= 3) ===
  const auto valid = EigenstateSynthesis(cobordism, 1).dualComplexValid();
  if (!valid.first)
    throw std::runtime_error(
        "BipartiteCreationTopology: seed failed the dual-complex manifold gate: " +
        valid.second);

  return cobordism;
}

void BipartiteCreationTopology::readoutHoles(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<std::vector<std::uint64_t>> &inputHoles,
    std::vector<std::complex<double>> &inputTargets,
    std::vector<std::vector<std::uint64_t>> &resultHoles,
    std::vector<int> &resultSigns) const {
  // Pin the single supplied seed state on the seed window (window 0) over
  // residualForPeriods (signed by its induced-orientation covector); the quark
  // window (window 1) is returned as the EMERGENT result block, read after the
  // relax. The antiquark window (window 2) is a SECOND emergent result that
  // TransportCobordism's single-result read-out cannot hold; it is read separately
  // by the bridge via antiquarkWindow()/antiquarkSigns().
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  resultSigns.clear();
  if (blockHoles_.size() < 3 || !cobordism) return;

  if (!states.empty())
    for (std::size_t k = 0; k < blockHoles_[0].size(); ++k) {
      inputHoles.push_back(blockHoles_[0][k]);
      const std::complex<double> a =
          k < states[0].size() ? states[0][k] : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(signTable_[0][k]) * a);
    }

  for (std::size_t k = 0; k < blockHoles_[1].size(); ++k) {
    resultHoles.push_back(blockHoles_[1][k]);
    resultSigns.push_back(signTable_[1][k]);
  }
}

std::vector<std::vector<std::uint64_t>> BipartiteCreationTopology::seedWindow()
    const {
  return blockHoles_.empty() ? std::vector<std::vector<std::uint64_t>>{}
                             : blockHoles_[0];
}

std::vector<std::vector<std::uint64_t>> BipartiteCreationTopology::quarkWindow()
    const {
  return blockHoles_.size() < 2 ? std::vector<std::vector<std::uint64_t>>{}
                                : blockHoles_[1];
}

std::vector<std::vector<std::uint64_t>>
BipartiteCreationTopology::antiquarkWindow() const {
  return blockHoles_.size() < 3 ? std::vector<std::vector<std::uint64_t>>{}
                                : blockHoles_[2];
}

std::vector<int> BipartiteCreationTopology::seedSigns() const {
  return signTable_.empty() ? std::vector<int>{} : signTable_[0];
}

std::vector<int> BipartiteCreationTopology::quarkSigns() const {
  return signTable_.size() < 2 ? std::vector<int>{} : signTable_[1];
}

std::vector<int> BipartiteCreationTopology::antiquarkSigns() const {
  return signTable_.size() < 3 ? std::vector<int>{} : signTable_[2];
}

}  // namespace tessera::cobordism
