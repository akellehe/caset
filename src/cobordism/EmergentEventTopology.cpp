// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/EmergentEventTopology.h"

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

void EmergentEventTopology::setLayers(int nLayers) {
  if (nLayers < 2)
    throw std::invalid_argument(
        "EmergentEventTopology: nLayers must be >= 2 (a middle, emergent slice "
        "must exist between the pinned bottom and top temporal layers)");
  nLayers_ = nLayers;
}

void EmergentEventTopology::setLorentzianWorldlines(double worldlineLsq) {
  lorentzian_ = true;
  lorentzWorldlineLsq_ = worldlineLsq;
}

void EmergentEventTopology::setUTurnTwist(bool on) { uTurnTwist_ = on; }

void EmergentEventTopology::setFrequency(int frequency) {
  if (frequency < 2)
    throw std::invalid_argument(
        "EmergentEventTopology: frequency must be >= 2 (N=2 is the base that "
        "hosts the disjoint holes; larger N refines the lattice)");
  frequency_ = frequency;
}

void EmergentEventTopology::validateStateDim(std::size_t d) const {
  if (d != 3)
    throw std::invalid_argument(
        "TransportCobordism (emergent event topology): state dimension must be 3 "
        "(the color triple on the Sigma=0 hyperplane)");
}

std::shared_ptr<Spacetime> EmergentEventTopology::build(
    std::size_t /*stateDim*/, std::uint64_t /*seed*/,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  using Face = std::vector<std::uint64_t>;
  // ONE connected event cobordism, built NON-WELDED: the shared frequency-N
  // symmetric-window geodesic icosahedron (S^2 minus the four A4 windows of three
  // color holes each) punched into a holed surface, then stacked over nLayers_
  // TEMPORAL layers by the dimension-generic staircase (prismCells). The layers
  // are tube-connected through the shared hole-tube walls (#378), so each window's
  // three color holes are readable at EVERY temporal slice (the base holes shifted
  // by layer*stride). The bottom (ell=0) and top (ell=nLayers_) slices are pinned;
  // the middle slices relax (the emergent intermediates).
  const auto surface = SymmetricWindowSurface::build(frequency_);
  const auto &faces = surface.faces;

  // Flatten all four windows' (A,B,C,R) twelve holes in window order, for the
  // base-layer endSignCovector and for the per-layer accessors.
  std::vector<Face> holes;
  holes.reserve(12);
  for (std::size_t w = 0; w < surface.windows.size(); ++w)
    for (const auto &h : surface.windows[w]) holes.push_back(h);

  blockHoles_.assign(surface.windows.size(), {});
  for (std::size_t w = 0; w < surface.windows.size(); ++w)
    for (const auto &h : surface.windows[w]) blockHoles_[w].push_back(h);

  // Holed surface -> 3-complex: the staircase prism over [0, nLayers_]. An
  // interval (NOT looped), so the two end caps + the hole-tube walls are the
  // boundary. Identity twist (the U-turn is a readout-level sign reversal, below).
  const std::set<Face> holeSet(holes.begin(), holes.end());
  std::vector<Face> holed;
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);

  stride_ = 0;  // per-layer vertex stride (matches Spacetime::prismCells)
  for (const auto &f : holed)
    for (const auto v : f) stride_ = std::max(stride_, v + 1);

  const auto prism = Spacetime::prismCells(holed, /*layers=*/nLayers_);
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(prism.size());
  for (auto c : prism) {
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  auto cobordism = Spacetime::fromCells(3, cells, 1.0, 0.0);

  // The symmetric UNIFORM seed (l^2 = 1): the symmetric windows need a
  // symmetry-respecting metric for the transport to intertwine the color Z3, and
  // on the (g-invariant) uniform point the Stokes relations are exact. REGGE:
  // causal type emergent.
  for (auto *e : cobordism->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(1.0, 0.0));

  // LORENTZIAN worldlines (the cross-temporal-layer edges timelike): l^2 < 0 on
  // any edge spanning two layers, so the dual Regge action is complex (Im S != 0)
  // and the field strength has a non-empty electric (timelike-leg) sector -- the
  // precondition for a non-degenerate emergent Gauss-law charge. A vertex id's
  // layer is id / stride.
  if (lorentzian_) {
    const auto layerOf = [this](std::uint64_t id) -> std::uint64_t {
      return stride_ ? id / stride_ : 0u;
    };
    for (auto *e : cobordism->getEdgeList()->toVector())
      if (layerOf(e->getSource()->getId()) != layerOf(e->getTarget()->getId()))
        e->setSquaredLength(std::complex<double>(lorentzWorldlineLsq_, 0.0));
  }

  // Per-hole induced-orientation signs (endSignCovector over the 12 base holes,
  // grouped per window A,B,C,R). The carried condition is sign . psi = 0, so
  // targets are pre-multiplied by it; the emergent windows are read in the SAME
  // global orientation, making sigma the relabeling-invariant Stokes charge (#412).
  const std::vector<int> covec = ChainComplex::endSignCovector(faces, holes);
  if (covec.size() != holes.size())
    throw std::runtime_error(
        "EmergentEventTopology: endSignCovector returned " +
        std::to_string(covec.size()) + " signs, expected " +
        std::to_string(holes.size()));
  signTable_.assign(blockHoles_.size(), std::vector<int>(3, 1));
  for (std::size_t w = 0; w < blockHoles_.size(); ++w)
    for (std::size_t k = 0; k < 3; ++k)
      signTable_[w][k] = covec[w * 3 + k];

  // The orientation-reversing U-TURN TWIST (#416): reverse every window's
  // induced-orientation covector, so each carried period (and emergent charge) is
  // the sign-flipped image of the untwisted (proton) sector -- the anti-baryon
  // (anti-proton) sector, qbar = q backward in time. Realized as a readout-level
  // sign reversal (the within-hole-transposition form), never a re-welded geometry.
  if (uTurnTwist_)
    for (auto &row : signTable_)
      for (auto &s : row) s = -s;

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
          "EmergentEventTopology: non-manifold seed (a triangle has > 2 "
          "cofaces)");
    if (kv.second == 1) boundaryCells.push_back(kv.first);
  }

  // === manifold gate: dualComplexValid (rigorous for n <= 3) ===
  const auto valid = EigenstateSynthesis(cobordism, 1).dualComplexValid();
  if (!valid.first)
    throw std::runtime_error(
        "EmergentEventTopology: seed failed the dual-complex manifold gate: " +
        valid.second);

  return cobordism;
}

std::vector<std::vector<std::uint64_t>> EmergentEventTopology::windowHolesAtLayer(
    std::size_t w, int layer) const {
  std::vector<std::vector<std::uint64_t>> out;
  if (w >= blockHoles_.size() || layer < 0 || layer > nLayers_) return out;
  const std::uint64_t off = static_cast<std::uint64_t>(layer) * stride_;
  for (const auto &h : blockHoles_[w]) {
    std::vector<std::uint64_t> shifted;
    shifted.reserve(h.size());
    for (const auto v : h) shifted.push_back(v + off);
    std::sort(shifted.begin(), shifted.end());
    out.push_back(std::move(shifted));
  }
  return out;
}

std::vector<int> EmergentEventTopology::windowSignsAtLayer(std::size_t w,
                                                           int /*layer*/) const {
  // The signs are layer-independent (the induced orientation of a hole-tube is the
  // same at every slice); the U-turn twist (if set) is already folded into
  // signTable_ at build time.
  return w < signTable_.size() ? signTable_[w] : std::vector<int>{};
}

void EmergentEventTopology::readoutHoles(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<std::vector<std::uint64_t>> &inputHoles,
    std::vector<std::complex<double>> &inputTargets,
    std::vector<std::vector<std::uint64_t>> &resultHoles,
    std::vector<int> &resultSigns) const {
  // BILATERAL pinning (the #434 experiment): pin the three input windows A,B,C at
  // the BOTTOM layer (ell=0) and the result window R at the TOP layer
  // (ell=nLayers_), both over the EXACT residualForPeriods (signed by their
  // induced-orientation covectors). The middle layers are pinned NOWHERE -- they
  // are the variable interior whose intermediates emerge. The canonical emergent
  // intermediate (window R at the middle layer) is returned as resultHoles so
  // TransportCobordism::result reads it directly.
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  resultSigns.clear();
  if (blockHoles_.size() < 4 || !cobordism) return;

  // The four pinned states in order A,B,C,R. A,B,C pin at the bottom slice; R pins
  // at the top slice. A missing state pins to zero amplitude.
  const std::size_t kResult = 3;  // window R
  const int bottom = 0;
  const int top = nLayers_;
  for (std::size_t w = 0; w < 4; ++w) {
    const int layer = (w == kResult) ? top : bottom;
    const auto winHoles = windowHolesAtLayer(w, layer);
    for (std::size_t k = 0; k < winHoles.size(); ++k) {
      inputHoles.push_back(winHoles[k]);
      const std::complex<double> a =
          (w < states.size() && k < states[w].size()) ? states[w][k]
                                                       : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(signTable_[w][k]) * a);
    }
  }

  // The emergent intermediate read directly into TransportCobordism::result: the
  // result window R at the middle temporal slice (floor(nLayers_/2)).
  const int mid = nLayers_ / 2;
  const auto midR = windowHolesAtLayer(kResult, mid);
  for (std::size_t k = 0; k < midR.size(); ++k) {
    resultHoles.push_back(midR[k]);
    resultSigns.push_back(signTable_[kResult][k]);
  }
}

}  // namespace tessera::cobordism
