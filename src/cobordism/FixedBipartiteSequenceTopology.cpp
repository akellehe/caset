// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/FixedBipartiteSequenceTopology.h"

#include <cmath>
#include <stdexcept>

namespace tessera::cobordism {

std::vector<std::complex<double>>
FixedBipartiteSequenceTopology::diquarkColorFor(
    const std::vector<std::vector<std::complex<double>>> &states) const {
  // An explicit override (or a fifth, diquark, state) wins.
  if (!diquarkColor_.empty()) {
    if (diquarkColor_.size() != 3)
      throw std::invalid_argument(
          "FixedBipartiteSequenceTopology: the diquark color must be a 3-vector "
          "(the color triple over window R's three holes)");
    return diquarkColor_;
  }
  if (states.size() >= 5 && states[4].size() == 3) return states[4];

  // The default colored 3bar: the antisymmetric (wedge / cross-product)
  // combination d = q_A ^ q_B of the two quark inputs -- the textbook
  // 3 (x) 3 -> 3bar antisymmetric diquark. Two color-indefinite (omega-rep)
  // inputs wedge to a definite anti-triplet, generically colored (sigma != 0).
  if (states.size() < 2 || states[0].size() != 3 || states[1].size() != 3)
    throw std::invalid_argument(
        "FixedBipartiteSequenceTopology: pinning the intermediate diquark needs "
        "at least the two quark inputs A,B (each a color triple) to form the "
        "antisymmetric 3bar, or an explicit setDiquarkColor / fifth state");
  const auto &a = states[0];
  const auto &b = states[1];
  std::vector<std::complex<double>> d = {a[1] * b[2] - a[2] * b[1],
                                         a[2] * b[0] - a[0] * b[2],
                                         a[0] * b[1] - a[1] * b[0]};
  // Normalize to the singlet norm sqrt(3) so r_U is on the same scale as A's
  // (residualForPeriods is covariant in the target norm).
  double n2 = 0.0;
  for (const auto &z : d) n2 += std::norm(z);
  const double norm = std::sqrt(n2);
  if (norm < 1e-12)
    throw std::invalid_argument(
        "FixedBipartiteSequenceTopology: the two quark inputs are color-"
        "parallel (q_A ^ q_B = 0); the antisymmetric 3bar is undefined -- supply "
        "a non-degenerate pair or an explicit setDiquarkColor");
  const double scale = std::sqrt(3.0) / norm;
  for (auto &z : d) z *= scale;
  return d;
}

std::vector<std::vector<std::uint64_t>>
FixedBipartiteSequenceTopology::diquarkHoles() const {
  return windowHolesAtLayer(/*w=*/3, nLayers() / 2);  // window R at the mid slice
}

std::vector<int> FixedBipartiteSequenceTopology::diquarkSigns() const {
  // The 3bar = orientation-reversed (#416-twisted) image of the R-window's
  // proton-sector induced-orientation covector.
  std::vector<int> signs = windowSignsAtLayer(/*w=*/3, nLayers() / 2);
  for (auto &s : signs) s = -s;
  return signs;
}

void FixedBipartiteSequenceTopology::readoutHoles(
    const std::shared_ptr<Spacetime> &cobordism,
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<std::vector<std::uint64_t>> &inputHoles,
    std::vector<std::complex<double>> &inputTargets,
    std::vector<std::vector<std::uint64_t>> &resultHoles,
    std::vector<int> &resultSigns) const {
  inputHoles.clear();
  inputTargets.clear();
  resultHoles.clear();
  resultSigns.clear();
  if (windowCount() < 4 || !cobordism) return;

  const std::size_t kResult = 3;        // window R
  const int top = nLayers();            // R pins at the top slice
  const int mid = top / 2;              // the canonical intermediate slice

  // === BILATERAL endpoints (identical to #434): A,B,C @ bottom, R @ top. The
  // endpoints stay color-emergent (the inputs are color-indefinite, the singlet
  // emerges). signed by the induced-orientation covector (#412). ===
  for (std::size_t w = 0; w < 4; ++w) {
    const int layer = (w == kResult) ? top : 0;
    const auto winHoles = windowHolesAtLayer(w, layer);
    const auto winSigns = windowSignsAtLayer(w, layer);
    for (std::size_t k = 0; k < winHoles.size(); ++k) {
      inputHoles.push_back(winHoles[k]);
      const std::complex<double> a =
          (w < states.size() && k < states[w].size()) ? states[w][k]
                                                       : std::complex<double>(0);
      inputTargets.push_back(static_cast<double>(winSigns[k]) * a);
    }
  }

  // === THE EXPERIMENT (B over A): additionally pin the intermediate result
  // window R at every strictly-interior temporal layer to the colored 3bar
  // diquark (the #416-twisted antisymmetric combination of the quark inputs). ===
  if (pinIntermediate_) {
    const auto diquark = diquarkColorFor(states);  // a 3-vector
    for (int ell = 1; ell < top; ++ell) {
      const auto winHoles = windowHolesAtLayer(kResult, ell);
      const auto winSigns = windowSignsAtLayer(kResult, ell);
      for (std::size_t k = 0; k < winHoles.size() && k < diquark.size(); ++k) {
        inputHoles.push_back(winHoles[k]);
        // 3bar: the orientation-reversing twist negates the sign covector.
        inputTargets.push_back(-static_cast<double>(winSigns[k]) * diquark[k]);
      }
    }
  }

  // === the emergent result block: window R at the middle slice (the imposed
  // diquark), twisted-signed so TransportCobordism::result mirrors A's
  // intermediate read-out for the A-vs-B comparison. ===
  const auto midR = windowHolesAtLayer(kResult, mid);
  const auto midS = windowSignsAtLayer(kResult, mid);
  for (std::size_t k = 0; k < midR.size(); ++k) {
    resultHoles.push_back(midR[k]);
    resultSigns.push_back(-midS[k]);  // 3bar twist
  }
}

}  // namespace tessera::cobordism
