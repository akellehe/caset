// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SPECTRAL_H
#define TESSERA_OBSERVABLES_SPECTRAL_H

#include <memory>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # SpectralGap
///
/// Observable: the first spectral gap \f$ \lambda_1 - \lambda_0 \f$ of the
/// **U(1) connection** graph Laplacian
/// (`cobordism::HodgeLaplacian::connectionEigenvalues`) on a triangulation — the
/// scalar wrapper over its ascending eigenvalues. It is the gauge-invariant
/// interference signature the cobordism spec's check **C4** watches: on the
/// triangle it collapses from \f$ 3 \f$ at zero flux to \f$ 0 \f$ at half a flux
/// quantum (\f$ \Phi = \pi \f$), where the two lowest modes become degenerate.
/// Returns 0 when there are fewer than two vertices (no gap).
///
/// It is deliberately NOT the gap of the Hodge \f$ L_0 \f$
/// (`HodgeLaplacian::eigenvalues(0)`): that operator is built from
/// \f$ \partial_1 \f$ and the weight alone, so its gap is a different number and
/// carries no flux dependence.
///
/// Mirrors how the characteristic-number Observables wrap `ChainComplex`: the rich
/// operator stays a plain class, the scalar measurement is an `Observable`. Because
/// the U(1) connection lives on each `Edge`, a `Spacetime` carries everything the
/// measurement needs.
class SpectralGap : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # HarmonicDimension
///
/// Observable: \f$ \dim \ker L^{U(1)} \f$ — the number of harmonic zero-modes of
/// the **U(1) connection** graph Laplacian (the count of
/// `cobordism::HodgeLaplacian::connectionHarmonics`). At zero flux this equals the
/// number of connected components \f$ b_0 \f$; a nonzero U(1) flux **lifts** the
/// zero-mode (magnetic frustration), so it drops *below* the flux-independent
/// topological count from `ChainComplex` — the contrast checks **C4/C5** rest on.
/// Returns 0 for the empty complex.
///
/// It is deliberately NOT \f$ \dim \ker L_0 \f$ of the Hodge operator: the row
/// sums of \f$ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f$ vanish
/// identically, so the constant is harmonic at any weights and
/// \f$ \dim \ker L_0 = b_0 \f$ always — reading it here would duplicate
/// `ChainComplex::bettiNumbers` and delete the flux content.
class HarmonicDimension : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SPECTRAL_H
