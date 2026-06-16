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
/// Hermitian-weighted Hodge Laplacian (`cobordism::HodgeLaplacian`, \f$ k = 0 \f$)
/// on a triangulation — the scalar wrapper over its ascending eigenvalues. It is
/// the gauge-invariant interference signature the cobordism spec's check **C4**
/// watches: on the triangle it collapses from \f$ 3 \f$ at zero flux to \f$ 0 \f$
/// at half a flux quantum (\f$ \Phi = \pi \f$), where the two lowest modes become
/// degenerate. Returns 0 when there are fewer than two vertices (no gap).
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
/// Observable: \f$ \dim \ker L_0 \f$ — the number of harmonic zero-modes of the
/// Hermitian-weighted Hodge Laplacian (the column count of
/// `cobordism::HodgeLaplacian::harmonics`). At zero flux this equals the number of
/// connected components \f$ b_0 \f$; a nonzero U(1) flux **lifts** the zero-mode
/// (magnetic frustration), so it drops *below* the flux-independent topological
/// count from `ChainComplex` — the contrast checks **C4/C5** rest on. Returns 0
/// for the empty complex.
class HarmonicDimension : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SPECTRAL_H
