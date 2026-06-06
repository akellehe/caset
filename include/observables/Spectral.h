// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
