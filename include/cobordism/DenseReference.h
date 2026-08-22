// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_DENSEREFERENCE_H
#define TESSERA_COBORDISM_DENSEREFERENCE_H

#include <complex>
#include <cstddef>
#include <vector>

#include "cobordism/Certificate.h"

namespace tessera::cobordism {

/// # DenseReference
///
/// The dense reference kernels of the analytic-first contract (#764), used
/// ONLY below a configurable dimension crossover: on small fixtures they
/// supply the independent answer a structured path is compared against (the
/// `denseReferenceError` field of a `Certificate`); above the crossover they
/// REFUSE (throw) — a dense global solve is the prohibited default at scale,
/// never a silent fallback (design spec section 18).
///
/// Kernels:
///  - `solve` — dense partial-pivot LU factor solve (a factor solve computes
///    the same object an explicit inverse would; the inverse is never
///    formed);
///  - `spectrum` — dense eigenvalues; the self-adjoint solver is applied
///    only after VERIFYING Hermiticity (`||A - A^dagger|| <= tol * ||A||`),
///    otherwise the general non-normal solver runs and the certificate says
///    so (a self-adjoint solver is never applied to a non-self-adjoint
///    operator);
///  - `fockSpectrum` — the dense-Fock oracle at the SPECTRUM level: dense
///    one-particle eigensolve, then explicit occupation subset-sum
///    enumeration. The reference the structured `OccupationSpectra` path and
///    the quasi-free (Wick) reads are validated against on crossover
///    fixtures; Fock OPERATOR matrices (creation/annihilation, wedge) are
///    the exterior-algebra track's and are not built here.
///
/// Every result carries a `Certificate` with the measured residual and
/// conditioning; nothing is returned bare.
class DenseReference {
  public:
    /// Ships small: dense kernels are for fixtures, not production scale.
    static constexpr int kDefaultCrossoverDimension = 512;

    /// @param crossoverDimension The dimension at and above which every
    ///   dense kernel refuses. @throws std::invalid_argument if < 1.
    explicit DenseReference(int crossoverDimension = kDefaultCrossoverDimension);

    /// The dimension at and above which dense kernels refuse.
    [[nodiscard]] int crossoverDimension() const noexcept { return crossover_; }
    /// Reconfigure the crossover. @throws std::invalid_argument if < 1.
    void setCrossoverDimension(int crossoverDimension);
    /// Whether a `dim`-dimensional dense computation is permitted.
    [[nodiscard]] bool belowCrossover(int dim) const noexcept {
      return dim < crossover_;
    }

    /// Dense LU solve of \f$ Ax = b \f$ (flat row-major `dim` x `dim`).
    /// Certificate: structure-exact factor solve with measured relative
    /// residual \f$ \|Ax-b\|/\|b\| \f$ and the LU condition estimate.
    /// @throws std::invalid_argument on size mismatch; std::length_error at
    ///   or above the crossover.
    [[nodiscard]] CertifiedVector solve(const std::vector<std::complex<double>> &matrix,
                                        int dim,
                                        const std::vector<std::complex<double>> &rhs,
                                        double tolerance = 1e-12) const;

    /// Dense eigenvalues of a `dim` x `dim` operator, sorted ascending by
    /// \f$ (\mathrm{Re}, \mathrm{Im}) \f$. `selfAdjoint = true` REQUESTS the
    /// self-adjoint solver; it is honored only when
    /// \f$ \|A-A^\dagger\| \le \text{tol}\cdot\|A\| \f$ is verified, else
    /// the general solver runs and the certificate's regime reports
    /// `NonNormal`. Residual: \f$ \max_i \|Av_i-\lambda_iv_i\| /
    /// \|A\| \f$ over the computed pairs; conditioning: the eigenvector
    /// matrix condition estimate (1 for the verified self-adjoint path).
    /// @throws std::invalid_argument on size mismatch; std::length_error at
    ///   or above the crossover.
    [[nodiscard]] CertifiedVector spectrum(const std::vector<std::complex<double>> &matrix,
                                           int dim, bool selfAdjoint,
                                           double tolerance = 1e-10) const;

    /// The dense-Fock oracle at the spectrum level: eigenvalues of the
    /// one-particle operator (as `spectrum`), then the exact
    /// \f$ \binom{n}{N} \f$ occupation subset sums for the `particles`
    /// sector. The certificate inherits the eigensolve's measured residual,
    /// regime, and conditioning (the enumeration adds only rounding).
    /// @throws as `spectrum`, plus std::length_error when the sector itself
    ///   is unmaterializable.
    [[nodiscard]] CertifiedVector fockSpectrum(
        const std::vector<std::complex<double>> &oneParticle, int dim,
        int particles, bool selfAdjoint, double tolerance = 1e-10) const;

  private:
    void requireBelowCrossover(int dim, const char *kernel) const;

    int crossover_{kDefaultCrossoverDimension};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DENSEREFERENCE_H
