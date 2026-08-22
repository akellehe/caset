// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LOWRANKUPDATE_H
#define TESSERA_COBORDISM_LOWRANKUPDATE_H

#include <Eigen/Core>
#include <Eigen/LU>

#include <complex>
#include <vector>

#include "cobordism/Certificate.h"

namespace tessera::cobordism {

/// # LowRankUpdate
///
/// Structure-exact Woodbury and secular update helpers for genuinely
/// low-rank local operator changes (#764) — the incremental path a local
/// metric or topology move takes instead of rebuilding a global operator.
///
/// **Woodbury identity (solves).** With the base operator \f$ A \f$ factored
/// once (partial-pivot LU — general complex square, NO positive-definite or
/// Hermitian assumption) and a registered change \f$ \Delta = UW \f$
/// (\f$ U: n\times r,\ W: r\times n \f$),
/// \f[ (A+UW)^{-1}b = A^{-1}b - A^{-1}U\,(I_r + WA^{-1}U)^{-1}\,W A^{-1} b, \f]
/// evaluated entirely by factor solves — the capacitance
/// \f$ I_r + WA^{-1}U \f$ is itself LU-factored; no explicit inverse is ever
/// formed. The result is STRUCTURE-EXACT: exact given the verified premise
/// that \f$ UW \f$ spans the FULL affected change. That premise is never
/// assumed — `factorsFromTouched` constructs factors that span the change by
/// construction and reports when the actual delta leaks outside the declared
/// touched set, `spansAffectedChange` re-verifies a claimed factorization,
/// and the cold-recompute fallback (`refactor`) replaces the base
/// factorization outright. Every solve reports its measured residual
/// \f$ \|(A+UW)x-b\|/\|b\| \f$ and the LU condition estimates.
///
/// **Secular identity (eigenvalues).** For a HERMITIAN operator with known
/// eigenvalues \f$ d_1\le\dots\le d_n \f$ and a rank-one Hermitian update
/// \f$ \rho\,zz^\dagger \f$ expressed in the eigenbasis, the updated
/// eigenvalues are the roots of the secular equation
/// \f[ f(\lambda) = 1 + \rho\sum_i \frac{|z_i|^2}{d_i-\lambda} = 0, \f]
/// one per interlacing interval, found by bisection to machine bracket
/// width. Domain: Hermitian (indefinite allowed); the non-normal Lorentzian
/// regime is REFUSED — interlacing does not hold there, and a self-adjoint
/// method is never applied to a non-self-adjoint operator.
class LowRankUpdate {
  public:
    /// The factors of a touched-star operator change, as built by
    /// `factorsFromTouched`: \f$ \Delta = \text{left}\cdot\text{right} \f$
    /// with rank at most twice the touched-index count. `spansChange` is the
    /// exactness verdict — false means the actual delta has support outside
    /// the declared touched rows/columns, the factors are NOT exact, and the
    /// caller must fall back to a cold recompute.
    struct TouchedFactors {
      bool spansChange{false};
      int rank{0};
      std::vector<std::complex<double>> left{};   ///< dim x rank, row-major
      std::vector<std::complex<double>> right{};  ///< rank x dim, row-major
    };

    /// Factor the base operator (flat row-major `dim` x `dim`) with
    /// partial-pivot LU. @throws std::invalid_argument on a size mismatch.
    LowRankUpdate(const std::vector<std::complex<double>> &base, int dim);

    [[nodiscard]] int dimension() const noexcept { return dim_; }
    /// Rank of the registered change (0 = none).
    [[nodiscard]] int updateRank() const noexcept { return rank_; }

    /// Register the pending change \f$ \Delta = UW \f$ (`left`: dim x rank,
    /// `right`: rank x dim, both flat row-major), replacing any previous one.
    /// @throws std::invalid_argument on size mismatch.
    void setUpdate(const std::vector<std::complex<double>> &left,
                   const std::vector<std::complex<double>> &right, int rank);
    /// Drop the registered change (back to the bare base operator).
    void clearUpdate() noexcept;

    /// Solve \f$ (A + UW)x = b \f$ by the Woodbury identity above. The
    /// certificate is structure-exact with the measured relative residual,
    /// conditioning \f$ \max(\kappa_{LU}(A), \kappa_{LU}(I+WA^{-1}U)) \f$,
    /// and the given tolerance. @throws std::invalid_argument on rhs size.
    [[nodiscard]] CertifiedVector solve(const std::vector<std::complex<double>> &rhs,
                                        double tolerance = 1e-12) const;

    /// Apply the updated operator: \f$ y = (A + UW)x \f$ (for external
    /// residual checks and benchmarks).
    [[nodiscard]] std::vector<std::complex<double>> apply(
        const std::vector<std::complex<double>> &x) const;

    /// Exactness check: whether the registered \f$ UW \f$ spans the FULL
    /// change to `updated`, i.e. \f$ \|(\text{updated}-A) - UW\|_F \le
    /// \text{tolerance}\cdot\|\text{updated}\|_F \f$. A false return means
    /// the low-rank path may NOT be called exact — cold-recompute instead.
    [[nodiscard]] bool spansAffectedChange(
        const std::vector<std::complex<double>> &updated,
        double tolerance = 1e-12) const;

    /// Build exact factors for the change `base` -> `updated` from the
    /// declared touched row/column index set: rows in `touched` are captured
    /// by identity-selector left factors, remaining touched columns by the
    /// delta's columns, so \f$ UW \f$ equals the delta EXACTLY whenever the
    /// delta's support lies in the touched rows/columns — `spansChange`
    /// reports whether it does (false = support leaked outside the declared
    /// star; use the cold path). All-zero rows/columns are trimmed, so the
    /// rank is at most twice the number of ACTIVE touched indices.
    /// @throws std::invalid_argument on size mismatch or out-of-range index.
    [[nodiscard]] static TouchedFactors factorsFromTouched(
        const std::vector<std::complex<double>> &base,
        const std::vector<std::complex<double>> &updated, int dim,
        const std::vector<int> &touched);

    /// Cold-recompute fallback: refactor `base` as the new base operator and
    /// clear any registered update.
    void refactor(const std::vector<std::complex<double>> &base, int dim);

    /// Secular rank-one Hermitian eigenvalue update: the ascending
    /// eigenvalues of \f$ \mathrm{diag}(d) + \rho zz^\dagger \f$ with `z`
    /// given in the eigenbasis of the base operator. Certified numerical:
    /// the residual is the largest of the final relative bisection bracket
    /// widths, the exact trace-identity defect
    /// \f$ |\sum\lambda' - \sum d - \rho\|z\|^2| \f$ (relative), and the
    /// deflated-weight bound. Domain: Hermitian (indefinite allowed) —
    /// `eigenvalues` must be real ascending; the non-normal regime must use
    /// a general eigensolve instead. @throws std::invalid_argument when the
    /// input is not ascending or sizes mismatch.
    [[nodiscard]] static CertifiedVector rankOneEigenvalues(
        const std::vector<double> &eigenvalues,
        const std::vector<std::complex<double>> &z, double rho,
        double tolerance = 1e-10);

  private:
    [[nodiscard]] Eigen::MatrixXcd deltaMatrix() const;

    int dim_{0};
    int rank_{0};
    Eigen::MatrixXcd base_{};
    Eigen::PartialPivLU<Eigen::MatrixXcd> baseFactorization_{};
    /// 1/rcond of the base LU, computed once per (re)factorization.
    double baseConditioning_{1.0};
    Eigen::MatrixXcd left_{};
    Eigen::MatrixXcd right_{};
    /// \f$ Z = A^{-1}U \f$ and the LU of the capacitance
    /// \f$ I_r + WZ \f$, computed once per `setUpdate` (they depend only on
    /// the factors, not the right-hand side).
    Eigen::MatrixXcd capacitanceSolvedLeft_{};
    Eigen::PartialPivLU<Eigen::MatrixXcd> capacitanceFactorization_{};
    double capacitanceConditioning_{1.0};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LOWRANKUPDATE_H
