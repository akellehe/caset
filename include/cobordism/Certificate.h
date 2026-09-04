// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CERTIFICATE_H
#define TESSERA_COBORDISM_CERTIFICATE_H

#include <complex>
#include <limits>
#include <string>
#include <vector>

namespace tessera::cobordism {

/// How a result was obtained, in decreasing order of a-priori strength. The
/// grade names the CLAIM CLASS — what kind of statement the producer is
/// making — while `Certificate::holds()` reports whether the measured
/// residual actually met the declared tolerance. The vocabulary is shared by
/// every analytic-first kernel (epic #763): a consumer that needs an exact
/// quantity refuses anything below its required grade instead of silently
/// accepting a looser one.
enum class CertificateGrade {
  /// A closed-form identity evaluated in floating point — e.g. the
  /// Kronecker-sum spectrum rule or a subset-sum enumeration. The only error
  /// source is rounding, so the residual is expected at machine precision
  /// (~1e-15 relative for doubles).
  AlgebraicallyExact,
  /// Exact GIVEN a verified structural premise — e.g. a Woodbury solve is
  /// exact given that the registered low-rank factors span the FULL operator
  /// change (`LowRankUpdate::spansAffectedChange`). The premise check and the
  /// arithmetic residual are both reported.
  StructureExact,
  /// An iterative or truncated computation carrying an explicit residual,
  /// conditioning number, and — on crossover fixtures — a dense-reference
  /// error. Honest label for anything looser than machine precision.
  CertifiedNumerical,
  /// An uncertified proposal (search heuristics, discovery scores). Never
  /// `holds()`; it must be re-derived through a certified path before any
  /// downstream claim is made.
  HeuristicDiscovery,
};

/// The spectral domain a certificate speaks for. `Static` is the
/// zero-frequency/whole-operator statement; `BandWindow` restricts the claim
/// to an explicit frequency window \f$ \Omega \f$ — no nonzero-spectrum
/// claim is ever attached to a static reduction.
enum class CertificateDomain { Static, BandWindow };

/// The metric regime the producing kernel verified.
/// A self-adjoint solver is never applied outside `PositiveSemidefinite` /
/// `HermitianIndefinite`; `NonNormal` results carry general-eigensolver
/// conditioning instead.
/// `ComplexSymmetricPencil` is the chain-level Whitney pencil's own regime
/// (specification §3, §6, §9): the operator is symmetric for a COMPLEX
/// SYMMETRIC chain metric \f$ M \f$, \f$ M L = (M L)^T \f$ (for a dressed
/// connection, \f$ (\tilde A^U)^T = \tilde A^{U^{-1}} \f$), verified before it
/// is claimed. Its pairings are bilinear, so a band carries `det B_C`,
/// `cond B_C`, and an isotropy certificate and NO inertia; nothing Hermitian
/// is asserted, and it is never silently folded into `NonNormal`.
enum class CertificateRegime {
  PositiveSemidefinite,
  HermitianIndefinite,
  NonNormal,
  ComplexSymmetricPencil,
};

/// # Certificate
///
/// The certification record attached to every analytic-first kernel result
/// (#764): the claim grade, its domain and metric regime, the measured
/// residual, the conditioning of the computation, the dense-reference error
/// where one was measured, and the tolerance the producer declared. All
/// quantities are RELATIVE (scale-free) unless the producer documents
/// otherwise; quantities that were not measured are quiet NaN, never zero —
/// a zero would claim a perfect measurement that was not made.
class Certificate {
  public:
    /// Not-measured marker for the optional fields.
    static constexpr double kUnmeasured = std::numeric_limits<double>::quiet_NaN();

    /// Default: an uncertified `HeuristicDiscovery` with nothing measured.
    Certificate() = default;

    /// An `AlgebraicallyExact` claim with its measured rounding residual.
    [[nodiscard]] static Certificate algebraicallyExact(CertificateDomain domain,
                                                        CertificateRegime regime,
                                                        double residual,
                                                        double tolerance);

    /// A `StructureExact` claim: exact given the verified structural premise;
    /// `conditioning` is the condition estimate of the linear algebra that
    /// evaluated it (e.g. the Woodbury capacitance).
    [[nodiscard]] static Certificate structureExact(CertificateDomain domain,
                                                    CertificateRegime regime,
                                                    double residual,
                                                    double conditioning,
                                                    double tolerance);

    /// A `CertifiedNumerical` claim with residual and conditioning.
    [[nodiscard]] static Certificate certifiedNumerical(CertificateDomain domain,
                                                        CertificateRegime regime,
                                                        double residual,
                                                        double conditioning,
                                                        double tolerance);

    /// An uncertified `HeuristicDiscovery` marker (never `holds()`).
    [[nodiscard]] static Certificate heuristicDiscovery(CertificateDomain domain,
                                                        CertificateRegime regime);

    /// The claim class (see `CertificateGrade`).
    [[nodiscard]] CertificateGrade grade() const noexcept { return grade_; }
    /// The spectral domain the claim speaks for.
    [[nodiscard]] CertificateDomain domain() const noexcept { return domain_; }
    /// The metric regime the producing kernel verified.
    [[nodiscard]] CertificateRegime regime() const noexcept { return regime_; }

    /// Measured relative residual of the produced result (NaN = not measured).
    [[nodiscard]] double residual() const noexcept { return residual_; }

    /// Condition estimate of the computation (>= 1; NaN = not measured).
    [[nodiscard]] double conditioning() const noexcept { return conditioning_; }

    /// Relative error against the dense reference kernel, when the result was
    /// cross-checked on a crossover fixture (NaN = not cross-checked).
    [[nodiscard]] double denseReferenceError() const noexcept {
      return denseReferenceError_;
    }
    /// Record the dense-reference error measured on a crossover fixture.
    void setDenseReferenceError(double error) noexcept {
      denseReferenceError_ = error;
    }

    /// The tolerance the producer declared for `holds()`.
    [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

    /// Whether the certificate stands: a certified grade whose measured
    /// residual met the declared tolerance. `HeuristicDiscovery` never holds;
    /// an unmeasured (NaN) residual never holds.
    [[nodiscard]] bool holds() const noexcept {
      return grade_ != CertificateGrade::HeuristicDiscovery &&
             residual_ <= tolerance_;
    }

    /// One-line human-readable summary (grade/domain/regime + numbers).
    [[nodiscard]] std::string describe() const;

  private:
    Certificate(CertificateGrade grade, CertificateDomain domain,
                CertificateRegime regime, double residual, double conditioning,
                double tolerance) noexcept
        : grade_(grade), domain_(domain), regime_(regime), residual_(residual),
          conditioning_(conditioning), tolerance_(tolerance) {}

    CertificateGrade grade_{CertificateGrade::HeuristicDiscovery};
    CertificateDomain domain_{CertificateDomain::Static};
    CertificateRegime regime_{CertificateRegime::NonNormal};
    double residual_{kUnmeasured};
    double conditioning_{kUnmeasured};
    double denseReferenceError_{kUnmeasured};
    double tolerance_{0.0};
};

/// A vector-valued kernel result (a solution, an eigenvalue list, a spectrum)
/// together with the `Certificate` that grades it. The uniform return record
/// of the analytic-first kernels, so no result travels without its
/// certification.
struct CertifiedVector {
  /// The result values (a solution vector or a sorted eigenvalue list).
  std::vector<std::complex<double>> values{};
  /// The certification record grading `values`.
  Certificate certificate{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CERTIFICATE_H
