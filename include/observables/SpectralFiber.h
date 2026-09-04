// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SPECTRALFIBER_H
#define TESSERA_OBSERVABLES_SPECTRALFIBER_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "cobordism/HodgeLaplacian.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::cobordism {
  class AnalyticCache;
}
namespace tessera::observables {

/// Configuration of the spectral-band detector and tracker (ticket #769,
/// whitepaper "Recursive spectral fibers").  All thresholds are analysis
/// parameters: they select which bands are *certified*, never which
/// eigenvalues exist, and none of them is a Betti-number oracle (the zero
/// band is found by the same relative gap rule as every other band).
struct SpectralFiberConfig {
  /// Form degrees enumerated by
  /// :func:`SpectralFiberTracker::enumerateOnComponents`.  The detector
  /// enumerates whatever ranks the gap rule produces at these degrees; it
  /// never requests a particular rank (no built-in color answer).
  std::vector<int> degrees{0, 1, 2};
  /// Relative band-grouping width: consecutive eigenvalues (sorted by
  /// (Re, Im)) belong to one band when their distance is at most
  /// `groupingTolerance * scale`, `scale` the spectral scale
  /// (max |eigenvalue|, or 1 for an identically zero operator).
  double groupingTolerance = 1e-8;
  /// Isolation floor: a band is certified only when its separation from
  /// the NEAREST DISCARDED eigenvalue in the complex plane
  /// (`SpectralBandCertificate::nearestDiscardedSeparation`) is at least
  /// `minRelativeGap * scale`.  A closing gap therefore returns an
  /// UNCERTIFIED band — never a discontinuous identity change.
  double minRelativeGap = 1e-6;
  /// A certified band's separation must also dominate its own spread:
  /// separation >= gapDominance * (in-band eigenvalue spread).
  double gapDominance = 4.0;
  /// Certification cap on the relative residuals (eigen, left, projector
  /// idempotency), all measured relative to the operator's Frobenius norm.
  double residualTolerance = 1e-9;
  /// Certification cap on the weighted Gram / signature defect
  /// epsilon_G = ||Phi^dagger W Phi - J||.
  double gramDefectTolerance = 1e-8;
  /// Certification cap on the band projector norm ||P||_2 (Kato's
  /// condition number of the spectral projector — 1 for an orthogonal
  /// projector, larger the more oblique the band).  This is the
  /// GAUGE-INVARIANT conditioning of the band; the FRAME condition number
  /// (`SpectralBandCertificate::frameConditionNumber`) depends on the
  /// in-band basis choice and is reported, not capped.
  double projectorNormCap = 1e8;
  /// Certification cap on the band's LOCALIZATION EXCESS
  /// (`SpectralBandCertificate::localizationExcess`) — the whitepaper
  /// acceptance conjunct "a localized spectral projector with stable
  /// rank", enforced HERE, in fiber acceptance, where the specification
  /// puts it.  The excess is 0 for a band as concentrated as its rank
  /// permits and exactly 1 for a perfectly delocalized one, so the default
  /// 0.5 certifies a band no more than halfway from maximally localized to
  /// fully spread.  1.0 accepts any MEASURED localization (an unmeasured
  /// NaN still fails) and reproduces the pre-#808 behaviour.
  double maxLocalizationExcess = 0.5;
  /// Dimension at and above which the self-adjoint paths switch from the
  /// exact dense solve to the sparse block solve (mirrors
  /// `DenseReference::kDefaultCrossoverDimension`).
  int denseCrossover = 512;
  /// Number of lowest eigenpairs the sparse block path computes.  The read
  /// is marked `truncated` when this covers only part of the spectrum; the
  /// first uncovered Ritz value bounds the last covered band's upper gap.
  int requestedEigenpairs = 32;
  /// Extra Ritz vectors carried by the sparse block solve beyond
  /// `requestedEigenpairs` (improves convergence and supplies the shield
  /// value bounding the last covered gap).
  int oversample = 8;
  /// Sparse block solve iteration cap and relative residual target.
  int maxSolverIterations = 400;
  double solverTolerance = 1e-12;
  /// Seed of the deterministic sparse-path start block.
  std::uint64_t solverSeed = 0;
  /// Minimum subspace overlap for a certified track continuation in
  /// :func:`SpectralFiberTracker::matchFibers`.
  double trackOverlapThreshold = 0.5;
  /// When true and the component is below `denseCrossover`, every solve is
  /// cross-checked against the independent `DenseReference` kernel and the
  /// measured deviation is recorded on the certificate
  /// (`Certificate::denseReferenceError`).
  bool crossValidateDense = false;
  /// Chain-level Whitney pencil path only (`HodgeLaplacian::MetricSource::WhitneyPencil`):
  /// the trapezoidal node count of the circular Riesz contour drawn around
  /// each band, and the relative tolerance below which the band's bilinear
  /// pairing `B_C` is declared isotropic (an exceptional point: no left frame).
  int contourNodes = 64;
  double isotropyTolerance = 1e-10;
};

/// # SpectralBandCertificate
///
/// The certification record of one whole spectral band: what was measured
/// about the band, in which metric regime,
/// and whether the band is certified.  Quantities that were not measured
/// are quiet NaN (`cobordism::Certificate::kUnmeasured`), never zero.
///
/// A degenerate band is a single object of rank >= 2.  An unexplained
/// multiplicity is reported exactly as its rank — it is never labeled a
/// Kähler–Dirac taste, a flavor, or a color (rank-three selection belongs
/// to a later classifier, not to this detector).  A negative Krein
/// signature is likewise a *certificate*, never an automatic antiparticle
/// identification.
struct SpectralBandCertificate {
  /// Form degree k of the restricted Hodge operator the band lives on.
  int degree = 0;
  /// Band rank r = number of eigenvalues in the band (with multiplicity).
  std::size_t rank = 0;
  /// Distance from the band to the SORT-ADJACENT eigenvalue below / above
  /// it (complex modulus, over the (Re, Im)-sorted spectrum).  REPORTED
  /// DIAGNOSTICS ONLY: with a genuinely complex spectrum the sorted
  /// neighbour need not be the nearest eigenvalue in the plane, so the
  /// isolation conjunct is enforced on `nearestDiscardedSeparation`
  /// instead.  +infinity when no eigenvalue exists on that side; NaN when
  /// the side is UNKNOWN (the truncated sparse top).
  double lowerGap = std::numeric_limits<double>::infinity();
  double upperGap = std::numeric_limits<double>::infinity();
  /// The BAND GAP the whitepaper names — "a nonzero band gap separating it
  /// from discarded modes": the distance IN THE COMPLEX PLANE from the
  /// band to the nearest eigenvalue outside it,
  /// min over discarded j, in-band i of |lambda_i - lambda_j|, which on a
  /// truncated sparse read is additionally bounded by the shield value.
  /// +infinity when nothing was discarded; NaN when an uncovered side
  /// leaves it UNKNOWN.  This is the quantity acceptance gates on.
  double nearestDiscardedSeparation = std::numeric_limits<double>::infinity();
  /// Inverse participation ratio of the band projector's diagonal density
  /// p_i = |P_ii| / sum_j |P_jj|: localization = sum_i p_i^2, in
  /// [1/n, 1/rank] — 1/rank fully localized on `rank` cells (1 for a
  /// rank-one band on a single cell), 1/n perfectly spread.
  /// Gauge-invariant (reads only the projector) and relabeling-invariant.
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// The effective support fraction n_eff / n = 1 / (n * localization) in
  /// [rank/n, 1]: exactly 1 for a perfectly delocalized band (uniform
  /// projector diagonal), rank/n for a band concentrated on `rank` cells.
  /// Gauge- and relabeling-invariant and comparable across dimensions, but
  /// NOT across ranks — a rank-r band cannot read below r/n.  Reported.
  double localizationSupportFraction = std::numeric_limits<double>::quiet_NaN();
  /// The localization datum the acceptance conjunct GATES on: the
  /// rank-normalized excess (n_eff - rank) / (n - rank) in [0, 1] — 0 when
  /// the band is as concentrated as a rank-`rank` projector can be, 1
  /// exactly when it is perfectly delocalized.  Defined as 0 when the band
  /// spans the whole operator (n == rank): a full-space band leaves no
  /// room to be localized in, so localization says nothing about it.
  double localizationExcess = std::numeric_limits<double>::quiet_NaN();
  /// Idempotency defect ||P^2 - P||_F / max(1, ||P||_F).
  double projectorResidual = std::numeric_limits<double>::quiet_NaN();
  /// epsilon_eig = ||L Phi - Phi Lambda||_F / ||L||_F on the eigen-paired
  /// right frame.
  double eigenResidual = std::numeric_limits<double>::quiet_NaN();
  /// Left-frame residual ||L^dagger Y - Y Lambda^dagger||_F / ||L||_F with
  /// Y = W^dagger Psi the Euclidean left frame.  Equal to `eigenResidual`
  /// on the self-adjoint path (the frames coincide there).
  double leftResidual = std::numeric_limits<double>::quiet_NaN();
  /// epsilon_G: in the self-adjoint / Krein-normalizable regimes
  /// ||Phi^dagger W Phi - J||_F with J = diag(I_p, -I_q); on the
  /// biorthogonal path ||Psi^dagger W Phi - I||_F.
  double gramDefect = std::numeric_limits<double>::quiet_NaN();
  /// Band projector norm ||P||_2 — the spectral norm of the band projector
  /// (Kato's condition number of the SPECTRAL PROJECTOR; 1 for an
  /// orthogonal projector, larger the more oblique the band).  Depends
  /// only on the band's ranges, so it is gauge-invariant; this is what
  /// `SpectralFiberConfig::projectorNormCap` caps.
  double projectorNorm = std::numeric_limits<double>::quiet_NaN();
  /// The FRAME condition number the whitepaper asks for in the non-normal
  /// regime ("use matched right and left frames ... and report both
  /// residuals and the frame condition number"):
  /// max(kappa(Phi), kappa(Psi)) with
  /// kappa(X) = sqrt(lambda_max / lambda_min) of X^dagger |W| X — the
  /// Riesz condition of each reported frame in the same |W| metric the
  /// Gram certificate is measured in.  Exactly 1 on the self-adjoint path
  /// (a W-orthonormal frame is perfectly conditioned) and larger the more
  /// oblique the matched pair.  A DIFFERENT quantity from `projectorNorm`:
  /// it is a property of the reported frames, so a different in-band basis
  /// changes it, which is why acceptance caps the gauge-invariant
  /// projector norm and reports this one.
  double frameConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// Krein inertia of Phi^dagger W Phi: positive / negative eigenvalue
  /// counts (p, q).  Neutral directions are rank - p - q (nonzero only
  /// when the W-Gram is singular, e.g. complex-eigenvalue Krein bands).
  /// In the positive regime p = rank, q = 0.  Negative signature is a
  /// certificate — never an automatic antiparticle identification.
  int positiveSignature = 0;
  int negativeSignature = 0;
  /// Chain-level Whitney pencil regime (`CertificateRegime::ComplexSymmetricPencil`)
  /// only; quiet NaN / false otherwise. The pairing is the complex BILINEAR
  /// restriction `B_C = (Phi^vee)^T G^U Phi` of the band (specification §6):
  /// its determinant and condition number are reported, NO sign or inertia
  /// is extracted from it, and `isotropic` marks `det B_C = 0` (the
  /// exceptional-point indicator), where the canonical left frame is refused
  /// with `leftFrameRefusal` naming the reason. `metricSymmetryDefect` is the
  /// regime's verification residual, `M L = (M L)^T`.
  std::complex<double> pairingDeterminant{std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::quiet_NaN()};
  double pairingCondition = std::numeric_limits<double>::quiet_NaN();
  double pairingScale = std::numeric_limits<double>::quiet_NaN();
  bool isotropic = false;
  std::string leftFrameRefusal{};
  double metricSymmetryDefect = std::numeric_limits<double>::quiet_NaN();
  /// The band's frequency window [min Re(lambda), max Re(lambda)] — the
  /// plain-data window handed to the response API (see
  /// :class:`SpectralBandWindow`).
  double frequencyLower = std::numeric_limits<double>::quiet_NaN();
  double frequencyUpper = std::numeric_limits<double>::quiet_NaN();
  /// Whether the band was produced by a VERIFIED self-adjoint solve (the
  /// positive regime; Hermiticity / symmetry checked before the solver was
  /// applied — a self-adjoint solver is never applied to a
  /// non-self-adjoint operator).
  bool selfAdjoint = false;
  /// Whether the band met every certification threshold of the producing
  /// `SpectralFiberConfig` (isolation on `nearestDiscardedSeparation`,
  /// LOCALIZATION, residuals, Gram defect, projector conditioning).  When
  /// false the band is still reported — an uncertified read, not a
  /// discontinuous identity change.
  bool accepted = false;
  /// The graded claim (#764 vocabulary): domain `BandWindow`, regime as
  /// verified, grade `CertifiedNumerical` (or `AlgebraicallyExact` where
  /// the arithmetic is closed-form) when accepted; an uncertified band
  /// carries `HeuristicDiscovery`, which never `holds()`.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.  States rank, window, gaps,
  /// signature, and residuals; reports a degenerate rank as an
  /// uninterpreted multiplicity.
  [[nodiscard]] std::string describe() const;
};

/// Principal-angle / support comparison of two fibers (the tracking
/// primitive).  Angles are computed between the two frames' column spans
/// restricted to the shared cells (cells matched by their sorted vertex-id
/// tuples — never by index or imposed order), so the read is gauge-
/// invariant (only the projector ranges enter) and relabeling-invariant.
struct FiberOverlapRead {
  /// Jaccard overlap of the two fibers' k-cell supports.
  double supportOverlap = 0.0;
  /// Number of shared k-cells.
  std::size_t sharedCells = 0;
  /// Principal angles (radians, ascending) between the two subspaces
  /// restricted to the shared cells.
  std::vector<double> principalAngles{};
  /// (sum_i cos^2 theta_i) / max(rank_a, rank_b) in [0, 1]; 1 exactly when
  /// the ranks agree and the restricted subspaces coincide.
  double subspaceOverlap = 0.0;
};

/// # SpectralFiber
///
/// One whole isolated spectral band of a component-restricted Hodge
/// operator: the right/left frames, the band projector, the eigenvalues,
/// and the :class:`SpectralBandCertificate`.
///
/// The band is REPRESENTED BY ITS PROJECTOR `P = Phi Psi^dagger W`
/// (`Psi^dagger W Phi = I`): individual eigenvectors are a gauge choice
/// and never determine an identity or a downstream observable.  On the
/// self-adjoint path `Psi = Phi` (W-orthonormal frame, `J = I`); in the
/// Krein-normalizable signed regime `Psi = Phi J` with
/// `Phi^dagger W Phi = J = diag(I_p, -I_q)`; on the biorthogonal
/// (non-normal) path `Phi`, `Psi` are matched right/left subspace bases.
///
/// Instances are immutable value objects produced by
/// :class:`SpectralFiberTracker` (or rehydrated by `fromRecord`).
class SpectralFiber {
  public:
    SpectralFiber() = default;

    /// Assemble a fiber from its parts (the tracker's constructor;
    /// exposed so replay/serialization tests can rebuild fibers).  `cells`
    /// are the k-cell sorted vertex-id tuples in the row order of the
    /// frames; `weights` is the diagonal W restricted to those cells.
    SpectralFiber(std::vector<std::vector<std::uint64_t>> cells,
                  std::vector<std::complex<double>> eigenvalues,
                  Eigen::MatrixXcd rightFrame, Eigen::MatrixXcd leftFrame,
                  Eigen::VectorXcd weights, SpectralBandCertificate certificate);

    /// Form degree of the certificate.
    [[nodiscard]] int degree() const noexcept { return certificate_.degree; }
    /// Band rank (columns of the frames).
    [[nodiscard]] std::size_t rank() const noexcept { return certificate_.rank; }
    /// Whether the certificate accepted the band.
    [[nodiscard]] bool accepted() const noexcept { return certificate_.accepted; }

    /// Right frame Phi (cells x rank).
    [[nodiscard]] Eigen::MatrixXcd rightFrame() const { return right_; }
    /// Left frame Psi (cells x rank), normalized to Psi^dagger W Phi = I.
    [[nodiscard]] Eigen::MatrixXcd leftFrame() const { return left_; }
    /// The band projector P = Phi Psi^dagger W (cells x cells), assembled
    /// on demand from the stored frames.
    [[nodiscard]] Eigen::MatrixXcd projector() const;
    /// The diagonal inner-product weights W restricted to the band's cells
    /// (the metric the Gram/signature certificate is measured in).
    [[nodiscard]] Eigen::VectorXcd weightDiagonal() const { return weights_; }

    /// The band's eigenvalues (with multiplicity), sorted by (Re, Im).
    [[nodiscard]] const std::vector<std::complex<double>> &eigenvalues() const noexcept {
      return eigenvalues_;
    }
    /// Mean of the band eigenvalues.
    [[nodiscard]] std::complex<double> bandCenter() const;

    /// The k-cells carrying the band, as sorted vertex-id tuples in frame
    /// row order (a single-vertex tuple per row at degree 0).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cellVertices() const noexcept {
      return cells_;
    }

    /// The band certificate.
    [[nodiscard]] const SpectralBandCertificate &certificate() const noexcept {
      return certificate_;
    }

    /// Principal-angle / support comparison against another fiber (see
    /// :class:`FiberOverlapRead`).  Cells are matched by vertex-id tuple.
    [[nodiscard]] static FiberOverlapRead overlap(const SpectralFiber &a,
                                                  const SpectralFiber &b);

    /// Checkpoint serialization: the JSON-able :class:`Record` of the
    /// fiber (schema-versioned; complex leaves split `{name}_re`/`{name}_im`
    /// per the #580 convention).
    [[nodiscard]] Record toRecord() const;
    /// Rehydrate a fiber from `toRecord()` output.  Rejects an unknown
    /// `schema_version` (std::invalid_argument), matching the checkpoint
    /// checkpoint reader contract.
    [[nodiscard]] static SpectralFiber fromRecord(const Record &record);

  private:
    std::vector<std::vector<std::uint64_t>> cells_{};
    std::vector<std::complex<double>> eigenvalues_{};
    Eigen::MatrixXcd right_{};
    Eigen::MatrixXcd left_{};
    Eigen::VectorXcd weights_{};
    SpectralBandCertificate certificate_{};
};

/// An accepted band's frequency window as PLAIN DATA for the shifted /
/// AMLS response consumer (#768): the lower/upper frequency bounds plus
/// the band certificate.  Deliberately carries no operator, no frame, and
/// no reference to any response/quotient type.
struct SpectralBandWindow {
  int degree = 0;
  std::size_t rank = 0;
  double frequencyLower = std::numeric_limits<double>::quiet_NaN();
  double frequencyUpper = std::numeric_limits<double>::quiet_NaN();
  SpectralBandCertificate certificate{};
};

/// One matched fiber pair across frames or resolutions.
struct FiberMatchRead {
  /// Positions in the `from` / `to` fiber lists handed to `matchFibers`.
  std::size_t fromIndex = 0;
  std::size_t toIndex = 0;
  int degree = 0;
  FiberOverlapRead overlap{};
  bool ranksEqual = false;
  /// Certified continuation: both endpoint bands accepted, equal rank, and
  /// subspace overlap at least the configured threshold.  When an endpoint
  /// band's gap closed (it is uncertified) the match is reported but NOT
  /// certified — identity never flips discontinuously.
  bool certifiedContinuation = false;
};

/// The band enumeration of one (component, degree) pair.
struct ComponentBandRead {
  /// The component's level-0 cell ids (vertex identifiers) — the
  /// `ComponentRead::support` convention of #765 and the `AnalyticCache`
  /// component key.
  std::vector<std::uint64_t> support{};
  int degree = 0;
  /// Number of k-cells of the restricted operator (its dimension).
  std::size_t dimension = 0;
  /// The restricted operator's k-cells as sorted vertex-id tuples, in the
  /// canonical (ChainComplex-ordered) row/column order.
  std::vector<std::vector<std::uint64_t>> cellVertices{};
  /// The verified metric regime of the solve.
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// Which solver ran: "dense-self-adjoint", "sparse-block-self-adjoint",
  /// or "dense-general".
  std::string solverPath{};
  /// Whether the sparse path covered only the lowest part of the spectrum.
  bool truncated = false;
  /// The computed (covered) eigenvalues, sorted by (Re, Im).
  std::vector<std::complex<double>> coveredEigenvalues{};
  /// The enumerated bands (every band reported, certified or not).
  std::vector<SpectralFiber> fibers{};
  /// The eigensolve's own certificate (regime, residual, conditioning; the
  /// dense-reference deviation when cross-validation ran).
  cobordism::Certificate solveCertificate{};

  /// Checkpoint serialization of the whole read (fibers included).
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static ComponentBandRead fromRecord(const Record &record);
};

/// # SpectralFiberTracker
///
/// Extraction and tracking of whole isolated localized Hodge bands on
/// persistent components (ticket #769).
///
/// **Identity implemented.**  For a component support `S` (vertex ids) the
/// tracker assembles the weighted Hodge operator of the FULL INDUCED
/// SUBCOMPLEX on `S` — every simplex of the complex all of whose vertices
/// lie in `S` — in the canonical ChainComplex cell order, with the same
/// diagonal inner-product weights `W_k` the whole-complex `HodgeLaplacian`
/// uses (consumed read-only; identical conventions, including the
/// degenerate-cell +1 fallback and the `WeightConvention`):
///
///   L_k^S = (W_k^S)^{-1} (d_k^S)^T W_{k-1}^S d_k^S
///           + d_{k+1}^S (W_{k+1}^S)^{-1} (d_{k+1}^S)^T W_k^S ,
///
/// where `d^S` restricts the integer boundary maps to the cells inside
/// `S`.  When `S` is the whole vertex set this is the whole-complex
/// operator: on the signed/complex-weight paths it equals
/// `HodgeLaplacian::laplacian(k)` entry for entry (the tests pin this via
/// the spectral resolution `sum_bands Phi Lambda Psi^dagger W = L`); on
/// the verified positive path the SOLVED object is the symmetric
/// W-orthonormal similarity `B_k^T B_k + B_{k+1} B_{k+1}^T` with the
/// identical spectrum, and the frames are mapped back to cochain
/// coordinates (`Phi = W^{-1/2} U`), so the same resolution identity
/// holds.  At degree 0 it is the induced-subgraph Hermitian U(1) CONNECTION
/// graph Laplacian under exactly `HodgeLaplacian::connectionLaplacian`'s
/// conventions (A_ij = sum l^2 e^{i phase}, magnitude degree) — NOT the Hodge
/// `laplacian(0)`, which is d_1 W_1^-1 d_1^T and a different operator (#805).
/// The band structure a degree-0 fiber tracks is the Aharonov-Bohm one, which
/// only the connection operator carries.  A `(k+1)`-cell with a
/// vertex outside `S` does not contribute (the component is read as a
/// complex in its own right).
///
/// **Regimes** (the regime is VERIFIED, never assumed):
///  - positive: all participating weights real positive (and the degree-0
///    operator Hermitian by measurement) — the symmetric representation
///    `B_k^T B_k + B_{k+1} B_{k+1}^T` is solved self-adjointly: the exact
///    dense solve below `denseCrossover`, the deterministic sparse block
///    (shift-invert subspace) solve at and above it;
///  - Hermitian signed (Krein): weights real with negative entries; the
///    operator is W-self-adjoint (`W L = (W L)^T`, verified); a general
///    dense solve supplies the band, and the certificate records the Krein
///    inertia of `Phi^dagger W Phi` normalized to `diag(I_p, -I_q)`;
///  - non-normal: complex weights (or a failed self-adjointness
///    verification); matched right/left subspaces with
///    `Psi^dagger W Phi = I` and the biorthogonal Riesz projector, both
///    residuals and the band conditioning reported.
///
/// **Band rule.**  Eigenvalues sorted by (Re, Im) are grouped into bands
/// by the relative gap rule of `SpectralFiberConfig`; every band is
/// reported with its projector and certificate; certification requires
/// ISOLATION FROM THE NEAREST DISCARDED EIGENVALUE IN THE COMPLEX PLANE
/// (sorting supplies the grouping, never the isolation measurement),
/// LOCALIZATION (the whitepaper conjunct, capped by
/// `maxLocalizationExcess`), residuals, Gram defect, and
/// projector conditioning — a closing gap yields an uncertified band,
/// never a different identity.  The detector enumerates ranks; it NEVER
/// requests rank three, and no eigenvalue threshold is used as a
/// Betti-number oracle.
///
/// **Read-only observable.**  Never calls a solver on the spacetime and
/// never mutates it; nothing here enters any emergence objective.
class SpectralFiberTracker {
  public:
    /// `AnalyticCache` kind string of the per-(component, degree) payload.
    static constexpr const char *kCacheKind = "spectral-fiber";

    /// Bind to the spacetime to read (kept alive by the shared_ptr), a
    /// configuration, and the Hodge weight convention (defaults to the
    /// process-wide `HodgeLaplacian::defaultWeightConvention()`).
    explicit SpectralFiberTracker(
        std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg = {},
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention());

    /// Bind with an explicit metric source. Under
    /// `HodgeLaplacian::MetricSource::WhitneyPencil` every degree \f$ k \ge 1 \f$
    /// is read on the chain-level Whitney pencil of the induced subcomplex
    /// (`chainhodge::CovariantChainHodge`): the operator is \f$ h_k(s,U) \f$,
    /// the regime is the VERIFIED `ComplexSymmetricPencil` (or `NonNormal`
    /// when the transpose identity fails), and every band is the Riesz
    /// projector of a circular contour drawn around the gap-rule group, with
    /// right frame `Phi`, canonical left frame `Phi~`, and the bilinear pairing
    /// certificates of specification §6 (`pairingDeterminant`,
    /// `pairingCondition`, `pairingScale`, `isotropic`). Degree 0 keeps the
    /// U(1) connection operator (#805) under either source.
    SpectralFiberTracker(std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg,
                         cobordism::HodgeLaplacian::MetricSource source);

    [[nodiscard]] cobordism::HodgeLaplacian::MetricSource metricSource() const noexcept {
      return metricSource_;
    }

    [[nodiscard]] const SpectralFiberConfig &config() const noexcept {
      return cfg_;
    }
    [[nodiscard]] cobordism::HodgeLaplacian::WeightConvention
    weightConvention() const noexcept {
      return weights_;
    }

    /// Enumerate the bands of one component at one form degree (Algorithm
    /// B steps 1-6).  `support` is the component's vertex-id set (input
    /// order irrelevant).  Unknown ids are ignored; an empty restricted
    /// operator yields a read with `dimension == 0` and no fibers.
    /// @throws std::invalid_argument for a negative degree.
    [[nodiscard]] ComponentBandRead enumerateBands(
        const std::vector<std::uint64_t> &support, int degree) const;

    /// Enumerate every configured degree on every component of a #765
    /// discovery result, in (component, degree) order.
    [[nodiscard]] std::vector<ComponentBandRead> enumerateOnComponents(
        const std::vector<ComponentRead> &components) const;

    /// `enumerateBands` through the #764 `AnalyticCache` contract: served
    /// from the cache while the component's star is untouched, recomputed
    /// (and re-stored under the current revision) otherwise.  The cache
    /// must be bound to the same spacetime.  Payload kind `kCacheKind`,
    /// parameter = degree.
    [[nodiscard]] ComponentBandRead enumerateBandsCached(
        cobordism::AnalyticCache &cache,
        const std::vector<std::uint64_t> &support, int degree) const;

    /// The accepted bands' frequency windows as plain data for the
    /// response consumer (#768).
    [[nodiscard]] static std::vector<SpectralBandWindow> acceptedWindows(
        const std::vector<ComponentBandRead> &reads);

    /// Track fibers across frames/resolutions: for every `from` fiber the
    /// best `to` partner of the same degree by subspace overlap (principal
    /// angles on shared cells), ties broken by support overlap then
    /// position.  Fibers with no positive-overlap partner are omitted.
    /// `certifiedContinuation` additionally requires both endpoint bands
    /// accepted, equal ranks, and overlap >= `overlapThreshold`.
    [[nodiscard]] static std::vector<FiberMatchRead> matchFibers(
        const std::vector<SpectralFiber> &from,
        const std::vector<SpectralFiber> &to, double overlapThreshold = 0.5);

  private:
    struct RestrictedOperator;  // assembled restricted Hodge data
    struct SolveOutput;         // one solve path's eigen-paired output

    std::shared_ptr<Spacetime> st_{};
    SpectralFiberConfig cfg_{};
    cobordism::HodgeLaplacian::WeightConvention weights_{
        cobordism::HodgeLaplacian::WeightConvention::SquaredContent};
    cobordism::HodgeLaplacian::MetricSource metricSource_{
        cobordism::HodgeLaplacian::MetricSource::DiagonalWeights};

    [[nodiscard]] RestrictedOperator assembleRestricted(
        const std::vector<std::uint64_t> &support, int degree) const;

    // The three solve paths.
    void solveDenseSelfAdjoint(const RestrictedOperator &op,
                               ComponentBandRead &read) const;
    void solveSparseSelfAdjoint(const RestrictedOperator &op,
                                ComponentBandRead &read) const;
    // The chain-level pencil path: Riesz bands on circular contours.
    void solvePencilBands(const RestrictedOperator &op,
                          ComponentBandRead &read) const;
    void solveDenseGeneral(const RestrictedOperator &op,
                           ComponentBandRead &read) const;

    // Band grouping + per-band measurement/certification (steps 3-6).
    void buildFibers(const RestrictedOperator &op, const SolveOutput &out,
                     ComponentBandRead &read) const;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SPECTRALFIBER_H
