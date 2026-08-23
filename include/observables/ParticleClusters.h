// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PARTICLECLUSTERS_H
#define TESSERA_OBSERVABLES_PARTICLECLUSTERS_H

// Particle classification from persistent modular spectral components
// (issue #773, Wave 3 of the recursive spectral-fiber program — design spec
// §16 "Algorithm I — quark and baryon discovery" (the §16.1 quark classifier),
// §6.8 "Particle reads", and the whitepaper section "Quarks as modular
// clusters").
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • ParticleClusters       — the QUARK/ANTIQUARK classifier: combines the
//                              already-certified Wave 1/2 evidence
//                              (persistence/localization, odd exterior
//                              parity, an accepted rank-three color band,
//                              the calibrated oriented-triangle anchor
//                              profile, bounded transport leakage, and the
//                              certified determinant-line winding) into a
//                              QuarkRead; searches for the unlabeled
//                              two-state flavor subclass; adapts the
//                              EXISTING Gauss-flux read onto nested
//                              enclosing surfaces; and verifies conjugate-
//                              pair conservation.  #774 (octet/gluon/meson/
//                              diquark) and #775 (bound supercomponent,
//                              color singlet, proton) extend this class
//                              BESIDE the quark surface — nothing here
//                              classifies those sectors.
//   • QuarkCandidateEvidence — the assembled evidence bundle: every field
//                              is a read produced by the merged upstream
//                              kernels (#765/#767/#769/#770/#772/#780);
//                              nothing is recomputed here.
//   • QuarkRead              — the spec §6.8 particle read (spec field
//                              names preserved), plus the additive evidence
//                              summary the classification consumed, the
//                              recorded thresholds, and the #764
//                              certificate.  Unknown or uncertified values
//                              are NULL (empty optional / NaN / 0 for the
//                              sign-valued ints), never zero-filled.
//   • FlavorDoubletRead      — the emergent transported two-state spectral
//                              subclass (searched WITHOUT a requested
//                              dimension).
//   • GaussFluxRead          — the nested-enclosing-surface consistency
//                              read over the EXISTING
//                              EigenstateSynthesis::gaussLawCharge.
//   • ConjugatePairRead      — pair-conservation verification of a
//                              certified conjugate creation homotopy.
//
// ─── Exact identities / certified combinations, and their domains ───────
//
//   • The classification verdict is an exact boolean combination of
//     CONSUMED certificates (a StructureExact claim: exact GIVEN the
//     verified upstream certificates; the residual reported is the maximum
//     residual of the held certificates it consumed).  No anchor, band,
//     transport, winding, parity, or Wick quantity is recomputed here —
//     recomputing any of them would fork the upstream kernels.
//   • Exterior parity and total occupation are the #780 Wick reads on the
//     candidate's carried quasi-free state: ⟨(−1)^N⟩ = det(I − 2Γ) and
//     ⟨N⟩ = tr Γ — algebraically exact on the covariance (the #766 grading
//     evaluated through the quasi-free layer).  Domain: the caller's
//     carried CovarianceState for the component's modes.
//   • Baryon flux: B = ν/3 with ν the CERTIFIED determinant-line winding
//     of #770 (closed full-rank gapped family, or an open cobordism
//     segment closed by its RECORDED matched-reference / boundary-register
//     trivialization — `DeterminantWindingRead::windingClosure` travels on
//     the QuarkRead).  A raw endpoint phase is never baryon-flux evidence:
//     with no certified winding the flux is unknown (null), never zero.
//     The quark/antiquark verdict additionally requires ν = ±1; a
//     certified ν = 0 is a certified ZERO flux (needed by the #774 gluon
//     sector), not a quark.
//   • Quark vs antiquark = determinant-line ORIENTATION (the sign of ν;
//     reversing the world-tube reverses it and transports in the dual
//     color representation), never the color representation alone: the
//     Λ²C³ anti-triplet of two quarks is excluded by its EVEN parity and
//     total occupation two, exactly the whitepaper distinction.
//   • The Gauss-flux electric read REUSES
//     `cobordism::EigenstateSynthesis::gaussLawCharge` verbatim on a
//     family of nested enclosing surfaces (closed stars of nested vertex
//     sets); the flux sum is algebraically exact in the supplied
//     field-strength 2-cochain, and the consistency claim is the measured
//     max deviation across the surfaces.
//   • Q = I3 + B/2 (the Gell-Mann–Nishijima relation on the accepted
//     doublet hypothesis) is TESTED, never asserted: only when baryon
//     flux, the emergent flavor doublet, and the Gauss read are all
//     independently certified, and a pass is recorded as the PROPOSED u/d
//     identification (`udIdentificationProposed`), not a charge
//     definition.
//
// ─── Thresholds ──────────────────────────────────────────────────────────
//
// "Exact" is exact where algebraic; every ACCEPTANCE threshold here is an
// analysis parameter (`ParticleClustersConfig`), echoed verbatim on every
// read (`QuarkRead::thresholds`) so a checkpoint carries the configuration
// that produced its verdicts.
//
// ─── Boundaries ──────────────────────────────────────────────────────────
//
// Read-only observable: consumes caller-assembled reads, never calls a
// solver, never mutates the spacetime (the Gauss adapter constructs a
// degree-2 EigenstateSynthesis solely for its documented read-only
// charge/curvature entry points), and NOTHING here enters any emergence
// objective.  No "quark = hole", no hard-coded u/d labels, no baryon
// number without determinant-winding evidence, and no rank-three band is
// called a quark without its anchor, parity, persistence, and leakage
// certificates — a missing certificate is NAMED in `failedCertificates`.

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "observables/ColorFiber.h"
#include "observables/FiberConnection.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"
#include "observables/SpectralFiber.h"
#include "quantum/CovarianceState.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::cobordism {
  class AnalyticCache;
}
namespace tessera::observables {

/// Analysis thresholds of the particle classification (#773).  Every value
/// selects which reads are CERTIFIED, never which value is reported, and
/// the whole configuration is echoed on every read
/// (`QuarkRead::thresholds`) per the "classification thresholds are
/// configuration, recorded in every read" contract.
struct ParticleClustersConfig {
  /// |⟨(−1)^N⟩ ∓ 1| cap for a definite exterior-parity sign (the #780
  /// Wick parity is exact on the covariance; this absorbs rounding).
  double parityTolerance = 1e-9;
  /// |⟨N⟩ − 1| cap for the single-fermion occupation certificate (the
  /// total-occupation channel distinguishing an antiquark from the
  /// two-quark anti-triplet).
  double occupationTolerance = 1e-9;
  /// Calibrated triangle-anchor atlas-score floor (a² ∈ [0, 1]).
  double minAnchorScore = 0.5;
  /// Determinant-phase coherence floor of the anchor profile (∈ [0, 1]).
  double minPhaseCoherence = 0.5;
  /// Cap on the worst lifetime transport leakage (the #770 regime-
  /// appropriate isometry defect); mirrors
  /// `FiberConnectionConfig::leakageTolerance`.
  double maxTransportLeakage = 1e-6;
  /// Minimum persistence lifetime (#765 track diagnostics: covered
  /// slices/frames).
  double minPersistenceLifetime = 2.0;
  /// Minimum adjacent-slice support overlap along the persistence track.
  double minPersistenceOverlap = 0.5;
  /// Band-localization floor (inverse participation ratio of the color
  /// band, ∈ [1/n, 1]).  The default 0 accepts any MEASURED localization
  /// (an unmeasured NaN still fails); raise it to demand concentration.
  double minLocalization = 0.0;
  /// Minimum band subspace overlap across a refinement for the
  /// refinement-stability certificate.
  double minRefinementOverlap = 0.9;
  /// Subspace-overlap threshold of the flavor-doublet tracking (passed to
  /// `SpectralFiberTracker::matchFibers`).
  double doubletOverlapThreshold = 0.5;
  /// Minimum number of frames a flavor subclass must persist through.
  std::size_t minDoubletFrames = 2;
  /// |I3 ∓ 1/2| cap for a definite doublet-member occupancy.
  double isospinTolerance = 1e-9;
  /// Max deviation across nested enclosing surfaces (and |Im| leakage)
  /// for a consistent Gauss flux.
  double gaussTolerance = 1e-9;
  /// Minimum number of nested enclosing surfaces for a consistency claim.
  std::size_t minEnclosingSurfaces = 2;
  /// |Q_gauss − (I3 + B/2)| cap for the proposed u/d identification.
  double udTolerance = 1e-9;
};

/// # GaussFluxRead
///
/// The electric Gauss-flux consistency read over nested enclosing surfaces
/// (#773 scope: "reuse the existing Gauss-flux read on nested enclosing
/// surfaces").  Each per-surface flux is the EXISTING
/// `cobordism::EigenstateSynthesis::gaussLawCharge` value — an
/// orientation-signed sum of the supplied field-strength 2-cochain over
/// the closed star boundary of one enclosed vertex set, restricted to the
/// electric (timelike-leg) plaquettes when `electricOnly`.  Charge is
/// certified only when the read is CONSISTENT across the surfaces; an
/// inconsistent or single-surface read reports an unknown flux (null),
/// never zero.
struct GaussFluxRead {
  /// The per-surface complex fluxes, in the nested-surface input order.
  std::vector<std::complex<double>> fluxes{};
  /// Number of enclosed vertices of each surface (the nesting witness).
  std::vector<std::size_t> surfaceVertexCounts{};
  /// Whether only electric (timelike-leg) plaquettes were summed.
  bool electricOnly = true;
  /// Max |flux_i − flux_j| over all surface pairs (0 for < 2 surfaces).
  double maxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// Max |Im flux_i| — the imaginary leakage of the real-by-construction
  /// electric charge (never silently discarded).
  double imagLeakage = std::numeric_limits<double>::quiet_NaN();
  /// Whether the read met `gaussTolerance` across at least
  /// `minEnclosingSurfaces` surfaces.
  bool consistent = false;
  /// The consistent electric flux: Re(mean of the per-surface values)
  /// when `consistent`; EMPTY (unknown) otherwise — never a default zero.
  std::optional<double> electricFlux{};
  /// Names of the failed/missing consistency certificates ("" when
  /// consistent): "gauss-consistency".
  std::vector<std::string> failedCertificates{};
  /// AlgebraicallyExact (the flux sums are exact signed sums of the
  /// supplied cochain) with residual = max(maxDeviation, imagLeakage)
  /// against `gaussTolerance`; HeuristicDiscovery when inconsistent.
  cobordism::Certificate certificate{};
};

/// # FlavorDoubletRead
///
/// The emergent, unlabeled, transported two-state spectral subclass that
/// could carry isospin (#773 scope; design spec §16.1).  The search runs
/// WITHOUT a requested dimension: every stable transported subclass of the
/// candidate's band enumeration is followed, and "two-state" is an
/// OUTCOME (`stableSubclassRanks` reports every stable rank found).  The
/// doublet's stored first-frame fiber is the RECORDED member
/// trivialization — a compilation convention like a declared anchor
/// weighting, never a physical u/d label.
struct FlavorDoubletRead {
  /// Whether exactly one stable transported two-state subclass emerged.
  bool found = false;
  /// Form degree of the doublet band (meaningful when `found`).
  int degree = 0;
  /// Rank of the accepted subclass (2 when `found`; never requested).
  std::size_t rank = 0;
  /// Number of frames the winning subclass persisted through.
  std::size_t framesTracked = 0;
  /// Smallest certified continuation overlap along the winning track.
  double minContinuationOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Worst band isolation min(lowerGap, upperGap) along the winning track.
  double minIsolation = std::numeric_limits<double>::quiet_NaN();
  /// Ranks of ALL stable full-length certified subclasses found, in
  /// first-frame band order — the no-requested-dimension witness.
  std::vector<std::size_t> stableSubclassRanks{};
  /// Number of stable TWO-state subclasses (found requires exactly one;
  /// 2+ is an ambiguous doublet hypothesis and stays uncertified).
  std::size_t twoStateCount = 0;
  /// The winning subclass's first-frame fiber — the recorded member
  /// trivialization isospin occupancy is measured against.  Default-
  /// constructed when not `found`.
  SpectralFiber doublet{};
  /// Names of the failed/missing certificates ("flavor-doublet" when the
  /// doublet hypothesis is uncertified).
  std::vector<std::string> failedCertificates{};
  /// Why the doublet is uncertified, when it is ("" when found):
  /// "insufficient-frames", "no-stable-two-state-subclass", or
  /// "ambiguous-two-state-subclasses".
  std::string invalidationReason{};
  /// CertifiedNumerical (residual = 1 − minContinuationOverlap against
  /// 1 − doubletOverlapThreshold) when found; HeuristicDiscovery
  /// otherwise.
  cobordism::Certificate certificate{};
};

/// # QuarkCandidateEvidence
///
/// The assembled evidence bundle of ONE candidate — every field is a read
/// PRODUCED BY the merged upstream kernels; this header never recomputes
/// any of them.  Unsupplied evidence (a default-constructed / NaN / empty
/// field) is MISSING evidence: the corresponding certificate fails by
/// name, it is never presumed to pass.
struct QuarkCandidateEvidence {
  /// The candidate's #765 label-free identity.
  ComponentId component{};
  /// The candidate's selected color band (#769) — the classifier only
  /// reads its rank/acceptance/localization; rank three is REQUIRED,
  /// never requested from the detector.
  SpectralFiber colorBand{};
  /// The #767 calibrated oriented-triangle anchor profile of `colorBand`
  /// (`ColorAnchor::evaluate` output with its pre-declared weighting).
  AnchorProfile anchor{};
  /// The candidate's lifetime transports (#770 world-tube family): every
  /// link must be accepted with leakage under the configured cap.
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// The #770 determinant-line winding of the lifetime family: a closed
  /// full-rank family, or an open cobordism segment under its RECORDED
  /// closure specification.  An invalidated/unclosed winding leaves the
  /// baryon flux unknown.
  DeterminantWindingRead winding{};
  /// The #780 Wick parity ⟨(−1)^N⟩ of the candidate's carried quasi-free
  /// state (`CovarianceState::wickParity`).
  quantum::WickCertificateRead parityRead{};
  /// The #780 Wick total occupation ⟨N⟩
  /// (`CovarianceState::wickTotalNumber`).
  quantum::WickCertificateRead occupationRead{};
  /// #765 persistence-track lifetime (covered slices/frames; NaN =
  /// missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// #765 smallest adjacent-slice support overlap along the track.
  double persistenceMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Band subspace overlap across a refinement
  /// (`SpectralFiber::overlap(...).subspaceOverlap` between the band and
  /// its refined re-extraction; NaN = missing).
  double refinementOverlap = std::numeric_limits<double>::quiet_NaN();
  /// The flavor-doublet search result (`flavorDoubletSearch`); absent =
  /// no doublet evidence, flavor unknown.
  std::optional<FlavorDoubletRead> flavor{};
  /// The candidate's amplitudes on the two doublet members, in the
  /// recorded trivialization (the doublet fiber's stored column order);
  /// absent = occupancy unknown.
  std::optional<Eigen::Vector2cd> doubletOccupancy{};
  /// The declared doublet orientation s ∈ {+1, −1}: which member carries
  /// I3 = +1/2 under the PROPOSED identification (a recorded convention,
  /// like a declared anchor weighting — never a hidden label).
  int doubletOrientation = +1;
  /// The nested-surface Gauss-flux read (`gaussFluxOnSurfaces`); absent =
  /// no charge evidence, charge unknown.
  std::optional<GaussFluxRead> charge{};
};

/// # QuarkRead
///
/// The quark/antiquark particle read (design spec §6.8 — the spec field
/// names are preserved verbatim), extended with the evidence summary the
/// classification consumed, the recorded thresholds, and the #764
/// certificate.  Unknown or uncertified values are NULL (empty optional;
/// NaN for unmeasured doubles; 0 for the sign-valued ints), never zero-
/// filled, and every gap is NAMED in `failedCertificates`.
struct QuarkRead {
  /// The candidate's #765 label-free identity.
  ComponentId component{};
  /// Certified exterior parity: −1 (odd) / +1 (even) / 0 = unknown (an
  /// uncertified parity read never emits a sign).
  int exteriorParity = 0;
  /// Rank of the supplied color band (0 when none was supplied).
  int colorRank = 0;
  /// The calibrated anchor atlas score a² (NaN when no anchor evidence).
  double triangleAnchorScore = std::numeric_limits<double>::quiet_NaN();
  /// max_τ |det A_τ|² of the anchor profile.
  double triangleAnchorMaxTerm = std::numeric_limits<double>::quiet_NaN();
  /// Participation ratio of the anchor term distribution.
  double triangleAnchorParticipation =
      std::numeric_limits<double>::quiet_NaN();
  /// Determinant-phase dispersion (1 − coherence) of the anchor profile.
  double anchorPhaseDispersion = std::numeric_limits<double>::quiet_NaN();
  /// The pre-declared anchor weighting rule ("uniform" / "declared").
  std::string anchorWeightingId{};
  /// The certified determinant-line winding ν; EMPTY when the family was
  /// invalidated or no closure was declared (spec §5.11).
  std::optional<int> determinantWinding{};
  /// The recorded winding closure specification: "closed-family",
  /// "matched-reference", "endpoint-trivialization", or "none".
  std::string windingClosure{"none"};
  /// The caller's closure reference identifier ("" when none).
  std::string windingReferenceId{};
  /// B = ν/3 under a CERTIFIED winding (a certified ν = 0 is a certified
  /// zero flux); EMPTY (unknown) without the winding certificate — baryon
  /// number is never inserted by definition.  The spec sketches this
  /// field as a bare double; its own "unknown or uncertified values are
  /// null, not zero" prose is encoded as the optional.
  std::optional<double> baryonFlux{};
  /// I3 = ±1/2 under the certified doublet hypothesis and the declared
  /// orientation; EMPTY when the doublet is missing/unstable or the
  /// occupancy is not a definite member.
  std::optional<double> isospin{};
  /// The Gauss-consistent electric flux; EMPTY unless BOTH the nested-
  /// surface Gauss read is consistent AND the flavor doublet is certified
  /// (#773 acceptance: a missing/unstable doublet yields unknown flavor
  /// AND charge).
  std::optional<double> electricFlux{};
  /// Passed-fraction of the ten core quark certificates (persistence,
  /// localization, parity-odd, occupation-one, color-rank-three, anchor,
  /// transport-leakage, winding, winding-unit, refinement-stability):
  /// 1.0 exactly when the candidate is a certified quark/antiquark.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed core order
  /// then the flavor/charge order ("flavor-doublet", "isospin",
  /// "gauss-consistency", "ud-identification").  Empty for a fully
  /// certified u/d-identified quark.
  std::vector<std::string> failedCertificates{};

  // ── additive evidence summary (consumed by #774/#775 unchanged) ──────

  /// "quark" (ν = +1), "antiquark" (ν = −1), or "none".
  std::string classification{"none"};
  /// ⟨N⟩ of the carried state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// Determinant-phase coherence of the anchor profile (1 − dispersion).
  double anchorPhaseCoherence = std::numeric_limits<double>::quiet_NaN();
  /// Number of lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst lifetime transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// #765 persistence lifetime consumed (NaN = missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// #765 minimum track overlap consumed.
  double persistenceMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Band localization consumed (from the color-band certificate).
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// Refinement subspace overlap consumed (NaN = missing).
  double refinementOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Whether Q = I3 + B/2 was tested AND held — the proposed u/d
  /// identification (never a general charge definition).
  bool udIdentificationProposed = false;
  /// The declared doublet orientation the isospin was reported under
  /// (0 when no isospin was reported).
  int doubletOrientation = 0;
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (exact boolean combination GIVEN the consumed held
  /// certificates; residual = their maximum residual) for a certified
  /// quark/antiquark; HeuristicDiscovery (never holds) otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary (classification, ν, B, parity,
  /// failed certificates).
  [[nodiscard]] std::string describe() const;

  /// Checkpoint serialization (design spec §20 `particles.quarks`): every
  /// spec field, the evidence summary, the failed-certificate names, and
  /// the threshold echo travel together; unknown values serialize as
  /// null, never zero.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate from `toRecord()` output; rejects an unknown
  /// `schema_version` (std::invalid_argument) per the checkpoint-reader
  /// contract.
  [[nodiscard]] static QuarkRead fromRecord(const Record &record);
};

/// # ConjugatePairRead
///
/// Pair-conservation verification of a conjugate quark-antiquark creation
/// path (#773 scope: verified only for a CERTIFIED conjugate homotopy).  A
/// gap-preserving conjugate path has certified windings on both legs and
/// they cancel exactly; a singular (rank/gap-closing) leg leaves the total
/// flux UNKNOWN — never zero by assumption.
struct ConjugatePairRead {
  /// ν_a + ν_b when both windings are certified; EMPTY otherwise.
  std::optional<int> totalWinding{};
  /// B_a + B_b when both baryon fluxes are known; EMPTY otherwise (the
  /// "singular path returns unknown flux" acceptance channel).
  std::optional<double> totalBaryonFlux{};
  /// Product of the two certified parities (+1 even / −1 odd / 0 =
  /// unknown).
  int totalParity = 0;
  /// Whether the certified total parity is even (+1).
  bool parityEven = false;
  /// Certified conservation: both windings certified, total winding 0,
  /// and even total parity.
  bool conserved = false;
  /// Names of the failed/missing certificates ("winding-first",
  /// "winding-second", "parity-first", "parity-second",
  /// "winding-conservation", "parity-even").
  std::vector<std::string> failedCertificates{};
  /// StructureExact (integer sums are exact GIVEN certified integer
  /// windings/parities) when conserved; HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};
};

/// # ParticleClusters
///
/// The #773 quark/antiquark classifier over persistent modular spectral
/// components (design spec §16.1; whitepaper "Quarks as modular
/// clusters").  See the file banner for the identities implemented, their
/// domains, and the certificate vocabulary.
///
/// **Composition, not recomputation.**  Every certificate consumed here is
/// produced by a merged upstream kernel: #765 persistence diagnostics,
/// #769 band certificates and tracking, #767 anchor profiles, #770
/// transports and determinant windings (with their recorded closure
/// specifications), #772-adjacent parity via the #780 Wick reads, and the
/// EXISTING Gauss-flux read (`EigenstateSynthesis::gaussLawCharge`).  The
/// classifier's own claim is the exact boolean combination.
///
/// **Certificate name vocabulary** (`failedCertificates`): "persistence",
/// "localization", "parity-odd", "occupation-one", "color-rank-three",
/// "anchor", "transport-leakage", "winding", "winding-unit",
/// "refinement-stability" (the ten core gates, in that order), then
/// "flavor-doublet", "isospin", "gauss-consistency", "ud-identification"
/// (the flavor/charge gates, which never veto quark-ness — they only
/// leave their own fields unknown).
///
/// **Read-only observable.**  Never calls a solver, never mutates the
/// spacetime, and no output of this class may enter any emergence
/// objective ("no quark-specific quantity enters the emergence
/// objective" is a tested acceptance bullet).
class ParticleClusters {
  public:
    /// `AnalyticCache` kind string of a cached QuarkRead (#764 contract).
    static constexpr const char *kCacheKind = "quark-read";

    /// Bind the classification thresholds (echoed on every read).
    explicit ParticleClusters(ParticleClustersConfig cfg = {});

    /// The threshold configuration this instance classifies with.
    [[nodiscard]] const ParticleClustersConfig &config() const noexcept {
      return cfg_;
    }

    // ── the quark classifier (design spec §16.1) ────────────────────────

    /// Classify one candidate from its assembled evidence: evaluate the
    /// ten core certificates, derive quark vs antiquark from the
    /// determinant-line orientation, attach the provisional baryon flux
    /// under the certified winding, and fill isospin/charge only from
    /// their own independent certificates.  Never throws on missing
    /// evidence — missing evidence is a NAMED failed certificate.
    [[nodiscard]] QuarkRead classifyQuark(
        const QuarkCandidateEvidence &evidence) const;

    /// `classifyQuark` over a candidate stream, in input order.
    [[nodiscard]] std::vector<QuarkRead> classifyQuarks(
        const std::vector<QuarkCandidateEvidence> &candidates) const;

    /// `classifyQuark` through the #764 `AnalyticCache` contract: keyed by
    /// the color band's cell-vertex set (kind `kCacheKind`, parameter =
    /// `evidenceFingerprint`), served while the band's star is untouched
    /// AND the evidence fingerprint matches, recomputed (and re-stored)
    /// otherwise.  Cached results equal cold recomputation.
    [[nodiscard]] QuarkRead classifyQuarkCached(
        cobordism::AnalyticCache &cache,
        const QuarkCandidateEvidence &evidence) const;

    /// Order-sensitive content fingerprint of the decision-relevant
    /// evidence AND the threshold configuration (the cache parameter): any
    /// change in either recomputes rather than serving a stale verdict.
    [[nodiscard]] std::uint64_t evidenceFingerprint(
        const QuarkCandidateEvidence &evidence) const;

    // ── conjugate-pair conservation ─────────────────────────────────────

    /// Verify pair conservation of a conjugate creation path from the two
    /// endpoint reads: total certified winding, total baryon flux, and
    /// total parity (see `ConjugatePairRead`).  A singular leg (unknown
    /// winding) leaves the totals unknown.
    [[nodiscard]] ConjugatePairRead conjugatePair(const QuarkRead &first,
                                                  const QuarkRead &second) const;

    // ── the emergent flavor doublet (no requested dimension) ────────────

    /// Search the candidate's band enumeration ACROSS FRAMES for a stable
    /// transported two-state subclass (design spec §16.1): follow every
    /// #769 band through consecutive frames via
    /// `SpectralFiberTracker::matchFibers` (certified continuations only,
    /// unambiguous — two chains merging onto one band invalidate each
    /// other), report every stable subclass rank, and accept exactly one
    /// full-length rank-two subclass.  `frames[t]` is the candidate's
    /// `ComponentBandRead` at time/scale sample t (one degree per call —
    /// pass each degree's enumeration separately or concatenated fibers
    /// of one read).  No dimension is ever requested.
    [[nodiscard]] FlavorDoubletRead flavorDoubletSearch(
        const std::vector<ComponentBandRead> &frames) const;

    // ── the reused Gauss-flux electric read ─────────────────────────────

    /// The EXISTING Gauss-flux read on nested enclosing surfaces: for each
    /// enclosed vertex set, `EigenstateSynthesis(st, 2).gaussLawCharge(F,
    /// set, electricOnly)` — the closed-star boundary flux of the supplied
    /// field-strength 2-cochain `F` (canonical degree-2 cell order) —
    /// then the consistency combination `gaussFluxConsistency`.  Read-only
    /// on the spacetime.
    /// @throws std::invalid_argument on an empty surface list;
    ///   std::runtime_error from the underlying read on a malformed `F`.
    [[nodiscard]] GaussFluxRead gaussFluxOnSurfaces(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::complex<double>> &fieldStrength,
        const std::vector<std::vector<std::uint64_t>> &enclosedVertexSets,
        bool electricOnly = true) const;

    /// The pure consistency combination over precomputed per-surface
    /// fluxes (the spacetime path delegates here): consistent when at
    /// least `minEnclosingSurfaces` surfaces agree within
    /// `gaussTolerance` (max pairwise deviation and |Im| leakage).
    [[nodiscard]] GaussFluxRead gaussFluxConsistency(
        const std::vector<std::complex<double>> &fluxes,
        const std::vector<std::size_t> &surfaceVertexCounts = {},
        bool electricOnly = true) const;

    /// Nested enclosing vertex sets by breadth-first shell growth: returns
    /// exactly `shells` sets, sets[0] = the seed ids (deduplicated,
    /// restricted to vertices present in the complex), sets[k] = sets[k−1]
    /// plus every vertex sharing an edge with it.  Nested by construction
    /// (growth saturates at the whole one-skeleton component); read-only.
    /// @throws std::invalid_argument on an empty seed, no seed vertex in
    ///   the complex, or shells < 1.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>>
    nestedEnclosures(const std::shared_ptr<Spacetime> &st,
                     const std::vector<std::uint64_t> &seedVertexIds,
                     std::size_t shells);

    // ── candidate tracking across scale/time ────────────────────────────

    /// Track candidates across frames by their color bands (#769
    /// delegation: `SpectralFiberTracker::matchFibers` on the evidence
    /// bands with `doubletOverlapThreshold`-independent `overlapThreshold`
    /// — the certified-continuation semantics).  Entry (fromIndex,
    /// toIndex) indexes the input evidence lists.
    [[nodiscard]] static std::vector<FiberMatchRead> trackCandidates(
        const std::vector<QuarkCandidateEvidence> &from,
        const std::vector<QuarkCandidateEvidence> &to,
        double overlapThreshold = 0.5);

  private:
    ParticleClustersConfig cfg_{};

    /// One core certificate evaluation: append `name` to `failed` when
    /// `passed` is false; returns `passed` (shared bookkeeping).
    static bool gate(bool passed, const char *name,
                     std::vector<std::string> &failed);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_PARTICLECLUSTERS_H
