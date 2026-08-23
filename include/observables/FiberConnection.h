// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_FIBERCONNECTION_H
#define TESSERA_OBSERVABLES_FIBERCONNECTION_H

// Derived U(r) fiber transport and center-aware rank-three holonomies
// (issue #770, Wave 2 of the recursive spectral-fiber program — design spec
// §12 "Algorithm E", §6.6 "Derived transport", §5.5 "Transport leakage",
// §5.11 "Relative determinant winding", and the whitepaper section "Color
// transport and Wilson loops without a new gauge field").
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • FiberConnection      — the derived-connection kernel: assembles the
//                            chain transfer T_AB from EXISTING machinery
//                            (the Hodge d'Alembertian's intercomponent
//                            block, or a RecursiveQuotient response-network
//                            block), forms the overlap
//                            M_AB = Φ_A†W_A T_AB Φ_B (Ψ_A† on the
//                            biorthogonal path), reports every
//                            pre-normalization diagnostic, gates, and only
//                            then reduces to the polar U(r) /
//                            pseudo-unitary factor; composes accepted maps
//                            into Wilson observables; continues rank-three
//                            cube-root branches; and certifies determinant
//                            windings for closed families and declared
//                            open-segment closures.
//   • FiberTransportRead   — one derived transport A ← B with its raw map,
//                            diagnostics, normalized factor, determinant
//                            phase, and #764 certificate (spec §6.6; the
//                            spec's per-transport winding/center fields
//                            materialize on the dedicated family reads
//                            below, because an integer winding exists only
//                            for a declared family/closure and a center
//                            sector only for a declared lift path).
//   • WilsonHolonomyRead   — the loop product: full U(r) holonomy,
//                            normalized trace, determinant line, and the
//                            center-blind adjoint reads.
//   • FundamentalLiftRead  — the explicitly lifted SU(3) fundamental
//                            holonomy with its declared base branch and
//                            accumulated Z₃ center sector.
//   • DeterminantWindingRead / WindingClosureSpec
//                          — the integer determinant winding of a closed
//                            full-rank family, or the RELATIVE winding of
//                            an open cobordism segment under a recorded
//                            closure specification (matched-reference or
//                            endpoint trivialization); unknown when no
//                            closure is declared.
//
// ─── Exact identities implemented, and their domains ────────────────────
//
//   • Chain transfer (both sources are wrappers over existing machinery,
//     never a sampled gauge field):
//       -- `chainTransfer` reads the off-diagonal block
//          T_AB = L_k[cells(A), cells(B)] of the whole-complex weighted
//          Hodge operator `cobordism::HodgeLaplacian::laplacian(k)` (the
//          cochain-coordinate d'Alembertian
//          L_k = W_k⁻¹d_kᵀW_{k-1}d_k + d_{k+1}W_{k+1}⁻¹d_{k+1}ᵀW_k; at
//          k = 0 the Hermitian U(1) graph Laplacian).  This block equals
//          the same block of the operator of the induced subcomplex on
//          support(A) ∪ support(B) EXACTLY: every coupling path between a
//          k-cell of A and a k-cell of B — a shared (k−1)-facet or a
//          common (k+1)-coface — has all its vertices inside the two
//          cells, hence inside the union support, and the per-cell weights
//          are identical.  Domain: any degree with both fibers' cells
//          present in the complex.
//       -- `responseTransfer` returns the effective response block of an
//          existing `cobordism::RecursiveQuotient::ResponseNetworkRead`
//          edge (rows = A's stalk, columns = B's stalk) — the
//          coarse-level induced transfer.
//   • Overlap and leakage (spec §5.5): M_AB = Φ_A†W_A T_AB Φ_B in the
//     self-adjoint regimes, M_AB = Ψ_A†W_A T_AB Φ_B (Ψ_A†W_AΦ_A = I) on
//     the biorthogonal path; leakage η = ‖M†M − I‖₂ in the positive
//     regime, the J-isometry defect ‖M†J_A M − J_B‖₂ in the Krein regime
//     (J = diag(I_p, −I_q) from the band signatures; identity in the
//     positive regime, so the two coincide there).
//   • Polar reduction (positive regime): V = M(M†M)^{−1/2} ∈ U(r) via
//     SVD, exactly unitary-equivariant — Φ_A ↦ Φ_A g_A, Φ_B ↦ Φ_B g_B
//     gives M ↦ g_A†Mg_B and V ↦ g_A†Vg_B (bifundamental; tested with
//     independent random U(r) changes at every component).  Domain:
//     accepted equal-rank bands, full numerical rank, leakage and
//     conditioning below their gates.
//   • Pseudo-unitary reduction (Krein regime, MATCHING signatures only):
//     V = M·K^{−1/2}, K = J_B M†J_A M (K is J_B-self-adjoint; the
//     principal square root is well defined for the near-J-isometric maps
//     the leakage gate admits), giving V†J_A V = J_B exactly in exact
//     arithmetic — inertia is retained, never silently Euclideanized.  A
//     signature mismatch REJECTS before reduction.
//   • Non-normal regime: the raw GL(r,C) transport is retained and
//     certified (rank, singular values, conditioning); no U(r) or SU(3)
//     value is emitted by applying the positive-metric formula outside
//     its domain.
//   • Wilson observables: H(γ) = Π_{(AB)∈γ} V_AB with V_AB : fiber(B) →
//     fiber(A); under local frame changes a CLOSED holonomy is conjugated
//     at its base component, H ↦ g_{A₀}†Hg_{A₀}, so the normalized trace
//     Tr H / r, det H, and the adjoint reads are base-point-conjugation
//     observables.  The determinant-line, projective/adjoint (center
//     blind: χ_adj = |Tr H|² − 1; at rank three the faithful PU(3) image
//     Ad(H) built with `ColorFiber::adjointOctetProjector`), and the
//     explicitly lifted fundamental are exposed as DISTINCT observables.
//   • Rank-three center structure: the read stores the full U(3) factor
//     AND det V ∈ U(1) — the cube-root ambiguity V ↦ V/(det V)^{1/3} is
//     Z₃, so the faithful data are (V, det V, [V] ∈ PU(3)).  A requested
//     fundamental lift continues a cube-root branch from a declared base
//     branch s₀ ∈ {0,1,2}: with per-link principal phases
//     θ_j = Arg det V_j ∈ (−π, π] and Θ = Σθ_j the accumulated
//     determinant phase, H̃ = H·e^{−iΘ/3}·ω^{−s₀} ∈ SU(3) exactly
//     (det H = e^{iΘ} by construction), and the accumulated center sector
//     m ≡ (Θ − Arg e^{iΘ})/2π (mod 3) is RECORDED — branch-independent,
//     while the lift itself shifts by ω^{−s₀} across branches and every
//     projective/adjoint read of the lift is branch-independent.
//   • Determinant winding (spec §5.11): for a CLOSED, continuous,
//     full-rank, gapped family of accepted transports the integer winding
//     ν = (1/2π)Σ principal steps of arg det around the cycle; the read
//     is INVALIDATED (winding = nullopt, reason recorded) when any sample
//     is unaccepted (a closed gap or lost rank), ranks disagree, or a
//     phase step reaches π (aliasing).  An OPEN segment gets only the
//     RELATIVE winding of the closed composite under a recorded closure
//     specification: matched-reference (caller-supplied reference
//     transports traversed backwards) or endpoint trivializations
//     (register-supplied frames; the four principal legs close the
//     determinant path exactly).  With no declared closure the winding is
//     UNKNOWN — a raw endpoint phase difference is never promoted to an
//     integer.  Interpretation as baryon flux is out of scope (#773).
//
// ─── Caching ─────────────────────────────────────────────────────────────
//
// Spacetime-backed transports and loop products go through the #764
// `AnalyticCache` contract: entries are keyed by the participating fibers'
// cell-vertex sets, so a published `TouchedStar` invalidates ONLY the
// transports/loops touching the changed star — disjoint siblings are
// served from cache, and cached results equal cold recomputation.
//
// ─── Boundaries ──────────────────────────────────────────────────────────
//
// Read-only observable: consumes accepted #769 fibers, never re-extracts
// bands, never mutates the spacetime, and nothing here enters any
// emergence objective.  The link matrix is always reconstructed from
// neighboring Hodge frames with a leakage certificate — no independently
// sampled gauge connection.  Polar normalization never conceals a bad
// assignment: every gate fires BEFORE reduction, and a rejected read
// still carries its raw map and full diagnostics.  Exchange characters,
// rotation loops, and spin lifts belong to #772; quark classification and
// baryon-flux interpretation to #773.

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/RecursiveQuotient.h"
#include "observables/Record.h"
#include "observables/SpectralFiber.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::cobordism {
  class AnalyticCache;
}
namespace tessera::observables {

/// Threshold configuration of the derived-transport gates (#770).  All
/// gates fire BEFORE polar/pseudo-unitary reduction — a failed gate yields
/// a rejected read that still reports its raw map and diagnostics.
struct FiberConnectionConfig {
  /// Relative singular-value cut for the numerical rank of the overlap
  /// (σ_i > rankTolerance · σ_max counts toward the rank).
  double rankTolerance = 1e-9;
  /// Cap on the regime-appropriate isometry leakage η before a unitary /
  /// pseudo-unitary factor may be emitted (spec §5.5).
  double leakageTolerance = 1e-6;
  /// Cap on frame/overlap conditioning: each endpoint band's certified
  /// condition number and the overlap's σ_max/σ_min must stay below it.
  double conditionNumberCap = 1e8;
  /// Absolute floor on each endpoint band's isolation min(lowerGap,
  /// upperGap).  0 = rely on the bands' own certification.
  double minEndpointGap = 0.0;
  /// Require both endpoint bands to be certificate-accepted (a closing
  /// gap makes a band uncertified, which rejects the transport here).
  bool requireCertifiedFibers = true;
  /// Tolerance the emitted #764 certificates hold against (polar /
  /// pseudo-unitary residuals, winding closure defects).
  double certificateTolerance = 1e-10;
  /// Relative endpoint-mismatch cap for a certified matched-reference /
  /// closed-family winding closure.
  double closureTolerance = 1e-9;
};

/// # FiberTransportRead
///
/// One derived fiber transport A ← B (design spec §6.6): the raw overlap
/// map, every pre-normalization diagnostic, the normalized factor when its
/// gates passed, the determinant-line datum, and the #764 certificate.
/// Quantities that were not measured are quiet NaN, never zero.
struct FiberTransportRead {
  /// Order-independent identifier of the DESTINATION fiber A (the
  /// `Fingerprint::fingerprintOf` hash of its deduplicated cell-vertex-id
  /// set — the `AnalyticCache::componentKey` convention).
  std::uint64_t toKey = 0;
  /// Order-independent identifier of the SOURCE fiber B.
  std::uint64_t fromKey = 0;
  /// Form degree of the two bands.
  int degree = 0;
  /// Common frame rank r (columns of both frames); the reported reads are
  /// r×r.  When the ranks disagree the read is rejected and `rank` holds
  /// the DESTINATION rank while `rawMap` stays rectangular.
  int rank = 0;
  /// The raw overlap M_AB = Φ_A†W_A T_AB Φ_B (Ψ_A† on the biorthogonal
  /// path) BEFORE any normalization.
  Eigen::MatrixXcd rawMap{};
  /// Singular values of `rawMap`, descending.
  std::vector<double> singularValues{};
  /// Numerical rank of `rawMap` at the configured relative tolerance.
  int numericalRank = 0;
  /// Regime-appropriate isometry leakage: ‖M†M − I‖₂ (positive /
  /// non-normal report) or the Krein J-isometry defect ‖M†J_A M − J_B‖₂.
  double leakage = std::numeric_limits<double>::quiet_NaN();
  /// σ_max/σ_min of the overlap (∞ when singular).
  double overlapConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// Endpoint band isolation min(lowerGap, upperGap) of the destination A.
  double toGap = std::numeric_limits<double>::infinity();
  /// Endpoint band isolation min(lowerGap, upperGap) of the source B.
  double fromGap = std::numeric_limits<double>::infinity();
  /// Krein inertia p of the destination band (from its certificate).
  int toPositiveSignature = 0;
  /// Krein inertia q of the destination band.
  int toNegativeSignature = 0;
  /// Krein inertia p of the source band.
  int fromPositiveSignature = 0;
  /// Krein inertia q of the source band.
  int fromNegativeSignature = 0;
  /// Destination band condition number (‖P‖₂ from its certificate).
  double toConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// Source band condition number (‖P‖₂ from its certificate).
  double fromConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// max of the endpoint condition numbers — the spec §6.6 field.
  double frameConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// The metric regime this transport was computed in (the paired
  /// endpoint regimes; NonNormal whenever either endpoint is).
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// The normalized factor: the polar V_AB ∈ U(r) in the positive regime,
  /// the pseudo-unitary V (V†J_A V = J_B) on matching Krein signatures.
  /// EMPTY (0×0) when not emitted — a rejected map or the certified
  /// GL(r,C) non-normal transport, which deliberately retains `rawMap`.
  Eigen::MatrixXcd unitaryMap{};
  /// det of the emitted factor (∈ U(1) to `determinantResidual`); for a
  /// certified GL transport the PHASE det(M)/|det M| of the raw map — the
  /// determinant phase is never discarded.  0 only when nothing could be
  /// measured (rank-deficient raw map).
  std::complex<double> determinantPhase{0.0, 0.0};
  /// ‖V†V − I‖₂ (positive) or ‖V†J_A V − J_B‖₂ (Krein) of the emitted
  /// factor; NaN when no factor was emitted.
  double polarResidual = std::numeric_limits<double>::quiet_NaN();
  /// | |det V| − 1 | of the emitted factor; NaN when none / GL-only.
  double determinantResidual = std::numeric_limits<double>::quiet_NaN();
  /// True when a rank-three factor was emitted whose determinant-line
  /// datum failed its residual check while the projective class remains
  /// certified — the read is then trustworthy only in PU(3).
  bool projectiveOnly = false;
  /// Whether the transport passed every applicable gate (positive/Krein:
  /// reduction emitted; non-normal: certified GL transport).
  bool accepted = false;
  /// Human-readable reason of the FIRST failed gate ("" when accepted).
  std::string rejectionReason{};
  /// The graded #764 record: BandWindow domain, the detected regime,
  /// CertifiedNumerical with the reduction residual (or the GL rank/
  /// conditioning claim); a rejected read carries HeuristicDiscovery,
  /// which never holds.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary (direction, rank, leakage, gates).
  [[nodiscard]] std::string describe() const;

  /// Checkpoint serialization (design spec section 20, the `transports`
  /// array): the JSON-able :class:`Record` of the read — at rank three the
  /// full U(3) factor, det V (U(1)), and thereby the PU(3) class (the
  /// class is `[V]` = V modulo center, determined by the serialized V) all
  /// travel; complex leaves split `{name}_re`/`{name}_im` per #580.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate from `toRecord()` output; rejects an unknown
  /// `schema_version` (std::invalid_argument) per the checkpoint-reader
  /// contract.
  [[nodiscard]] static FiberTransportRead fromRecord(const Record &record);
};

/// # WilsonHolonomyRead
///
/// The product of accepted transports around a loop (or along an open
/// composite): the full holonomy, its normalized trace, determinant line,
/// and center-blind adjoint reads (design spec §12 step 8).
struct WilsonHolonomyRead {
  /// Common rank r of every link.
  int rank = 0;
  /// Number of links multiplied.
  std::size_t loopLength = 0;
  /// Whether the links chain (link i's source = link i+1's destination)
  /// and the last source equals the first destination.  Base-point
  /// conjugation covariance is a CLOSED-loop statement.
  bool closed = false;
  /// The base component (first link's destination key).
  std::uint64_t baseKey = 0;
  /// H(γ) = Π_i V_i — the full U(r) holonomy when every link carried a
  /// unitary/pseudo-unitary factor, the certified GL(r,C) product
  /// otherwise (see `unitary`).
  Eigen::MatrixXcd holonomy{};
  /// Tr H / r — conjugation-invariant on a closed loop.
  std::complex<double> normalizedTrace{0.0, 0.0};
  /// det H — the determinant-line observable (∈ U(1) when `unitary`).
  std::complex<double> determinant{0.0, 0.0};
  /// The center-blind adjoint character |Tr H|² − 1 (χ_adj at rank 3;
  /// the analogous traceless-sector character at generic rank).
  std::complex<double> adjointTrace{0.0, 0.0};
  /// Rank 3 only: the faithful PU(3) image — the 9×9 matrix of
  /// M ↦ H M H† restricted to the traceless octet by
  /// `ColorFiber::adjointOctetProjector` (empty at other ranks).
  Eigen::MatrixXcd adjointMatrix{};
  /// Metric-appropriate isometry defect of the product: ‖H†H − I‖₂ in the
  /// positive regime (and reported for GL products), the base-point
  /// J-isometry defect ‖H†J H − J‖₂ on a Krein loop — a pseudo-unitary
  /// product is exactly J-unitary and is never graded against the
  /// Euclidean metric.
  double unitarityResidual = std::numeric_limits<double>::quiet_NaN();
  /// True when every link supplied a unitary/pseudo-unitary factor.
  bool unitary = false;
  /// CertifiedNumerical/BandWindow in the links' shared regime.
  cobordism::Certificate certificate{};
};

/// # FundamentalLiftRead
///
/// The explicitly lifted SU(3) fundamental holonomy: a cube-root branch
/// continued from a declared base branch, with the accumulated Z₃ center
/// sector recorded (design spec §12 step 7).  Requested at rank three
/// only — SU(3) is never hard-coded at generic rank.
struct FundamentalLiftRead {
  /// Rank of the links (3 when valid).
  int rank = 0;
  /// The declared base branch s₀ ∈ {0, 1, 2}.
  int baseBranch = 0;
  /// H̃ = H · e^{−iΘ/3} · ω^{−s₀} ∈ SU(3), Θ the accumulated determinant
  /// phase; empty when `valid` is false.
  Eigen::MatrixXcd lift{};
  /// Tr H̃ — the branch-DEPENDENT fundamental Wilson value (shifts by
  /// ω^{−s₀} across branches).
  std::complex<double> liftTrace{0.0, 0.0};
  /// The accumulated Z₃ center sector m mod 3, m = (Θ − Arg e^{iΘ})/2π —
  /// branch-independent, the recorded "measured sector".
  int centerSector = 0;
  /// Θ = Σ_j Arg det V_j (per-link principal phases, running sum).
  double accumulatedDeterminantPhase = 0.0;
  /// max_j |Arg det V_j| — how far any single link sits from the branch
  /// cut (π means an ambiguous link).
  double maxDeterminantPhaseStep = 0.0;
  /// |det H̃ − 1|.
  double detResidual = std::numeric_limits<double>::quiet_NaN();
  /// Whether the lift was emitted (rank 3, every link accepted+unitary).
  bool valid = false;
  /// Why not, when not ("" when valid).
  std::string invalidReason{};
  /// CertifiedNumerical against `detResidual` when valid.
  cobordism::Certificate certificate{};

  /// Checkpoint serialization: the lift matrix and its ACCUMULATED center
  /// sector travel together (the ticket's "continuously chosen SU(3) lift
  /// with its accumulated center sector").
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static FundamentalLiftRead fromRecord(const Record &record);
};

/// The declared closure of an open-segment determinant winding (design
/// spec §5.11): HOW the open composite is closed is part of the
/// certificate.  With `Mode::None` the winding is left unknown.
struct WindingClosureSpec {
  /// The declared closure convention.
  enum class Mode {
    /// No closure declared — the winding is reported UNKNOWN (a raw
    /// endpoint phase difference is never promoted to an integer).
    None,
    /// Close with the inverse of a matched reference transport family
    /// (caller-supplied, e.g. a non-exchanging reference construction):
    /// the composite traverses the segment forward and the reference
    /// backward; the endpoint mismatch is the closure defect.
    MatchedReference,
    /// Close through fixed endpoint trivializations supplied by the
    /// boundary registers: four principal determinant legs
    /// T₀ → V(0) → … → V(n−1) → T₁ → T₀ close the phase path exactly.
    EndpointTrivialization,
  };
  /// The declared closure mode (see `Mode`).
  Mode mode = Mode::None;
  /// Caller-supplied identifier of the reference specification, recorded
  /// verbatim on the read.
  std::string referenceId{};
  /// MatchedReference: the reference transports over the SAME parameter
  /// samples, same orientation as the segment (traversed backwards by the
  /// closure).
  std::vector<Eigen::MatrixXcd> referenceTransports{};
  /// EndpointTrivialization: the register-supplied r×r frame at the START
  /// of the segment.
  Eigen::MatrixXcd startTrivialization{};
  /// EndpointTrivialization: the register-supplied r×r frame at the END
  /// of the segment.
  Eigen::MatrixXcd endTrivialization{};
};

/// # DeterminantWindingRead
///
/// The integer determinant winding of a closed full-rank transport family,
/// or the relative winding of an open segment under a recorded closure
/// (design spec §5.11 / §12 step 9).  `winding` is EMPTY when invalidated
/// (a closed gap / lost rank / aliasing step) or when no closure was
/// declared — never a silently wrong integer.
struct DeterminantWindingRead {
  /// ν ∈ ℤ, or nullopt (unknown / invalidated).
  std::optional<int> winding{};
  /// "closed-family", "matched-reference", "endpoint-trivialization", or
  /// "none" — the recorded closure specification (spec §6.6 field).
  std::string windingClosure{"none"};
  /// The caller's reference identifier ("" when none).
  std::string windingReferenceId{};
  /// Total unwrapped determinant phase of the CLOSED composite (2πν when
  /// valid); for an undeclared closure, the raw open-path phase — which
  /// is deliberately NOT an integer certificate.
  double accumulatedPhase = 0.0;
  /// Largest single principal step (radians) across the composite,
  /// closure legs included — the aliasing guard (must stay < π).
  double maxPhaseStep = 0.0;
  /// maxPhaseStep / π (the `phaseStepMargin` convention of
  /// `RecursiveQuotient::MultiplicityRead`).
  double phaseStepMargin = 0.0;
  /// Relative closure mismatch: 0 structurally for a closed family (the
  /// cycle closes by construction — the n−1 → 0 step is an ordinary
  /// phase step), segment-vs-reference endpoint matrix mismatch for
  /// matched-reference, trivialization unitarity defect for endpoint
  /// trivializations.
  double closureDefect = 0.0;
  /// Why the winding is absent, when it is ("" otherwise).
  std::string invalidationReason{};
  /// CertifiedNumerical/BandWindow with residual = closure defect and the
  /// step margin as conditioning; HeuristicDiscovery when invalidated.
  cobordism::Certificate certificate{};

  /// Checkpoint serialization: the closure SPECIFICATION travels with the
  /// integer (spec section 5.11 — the closure is part of the certificate);
  /// an unknown winding serializes as unknown, never as zero.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static DeterminantWindingRead fromRecord(const Record &record);
};

/// # FiberConnection
///
/// Derived spectral-frame transport and Wilson observables (ticket #770;
/// design spec §12, Algorithm E).  See the file banner for the exact
/// identities and their domains.  Pure read layer: consumes accepted #769
/// `SpectralFiber`s and existing chain/response operators, mutates
/// nothing, and none of its outputs enters any emergence objective.
class FiberConnection {
  public:
    /// `AnalyticCache` kind string of a cached transport read.
    static constexpr const char *kTransportCacheKind = "fiber-transport";
    /// `AnalyticCache` kind string of a cached Wilson loop product.
    static constexpr const char *kHolonomyCacheKind = "wilson-product";

    /// Bind the gate thresholds this connection derives transports under.
    explicit FiberConnection(FiberConnectionConfig cfg = {});

    /// The threshold configuration this instance gates with.
    [[nodiscard]] const FiberConnectionConfig &config() const noexcept {
      return cfg_;
    }

    // ── chain-transfer sources (wrappers over existing machinery) ──────

    /// The chain transfer T_AB induced by the connecting simplices: the
    /// off-diagonal block L_k[cells(to), cells(from)] of the whole-complex
    /// weighted Hodge operator (see the file banner for the exact block
    /// identity).  Cells are matched by sorted vertex-id tuple against the
    /// canonical `ChainComplex` order — no vertex order is ever imposed.
    /// @throws std::invalid_argument for an unknown cell or negative
    ///   degree.
    [[nodiscard]] static Eigen::MatrixXcd chainTransfer(
        const std::shared_ptr<Spacetime> &st, int degree,
        const std::vector<std::vector<std::uint64_t>> &toCells,
        const std::vector<std::vector<std::uint64_t>> &fromCells,
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention());

    /// The effective response block of an existing #768 response network:
    /// the edge block with rows = `toComponent`'s stalk and columns =
    /// `fromComponent`'s stalk (zero block of the right shape when the
    /// network carries no such edge).
    /// @throws std::out_of_range on a bad component index.
    [[nodiscard]] static Eigen::MatrixXcd responseTransfer(
        const cobordism::RecursiveQuotient::ResponseNetworkRead &network,
        int toComponent, int fromComponent);

    // ── the derived transport ───────────────────────────────────────────

    /// Derive the transport A ← B from an explicit transfer block
    /// (rows = A's cells, columns = B's cells): compute the overlap in the
    /// paired regime, report EVERY diagnostic, gate, and only then reduce
    /// (Algorithm E steps 2–7).  A rejected read still carries the raw
    /// map, singular values, leakage, gaps, signatures, and conditioning.
    /// @throws std::invalid_argument on a transfer/frame shape mismatch.
    [[nodiscard]] FiberTransportRead transport(
        const SpectralFiber &to, const SpectralFiber &from,
        const Eigen::MatrixXcd &transfer) const;

    /// `transport` with the reverse-oriented transfer: the W-adjoint
    /// reverse block T_BA = W_B⁻¹ T_AB† W_A, which is the exact reverse
    /// chain transfer whenever the underlying operator is W-self-adjoint
    /// (the positive and Krein regimes) — there reversing a link returns
    /// the adjoint/inverse: M_BA = M_AB† and V_BA = V_AB†.
    [[nodiscard]] FiberTransportRead transportReverse(
        const SpectralFiber &to, const SpectralFiber &from,
        const Eigen::MatrixXcd &transfer) const;

    /// Derive the transport A ← B on a spacetime: assembles the chain
    /// transfer from the Hodge operator and calls `transport`.  Both
    /// fibers must be same-degree bands of components of `st`.
    [[nodiscard]] FiberTransportRead transportOnSpacetime(
        const std::shared_ptr<Spacetime> &st, const SpectralFiber &to,
        const SpectralFiber &from,
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention()) const;

    /// `transportOnSpacetime` through the #764 `AnalyticCache` contract:
    /// served while both fibers' cell-vertex stars are untouched,
    /// recomputed (and re-stored) otherwise.  Key: the union of the two
    /// fibers' cell-vertex-id sets; kind `kTransportCacheKind`;
    /// parameter = degree.  Cached results equal cold recomputation.
    [[nodiscard]] FiberTransportRead transportOnSpacetimeCached(
        cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
        const SpectralFiber &to, const SpectralFiber &from,
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention()) const;

    // ── Wilson observables ──────────────────────────────────────────────

    /// Multiply accepted transports along `links` (link i maps its source
    /// fiber to its destination; the product H = V₀V₁⋯V_{n−1} maps the
    /// LAST source into the FIRST destination — the base component).
    /// Emits the full holonomy, normalized trace, determinant line, and
    /// the center-blind adjoint reads; `closed` reports whether the keys
    /// chain into a loop.  Uses the unitary factors when every link has
    /// one, the certified GL raw maps otherwise — never a mixture.
    /// @throws std::invalid_argument on an empty chain, a rank mismatch,
    ///   or an unaccepted link (only ACCEPTED maps are multiplied).
    [[nodiscard]] WilsonHolonomyRead holonomy(
        const std::vector<FiberTransportRead> &links) const;

    /// Wilson loop over an ordered cycle of fibers on a spacetime:
    /// links fibers[0] ← fibers[1] ← … ← fibers[n−1] ← fibers[0], each
    /// derived by `transportOnSpacetime`, then the product.
    /// @throws std::invalid_argument when fewer than two fibers.
    [[nodiscard]] WilsonHolonomyRead holonomyOnSpacetime(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<SpectralFiber> &fibers,
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention()) const;

    /// `holonomyOnSpacetime` through the `AnalyticCache`: each link via
    /// `transportOnSpacetimeCached`, and the loop product itself cached
    /// under the union of ALL participating fibers' vertex sets (kind
    /// `kHolonomyCacheKind`; parameter = an order-sensitive hash of the
    /// fiber sequence folded with the degree, so distinct loop orders
    /// never collide).  A published `TouchedStar` invalidates ONLY the
    /// loops (and links) whose fibers meet the star.
    [[nodiscard]] WilsonHolonomyRead holonomyOnSpacetimeCached(
        cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
        const std::vector<SpectralFiber> &fibers,
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention()) const;

    // ── rank-three center structure ─────────────────────────────────────

    /// A canonical PU(3) class representative of an emitted rank-three
    /// factor: V·δ^{−1/3} with the PRINCIPAL cube root of δ = det V.  The
    /// projective CLASS {U, ωU, ω²U} is the faithful datum; this fixes
    /// one representative deterministically (tests exercise all three
    /// branches via `fundamentalLift`).
    /// @throws std::invalid_argument unless `unitary` is 3×3.
    [[nodiscard]] static Eigen::MatrixXcd projectiveRepresentative(
        const Eigen::MatrixXcd &unitary);

    /// The faithful PU(3) image of a 3×3 unitary: the 9×9 matrix of
    /// M ↦ U M U† restricted to the traceless octet
    /// (`ColorFiber::adjointOctetProjector` — the #767 conventions are
    /// consumed, never reimplemented).  Center-blind: Ad(ωU) = Ad(U).
    /// @throws std::invalid_argument unless `unitary` is 3×3.
    [[nodiscard]] static Eigen::MatrixXcd adjointRepresentation(
        const Eigen::MatrixXcd &unitary);

    /// Continue a cube-root branch along `links` from the declared base
    /// branch and RECORD the accumulated Z₃ center sector (file banner
    /// for the exact lift identity).  Requires rank three, unitary links,
    /// and the POSITIVE regime; an invalid request reports
    /// `valid = false` with the reason — SU(3) is never emitted at
    /// generic rank, from a GL transport, or from a pseudo-unitary
    /// (Krein) factor outside U(3).
    /// @throws std::invalid_argument when `baseBranch` ∉ {0, 1, 2} or
    ///   `links` is empty.
    [[nodiscard]] FundamentalLiftRead fundamentalLift(
        const std::vector<FiberTransportRead> &links,
        int baseBranch = 0) const;

    // ── determinant winding ─────────────────────────────────────────────

    /// The integer determinant winding of a CLOSED transport family
    /// (world-tube samples V(t₀), …, V(t_{n−1}), traversed cyclically —
    /// the closing step returns to sample 0).  Invalidated (nullopt +
    /// reason) when any sample is unaccepted (gap/rank closed), ranks
    /// disagree, or a phase step reaches π.
    /// @throws std::invalid_argument on an empty family.
    [[nodiscard]] DeterminantWindingRead closedFamilyWinding(
        const std::vector<FiberTransportRead> &family) const;

    /// The RELATIVE determinant winding of an OPEN cobordism segment
    /// under the declared closure (spec §5.11): matched-reference or
    /// endpoint-trivialization, with the specification recorded on the
    /// read.  `Mode::None` reports the raw open-path phase and an UNKNOWN
    /// winding.  Same invalidation rules as `closedFamilyWinding`.
    /// @throws std::invalid_argument on an empty segment or a malformed
    ///   closure (wrong reference length / trivialization shape).
    [[nodiscard]] DeterminantWindingRead openSegmentWinding(
        const std::vector<FiberTransportRead> &segment,
        const WindingClosureSpec &closure) const;

    // ── shared key helpers ──────────────────────────────────────────────

    /// The order-independent key of a fiber: `Fingerprint::fingerprintOf`
    /// over its DEDUPLICATED cell-vertex-id set (the `AnalyticCache`
    /// component-key convention; relabeling-covariant, order-invariant).
    [[nodiscard]] static std::uint64_t fiberKey(const SpectralFiber &fiber);

  private:
    FiberConnectionConfig cfg_{};

    /// Deduplicated, sorted union of the fibers' cell vertex ids (the
    /// cache component key material).
    [[nodiscard]] static std::vector<std::uint64_t> unionVertexIds(
        const std::vector<const SpectralFiber *> &fibers);

    /// Per-regime overlap + gates + reduction core.
    [[nodiscard]] FiberTransportRead deriveTransport(
        const SpectralFiber &to, const SpectralFiber &from,
        const Eigen::MatrixXcd &transfer) const;

    /// Shared winding core: gates the family, walks the principal legs of
    /// the (cyclic or closure-completed) determinant path, and grades the
    /// result.  `closure` is null for the cyclic closed-family read.
    [[nodiscard]] DeterminantWindingRead windingRead(
        const std::vector<FiberTransportRead> &family, bool cyclic,
        const WindingClosureSpec *closure) const;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_FIBERCONNECTION_H
