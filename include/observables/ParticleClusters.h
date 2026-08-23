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
//                              pair conservation.  #774 adds the EVEN
//                              sector reads BESIDE the quark surface: the
//                              quasi-free octet bilinear read on the #780
//                              covariance layer, and the gluon-candidate /
//                              meson-candidate / diquark-candidate
//                              classifiers (see below).  #775 adds the
//                              THREE-cluster sector beside these: the
//                              bound-supercomponent search, the color
//                              singlet, and the complete proton
//                              certificate (see below).
//   • OctetBilinearRead      — the #774 quasi-free traceless-bilinear read
//                              of three declared color modes of a carried
//                              #780 CovarianceState: the bilinear matrix
//                              M_ij = ⟨a_i†a_j⟩ = (Γ_S)ᵀ, its exact
//                              1 ⊕ 8 split (delegated to
//                              ColorFiber::octetRead / tracelessPart /
//                              adjointOctetProjector — never re-derived),
//                              the adjoint Casimir, the quartic-Wick color
//                              Casimir expectation, subset occupation and
//                              parity.  Dense Fock vectors appear only in
//                              tests; #771 stays the oracle layer.
//   • GluonCandidateEvidence / GluonRead
//                            — the #774 gluon-candidate classifier: a
//                              persistent transported octet excitation
//                              with certified even parity, certified ZERO
//                              total determinant winding (zero baryon
//                              flux), and accepted rank-three transports.
//                              "Candidate" is the strongest claim ever
//                              emitted — no even octet excitation is
//                              classified a physical gluon.
//   • CompositeCandidateEvidence / MesonRead / DiquarkRead
//                            — the #774 TWO-cluster composite classifiers
//                              over constituent QuarkReads: meson = one
//                              certified quark + one certified antiquark,
//                              even composite parity (the exact graded
//                              product of the certified constituent
//                              parities), color-singlet pairing (the 1 ⊕ 8
//                              split of the caller-assembled pair
//                              bilinear), and zero total winding/baryon
//                              flux; diquark = two certified quarks, even
//                              parity, a certified Λ²C³ anti-triplet
//                              wedge occupation (#780 Gram determinant —
//                              exactly zero for duplicated color modes),
//                              and the PRESERVED constituent baryon flux
//                              B = 2/3 (≠ an antiquark's −1/3; occupation
//                              two and even parity are the recorded
//                              distinction channels, exactly the #773
//                              anti-triplet fixture).
//   • BoundCandidateEvidence / BoundSupercomponentRead
//                            — the #775 bound-supercomponent search
//                              (design spec §16.2): enumerate the NEXT
//                              modular level for components containing
//                              exactly three certified quark candidates
//                              whose #765 persistence windows overlap and
//                              whose mutual #770 transports stay bounded
//                              inside the supercomponent.
//   • ScaleProfileSample / ScaleProfileRead
//                            — the #775 refinement-window read over the
//                              EXISTING #575/#566/#593 mass-radius
//                              battery (`InteriorHinges` via
//                              `RegisterContext::interiorHinges`): a
//                              finite emergent radius plus the
//                              refinement stability of the DIMENSIONLESS
//                              channels.  See "The form-factor situation"
//                              below — nothing here is a form factor.
//   • BaryonCandidateEvidence / BaryonRead
//                            — the #775 THREE-cluster classifier and the
//                              complete proton certificate: the invariant
//                              color volume det C and Gram determinant
//                              det(C†C), the independent vanishing
//                              net-color-flux diagnostic, summed certified
//                              determinant winding and exterior parity,
//                              the reused #773 flavor/charge reads, the
//                              #772 Berry-cancelled 2π character and spin
//                              lift, and the SHARP total-space spin
//                              certificate ⟨J²⟩ = 3/4 with Var(J²) ≈ 0
//                              evaluated by exact Wick contraction on the
//                              #780 covariance layer.  Four verdicts:
//                              "no-baryon", "baryon-candidate",
//                              "certified-proton", and
//                              "quasi-free-sharp-spin-obstruction".
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
// ─── #775: the three-cluster (baryon / proton) identities ────────────────
//
//   • Invariant color volume: S_ABC = det[c_A c_B c_C] = ε_ijk c_A^i c_B^j
//     c_C^k, delegated VERBATIM to `ColorFiber::colorWedge`; invariant
//     under a common g ∈ SU(3) because det(gC) = det(g) det(C) = det(C).
//     Its squared magnitude is the singlet Gram certificate
//     |S_ABC|² = det(C†C) ∈ [0, 1] (`ColorFiber::singletGram`) — exactly
//     one for an orthonormal anchored triad, exactly zero for duplicated
//     color modes (the Pauli/Gram alternation).  Domain: three normalized
//     color columns.
//   • THE WEDGE IS BUILT ONCE.  The antisymmetry of the three-mode wedge
//     IS the ε tensor inside the determinant: no extra fermion permutation
//     sign is ever multiplied onto it.  A transposition of two color
//     columns therefore flips `colorWedge` (det C) and leaves the SINGLET
//     certificate det(C†C) = |det C|² exactly invariant; the composite's
//     fermionic statistics come from the graded product of the three
//     CONSTITUENT parities, never from re-signing the wedge.
//   • Net color flux: the octet (traceless) Frobenius weight
//     ‖M − (Tr M/3) I‖_F² of the bound object's color bilinear
//     M_ij = ⟨a_i†a_j⟩ under the exact 1 ⊕ 8 split — the #774
//     `octetBilinearRead` REUSED, not re-derived.  Zero octet weight = no
//     net color polarization.  This is an INDEPENDENT diagnostic on a
//     FINITE complex: it is deliberately not presented, by itself, as a
//     proof of confinement, and no method here makes that claim.
//   • Composite baryon flux: ν = ν_A + ν_B + ν_C over the three CERTIFIED
//     determinant windings (each carrying its own recorded #770 closure
//     specification), provisionally B = ν/3 — so a certified proton reads
//     ν = 3 ⇒ B = +1.  An uncertified leg leaves the total UNKNOWN (null),
//     never zero: the closure specification is part of the certificate
//     (spec §5.11), and this class only ever sums integers that already
//     carry one.
//   • Composite exterior parity: the exact graded product of the three
//     certified constituent parities (parity adds mod 2) — odd for three
//     odd clusters.  Unknown when any constituent parity is uncertified.
//   • Flavor / charge: the #773 constituent reads CONSUMED VERBATIM.  The
//     `uud` pattern is the certified-isospin multiset {+1/2, +1/2, −1/2}
//     under each constituent's own recorded doublet orientation, and the
//     electric flux is the sum of the three CERTIFIED Gauss-consistent
//     constituent fluxes (2/3 + 2/3 − 1/3 = +1 for a proton).  Nothing is
//     re-derived and no u/d label is inserted.
//   • Sharp total-space spin: ⟨J²⟩ = Σ_α ⟨dΓ(J_α)²⟩ and
//     Var(J²) = ⟨(J²)²⟩ − ⟨J²⟩², both exact finite Wick sums on the #780
//     covariance (`CovarianceState::wickSpinSquaredExpectation` /
//     `wickSpinSquaredVariance`) — the TOTAL-SPACE operator, never a
//     product of per-hole or per-edge spinors.  A candidate carried as an
//     explicit composite state instead supplies the #772 dense oracle
//     (`ExchangeHolonomy::totalJSquared`), which certifies ⟨J²⟩ and leaves
//     Var(J²) UNKNOWN — expectation alone is never a sharp-spin
//     certificate (spec §5.12).
//   • The reference-normalized 2π character is the #772 `rotationCharacter`
//     read (channel `PhysicalRotation`), required at characterSign = −1;
//     the SO(d) → Spin(d) lift is required only when the caller declares a
//     CONTINUUM spin claim (spec §16.4: "accepted when a continuum spin
//     claim is made").  The #772 PARTICLE-EXCHANGE channel is reused as a
//     REPORT channel only — `exchangeCharacter` and the doubly cancelled
//     spin-statistics ratio χ̂(exchange)·χ̂(2π)^{-1} travel on the read,
//     but neither the ticket's proton-certificate list nor spec §16.4
//     carries an exchange row, so neither gates the verdict.  A read
//     tagged with the wrong channel is refused, never reinterpreted.
//   • Verdicts (spec §16.4): "no-baryon" when the structural gates fail;
//     "certified-proton" when every certificate in the table holds;
//     "quasi-free-sharp-spin-obstruction" when the ONLY failure is
//     `sharp-spin` AND the accepted covariance-only class was swept and its
//     Var(J²) floor stays above tolerance; "baryon-candidate" otherwise.
//     The obstruction is a BRANCH POINT mandating an explicit non-Gaussian
//     mechanism (its own scope decision and ticket) — never a refutation of
//     the geometry, and nothing here silently adds such a mechanism.
//
// ─── #775: the form-factor situation (stated plainly) ────────────────────
//
// The ticket and the whitepaper ask for "stable spectral-mass/form-factor
// reads".  THERE IS NO FORM-FACTOR IMPLEMENTATION IN THIS REPOSITORY, and
// none is invented here.  What the refinement certificate actually consumes
// is the EXISTING #575/#566/#593 mass-radius battery (`InteriorHinges`,
// read through `RegisterContext::interiorHinges`, exactly what
// `EmergentRadius` / `EmergentMass` read):
//
//   • the emergent radius r = V_dual^{1/4} (DIMENSIONFUL, lattice units —
//     only its FINITENESS is certified, never an absolute value);
//   • the mean interior deficit angle m_shell (an angle: dimensionless in
//     lattice units) as the spectral-mass channel;
//   • the participation ratio of the curvature weight; and
//   • `radialWeightProfile` — the per-BFS-shell share of the |Re ε · ★h|
//     curvature weight: a normalized RADIAL CURVATURE-WEIGHT DENSITY.
//
// `radialWeightProfile` is NOT a form factor.  No momentum-transfer Fourier
// transform of a charge density is computed anywhere in this tree, no
// charge radius is extracted from a slope at q² → 0, and the profiled
// density is CURVATURE weight, not electromagnetic charge.  The certificate
// it grades is named `profile-stability`, never "form-factor": the ticket's
// form-factor requirement is met ONLY in the weaker sense of "a
// refinement-stable dimensionless radial profile", and that gap is recorded
// here rather than papered over.
//
// Any DIMENSIONFUL mass stays UNKNOWN: `ScaleProfileRead::physicalMass` and
// `BaryonRead::physicalMass` are ALWAYS empty, because converting a
// deficit-angle reading into a physical mass needs a physical scale this
// program has not independently established.
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
// No proton certificate, and no quantity derived from one, enters any
// emergence objective; software completion is not evidence that an unforced
// proton exists.

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "observables/ColorFiber.h"
#include "observables/ExchangeHolonomy.h"
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

class RegisterContext;  // the #593 validated read context (RegisterContext.h)

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

  // ── #774 even-sector thresholds ────────────────────────────────────────

  /// Floor on the octet Frobenius weight ‖M − (Tr M/3) I‖_F² of a gluon
  /// candidate's bilinear — a genuinely nonzero color polarization (the
  /// vacuum reads exactly zero and fails by name).
  double minOctetWeight = 1e-9;
  /// Cap on the octet-projector residual ‖(I₉ − P₈) vec(M₈)‖ / ‖M₈‖_F of
  /// the excitation: the traceless bilinear lies in the 8 EXACTLY (the
  /// residual is rounding), so this is a machine-precision certificate.
  double octetPurityTolerance = 1e-9;
  /// Cap on the octet fraction octet/(octet + singlet) of a meson's pair
  /// color bilinear — the color-SINGLET certificate of the q-q̄ composite.
  double compositeOctetTolerance = 1e-9;
  /// Floor on the certified Λ²C³ anti-triplet wedge occupation
  /// det(C†ΓC) ∈ [0, 1] of a diquark candidate (exactly 1 for the
  /// two-orthonormal-column Slater fixture, exactly 0 for a duplicated
  /// color mode — the Pauli/Gram identity).
  double minAntiTripletWeight = 0.5;

  // ── #775 baryon / proton thresholds ────────────────────────────────────

  /// |det(C†C) − 1| cap of the COLOR-SINGLET certificate of the three
  /// normalized anchored color columns (`ColorFiber::singletGram`):
  /// exactly 1 for an orthonormal triad, exactly 0 for duplicate color
  /// modes, so this only absorbs rounding.
  double colorGramTolerance = 1e-9;
  /// Cap on the composite's NET COLOR FLUX — the octet (traceless)
  /// Frobenius weight of the bound object's color bilinear.  An
  /// INDEPENDENT confinement diagnostic on a finite complex, never a
  /// proof of confinement on its own.
  double colorFluxTolerance = 1e-9;
  /// |⟨J²⟩ − 3/4| cap of the total-space spin expectation.
  double spinExpectationTolerance = 1e-9;
  /// |Var(J²)| cap of the SHARP-spin certificate (design spec §5.12): the
  /// quantity that separates a proton certificate from an accidental
  /// expectation value.
  double spinVarianceTolerance = 1e-9;
  /// Minimum fraction of a constituent's level-0 support that must lie
  /// inside a candidate supercomponent (1.0 = full containment).
  double minSupportContainment = 1.0;
  /// Minimum number of SHARED persistence slices across the three
  /// constituents' #765 lifetime windows.
  double minLifetimeOverlap = 1.0;
  /// Strict floor a finite emergent radius must exceed.
  double minRadius = 0.0;
  /// Cap on the deviation of every DIMENSIONLESS scale channel across the
  /// configured refinement window (relative spread for the scalars,
  /// max absolute per-shell deviation for the radial weight profile).
  double maxProfileDeviation = 1e-6;
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

/// # OctetBilinearRead
///
/// The #774 quasi-free traceless-bilinear (octet) read of THREE declared
/// color modes of a carried #780 `quantum::CovarianceState` — evaluated on
/// the covariance layer (polynomial in the mode count, no Fock vector;
/// the #771 lazy engine remains the dense oracle in tests and the carrier
/// of explicitly non-Gaussian boundary data).
///
/// Exact identities (all delegations, tested to rounding):
///   • M_ij = ⟨a_i†a_j⟩ = Γ_{ji} on the declared modes — `bilinear` is the
///     transposed principal submatrix of Γ.
///   • 1 ⊕ 8 resolution: `singletWeight`/`octetWeight` are EXACTLY
///     `ColorFiber::octetRead(bilinear)`; `octetComponent` is EXACTLY
///     `ColorFiber::tracelessPart(bilinear)`, which lies in the octet BY
///     THE ALGEBRA — `octetProjectorResidual` measures
///     ‖(I₉ − P₈) vec(M₈)‖/‖M₈‖_F ≈ 0 (rounding only).
///   • `casimir` = `ColorFiber::adjointCasimir(octetComponent)` = 3 for a
///     nonzero excitation (C = 3 P₈).
///   • `casimirExpectation` = ⟨Σ_a dΓ(λ_a/2)²⟩ by quartic Wick sums
///     (#780 `wickBilinearMoment` with the embedded Gell-Mann halves):
///     exactly 0 on the vacuum and the fully occupied singlet, exactly
///     C₂ = 4/3 on the fundamental (N = 1) and anti-triplet (N = 2)
///     Slater states.
///   • `gellMannComponents[a-1]` = Tr(λ_a M)/2 — the octet coordinates:
///     M = (Tr M/3) I + Σ_a comp_a λ_a exactly.
///
/// Adding microscopic modes (vacuum embedding, #771 `embedInVacuum` or a
/// zero-block covariance extension) leaves this read unchanged: arbitrarily
/// many collective excitations are represented WITHOUT changing any
/// two-dimensional edge-mode factor.  Unknown values are NaN/0-sign, never
/// zero-filled.
struct OctetBilinearRead {
  /// The three declared color modes, in the caller's recorded order (the
  /// color trivialization; a compilation convention like a declared anchor
  /// weighting).
  std::vector<std::size_t> colorModes{};
  /// Certified subset occupation ⟨N_S⟩ = tr Γ_S (NaN when the constituent
  /// Wick reads did not certify).
  double occupation = std::numeric_limits<double>::quiet_NaN();
  /// Certified subset parity sign of ⟨(−1)^{N_S}⟩: +1 / −1 / 0 = unknown
  /// or indefinite (never a forced sign).
  int subsetParity = 0;
  /// The bilinear matrix M_ij = ⟨a_i†a_j⟩ on the declared modes.
  Eigen::Matrix3cd bilinear = Eigen::Matrix3cd::Zero();
  /// The traceless octet component `ColorFiber::tracelessPart(bilinear)`.
  Eigen::Matrix3cd octetComponent = Eigen::Matrix3cd::Zero();
  /// ‖M − (Tr M/3) I‖_F² — the octet Frobenius weight
  /// (`ColorFiber::octetRead`).
  double octetWeight = std::numeric_limits<double>::quiet_NaN();
  /// |Tr M|²/3 — the singlet (trace/number) Frobenius weight.
  double singletWeight = std::numeric_limits<double>::quiet_NaN();
  /// ‖(I₉ − P₈) vec(octetComponent)‖ / ‖octetComponent‖_F — rounding-level
  /// for any state (the excitation is octet by the algebra); NaN when the
  /// excitation vanishes (an undefined residual is unknown, never zero).
  double octetProjectorResidual = std::numeric_limits<double>::quiet_NaN();
  /// `ColorFiber::adjointCasimir(octetComponent)` ∈ [0, 3]; 3 exactly for
  /// a nonzero excitation; NaN when it vanishes.
  double casimir = std::numeric_limits<double>::quiet_NaN();
  /// ⟨Σ_a dΓ(λ_a/2)²⟩ — the quartic-Wick color Casimir expectation of the
  /// carried state on the declared modes (NaN when uncertified).
  double casimirExpectation = std::numeric_limits<double>::quiet_NaN();
  /// Tr(λ_a M)/2 for a = 1..8 — the octet coordinates of the bilinear.
  std::vector<std::complex<double>> gellMannComponents{};
  /// Max residual of the consumed #780 Wick reads (their certificates all
  /// travel through `certificate`).
  double residual = std::numeric_limits<double>::quiet_NaN();
  /// AlgebraicallyExact / Static in the covariance's verified regime
  /// (residual = the consumed Wick residual maximum against the #780 read
  /// tolerance): the read is a finite exact Wick sum on the covariance.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization (complex leaves split `{name}_re`/`{name}_im`
  /// per #580; unknown values serialize as null/NaN, never zero).
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static OctetBilinearRead fromRecord(const Record &record);
};

/// # GluonCandidateEvidence
///
/// The assembled evidence bundle of ONE gluon candidate (#774; design spec
/// §14.3): every field is a read produced by a merged upstream kernel —
/// the #780 carried-state Wick parity/occupation, this class's own
/// quasi-free octet bilinear read, the #770 lifetime transports and
/// determinant winding, and the #765 persistence diagnostics.  Missing
/// evidence fails its certificate BY NAME, it is never presumed.
struct GluonCandidateEvidence {
  /// The excitation's #765 label-free identity.
  ComponentId component{};
  /// The component the excitation is bound to / lives on (reported
  /// verbatim — the "binding component" of the ticket's report set).
  ComponentId bindingComponent{};
  /// The quasi-free octet bilinear read of the carried state
  /// (`octetBilinearRead`).
  OctetBilinearRead octet{};
  /// The #780 Wick parity ⟨(−1)^N⟩ of the WHOLE carried state
  /// (`CovarianceState::wickParity`) — the even-parity gate.
  quantum::WickCertificateRead parityRead{};
  /// The #780 Wick total occupation ⟨N⟩ (`wickTotalNumber`; report-only).
  quantum::WickCertificateRead occupationRead{};
  /// The candidate's lifetime transports (#770): every link must be
  /// accepted, rank three, with leakage under the configured cap — the
  /// "accepted octet transport" gate (the adjoint action on the octet is
  /// exact GIVEN the accepted rank-three factor:
  /// `FiberConnection::adjointRepresentation`).
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// The #770 determinant-line winding of the lifetime family — the gluon
  /// gate requires a CERTIFIED ν = 0 (zero baryon flux is evidence, never
  /// a default: an unknown winding leaves the flux unknown).
  DeterminantWindingRead winding{};
  /// #765 persistence-track lifetime (NaN = missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
};

/// # GluonRead
///
/// The #774 gluon-candidate read: a persistent transported octet
/// excitation with certified even parity and certified zero total
/// determinant winding / baryon flux.  `classification` is
/// "gluon-candidate" or "none" — NEVER "gluon": no even octet excitation
/// is claimed to be a physical gluon (ticket out-of-scope list).  Unknown
/// or uncertified values are NULL (empty optional / NaN / 0-sign), never
/// zero-filled, and every gap is NAMED in `failedCertificates`.
///
/// Certificate name vocabulary (fixed order): "parity-even",
/// "octet-excitation", "octet-purity", "octet-transport", "winding-zero",
/// "persistence".
struct GluonRead {
  /// The excitation's #765 identity.
  ComponentId component{};
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// "gluon-candidate" or "none".
  std::string classification{"none"};
  /// Certified carried-state exterior parity: +1 (even) / −1 (odd) / 0 =
  /// unknown.
  int exteriorParity = 0;
  /// ⟨N⟩ of the whole carried state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// Flat consumed-scalar summaries of the octet evidence (the QuarkRead
  /// anchor-profile convention — ONE source of truth: the full
  /// `OctetBilinearRead` travels on the EVIDENCE and serializes itself;
  /// the read carries only the consumed decision/report scalars, NaN when
  /// missing): the adjoint Casimir, the quartic-Wick color Casimir
  /// expectation, the octet-projector residual, and the 1 ⊕ 8 Frobenius
  /// weights.
  double casimir = std::numeric_limits<double>::quiet_NaN();
  double casimirExpectation = std::numeric_limits<double>::quiet_NaN();
  double octetProjectorResidual = std::numeric_limits<double>::quiet_NaN();
  double octetWeight = std::numeric_limits<double>::quiet_NaN();
  double singletWeight = std::numeric_limits<double>::quiet_NaN();
  /// The certified determinant winding (0 for a certified gluon
  /// candidate); EMPTY when invalidated/unclosed.
  std::optional<int> determinantWinding{};
  /// The recorded winding closure specification (#770 vocabulary).
  std::string windingClosure{"none"};
  /// The caller's closure reference identifier ("" when none).
  std::string windingReferenceId{};
  /// B = ν/3 under a CERTIFIED winding — 0.0 is a CERTIFIED zero flux;
  /// EMPTY (unknown) without the winding certificate, never zero.
  std::optional<double> baryonFlux{};
  /// Number of lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst lifetime transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// #765 persistence lifetime consumed (the "lifetime" report; NaN =
  /// missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the six gluon certificates; 1.0 exactly for a
  /// certified gluon candidate.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (exact boolean combination GIVEN the consumed held
  /// certificates) for a certified candidate; HeuristicDiscovery
  /// otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization (design spec §20 `particles.gluons`).
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static GluonRead fromRecord(const Record &record);
};

/// # CompositeCandidateEvidence
///
/// The assembled evidence bundle of ONE two-cluster composite (#774 meson
/// and diquark candidates; the THREE-cluster bundle is the #775
/// `BaryonCandidateEvidence` below).  The
/// constituents are #773 `QuarkRead`s CONSUMED VERBATIM — their windings,
/// parities, and certificates are never recomputed here; the composite
/// channels (carried-state occupation, the pair color bilinear, the
/// anti-triplet wedge read) are caller-assembled upstream reads.
struct CompositeCandidateEvidence {
  /// The component binding the two clusters (reported verbatim).
  ComponentId bindingComponent{};
  /// The first constituent's #773 read.
  QuarkRead first{};
  /// The second constituent's #773 read.
  QuarkRead second{};
  /// The #780 Wick total occupation ⟨N⟩ of the carried composite state
  /// (report-only; NaN/uncertified = unknown).
  quantum::WickCertificateRead occupationRead{};
  /// Meson channel: the pair color bilinear M_ij pairing constituent color
  /// i with conjugate color j (3 ⊗ 3̄) — a singlet composite has M ∝ I.
  /// ABSENT = no pairing evidence: the color-singlet certificate fails by
  /// name.
  std::optional<Eigen::Matrix3cd> colorPairing{};
  /// Diquark channel: the certified Λ²C³ wedge occupation det(C†ΓC) of
  /// the two constituent color columns on the carried state (#780
  /// `wickGramDeterminant` / `wickColorWedgeSquared` family) — exactly
  /// zero for duplicated color modes (Pauli).  Default-constructed =
  /// missing evidence.
  quantum::WickCertificateRead antiTripletRead{};
  /// The composite's lifetime transports (#770; report-only for the
  /// two-cluster reads — the max leakage travels on the read).
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// #765 persistence-track lifetime of the composite (NaN = missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
};

/// # MesonRead
///
/// The #774 meson-candidate read: one certified quark plus one certified
/// antiquark (order-insensitive), EVEN composite parity — the exact graded
/// product of the certified constituent parities (whitepaper "Fermion
/// statistics from simplicial orientation": parity adds mod 2) — a
/// color-SINGLET pair bilinear under the exact 1 ⊕ 8 split, and zero total
/// certified winding/baryon flux (the #773 conjugate-pair integer sums).
///
/// Certificate name vocabulary (fixed order): "constituent-quark",
/// "constituent-antiquark", "parity-even", "color-singlet", "flux-zero".
struct MesonRead {
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// The two constituents' #765 identities, in evidence order.
  ComponentId firstConstituent{};
  ComponentId secondConstituent{};
  /// "meson-candidate" or "none".
  std::string classification{"none"};
  /// The composite exterior parity: the exact product of the two certified
  /// constituent parities (+1 even / −1 odd / 0 = unknown).
  int exteriorParity = 0;
  /// ⟨N⟩ of the carried composite state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// The 1 ⊕ 8 Frobenius weights of the pair color bilinear
  /// (`ColorFiber::octetRead`; NaN when no pairing evidence).
  double pairingSingletWeight = std::numeric_limits<double>::quiet_NaN();
  double pairingOctetWeight = std::numeric_limits<double>::quiet_NaN();
  /// octet/(octet + singlet) of the pairing (NaN when missing or zero).
  double pairingOctetFraction = std::numeric_limits<double>::quiet_NaN();
  /// ν₁ + ν₂ when both constituent windings are certified; EMPTY
  /// otherwise (unknown, never zero).
  std::optional<int> totalWinding{};
  /// B₁ + B₂ when both baryon fluxes are known; EMPTY otherwise.
  std::optional<double> totalBaryonFlux{};
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// #765 composite persistence lifetime (the "lifetime" report).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the five meson certificates.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact GIVEN the consumed held certificates when certified;
  /// HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static MesonRead fromRecord(const Record &record);
};

/// # DiquarkRead
///
/// The #774 diquark-candidate read: TWO certified quarks (ν = +1 each),
/// EVEN composite parity (exact graded product), a certified Λ²C³
/// anti-triplet wedge occupation, and the PRESERVED constituent baryon
/// flux B = B₁ + B₂ = 2/3.  A diquark is explicitly NOT an antiquark: the
/// color representation alone (3̄) coincides, but the recorded distinction
/// channels are total occupation TWO, EVEN parity, and B = +2/3 (an
/// antiquark is occupation one, odd, B = −1/3) — exactly the #773
/// anti-triplet fixture's channels.
///
/// Certificate name vocabulary (fixed order): "constituent-quarks",
/// "parity-even", "anti-triplet", "baryon-flux-two-thirds".
struct DiquarkRead {
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// The two constituents' #765 identities, in evidence order.
  ComponentId firstConstituent{};
  ComponentId secondConstituent{};
  /// "diquark-candidate" or "none".
  std::string classification{"none"};
  /// The composite exterior parity (exact constituent product; 0 =
  /// unknown).
  int exteriorParity = 0;
  /// ⟨N⟩ of the carried composite state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// The certified anti-triplet wedge occupation det(C†ΓC) (NaN when the
  /// wedge read is missing/uncertified).
  double antiTripletWeight = std::numeric_limits<double>::quiet_NaN();
  /// ν₁ + ν₂ when both constituent windings are certified (2 for a
  /// certified diquark); EMPTY otherwise.
  std::optional<int> totalWinding{};
  /// B₁ + B₂ (2/3 for a certified diquark — the constituent flux is
  /// PRESERVED, never re-derived); EMPTY when unknown.
  std::optional<double> totalBaryonFlux{};
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// #765 composite persistence lifetime (the "lifetime" report).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the four diquark certificates.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact GIVEN the consumed held certificates when certified;
  /// HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static DiquarkRead fromRecord(const Record &record);
};

/// # BoundCandidateEvidence
///
/// One constituent's datum for the #775 bound-supercomponent search
/// (design spec §16.2).  Every field is an upstream read consumed
/// verbatim: the #773 quark verdict, the #765 level-0 support and
/// persistence window, and the #770 transports to the other constituents.
struct BoundCandidateEvidence {
  /// The candidate's #773 read (only a CERTIFIED "quark" verdict counts
  /// toward the three-quark census).
  QuarkRead quark{};
  /// The candidate's level-0 cell support (`ComponentRead::support`) —
  /// the set tested for containment in the supercomponent.  Empty =
  /// missing evidence: the containment certificate fails by name.
  std::vector<std::uint64_t> support{};
  /// The candidate's #765 persistence window
  /// {`PersistenceTrack::firstSlice`, `PersistenceTrack::lastSlice`}
  /// (inclusive slice indices).  EMPTY = no lifetime evidence — the
  /// overlap certificate fails by name, never presumed.
  std::optional<std::pair<std::size_t, std::size_t>> lifetime{};
  /// The #770 transports between this constituent and the others.  Every
  /// supplied link must be ACCEPTED with leakage under
  /// `maxTransportLeakage` — the "mutual transport remains inside the
  /// supercomponent" requirement, stated in the #770 isometry-defect
  /// vocabulary (a leaking transfer is the tracked subspace turning away
  /// from its successor).  Empty = missing evidence.
  std::vector<FiberTransportRead> mutualTransports{};
};

/// # BoundSupercomponentRead
///
/// One next-modular-level component examined by the #775 bound-
/// supercomponent search: which certified quark candidates it contains,
/// their shared lifetime window, and the containment / transport
/// certificates.  Unknown values are NaN / empty, never zero-filled.
///
/// Certificate name vocabulary (fixed order): "supercomponent-level",
/// "quark-count", "support-containment", "lifetime-overlap",
/// "transport-containment".
struct BoundSupercomponentRead {
  /// The examined next-level component's #765 identity.
  ComponentId boundComponent{};
  /// The CONTAINED certified quark candidates' identities, in input
  /// order (three when `found`).
  std::vector<ComponentId> quarks{};
  /// Indices of the contained candidates in the input candidate list.
  std::vector<std::size_t> quarkIndices{};
  /// Whether this component is a certified bound supercomponent of
  /// exactly three lifetime-overlapping certified quark candidates.
  bool found = false;
  /// The shared lifetime window [first, last] of the contained
  /// candidates; EMPTY when any lifetime is missing or the windows are
  /// disjoint.
  std::optional<std::pair<std::size_t, std::size_t>> lifetimeWindow{};
  /// Number of SHARED persistence slices (0 when disjoint/unknown).
  double lifetimeOverlap = 0.0;
  /// Smallest per-constituent support-containment fraction (NaN when a
  /// constituent supplied no support).
  double minContainment = std::numeric_limits<double>::quiet_NaN();
  /// Worst mutual-transport leakage over the contained candidates (NaN
  /// when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Number of mutual transports consumed.
  std::size_t transportCount = 0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (the containment/overlap decisions are exact set and
  /// integer-interval statements GIVEN the consumed certified constituent
  /// reads; residual = their maximum) when `found`; HeuristicDiscovery
  /// otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
};

/// # ScaleProfileSample
///
/// One refinement-window sample of the EXISTING #575/#566/#593 mass-radius
/// battery (`InteriorHinges`).  See the file banner section "the
/// form-factor situation": `radialWeightProfile` is a radial CURVATURE-
/// WEIGHT density, not a momentum-transfer form factor.
struct ScaleProfileSample {
  /// The emergent radius r = V_dual^{1/4} (`InteriorHinges::Radii::rDual`)
  /// — DIMENSIONFUL lattice units; only its finiteness is certified.
  double radius = std::numeric_limits<double>::quiet_NaN();
  /// The primal cross-check radius V_primal^{1/4}
  /// (`InteriorHinges::Radii::rPrimal`) — DIMENSIONFUL; its RATIO to
  /// `radius` is the dimensionless channel.
  double radiusCrossCheck = std::numeric_limits<double>::quiet_NaN();
  /// The spectral-mass channel: the intensive shell mass
  /// (`InteriorHinges::Masses::mShell`), a mean interior deficit ANGLE
  /// and therefore dimensionless in lattice units.
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  /// The curvature-weight participation ratio
  /// (`InteriorHinges::Localization::pr` ∈ (0, 1]) — dimensionless.
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// The per-BFS-shell share of the |Re ε · ★h| curvature weight
  /// (`InteriorHinges::Localization::shellProfile[k].weightShare`), shell
  /// ascending — dimensionless.  Empty when the interior carried no
  /// shell seeds (no register holes): then the profile is UNKNOWN and the
  /// stability certificate fails by name.
  std::vector<double> radialWeightProfile{};
};

/// # ScaleProfileRead
///
/// The #775 refinement-window certificate over `ScaleProfileSample`s: a
/// FINITE emergent radius plus the refinement stability of every
/// DIMENSIONLESS channel.  A dimensionful mass is never emitted
/// (`physicalMass` is always empty — no physical scale has been
/// independently established).
///
/// Certificate name vocabulary (fixed order): "refinement-window",
/// "finite-radius", "radius-ratio-stability", "spectral-mass-stability",
/// "localization-stability", "profile-stability".
struct ScaleProfileRead {
  /// Number of refinement samples consumed.
  std::size_t sampleCount = 0;
  /// The reported emergent radius (the FIRST sample's `radius`).
  double radius = std::numeric_limits<double>::quiet_NaN();
  /// Whether every sample's radius is finite and above `minRadius`.
  bool radiusFinite = false;
  /// The dimensionless radius ratio r_dual / r_primal of the first
  /// sample, and its normalized spread (max − min)/max(|mean|, 1) over the
  /// window — RELATIVE for O(1)-and-larger channels, ABSOLUTE for channels
  /// near zero (so a near-zero channel is never reported as infinitely
  /// unstable).  Every spread below uses the same normalization.
  double radiusRatio = std::numeric_limits<double>::quiet_NaN();
  double radiusRatioSpread = std::numeric_limits<double>::quiet_NaN();
  /// The dimensionless spectral-mass channel of the first sample and its
  /// relative spread over the window.
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  double spectralMassSpread = std::numeric_limits<double>::quiet_NaN();
  /// The localization channel of the first sample and its relative
  /// spread over the window.
  double localization = std::numeric_limits<double>::quiet_NaN();
  double localizationSpread = std::numeric_limits<double>::quiet_NaN();
  /// Max absolute per-shell deviation of the radial weight profiles
  /// across the window (NaN when a profile was missing or the shell
  /// counts disagree — unknown, never zero).
  double profileMaxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// Number of shells the profiles share (0 when unavailable).
  std::size_t profileShells = 0;
  /// The DIMENSIONFUL mass — ALWAYS empty: unknown until a physical scale
  /// is independently established (never zero, never a lattice number
  /// relabeled as a mass).
  std::optional<double> physicalMass{};
  /// Whether every listed certificate held.
  bool stable = false;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// CertifiedNumerical (the spreads are measured deviations of finite
  /// sums; residual = the worst measured deviation against
  /// `maxProfileDeviation`) when `stable`; HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
};

/// # BaryonCandidateEvidence
///
/// The assembled evidence bundle of ONE three-cluster candidate (#775).
/// Every field is a read produced by a merged upstream kernel — the #773
/// constituent verdicts, this class's own bound-supercomponent search and
/// octet bilinear read, the #767 color columns, the #772 rotation
/// character and spin lift, the #780 Wick spin reads, and the #575
/// mass-radius samples.  Missing evidence fails its certificate BY NAME;
/// it is never presumed to pass.
struct BaryonCandidateEvidence {
  /// The bound supercomponent's #765 identity (reported verbatim).
  ComponentId boundComponent{};
  /// The three constituents' #773 reads, CONSUMED VERBATIM.
  std::array<QuarkRead, 3> quarks{};
  /// The bound-supercomponent search result (`boundSupercomponentSearch`).
  BoundSupercomponentRead binding{};
  /// The three NORMALIZED anchored color columns C = [c_A c_B c_C], in
  /// constituent order.  The three-mode wedge is built ONCE from this
  /// matrix — no extra fermion sign is multiplied onto the color ε.
  Eigen::Matrix3cd colorColumns = Eigen::Matrix3cd::Zero();
  /// The bound object's octet bilinear read (`octetBilinearRead` on the
  /// composite's carried state and its three declared color modes) — the
  /// INDEPENDENT net-color-flux diagnostic.  Default-constructed =
  /// missing evidence.
  OctetBilinearRead colorFlux{};
  /// The #772 Berry-cancelled PHYSICAL-ROTATION character of the closed
  /// total-space 2π cluster-frame cycle against its matched co-moving
  /// non-rotating reference (`ExchangeHolonomy::rotationCharacter`).
  HolonomyCharacterRead rotation{};
  /// The #772 Berry-cancelled PARTICLE-EXCHANGE character
  /// (`ExchangeHolonomy::exchangeCharacter`), when the caller ran the
  /// exchange experiment.  REPORT-ONLY: neither the ticket's proton-
  /// certificate list nor design spec §16.4 has an exchange row, so it
  /// gates nothing — it only fills `BaryonRead::exchangeCharacter` and the
  /// doubly cancelled spin-statistics ratio.  A read tagged with the wrong
  /// channel is refused (the #772 channels are never interchangeable).
  std::optional<HolonomyCharacterRead> exchange{};
  /// Whether the caller is making a CONTINUUM spin claim.  When true the
  /// SO(d) → Spin(d) lift is REQUIRED (a missing/obstructed lift fails by
  /// name); when false the lift is not applicable and is never demanded
  /// (design spec §16.4).
  bool continuumSpinClaim = false;
  /// The #772 `ExchangeHolonomy::spinLift` decision, when one was made.
  std::optional<SpinLiftRead> spinLift{};
  /// The #780 total-space ⟨J²⟩ of the carried quasi-free state
  /// (`CovarianceState::wickSpinSquaredExpectation`).
  quantum::WickCertificateRead spinSquaredRead{};
  /// The #780 Var(J²) of the carried quasi-free state
  /// (`CovarianceState::wickSpinSquaredVariance`) — the sharp-spin
  /// certificate.
  quantum::WickCertificateRead spinVarianceRead{};
  /// The ACCEPTED COVARIANCE-ONLY CLASS: the Var(J²) read of every
  /// quasi-free candidate of the class the obstruction verdict quantifies
  /// over (the candidate's own read included by the caller).  Empty or
  /// partially uncertified = the class was NOT swept, so a variance
  /// failure is an unknown, never an obstruction.
  std::vector<quantum::WickCertificateRead> classVarianceReads{};
  /// The #772 DENSE total-space ⟨J²⟩ oracle
  /// (`ExchangeHolonomy::totalJSquared`) for a candidate carried as an
  /// explicit composite state rather than a covariance.  Consulted only
  /// when the #780 Wick expectation is absent/uncertified, and it never
  /// supplies a variance: expectation alone is not a sharp-spin
  /// certificate.
  std::optional<double> totalSpaceJ2{};
  /// The refinement-window samples of the existing mass-radius battery
  /// (`scaleProfileSample`).
  std::vector<ScaleProfileSample> scaleSamples{};
  /// #765 persistence-track lifetime of the BOUND component (NaN =
  /// missing; report-only — the persistence requirement lives in the
  /// supercomponent search's lifetime-overlap certificate).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// The composite's lifetime transports (#770; report-only — the max
  /// leakage travels on the read).
  std::vector<FiberTransportRead> lifetimeTransports{};
};

/// # BaryonRead
///
/// The #775 three-quark baryon read and complete proton certificate
/// (design spec §6.8 — the spec field names are preserved verbatim —
/// and §16.2-§16.4), extended with the evidence summary the verdict
/// consumed, the recorded thresholds, and the #764 certificate.
///
/// `classification` is one of "no-baryon", "baryon-candidate",
/// "certified-proton", or "quasi-free-sharp-spin-obstruction" (the
/// hyphenated spelling of the spec's `quasi_free_sharp_spin_obstruction`,
/// matching this class's existing "gluon-candidate"/"meson-candidate"
/// verdict vocabulary).
///
/// Certificate name vocabulary — the two STRUCTURAL gates first
/// (a failure of either is "no-baryon"): "constituent-quarks",
/// "bound-supercomponent"; then the proton-certificate gates:
/// "color-singlet", "color-flux-zero", "baryon-flux-unit",
/// "composite-parity-odd", "flavor-uud", "electric-flux-unit",
/// "spin-expectation", "sharp-spin", "rotation-character", "spin-lift",
/// "finite-radius", "profile-stability".
///
/// Unknown or uncertified values are NULL (empty optional; NaN for
/// unmeasured doubles; 0 for the sign-valued ints), never zero-filled, and
/// every gap is NAMED in `failedCertificates`.
struct BaryonRead {
  /// The three constituents' #765 identities, in evidence order.
  std::array<ComponentId, 3> quarks{};
  /// The bound supercomponent's #765 identity.
  ComponentId boundComponent{};
  /// det(C†C) = |det C|² of the three normalized color columns (NaN when
  /// no color evidence was supplied).
  double colorGramDeterminant = std::numeric_limits<double>::quiet_NaN();
  /// The NET COLOR FLUX diagnostic: the octet Frobenius weight of the
  /// bound object's color bilinear (NaN when the octet read is missing or
  /// uncertified).  An independent finite-complex diagnostic — never on
  /// its own a proof of confinement.
  double colorFlux = std::numeric_limits<double>::quiet_NaN();
  /// B = ν/3 with ν the SUM of the three certified constituent windings
  /// (+1 for a certified proton); EMPTY when any leg is uncertified.
  std::optional<double> baryonFlux{};
  /// The summed CERTIFIED constituent electric Gauss fluxes (+1 for a
  /// certified proton); EMPTY when any constituent's charge is unknown.
  std::optional<double> electricFlux{};
  /// The certified total-space ⟨J²⟩ (3/4 for a proton, 15/4 for a Δ);
  /// EMPTY when neither the #780 Wick read nor the #772 dense oracle
  /// certified it.
  std::optional<double> totalJ2{};
  /// The certified Var(J²) (≈ 0 for a sharp spin); EMPTY when the
  /// quasi-free variance read is missing/uncertified — UNKNOWN, never
  /// zero, and never inferred from the expectation.
  std::optional<double> totalJ2Variance{};
  /// The #772 Berry-cancelled 2π character; EMPTY when the rotation read
  /// did not certify.
  std::optional<std::complex<double>> rotationCharacter{};
  /// "no-baryon", "baryon-candidate", "certified-proton", or
  /// "quasi-free-sharp-spin-obstruction".
  std::string classification{"no-baryon"};
  /// The bound component's #765 persistence lifetime (NaN = missing).
  double persistence = std::numeric_limits<double>::quiet_NaN();
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};

  // ── additive evidence summary (consumed by #776/#777/#778 unchanged) ──

  /// The invariant color volume S_ABC = det[c_A c_B c_C] — built ONCE
  /// (NaN when no color evidence).  A transposition of two constituents
  /// flips this sign and leaves `colorGramDeterminant` invariant.
  std::complex<double> colorWedge{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
  /// ν = ν_A + ν_B + ν_C over the certified constituent windings (3 for a
  /// certified proton); EMPTY when any leg is uncertified.
  std::optional<int> totalWinding{};
  /// The composite exterior parity: the exact graded product of the three
  /// certified constituent parities (−1 odd / +1 even / 0 = unknown).
  int exteriorParity = 0;
  /// The certified flavor occupation pattern under the constituents'
  /// recorded doublet orientations ("uud", "uuu", "udd", "ddd"); EMPTY
  /// when any constituent's isospin is unknown.
  std::string flavorPattern{};
  /// The summed certified constituent I3 (+1/2 for uud); EMPTY when any
  /// constituent's isospin is unknown.
  std::optional<double> totalIsospin{};
  /// −1 / +1 when the #772 rotation read emitted a sign; 0 otherwise.
  int rotationCharacterSign = 0;
  /// The #772 Berry-cancelled PARTICLE-EXCHANGE character, when the caller
  /// supplied a certified exchange read; EMPTY otherwise.  REPORT-ONLY.
  std::optional<std::complex<double>> exchangeCharacter{};
  /// The doubly cancelled spin-statistics ratio
  /// χ̂(exchange) · χ̂(2π)^{-1} (`ExchangeHolonomy::doublyCancelledRatio`;
  /// +1 on a spin-½ fixture, each factor separately near −1) — filled only
  /// when BOTH channels are certified and correctly tagged.  REPORT-ONLY.
  std::optional<std::complex<double>> spinStatisticsRatio{};
  /// Whether a CONTINUUM spin claim was declared (so the lift was
  /// demanded) and, when demanded, whether it was accepted.
  bool spinLiftApplicable = false;
  bool spinLiftAccepted = false;
  /// Whether the sharp-spin certificate (certified Var(J²) within
  /// `spinVarianceTolerance`) held.
  bool sharpSpin = false;
  /// Whether the accepted covariance-only class was swept (every class
  /// variance read supplied AND certified) — the premise the obstruction
  /// verdict quantifies over.
  bool quasiFreeClassSwept = false;
  /// min over the swept class of |Var(J²)| (NaN when not swept): the
  /// floor the variance failed to converge below.
  double classVarianceFloor = std::numeric_limits<double>::quiet_NaN();
  /// The consumed scale-profile scalars (the QuarkRead flat-summary
  /// convention — the full `ScaleProfileRead` is recomputed by
  /// `scaleProfile` from the same evidence): the emergent radius, its
  /// finiteness, the dimensionless spectral-mass channel, the
  /// dimensionless radius ratio, the worst dimensionless deviation across
  /// the refinement window, and whether every dimensionless channel was
  /// stable.
  double radius = std::numeric_limits<double>::quiet_NaN();
  bool radiusFinite = false;
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  double radiusRatio = std::numeric_limits<double>::quiet_NaN();
  double profileMaxDeviation = std::numeric_limits<double>::quiet_NaN();
  bool profileStable = false;
  /// The DIMENSIONFUL mass — ALWAYS empty (see the file banner).
  std::optional<double> physicalMass{};
  /// Number of shared persistence slices of the three constituents.
  double lifetimeOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the fourteen certificates listed above; 1.0
  /// exactly for a certified proton.
  double confidence = 0.0;
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (exact boolean combination GIVEN the consumed held
  /// certificates; residual = their maximum) for a certified proton;
  /// HeuristicDiscovery (never holds) otherwise — including for the
  /// obstruction verdict, which is a reported branch point, not a held
  /// claim.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization (design spec §20 `particles.baryons`): every
  /// spec field, the evidence summary, the failed-certificate names, and
  /// the threshold echo travel together; unknown values serialize as
  /// null, never zero.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate from `toRecord()` output; rejects an unknown
  /// `schema_version` (std::invalid_argument).
  [[nodiscard]] static BaryonRead fromRecord(const Record &record);
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

    // ── #774 even sectors: octet bilinear read + gluon/meson/diquark ────

    /// `AnalyticCache` kind string of a cached octet bilinear read.
    static constexpr const char *kOctetCacheKind = "octet-bilinear-read";

    /// The quasi-free traceless-bilinear (octet) read of three declared
    /// color modes of a carried #780 covariance (see `OctetBilinearRead`
    /// for the exact identities).  Every quantity is a finite exact Wick
    /// sum evaluated through the PUBLIC `CovarianceState` reads — dense
    /// Fock objects never appear on this path; adding vacuum-embedded
    /// microscopic modes leaves the read unchanged.
    /// @throws std::invalid_argument unless `colorModes` names exactly
    ///   three DISTINCT in-range modes.
    [[nodiscard]] OctetBilinearRead octetBilinearRead(
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// `octetBilinearRead` through the #764 `AnalyticCache` contract:
    /// keyed by the caller's component vertex set (kind `kOctetCacheKind`,
    /// parameter = `octetFingerprint` — the covariance hash, the declared
    /// modes, and the sign-extraction threshold), served while the
    /// component's star is untouched AND the fingerprint matches (a Γ
    /// change is a state change), recomputed and re-stored otherwise.
    /// Cached results equal cold recomputation.
    [[nodiscard]] OctetBilinearRead octetBilinearReadCached(
        cobordism::AnalyticCache &cache,
        const std::vector<std::uint64_t> &componentVertexIds,
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// Order-sensitive content fingerprint of an octet-read request: the
    /// exact covariance hash, the declared color modes, and the decision
    /// thresholds (the cache parameter of `octetBilinearReadCached`).
    [[nodiscard]] std::uint64_t octetFingerprint(
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// Classify one gluon candidate from its assembled evidence (design
    /// spec §14.3): certified even parity, a nonzero certified octet
    /// excitation with machine-level octet purity, accepted rank-three
    /// lifetime transports under the leakage cap, a CERTIFIED zero total
    /// determinant winding (zero baryon flux — evidence, never a default),
    /// and sufficient persistence.  Never throws on missing evidence —
    /// missing evidence is a NAMED failed certificate.
    [[nodiscard]] GluonRead classifyGluon(
        const GluonCandidateEvidence &evidence) const;

    /// Classify one meson candidate: a certified quark + certified
    /// antiquark pair (order-insensitive), even composite parity (exact
    /// constituent product), color-singlet pairing, and zero total
    /// certified winding/flux (see `MesonRead`).
    [[nodiscard]] MesonRead classifyMeson(
        const CompositeCandidateEvidence &evidence) const;

    /// Classify one diquark candidate: two certified quarks, even
    /// composite parity, a certified anti-triplet wedge occupation, and
    /// the preserved constituent baryon flux B = 2/3 (see `DiquarkRead`).
    [[nodiscard]] DiquarkRead classifyDiquark(
        const CompositeCandidateEvidence &evidence) const;

    // ── #775 three-cluster sector: baryons and the proton certificate ───

    /// The bound-supercomponent search (design spec §16.2): for each
    /// NEXT-modular-level component, report which certified quark
    /// candidates it contains, their shared #765 lifetime window, and the
    /// containment / mutual-transport certificates.  One read is emitted
    /// per component containing at least one contained candidate, in
    /// input order; `found` requires the component to sit at a strictly
    /// higher modular level than every constituent, to contain EXACTLY
    /// three certified quark candidates, and for their lifetimes to
    /// overlap and their mutual transports to stay bounded.  Never throws
    /// on missing evidence — missing evidence is a NAMED failed
    /// certificate.
    [[nodiscard]] std::vector<BoundSupercomponentRead>
    boundSupercomponentSearch(
        const std::vector<ComponentRead> &nextLevelComponents,
        const std::vector<BoundCandidateEvidence> &candidates) const;

    /// One refinement sample of the EXISTING #575/#566/#593 mass-radius
    /// battery, read through the #593 validated context exactly as
    /// `EmergentRadius` / `EmergentMass` do
    /// (`RegisterContext::interiorHinges`): the dual/primal radii, the
    /// intensive shell mass, the curvature-weight participation ratio,
    /// and the per-shell curvature-weight profile.  Read-only; nothing is
    /// recomputed and no solver is called.
    [[nodiscard]] static ScaleProfileSample scaleProfileSample(
        const RegisterContext &ctx);

    /// The refinement-window certificate over `ScaleProfileSample`s: a
    /// finite emergent radius plus the refinement stability of every
    /// DIMENSIONLESS channel (see `ScaleProfileRead`, and the file banner
    /// section "the form-factor situation" — nothing here is a form
    /// factor and no dimensionful mass is ever emitted).
    [[nodiscard]] ScaleProfileRead scaleProfile(
        const std::vector<ScaleProfileSample> &samples) const;

    /// Classify one three-cluster candidate and evaluate the complete
    /// proton certificate (design spec §16.2-§16.4): the two structural
    /// gates, the color volume / Gram determinant with the wedge built
    /// ONCE, the independent net-color-flux diagnostic, the summed
    /// certified winding and graded parity, the reused #773 flavor/charge
    /// reads, the #772 2π character and (when a continuum spin claim is
    /// declared) the spin lift, the SHARP total-space spin certificate,
    /// and the refinement-window scale reads.  Returns "no-baryon",
    /// "baryon-candidate", "certified-proton", or
    /// "quasi-free-sharp-spin-obstruction" with every failed or unknown
    /// certificate NAMED.  Never throws on missing evidence.
    [[nodiscard]] BaryonRead classifyBaryon(
        const BaryonCandidateEvidence &evidence) const;

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
