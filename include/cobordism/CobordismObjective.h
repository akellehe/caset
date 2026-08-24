// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COBORDISMOBJECTIVE_H
#define TESSERA_COBORDISM_COBORDISMOBJECTIVE_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/HodgeLaplacian.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # ObjectiveTerms
///
/// The COMPLETE, enumerable term list a scalar cobordism objective is the sum
/// of. `CobordismObjective::total` is static over this record, so the collapse
/// from a decomposition to the number the optimizer compares provably reads
/// nothing else. Every member is a geometric or target quantity; the last is
/// the one permitted state channel.
///
/// Declared at namespace scope so an objective can be written without
/// depending on `MultiCobordism`; `MultiCobordism::ObjectiveTerms` aliases this
/// type, so every existing use of that name is unchanged.
struct ObjectiveTerms {
  /// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2\f$ — 0 when the Einstein-Hilbert
  /// term is deselected.
  double reggeStationarity = 0.0;
  /// \f$\eta_H\sum_k\|\nabla_zS_{{\rm Hodge},k}\|^2\f$ — joint stationarity.
  double hodgeStationarity = 0.0;
  /// \f$\gamma r_U\f$ — the target-conditioned register residual.
  double registerResidual = 0.0;
  /// \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$'s action magnitude.
  double actionMagnitude = 0.0;
  /// \f$\beta_E E_{\rm carried}(\Gamma,g)\f$ — the ONE permitted state
  /// channel, exactly 0.0 outside the certificates-blind mean-field sub-mode.
  double carriedStateEnergy = 0.0;
};

/// # ObjectiveContext
///
/// **The no-feedback firewall, restated for an injected objective.**
///
/// The guarantee this replaces was mechanical rather than conventional:
/// `objectiveOf` was `static`, so it had no `this` and therefore no pointer
/// through which to dereference a member where an analysis result could live.
/// Injection reintroduces a `this` — an objective is an object and could in
/// principle hold a node reference or cache reads across calls — so the
/// guarantee has to be re-established on the INPUT side instead.
///
/// It is re-established by this type being PLAIN DATA. It carries geometry, a
/// region, that region's declared target states, and scalar configuration.
/// There is no `MultiCobordism` reference, no pointer to one, and — deliberately
/// — no `std::function`, because a bound callable would capture the node and
/// smuggle exactly the reachability the static function denied. An objective
/// therefore cannot consult a component, fiber, transport, amplitude, colour,
/// charge, flavour, exchange, spin certificate or verdict: not because it is
/// written not to, but because it is handed nothing that leads there.
/// `inputNames()` enumerates every field so a test asserts the list rather than
/// trusting this comment.
///
/// Geometry access is intended and necessary — an objective must read the
/// complex it scores. What is impossible is reaching the analysis products OF
/// that geometry, all of which land in the checkpoint document and nowhere a
/// context can see.
struct ObjectiveContext {
  /// The complex being scored.
  std::shared_ptr<Spacetime> spacetime;

  /// The region of `spacetime` this objective is scored over, as a vertex set.
  /// EMPTY means the whole complex. An objective is always evaluated against a
  /// region rather than implicitly against "the node", so several objectives —
  /// one per pinned region, say — can coexist on one complex without any of
  /// them assuming it is the only one.
  std::set<std::uint64_t> region;

  /// The edge coordinates this objective's sums run over, as indices into the
  /// complex's edge list.
  ///
  /// ABSENT means every edge — the whole-cobordism scope, and the
  /// single-objective run that stays bit-identical. A PRESENT but empty list
  /// means score nothing, which is a real and different thing: a region whose
  /// interior contains no edge, with the straddling edges declared out, scores
  /// no coordinate at all. Collapsing the two would silently promote such a
  /// region to scoring the entire complex.
  ///
  /// The ENGINE resolves this from the objective's declared `ObjectiveScope` —
  /// which region, and whether the straddling edges count — so the declaration
  /// is honoured rather than re-derived, and an objective never recomputes edge
  /// membership from `region`. That is why the scope is a declaration the
  /// engine reads and not a rule the engine applies by role.
  std::optional<std::vector<std::size_t>> scoredEdges;

  /// The target states the region is scored against, for a target-conditioned
  /// objective. Empty for a purely geometric one.
  std::vector<std::vector<std::complex<double>>> regionTargets;

  /// The register degrees the objective is declared over.
  std::vector<int> registerDegrees;
  /// \f$\beta_R\f$, the Regge stationarity weight.
  double reggeWeight = 1.0;
  /// \f$\eta_H\f$, the Hodge-entropy stationarity weight.
  double hodgeEntropyWeight = 0.0;
  /// \f$\gamma\f$, the register-residual weight.
  double gamma = 1.0;
  /// \f$\beta_E\f$, the carried-state energy weight. Exactly zero outside the
  /// certificates-blind mean-field sub-mode.
  double carriedStateEnergyWeight = 0.0;
  /// Whether the Einstein-Hilbert term is selected.
  bool einsteinHilbert = true;
  /// Which entropy the Hodge term reads: the complex operator or its
  /// phase-blind entrywise ablation.
  HodgeLaplacian::EntropyPhaseMode hodgeEntropyPhaseMode =
      HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase;

  /// \f$r_U\f$ on this region, computed by the engine and passed as a NUMBER
  /// rather than as a callable, so no node is reachable from here. Computed
  /// only when the objective declares `needsRegisterResidual`; NaN otherwise,
  /// never a silent zero.
  double registerResidual = std::numeric_limits<double>::quiet_NaN();
  /// \f$E_{\rm carried}(\Gamma,g)\f$, likewise a precomputed number. Exactly
  /// zero where the weight is zero.
  double carriedStateEnergy = 0.0;

  /// The names of every field above, in declaration order — the firewall list
  /// a structural test asserts against, exactly as `objectiveTermNames` does
  /// for the output side.
  [[nodiscard]] static std::vector<std::string> inputNames();
};

/// # ObjectiveDirection
///
/// A stage-2 search direction together with the exact objective value at the
/// point it was taken from. `baselineComputed` is false when the objective did
/// not assemble its scalar while building the direction, in which case the
/// engine evaluates the scalar itself rather than trusting an accumulated
/// stage-1 trace.
struct ObjectiveDirection {
  /// The ascent displacement. Stage 2 subtracts a scaled multiple of it.
  Eigen::VectorXcd ascent;
  /// The exact objective at the current point, when the direction's assembly
  /// already produced it.
  double baseline = 0.0;
  /// Whether `baseline` is meaningful.
  bool baselineComputed = false;
};

/// # ObjectiveDirectionContext
///
/// `ObjectiveContext` plus the extra data a stage-2 direction needs. Plain
/// data for the same reason, so the direction path cannot reach a node either.
struct ObjectiveDirectionContext {
  /// The scalar inputs, unchanged.
  ObjectiveContext scalar;
  /// The number of edge coordinates the direction is taken over.
  std::size_t edgeCount = 0;
  /// \f$\partial E_{\rm carried}/\partial z\f$, exact and analytic, computed by
  /// the engine. Empty where the carried-state weight is zero.
  std::vector<std::complex<double>> carriedStateEnergyGradient;
};

/// # RegionHandle
///
/// A reference to a DECLARED pinned region. The whole point of the type is
/// that a caller cannot fabricate one: the only non-empty handle comes from
/// `MultiCobordism::regionHandle`, which looks the name up among the declared
/// regions and throws by name if it is not there.
///
/// That is what makes a mis-spelling impossible rather than merely
/// discouraged. A bare `std::string region` would let `"boundary"` and
/// `"boundry"` both compile and one of them silently score nothing; a handle
/// cannot be spelled at all, only obtained.
class RegionHandle {
 public:
  /// The whole cobordism — the default, and what declaring nothing produces.
  RegionHandle() = default;

  /// Whether this handle references the whole cobordism rather than a region.
  [[nodiscard]] bool isWholeCobordism() const noexcept { return name_.empty(); }
  /// The declared region's name, for reporting. Empty for the whole cobordism.
  [[nodiscard]] const std::string &name() const noexcept { return name_; }

  [[nodiscard]] bool operator==(const RegionHandle &other) const noexcept {
    return name_ == other.name_;
  }

 private:
  /// Only the engine mints a handle, and only for a region it has verified is
  /// declared.
  friend class MultiCobordism;
  explicit RegionHandle(std::string name) : name_(std::move(name)) {}
  std::string name_;
};

/// # ObjectiveScope
///
/// What an objective DECLARES that it references: a pinned region, or — by
/// declaring nothing — the whole cobordism. The engine honours the
/// declaration; it does not infer a scope from an objective's role or decide
/// one by convention.
///
/// The default is the whole cobordism, which is the single-objective run that
/// exists today and must stay bit-identical to it.
///
/// Scope is deliberately independent of whether the referenced region's
/// coordinates are frozen. Pinning does two unrelated jobs — it NAMES a region
/// so an objective can reference it, and it CONSTRAINS relaxation by zeroing a
/// pinned edge's descent component before the line search — and neither
/// justifies the other. A pinned edge does not vary, yet it is still scored,
/// and the bulk objective scores the pinned interior along with everything
/// else. An objective scoped to a region must not have to know or care whether
/// that region's coordinates move.
struct ObjectiveScope {
  /// The region referenced. Default-constructed means the whole cobordism.
  /// Obtainable only from `MultiCobordism::regionHandle`, so it cannot name a
  /// region that was never declared.
  RegionHandle region;

  /// Whether edges with a single endpoint in `region` — the straddling edges —
  /// enter this objective's score. Part of the same scope declaration rather
  /// than a separate mechanism, and meaningless for a whole-cobordism scope,
  /// which has no border to straddle.
  ///
  /// A region-scoped objective will normally declare `false`, so the edges
  /// tying its region to the bulk are scored by the bulk's objective and not
  /// twice; a caller that wants them counted may say otherwise. The border is
  /// the one the node already defines — `MultiCobordism::edgeIsPinned` holds
  /// exactly when a SINGLE region contains both endpoints — so a straddling
  /// edge is one with a single endpoint in the region. That predicate is the
  /// definition; nothing here restates it.
  bool includesStraddlingEdges = true;

  /// Whether this scope is the whole cobordism, i.e. nothing was declared.
  [[nodiscard]] bool isWholeCobordism() const {
    return region.isWholeCobordism();
  }
};

/// # ObjectiveName
///
/// The identifiers objectives are known by, as named constants rather than
/// string literals repeated at each site. Every identifier is written once
/// where the objective declares it and compared against these where a caller
/// selects or asserts one; a typo in a literal would not fail to compile, it
/// would silently fail to match.
class ObjectiveName {
 public:
  static constexpr const char *kJointStationarity = "joint_stationarity";
  static constexpr const char *kLegacy = "legacy";
  static constexpr const char *kMediatedCorrespondence =
      "mediated_correspondence";
};

/// # ObjectiveTermName
///
/// The declared term slots, likewise named. `CobordismObjective::
/// declaredTermNames` is assembled from these, so the list and the constants
/// cannot drift apart.
class ObjectiveTermName {
 public:
  static constexpr const char *kReggeStationarity = "regge_stationarity";
  static constexpr const char *kHodgeStationarity = "hodge_stationarity";
  static constexpr const char *kRegisterResidual = "register_residual";
  static constexpr const char *kActionMagnitude = "action_magnitude";
  static constexpr const char *kCarriedStateEnergy = "carried_state_energy";
};

/// # CobordismObjective
///
/// The functional `MultiCobordism` descends, as an injected specification
/// rather than a value of a closed enum. An implementation declares the terms
/// it is the sum of, decomposes itself over a REGION of a complex, and supplies
/// a stage-2 search direction. It knows nothing about the engine that drives
/// it, and the engine knows nothing about which objective it holds.
///
/// An objective is scored against a region, never implicitly against a whole
/// node, so more than one may coexist on a single complex — a pinned region
/// carrying its own objective alongside the node's, for instance.
class CobordismObjective {
 public:
  virtual ~CobordismObjective() = default;

  /// A stable identifier, stamped on records so a run says what it descended.
  [[nodiscard]] virtual std::string name() const = 0;

  /// The complete, enumerable term list this objective is the sum of. The
  /// names are the record's keys.
  [[nodiscard]] virtual std::vector<std::string> termNames() const = 0;

  /// Decompose this objective over the context's region.
  [[nodiscard]] virtual ObjectiveTerms terms(
      const ObjectiveContext &context) const = 0;

  /// The stage-2 search direction over the context's region.
  [[nodiscard]] virtual ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const = 0;

  /// Whether this objective's value depends on prescribed target states rather
  /// than on the geometry alone. A search policy that must stay unforced
  /// consults this rather than testing for an objective by name.
  [[nodiscard]] virtual bool isTargetConditioned() const = 0;

  /// What this objective references: a named pinned region, or — by declaring
  /// nothing — the whole cobordism. The engine honours the declaration rather
  /// than inferring one from the objective's role.
  ///
  /// Scope is a property of the INSTANCE, not of the class, so an existing
  /// objective can be pointed at a region without writing a new type: the same
  /// functional is a perfectly good thing to hold a boundary to. An
  /// implementation may still override this where its scope is intrinsic.
  [[nodiscard]] virtual ObjectiveScope scope() const { return scope_; }

  /// Declare what this objective references. Default-constructed means the
  /// whole cobordism, which is what an objective that never calls this
  /// declares.
  void setScope(ObjectiveScope scope) { scope_ = std::move(scope); }

  /// Whether this objective reads \f$r_U\f$. The engine computes that residual
  /// only when an objective asks for it, so a purely geometric objective never
  /// pays for a target-conditioned quantity it does not use.
  [[nodiscard]] virtual bool needsRegisterResidual() const { return false; }

  /// Whether a candidate move's objective change may be scored by a LOCALIZED
  /// exact delta instead of by re-evaluating the whole functional.
  ///
  /// An objective built from global spectra or action magnitudes changes
  /// everywhere when one cell changes, so its true scalar difference is the
  /// only honest score and the engine pays for a full evaluation per candidate.
  /// An objective assembled from per-cell contributions can instead be
  /// differenced exactly over the cells a move touches. Declaring `false` is
  /// always CORRECT and merely more expensive, which is why it is the default:
  /// an objective opts in only where its decomposition genuinely supports the
  /// cheaper route.
  [[nodiscard]] virtual bool supportsLocalizedDelta() const { return false; }

  /// The lowest register degree over which this objective is DECLARED. The
  /// engine refuses to install it on a node carrying a lower degree, so a
  /// declared domain restriction travels with the objective that declares it
  /// rather than living in the engine as a special case.
  ///
  /// Zero — no restriction — is the default.
  [[nodiscard]] virtual int minimumRegisterDegree() const { return 0; }

  /// The weight this objective puts on a NUMERICALLY differentiated
  /// register-residual direction, given its configuration; zero for an
  /// objective that supplies an analytic direction for every term it has.
  ///
  /// Returned as a weight rather than performed here on purpose: differencing
  /// a scalar over edge coordinates is engine machinery, and handing an
  /// objective a callable that could do it would mean handing it a closure over
  /// the node. The engine applies this weight to its own differentiation of
  /// \f$r_U\f$.
  [[nodiscard]] virtual double numericalRegisterResidualWeight(
      const ObjectiveContext &) const {
    return 0.0;
  }

  /// The scalar: the plain sum of the declared terms. STATIC by design — the
  /// collapse from the decomposition to the number the optimizer compares has
  /// no `this` and so cannot reach any state at all.
  [[nodiscard]] static double total(const ObjectiveTerms &terms);

  /// The declaration order of `ObjectiveTerms`' members. Every objective
  /// records into the same enumerable slots, so a record stays comparable
  /// across objectives and a structural test can assert the list.
  [[nodiscard]] static std::vector<std::string> declaredTermNames();

 private:
  /// The declared scope. Default-constructed is the whole cobordism, so an
  /// objective that never declares one behaves exactly as it did before scopes
  /// existed.
  ObjectiveScope scope_;
};

/// # JointStationarityObjective
///
/// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2+\eta_H\sum_k\|\nabla_zS_{{\rm
/// Hodge},k}\|^2\f$ — both the Regge action and the Hodge entropy stationary
/// at the same metric. The objective the whitepaper describes, and the only
/// one of the three built-ins that is not target-conditioned.
class JointStationarityObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return false; }
  /// A DECLARED domain restriction, not a capability limit. Since the
  /// degree-zero \f$L_0=d_1W_1^{-1}d_1^{\mathsf T}\f$ is holomorphic in
  /// \f$z\f$ and its entropy gradient is exact, the gradient exists at degree
  /// zero too; widening this objective's declared domain to reach it is a
  /// separate decision about the objective, taken deliberately or not at all.
  [[nodiscard]] int minimumRegisterDegree() const override { return 1; }
};

/// # LegacyObjective
///
/// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2+\gamma r_U\f$ — the compatibility
/// objective. Target-conditioned through \f$r_U\f$.
class LegacyObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return true; }
  [[nodiscard]] bool needsRegisterResidual() const override { return true; }
  /// The Regge half has an exact per-cell delta and the register half is an
  /// exact \f$\gamma\,\Delta r_U\f$, so a candidate move is scored by
  /// differencing over the cells it touches rather than by re-evaluating the
  /// whole functional.
  [[nodiscard]] bool supportsLocalizedDelta() const override { return true; }
  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override;
};

/// # MediatedCorrespondenceObjective
///
/// \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$ — the historical operator-cobordism
/// experiment. Target-conditioned through \f$r_U\f$.
class MediatedCorrespondenceObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return true; }
  [[nodiscard]] bool needsRegisterResidual() const override { return true; }
  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISMOBJECTIVE_H
