// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COBORDISMOBJECTIVE_H
#define TESSERA_COBORDISM_COBORDISMOBJECTIVE_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
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

  /// Whether this objective reads \f$r_U\f$. The engine computes that residual
  /// only when an objective asks for it, so a purely geometric objective never
  /// pays for a target-conditioned quantity it does not use.
  [[nodiscard]] virtual bool needsRegisterResidual() const { return false; }

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
