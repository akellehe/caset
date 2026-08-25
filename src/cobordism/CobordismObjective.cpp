// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CobordismObjective.h"

#include <cmath>
#include <limits>

#include "cobordism/HodgeLaplacian.h"
#include "matter/MatterConfiguration.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using complexd = std::complex<double>;

// ---------------------------------------------------------------- context

std::vector<std::string> ObjectiveContext::inputNames() {
  // The declaration order of `ObjectiveContext`. Enumerated as data so the
  // no-feedback firewall stays CHECKABLE for an injected objective exactly as
  // it was for the static `objectiveOf`: a test reads this list and confirms
  // that nothing on it is, or leads to, a component, fiber, transport,
  // amplitude, colour, particle, charge, flavour, exchange, spin certificate
  // or verdict. Every entry is geometry, a region, a declared target, a
  // configured weight, or a precomputed geometric scalar.
  return {"spacetime",
          "region",
          "scored_edges",
          "region_targets",
          "register_degrees",
          "hodge_degrees",
          "hodge_degree_weights",
          "regge_weight",
          "hodge_entropy_weight",
          "connection_entropy_weight",
          "gamma",
          "carried_state_energy_weight",
          "einstein_hilbert",
          "hodge_entropy_phase_mode",
          "register_residual",
          "carried_state_energy"};
}

// ---------------------------------------------------------------- base

double CobordismObjective::total(const ObjectiveTerms &terms) {
  // STATIC: no `this`, so the scalar the optimizer descends provably depends
  // on nothing but the declared terms.
  return terms.reggeStationarity + terms.hodgeStationarity +
         terms.connectionStationarity + terms.registerResidual +
         terms.actionMagnitude + terms.carriedStateEnergy;
}

std::vector<std::string> CobordismObjective::declaredTermNames() {
  // Assembled from the named constants, so the list and the names a caller
  // compares against cannot drift apart.
  return {ObjectiveTermName::kReggeStationarity,
          ObjectiveTermName::kHodgeStationarity,
          ObjectiveTermName::kConnectionStationarity,
          ObjectiveTermName::kRegisterResidual,
          ObjectiveTermName::kActionMagnitude,
          ObjectiveTermName::kCarriedStateEnergy};
}

namespace {

/// The one permitted state channel, shared by every built-in objective. The
/// engine passes the energy as a number that is exactly zero outside the
/// certificates-blind mean-field sub-mode.
double carriedStateTerm(const ObjectiveContext &context) {
  if (context.carriedStateEnergyWeight == 0.0) return 0.0;
  return context.carriedStateEnergyWeight * context.carriedStateEnergy;
}

/// A per-edge mask from the resolved scope. An EMPTY mask means every edge —
/// the whole-cobordism scope — and every sum below then runs exactly the loop
/// it ran before scopes existed, which is what keeps the single-objective run
/// bit-identical. A present-but-empty scope yields an all-false mask, which
/// scores nothing; that is deliberately NOT the same as the whole cobordism.
std::vector<bool> scopeMask(const ObjectiveContext &context,
                            std::size_t edgeCount) {
  if (!context.scoredEdges.has_value()) return {};
  std::vector<bool> mask(edgeCount, false);
  for (std::size_t edgeIndex : *context.scoredEdges)
    if (edgeIndex < edgeCount) mask[edgeIndex] = true;
  return mask;
}

/// Whether an edge coordinate is in scope. An empty mask is the whole
/// cobordism, so everything is.
bool edgeInScope(const std::vector<bool> &mask, std::size_t edgeIndex) {
  return mask.empty() || mask[edgeIndex];
}

/// `β_R ‖∇_z S_Regge‖²` over the edges in scope, computed from the geometry
/// alone, or zero where the Einstein-Hilbert term is deselected or unweighted.
double reggeTerm(const ObjectiveContext &context) {
  if (!context.einsteinHilbert || context.reggeWeight == 0.0) return 0.0;
  const auto gradient =
      ReggeSolver(context.spacetime, MatterConfiguration()).actionGradientExact();
  double normSquared = 0.0;
  if (!context.scoredEdges.has_value()) {
    for (const auto &component : gradient) normSquared += std::norm(component);
  } else {
    const auto mask = scopeMask(context, gradient.size());
    for (std::size_t edgeIndex = 0; edgeIndex < gradient.size(); ++edgeIndex)
      if (edgeInScope(mask, edgeIndex))
        normSquared += std::norm(gradient[edgeIndex]);
  }
  return context.reggeWeight * normSquared;
}

/// The declared weight on the `index`-th entry of `hodgeDegrees`. An empty
/// weight list means uniform, and multiplying by exactly 1 is exact in binary
/// floating point, so an explicitly-configured single-degree run reproduces a
/// pre-weights run to the bit.
double hodgeDegreeWeight(const ObjectiveContext &context, std::size_t index) {
  if (context.hodgeDegreeWeights.empty()) return 1.0;
  if (index >= context.hodgeDegreeWeights.size()) return 1.0;
  return context.hodgeDegreeWeights[index];
}

/// \f$\|\nabla_zS_k\|^2\f$ over the edges in scope, UNWEIGHTED.
///
/// The whole-cobordism path calls the same primitive the single-objective run
/// has always called, so that path stays bit-identical; a scope restricts the
/// identical sum to its own coordinates.
double hodgeGradientNormSquared(const ObjectiveContext &context, int degree) {
  const HodgeLaplacian hodge(context.spacetime);
  if (!context.scoredEdges.has_value())
    return hodge.spectralEntropyGradientNorm(degree,
                                             context.hodgeEntropyPhaseMode);
  const auto components =
      hodge.spectralEntropyGradient(degree, context.hodgeEntropyPhaseMode);
  const auto mask = scopeMask(context, components.size());
  double normSquared = 0.0;
  for (std::size_t edgeIndex = 0; edgeIndex < components.size(); ++edgeIndex)
    if (edgeInScope(mask, edgeIndex))
      normSquared += std::norm(components[edgeIndex]);
  return normSquared;
}

/// Every declared degree's share of the Hodge term, in declaration order.
///
/// The single place the breakdown is computed: `terms`, the direction's
/// baseline and the reported contributions all read this, so a reported share
/// cannot disagree with the number that was descended.
std::vector<HodgeDegreeContribution> hodgeContributions(
    const ObjectiveContext &context) {
  std::vector<HodgeDegreeContribution> contributions;
  if (!context.spacetime || context.hodgeEntropyWeight == 0.0)
    return contributions;
  contributions.reserve(context.hodgeDegrees.size());
  for (std::size_t index = 0; index < context.hodgeDegrees.size(); ++index) {
    HodgeDegreeContribution contribution;
    contribution.degree = context.hodgeDegrees[index];
    contribution.weight = hodgeDegreeWeight(context, index);
    contribution.gradientNormSquared =
        hodgeGradientNormSquared(context, contribution.degree);
    contribution.contribution = context.hodgeEntropyWeight *
                                contribution.weight *
                                contribution.gradientNormSquared;
    contributions.push_back(contribution);
  }
  return contributions;
}

/// The Hodge stationarity term from an already-computed breakdown.
///
/// Accumulates the WEIGHTED norms and applies the entropy weight ONCE at the
/// end, which is the order the term has always been summed in. That order is
/// what preserves bit-identity: with uniform weights, multiplying each norm by
/// exactly 1 leaves the partial sums untouched and the single final multiply is
/// the same operation as before. Distributing the entropy weight across the
/// degrees instead would be algebraically equal and numerically different.
double hodgeTermFrom(const ObjectiveContext &context,
                     const std::vector<HodgeDegreeContribution> &contributions) {
  double weightedNormSum = 0.0;
  for (const auto &contribution : contributions)
    weightedNormSum += contribution.weight * contribution.gradientNormSquared;
  return context.hodgeEntropyWeight * weightedNormSum;
}

/// The connection-entropy stationarity term, and the phase gradient it was
/// assembled from. Returns the gradient through an out-parameter so a caller
/// wanting both the scalar and the direction pays for one eigendecomposition
/// rather than two.
///
/// Scope masks the GRADIENT, for the same reason the Regge term does: the
/// restricted functional is the sum over the region, whose Wirtinger derivative
/// is the masked gradient — not the masked derivative of the whole.
double connectionStationarityTerm(
    const ObjectiveContext &context, std::size_t edgeCount,
    const std::vector<bool> &mask,
    std::vector<std::complex<double>> *phaseGradient) {
  if (phaseGradient) phaseGradient->assign(edgeCount, complexd{0.0, 0.0});
  if (!context.spacetime || context.connectionEntropyWeight == 0.0) return 0.0;
  const HodgeLaplacian hodge(context.spacetime);
  const auto components = hodge.connectionSpectralEntropyPhaseGradient();
  double normSquared = 0.0;
  for (std::size_t edgeIndex = 0;
       edgeIndex < components.size() && edgeIndex < edgeCount; ++edgeIndex) {
    if (!edgeInScope(mask, edgeIndex)) continue;
    normSquared += std::norm(components[edgeIndex]);
    if (phaseGradient) (*phaseGradient)[edgeIndex] = components[edgeIndex];
  }
  return context.connectionEntropyWeight * normSquared;
}

/// The register-residual term, or zero when the engine did not compute it.
double registerResidualTerm(const ObjectiveContext &context, double weight) {
  if (!std::isfinite(context.registerResidual)) return 0.0;
  return weight * context.registerResidual;
}

/// The exact analytic ascent displacement of `β_R ‖∇_z S_Regge‖²` over the
/// edges in scope, together with the squared gradient norm it was assembled
/// from. The direction is `2 conj(H) g`; the norm is returned so a caller
/// wanting the exact baseline does not recompute the same gradient.
///
/// Scope enters by masking the GRADIENT VECTOR rather than the finished ascent,
/// and that is the mathematically exact restriction rather than a convenience:
/// the restricted functional is \f$\sum_{e\in R}|g_e|^2\f$, whose Wirtinger
/// derivative is \f$2\,\overline{H}g_R\f$ with \f$g_R\f$ the gradient masked to
/// the region. Masking the ascent afterwards would instead discard couplings
/// the restricted functional genuinely has. An empty mask leaves every
/// coordinate in, so the whole-cobordism path is untouched.
Eigen::VectorXcd reggeStationarityAscent(
    const std::shared_ptr<Spacetime> &spacetime, std::size_t edgeCount,
    double reggeWeight, const std::vector<bool> &mask,
    double *gradientNormSquared) {
  Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
  if (gradientNormSquared) *gradientNormSquared = 0.0;
  ReggeSolver reggeSolver(spacetime, MatterConfiguration());
  const auto gradientComponents = reggeSolver.actionGradientExact();
  const auto hessianRows = reggeSolver.actionHessianExact();
  Eigen::VectorXcd gradientVector(edgeCount);
  double normSquared = 0.0;
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
    if (!edgeInScope(mask, edgeIndex)) {
      gradientVector(edgeIndex) = complexd{0.0, 0.0};
      continue;
    }
    gradientVector(edgeIndex) = gradientComponents[edgeIndex];
    normSquared += std::norm(gradientComponents[edgeIndex]);
  }
  Eigen::MatrixXcd hessianMatrix(edgeCount, edgeCount);
  for (std::size_t rowIndex = 0; rowIndex < edgeCount; ++rowIndex)
    for (std::size_t columnIndex = 0; columnIndex < edgeCount; ++columnIndex)
      hessianMatrix(rowIndex, columnIndex) = hessianRows[rowIndex][columnIndex];
  ascent += reggeWeight * 2.0 * (hessianMatrix.conjugate() * gradientVector);
  if (gradientNormSquared) *gradientNormSquared = normSquared;
  return ascent;
}

/// The one permitted state channel's exact analytic contribution to the
/// direction. The gradient arrives as data and is empty outside the
/// certificates-blind mean-field sub-mode.
void addCarriedStateAscent(const ObjectiveDirectionContext &context,
                           Eigen::VectorXcd *ascent, double *baseline) {
  const auto &scalar = context.scalar;
  if (scalar.carriedStateEnergyWeight == 0.0) return;
  if (context.carriedStateEnergyGradient.size() != context.edgeCount) return;
  for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount; ++edgeIndex)
    (*ascent)(edgeIndex) += scalar.carriedStateEnergyWeight *
                            context.carriedStateEnergyGradient[edgeIndex];
  if (baseline)
    *baseline += scalar.carriedStateEnergyWeight * scalar.carriedStateEnergy;
}

}  // namespace

// ---------------------------------------------------------------- joint

std::string JointStationarityObjective::name() const {
  return ObjectiveName::kJointStationarity;
}

std::vector<std::string> JointStationarityObjective::termNames() const {
  return CobordismObjective::declaredTermNames();
}

ObjectiveTerms JointStationarityObjective::terms(
    const ObjectiveContext &context) const {
  ObjectiveTerms terms;
  if (!context.spacetime) {
    terms.reggeStationarity = std::numeric_limits<double>::infinity();
    return terms;
  }
  terms.carriedStateEnergy = carriedStateTerm(context);
  terms.reggeStationarity = reggeTerm(context);

  // Summed over the DECLARED Hodge degrees, which are configured independently
  // of the register degrees and never read from them.
  terms.hodgeStationarity =
      hodgeTermFrom(context, hodgeContributions(context));

  // The ONLY term with a phi gradient. Every L_k is blind to the connection, so
  // without this one phi is a declared field that no update can move.
  const auto edgeCount =
      context.spacetime && context.spacetime->getEdgeList()
          ? context.spacetime->getEdgeList()->toVector().size()
          : std::size_t{0};
  terms.connectionStationarity = connectionStationarityTerm(
      context, edgeCount, scopeMask(context, edgeCount), nullptr);
  return terms;
}

std::vector<HodgeDegreeContribution>
JointStationarityObjective::hodgeDegreeContributions(
    const ObjectiveContext &context) const {
  return hodgeContributions(context);
}

ObjectiveDirection JointStationarityObjective::direction(
    const ObjectiveDirectionContext &context) const {
  const auto &scalar = context.scalar;
  ObjectiveDirection result;
  result.ascent = Eigen::VectorXcd::Zero(context.edgeCount);
  // Joint mode assembles its exact baseline from the very gradients its
  // direction needs, so the line search never has to trust an accumulated
  // stage-1 trace.
  result.baselineComputed = true;
  double baseline = 0.0;
  const auto mask = scopeMask(scalar, context.edgeCount);

  if (scalar.einsteinHilbert && scalar.reggeWeight != 0.0) {
    double reggeGradientNormSquared = 0.0;
    result.ascent += reggeStationarityAscent(scalar.spacetime,
                                             context.edgeCount,
                                             scalar.reggeWeight, mask,
                                             &reggeGradientNormSquared);
    baseline += scalar.reggeWeight * reggeGradientNormSquared;
  }

  if (scalar.hodgeEntropyWeight != 0.0) {
    // For each entropy S_k, h is its exact complex-z gradient. The real
    // Hessian-vector product needed by grad ||h||^2 is the directional
    // derivative of h along conj(h), and it is CLOSED FORM
    // (`spectralEntropyGradientDirectionalDerivative`): the simplex volume
    // Hessian, the second derivative of L_k contracted against the direction,
    // and the Daleckii-Krein derivative of dS/dA on the same fixed-rank
    // stratum the value uses. No step size and no finite difference enter the
    // descent direction. The resulting ascent displacement is 2 conj(dh).
    double weightedNormSum = 0.0;
    for (std::size_t degreeIndex = 0;
         degreeIndex < scalar.hodgeDegrees.size(); ++degreeIndex) {
      const int degree = scalar.hodgeDegrees[degreeIndex];
      const double degreeWeight = hodgeDegreeWeight(scalar, degreeIndex);
      const HodgeLaplacian hodge(scalar.spacetime);
      const auto baseComponents =
          hodge.spectralEntropyGradient(degree, scalar.hodgeEntropyPhaseMode);
      double entropyGradientNormSquared = 0.0;
      std::vector<complexd> entropyAscent(context.edgeCount);
      // Scope masks the GRADIENT the HVP is contracted along, which is the
      // exact restriction of the sum for the same reason it is in the Regge
      // term: the restricted functional is the sum over the region, so the
      // direction it moves along is the region-masked gradient.
      for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount;
           ++edgeIndex) {
        if (!edgeInScope(mask, edgeIndex)) {
          entropyAscent[edgeIndex] = complexd{0.0, 0.0};
          continue;
        }
        entropyGradientNormSquared += std::norm(baseComponents[edgeIndex]);
        entropyAscent[edgeIndex] = std::conj(baseComponents[edgeIndex]);
      }
      // Accumulated and weighted ONCE at the end, matching how the scalar term
      // is summed, so an explicitly-configured single-degree run reproduces a
      // pre-weights baseline to the bit.
      weightedNormSum += degreeWeight * entropyGradientNormSquared;
      if (entropyGradientNormSquared == 0.0)
        continue;  // the exact HVP of the zero direction is zero
      const auto directionalComponents =
          hodge.spectralEntropyGradientDirectionalDerivative(
              degree, entropyAscent, scalar.hodgeEntropyPhaseMode);
      Eigen::VectorXcd directionalDerivative(context.edgeCount);
      for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount;
           ++edgeIndex)
        directionalDerivative(edgeIndex) = directionalComponents[edgeIndex];
      // The degree's weight scales its own share of the displacement, which is
      // the exact derivative of the weighted sum rather than a reweighting of
      // the finished direction.
      result.ascent += scalar.hodgeEntropyWeight * degreeWeight * 2.0 *
                       directionalDerivative.conjugate();
    }
    baseline += scalar.hodgeEntropyWeight * weightedNormSum;
  }

  addCarriedStateAscent(context, &result.ascent, &baseline);
  // The connection term. Its gradient is with respect to PHI, not z, so it
  // contributes to `phaseAscent` and never to `ascent` — the two fields move on
  // their own coordinates and are never mixed. `grad ||h||^2 = 2 conj(H) h`
  // needs the phi-Hessian; the term is instead descended by its own gradient
  // scaled by the residual, which is exact for the STATIONARITY functional in
  // the same sense the Regge term's is: both descend `||h||^2` along `conj(h)`.
  if (scalar.connectionEntropyWeight != 0.0) {
    std::vector<complexd> phaseGradient;
    const double term = connectionStationarityTerm(scalar, context.edgeCount,
                                                   mask, &phaseGradient);
    baseline += term;
    result.phaseAscent = Eigen::VectorXcd::Zero(context.edgeCount);
    for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount; ++edgeIndex)
      result.phaseAscent(edgeIndex) =
          2.0 * scalar.connectionEntropyWeight *
          std::conj(phaseGradient[edgeIndex]);
  }

  result.baseline = baseline;
  return result;
}

// ---------------------------------------------------------------- legacy

std::string LegacyObjective::name() const {
  return ObjectiveName::kLegacy;
}

std::vector<std::string> LegacyObjective::termNames() const {
  return CobordismObjective::declaredTermNames();
}

ObjectiveTerms LegacyObjective::terms(const ObjectiveContext &context) const {
  ObjectiveTerms terms;
  if (!context.spacetime) {
    terms.reggeStationarity = std::numeric_limits<double>::infinity();
    return terms;
  }
  terms.carriedStateEnergy = carriedStateTerm(context);
  terms.reggeStationarity = reggeTerm(context);
  terms.registerResidual = registerResidualTerm(context, context.gamma);
  return terms;
}

double LegacyObjective::numericalRegisterResidualWeight(
    const ObjectiveContext &context) const {
  // In residual-only legacy mode there is no Regge ray to search, so the
  // complete r_U gradient is needed. When both historical terms are enabled,
  // Legacy intentionally preserves its former analytic Regge direction (the
  // exact legacy scalar still gates the line search): evaluating the composite
  // block/target r_U at 4|E| coordinates made the compatibility mode orders of
  // magnitude slower.
  if (context.einsteinHilbert && context.reggeWeight != 0.0) return 0.0;
  return context.gamma;
}

ObjectiveDirection LegacyObjective::direction(
    const ObjectiveDirectionContext &context) const {
  const auto &scalar = context.scalar;
  ObjectiveDirection result;
  result.ascent = Eigen::VectorXcd::Zero(context.edgeCount);
  if (scalar.einsteinHilbert && scalar.reggeWeight != 0.0)
    result.ascent += reggeStationarityAscent(
        scalar.spacetime, context.edgeCount, scalar.reggeWeight,
        scopeMask(scalar, context.edgeCount), nullptr);
  addCarriedStateAscent(context, &result.ascent, nullptr);
  return result;
}

// -------------------------------------------------- mediated correspondence

std::string MediatedCorrespondenceObjective::name() const {
  return ObjectiveName::kMediatedCorrespondence;
}

std::vector<std::string> MediatedCorrespondenceObjective::termNames() const {
  return CobordismObjective::declaredTermNames();
}

ObjectiveTerms MediatedCorrespondenceObjective::terms(
    const ObjectiveContext &context) const {
  ObjectiveTerms terms;
  if (!context.spacetime) {
    terms.reggeStationarity = std::numeric_limits<double>::infinity();
    return terms;
  }
  terms.carriedStateEnergy = carriedStateTerm(context);
  terms.actionMagnitude =
      context.einsteinHilbert && context.reggeWeight != 0.0
          ? context.reggeWeight *
                std::abs(ReggeSolver(context.spacetime, MatterConfiguration())
                             .dualReggeAction())
          : 0.0;
  terms.registerResidual = registerResidualTerm(context, 1.0);
  return terms;
}

double MediatedCorrespondenceObjective::numericalRegisterResidualWeight(
    const ObjectiveContext &) const {
  return 1.0;
}

ObjectiveDirection MediatedCorrespondenceObjective::direction(
    const ObjectiveDirectionContext &context) const {
  const auto &scalar = context.scalar;
  ObjectiveDirection result;
  result.ascent = Eigen::VectorXcd::Zero(context.edgeCount);

  if (scalar.einsteinHilbert && scalar.reggeWeight != 0.0) {
    ReggeSolver reggeSolver(scalar.spacetime, MatterConfiguration());
    const complexd action = reggeSolver.dualReggeAction();
    if (std::abs(action) > 0.0) {
      const auto actionGradient = reggeSolver.actionGradientExact();
      for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount;
           ++edgeIndex)
        result.ascent(edgeIndex) += scalar.reggeWeight *
                                    (action / std::abs(action)) *
                                    std::conj(actionGradient[edgeIndex]);
    }
  }
  addCarriedStateAscent(context, &result.ascent, nullptr);
  return result;
}

}  // namespace tessera::cobordism
