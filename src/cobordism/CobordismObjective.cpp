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
          "region_targets",
          "register_degrees",
          "regge_weight",
          "hodge_entropy_weight",
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
         terms.registerResidual + terms.actionMagnitude +
         terms.carriedStateEnergy;
}

std::vector<std::string> CobordismObjective::declaredTermNames() {
  return {"regge_stationarity", "hodge_stationarity", "register_residual",
          "action_magnitude", "carried_state_energy"};
}

namespace {

/// The one permitted state channel, shared by every built-in objective. The
/// engine passes the energy as a number that is exactly zero outside the
/// certificates-blind mean-field sub-mode.
double carriedStateTerm(const ObjectiveContext &context) {
  if (context.carriedStateEnergyWeight == 0.0) return 0.0;
  return context.carriedStateEnergyWeight * context.carriedStateEnergy;
}

/// `β_R ‖∇_z S_Regge‖²`, computed from the geometry alone, or zero where the
/// Einstein-Hilbert term is deselected or unweighted.
double reggeTerm(const ObjectiveContext &context) {
  if (!context.einsteinHilbert || context.reggeWeight == 0.0) return 0.0;
  const auto gradient =
      ReggeSolver(context.spacetime, MatterConfiguration()).actionGradientExact();
  double normSquared = 0.0;
  for (const auto &component : gradient) normSquared += std::norm(component);
  return context.reggeWeight * normSquared;
}

/// The register-residual term, or zero when the engine did not compute it.
double registerResidualTerm(const ObjectiveContext &context, double weight) {
  if (!std::isfinite(context.registerResidual)) return 0.0;
  return weight * context.registerResidual;
}

/// The exact analytic ascent displacement of `β_R ‖∇_z S_Regge‖²`, together
/// with the squared gradient norm it was assembled from. The direction is
/// `2 conj(H) g`; the norm is returned so a caller wanting the exact baseline
/// does not recompute the same gradient.
Eigen::VectorXcd reggeStationarityAscent(
    const std::shared_ptr<Spacetime> &spacetime, std::size_t edgeCount,
    double reggeWeight, double *gradientNormSquared) {
  Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
  if (gradientNormSquared) *gradientNormSquared = 0.0;
  ReggeSolver reggeSolver(spacetime, MatterConfiguration());
  const auto gradientComponents = reggeSolver.actionGradientExact();
  const auto hessianRows = reggeSolver.actionHessianExact();
  Eigen::VectorXcd gradientVector(edgeCount);
  double normSquared = 0.0;
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
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
  return "joint_stationarity";
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

  double entropyStationarity = 0.0;
  if (context.hodgeEntropyWeight != 0.0)
    for (int degree : context.registerDegrees)
      entropyStationarity +=
          HodgeLaplacian(context.spacetime)
              .spectralEntropyGradientNorm(degree,
                                           context.hodgeEntropyPhaseMode);
  terms.hodgeStationarity = context.hodgeEntropyWeight * entropyStationarity;
  return terms;
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

  if (scalar.einsteinHilbert && scalar.reggeWeight != 0.0) {
    double reggeGradientNormSquared = 0.0;
    result.ascent += reggeStationarityAscent(scalar.spacetime,
                                             context.edgeCount,
                                             scalar.reggeWeight,
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
    for (int degree : scalar.registerDegrees) {
      const HodgeLaplacian hodge(scalar.spacetime);
      const auto baseComponents =
          hodge.spectralEntropyGradient(degree, scalar.hodgeEntropyPhaseMode);
      double entropyGradientNormSquared = 0.0;
      std::vector<complexd> entropyAscent(context.edgeCount);
      for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount;
           ++edgeIndex) {
        entropyGradientNormSquared += std::norm(baseComponents[edgeIndex]);
        entropyAscent[edgeIndex] = std::conj(baseComponents[edgeIndex]);
      }
      baseline += scalar.hodgeEntropyWeight * entropyGradientNormSquared;
      if (entropyGradientNormSquared == 0.0)
        continue;  // the exact HVP of the zero direction is zero
      const auto directionalComponents =
          hodge.spectralEntropyGradientDirectionalDerivative(
              degree, entropyAscent, scalar.hodgeEntropyPhaseMode);
      Eigen::VectorXcd directionalDerivative(context.edgeCount);
      for (std::size_t edgeIndex = 0; edgeIndex < context.edgeCount;
           ++edgeIndex)
        directionalDerivative(edgeIndex) = directionalComponents[edgeIndex];
      result.ascent +=
          scalar.hodgeEntropyWeight * 2.0 * directionalDerivative.conjugate();
    }
  }

  addCarriedStateAscent(context, &result.ascent, &baseline);
  result.baseline = baseline;
  return result;
}

// ---------------------------------------------------------------- legacy

std::string LegacyObjective::name() const { return "legacy"; }

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
    result.ascent += reggeStationarityAscent(scalar.spacetime,
                                             context.edgeCount,
                                             scalar.reggeWeight, nullptr);
  addCarriedStateAscent(context, &result.ascent, nullptr);
  return result;
}

// -------------------------------------------------- mediated correspondence

std::string MediatedCorrespondenceObjective::name() const {
  return "mediated_correspondence";
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
