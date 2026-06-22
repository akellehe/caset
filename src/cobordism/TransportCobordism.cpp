// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/TransportCobordism.h"

#include <set>
#include <stdexcept>
#include <utility>

#include "cobordism/CobordismRelaxer.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/RegisterTopology.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;

TransportCobordism::TransportCobordism(
    const std::vector<std::vector<std::complex<double>>> &inputStates,
    double beta, double epsilon, int maxIters, std::uint64_t seed,
    std::shared_ptr<TopologyBuilder> topology, bool verbose,
    StateResidualMode stateMode)
    : inputStates_(inputStates),
      beta_(beta),
      epsilon_(epsilon),
      maxIters_(maxIters),
      seed_(seed),
      verbose_(verbose),
      topology_(topology ? std::move(topology)
                         : std::make_shared<RegisterTopology>()),
      stateMode_(stateMode) {
  if (inputStates_.empty())
    throw std::invalid_argument("TransportCobordism: inputStates is empty");
  stateDim_ = inputStates_.front().size();
  // The admissible state dimension travels with the topology (the register: the
  // color triple d = 3).
  topology_->validateStateDim(stateDim_);

  buildSeed();
  computeStateTargets();
  if (stateHoles_.empty())
    throw std::invalid_argument(
        "TransportCobordism: the topology supplied no triangle-hole read-out "
        "(readoutHoles); use a transport topology (RegisterTopology / "
        "TripartiteRegisterTopology), not an edge-loop operator topology");
  optimize();
}

void TransportCobordism::buildSeed() {
  cobordism_ = topology_->build(stateDim_, seed_, boundaryCells_);
}

void TransportCobordism::computeStateTargets() {
  // The transport pins INPUTS only and reads the EMERGENT result block: the
  // topology's exact triangle-hole read-out gives the pinned input holes + their
  // induced-orientation targets, and the first unpinned window (the result block).
  topology_->readoutHoles(cobordism_, inputStates_, stateHoles_, holeTargets_,
                          resultHoles_, resultSigns_);
}

void TransportCobordism::optimize() {
  // Relax the interior edge lengths to a stationary point of the dual Regge
  // action under the input-pinning residual r = beta||grad S||^2 + r_state, scored
  // over the EXACT triangle holes. The result block is left free -- it emerges.
  const bool periodPin = (stateMode_ == StateResidualMode::PeriodPin);
  CobordismRelaxer::relaxInterior(cobordism_, beta_, /*stateLoops=*/{},
                                  /*stateTargets=*/{}, stateHoles_, holeTargets_,
                                  maxIters_, stats_.relaxIterations, periodPin,
                                  epsilon_, verbose_,
                                  static_cast<int>(topology_->registerDegree()));
  readResult();
  stats_.converged = stats_.residual < epsilon_;
}

void TransportCobordism::readResult() {
  // The register read-out degree travels with the topology: k=1 (b_1, triangle
  // holes) on S^2, k=2 (b_2, tetrahedral holes) on S^3. cyclePeriods /
  // residualForPeriods are degree-general (a hole is a (k+2)-vertex tuple).
  EigenstateSynthesis es(cobordism_, static_cast<int>(topology_->registerDegree()));
  bulkCells_ = es.bulkMinusBoundaryCells();

  // === topology stats ===
  const auto bws = CobordismRelaxer::betti(*cobordism_);
  stats_.bettiCobordism = bws;
  stats_.b1Bulk = (bws.size() > 1) ? bws[1] : 0;
  stats_.interiorVertices = static_cast<int>(es.interiorVertexCount());
  stats_.topology = topology_->name();

  // === emergent result read-out ===
  // The relaxed geometry's harmonic periods over the result block's color holes
  // (cyclePeriods): the inputs carried through the bulk to the result window. The
  // result is the first-harmonic row; its sum is the color charge sigma_R.
  result_.clear();
  if (!resultHoles_.empty()) {
    const auto P = es.cyclePeriods(resultHoles_);
    const std::size_t m = resultHoles_.size();
    if (m > 0 && P.size() >= m && P.size() % m == 0)
      result_.assign(P.begin(), P.begin() + static_cast<std::ptrdiff_t>(m));
  }
  // Apply the result block's induced-orientation signs (symmetric with the signed
  // input targets) so sigma_R = sum(result_) is the relabeling-invariant Stokes
  // charge, not a bare sum of per-hole sorted-reference periods (#412). Empty signs
  // (e.g. the bipartite register) leave the raw periods unchanged.
  if (!resultSigns_.empty() && resultSigns_.size() == result_.size())
    for (std::size_t k = 0; k < result_.size(); ++k)
      result_[k] *= static_cast<double>(resultSigns_[k]);

  // === residual read-out: beta*||grad_I S||^2 + r_state at the relaxed metric ===
  ReggeSolver residualSolver(cobordism_, MatterConfiguration());
  const auto gRes = residualSolver.actionGradientExact();
  std::set<std::pair<std::uint64_t, std::uint64_t>> interiorSet;
  for (const auto &uv : es.interiorEdges()) interiorSet.insert(uv);
  const auto resEdges = cobordism_->getEdgeList()->toVector();
  double actionN2 = 0.0;  // free (interior) edges only -- dW is fixed
  for (std::size_t i = 0; i < resEdges.size() && i < gRes.size(); ++i)
    if (interiorSet.count(CobordismRelaxer::edgeKey(resEdges[i])))
      actionN2 += std::norm(gRes[i]);
  stats_.statActionResidual = beta_ * actionN2;
  stats_.dualAction = residualSolver.dualReggeAction();
  const bool periodPin = (stateMode_ == StateResidualMode::PeriodPin);
  stats_.stateMode = periodPin ? "r_psi" : "r_U";
  stats_.stateResidual =
      stateHoles_.empty()
          ? 0.0
          : (periodPin ? es.periodGapForPeriods(stateHoles_, holeTargets_)
                       : es.residualForPeriods(stateHoles_, holeTargets_));
  stats_.residual = stats_.statActionResidual + stats_.stateResidual;
}

}  // namespace tessera::cobordism
