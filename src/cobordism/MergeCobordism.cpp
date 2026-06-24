// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MergeCobordism.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>


#include "cobordism/ChainComplex.h"
#include "cobordism/CobordismRelaxer.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/RegisterTopology.h"
#include "cobordism/TorusOperatorTopology.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using ::tessera::spacetime::Spacetime;


MergeCobordism::MergeCobordism(
    const std::vector<std::vector<std::complex<double>>> &inputStates,
    const std::vector<std::vector<std::complex<double>>> &outputStates,
    const std::vector<std::complex<double>> &U, double beta, double epsilon,
    int maxIters, std::uint64_t seed, std::shared_ptr<TopologyBuilder> topology,
    bool verbose, StateResidualMode stateMode)
    : inputStates_(inputStates),
      outputStates_(outputStates),
      beta_(beta),
      epsilon_(epsilon),
      maxIters_(maxIters),
      seed_(seed),
      verbose_(verbose),
      topology_(topology ? std::move(topology)
                         : std::make_shared<TorusOperatorTopology>()),
      stateMode_(stateMode) {
  if (inputStates_.empty())
    throw std::invalid_argument("MergeCobordism: inputStates is empty");
  stateDim_ = inputStates_.front().size();
  // The admissible state dimension travels with the topology (the operator: a
  // power of two). The carried-rep TRANSPORT (the #353 color register, the #396
  // proton junction) is TransportCobordism, not this merge.
  topology_->validateStateDim(stateDim_);

  if (!U.empty()) {
    // U-supplied mode: the output emerges, so outputStates must be OMITTED (the
    // two modes are mutually exclusive -- supplying both would silently discard
    // the caller's outputStates when computeOutputsFromOperator overwrites them).
    if (!outputStates_.empty())
      throw std::invalid_argument(
          "MergeCobordism: supply either outputStates or U, not both");
    if (U.size() != stateDim_ * stateDim_)
      throw std::invalid_argument(
          "MergeCobordism: U must be a d x d row-major operator");
    computeOutputsFromOperator(U);
  }
  if (outputStates_.empty())
    throw std::invalid_argument(
        "MergeCobordism: outputStates (or U) is required -- the merge pins inputs "
        "AND outputs and reads the operator / final state. For pinning inputs "
        "alone and reading the emergent result, use TransportCobordism.");

  buildSeed();
  computeStateTargets();
  optimize();
}

void MergeCobordism::computeOutputsFromOperator(
    const std::vector<std::complex<double>> &U) {
  const std::size_t d = stateDim_;
  outputStates_.clear();
  for (const auto &psi : inputStates_) {
    if (psi.size() != d)
      throw std::invalid_argument("MergeCobordism: ragged input state dimension");
    std::vector<std::complex<double>> out(d, {0.0, 0.0});
    for (std::size_t i = 0; i < d; ++i)
      for (std::size_t j = 0; j < d; ++j) out[i] += U[i * d + j] * psi[j];
    outputStates_.push_back(std::move(out));
  }
}

void MergeCobordism::computeStateTargets() {
  // The merge pins inputs AND outputs over the operator topology's edge-loop
  // read-out (the S^1 cycles), scored over residualForLoops. (The exact
  // triangle-hole transport read-out lives in TransportCobordism.)
  std::vector<std::vector<std::complex<double>>> states;
  states.reserve(inputStates_.size() + outputStates_.size());
  for (const auto &s : inputStates_) states.push_back(s);
  for (const auto &s : outputStates_) states.push_back(s);
  topology_->readout(cobordism_, states, stateLoops_, stateTargets_);
}

void MergeCobordism::buildSeed() {
  cobordism_ = topology_->build(stateDim_, seed_, boundaryCells_);
}

void MergeCobordism::optimize() {
  // Relax the interior edge lengths to a stationary point of the dual Regge
  // action under the state-pinning residual r = beta||grad S||^2 + r_state. No
  // combinatorial move-search: the relaxed seed triangulation is a local optimum
  // (every boundary-fixed Pachner move only raised the residual in testing), so
  // the relax alone determines the geometry. [#388]
  const bool periodPin = (stateMode_ == StateResidualMode::PeriodPin);
  CobordismRelaxer::relaxInterior(cobordism_, beta_, stateLoops_, stateTargets_,
                                  /*stateHoles=*/{}, /*holeTargets=*/{}, maxIters_,
                                  stats_.relaxIterations, periodPin, epsilon_,
                                  verbose_,
                                  static_cast<int>(topology_->registerDegree()));
  extractOperator();
  stats_.converged = stats_.residual < epsilon_;
}

void MergeCobordism::extractOperator() {
  EigenstateSynthesis es(cobordism_, 1);
  bulkCells_ = es.bulkMinusBoundaryCells();
  const auto H = es.bulkMinusBoundaryHarmonicMatrix();
  const std::size_t ncols = bulkCells_.size();
  const std::size_t dim = ncols ? H.size() / ncols : 0;

  // === topology stats ===
  stats_.kerL1Bulk = static_cast<int>(dim);
  const auto bws = CobordismRelaxer::betti(*cobordism_);
  stats_.bettiCobordism = bws;
  stats_.b1Bulk = (bws.size() > 1) ? bws[1] : 0;
  stats_.interiorVertices = static_cast<int>(es.interiorVertexCount());
  stats_.topology = topology_->name() + "  ker L1(W-dW)=" + std::to_string(dim);

  // === operator read-out — DEFERRED (#376) ===
  // unvec(ker L_1(W - dW)) is the operator, but ker L_1(W - dW) here is a
  // (d^2-1)-dim subspace of the interior 1-cochains (C^|interior C_1|) with no
  // basis-independent map to the d x d operator: the kernel basis is only fixed
  // up to an O(dim) rotation, so a reshape is frame-dependent. A principled read
  // needs distinguished interior Choi-cycles the current topology does not
  // supply (the interior-handle operator-topology rework). Rather than fabricate
  // a frame-dependent value, the operator stays empty (honest floor signal).
  choiState_.clear();
  operatorU_.clear();

  // === emergent final-state read-out (#376) ===
  // The output the relaxed geometry produces from the inputs: carry the pinned
  // input periods as a metric L_1(W) harmonic and read its periods over the OUTPUT
  // cycles. Primary in U-supplied mode (the final state is the emergent quantity);
  // a consistency read (emergent vs. the pinned target) in output-supplied mode.
  // The topology emits loopsPerState() cycles per pinned state (inputs first), so
  // the input/output split is at nIn = loopsPerState * (#input states); skip the
  // read when the split is not determinate (e.g. some state went unpinned).
  outputState_.clear();
  const std::size_t totalStates = inputStates_.size() + outputStates_.size();
  const std::size_t loopsPerState = topology_->loopsPerState();
  if (totalStates > 0 && loopsPerState > 0 &&
      stateLoops_.size() == loopsPerState * totalStates &&
      stateTargets_.size() == stateLoops_.size()) {
    const std::size_t nIn = loopsPerState * inputStates_.size();
    if (nIn > 0 && nIn < stateLoops_.size()) {
      const std::vector<TopologyBuilder::EdgeLoop> inLoops(
          stateLoops_.begin(),
          stateLoops_.begin() + static_cast<std::ptrdiff_t>(nIn));
      const std::vector<std::complex<double>> inTargets(
          stateTargets_.begin(),
          stateTargets_.begin() + static_cast<std::ptrdiff_t>(nIn));
      const std::vector<TopologyBuilder::EdgeLoop> outLoops(
          stateLoops_.begin() + static_cast<std::ptrdiff_t>(nIn),
          stateLoops_.end());
      const std::vector<std::complex<double>> psiIn =
          es.carriedRepresentativeOverLoops(inLoops, inTargets);
      if (!psiIn.empty())
        outputState_ = es.periodsOfCochainOverLoops(psiIn, outLoops);
    }
  }

  // === residual read-out: beta*||grad_I S||^2 + r_state at the relaxed metric ===
  ReggeSolver residualSolver(cobordism_, MatterConfiguration());
  const auto gRes = residualSolver.actionGradientExact();
  std::set<std::pair<std::uint64_t, std::uint64_t>> interiorSet;
  for (const auto &uv : es.interiorEdges()) interiorSet.insert(uv);
  const auto resEdges = cobordism_->getEdgeList()->toVector();
  double actionN2 = 0.0;  // free (interior) edges only — dW is fixed
  for (std::size_t i = 0; i < resEdges.size() && i < gRes.size(); ++i)
    if (interiorSet.count(CobordismRelaxer::edgeKey(resEdges[i])))
      actionN2 += std::norm(gRes[i]);
  stats_.statActionResidual = beta_ * actionN2;
  stats_.dualAction = residualSolver.dualReggeAction();
  const bool periodPin = (stateMode_ == StateResidualMode::PeriodPin);
  stats_.stateMode = periodPin ? "r_psi" : "r_U";
  // The pinned-state residual over the operator's edge-loops (r_U or r_psi).
  stats_.stateResidual =
      stateLoops_.empty()
          ? 0.0
          : (periodPin ? es.periodGapForLoops(stateLoops_, stateTargets_)
                       : es.residualForLoops(stateLoops_, stateTargets_));
  stats_.residual = stats_.statActionResidual + stats_.stateResidual;
}

}  // namespace tessera::cobordism
