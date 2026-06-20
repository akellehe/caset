// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MergeCobordism.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "cobordism/ChainComplex.h"
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

namespace {

// The sorted (min,max) endpoint key of a mesh edge.
std::pair<std::uint64_t, std::uint64_t> edgeKey(const ::tessera::mesh::Edge *e) {
  const auto a = e->getSource()->getId();
  const auto b = e->getTarget()->getId();
  return {std::min(a, b), std::max(a, b)};
}

// Betti numbers of a complex (combinatorial, geometry-free).
std::vector<int> betti(const Spacetime &st) {
  return ChainComplex::fromSpacetime(st).bettiNumbers();
}

// One bounded Gauss-Newton / Levenberg-Marquardt descent of the total residual
// r = beta*||grad_I S||^2 + r_state over the interior edge squared-lengths, using
// the EXACT analytic gradient and the SPARSE analytic Hessian of the dual Regge
// action (actionHessianExactSparse, #381 — assembled at O(nnz), the interior
// block extracted from it). dW is held fixed (Dirichlet), so the Regge
// stationarity is over interior edges only. The r_state term is selected by
// `mode`: r_U realizability or r_psi hard period-pin. The pinned states are
// scored over EITHER the topology's EXACT triangle holes (stateHoles, the #353
// register's residualForPeriods/periodGapForPeriods) OR its SOFT edge-loops
// (stateLoops, the operator's S^1 residualForLoops/periodGapForLoops) -- whichever
// the topology supplied (mutually exclusive; holes take precedence). Returns the
// final r; leaves the interior edges at the best point found.
double relaxInterior(
    const std::shared_ptr<Spacetime> &st, double beta,
    const std::vector<EigenstateSynthesis::EdgeLoop> &stateLoops,
    const std::vector<std::complex<double>> &stateTargets,
    const std::vector<std::vector<std::uint64_t>> &stateHoles,
    const std::vector<std::complex<double>> &holeTargets,
    int maxIters, int &iterCounter, MergeCobordism::StateResidualMode mode,
    double stateEpsilon, bool verbose = false) {
  EigenstateSynthesis es(st, 1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> interiorRank;
  for (const auto &uv : es.interiorEdges()) interiorRank.emplace(uv, 0);
  const auto edges = st->getEdgeList()->toVector();
  std::vector<std::size_t> interiorIdx;
  std::vector<::tessera::mesh::Edge *> interiorEdgePtr;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (interiorRank.count(edgeKey(edges[i]))) {
      interiorIdx.push_back(i);
      interiorEdgePtr.push_back(edges[i]);
    }
  }
  const std::size_t nI = interiorIdx.size();
  const bool periodPin = (mode == MergeCobordism::StateResidualMode::PeriodPin);
  // EXACT triangle-hole path (#353 register) vs SOFT edge-loop path (operator).
  const bool useHoles = !stateHoles.empty();

  // r_state: the selected state residual of the pinned states over the boundary
  // cycles, read against the live metric so it tracks the relaxation -- r_psi
  // (carried-vs-target period gap) or r_U (exact-period state non-harmonicity).
  auto stateCost = [&]() -> double {
    if (useHoles)
      return periodPin ? es.periodGapForPeriods(stateHoles, holeTargets)
                       : es.residualForPeriods(stateHoles, holeTargets);
    if (stateLoops.empty()) return 0.0;
    return periodPin ? es.periodGapForLoops(stateLoops, stateTargets)
                     : es.residualForLoops(stateLoops, stateTargets);
  };
  // The action residual is the Regge stationarity over the FREE (interior) edges
  // only: dW is fixed, so its action gradient is an irreducible Dirichlet
  // reaction, not part of the interior stationarity.
  auto actionNorm2 = [&]() {
    ReggeSolver solver(st, MatterConfiguration());
    const auto g = solver.actionGradientExact();
    double n2 = 0.0;
    for (const std::size_t e : interiorIdx) n2 += std::norm(g[e]);
    return n2;
  };

  if (nI == 0) return beta * actionNorm2() + stateCost();

  // cellSimplices order (the state-residual gradient) -> interior param index, so
  // the analytic state-residual gradient folds into the action gradient.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> paramOf;
  for (std::size_t c = 0; c < nI; ++c) paramOf[edgeKey(interiorEdgePtr[c])] = c;
  const auto &cells = es.cellSimplices();
  std::vector<int> cellToParam(cells.size(), -1);
  for (std::size_t j = 0; j < cells.size(); ++j) {
    if (cells[j].size() < 2) continue;
    const std::pair<std::uint64_t, std::uint64_t> key{
        std::min(cells[j][0], cells[j][1]), std::max(cells[j][0], cells[j][1])};
    const auto it = paramOf.find(key);
    if (it != paramOf.end()) cellToParam[j] = static_cast<int>(it->second);
  }

  auto cost = [&]() { return beta * actionNorm2() + stateCost(); };

  double mu = 1e-3;
  double best = cost();
  for (int it = 0; it < maxIters; ++it, ++iterCounter) {
    ReggeSolver solver(st, MatterConfiguration());
    const auto gVec = solver.actionGradientExact();          // |E| complex
    const auto hSparse = solver.actionHessianExactSparse();  // |E|x|E| sparse

    // The interior gradient g_I and the interior-interior Hessian block H_II,
    // extracted from the sparse Hessian (no dense |E|^2 assembly).
    Eigen::VectorXcd gI(nI);
    for (std::size_t c = 0; c < nI; ++c) gI(c) = gVec[interiorIdx[c]];
    Eigen::MatrixXcd HII(nI, nI);
    for (std::size_t r = 0; r < nI; ++r)
      for (std::size_t c = 0; c < nI; ++c)
        HII(r, c) = hSparse.coeff(static_cast<Eigen::Index>(interiorIdx[r]),
                                  static_cast<Eigen::Index>(interiorIdx[c]));

    // Analytic gradient and GN Hessian of beta*||grad_I S||^2 over the interior
    // lengths, plus the analytic state-residual gradient (cellSimplices order).
    // The state-residual gradient is folded in ONLY while r_state is ABOVE its
    // convergence floor. When the pinned inputs already carry exactly (r_state ->
    // 0 -- e.g. the distinct-window junction, where every neutral input is
    // carried on its own independent cycle), its analytic gradient is numerical
    // noise that, divided by the ill-conditioned action GN Hessian, explodes the
    // step (the relaxation then stalls at iteration 0). Once the state term is
    // converged we minimize the action alone. [#396]
    const double rStateNow = stateCost();
    Eigen::VectorXd grad = (2.0 * beta * (HII.adjoint() * gI)).real();
    if (rStateNow > stateEpsilon && (useHoles || !stateLoops.empty())) {
      const auto rg =
          useHoles
              ? (periodPin
                     ? es.periodGapForPeriodsGradient(stateHoles, holeTargets)
                     : es.residualForPeriodsGradient(stateHoles, holeTargets))
              : (periodPin
                     ? es.periodGapForLoopsGradient(stateLoops, stateTargets)
                     : es.residualForLoopsGradient(stateLoops, stateTargets));
      for (std::size_t j = 0; j < rg.size() && j < cellToParam.size(); ++j)
        if (cellToParam[j] >= 0) grad(cellToParam[j]) += rg[j];
    }
    Eigen::MatrixXd B = (2.0 * beta * (HII.adjoint() * HII)).real();

    Eigen::VectorXd x0(nI);
    for (std::size_t c = 0; c < nI; ++c)
      x0(c) = interiorEdgePtr[c]->getSquaredLength().real();

    bool improved = false;
    for (int ls = 0; ls < 12; ++ls) {  // mu line search
      Eigen::MatrixXd A = B;
      for (std::size_t c = 0; c < nI; ++c) A(c, c) += mu * (B(c, c) + 1.0);
      const Eigen::VectorXd delta = A.ldlt().solve(-grad);
      Eigen::VectorXd x = x0 + delta;
      // Emergent causal type: l^2 is free (spacelike/null/timelike), no clamp; a
      // degenerate step blows the action up and the line search rejects it.
      for (std::size_t c = 0; c < nI; ++c)
        interiorEdgePtr[c]->setSquaredLength(std::complex<double>(x(c), 0.0));
      // A degenerate step (a null/zero-area edge) makes the action / period
      // read-out fail loudly; treat the throw as no improvement so the line
      // search damps mu and retries, rather than aborting the whole relaxation.
      double trial;
      try {
        trial = cost();
      } catch (...) {
        trial = std::numeric_limits<double>::infinity();
      }
      if (trial < best) {
        best = trial;
        mu = std::max(mu * 0.5, 1e-9);
        improved = true;
        break;
      }
      mu = std::min(mu * 4.0, 1e9);
    }
    if (verbose && (it % 25 == 0 || it + 1 == maxIters || !improved))
      std::cerr << "[merge relax] iter " << it << "/" << maxIters << "  r=" << best
                << "\n";
    if (!improved) {  // restore best point and stop
      for (std::size_t c = 0; c < nI; ++c)
        interiorEdgePtr[c]->setSquaredLength(std::complex<double>(x0(c), 0.0));
      break;
    }
  }
  return best;
}

}  // namespace

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
                         : std::make_shared<RegisterTopology>()),
      stateMode_(stateMode) {
  if (inputStates_.empty())
    throw std::invalid_argument("MergeCobordism: inputStates is empty");
  stateDim_ = inputStates_.front().size();
  // The admissible state dimension travels with the topology (operator: a power
  // of two; register: the color triple d = 3).
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
  if (outputStates_.empty() && !topology_->emergesResult())
    throw std::invalid_argument(
        "MergeCobordism: outputStates is required when U is not supplied (this "
        "topology pins the result; only an emergent-result topology -- the #353 "
        "register -- may pin inputs alone and read the emergent result block)");

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
  // The states (inputs then outputs) pinned over the topology's read-out cycles.
  // A topology supplies EXACTLY ONE of the two read-outs: edge-loops (operator's
  // S^1, scored over the SOFT residualForLoops) or triangle holes (#353 register,
  // scored over the EXACT residualForPeriods + the result block read over
  // cyclePeriods). The unused one stays empty, so the merge dispatches on which.
  std::vector<std::vector<std::complex<double>>> states;
  states.reserve(inputStates_.size() + outputStates_.size());
  for (const auto &s : inputStates_) states.push_back(s);
  for (const auto &s : outputStates_) states.push_back(s);
  topology_->readout(cobordism_, states, stateLoops_, stateTargets_);
  // The operator pins inputs AND outputs (readout() above gets both); the #353
  // register pins INPUTS only and reads the EMERGENT result block (the first
  // unpinned block), so readoutHoles gets the inputs alone.
  topology_->readoutHoles(cobordism_, inputStates_, stateHoles_, holeTargets_,
                          resultHoles_);
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
  relaxInterior(cobordism_, beta_, stateLoops_, stateTargets_, stateHoles_,
                holeTargets_, maxIters_, stats_.relaxIterations, stateMode_,
                epsilon_, verbose_);
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
  const auto bws = betti(*cobordism_);
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

  // === emergent output state read-out (#376) ===
  // The output the relaxed geometry produces from the INPUTS alone (the #353
  // inputs -> emergent output flow): carry the pinned input periods as a metric
  // L_1(W) harmonic and read its periods over the OUTPUT cycles. Populated in
  // both modes — primary in U-supplied mode, a consistency read (emergent vs the
  // pinned target) in output-supplied mode. Read from the relaxed (live) metric,
  // so it is emergent, not the seed. The topology's readout() emits the cycles
  // state-by-state (inputs then outputs) with a fixed loops-per-state, so the
  // input/output split is at nIn = loopsPerState * (#input states); skip the
  // read when that split is not determinate (e.g. some states went unpinned).
  outputState_.clear();
  if (!resultHoles_.empty()) {
    // === EXACT (#353 register) read ===
    // The relaxed geometry's harmonic periods over the EMERGENT result block's
    // color holes (cyclePeriods): carriedDim rows x (#result holes), row-major.
    // Row 0 is the result block's emergent color amplitudes (the #353
    // cyclePeriods(holes)[0, result-block] read); their sum is the color charge
    // sigma_R (-> 0 iff the emergent result is color-neutral). EXACT, not the soft
    // edge-loop carry -- so this never re-introduces the loop residual floor.
    const auto P = es.cyclePeriods(resultHoles_);
    const std::size_t m = resultHoles_.size();
    if (m > 0 && P.size() >= m && P.size() % m == 0)
      outputState_.assign(P.begin(), P.begin() + static_cast<std::ptrdiff_t>(m));
  } else {
    // === SOFT (operator S^1) read ===
    // The output the relaxed geometry produces from the INPUTS alone: carry the
    // pinned input periods as a metric L_1(W) harmonic and read its periods over
    // the OUTPUT cycles. The topology emits loopsPerState() cycles per pinned
    // state, inputs first, and pins at most its state capacity -- so
    // stateLoops_.size() == loopsPerState * totalStates holds IFF every state was
    // pinned (no truncation). Take loopsPerState from the topology rather than
    // inferring it by division (which silently mis-splits when states are
    // dropped), and require an exact, untruncated, target-matched read; otherwise
    // skip (an honest empty, not a frame-/split-dependent value).
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
  }

  // === residual read-out: beta*||grad_I S||^2 + r_state at the relaxed metric ===
  ReggeSolver residualSolver(cobordism_, MatterConfiguration());
  const auto gRes = residualSolver.actionGradientExact();
  std::set<std::pair<std::uint64_t, std::uint64_t>> interiorSet;
  for (const auto &uv : es.interiorEdges()) interiorSet.insert(uv);
  const auto resEdges = cobordism_->getEdgeList()->toVector();
  double actionN2 = 0.0;  // free (interior) edges only — dW is fixed
  for (std::size_t i = 0; i < resEdges.size() && i < gRes.size(); ++i)
    if (interiorSet.count(edgeKey(resEdges[i]))) actionN2 += std::norm(gRes[i]);
  stats_.statActionResidual = beta_ * actionN2;
  stats_.dualAction = residualSolver.dualReggeAction();
  const bool periodPin = (stateMode_ == StateResidualMode::PeriodPin);
  stats_.stateMode = periodPin ? "r_psi" : "r_U";
  // The pinned-input state residual, scored over the EXACT triangle holes (#353
  // register) or the SOFT edge-loops (operator) -- whichever the topology used.
  if (!stateHoles_.empty())
    stats_.stateResidual =
        periodPin ? es.periodGapForPeriods(stateHoles_, holeTargets_)
                  : es.residualForPeriods(stateHoles_, holeTargets_);
  else
    stats_.stateResidual =
        stateLoops_.empty()
            ? 0.0
            : (periodPin ? es.periodGapForLoops(stateLoops_, stateTargets_)
                         : es.residualForLoops(stateLoops_, stateTargets_));
  stats_.residual = stats_.statActionResidual + stats_.stateResidual;
}

}  // namespace tessera::cobordism
