// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MergeCobordism.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
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
// r = beta*||grad_I S||^2 + r_psi over the interior edge squared-lengths, using
// the EXACT analytic gradient and the SPARSE analytic Hessian of the dual Regge
// action (actionHessianExactSparse, #381 — assembled at O(nnz), the interior
// block extracted from it). dW is held fixed (Dirichlet), so the Regge
// stationarity is over interior edges only. Returns the final r; leaves the
// interior edges at the best point found.
double relaxInterior(
    const std::shared_ptr<Spacetime> &st, double beta,
    const std::vector<EigenstateSynthesis::EdgeLoop> &stateLoops,
    const std::vector<std::complex<double>> &stateTargets,
    int maxIters, int &iterCounter, bool verbose = false) {
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

  // r_psi: the carried-harmonic residual of the pinned states over the boundary
  // cycles, read against the live metric so it tracks the relaxation.
  auto stateCost = [&]() {
    return stateLoops.empty() ? 0.0
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

  // cellSimplices order (residualForLoopsGradient) -> interior param index, so
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
    Eigen::VectorXd grad = (2.0 * beta * (HII.adjoint() * gI)).real();
    if (!stateLoops.empty()) {
      const auto rg = es.residualForLoopsGradient(stateLoops, stateTargets);
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
      const double trial = cost();
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
    std::uint64_t seed, std::shared_ptr<TopologyBuilder> topology, bool verbose)
    : inputStates_(inputStates),
      outputStates_(outputStates),
      beta_(beta),
      epsilon_(epsilon),
      seed_(seed),
      verbose_(verbose),
      topology_(topology ? std::move(topology)
                         : std::make_shared<TorusOperatorTopology>()) {
  if (inputStates_.empty())
    throw std::invalid_argument("MergeCobordism: inputStates is empty");
  stateDim_ = inputStates_.front().size();
  if (stateDim_ < 2 || (stateDim_ & (stateDim_ - 1)) != 0)
    throw std::invalid_argument(
        "MergeCobordism: state dimension must be a power of two >= 2");

  if (!U.empty()) {
    // U-supplied: compute the output state(s) from U, then ignore U.
    if (U.size() != stateDim_ * stateDim_)
      throw std::invalid_argument(
          "MergeCobordism: U must be a d x d row-major operator");
    computeOutputsFromOperator(U);
  }
  if (outputStates_.empty())
    throw std::invalid_argument(
        "MergeCobordism: outputStates is required when U is not supplied");

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
  // action under the state-pinning residual r = beta||grad S||^2 + r_psi. No
  // combinatorial move-search: the relaxed seed triangulation is a local optimum
  // (every boundary-fixed Pachner move only raised the residual in testing), so
  // the relax alone determines the geometry. [#388]
  relaxInterior(cobordism_, beta_, stateLoops_, stateTargets_, /*maxIters=*/400,
                stats_.relaxIterations, verbose_);
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

  // === operator read-out — DEFERRED to #6 (operator-recovery verdict) ===
  // The principled map ker L_1(W) -> U^Choi (the cycle-Choi correspondence) is
  // the #6 work; the clean base deliberately does NOT carry the old first-d^2
  // placeholder. choiState_/operatorU_ stay empty until #6 lands.
  choiState_.clear();
  operatorU_.clear();

  // === residual read-out: beta*||grad_I S||^2 + r_psi at the relaxed metric ===
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
  stats_.stateResidual =
      stateLoops_.empty() ? 0.0 : es.residualForLoops(stateLoops_, stateTargets_);
  stats_.residual = stats_.statActionResidual + stats_.stateResidual;
}

}  // namespace tessera::cobordism
