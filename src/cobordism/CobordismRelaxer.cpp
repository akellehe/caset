// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CobordismRelaxer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "cobordism/ChainComplex.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;

std::pair<std::uint64_t, std::uint64_t> CobordismRelaxer::edgeKey(
    const ::tessera::mesh::Edge *e) {
  const auto a = e->getSource()->getId();
  const auto b = e->getTarget()->getId();
  return {std::min(a, b), std::max(a, b)};
}

std::vector<int> CobordismRelaxer::betti(const Spacetime &st) {
  return ChainComplex::fromSpacetime(st).bettiNumbers();
}

double CobordismRelaxer::relaxInterior(
    const std::shared_ptr<Spacetime> &st, double beta,
    const std::vector<EigenstateSynthesis::EdgeLoop> &stateLoops,
    const std::vector<std::complex<double>> &stateTargets,
    const std::vector<std::vector<std::uint64_t>> &stateHoles,
    const std::vector<std::complex<double>> &holeTargets, int maxIters,
    int &iterCounter, bool periodPin, double stateEpsilon, bool verbose) {
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
  // EXACT triangle-hole path (register/junction) vs SOFT edge-loop path (operator).
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
      std::cerr << "[cobordism relax] iter " << it << "/" << maxIters
                << "  r=" << best << "\n";
    if (!improved) {  // restore best point and stop
      for (std::size_t c = 0; c < nI; ++c)
        interiorEdgePtr[c]->setSquaredLength(std::complex<double>(x0(c), 0.0));
      break;
    }
  }
  return best;
}

}  // namespace tessera::cobordism
