// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MergeCobordism.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "quantum/ChoiJamiolkowski.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include <iostream>

#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include "spacetime/topologies/SimplexBoundarySphere.h"
#include "spacetime/topologies/SimplicialProduct.h"
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::quantum::ChoiJamiolkowski;
using ::tessera::simulations::ReggeSolver;
using ::tessera::spacetime::AddMove;
using ::tessera::spacetime::PachnerMode;
using ::tessera::spacetime::RemoveMove;
using ::tessera::spacetime::Spacetime;

namespace {

// === (subdivided T^2 with 3 holes) x S^1 = T^3 minus 3 solid tori (#363) ===
// dW = 3 tori (the qubit registers, b_1 = 2 each); interior b_1 = 2g + h - 2 = 3
// for genus g = 1, h = 3 holes (the operator handles). A genus-1 base is required:
// a sphere base would leave only the S^1 interior (b_1 = 1). The torus cycles are
// genuine non-contractible cycles a state residual can read.

// One geodesic (1 -> 4) subdivision of a triangulated surface, so a 3-fold
// vertex-disjoint hole triple exists (the minimal S^1 x S^1 torus has only 2).
std::vector<std::vector<std::uint64_t>> subdivideFaces(
    const std::vector<std::vector<std::uint64_t>> &faces) {
  std::uint64_t nxt = 0;
  for (const auto &f : faces)
    for (const auto v : f) nxt = std::max(nxt, v + 1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> mid;
  auto m = [&](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
    const auto key = std::make_pair(std::min(a, b), std::max(a, b));
    const auto it = mid.find(key);
    if (it != mid.end()) return it->second;
    const std::uint64_t id = nxt++;
    mid[key] = id;
    return id;
  };
  std::vector<std::vector<std::uint64_t>> out;
  for (const auto &f : faces) {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const std::uint64_t ab = m(a, b), bc = m(b, c), ca = m(c, a);
    std::vector<std::vector<std::uint64_t>> tris = {
        {a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}};
    for (auto &s : tris) {
      std::sort(s.begin(), s.end());
      out.push_back(s);
    }
  }
  return out;
}

// The first k pairwise vertex-disjoint faces — the holonomy holes whose
// boundary circles become the qubit tori (each circle x S^1).
std::vector<std::vector<std::uint64_t>> disjointHoles(
    const std::vector<std::vector<std::uint64_t>> &faces, std::size_t k) {
  std::vector<std::vector<std::uint64_t>> holes;
  std::set<std::uint64_t> used;
  for (const auto &f : faces) {
    if (holes.size() >= k) break;
    bool overlap = false;
    for (const auto v : f)
      if (used.count(v)) { overlap = true; break; }
    if (!overlap) {
      holes.push_back(f);
      for (const auto v : f) used.insert(v);
    }
  }
  return holes;
}

// The 3 prism tets of (triangle x edge) from layer offset `bot` to `top` — the
// dimension-generic staircase, looped to make x S^1 rather than x [0,1].
void appendStaircase(const std::vector<std::vector<std::uint64_t>> &faces,
                     std::uint64_t bot, std::uint64_t top,
                     std::set<std::vector<std::uint64_t>> &cells) {
  for (const auto &f : faces) {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const std::uint64_t b0[3] = {a + bot, b + bot, c + bot};
    const std::uint64_t t0[3] = {a + top, b + top, c + top};
    std::vector<std::vector<std::uint64_t>> tets = {
        {b0[0], b0[1], b0[2], t0[2]},
        {b0[0], b0[1], t0[1], t0[2]},
        {b0[0], t0[0], t0[1], t0[2]}};
    for (auto &t : tets) {
      std::sort(t.begin(), t.end());
      cells.insert(t);
    }
  }
}

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

// One bounded Gauss-Newton / Levenberg-Marquardt descent of the stationary-action
// residual beta*||grad S||^2 over the interior edge squared-lengths, using the
// EXACT analytic gradient and Hessian of the dual Regge action (no finite
// differences). Returns the final beta*||grad S||^2. Leaves the interior edges at
// the best point found.
//
// The least-squares residual is f = grad S (over all edges), its Jacobian w.r.t.
// the interior edge lengths is the action Hessian H restricted to the interior
// columns; the GN normal equations (Re(H_I^H H_I) + mu I) delta = -Re(H_I^H g)
// are the analytic-Jacobian Levenberg-Marquardt step the spec calls for.
// Boundary-fixed Pachner moves per relaxation round (#363): a batch jumps
// further through triangulation space before each (expensive) relax, so the
// combinatorial search covers far more updates between relaxations.
constexpr int kMovesPerRound = 8;

double relaxInterior(
    const std::shared_ptr<Spacetime> &st, double beta,
    const std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> &stateLoops,
    const std::vector<std::complex<double>> &stateTargets,
    int maxIters, int &iterCounter, bool verbose = false) {
  // The free parameters: interior edges (both endpoints off dW). dW is held
  // fixed, so the Regge stationarity we relax is over these edges only. The
  // EdgeList order is the order of actionGradientExact / actionHessianExact.
  EigenstateSynthesis es(st, 1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> interiorRank;
  for (const auto &uv : es.interiorEdges()) interiorRank.emplace(uv, 0);
  const auto edges = st->getEdgeList()->toVector();
  std::vector<std::size_t> interiorIdx;   // EdgeList indices of the free edges
  std::vector<::tessera::mesh::Edge *> interiorEdgePtr;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (interiorRank.count(edgeKey(edges[i]))) {
      interiorIdx.push_back(i);
      interiorEdgePtr.push_back(edges[i]);
    }
  }
  const std::size_t nI = interiorIdx.size();

  // r_psi: the carried-harmonic residual of the pinned states over the boundary
  // cycles. Reads the live metric, so it tracks the interior relaxation.
  auto stateCost = [&]() {
    return stateLoops.empty() ? 0.0
                              : es.residualForLoops(stateLoops, stateTargets);
  };
  // The action residual is the Regge stationarity over the FREE (interior) edges
  // ONLY: dW is fixed (Dirichlet), so its action gradient is an irreducible
  // reaction, not part of the interior stationarity — summing it over all edges
  // would floor r at the boundary's |grad S|^2.
  auto actionNorm2 = [&]() {
    ReggeSolver solver(st, MatterConfiguration());
    const auto g = solver.actionGradientExact();
    double n2 = 0.0;
    for (const std::size_t e : interiorIdx) n2 += std::norm(g[e]);
    return n2;
  };

  if (nI == 0) return beta * actionNorm2() + stateCost();  // no free edges to relax

  // cellSimplices order (residualForPeriodsGradient) -> interior param index,
  // so the analytic state-residual gradient folds into the action gradient.
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
    const auto gVec = solver.actionGradientExact();   // |E| complex, EdgeList order
    const auto hMat = solver.actionHessianExact();     // |E|x|E| complex

    // The interior gradient g_I and the interior-interior Hessian block H_II — the
    // residual is the stationarity over the FREE edges only (see actionNorm2).
    Eigen::VectorXcd gI(nI);
    for (std::size_t c = 0; c < nI; ++c) gI(c) = gVec[interiorIdx[c]];
    Eigen::MatrixXcd HII(nI, nI);
    for (std::size_t r = 0; r < nI; ++r)
      for (std::size_t c = 0; c < nI; ++c)
        HII(r, c) = hMat[interiorIdx[r]][interiorIdx[c]];

    // Analytic gradient and GN Hessian of beta*||grad_I S||^2 over the interior
    // lengths, plus the analytic state-residual gradient (cellSimplices order).
    Eigen::VectorXd grad = (2.0 * beta * (HII.adjoint() * gI)).real();
    if (!stateLoops.empty()) {
      const auto rg = es.residualForLoopsGradient(stateLoops, stateTargets);
      for (std::size_t j = 0; j < rg.size() && j < cellToParam.size(); ++j)
        if (cellToParam[j] >= 0) grad(cellToParam[j]) += rg[j];
    }
    Eigen::MatrixXd B = (2.0 * beta * (HII.adjoint() * HII)).real();

    // Current interior lengths (real squared-lengths; spacelike).
    Eigen::VectorXd x0(nI);
    for (std::size_t c = 0; c < nI; ++c)
      x0(c) = interiorEdgePtr[c]->getSquaredLength().real();

    bool improved = false;
    for (int ls = 0; ls < 12; ++ls) {  // mu line search
      Eigen::MatrixXd A = B;
      for (std::size_t c = 0; c < nI; ++c) A(c, c) += mu * (B(c, c) + 1.0);
      const Eigen::VectorXd delta = A.ldlt().solve(-grad);
      Eigen::VectorXd x = x0 + delta;
      // Emergent causal type: l^2 is free to be positive (spacelike), negative
      // (timelike), or pass through zero (null) — no clamp. A degenerate/null
      // step blows the action up, so the line search below rejects it loudly.
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
      std::cerr << "[merge phase2] iter " << it << "/" << maxIters
                << "  r=" << best << "\n";
    if (!improved) {
      // restore best point (x0) and stop — no step lowered the residual
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
    int maxAttempts, std::uint64_t seed, bool verbose)
    : inputStates_(inputStates),
      outputStates_(outputStates),
      beta_(beta),
      epsilon_(epsilon),
      maxAttempts_(maxAttempts),
      seed_(seed),
      verbose_(verbose) {
  if (inputStates_.empty())
    throw std::invalid_argument("MergeCobordism: inputStates is empty");
  stateDim_ = inputStates_.front().size();
  if (stateDim_ < 2 || (stateDim_ & (stateDim_ - 1)) != 0)
    throw std::invalid_argument(
        "MergeCobordism: state dimension must be a power of two >= 2");

  if (!U.empty()) {
    // U-supplied mode: compute the output state(s) from U, then ignore U.
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
  optimize();  // also reads the operator out (the best round's), via extractOperator
}

void MergeCobordism::computeOutputsFromOperator(
    const std::vector<std::complex<double>> &U) {
  // Apply the (row-major d x d) operator to each input: psi_out = U psi_in. The
  // boundary then pins the U-predicted final state and the emergent operator is
  // checked against U.
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
  // dW = 3 tori carry psi_A, psi_B, psi_AB. Each qubit's two components are the
  // periods over its torus's two cycles: the hole-circle (the removed face's
  // boundary) and the S^1 (the time loop of a hole vertex). residualForLoops
  // over ALL six cycles jointly scores how far the bulk is from carrying the
  // states; the six cycles span b_1(W) = 5 (over-determined by the one
  // charge-conservation relation, the register's m = nd + 1).
  stateLoops_.clear();
  stateTargets_.clear();
  if (holes_.size() < 3 || layerStride_ == 0) return;
  std::vector<std::vector<std::complex<double>>> states;
  for (const auto &s : inputStates_) states.push_back(s);
  for (const auto &s : outputStates_) states.push_back(s);
  const std::uint64_t N = layerStride_;
  const std::size_t nStates = std::min(states.size(), holes_.size());
  for (std::size_t i = 0; i < nStates; ++i) {
    std::vector<std::uint64_t> h(holes_[i]);
    std::sort(h.begin(), h.end());
    const std::complex<double> a0 = states[i].size() > 0 ? states[i][0] : std::complex<double>(0);
    const std::complex<double> a1 = states[i].size() > 1 ? states[i][1] : std::complex<double>(0);
    // cycle 1: the hole-circle (removed face boundary) -> psi_i[0]
    stateLoops_.push_back({{h[0], h[1]}, {h[1], h[2]}, {h[2], h[0]}});
    stateTargets_.push_back(a0);
    // cycle 2: the S^1 (vertical loop of hole vertex h[0]) -> psi_i[1]
    stateLoops_.push_back(
        {{h[0], h[0] + N}, {h[0] + N, h[0] + 2 * N}, {h[0] + 2 * N, h[0]}});
    stateTargets_.push_back(a1);
  }
}

void MergeCobordism::buildSeed() {
  // dW = geo(psi_A) u geo(psi_B) u geo(psi_AB): three qubit registers, each a
  // torus (b_1 = 2). The bulk between them carries the operator as the interior
  // handles, ker L_1(W - dW) = d^2 - 1 = 3 for a qubit. Realized as (subdivided
  // T^2 with 3 holes) x S^1 = T^3 minus the three (hole x S^1) solid tori: the
  // hole-circles x S^1 are the boundary tori, the genus + S^1 cycles the interior
  // operator. REGGE (not CDT): edge lengths free, causal type emergent.
  Signature sig2(2, SignatureType::Lorentzian);
  auto metric2 = std::make_shared<Metric>(true, sig2);
  auto s1a = std::make_shared<SimplexBoundarySphere>(1);
  auto s1b = std::make_shared<SimplexBoundarySphere>(1);
  std::shared_ptr<Topology> t2 = std::make_shared<SimplicialProduct>(s1a, s1b);
  auto torus = std::make_shared<Spacetime>(metric2, SpacetimeType::REGGE, 1.0, 1.0,
                                           Foliation::NONE, t2);
  torus->build();

  const auto faces =
      subdivideFaces(ChainComplex::fromSpacetime(*torus).kSimplexVertices(2));
  const int holeCount = static_cast<int>(stateDim_ * stateDim_) - 1;  // 3 for a qubit
  const auto holes = disjointHoles(faces, static_cast<std::size_t>(holeCount));
  if (static_cast<int>(holes.size()) < holeCount)
    throw std::runtime_error(
        "MergeCobordism: torus base lacks enough vertex-disjoint holes");
  const std::set<std::vector<std::uint64_t>> holeSet(holes.begin(), holes.end());

  std::uint64_t N = 0;
  for (const auto &f : faces)
    for (const auto v : f) N = std::max(N, v + 1);
  holes_ = holes;       // the qubit boundary holes, for computeStateTargets
  layerStride_ = N;     // the S^1 layer stride (vertex-id offset per layer)
  std::vector<std::vector<std::uint64_t>> holed;  // torus with the holes removed
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);

  // Staircase the holed torus over S^1 (three layers, looped 0 -> N -> 2N -> 0).
  std::set<std::vector<std::uint64_t>> cellSet;
  for (std::uint64_t layer = 0; layer < 3; ++layer)
    appendStaircase(holed, layer * N, ((layer + 1) % 3) * N, cellSet);

  Signature sig3(3, SignatureType::Lorentzian);
  auto metric3 = std::make_shared<Metric>(true, sig3);
  cobordism_ = std::make_shared<Spacetime>(metric3, SpacetimeType::REGGE, 1.0, 1.0,
                                           Foliation::NONE, std::nullopt);
  std::set<std::uint64_t> verts;
  for (const auto &c : cellSet)
    for (const auto v : c) verts.insert(v);
  std::map<std::uint64_t, ::tessera::mesh::Vertex *> vmap;
  for (const auto id : verts) vmap[id] = cobordism_->createVertex(id);
  for (const auto &c : cellSet) {
    std::vector<::tessera::mesh::Vertex *> vs;
    vs.reserve(c.size());
    for (const auto v : c) vs.push_back(vmap[v]);
    cobordism_->createSimplex(vs);
  }

  // Seed the metric off the degenerate uniform point. At l^2 = 1 a non-null
  // metric-Hodge eigenvalue sits at ~0, where the period-residual gradient's
  // eigenvector perturbation (~ -1/lambda) blows up (|grad| ~ 1e12), so the
  // first relax step cannot descend. A small seeded spread breaks the
  // degeneracy (|grad| ~ 1e-6, FD-validated); the relaxation moves freely from
  // here. l^2 stays positive (spacelike) — the causal type is still emergent.
  std::mt19937 jrng(static_cast<std::uint32_t>(seed_) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> jitter(0.7, 1.3);
  for (auto *e : cobordism_->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(jitter(jrng), 0.0));

  // dW = the single-coface triangles: the three qubit tori (54 faces = 3 x T^2).
  std::map<std::vector<std::uint64_t>, int> cofaceCount;
  for (const auto &c : cellSet)
    for (std::size_t i = 0; i < c.size(); ++i) {
      std::vector<std::uint64_t> tri;
      tri.reserve(c.size() - 1);
      for (std::size_t j = 0; j < c.size(); ++j)
        if (j != i) tri.push_back(c[j]);
      ++cofaceCount[tri];
    }
  boundaryCells_.clear();
  for (const auto &kv : cofaceCount)
    if (kv.second == 1) boundaryCells_.push_back(kv.first);
  stats_.b1Bulk = 0;  // filled after relaxation in extractOperator()
}

void MergeCobordism::optimize() {
  // Joint search (#363). The dual action is stationary over edge lengths AND
  // combinatorics jointly, so we cannot extremize one then the other (a fixed
  // metric makes the move-graph look flat — a combinatorial-first pass accepts
  // zero moves). Each round instead applies a batch of boundary-fixed Pachner
  // moves (a walk over all five bistellar types), relaxes the metric against the
  // FULL residual r = beta||grad S||^2 + r_psi, and scores the relaxed result;
  // we keep the lowest-residual operator found. A move is only as good as the
  // metric it can be relaxed into.
  std::mt19937 rng(static_cast<std::uint32_t>(seed_));

  relaxInterior(cobordism_, beta_, stateLoops_, stateTargets_, /*maxIters=*/150,
                stats_.relaxIterations, verbose_);
  extractOperator();
  double best = stats_.residual;
  std::vector<std::complex<double>> bestU = operatorU_, bestChoi = choiState_;
  Stats bestSnap = stats_;
  if (verbose_)
    std::cerr << "[merge] seed r=" << best << " -- joint search (up to "
              << maxAttempts_ << " rounds)\n";

  for (int attempt = 0; attempt < maxAttempts_ && best >= epsilon_; ++attempt) {
    auto applyMove = [&](auto &mv, int &counter) {
      if (mv.propose() && mv.apply()) ++counter;  // a walk: no per-move rollback
    };
    for (int k = 0; k < kMovesPerRound; ++k) {
      const std::uint64_t s = static_cast<std::uint64_t>(rng());
      switch ((attempt * kMovesPerRound + k) % 5) {
        case 0: { FlipMove mv(cobordism_.get(), s, PachnerMode::PreGeometric, true); applyMove(mv, stats_.flipMoves); break; }
        case 1: { IFlipMove mv(cobordism_.get(), s, PachnerMode::PreGeometric, true); applyMove(mv, stats_.flipMoves); break; }
        case 2: { ShiftMove mv(cobordism_.get(), s, PachnerMode::PreGeometric, true); applyMove(mv, stats_.flipMoves); break; }
        case 3: { AddMove mv(cobordism_.get(), s, true, PachnerMode::PreGeometric, true); applyMove(mv, stats_.addMoves); break; }
        default: { RemoveMove mv(cobordism_.get(), s, PachnerMode::PreGeometric, true); applyMove(mv, stats_.removeMoves); break; }
      }
    }
    stats_.attempts = attempt + 1;
    relaxInterior(cobordism_, beta_, stateLoops_, stateTargets_, /*maxIters=*/150,
                  stats_.relaxIterations, verbose_);
    extractOperator();
    if (verbose_)
      std::cerr << "[merge joint] round " << (attempt + 1) << "/" << maxAttempts_
                << "  r=" << stats_.residual << "  best=" << best
                << "  (flip=" << stats_.flipMoves << " add=" << stats_.addMoves
                << " rm=" << stats_.removeMoves << ")\n";
    if (stats_.residual < best) {
      best = stats_.residual;
      bestU = operatorU_;
      bestChoi = choiState_;
      bestSnap = stats_;
    }
  }

  // Return the lowest-residual operator found; the move counters stay cumulative.
  stats_.converged = best < epsilon_;
  operatorU_ = bestU;
  choiState_ = bestChoi;
  stats_.residual = bestSnap.residual;
  stats_.statActionResidual = bestSnap.statActionResidual;
  stats_.stateResidual = bestSnap.stateResidual;
  stats_.dualAction = bestSnap.dualAction;
  stats_.kerL1Bulk = bestSnap.kerL1Bulk;
  stats_.bettiCobordism = bestSnap.bettiCobordism;
  stats_.b1Bulk = bestSnap.b1Bulk;
  stats_.interiorVertices = bestSnap.interiorVertices;
  stats_.topology = bestSnap.topology;
}

void MergeCobordism::extractOperator() {
  // The discovered operator: U_AB = unvec(ker L_1(W - dW)). Read the carried
  // harmonic of the bulk (interior subcomplex) and un-vectorize it into the
  // d x d operator.
  EigenstateSynthesis es(cobordism_, 1);
  bulkCells_ = es.bulkMinusBoundaryCells();
  const auto H = es.bulkMinusBoundaryHarmonicMatrix();
  const std::size_t ncols = bulkCells_.size();
  const std::size_t dim = ncols ? H.size() / ncols : 0;

  stats_.kerL1Bulk = static_cast<int>(dim);
  const auto bws = betti(*cobordism_);
  stats_.bettiCobordism = bws;
  stats_.b1Bulk = (bws.size() > 1) ? bws[1] : 0;
  // interior vertex count off the synthesizer.
  stats_.interiorVertices = static_cast<int>(es.interiorVertexCount());

  const std::size_t d = stateDim_;
  choiState_.clear();
  operatorU_.clear();
  outputState_.clear();
  const std::size_t registerDim = 0;

  // L_1(W) harmonic read-out. The OPERATOR emerges: it rides on the metric Hodge
  // harmonic of the WHOLE relaxed system that carries ALL the pinned states (so
  // the relaxed geometry enters it). The states (psi_A, psi_B AND psi_AB) all stay
  // pinned; the operator is what we read out. (The interior dim ker L_1(W - dW) is
  // kept as a topology stat above.)
  if (!stateLoops_.empty()) {
    const std::vector<std::complex<double>> psi =
        es.carriedRepresentativeOverLoops(stateLoops_, stateTargets_);
    if (!psi.empty()) {
      // The operator's Choi state: the carried metric harmonic on the first d^2
      // cells (state-dependent via the relaxed metric; a principled cycle-based
      // Choi is the next refinement).
      for (std::size_t j = 0; j < d * d && j < psi.size(); ++j)
        choiState_.push_back(psi[j]);
    }
    // Fidelity diagnostic only (NOT the operator): the emergent ending state from
    // the inputs alone — carry the inputs, read the output cycles — vs the pinned
    // target psi_AB.
    const std::size_t nIn = std::min(inputStates_.size() * 2, stateLoops_.size());
    if (nIn < stateLoops_.size()) {
      const std::vector<EigenstateSynthesis::EdgeLoop> inLoops(
          stateLoops_.begin(), stateLoops_.begin() + static_cast<std::ptrdiff_t>(nIn));
      const std::vector<std::complex<double>> inTargets(
          stateTargets_.begin(), stateTargets_.begin() + static_cast<std::ptrdiff_t>(nIn));
      const std::vector<EigenstateSynthesis::EdgeLoop> outLoops(
          stateLoops_.begin() + static_cast<std::ptrdiff_t>(nIn), stateLoops_.end());
      const std::vector<std::complex<double>> psiIn =
          es.carriedRepresentativeOverLoops(inLoops, inTargets);
      if (!psiIn.empty()) outputState_ = es.periodsOfCochainOverLoops(psiIn, outLoops);
    }
  }

  // Promote the carried Choi state to the operator via the Choi-Jamiolkowski
  // isomorphism (U = sqrt(d) * unvec(state), the inverse of the Choi-state map).
  if (!choiState_.empty())
    operatorU_ = ChoiJamiolkowski::operatorFromChoiState(choiState_,
                                                         static_cast<int>(d));

  // Topology call-out.
  {
    std::string t = "W: b = [";
    for (std::size_t i = 0; i < bws.size(); ++i)
      t += std::to_string(bws[i]) + (i + 1 < bws.size() ? "," : "");
    t += "]; ker L1(W-dW) = " + std::to_string(dim) +
         "; register ker L1(W) over holes = " + std::to_string(registerDim);
    stats_.topology = t;
  }

  // Residual read-out, so the combinatorial search can score each triangulation:
  // the dual Sorkin action stationarity beta||grad S||^2 plus the joint state
  // residual r_psi, both at the current metric.
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
