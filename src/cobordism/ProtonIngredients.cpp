// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ProtonIngredients.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "cobordism/MultiCobordism.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {
constexpr int kDim = 4;  // framework dimension; the seed is a single Δ⁴ simplex
// How many settle-and-verify evolution+relaxation passes an attempt may take before it
// is judged non-persistent: the LAST pass must leave holes, b_k, and F stable, so early
// passes are allowed to finish settling a not-quite-stationary complex instead of
// throwing the attempt away.
constexpr int kMaxPersistencePasses = 3;
}  // namespace

ProtonIngredients::ProtonIngredients(std::uint64_t seed, int registerDegree,
                                     double gamma, double inputWeight, int precone,
                                     bool shouldUseDirectedSurgery)
    : proton_(seed, registerDegree, gamma, inputWeight, precone,
              shouldUseDirectedSurgery),
      baseSeed_(seed),
      registerDegree_(registerDegree),
      gamma_(gamma),
      inputResidualWeight_(inputWeight),
      precone_(precone),
      shouldUseDirectedSurgery_(shouldUseDirectedSurgery),
      // The joint-arm per-block reads stay NaN unless buildJoint() runs — the
      // two-step's step B has no localized output blocks to read.
      baryonResidual_(std::numeric_limits<double>::quiet_NaN()),
      antibaryonResidual_(std::numeric_limits<double>::quiet_NaN()) {}

std::shared_ptr<Spacetime> ProtonIngredients::buildMinimalSeed() {
  using namespace ::tessera::spacetime;
  // Mirrors Proton::buildMinimalSeed (private there): one Δ⁴ pentatope, uniform
  // ℓ² = +1. The metric is deliberately all-spacelike — at initialization no time has
  // passed, so no causal structure is put in by hand; any causal content must emerge.
  auto metric =
      std::make_shared<Metric>(true, Signature(kDim, SignatureType::Lorentzian));
  std::shared_ptr<Topology> topology = std::make_shared<SolidSimplex>(kDim);
  auto host = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0,
                                          Foliation::PREFERRED, topology);
  host->build();
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setSquaredLength(complexd(1.0, 0.0));
  return host;
}

std::shared_ptr<MultiCobordism> ProtonIngredients::recombinationNode(
    std::uint64_t seed) const {
  // Step A is the canonical arm's node, verbatim — the composed Proton is the single
  // source of truth for the recombination setup.
  return proton_.recombinationNode(seed);
}

std::shared_ptr<MultiCobordism> ProtonIngredients::formationNode(
    std::uint64_t seed) const {
  // Step B with nothing pinned: the same seed complex and the same ideal diquark
  // {1,ω} + third quark {ω²} inputs as Proton::formationNode — but the output-target
  // list is EMPTY, so the objective's matter term is the inputs' residuals alone and
  // the whole's final state emerges. (MultiCobordism supports the empty list: with no
  // output targets, rU sums only the input blocks.)
  const complexd w = Proton::omega();
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> thirdQuark = {w * w};
  auto host = buildMinimalSeed();
  // Capture the seed vertex IDS before constructing the node (see
  // Proton::recombinationNode): precone_ > 0 regrows the complex in the ctor, but the
  // seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, std::vector<std::vector<complexd>>{diquark, thirdQuark},
      std::vector<std::vector<complexd>>{},  // nothing pinned — the final state emerges
      std::vector<int>{registerDegree_}, gamma_, seed, precone_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  return node;
}

std::shared_ptr<MultiCobordism> ProtonIngredients::jointNode(
    std::uint64_t seed) const {
  // The JOINT node (#560, rung 1): the two-step event graph collapsed into one
  // co-optimized node. Inputs: the Z₃ ORBIT of the neutral q-q̄ pair — {1,-1,0} and its
  // two cyclic rotations (the #398 symmetric-input lesson) — at v0,v1,v2. Outputs: TWO
  // localized blocks (the multi-output rU branch, exactly as the 2→2 recombination) —
  // the baryon [1,ω,ω²] at v3 ⊔ its component-wise conjugate, the antibaryon
  // [1,ω̄,ω̄²], at v4. The conjugate is a component-permutation of the singlet
  // (ω̄ = ω²) and the block residual is relabeling-invariant, so the conjugation is
  // carried by the block's location in the emergent complex, not the residual's value
  // (see the header note).
  const std::vector<std::vector<complexd>> pairTriple = {
      {complexd(1.0, 0.0), complexd(-1.0, 0.0), complexd(0.0, 0.0)},
      {complexd(0.0, 0.0), complexd(1.0, 0.0), complexd(-1.0, 0.0)},
      {complexd(-1.0, 0.0), complexd(0.0, 0.0), complexd(1.0, 0.0)}};
  const std::vector<complexd> baryon = Proton::singlet();  // {1, ω, ω²}
  std::vector<complexd> antibaryon;
  antibaryon.reserve(baryon.size());
  for (const auto &component : baryon) antibaryon.push_back(std::conj(component));
  auto host = buildMinimalSeed();
  // Capture the seed vertex IDS before constructing the node (see
  // Proton::recombinationNode): precone_ > 0 regrows the complex in the ctor, but the
  // seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, pairTriple, std::vector<std::vector<complexd>>{baryon, antibaryon},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1], seedVertexIds[2]});
  node->seedOutputs({seedVertexIds[3], seedVertexIds[4]});
  return node;
}

void ProtonIngredients::driveNode(MultiCobordism &node, int initSteps,
                                  int evolveSteps, int stage1CandidateMoves,
                                  int stage1Patience, double stage2Beta,
                                  int stage2MaxIters) const {
  // Proton::build()'s exact drive per node: INITIALIZATION pass (grow_boundaries=true),
  // optional directed cone-out, EVOLUTION pass (∂W frozen), optional directed cone-in,
  // then the geometric relaxation. Shared verbatim by build() and buildJoint() — the
  // joint arm changes the event graph, never the drive.
  node.runStage1(initSteps, stage1CandidateMoves, stage1Patience,
                 /*growBoundaries=*/true);
  if (shouldUseDirectedSurgery_)
    (void)node.directedConeOut();
  node.runStage1(evolveSteps, stage1CandidateMoves, stage1Patience,
                 /*growBoundaries=*/false);
  if (shouldUseDirectedSurgery_)
    (void)node.directedConeIn();
  node.runStage2(stage2Beta, stage2MaxIters);
}

int ProtonIngredients::emergentHoleCount(
    const std::shared_ptr<Spacetime> &whole) const {
  return static_cast<int>(
      MultiCobordism::emergentHoles(*whole, registerDegree_).size());
}

int ProtonIngredients::bettiAtRegisterDegree(
    const std::shared_ptr<Spacetime> &whole) const {
  const auto betti = MultiCobordism::betti(*whole);
  return registerDegree_ < static_cast<int>(betti.size()) ? betti[registerDegree_]
                                                          : 0;
}

bool ProtonIngredients::settleAndVerifyPersistence(
    MultiCobordism &node, int evolveSteps, int stage1CandidateMoves,
    int stage1Patience, double stage2Beta, int stage2MaxIters,
    double persistRelTol) const {
  // Persistence: continued evolution (∂W frozen) + relaxation must leave the
  // answer-agnostic summary — the emergent hole count, b_k, and the objective;
  // deliberately NOT the singlet residual, since no answer-shaped quantity may steer
  // or gate these builds — stable. Up to kMaxPersistencePasses passes may settle a
  // not-quite-stationary complex; the LAST pass has to be the stable one.
  bool persisted = false;
  for (int pass = 0; pass < kMaxPersistencePasses && !persisted; ++pass) {
    const auto whole = node.spacetime();
    const int holesBefore = emergentHoleCount(whole);
    const int bettiBefore = bettiAtRegisterDegree(whole);
    const double objectiveBefore = node.objective();
    node.runStage1(evolveSteps, stage1CandidateMoves, stage1Patience,
                   /*growBoundaries=*/false);
    node.runStage2(stage2Beta, stage2MaxIters);
    const auto settled = node.spacetime();
    const double objectiveAfter = node.objective();
    persisted = emergentHoleCount(settled) == holesBefore &&
                bettiAtRegisterDegree(settled) == bettiBefore &&
                std::abs(objectiveAfter - objectiveBefore) <=
                    persistRelTol * std::max(std::abs(objectiveBefore), 1.0);
  }
  return persisted;
}

void ProtonIngredients::build(int maxRestarts, int initSteps, int evolveSteps,
                              int stage1CandidateMoves, int stage1Patience,
                              double stage2Beta, int stage2MaxIters,
                              double persistRelTol) {
  if (attempted_) return;
  attempted_ = true;

  double bestObjective = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination: best-effort; its r_U is reported, not gated. ----
    auto stepA = recombinationNode(seedA);
    driveNode(*stepA, initSteps, evolveSteps, stage1CandidateMoves, stage1Patience,
              stage2Beta, stage2MaxIters);
    const double diquarkR = stepA->rU(stepA->spacetime());

    // ---- Step B — formation with nothing pinned ----
    auto stepB = formationNode(seedB);
    driveNode(*stepB, initSteps, evolveSteps, stage1CandidateMoves, stage1Patience,
              stage2Beta, stage2MaxIters);

    const bool persisted = settleAndVerifyPersistence(
        *stepB, evolveSteps, stage1CandidateMoves, stage1Patience, stage2Beta,
        stage2MaxIters, persistRelTol);
    const bool isStationary = stepB->lastStage2Stationary();
    const bool ok = isStationary && persisted;

    const auto whole = stepB->spacetime();
    const double finalObjective = stepB->objective();

    // Keep the converged attempt, or the lowest-objective one so far otherwise — the
    // objective itself is the only ranking, never a target state.
    if (ok || finalObjective < bestObjective) {
      bestObjective = finalObjective;
      converged_ = ok;
      stationary_ = isStationary;
      persistent_ = persisted;
      keptSeed_ = seedA;
      spacetime_ = whole;
      emergentHoles_ = MultiCobordism::emergentHoles(*whole, registerDegree_);
      singletResidual_ = MultiCobordism::residualOfTargetStateAgainstHarmonic(
          whole, registerDegree_, Proton::singlet());  // diagnostic read, after the fact
      inputResidual_ = stepB->rU(whole);
      finalObjective_ = finalObjective;
      diquarkResidual_ = diquarkR;
    }
    if (ok) return;  // a stationary, persistent structure emerged — stop restarting
  }
}

double ProtonIngredients::outputBlockResidual(const MultiCobordism &node,
                                              std::size_t blockIndex) const {
  // The after-the-fact per-output-block read (#560): the block's own sub-complex —
  // the ambient complex's top cells all of whose vertices lie in the block's region —
  // scored against the block's target. Mirrors MultiCobordism's private
  // residualForBoundaryBlock at this class's single register degree (uniform-metric
  // sub-complex via fromCells), so the read matches how the drive's rU scored the
  // block; with no full cell inside the region, the full leak ‖target‖².
  const auto &block = node.outputs().at(blockIndex);
  std::vector<std::vector<std::uint64_t>> cellsInsideRegion;
  for (const auto &topSimplex : node.spacetime()->getTopSimplices()) {
    std::vector<std::uint64_t> cellVertexIds;
    bool cellIsInsideRegion = true;
    for (const auto *vertex : topSimplex->getVertices()) {
      if (!block.vertices.count(vertex->getId())) {
        cellIsInsideRegion = false;
        break;
      }
      cellVertexIds.push_back(vertex->getId());
    }
    if (cellIsInsideRegion) cellsInsideRegion.push_back(std::move(cellVertexIds));
  }
  if (cellsInsideRegion.empty()) {
    double targetNormSquared = 0.0;
    for (const auto &component : block.target)
      targetNormSquared += std::norm(component);
    return targetNormSquared;  // the full leak — nothing carries the block yet
  }
  const auto blockSubcomplex = Spacetime::fromCells(kDim, cellsInsideRegion, 1.0, 0.0);
  return MultiCobordism::residualOfTargetStateAgainstHarmonic(
      blockSubcomplex, registerDegree_, block.target);
}

void ProtonIngredients::buildJoint(int maxRestarts, int initSteps, int evolveSteps,
                                   int stage1CandidateMoves, int stage1Patience,
                                   double stage2Beta, int stage2MaxIters,
                                   double persistRelTol) {
  if (attempted_) return;
  attempted_ = true;

  // The joint arm has no step A — the diquark intermediate is collapsed away — so its
  // observable is explicitly not-a-number, never a stale zero.
  diquarkResidual_ = std::numeric_limits<double>::quiet_NaN();

  double bestObjective = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    // ONE node per attempt (the collapsed event graph is the experiment), so the
    // restart enumeration is dense: seed + attempt.
    const std::uint64_t seed = baseSeed_ + static_cast<std::uint64_t>(attempt);

    auto node = jointNode(seed);
    driveNode(*node, initSteps, evolveSteps, stage1CandidateMoves, stage1Patience,
              stage2Beta, stage2MaxIters);

    const bool persisted = settleAndVerifyPersistence(
        *node, evolveSteps, stage1CandidateMoves, stage1Patience, stage2Beta,
        stage2MaxIters, persistRelTol);
    const bool isStationary = node->lastStage2Stationary();
    const bool ok = isStationary && persisted;

    const auto whole = node->spacetime();
    const double finalObjective = node->objective();

    // Keep the converged attempt, or the lowest-objective one so far otherwise — the
    // objective itself is the only ranking, never a target state.
    if (ok || finalObjective < bestObjective) {
      bestObjective = finalObjective;
      converged_ = ok;
      stationary_ = isStationary;
      persistent_ = persisted;
      keptSeed_ = seed;
      spacetime_ = whole;
      emergentHoles_ = MultiCobordism::emergentHoles(*whole, registerDegree_);
      singletResidual_ = MultiCobordism::residualOfTargetStateAgainstHarmonic(
          whole, registerDegree_, Proton::singlet());  // diagnostic read, after the fact
      inputResidual_ = node->rU(whole);  // full matter term: inputs + output blocks
      finalObjective_ = finalObjective;
      // The per-output-block singlet residuals, read after the fact off each block's
      // own emergent sub-complex: baryon first, antibaryon second (jointNode's
      // seeding order).
      baryonResidual_ = outputBlockResidual(*node, 0);
      antibaryonResidual_ = outputBlockResidual(*node, 1);
      // Keep the blocks themselves as the after-the-fact provenance behind those two
      // residuals (region + target), so external readers can re-score them exactly.
      outputBlocks_ = node->outputs();
    }
    if (ok) return;  // a stationary, persistent structure emerged — stop restarting
  }
}

void ProtonIngredients::ensureBuilt() {
  if (!attempted_) build();
}

bool ProtonIngredients::converged() {
  ensureBuilt();
  return converged_;
}

bool ProtonIngredients::stationary() {
  ensureBuilt();
  return stationary_;
}

bool ProtonIngredients::persistent() {
  ensureBuilt();
  return persistent_;
}

std::uint64_t ProtonIngredients::seed() {
  ensureBuilt();
  return keptSeed_;
}

std::shared_ptr<Spacetime> ProtonIngredients::spacetime() {
  ensureBuilt();
  return spacetime_;
}

std::shared_ptr<Spacetime> ProtonIngredients::block() {
  ensureBuilt();
  return spacetime_;  // the emergent object IS the whole (parity with Proton::block)
}

std::vector<std::vector<std::uint64_t>> ProtonIngredients::emergentHoles() {
  ensureBuilt();
  return emergentHoles_;
}

double ProtonIngredients::singletResidual() {
  ensureBuilt();
  return singletResidual_;
}

double ProtonIngredients::inputResidual() {
  ensureBuilt();
  return inputResidual_;
}

double ProtonIngredients::finalObjective() {
  ensureBuilt();
  return finalObjective_;
}

double ProtonIngredients::diquarkResidual() {
  ensureBuilt();
  return diquarkResidual_;
}

double ProtonIngredients::baryonResidual() {
  ensureBuilt();
  return baryonResidual_;  // NaN unless buildJoint() ran (see the header note)
}

double ProtonIngredients::antibaryonResidual() {
  ensureBuilt();
  return antibaryonResidual_;  // NaN unless buildJoint() ran (see the header note)
}

std::vector<MultiCobordism::BoundaryBlock> ProtonIngredients::outputBlocks() {
  ensureBuilt();
  return outputBlocks_;  // empty unless buildJoint() ran (see the header note)
}

}  // namespace tessera::cobordism
