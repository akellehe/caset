// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/SurgicalCone.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using complexd = std::complex<double>;

namespace {
constexpr int kFrameworkDimension = 4;  // framework dimension (closed S^4 host)

// Sorted vertex-id tuple of a top simplex (the reference's `_top_tuple`).
std::vector<std::uint64_t> topTuple(const ::tessera::mesh::Simplex &simplex) {
  std::vector<std::uint64_t> sortedVertexIdentifiers;
  for (const auto *vertex : simplex.getVertices())
    sortedVertexIdentifiers.push_back(vertex->getId());
  std::sort(sortedVertexIdentifiers.begin(), sortedVertexIdentifiers.end());
  return sortedVertexIdentifiers;
}

std::pair<std::uint64_t, std::uint64_t> edgeKey(
    const ::tessera::mesh::Edge *edge) {
  const auto sourceVertexId = edge->getSource()->getId();
  const auto targetVertexId = edge->getTarget()->getId();
  return {std::min(sourceVertexId, targetVertexId),
          std::max(sourceVertexId, targetVertexId)};
}
}  // namespace

MultiCobordism::MultiCobordism(
    std::shared_ptr<Spacetime> host,
    const std::vector<std::vector<complexd>> &inputTargets,
    const std::vector<std::vector<complexd>> &outputTargets,
    const std::vector<int> &degrees, double gamma, std::uint64_t seed)
    : spacetime_(std::move(host)),
      inputTargets_(inputTargets),
      outputTargets_(outputTargets),
      registerDegrees_(degrees),
      dualComplexGateDegree_(
          registerDegrees_.empty()
              ? 0
              : *std::max_element(registerDegrees_.begin(),
                                  registerDegrees_.end())),
      gamma_(gamma),
      randomNumberGenerator_(seed) {}

std::vector<int> MultiCobordism::betti(const Spacetime &spacetime) {
  return ChainComplex::fromSpacetime(spacetime).bettiNumbers();
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::emergentHoles(
    const Spacetime &spacetime, int registerDegree) {
  // The (k+2)-vertex tuples all of whose drop-one facets are boundary facets.
  std::set<std::vector<std::uint64_t>> boundaryFacets;
  for (auto boundaryFacet : spacetime.getBoundary()) {
    std::sort(boundaryFacet.begin(), boundaryFacet.end());
    boundaryFacets.insert(std::move(boundaryFacet));
  }
  std::vector<std::vector<std::uint64_t>> emergentHoleTuples;
  if (boundaryFacets.empty() ||
      static_cast<int>(boundaryFacets.begin()->size()) !=
          registerDegree + 1)  // facets must be k-cells
    return emergentHoleTuples;
  std::set<std::uint64_t> boundaryVertexIds;
  for (const auto &boundaryFacet : boundaryFacets)
    for (auto vertexId : boundaryFacet) boundaryVertexIds.insert(vertexId);
  std::set<std::vector<std::uint64_t>> emergentHoleSet;
  for (const auto &boundaryFacet : boundaryFacets) {
    for (auto candidateVertexId : boundaryVertexIds) {
      if (std::find(boundaryFacet.begin(), boundaryFacet.end(),
                    candidateVertexId) != boundaryFacet.end())
        continue;
      std::vector<std::uint64_t> candidateHole = boundaryFacet;
      candidateHole.push_back(candidateVertexId);
      std::sort(candidateHole.begin(), candidateHole.end());
      bool allFacetsAreBoundary = true;
      for (std::size_t droppedIndex = 0; droppedIndex < candidateHole.size();
           ++droppedIndex) {
        std::vector<std::uint64_t> dropOneFacet;
        for (std::size_t copyIndex = 0; copyIndex < candidateHole.size();
             ++copyIndex)
          if (copyIndex != droppedIndex)
            dropOneFacet.push_back(candidateHole[copyIndex]);
        if (!boundaryFacets.count(dropOneFacet)) {
          allFacetsAreBoundary = false;
          break;
        }
      }
      if (allFacetsAreBoundary) emergentHoleSet.insert(candidateHole);
    }
  }
  emergentHoleTuples.assign(emergentHoleSet.begin(), emergentHoleSet.end());
  return emergentHoleTuples;
}

double MultiCobordism::reggeActionGradient(
    const std::shared_ptr<Spacetime> &spacetime) {
  ReggeSolver reggeSolver(spacetime, MatterConfiguration());
  double squaredGradientNorm = 0.0;
  for (const auto &gradientComponent : reggeSolver.actionGradientExact())
    squaredGradientNorm += std::norm(gradientComponent);
  return squaredGradientNorm;
}

double MultiCobordism::residualOfTargetStateAgainstHarmonic(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState) {
  const std::size_t targetDimension = targetState.size();
  Eigen::VectorXcd targetVector(targetDimension);
  for (std::size_t componentIndex = 0; componentIndex < targetDimension;
       ++componentIndex)
    targetVector(componentIndex) = targetState[componentIndex];
  const double fullLeakResidual = targetVector.squaredNorm();  // zero-filled leak

  const auto bettiNumbers = betti(*spacetime);
  if (registerDegree < 0 ||
      registerDegree >= static_cast<int>(bettiNumbers.size()))
    return fullLeakResidual;
  const int degreeBettiNumber = bettiNumbers[registerDegree];
  if (degreeBettiNumber == 0) return fullLeakResidual;
  auto emergentHoleTuples = emergentHoles(*spacetime, registerDegree);
  if (emergentHoleTuples.empty()) return fullLeakResidual;
  if (emergentHoleTuples.size() > targetDimension)
    emergentHoleTuples.resize(targetDimension);
  const std::size_t usableHoleCount = emergentHoleTuples.size();

  EigenstateSynthesis eigenstateSynthesis(spacetime, registerDegree);
  const auto flattenedCyclePeriods =
      eigenstateSynthesis.cyclePeriods(emergentHoleTuples);  // bk x m, row-major
  // periodMatrixTransposed: (d, bk), zero-filled beyond the usable hole columns.
  Eigen::MatrixXcd periodMatrixTransposed = Eigen::MatrixXcd::Zero(
      static_cast<int>(targetDimension), degreeBettiNumber);
  for (int bettiRowIndex = 0; bettiRowIndex < degreeBettiNumber; ++bettiRowIndex)
    for (std::size_t holeColumnIndex = 0; holeColumnIndex < usableHoleCount;
         ++holeColumnIndex)
      periodMatrixTransposed(static_cast<int>(holeColumnIndex), bettiRowIndex) =
          flattenedCyclePeriods[static_cast<std::size_t>(bettiRowIndex) *
                                    usableHoleCount +
                                holeColumnIndex];

  // min over permutations of the target components of ||pdT c - ts||^2 (lstsq c).
  std::vector<int> targetPermutation(targetDimension);
  std::iota(targetPermutation.begin(), targetPermutation.end(), 0);
  double bestResidual = std::numeric_limits<double>::infinity();
  Eigen::BDCSVD<Eigen::MatrixXcd> periodSvd(
      periodMatrixTransposed, Eigen::ComputeThinU | Eigen::ComputeThinV);
  do {
    Eigen::VectorXcd permutedTargetVector(targetDimension);
    for (std::size_t componentIndex = 0; componentIndex < targetDimension;
         ++componentIndex)
      permutedTargetVector(componentIndex) =
          targetVector(targetPermutation[componentIndex]);
    const Eigen::VectorXcd leastSquaresCoefficients =
        periodSvd.solve(permutedTargetVector);
    bestResidual = std::min(bestResidual,
                            (periodMatrixTransposed * leastSquaresCoefficients -
                             permutedTargetVector)
                                .squaredNorm());
  } while (std::next_permutation(targetPermutation.begin(),
                                 targetPermutation.end()));
  return bestResidual;
}

std::shared_ptr<Spacetime> MultiCobordism::subcomplexWithinVertexSet(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::set<std::uint64_t> &vertexSet) const {
  std::vector<std::vector<std::uint64_t>> cellsInsideVertexSet;
  for (const auto &topSimplex : spacetime->getTopSimplices()) {
    auto cellVertexIds = topTuple(*topSimplex);
    bool cellIsInsideVertexSet = true;
    for (auto vertexId : cellVertexIds)
      if (!vertexSet.count(vertexId)) {
        cellIsInsideVertexSet = false;
        break;
      }
    if (cellIsInsideVertexSet)
      cellsInsideVertexSet.push_back(std::move(cellVertexIds));
  }
  if (cellsInsideVertexSet.empty()) return nullptr;
  return Spacetime::fromCells(kFrameworkDimension, cellsInsideVertexSet, 1.0,
                              0.0);
}

double MultiCobordism::residualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime) const {
  auto blockSubcomplex =
      subcomplexWithinVertexSet(spacetime, boundaryBlock.vertices);
  double residual = 0.0;
  if (!blockSubcomplex) {
    Eigen::VectorXcd targetVector(boundaryBlock.target.size());
    for (std::size_t componentIndex = 0;
         componentIndex < boundaryBlock.target.size(); ++componentIndex)
      targetVector(componentIndex) = boundaryBlock.target[componentIndex];
    return static_cast<double>(registerDegrees_.size()) *
           targetVector.squaredNorm();
  }
  for (int registerDegree : registerDegrees_)
    residual += residualOfTargetStateAgainstHarmonic(
        blockSubcomplex, registerDegree, boundaryBlock.target);
  return residual;
}

double MultiCobordism::rU(const std::shared_ptr<Spacetime> &spacetime) const {
  // The cobordism residual. INPUTS are localized boundary sub-complexes (built near
  // a seed, held representable by these terms, not pinned) — each read off its own
  // region and weighted by inputResidualWeight_ so they are not out-competed by the
  // whole/output term.
  double totalResidual = 0.0;
  for (const auto &inputBlock : inputBlocks_)
    totalResidual +=
        inputResidualWeight_ * residualForBoundaryBlock(inputBlock, spacetime);
  if (outputTargets_.size() == 1) {
    // A SINGLE output is the whole cobordism's output boundary: as in the Python
    // reference it is "the harmonic of the entire structure", NEVER a pinned
    // region. Read it off the WHOLE complex so the bulk loop drives the whole to
    // carry it (the output EMERGES; it is not frozen by construct_outputs).
    for (int registerDegree : registerDegrees_)
      totalResidual += residualOfTargetStateAgainstHarmonic(
          spacetime, registerDegree, outputTargets_.front());
  } else {
    // Multiple outputs (e.g. a 2->2 recombination → diquark ⊔ antidiquark) live in
    // distinct regions: read each off its own constructed block.
    for (const auto &outputBlock : outputBlocks_)
      totalResidual += residualForBoundaryBlock(outputBlock, spacetime);
    if (inputBlocks_.empty() && outputBlocks_.empty())  // bare objective, nothing built yet
      for (int registerDegree : registerDegrees_)
        for (const auto &outputTarget : outputTargets_)
          totalResidual += residualOfTargetStateAgainstHarmonic(
              spacetime, registerDegree, outputTarget);
  }
  return totalResidual;
}

double MultiCobordism::objective() const {
  return reggeActionGradient(spacetime_) + gamma_ * rU(spacetime_);
}

std::set<std::uint64_t> MultiCobordism::boundaryVerts() const {
  // NOTHING is pinned. The boundary states are held representable by their r_U
  // terms — each input by its OWN residual r(input_i), and the single output by
  // the WHOLE cobordism's residual r(whole) (with the inputs as its boundary) —
  // NOT by freezing vertices. The particular input structures are free to change;
  // they must each merely continue to minimize their residual (keep representing
  // their state). With the Regge and residual terms on the same order (Gamma), the
  // optimizer cannot trade a boundary state away just to smooth the geometry.
  return {};
}

MultiCobordism::Snapshot MultiCobordism::snapshotOf(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> cellVertexTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    cellVertexTuples.push_back(topTuple(*topSimplex));
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> squaredLengthsByEdge;
  for (const auto *edge : spacetime.getEdgeList()->toVector())
    squaredLengthsByEdge[edgeKey(edge)] = edge->getSquaredLength();
  return {std::move(cellVertexTuples), std::move(squaredLengthsByEdge)};
}

MultiCobordism::Snapshot MultiCobordism::snapshot() const {
  return snapshotOf(*spacetime_);
}

std::shared_ptr<Spacetime> MultiCobordism::build(
    const Snapshot &complexSnapshot) const {
  auto rebuiltSpacetime = Spacetime::fromCells(kFrameworkDimension,
                                               complexSnapshot.first, 1.0, 0.0);
  for (auto *edge : rebuiltSpacetime->getEdgeList()->toVector()) {
    const auto savedEntry = complexSnapshot.second.find(edgeKey(edge));
    if (savedEntry != complexSnapshot.second.end())
      edge->setSquaredLength(savedEntry->second);
  }
  return rebuiltSpacetime;
}

MultiCobordism::MoveSpec MultiCobordism::drawRandomMoveSpecification(
    const Spacetime &spacetime) {
  static const char *moveKinds[] = {"add",   "remove",   "flip",
                                    "iflip", "cone_out", "cone_in"};
  const std::string moveKind = moveKinds[randomNumberGenerator_() % 6];
  if (moveKind == "add" || moveKind == "remove" || moveKind == "flip" ||
      moveKind == "iflip")
    return {moveKind,
            {static_cast<std::uint64_t>(randomNumberGenerator_() % (1u << 31))}};
  std::vector<std::vector<std::uint64_t>> topCellTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    topCellTuples.push_back(topTuple(*topSimplex));
  if (topCellTuples.empty()) return {"noop", {}};
  const auto &chosenCell =
      topCellTuples[randomNumberGenerator_() % topCellTuples.size()];
  if (moveKind == "cone_out") return {"cone_out", chosenCell};
  const std::size_t droppedVertexIndex =
      randomNumberGenerator_() % chosenCell.size();
  std::vector<std::uint64_t> coneInFace;
  for (std::size_t vertexIndex = 0; vertexIndex < chosenCell.size();
       ++vertexIndex)
    if (vertexIndex != droppedVertexIndex)
      coneInFace.push_back(chosenCell[vertexIndex]);
  return {"cone_in", coneInFace};
}

bool MultiCobordism::applyMoveSpecification(
    const std::shared_ptr<Spacetime> &spacetime,
    const MoveSpec &moveSpecification) {
  const auto &moveKind = moveSpecification.first;
  if (moveKind == "noop") return false;
  bool moveWasApplied = false;
  if (moveKind == "add" || moveKind == "remove" || moveKind == "flip" ||
      moveKind == "iflip") {
    std::mt19937 moveRandomEngine(
        static_cast<std::uint32_t>(moveSpecification.second[0]));
    using ::tessera::spacetime::PachnerMode;
    if (moveKind == "add") {
      ::tessera::spacetime::AddMove pachnerMove(
          spacetime.get(), &moveRandomEngine, false, PachnerMode::PreGeometric,
          false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == "remove") {
      ::tessera::spacetime::RemoveMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == "flip") {
      ::tessera::spacetime::FlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else {
      ::tessera::spacetime::IFlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    }
  } else if (moveKind == "cone_out") {
    moveWasApplied =
        SurgicalCone(spacetime.get()).coneOut(moveSpecification.second).first;
  } else {
    moveWasApplied =
        SurgicalCone(spacetime.get()).coneIn(moveSpecification.second).first;
  }
  if (!moveWasApplied) return false;
  std::set<std::uint64_t> liveVertexIds;
  for (const auto &topSimplex : spacetime->getTopSimplices())
    for (auto vertexId : topTuple(*topSimplex)) liveVertexIds.insert(vertexId);
  for (auto vertexId : boundaryVerts())
    if (!liveVertexIds.count(vertexId))
      return false;  // a pinned boundary vertex was removed
  return EigenstateSynthesis(spacetime, dualComplexGateDegree_)
      .dualComplexValid()
      .first;
}

double MultiCobordism::deltaF(
    const std::shared_ptr<Spacetime> &candidateSpacetime, double baseResidualU,
    const std::set<std::vector<std::uint64_t>> &baseCellSet) const {
  std::set<std::vector<std::uint64_t>> candidateCellSet;
  for (const auto &topSimplex : candidateSpacetime->getTopSimplices())
    candidateCellSet.insert(topTuple(*topSimplex));
  std::vector<std::vector<std::uint64_t>> touchedCells;
  for (const auto &cell : baseCellSet)
    if (!candidateCellSet.count(cell)) touchedCells.push_back(cell);
  for (const auto &cell : candidateCellSet)
    if (!baseCellSet.count(cell)) touchedCells.push_back(cell);
  ReggeSolver baseReggeSolver(spacetime_, MatterConfiguration());
  ReggeSolver candidateReggeSolver(candidateSpacetime, MatterConfiguration());
  std::set<std::pair<std::uint64_t, std::uint64_t>> affectedEdgeSet;
  for (const auto &edgeEndpoints :
       baseReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  for (const auto &edgeEndpoints :
       candidateReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> affectedEdges(
      affectedEdgeSet.begin(), affectedEdgeSet.end());
  const double gradientDelta =
      candidateReggeSolver.gradientNorm2OverEdges(affectedEdges) -
      baseReggeSolver.gradientNorm2OverEdges(affectedEdges);
  const double residualUDelta = rU(candidateSpacetime) - baseResidualU;
  return gradientDelta + gamma_ * residualUDelta;
}

double MultiCobordism::step(int nCandidateMoves) {
  const auto currentSnapshot = snapshot();
  const double baseResidualU = rU(spacetime_);
  std::set<std::vector<std::uint64_t>> baseCellSet;
  for (const auto &topSimplex : spacetime_->getTopSimplices())
    baseCellSet.insert(topTuple(*topSimplex));
  double bestObjectiveDelta = -convergenceTolerance_;
  bool foundImprovingMove = false;
  Snapshot bestSnapshot;
  for (int candidateIndex = 0; candidateIndex < nCandidateMoves; ++candidateIndex) {
    const auto moveSpecification = drawRandomMoveSpecification(*spacetime_);
    auto candidateSpacetime = build(currentSnapshot);
    if (!applyMoveSpecification(candidateSpacetime, moveSpecification)) continue;
    const double objectiveDelta =
        deltaF(candidateSpacetime, baseResidualU, baseCellSet);
    if (objectiveDelta < bestObjectiveDelta) {
      bestObjectiveDelta = objectiveDelta;
      bestSnapshot = snapshotOf(*candidateSpacetime);
      foundImprovingMove = true;
    }
  }
  if (foundImprovingMove) {
    spacetime_ = build(bestSnapshot);
    return bestObjectiveDelta;
  }
  return 0.0;
}

double MultiCobordism::trapDoorMove(int attempts) {
  const auto currentSnapshot = snapshot();
  const double baseResidualU = rU(spacetime_);
  std::set<std::vector<std::uint64_t>> baseCellSet;
  for (const auto &topSimplex : spacetime_->getTopSimplices())
    baseCellSet.insert(topTuple(*topSimplex));
  // Commit the first gated move we can apply from the FULL range
  // (drawRandomMoveSpecification: add/remove/flip/iflip/cone_out/cone_in). It need
  // NOT lower F — that is the escape; the gate (a valid manifold-with-boundary, no
  // pinned vertex removed) keeps it sound. Several tries because a given random
  // move may not propose/validate on the current complex.
  for (int attempt = 0; attempt < std::max(1, attempts); ++attempt) {
    const auto moveSpecification = drawRandomMoveSpecification(*spacetime_);
    auto candidateSpacetime = build(currentSnapshot);
    if (!applyMoveSpecification(candidateSpacetime, moveSpecification)) continue;
    const double objectiveDelta =
        deltaF(candidateSpacetime, baseResidualU, baseCellSet);
    spacetime_ = build(snapshotOf(*candidateSpacetime));
    return objectiveDelta;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

void MultiCobordism::growBoundaryRegions() {
  // Expand one block's region by a shell — the vertices of every top cell touching
  // it — so it tracks the bulk's growth and gets room to open the holes that carry
  // it. A block already carrying (residual < tolerance) is left alone, so it stops
  // growing once it represents its state.
  const auto growOneShell = [this](BoundaryBlock &block) {
    if (residualForBoundaryBlock(block, spacetime_) < inputCarriedTolerance_) return;
    std::set<std::uint64_t> expanded = block.vertices;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topTuple(*topSimplex);
      bool touchesRegion = false;
      for (auto vertexId : cellVertexIds)
        if (block.vertices.count(vertexId)) {
          touchesRegion = true;
          break;
        }
      if (touchesRegion)
        expanded.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    block.vertices = std::move(expanded);
  };
  for (auto &inputBlock : inputBlocks_) growOneShell(inputBlock);
  // Localized OUTPUT blocks (a 2→2 recombination's diquark ⊔ antidiquark) grow the
  // same way; a SINGLE output reads off the whole and has no block here, so this is
  // a no-op for the formation node.
  for (auto &outputBlock : outputBlocks_) growOneShell(outputBlock);
}

std::vector<double> MultiCobordism::runStage1(int maxSteps, int nCandidateMoves,
                                                 int patience, bool growBoundaries) {
  // The register is "carried" (converged) once the summed r_U is essentially zero.
  constexpr double kRegisterCarriedTolerance = 1e-3;
  std::vector<double> objectiveTrace = {objective()};
  int trapDoorGrows = 0;   // consecutive cone-ins since the last improving move
  Snapshot burstStart;     // complex state before the current unproductive grow burst
  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex) {
    // INITIALIZATION ONLY: while establishing the boundary, let each not-yet-carrying
    // region expand a shell so it can develop the holes that carry its state. Off
    // during the bulk evolution — the boundary ∂W is then frozen.
    if (growBoundaries) growBoundaryRegions();
    const double objectiveDelta = step(nCandidateMoves);
    if (objectiveDelta < -convergenceTolerance_) {
      // An F-lowering surgery move: progress. Any preceding cone-ins led here, so
      // keep them and reset the trap-door burst.
      objectiveTrace.push_back(objectiveTrace.back() + objectiveDelta);
      trapDoorGrows = 0;
      continue;
    }
    // No move lowered the objective. If the register is already carried, that IS
    // convergence — halt (the trap door is unnecessary, and growing further would
    // only disturb the carried state).
    if (rU(spacetime_) < kRegisterCarriedTolerance) break;
    // TRAP DOOR: grow via a gated cone-in so the optimizer escapes a too-small
    // complex instead of giving up.
    if (trapDoorGrows == 0) burstStart = snapshot();   // remember the pre-burst state
    const double growthDelta = trapDoorMove(nCandidateMoves);
    if (!std::isfinite(growthDelta)) {
      // No gated move applied in this batch of random tries. Reseed and retry on the
      // next iteration rather than halting the whole run: with an advanced RNG a
      // valid move almost always appears (one batch missing is not a true dead end).
      // `maxSteps` bounds the retries. This is the other half of in-run stall
      // recovery — without it a single missed batch ends a long call early, which
      // is exactly what chunking used to paper over.
      randomNumberGenerator_.seed(randomNumberGenerator_());
      continue;
    }
    objectiveTrace.push_back(objectiveTrace.back() + growthDelta);
    if (++trapDoorGrows >= patience) {
      // `patience` cone-ins with no improving move in between: this grow burst is
      // not helping. Revert the whole unproductive burst, reseed, and try a FRESH
      // trajectory from the reverted state rather than halting — the stall recovery
      // happens within this run, so a single long call is as robust as many short
      // ones (no need for the caller to chunk its run_stage1 budget). `maxSteps`
      // still bounds the loop.
      spacetime_ = build(burstStart);
      objectiveTrace.push_back(objective());
      randomNumberGenerator_.seed(randomNumberGenerator_());
      trapDoorGrows = 0;
    }
  }
  return objectiveTrace;
}

void MultiCobordism::constructInputs(const std::vector<std::uint64_t> &seeds,
                                        int rounds) {
  constructBlocks(seeds, inputTargets_, inputBlocks_, rounds);
}

void MultiCobordism::constructOutputs(const std::vector<std::uint64_t> &seeds,
                                         int rounds) {
  constructBlocks(seeds, outputTargets_, outputBlocks_, rounds);
}

void MultiCobordism::constructBlocks(
    const std::vector<std::uint64_t> &seeds,
    const std::vector<std::vector<complexd>> &targets,
    std::vector<BoundaryBlock> &destinationBlocks, int rounds) {
  // Region-restricted surgical solve per boundary block: grow whatever emergent
  // topology in the seed's neighbourhood carries the block's target (kept by Δr).
  for (std::size_t blockIndex = 0;
       blockIndex < targets.size() && blockIndex < seeds.size(); ++blockIndex) {
    const std::uint64_t seedVertexId = seeds[blockIndex];
    std::set<std::uint64_t> regionVertexIds;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topTuple(*topSimplex);
      if (std::find(cellVertexIds.begin(), cellVertexIds.end(), seedVertexId) !=
          cellVertexIds.end())
        regionVertexIds.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    BoundaryBlock boundaryBlock{regionVertexIds, targets[blockIndex]};
    double residual = residualForBoundaryBlock(boundaryBlock, spacetime_);
    for (int roundIndex = 0; roundIndex < rounds; ++roundIndex) {
      const auto roundSnapshot = snapshot();
      // a region-restricted move: cone on a cell inside the region.
      std::vector<std::vector<std::uint64_t>> cellsInsideRegion;
      for (const auto &topSimplex : spacetime_->getTopSimplices()) {
        auto cellVertexIds = topTuple(*topSimplex);
        bool cellIsInsideRegion = true;
        for (auto vertexId : cellVertexIds)
          if (!regionVertexIds.count(vertexId)) {
            cellIsInsideRegion = false;
            break;
          }
        if (cellIsInsideRegion)
          cellsInsideRegion.push_back(std::move(cellVertexIds));
      }
      if (cellsInsideRegion.empty()) break;
      const auto &chosenCell =
          cellsInsideRegion[randomNumberGenerator_() % cellsInsideRegion.size()];
      auto candidateSpacetime = build(roundSnapshot);
      bool moveWasApplied;
      if (randomNumberGenerator_() % 2) {
        // cone-out removes the whole chosen top cell.
        moveWasApplied =
            SurgicalCone(candidateSpacetime.get()).coneOut(chosenCell).first;
      } else {
        // cone-in GROWS: a fresh apex joins a d-vertex FACET of the cell (drop one
        // vertex) to form a new top cell. coneIn needs d targets, NOT the full
        // (d+1)-vertex cell — same payload drawRandomMoveSpecification builds for a
        // stage-1 cone_in. Passing the whole cell made every seeding cone-in fail
        // the arg-count check, so seeding could only ever shrink, never grow.
        std::vector<std::uint64_t> coneInFace = chosenCell;
        coneInFace.erase(coneInFace.begin() +
                         static_cast<std::ptrdiff_t>(randomNumberGenerator_() %
                                                     coneInFace.size()));
        moveWasApplied =
            SurgicalCone(candidateSpacetime.get()).coneIn(coneInFace).first;
      }
      if (!moveWasApplied ||
          !EigenstateSynthesis(candidateSpacetime, dualComplexGateDegree_)
               .dualComplexValid()
               .first)
        continue;
      const double newResidual =
          residualForBoundaryBlock(boundaryBlock, candidateSpacetime);
      if (newResidual < residual - convergenceTolerance_) {
        residual = newResidual;
        spacetime_ = build(snapshotOf(*candidateSpacetime));
        // refresh the region with any new vertices the move added near the seed.
        for (const auto &topSimplex : spacetime_->getTopSimplices()) {
          auto cellVertexIds = topTuple(*topSimplex);
          if (std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        seedVertexId) != cellVertexIds.end())
            regionVertexIds.insert(cellVertexIds.begin(), cellVertexIds.end());
        }
        boundaryBlock.vertices = regionVertexIds;
      }
    }
    destinationBlocks.push_back(boundaryBlock);
  }
}

std::vector<double> MultiCobordism::runStage2(double beta, int maxIters,
                                                 double alpha0) {
  auto edges = spacetime_->getEdgeList()->toVector();
  const std::size_t edgeCount = edges.size();
  auto fullObjective = [&]() {
    return beta * reggeActionGradient(spacetime_) + gamma_ * rU(spacetime_);
  };
  std::vector<double> objectiveTrace = {fullObjective()};
  double stepScale = alpha0;
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex) {
    ReggeSolver reggeSolver(spacetime_, MatterConfiguration());
    const auto gradientComponents = reggeSolver.actionGradientExact();
    const auto hessianRows = reggeSolver.actionHessianExact();  // rows of complex
    Eigen::VectorXcd gradientVector(edgeCount);
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      gradientVector(edgeIndex) = gradientComponents[edgeIndex];
    Eigen::MatrixXcd hessianMatrix(edgeCount, edgeCount);
    for (std::size_t rowIndex = 0; rowIndex < edgeCount; ++rowIndex)
      for (std::size_t columnIndex = 0; columnIndex < edgeCount; ++columnIndex)
        hessianMatrix(rowIndex, columnIndex) =
            hessianRows[rowIndex][columnIndex];
    const Eigen::VectorXcd descentDirection =
        beta * 2.0 * (hessianMatrix.conjugate() * gradientVector);
    Eigen::VectorXcd squaredLengths(edgeCount);
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      squaredLengths(edgeIndex) = edges[edgeIndex]->getSquaredLength();
    const double currentObjective = objectiveTrace.back();
    double trialStepScale = stepScale;
    bool objectiveImproved = false;
    for (int lineSearchIndex = 0; lineSearchIndex < 24; ++lineSearchIndex) {
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
        complexd trialSquaredLength = squaredLengths(edgeIndex) -
                                trialStepScale * descentDirection(edgeIndex);
        double boundedRealPart =
            std::min(std::max(trialSquaredLength.real(), 0.05),
                     20.0);  // bound the real part
        edges[edgeIndex]->setSquaredLength(
            complexd(boundedRealPart, trialSquaredLength.imag()));
      }
      double trialObjective;
      try {
        trialObjective = fullObjective();
      } catch (...) {
        trialObjective = std::numeric_limits<double>::infinity();
      }
      if (trialObjective < currentObjective - convergenceTolerance_) {
        objectiveTrace.push_back(trialObjective);
        stepScale = std::min(stepScale * 1.3, 1.0);
        objectiveImproved = true;
        break;
      }
      trialStepScale *= 0.5;
    }
    if (!objectiveImproved) {
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
        edges[edgeIndex]->setSquaredLength(squaredLengths(edgeIndex));
      break;
    }
  }
  return objectiveTrace;
}

}  // namespace tessera::cobordism
