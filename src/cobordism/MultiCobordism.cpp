// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "cobordism/DiracKahler.h"
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
using cd = std::complex<double>;

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
    const std::vector<std::vector<cd>> &inputTargets,
    const std::vector<std::vector<cd>> &outputTargets,
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
    const std::vector<cd> &targetState) {
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

double MultiCobordism::diracKahlerSpinCasimir(
    const std::shared_ptr<Spacetime> &spacetime) {
  DiracKahler diracKahler(spacetime);
  const auto flatGammas = diracKahler.gammas(/*lorentzian=*/false);
  const std::size_t fiberDimension = diracKahler.gammaDimension();
  std::vector<Eigen::MatrixXcd> gamma;
  gamma.reserve(flatGammas.size());
  for (const auto &flat : flatGammas) {
    Eigen::MatrixXcd matrix(static_cast<int>(fiberDimension),
                            static_cast<int>(fiberDimension));
    for (std::size_t row = 0; row < fiberDimension; ++row)
      for (std::size_t col = 0; col < fiberDimension; ++col)
        matrix(static_cast<int>(row), static_cast<int>(col)) =
            flat[row * fiberDimension + col];
    gamma.push_back(std::move(matrix));
  }
  // S_a = -i Sigma_{jk} over the spatial plane perpendicular to axis a (cyclic over the
  // spatial gammas 1,2,3); Sigma_ij = 1/4 [gamma_i, gamma_j] has eigenvalues +/- i/2, so
  // S_a has +/- 1/2 and the Casimir Sum_a S_a^2 = 3/4 * I on the spin-1/2 fiber.
  const int spatialPlane[3][2] = {{2, 3}, {3, 1}, {1, 2}};
  Eigen::MatrixXcd casimir = Eigen::MatrixXcd::Zero(
      static_cast<int>(fiberDimension), static_cast<int>(fiberDimension));
  for (int axis = 0; axis < 3; ++axis) {
    const int i = spatialPlane[axis][0];
    const int j = spatialPlane[axis][1];
    const Eigen::MatrixXcd sigma =
        0.25 * (gamma[i] * gamma[j] - gamma[j] * gamma[i]);
    const Eigen::MatrixXcd spinComponent =
        std::complex<double>(0.0, -1.0) * sigma;
    casimir += spinComponent * spinComponent;
  }
  return casimir.trace().real() / static_cast<double>(fiberDimension);  // -> 3/4
}

double MultiCobordism::holeDeficit(const std::shared_ptr<Spacetime> &spacetime,
                                   const std::vector<std::uint64_t> &hole) {
  const std::set<std::uint64_t> holeVertexSet(hole.begin(), hole.end());
  double totalDeficit = 0.0;
  for (const auto &simplex : spacetime->getSimplices()) {
    if (simplex->getVertices().size() != 3) continue;  // hinges (triangles) in 4D
    bool insideHole = true;
    for (const auto *vertex : simplex->getVertices())
      if (!holeVertexSet.count(vertex->getId())) {
        insideHole = false;
        break;
      }
    if (insideHole) totalDeficit += simplex->deficitAngle();
  }
  return totalDeficit;
}

double MultiCobordism::compositeSpinJ2(std::size_t outputBlockIndex) const {
  if (outputBlockIndex >= outputs_.size())
    throw std::runtime_error(
        "MultiCobordism::compositeSpinJ2: output block index out of range");
  const auto blockSubcomplex =
      subcomplexWithinVertexSet(spacetime_, outputs_[outputBlockIndex].vertices);
  if (!blockSubcomplex)
    throw std::runtime_error(
        "MultiCobordism::compositeSpinJ2: output block has no sub-complex");
  const auto holes = emergentHoles(*blockSubcomplex, 3);
  if (holes.size() < 3)
    throw std::runtime_error(
        "MultiCobordism::compositeSpinJ2: output block has no 3-hole (b3) register");
  // Materialize the skeleton once so the hinges carry deficit angles.
  const ReggeSolver reggeSolver(blockSubcomplex, MatterConfiguration());
  (void)reggeSolver;
  const double spinHalfCasimir = diracKahlerSpinCasimir(blockSubcomplex);  // 3/4
  // The two-quark pair-loop gamma_ij LITERALLY encircles holes i and j; by cycle additivity
  // gamma_ij = gamma_i + gamma_j, so the deficit it encloses is eps_i + eps_j (the curvature
  // at the two encircled holes), and <S_i.S_j> = 1/4 cos(eps_i + eps_j). No duality assumed
  // (this differs from the complementary-hole proxy by the Gauss-Bonnet constant).
  double holeDeficits[3];
  for (std::size_t holeIndex = 0; holeIndex < 3; ++holeIndex)
    holeDeficits[holeIndex] = holeDeficit(blockSubcomplex, holes[holeIndex]);
  static const int pair[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  double crossTermSum = 0.0;
  for (const auto &twoQuarkLoop : pair)
    crossTermSum +=
        0.25 * std::cos(holeDeficits[twoQuarkLoop[0]] + holeDeficits[twoQuarkLoop[1]]);
  return 3.0 * spinHalfCasimir + 2.0 * crossTermSum;
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
  // The symmetric cobordism residual: one r_U term per boundary block — every
  // input AND every output sub-complex (the bulk routes the connectivity between
  // them). The Regge extremization term lives in objective()/stages, not here.
  double totalResidual = 0.0;
  for (const auto &inputBlock : inputs_)
    totalResidual += residualForBoundaryBlock(inputBlock, spacetime);
  for (const auto &outputBlock : outputs_)
    totalResidual += residualForBoundaryBlock(outputBlock, spacetime);
  // No blocks yet (pre-construction): score the raw output targets as full leak so
  // the bare objective is well-defined and matches the single-merge convention.
  if (inputs_.empty() && outputs_.empty())
    for (int registerDegree : registerDegrees_)
      for (const auto &outputTarget : outputTargets_)
        totalResidual += residualOfTargetStateAgainstHarmonic(
            spacetime, registerDegree, outputTarget);
  return totalResidual;
}

double MultiCobordism::objective() const {
  return reggeActionGradient(spacetime_) + gamma_ * rU(spacetime_);
}

std::set<std::uint64_t> MultiCobordism::boundaryVerts() const {
  std::set<std::uint64_t> boundaryVertexIds;
  for (const auto &inputBlock : inputs_)
    boundaryVertexIds.insert(inputBlock.vertices.begin(),
                             inputBlock.vertices.end());
  for (const auto &outputBlock : outputs_)
    boundaryVertexIds.insert(outputBlock.vertices.begin(),
                             outputBlock.vertices.end());
  return boundaryVertexIds;
}

MultiCobordism::Snapshot MultiCobordism::snapshotOf(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> cellVertexTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    cellVertexTuples.push_back(topTuple(*topSimplex));
  std::map<std::pair<std::uint64_t, std::uint64_t>, cd> squaredLengthsByEdge;
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

double MultiCobordism::step(int nCandidates) {
  const auto currentSnapshot = snapshot();
  const double baseResidualU = rU(spacetime_);
  std::set<std::vector<std::uint64_t>> baseCellSet;
  for (const auto &topSimplex : spacetime_->getTopSimplices())
    baseCellSet.insert(topTuple(*topSimplex));
  double bestObjectiveDelta = -convergenceTolerance_;
  bool foundImprovingMove = false;
  Snapshot bestSnapshot;
  for (int candidateIndex = 0; candidateIndex < nCandidates; ++candidateIndex) {
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

std::vector<double> MultiCobordism::runStage1(int maxSteps, int nCandidates,
                                                 int patience) {
  std::vector<double> objectiveTrace = {objective()};
  int consecutiveStalls = 0;
  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex) {
    const double objectiveDelta = step(nCandidates);
    objectiveTrace.push_back(objectiveTrace.back() + objectiveDelta);
    if (objectiveDelta >= -convergenceTolerance_) {
      ++consecutiveStalls;
      randomNumberGenerator_.seed(randomNumberGenerator_());
      if (consecutiveStalls >= patience) break;
    } else {
      consecutiveStalls = 0;
    }
  }
  return objectiveTrace;
}

void MultiCobordism::constructInputs(const std::vector<std::uint64_t> &seeds,
                                        int rounds) {
  constructBlocks(seeds, inputTargets_, inputs_, rounds);
}

void MultiCobordism::constructOutputs(const std::vector<std::uint64_t> &seeds,
                                         int rounds) {
  constructBlocks(seeds, outputTargets_, outputs_, rounds);
}

void MultiCobordism::constructBlocks(
    const std::vector<std::uint64_t> &seeds,
    const std::vector<std::vector<cd>> &targets,
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
      const bool moveWasApplied =
          (randomNumberGenerator_() % 2)
              ? SurgicalCone(candidateSpacetime.get()).coneOut(chosenCell).first
              : SurgicalCone(candidateSpacetime.get()).coneIn(chosenCell).first;
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
        cd trialSquaredLength = squaredLengths(edgeIndex) -
                                trialStepScale * descentDirection(edgeIndex);
        double boundedRealPart =
            std::min(std::max(trialSquaredLength.real(), 0.05),
                     20.0);  // bound the real part
        edges[edgeIndex]->setSquaredLength(
            cd(boundedRealPart, trialSquaredLength.imag()));
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
