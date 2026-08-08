// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/SurgicalCone.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeKey.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
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

// Sorted vertex-id tuple of a top simplex.
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

// The boundary facets of `spacetime`, as sorted vertex-id tuples, for membership tests.
std::set<std::vector<std::uint64_t>> boundaryFacetSet(const Spacetime &spacetime) {
  std::set<std::vector<std::uint64_t>> facets;
  for (auto facet : spacetime.getBoundary()) {  // getBoundary() returns a fresh copy
    std::sort(facet.begin(), facet.end());
    facets.insert(facet);
  }
  return facets;
}

// Whether any `pinned` vertex is no longer live in `spacetime` (a stranded pinned vertex).
bool strandsPinned(const Spacetime &spacetime, const std::set<std::uint64_t> &pinned) {
  std::set<std::uint64_t> live;
  for (const auto *vertex : spacetime.getVertexList()->toVector())
    live.insert(vertex->getId());
  for (auto vertexId : pinned)
    if (!live.count(vertexId)) return true;
  return false;
}
}  // namespace

MultiCobordism::MultiCobordism(
    std::shared_ptr<Spacetime> host,
    const std::vector<std::vector<complexd>> &inputTargets,
    const std::vector<std::vector<complexd>> &outputTargets,
    const std::vector<int> &degrees, double gamma, std::uint64_t seed,
    int precone, bool shouldProposeDispositions)
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
      randomNumberGenerator_(seed) {
  // Assigned in the body rather than the init list: the member is declared last,
  // and C++ initializes in DECLARATION order, so an init-list entry here would
  // reorder-warn. It is a plain bool with an in-class default, so nothing depends
  // on it being set earlier.
  shouldProposeDispositions_ = shouldProposeDispositions;
  // Pre-grow the seed by `precone` gated cone-ins before any optimization, so the
  // stage-1 search starts from a larger complex grown emergently from the host (no
  // input/output block is seeded yet, so nothing is pinned — the gate is the only
  // constraint). `precone <= 0` leaves the host and RNG untouched.
  if (precone > 0) preconeCells(precone);
}

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
    // carry it (the output EMERGES; it is not frozen by seedOutputs).
    for (int registerDegree : registerDegrees_)
      totalResidual += residualOfTargetStateAgainstHarmonic(
          spacetime, registerDegree, outputTargets_.front());
  } else {
    // Multiple outputs (e.g. a 2->2 recombination → diquark ⊔ antidiquark) live in
    // distinct regions: read each off its own constructed block. EMPTY outputTargets
    // is the supported nothing-pinned-downstream shape (#555): no output term at
    // all — rU is the weighted input residuals alone, and the whole's final state
    // emerges (read after the fact, e.g. ProtonIngredients' singlet diagnostic).
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

std::set<std::uint64_t> MultiCobordism::pinnedBoundaryVertices() const {
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
  // #613: with emergent dispositions ON the draw also offers a TIMELIKE cone-in
  // and a disposition flip on an existing edge. Both are ordinary candidate moves:
  // proposed at random, scored by deltaF, committed only if they lower F. Nothing
  // prescribes causal structure -- the objective decides whether it wants any.
  //
  // The disposition is drawn as a DISCRETE move rather than left to stage 2 because
  // a continuous descent cannot carry l^2 across zero: that is a null, degenerate
  // configuration where the deficit angles and dual volumes are singular, so the
  // Euclidean orthant is a trap. Measured: every edge stays spacelike and Im S = 0
  // through 110+ relaxation iterations.
  static const char *baseMoveKinds[] = {kAddMove,  kRemoveMove, kFlipMove,
                                        kIFlipMove, kConeOut,   kConeIn};
  static const char *dispositionMoveKinds[] = {
      kAddMove, kRemoveMove,     kFlipMove,        kIFlipMove,
      kConeOut, kConeIn,         kConeInTimelike,  kFlipDisposition};
  const char *const *moveKinds =
      shouldProposeDispositions_ ? dispositionMoveKinds : baseMoveKinds;
  const std::size_t nMoveKinds = shouldProposeDispositions_ ? 8u : 6u;
  const std::string moveKind = moveKinds[randomNumberGenerator_() % nMoveKinds];

  // Flip the disposition of one existing edge, chosen uniformly. The payload is
  // the edge's two vertex ids.
  if (moveKind == kFlipDisposition) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edgeEndpoints;
    if (spacetime.getEdgeList())
      for (const auto *edge : spacetime.getEdgeList()->toVector())
        if (edge != nullptr && edge->getSource() != nullptr &&
            edge->getTarget() != nullptr)
          edgeEndpoints.emplace_back(edge->getSource()->getId(),
                                     edge->getTarget()->getId());
    if (edgeEndpoints.empty()) return {kNoop, {}};
    const auto &chosen =
        edgeEndpoints[randomNumberGenerator_() % edgeEndpoints.size()];
    return {kFlipDisposition, {chosen.first, chosen.second}};
  }
  if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove)
    return {moveKind,
            {static_cast<std::uint64_t>(randomNumberGenerator_() % (1u << 31))}};
  std::vector<std::vector<std::uint64_t>> topCellTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    topCellTuples.push_back(topTuple(*topSimplex));
  if (topCellTuples.empty()) return {kNoop, {}};
  const auto &chosenCell =
      topCellTuples[randomNumberGenerator_() % topCellTuples.size()];
  if (moveKind == kConeOut) return {kConeOut, chosenCell};
  // cone_in and cone_in_timelike share a payload (the facet to cone onto); only
  // the apex-edge disposition differs when applied.
  const std::size_t droppedVertexIndex =
      randomNumberGenerator_() % chosenCell.size();
  std::vector<std::uint64_t> coneInFace;
  for (std::size_t vertexIndex = 0; vertexIndex < chosenCell.size();
       ++vertexIndex)
    if (vertexIndex != droppedVertexIndex)
      coneInFace.push_back(chosenCell[vertexIndex]);
  return {moveKind, coneInFace};
}

bool MultiCobordism::applyMoveSpecification(
    const std::shared_ptr<Spacetime> &spacetime,
    const MoveSpec &moveSpecification) {
  const auto &moveKind = moveSpecification.first;
  if (moveKind == kNoop) return false;
  bool moveWasApplied = false;
  if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove) {
    std::mt19937 moveRandomEngine(
        static_cast<std::uint32_t>(moveSpecification.second[0]));
    using ::tessera::spacetime::PachnerMode;
    if (moveKind == kAddMove) {
      ::tessera::spacetime::AddMove pachnerMove(
          spacetime.get(), &moveRandomEngine, false, PachnerMode::PreGeometric,
          false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kRemoveMove) {
      ::tessera::spacetime::RemoveMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kFlipMove) {
      ::tessera::spacetime::FlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else {
      ::tessera::spacetime::IFlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    }
  } else if (moveKind == kConeOut) {
    moveWasApplied =
        SurgicalCone(spacetime.get()).coneOut(moveSpecification.second).first;
  } else if (moveKind == kFlipDisposition) {
    // #613: negate one edge's squared length, carrying it across the light cone.
    // Spacelike <-> timelike is a DISCRETE step stage 2 cannot take (it would have
    // to pass through the singular l^2 = 0), which is why it is a move. Not gated
    // here -- deltaF and step()'s acceptance test gate it, exactly as for every
    // other move.
    if (payloadNamesAnEdge(moveSpecification.second) &&
        spacetime->getEdgeList()) {
      // O(1) via the EdgeList's fingerprint -> slot map, not an O(|E|) scan:
      // EdgeKey canonicalizes the endpoint pair, so orientation does not matter.
      const ::tessera::mesh::EdgeKey key(moveSpecification.second[0],
                                         moveSpecification.second[1]);
      if (auto *edge =
              spacetime->getEdgeList()->get(key.fingerprint.fingerprint())) {
        edge->setSquaredLength(-edge->getSquaredLength());
        moveWasApplied = true;
      }
    }
  } else {
    moveWasApplied = SurgicalCone(spacetime.get())
                         .coneIn(moveSpecification.second,
                                 /*timelike=*/moveKind == kConeInTimelike)
                         .first;
  }
  if (!moveWasApplied) return false;
  std::set<std::uint64_t> liveVertexIds;
  for (const auto &topSimplex : spacetime->getTopSimplices())
    for (auto vertexId : topTuple(*topSimplex)) liveVertexIds.insert(vertexId);
  for (auto vertexId : pinnedBoundaryVertices())
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

void MultiCobordism::preconeCells(int count) {
  // Each cone-in cones a fresh apex onto a random codim-1 facet (a top cell with one
  // vertex dropped) and is committed only through applyMoveSpecification's
  // dualComplexValid gate — the same gated primitive the trap door uses, so the
  // pre-growth is sound (nothing inserted by fiat). On the single-Δ⁴ seed (a 4-ball)
  // a cone-in over a boundary facet is valid, so this enlarges the 4-ball; a draw
  // onto an already-saturated interior facet is rejected by the gate and retried.
  constexpr int kAttemptsPerCone = 16;  // gated tries before giving up on one cone
  for (int conedSoFar = 0; conedSoFar < count; ++conedSoFar) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime_->getTopSimplices())
      topCellTuples.push_back(topTuple(*topSimplex));
    if (topCellTuples.empty()) return;  // nothing to cone onto
    bool coned = false;
    for (int attempt = 0; attempt < kAttemptsPerCone && !coned; ++attempt) {
      const auto &chosenCell =
          topCellTuples[randomNumberGenerator_() % topCellTuples.size()];
      const std::size_t droppedVertexIndex =
          randomNumberGenerator_() % chosenCell.size();
      std::vector<std::uint64_t> coneInFace;  // a codim-1 facet: drop one vertex
      for (std::size_t vertexIndex = 0; vertexIndex < chosenCell.size();
           ++vertexIndex)
        if (vertexIndex != droppedVertexIndex)
          coneInFace.push_back(chosenCell[vertexIndex]);
      auto candidateSpacetime = build(snapshot());
      if (applyMoveSpecification(candidateSpacetime, {kConeIn, coneInFace})) {
        spacetime_ = build(snapshotOf(*candidateSpacetime));
        coned = true;
      }
    }
    if (!coned) return;  // no valid cone-in found for this cell; stop early
  }
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
    //
    // Growing a region CHANGES F and so must be booked into the trace (#607).
    // `growBoundaryRegions` mutates only the boundary blocks' vertex sets and never
    // touches `spacetime_`, so `reggeActionGradient` is provably unchanged and the
    // whole objective change is `gamma_ * Δr_U` — exact, not an approximation of the
    // kind `deltaF` makes for the gradient term. Leaving it unbooked let the
    // accumulated trace drift arbitrarily far from `objective()` (measured at tens of
    // thousands on preconed hosts), and since the SAME accumulated quantity gates
    // acceptance, moves were being committed against a number that was not F.
    if (growBoundaries) {
      const double residualBeforeGrowth = rU(spacetime_);
      growBoundaryRegions();
      const double growthObjectiveDelta =
          gamma_ * (rU(spacetime_) - residualBeforeGrowth);
      if (growthObjectiveDelta != 0.0)
        objectiveTrace.push_back(objectiveTrace.back() + growthObjectiveDelta);
    }
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

std::vector<double> MultiCobordism::runUnified(int maxSteps,
                                                int nCandidateMoves, double beta,
                                                double alpha0,
                                                double convergenceTarget) {
  // ONE stage. Combinatorial and geometric candidates are priced against the SAME
  // objective and the best is committed; when nothing lowers F the drive stops.
  //
  // There is no trap door here and no heuristic of any kind -- no move committed
  // without lowering F, no patience counter, no burst revert, no reseeding, no
  // backoff. runStage1 needs an escape because a combinatorial move is priced at the
  // un-relaxed metric and freshly grown cells therefore always look like a loss;
  // letting relaxation compete for the same step removes that penalty at its source,
  // which is the whole point of merging the stages. Adding an escape back would
  // destroy the measurement: a complex that grows because it was pushed says nothing
  // about whether growth pays.
  const auto fullObjectiveOf = [&](const std::shared_ptr<Spacetime> &candidate) {
    return beta * reggeActionGradient(candidate) + gamma_ * rU(candidate);
  };

  lastUnifiedExhaustedMoves_ = false;
  lastUnifiedMoveKinds_.clear();
  std::vector<double> objectiveTrace = {fullObjectiveOf(spacetime_)};
  double stepScale = alpha0;

  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex) {
    const double currentObjective = objectiveTrace.back();
    if (currentObjective <= convergenceTarget) break;   // the objective is MET

    const auto currentSnapshot = snapshot();
    double bestObjective = currentObjective;
    Snapshot bestSnapshot;
    std::string bestMoveKind;
    bool foundImprovingMove = false;

    // ---- combinatorial candidates ----
    // Priced by the FULL objective on the candidate complex, NOT by deltaF's
    // incremental sum over affected edges. deltaF and the geometric step measure
    // different quantities, and a move that won a step because of which estimator
    // scored it would be an artifact of the code rather than of the physics.
    for (int candidateIndex = 0; candidateIndex < nCandidateMoves; ++candidateIndex) {
      const auto moveSpecification = drawRandomMoveSpecification(*spacetime_);
      auto candidateSpacetime = build(currentSnapshot);
      if (!applyMoveSpecification(candidateSpacetime, moveSpecification)) continue;
      const double candidateObjective = fullObjectiveOf(candidateSpacetime);
      if (candidateObjective < bestObjective) {
        bestObjective = candidateObjective;
        bestSnapshot = snapshotOf(*candidateSpacetime);
        bestMoveKind = moveSpecification.first;
        foundImprovingMove = true;
      }
    }

    // ---- the geometric candidate ----
    // One relaxation step along the exact on-manifold descent direction: for real F
    // of a complex variable on the real axis dF/dx = 2 Re(dF/dz-bar), so the
    // direction is Re(2*beta*H-bar*g) (#589). Trials are constructed exactly real, so
    // Im l^2 == 0 holds by construction. Evaluated on a COPY, so a losing geometric
    // trial cannot perturb the complex the combinatorial candidates were priced
    // against.
    {
      auto geometricCandidate = build(currentSnapshot);
      auto candidateEdges = geometricCandidate->getEdgeList()->toVector();
      const std::size_t edgeCount = candidateEdges.size();
      if (edgeCount > 0) {
        ReggeSolver reggeSolver(geometricCandidate, MatterConfiguration());
        const auto gradientComponents = reggeSolver.actionGradientExact();
        const auto hessianRows = reggeSolver.actionHessianExact();
        Eigen::VectorXcd gradientVector(edgeCount);
        for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
          gradientVector(edgeIndex) = gradientComponents[edgeIndex];
        Eigen::MatrixXcd hessianMatrix(edgeCount, edgeCount);
        for (std::size_t rowIndex = 0; rowIndex < edgeCount; ++rowIndex)
          for (std::size_t columnIndex = 0; columnIndex < edgeCount; ++columnIndex)
            hessianMatrix(rowIndex, columnIndex) = hessianRows[rowIndex][columnIndex];
        const Eigen::VectorXd descentDirection =
            (beta * 2.0 * (hessianMatrix.conjugate() * gradientVector)).real();
        Eigen::VectorXd squaredLengths(edgeCount);
        for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
          squaredLengths(edgeIndex) = candidateEdges[edgeIndex]->getRealSquaredLength();

        // A backtracking ladder, which is not a heuristic but the standard way to
        // pick a step LENGTH along a direction that is itself exact: for smooth F
        // with non-zero gradient some small enough step lowers F, so exhausting the
        // ladder means there is no downhill along this direction. The best rung is
        // kept rather than the first that improves.
        double trialStepScale = stepScale;
        double bestGeometricObjective = currentObjective;
        Eigen::VectorXd bestGeometricLengths;
        for (int rung = 0; rung < 64; ++rung) {
          for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
            candidateEdges[edgeIndex]->setSquaredLength(
                complexd(squaredLengths(edgeIndex) -
                             trialStepScale * descentDirection(edgeIndex),
                         0.0));
          const double trialObjective = fullObjectiveOf(geometricCandidate);
          if (trialObjective < bestGeometricObjective) {
            bestGeometricObjective = trialObjective;
            bestGeometricLengths = Eigen::VectorXd(edgeCount);
            for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
              bestGeometricLengths(edgeIndex) =
                  candidateEdges[edgeIndex]->getRealSquaredLength();
          }
          trialStepScale *= 0.5;
        }
        if (bestGeometricLengths.size() > 0 &&
            bestGeometricObjective < bestObjective) {
          for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
            candidateEdges[edgeIndex]->setSquaredLength(
                complexd(bestGeometricLengths(edgeIndex), 0.0));
          bestObjective = bestGeometricObjective;
          bestSnapshot = snapshotOf(*geometricCandidate);
          bestMoveKind = "relax";
          foundImprovingMove = true;
          stepScale = std::min(stepScale * 1.3, 1.0);
        }
      }
    }

    if (!foundImprovingMove) {
      // No available move -- of EITHER kind -- lowers F. That is the result, and it
      // is reported rather than escaped.
      lastUnifiedExhaustedMoves_ = true;
      break;
    }
    spacetime_ = build(bestSnapshot);
    objectiveTrace.push_back(bestObjective);
    lastUnifiedMoveKinds_.push_back(bestMoveKind);
  }
  return objectiveTrace;
}

void MultiCobordism::seedInputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, inputTargets_, inputBlocks_);
}

void MultiCobordism::seedOutputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, outputTargets_, outputBlocks_);
}

void MultiCobordism::seedBlocks(
    const std::vector<std::uint64_t> &seeds,
    const std::vector<std::vector<complexd>> &targets,
    std::vector<BoundaryBlock> &destinationBlocks) {
  // Seed one boundary block per (seed vertex, target): its initial region is the seed
  // vertex's cell-neighbourhood. The block is NOT pre-grown here — runStage1's
  // growBoundaryRegions grows it under the objective, so the carrying topology is fully
  // emergent. The seed vertex is the only anchor (it distinguishes one input/output
  // from another); everything else emerges.
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
    destinationBlocks.push_back(BoundaryBlock{regionVertexIds, targets[blockIndex]});
  }
}

std::vector<double> MultiCobordism::runStage2(double beta, int maxIters,
                                                 double alpha0, double relTol) {
  auto edges = spacetime_->getEdgeList()->toVector();
  const std::size_t edgeCount = edges.size();
  auto fullObjective = [&]() {
    return beta * reggeActionGradient(spacetime_) + gamma_ * rU(spacetime_);
  };
  std::vector<double> objectiveTrace = {fullObjective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // set true only on the stationary break below
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
    // The exact gradient of F restricted to the real signed-l^2 manifold: for a
    // real-valued F of a complex variable evaluated on the real axis,
    // dF/dx = 2 Re(dF/dz̄), so the on-manifold descent direction is the REAL
    // PART of the Wirtinger direction 2β(H̄·g) (#589).
    const Eigen::VectorXd descentDirection =
        (beta * 2.0 * (hessianMatrix.conjugate() * gradientVector)).real();
    Eigen::VectorXd squaredLengths(edgeCount);
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      squaredLengths(edgeIndex) =
          edges[edgeIndex]->getRealSquaredLength();
    const double currentObjective = objectiveTrace.back();
    // Relative stationarity: accept a step only when it lowers F by more than relTol
    // scaled by the current magnitude (an absolute floor of relTol when |F| < 1). The
    // old absolute convergenceTolerance_ accepted ~1e-11 *relative* steps for F ~ 100
    // — the rounding floor; this scales the threshold with the objective instead.
    const double improvementThreshold =
        relTol * std::max(std::abs(currentObjective), 1.0);
    double trialStepScale = stepScale;
    bool objectiveImproved = false;
    for (int lineSearchIndex = 0; lineSearchIndex < 24; ++lineSearchIndex) {
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
        // The trial is UNBOUNDED on the real axis — fully Lorentzian, no
        // clamp, no causal guard (semantics: runStage2 in MultiCobordism.h).
        // Spacelike, timelike, and lightlike trials are all admissible, and
        // every trial is constructed EXACTLY real, so Im l^2 == 0 holds for
        // all time by construction — no backoff, no projection (#589).
        edges[edgeIndex]->setSquaredLength(complexd(
            squaredLengths(edgeIndex) -
                trialStepScale * descentDirection(edgeIndex),
            0.0));
      }
      // The objective is total on the real signed-l^2 manifold, so a trial
      // cannot fail to evaluate; a genuine error propagates loudly (#589).
      const double trialObjective = fullObjective();
      if (trialObjective < currentObjective - improvementThreshold) {
        objectiveTrace.push_back(trialObjective);
        stepScale = std::min(stepScale * 1.3, 1.0);
        objectiveImproved = true;
        break;
      }
      trialStepScale *= 0.5;
    }
    if (!objectiveImproved) {
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
        edges[edgeIndex]->setSquaredLength(
            complexd(squaredLengths(edgeIndex), 0.0));
      lastStage2Stationary_ = true;  // no line-search step beat the relative threshold
      break;
    }
  }
  return objectiveTrace;
}

int MultiCobordism::directedConeOut(HolePlacementStrategy strategy, int maxOpen) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;  // bound the scan; interior-first surfaces openers early
  constexpr int kProbeOpeners = 3;    // stop once a few openers are in hand
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  const auto pinned = pinnedBoundaryVertices();
  int opened = 0;
  for (int iteration = 0; iteration < maxOpen; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    std::set<std::uint64_t> holeVertices;
    for (const auto &hole : holesBefore) holeVertices.insert(hole.begin(), hole.end());
    const auto boundary = boundaryFacetSet(*spacetime);

    std::vector<std::vector<std::uint64_t>> cells;
    for (const auto *simplex : spacetime->getTopSimplices())
      cells.push_back(topTuple(*simplex));
    // Order interior-first (fewest boundary facets → hole-creators first); the secondary key
    // then places cells sharing vertices with the existing holes last (AdjacentHolesLast, a
    // separated register) or first (AdjacentHolesFirst, a clustered one).
    const auto orderKey = [&](const std::vector<std::uint64_t> &cell) {
      int boundaryFacets = 0;
      for (std::size_t i = 0; i < cell.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < cell.size(); ++j)
          if (j != i) facet.push_back(cell[j]);
        if (boundary.count(facet)) ++boundaryFacets;
      }
      int shared = 0;
      for (auto vertexId : cell)
        if (holeVertices.count(vertexId)) ++shared;
      return std::pair<int, int>(
          boundaryFacets,
          strategy == HolePlacementStrategy::AdjacentHolesFirst ? -shared : shared);
    };
    std::sort(cells.begin(), cells.end(),
              [&](const std::vector<std::uint64_t> &a,
                  const std::vector<std::uint64_t> &b) { return orderKey(a) < orderKey(b); });

    const double baseResidual = rU(spacetime);
    double bestResidual = baseResidual;
    std::vector<std::uint64_t> bestCell;
    int candidatesScanned = 0;
    int openersScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &cell : cells) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      if (!cone.coneOut(cell).first) continue;  // gate rejected; nothing applied
      const bool opensHole =
          !strandsPinned(*spacetime, pinned) &&
          emergentHoles(*spacetime, registerDegree).size() > holeCountBefore;
      if (opensHole) {
        const double candidateResidual = rU(spacetime);  // rU absorbs r_state (both steps)
        if (candidateResidual < bestResidual) {
          bestResidual = candidateResidual;
          bestCell = cell;
        }
        ++openersScanned;
      }
      cone.rollback();
      if (opensHole && openersScanned >= kProbeOpeners) break;
    }
    if (bestCell.empty()) break;  // no opener lowers rU
    if (!cone.coneOut(bestCell).first) break;
    if (strandsPinned(*spacetime, pinned)) {  // defensive: never strand a pinned vertex
      cone.rollback();
      break;
    }
    ++opened;
  }
  return opened;
}

int MultiCobordism::directedConeIn(int maxClose) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  int closed = 0;
  for (int iteration = 0; iteration < maxClose; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    if (holeCountBefore == 0) break;
    const auto boundary = boundaryFacetSet(*spacetime);

    // Cap facets: drop-one facets of the current holes that lie on the boundary — capping
    // one (a cone-in over it) closes that hole.
    std::set<std::vector<std::uint64_t>> seen;
    std::vector<std::vector<std::uint64_t>> capFacets;
    for (const auto &hole : holesBefore) {
      for (std::size_t i = 0; i < hole.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < hole.size(); ++j)
          if (j != i) facet.push_back(hole[j]);
        std::sort(facet.begin(), facet.end());
        if (boundary.count(facet) && seen.insert(facet).second) capFacets.push_back(facet);
      }
    }

    const double baseResidual = rU(spacetime);
    double bestResidual = baseResidual;
    std::vector<std::uint64_t> bestFacet;
    int candidatesScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &facet : capFacets) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      if (!cone.coneIn(facet).first) continue;
      if (emergentHoles(*spacetime, registerDegree).size() < holeCountBefore) {
        const double candidateResidual = rU(spacetime);
        if (candidateResidual < bestResidual) {
          bestResidual = candidateResidual;
          bestFacet = facet;
        }
      }
      cone.rollback();
    }
    if (bestFacet.empty()) break;  // no cap lowers rU
    if (!cone.coneIn(bestFacet).first) break;
    ++closed;
  }
  return closed;
}

void MultiCobordism::buildStep(BuildAction action, int maxSteps, int nCandidateMoves,
                               int patience, double stage2Beta, int stage2MaxIters,
                               double stage2Alpha0,
                               HolePlacementStrategy holePlacementStrategy) {
  switch (action) {
    case BuildAction::Grow:
      runStage1(maxSteps, nCandidateMoves, patience, /*growBoundaries=*/true);
      break;
    case BuildAction::Evolve:
      runStage1(maxSteps, nCandidateMoves, patience, /*growBoundaries=*/false);
      break;
    case BuildAction::Relax:
      runStage2(stage2Beta, stage2MaxIters, stage2Alpha0);
      break;
    case BuildAction::ConeOut:
      (void)directedConeOut(holePlacementStrategy);
      break;
    case BuildAction::ConeIn:
      (void)directedConeIn();
      break;
  }
}

}  // namespace tessera::cobordism
