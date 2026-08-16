// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>

#include "Logger.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
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

Eigen::VectorXcd MultiCobordism::targetStateVector(
    const std::vector<complexd> &targetState) {
  Eigen::VectorXcd targetVector(targetState.size());
  for (std::size_t componentIndex = 0; componentIndex < targetState.size();
       ++componentIndex)
    targetVector(componentIndex) = targetState[componentIndex];
  return targetVector;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::holesCarryingTheTarget(
    const Spacetime &spacetime, int registerDegree, std::size_t targetDimension) {
  auto emergentHoleTuples = emergentHoles(spacetime, registerDegree);
  // One target component per hole: holes beyond the target's width have no component
  // to carry and take no part in the fit.
  if (emergentHoleTuples.size() > targetDimension)
    emergentHoleTuples.resize(targetDimension);
  return emergentHoleTuples;
}

Eigen::MatrixXcd MultiCobordism::holePeriodMatrix(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    int degreeBettiNumber,
    const std::vector<std::vector<std::uint64_t>> &registerHoles,
    std::size_t targetDimension) {
  EigenstateSynthesis eigenstateSynthesis(spacetime, registerDegree);
  const auto flattenedCyclePeriods =
      eigenstateSynthesis.cyclePeriods(registerHoles);  // bk x m, row-major
  const std::size_t holeCount = registerHoles.size();
  Eigen::MatrixXcd periodMatrixTransposed = Eigen::MatrixXcd::Zero(
      static_cast<int>(targetDimension), degreeBettiNumber);
  for (int harmonicIndex = 0; harmonicIndex < degreeBettiNumber; ++harmonicIndex)
    for (std::size_t holeIndex = 0; holeIndex < holeCount; ++holeIndex)
      periodMatrixTransposed(static_cast<int>(holeIndex), harmonicIndex) =
          flattenedCyclePeriods[static_cast<std::size_t>(harmonicIndex) * holeCount +
                                holeIndex];
  return periodMatrixTransposed;
}

Eigen::VectorXcd MultiCobordism::relabeledTargetVector(
    const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling) {
  Eigen::VectorXcd relabeled(targetVector.size());
  for (std::size_t holeIndex = 0; holeIndex < relabeling.size(); ++holeIndex)
    relabeled(holeIndex) = targetVector(relabeling[holeIndex]);
  return relabeled;
}

MultiCobordism::RelabelingMatch MultiCobordism::bestRelabelingOfTarget(
    const Eigen::MatrixXcd &periodMatrixTransposed,
    const Eigen::VectorXcd &targetVector,
    const std::set<std::vector<int>> &claimedMatchings, bool skipClaimed) {
  // min over the relabelings of the target components of ||pdT c - ts||^2 (lstsq c).
  Eigen::BDCSVD<Eigen::MatrixXcd> periodSvd(
      periodMatrixTransposed, Eigen::ComputeThinU | Eigen::ComputeThinV);
  RelabelingMatch bestMatch;
  std::vector<int> relabeling(static_cast<std::size_t>(targetVector.size()));
  std::iota(relabeling.begin(), relabeling.end(), 0);
  do {
    if (skipClaimed && claimedMatchings.count(relabeling)) continue;
    const Eigen::VectorXcd relabeledTarget =
        relabeledTargetVector(targetVector, relabeling);
    const Eigen::VectorXcd leastSquaresCoefficients =
        periodSvd.solve(relabeledTarget);
    const double residual =
        (periodMatrixTransposed * leastSquaresCoefficients - relabeledTarget)
            .squaredNorm();
    if (!bestMatch.scored || residual < bestMatch.residual)
      bestMatch = {residual, relabeling, true};
  } while (std::next_permutation(relabeling.begin(), relabeling.end()));
  return bestMatch;
}

double MultiCobordism::residualOfTargetStateAgainstHarmonic(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState) {
  // No other register to collide with: an empty claim set excludes nothing, so this
  // is the unconstrained min over the relabelings (`r_state`, the reference read-out).
  std::set<std::vector<int>> claimedMatchings;
  return residualOfTargetStateAgainstHarmonicWithDistinctMatching(
      spacetime, registerDegree, targetState, claimedMatchings);
}

double MultiCobordism::residualOfTargetStateAgainstHarmonicWithDistinctMatching(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState,
    std::set<std::vector<int>> &claimedMatchings) {
  const Eigen::VectorXcd targetVector = targetStateVector(targetState);
  const double fullLeakResidual = targetVector.squaredNorm();  // zero-filled leak

  const auto bettiNumbers = betti(*spacetime);
  if (registerDegree < 0) //||  # TODO: why would we exit if registerDegree is higher than betti numbers?
    return fullLeakResidual;
  if (registerDegree >= static_cast<int>(bettiNumbers.size())) {
    CLOG(WARN_LEVEL, "register degree was higher than bettiNumbers!");
    return fullLeakResidual;
  }
  const int degreeBettiNumber = bettiNumbers[registerDegree];
  if (degreeBettiNumber == 0) return fullLeakResidual;

  const auto registerHoles =
      holesCarryingTheTarget(*spacetime, registerDegree, targetState.size());
  if (registerHoles.empty()) return fullLeakResidual;
  const Eigen::MatrixXcd periodMatrixTransposed =
      holePeriodMatrix(spacetime, registerDegree, degreeBettiNumber, registerHoles,
                       targetState.size());

  // The relabeling this register wins is withheld from the registers scored after it,
  // so no two of them are read against the same matching of components onto holes.
  RelabelingMatch match = bestRelabelingOfTarget(
      periodMatrixTransposed, targetVector, claimedMatchings, /*skipClaimed=*/true);
  if (!match.scored) {
    // Every relabeling is already claimed — more registers than the d! this target
    // admits. Restart the exclusion rather than return the empty minimum.
    claimedMatchings.clear();
    match = bestRelabelingOfTarget(periodMatrixTransposed, targetVector,
                                   claimedMatchings, /*skipClaimed=*/false);
  }
  claimedMatchings.insert(match.relabeling);
  return match.residual;
}

double MultiCobordism::residualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime) const {
  std::set<std::vector<int>> claimedMatchings;
  return residualForBoundaryBlockWithDistinctMatchings(boundaryBlock, spacetime,
                                                       claimedMatchings);
}

double MultiCobordism::residualForBoundaryBlockWithDistinctMatchings(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime,
    std::set<std::vector<int>> &claimedMatchings) const {
  auto blockSubcomplex = spacetime->subcomplexWithinVertexSet(
    boundaryBlock.vertices);
  double residual = 0.0;
  if (!blockSubcomplex)  // no complex to read: the target leaks in full, per degree
    return static_cast<double>(registerDegrees_.size()) *
           targetStateVector(boundaryBlock.target).squaredNorm();
  for (int registerDegree : registerDegrees_)
    residual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
        blockSubcomplex, registerDegree, boundaryBlock.target, claimedMatchings);
  return residual;
}

double MultiCobordism::rU(const std::shared_ptr<Spacetime> &spacetime) const {
  // The cobordism residual. INPUTS are localized boundary sub-complexes (built near
  // a seed, held representable by these terms, not pinned) — each read off its own
  // region and weighted by inputResidualWeight_ so they are not out-competed by the
  // whole/output term.
  //
  // ONE claim set spans the whole evaluation: every register here is scored by the
  // same min-over-relabelings, so without it they all pick the same argmin matching
  // and the sum is smallest when the registers carry identical weights. The set
  // records each register's winning matching and withholds it from the ones after.
  std::set<std::vector<int>> claimedMatchings;
  double totalResidual = 0.0;
  for (const auto &inputBlock : inputBlocks_)
    totalResidual += inputResidualWeight_ *
                     residualForBoundaryBlockWithDistinctMatchings(inputBlock, spacetime,
                                                   claimedMatchings);
  if (outputTargets_.size() == 1) {
    // A SINGLE output is the whole cobordism's output boundary: as in the Python
    // reference it is "the harmonic of the entire structure", NEVER a pinned
    // region. Read it off the WHOLE complex so the bulk loop drives the whole to
    // carry it (the output EMERGES; it is not frozen by seedOutputs).
    for (int registerDegree : registerDegrees_)
      totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
          spacetime, registerDegree, outputTargets_.front(), claimedMatchings);
  } else {
    // Multiple outputs (e.g. a 2->2 recombination → diquark ⊔ antidiquark) live in
    // distinct regions: read each off its own constructed block. EMPTY outputTargets
    // is the supported nothing-pinned-downstream shape (#555): no output term at
    // all — rU is the weighted input residuals alone, and the whole's final state
    // emerges (read after the fact, e.g. ProtonIngredients' singlet diagnostic).
    for (const auto &outputBlock : outputBlocks_)
      totalResidual +=
          residualForBoundaryBlockWithDistinctMatchings(outputBlock, spacetime,
                                                        claimedMatchings);
    if (inputBlocks_.empty() && outputBlocks_.empty())  // bare objective, nothing built yet
      for (int registerDegree : registerDegrees_)
        for (const auto &outputTarget : outputTargets_)
          totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
              spacetime, registerDegree, outputTarget, claimedMatchings);
  }
  // The pre-topological register signal (#644): the period residuals above are
  // STEP functions in the topology — exactly flat until a register exists — so
  // they carry no register-seeking gradient at a seed. The near-kernel residual
  // is the same functional continued below the topological threshold, and it
  // saturates at 0 the moment b_k reaches the expected count (see the header).
  const std::size_t expectedRegisters = expectedRegisterCount();
  if (expectedRegisters > 0)
    for (int registerDegree : registerDegrees_)
      totalResidual += nearKernelResidual(spacetime, registerDegree,
                                          expectedRegisters);
  return totalResidual;
}

std::size_t MultiCobordism::expectedRegisterCount() const {
  std::size_t expected = 0;
  for (const auto &target : inputTargets_)
    expected = std::max(expected, target.size());
  for (const auto &target : outputTargets_)
    expected = std::max(expected, target.size());
  return expected;
}

double MultiCobordism::nearKernelResidual(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    std::size_t expectedRegisterCount) {
  if (expectedRegisterCount == 0) return 0.0;
  cobordism::HodgeLaplacian laplacian(spacetime);
  // COMBINATORIAL operator (unit weights): its kernel is exactly the topology
  // (dim ker = b_k), so the term counts registers the way emergent_holes does.
  // The metric version was gameable: stage 2 discovered it could tune the
  // causal structure to the null-crossing locus and manufacture spectral
  // near-kernels with NO holes (a probe ended 18/18 edges timelike, 0 holes,
  // colorR at floor). With unit weights there is nothing to tune — only
  // stage-1 topology moves can lower it, which is exactly whose job opening
  // registers is (#644).
  const std::vector<std::complex<double>> flat =
      laplacian.laplacian(registerDegree, /*metric=*/false);
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  // No k-cells at all: every expected register is missing — the worst case on
  // the normalized scale, 1 per missing dimension.
  if (n == 0) return static_cast<double>(expectedRegisterCount);
  Eigen::MatrixXcd L(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * n + j];
  // Singular values of the NON-normal signed operator: the smooth surrogate for
  // the eigenvalue magnitudes (they share the kernel exactly).
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(L);
  const Eigen::VectorXd sigma = svd.singularValues();  // descending
  double total = 0.0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i) total += sigma[i] * sigma[i];
  // L identically zero: every mode is kernel — nothing left to open.
  if (total <= 0.0) return 0.0;
  const std::size_t m = std::min(expectedRegisterCount, n);
  double smallest = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double s = sigma[static_cast<Eigen::Index>(n - 1 - i)];
    smallest += s * s;
  }
  // n * (smallest m) / (all): scale-invariant (L is degree −1 in l^2, so a raw
  // spectral sum is a conformal-inflation descent channel; the ratio is degree
  // 0). Missing dimensions count 1 each — the generic-mode value.
  return static_cast<double>(n) * smallest / total +
         static_cast<double>(expectedRegisterCount - m);
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
    cellVertexTuples.push_back(topSimplex->topTuple());
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengthsByEdge;
  for (const auto *edge : spacetime.getEdgeList()->toVector())
    lengthsByEdge[edgeKey(edge)] = edge->getLength();  // verbatim, branch-exact
  return {std::move(cellVertexTuples), std::move(lengthsByEdge)};
}

MultiCobordism::Snapshot MultiCobordism::snapshot() const {
  return snapshotOf(*spacetime_);
}

std::shared_ptr<Spacetime> MultiCobordism::build(
    const Snapshot &complexSnapshot) const {
  auto rebuiltSpacetime = Spacetime::fromCells(spacetime_->getDimensions(),
                                               complexSnapshot.first, 1.0, 0.0);
  for (auto *edge : rebuiltSpacetime->getEdgeList()->toVector()) {
    const auto savedEntry = complexSnapshot.second.find(edgeKey(edge));
    if (savedEntry != complexSnapshot.second.end())
      edge->setLength(savedEntry->second);  // verbatim, branch-exact
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
    topCellTuples.push_back(topSimplex->topTuple());
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
  CLOG(INFO_LEVEL, "Applying a ", moveKind, " move.");
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
        edge->setLength(std::sqrt(-(edge->getLength() * edge->getLength())));
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
    for (auto vertexId : topSimplex->topTuple()) liveVertexIds.insert(vertexId);
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
    candidateCellSet.insert(topSimplex->topTuple());
  std::vector<std::vector<std::uint64_t>> touchedCells;
  for (const auto &cell : baseCellSet)
    if (!candidateCellSet.count(cell)) touchedCells.push_back(cell);
  for (const auto &cell : candidateCellSet)
    if (!baseCellSet.count(cell)) touchedCells.push_back(cell);

  // The touched-cell diff alone is NOT the whole affected set (#633): a
  // flip_disposition changes an edge's l^2 SIGN and changes no cells at all, so the
  // diff comes back empty and the geometry term would be scored as exactly 0 --
  // while flipping an edge between spacelike and timelike changes the deficit angle
  // of every hinge on it. So diff the edge l^2 values too, and pull in the top cells
  // incident to any edge that moved. Move-agnostic on purpose: this also covers
  // cone_in_timelike's apex edges and any future move that perturbs geometry without
  // changing cells, rather than special-casing a move kind.
  //
  // Widening is safe: Delta||grad S||^2 = after - before is exact over any FIXED
  // SUPERSET of the truly-affected edges, because every edge outside the set keeps
  // its gradient and cancels. A superset costs compute, never correctness.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> baseLengths;
  for (const auto *edge : spacetime_->getEdgeList()->toVector())
    baseLengths[edgeKey(edge)] = edge->getLength();
  std::set<std::pair<std::uint64_t, std::uint64_t>> movedEdges;
  for (const auto *edge : candidateSpacetime->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    const auto found = baseLengths.find(key);
    if (found == baseLengths.end() ||
        found->second != edge->getLength())
      movedEdges.insert(key);
    if (found != baseLengths.end()) baseLengths.erase(found);
  }
  for (const auto &leftover : baseLengths)  // in base, absent from candidate
    movedEdges.insert(leftover.first);
  if (!movedEdges.empty()) {
    std::set<std::vector<std::uint64_t>> incidentCells;
    const auto collectIncident = [&](const Spacetime &spacetime) {
      for (const auto &topSimplex : spacetime.getTopSimplices()) {
        auto cellVertexIds = topSimplex->topTuple();
        for (const auto &moved : movedEdges) {
          const bool cellHoldsBothEndpoints =
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.first) != cellVertexIds.end() &&
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.second) != cellVertexIds.end();
          if (cellHoldsBothEndpoints) {
            incidentCells.insert(std::move(cellVertexIds));
            break;
          }
        }
      }
    };
    collectIncident(*spacetime_);
    collectIncident(*candidateSpacetime);
    for (const auto &cell : incidentCells)
      if (std::find(touchedCells.begin(), touchedCells.end(), cell) ==
          touchedCells.end())
        touchedCells.push_back(cell);
  }

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
    baseCellSet.insert(topSimplex->topTuple());
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

void MultiCobordism::preconeCells(int count) {
  // Each cone-in cones a fresh apex onto a random codim-1 facet (a top cell with one
  // vertex dropped) and is committed only through applyMoveSpecification's
  // dualComplexValid gate — the same gated primitive the stage-1 draw uses, so the
  // pre-growth is sound (nothing inserted by fiat). On the single-Δ⁴ seed (a 4-ball)
  // a cone-in over a boundary facet is valid, so this enlarges the 4-ball; a draw
  // onto an already-saturated interior facet is rejected by the gate and retried.
  constexpr int kAttemptsPerCone = 20;  // gated tries before giving up on one cone
  for (int conedSoFar = 0; conedSoFar < count; ++conedSoFar) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime_->getTopSimplices())
      topCellTuples.push_back(topSimplex->topTuple());
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
      auto cellVertexIds = topSimplex->topTuple();
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
                                                 bool growBoundaries) {
  std::vector<double> objectiveTrace = {objective()};
  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex)
    if (!stage1Update(nCandidateMoves, growBoundaries, objectiveTrace)) break;
  return objectiveTrace;
}

bool MultiCobordism::stage1Update(int nCandidateMoves, bool growBoundaries,
                                  std::vector<double> &objectiveTrace) {
  // The register is "carried" (converged) once the summed r_U is essentially zero.
  constexpr double kRegisterCarriedTolerance = 1e-3;
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
    // An F-lowering surgery move: progress.
    objectiveTrace.push_back(objectiveTrace.back() + objectiveDelta);
    return true;
  }
  // No candidate in this batch lowered the objective. If the register is already
  // carried, that IS convergence — halt. Otherwise keep drawing: a batch is a
  // random sample, so one miss is not proof no improving move exists — under the
  // default disposition-enabled draw (#632) the seed always has an F-lowering
  // move — and the next iteration redraws fresh candidates. `maxSteps` bounds
  // the retries.
  return rU(spacetime_) >= kRegisterCarriedTolerance;
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
      auto cellVertexIds = topSimplex->topTuple();
      if (std::find(cellVertexIds.begin(), cellVertexIds.end(), seedVertexId) !=
          cellVertexIds.end())
        regionVertexIds.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    destinationBlocks.push_back(BoundaryBlock{regionVertexIds, targets[blockIndex]});
  }
}

std::vector<double> MultiCobordism::runStage2(double beta, int maxIters,
                                                 double alpha0, double relTol) {
  std::vector<double> objectiveTrace = {beta * reggeActionGradient(spacetime_) +
                                        gamma_ * rU(spacetime_)};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex)
    if (!stage2Update(beta, relTol, objectiveTrace, stepScale)) break;
  return objectiveTrace;
}

std::vector<double> MultiCobordism::run(int maxIters, int nCandidateMoves,
                                        bool growBoundaries, double beta,
                                        double alpha0, double relTol) {
  std::vector<double> objectiveTrace = {objective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex) {
    // Both updates run every iteration, so the optimizer takes whichever kind of
    // progress is available — combinatorial, geometric, or both. Neither stall is
    // final on its own: a stage-2 stationary point can be reopened by the next
    // stage-1 topology change (stage2Update re-reads the edge list each call) and
    // a stalled stage-1 search can be unstuck by relaxed geometry, so halt only
    // when BOTH halves stall within the same iteration.
    const bool combinatorialProgressing =
        stage1Update(nCandidateMoves, growBoundaries, objectiveTrace);

    bool geometricConverging = false;
    for (int i=0; i<maxIters; i++) {
      geometricConverging = stage2Update(beta, relTol, objectiveTrace, stepScale);
      if (!geometricConverging) break;
    }

    if (!combinatorialProgressing && !geometricConverging) break;
  }
  return objectiveTrace;
}

bool MultiCobordism::stage2Update(double beta, double relTol,
                                  std::vector<double> &objectiveTrace,
                                  double &stepScale) {
  // Reset-then-set: the flag reports THIS call's outcome, so in the combined
  // drive (`run`) it reflects the most recent geometric update instead of
  // latching true after a stationary point a later topology change reopened.
  // (`runStage2` is unaffected: there a true flag breaks its loop immediately.)
  lastStage2Stationary_ = false;
  // Within one `runStage2` call the topology is fixed (only edge lengths move), so
  // re-reading the edge list here is free; in the combined drive (`run`) it is what
  // picks up the edges a stage-1 move just created or removed.
  const auto &edges = spacetime_->getEdgeList()->toVector();
  const std::size_t edgeCount = edges.size();
  auto fullObjective = [&]() {
    return beta * reggeActionGradient(spacetime_) + gamma_ * rU(spacetime_);
  };
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

  // Descent direction allows descent in the complex (timelike) plane
  const Eigen::VectorXcd descentDirection = beta * 2.0 * (hessianMatrix.conjugate() * gradientVector);

  Eigen::VectorXcd lengths(edgeCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
    lengths(edgeIndex) =
        edges[edgeIndex]->getLength();
  const double currentObjective = objectiveTrace.back();
  // Relative stationarity: accept a step only when it lowers F by more than relTol
  // scaled by the current magnitude (an absolute floor of relTol when |F| < 1). The
  // old absolute convergenceTolerance_ accepted ~1e-11 *relative* steps for F ~ 100
  // — the rounding floor; this scales the threshold with the objective instead.
  const double improvementThreshold =
      relTol * std::max(std::abs(currentObjective), 1.0);
  auto trialStepScale = complexd(stepScale, stepScale);
  // Put the geometry back the way this call found it. Both early exits use it: the
  // line search that never beat the threshold, and an objective evaluation that threw
  // — a trial the caller never accepted must not survive as the complex's geometry.
  const auto restoreEdgeLengths = [&]() {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edges[edgeIndex]->setLength(lengths(edgeIndex));
  };
  bool objectiveImproved = false;
  try {
    for (int lineSearchIndex = 0; lineSearchIndex < 24; ++lineSearchIndex) {
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
        // The trial is UNBOUNDED on the real axis — fully Lorentzian, no
        // clamp, no causal guard (semantics: runStage2 in MultiCobordism.h).
        // Spacelike, timelike, and lightlike trials are all admissible, and
        // every trial is constructed EXACTLY real, so Im l^2 == 0 holds for
        // all time by construction — no backoff, no projection (#589).
        // TODO: setSquaredLength takes a complex number. the SQUARED length can only be real. We need to fix this to use
        //   imaginary length; we're losing detail on the real/imaginary split.
        // edges[edgeIndex]->setLength(std::sqrt(complexd(
            // squaredLengths(edgeIndex) - trialStepScale * descentDirection(edgeIndex),
            // 0.0)));
        edges[edgeIndex]->setLength(complexd(lengths(edgeIndex) - trialStepScale * descentDirection(edgeIndex)));
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
  } catch (...) {
    // The error still propagates loudly — it just does not take the geometry with
    // it. The throw comes from a TRIAL the line search had not accepted, so the
    // complex the caller still holds must be the one it had on entry, not a
    // half-applied step everything downstream would then read.
    restoreEdgeLengths();
    throw;
  }
  if (!objectiveImproved) {
    restoreEdgeLengths();
    lastStage2Stationary_ = true;  // no line-search step beat the relative threshold
    return false;
  }
  return true;
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
      cells.push_back(simplex->topTuple());
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
                               double stage2Beta, int stage2MaxIters,
                               double stage2Alpha0,
                               HolePlacementStrategy holePlacementStrategy) {
  switch (action) {
    case BuildAction::Grow:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/true);
      break;
    case BuildAction::Evolve:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/false);
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
