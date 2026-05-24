// InteractionSimulation — implementation.
//
// See docs/source/interaction-history-monte-carlo.md for the charter.
// The construction is purely local: each interaction works on the
// participating systems' one-qubit density matrices, the joint state
// ρ_AB = U(ρ_X⊗ρ_Y)U†, and conservation-law bookkeeping. No MPS, no
// Choi state, no global wavefunction — the global correlation structure
// lives in the geometry (the accumulated edge lengths / Regge action).

#include "simulations/InteractionSimulation.h"

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/SimplexFilter.h"
#include "mesh/Vertex.h"
#include "observables/MIUnits.hpp"
#include "quantum/Holography.hpp"
#include "quantum/KoashiImoto.hpp"
#include "quantum/QuantumState.hpp"
#include "spacetime/Metric.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <stdexcept>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

namespace {

using cd = std::complex<double>;

// ─── REMOVED in #56 ─────────────────────────────────────────────────────
// The pre-refactor file held a forest of small Eigen helpers in this
// anonymous namespace, all serving the Schwinger / qudit / Choi code
// paths that are gone with the rest of the v0.1/v0.2 machinery:
//
//   vonNeumannEntropy / vonNeumannEntropyAny  — replaced by
//                          tessera::quantum::QuantumState::entropy
//                          (per-vertex marginal) and by
//                          tessera::quantum::mutualInformation
//                          (joint, when needed).
//   randomCorrelatedState  — pre-refactor initial layer; replaced by
//                          QuantumState::randomMixed.
//   partialTrace           — replaced by tessera::quantum::partialTrace{A,B}.
//   sortedPair / swapQubits / tensor2 / kron4   — used by the old
//                          jointStateFor / computeInteraction; both
//                          methods are gone.
//   schwingerTwoSiteU / I_UNIT   — Schwinger two-site Hamiltonian's
//                          unitary exponential. Gone with the
//                          Hamiltonian.
//   jointMutualInformation  — used by computeInteraction.
//   quditChargeOp / quditChargeConj / quditSpinSx / quditSpinSy —
//                          v0.2 charge / spin operators.
//   quditPairU / quditTraceOutA / quditTraceOutB / quditTensor /
//   quditSwap / quditJointMI   — v0.2 16-dim joint machinery.
//   edgeLengthFromMI       — replaced by the inline `dVrOf` lambda in
//                          interact() (and by InteractionSimulation::edgeLength
//                          for free-form edge queries), since the d_VR
//                          formula now uses I_max-relative MI and no
//                          longer needs the epsilon floor at this site.
//
// Everything below this point is *still in use* by the post-#56
// interact() / unInteract() / Regge-action plumbing.

// Wick-rotated to Euclidean signature: every edge contributes +ℓ². The
// spacelike/timelike flag on each cell edge is now informational only —
// the Regge action is real and well-defined everywhere with no complex
// projection step.

double signedSquaredLength(double length, bool /*spacelike*/) {
    return length * length;
}

// The ten edges of the (2,3) cell, with CDT disposition. Local labels:
// 0 = X, 1 = Y, 2 = X', 3 = AB, 4 = Y'. Same time slice -> spacelike.
struct CellEdge {
    int u, v;
    bool spacelike;
};
const CellEdge kCellEdges[10] = {
    {0, 1, true},    // X-Y     Delaunay edge (t=0 slice)
    {0, 2, false},   // X-X'    temporal residual
    {0, 3, false},   // X-AB    temporal, joint MI
    {1, 3, false},   // Y-AB    temporal, joint MI
    {1, 4, false},   // Y-Y'    temporal residual
    {2, 3, true},    // X'-AB   spatial (t=1 slice)
    {3, 4, true},    // Y'-AB   spatial
    {2, 4, true},    // X'-Y'   spatial
    {0, 4, false},   // X-Y'    closure (cross slice)
    {1, 2, false},   // Y-X'    closure
};

// Sum of the hinge contributions A_h ε_h of a single (2,3) cell with the
// given ten signed squared edge lengths. Built in a throwaway Spacetime
// so a rejected Metropolis proposal never touches the live complex.
double cellHingeAction(const double edgeSq[10]) {
    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Euclidean));
    Spacetime st(metric, SpacetimeType::REGGE, 1.0, 1.0, Foliation::NONE,
                 std::nullopt);
    VertexPtr v[5];
    v[0] = st.createVertex(0, std::vector<double>{0.0});
    v[1] = st.createVertex(1, std::vector<double>{0.0});
    v[2] = st.createVertex(2, std::vector<double>{1.0});
    v[3] = st.createVertex(3, std::vector<double>{1.0});
    v[4] = st.createVertex(4, std::vector<double>{1.0});
    for (int e = 0; e < 10; ++e)
        (void)st.createEdge(v[kCellEdges[e].u], v[kCellEdges[e].v],
                            edgeSq[e]);
    auto [cell, created] = st.createSimplex(
        VertexPtrs{v[0], v[1], v[2], v[3], v[4]});
    (void)created;
    double s = 0.0;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets())
            if (hinge->getVertices().size() == 3)
                s += hinge->area() * hinge->deficitAngle();
    return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────

InteractionSimulation::InteractionSimulation(InteractionConfig config)
    : config_(std::move(config)), rng_(config_.seed) {
    if (config_.nSystems < 2)
        throw std::invalid_argument(
            "InteractionConfig.nSystems must be >= 2");
    if (config_.delaunayEdges.empty())
        throw std::invalid_argument(
            "InteractionConfig.delaunayEdges must be non-empty");
    for (auto const& [i, j] : config_.delaunayEdges)
        if (i < 0 || j < 0 || i >= config_.nSystems
            || j >= config_.nSystems || i == j)
            throw std::invalid_argument(
                "InteractionConfig.delaunayEdges has an out-of-range or "
                "degenerate site-index pair");
    if (config_.epsInit <= 0.0)
        throw std::invalid_argument("InteractionConfig.epsInit must be > 0");
    if (config_.dimPerVertex <= 1)
        throw std::invalid_argument(
            "InteractionConfig.dimPerVertex must be > 1");

    // ─── REMOVED in #56 ────────────────────────────────────────────────
    // Schwinger H_XY validation:   if (config_.a <= 0.0) throw ...
    // Charge code-path reconcile:  useCharges <-> featureCharges aliasing,
    //                              auto-clear of dependent flags.
    // Schwinger unitary build:     interactionU_ = schwingerTwoSiteU(...)
    // Qudit-pair unitary build:    if (featureQuditBasis) quditInteractionU_ = ...
    // Choi state build:            if (featureChoiSigmaAB) quditChoiU_ = ...

    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Euclidean));
    spacetime_ = std::make_shared<Spacetime>(
        metric, SpacetimeType::REGGE, 1.0, 1.0, Foliation::NONE,
        std::nullopt);

    // Global information ceiling: N · epsInit (issue #56). Edge lengths
    // are d_VR = -log(I / iMax_); set once at construction and never
    // recomputed.
    iMax_ = static_cast<double>(config_.nSystems) * config_.epsInit;

    buildInitialLayer();
}

InteractionSimulation::~InteractionSimulation() = default;

void InteractionSimulation::buildInitialLayer() {
    const int n = config_.nSystems;
    const int dim = config_.dimPerVertex;

    // Recipe: sample marginals, then dial in MIs (issue #56).
    //
    // First pass — sample per-vertex QuantumStates with marginal entropy
    // close to epsInit nats. We use the existing randomMixed factory,
    // which lays down a Haar-conjugated diagonal at the target entropy.
    // This already satisfies the "each vertex carries epsInit nats" floor.
    //
    // Second pass (TODO — placeholder for now) — inject random pairwise
    // MIs by entangling Delaunay-adjacent pairs. The current pass leaves
    // the initial joint as product so the simulation starts with all
    // MIs near zero; interactions then generate correlation via KI
    // applied to the product joints (which on a first interaction is
    // structurally trivial — interactions emerge as the lattice grows
    // and Σ vertices accumulate).
    std::uniform_int_distribution<std::uint64_t> seedDist(
        1, std::numeric_limits<std::uint64_t>::max());

    std::vector<VertexPtr> verts(static_cast<std::size_t>(n));
    for (int s = 0; s < n; ++s) {
        VertexPtr v = spacetime_->createVertex(
            nextVertexId_++, std::vector<double>{0.0});
        verts[static_cast<std::size_t>(s)] = v;

        // Per-vertex QuantumState: random mixed state at entropy
        // ≈ epsInit (clamped to log(dim) since that's the max entropy
        // on a d-dim Hilbert space).
        const double maxS = std::log(static_cast<double>(dim));
        const double targetS = std::min(config_.epsInit, maxS);
        v->quantumState() = tessera::quantum::QuantumState::randomMixed(
            dim, targetS, seedDist(rng_));

        addToFrontier(v);
    }

    // Delaunay edges with d_VR length from MI between endpoint states.
    // For the product-joint initial layer above, all MIs are zero, so
    // every edge is at +∞ length. We still create the edges to seed
    // the geometry; the Regge solver / spectral pipeline can collapse
    // any infinite-length edges or treat them as disconnected per its
    // own contract.
    for (auto const& [i, j] : config_.delaunayEdges) {
        VertexPtr a = verts[static_cast<std::size_t>(i)];
        VertexPtr b = verts[static_cast<std::size_t>(j)];
        const double len = edgeLength(a, b);
        (void)spacetime_->createEdge(a, b, len);
    }
}

void InteractionSimulation::addToFrontier(VertexPtr v) {
    if (v == nullptr || frontierIdx_.count(v)) return;
    frontierIdx_[v] = frontier_.size();
    frontier_.push_back(v);
}

void InteractionSimulation::removeFromFrontier(VertexPtr v) {
    auto it = frontierIdx_.find(v);
    if (it == frontierIdx_.end()) return;
    const std::size_t pos  = it->second;
    const std::size_t last = frontier_.size() - 1;
    if (pos != last) {
        frontier_[pos] = frontier_[last];
        frontierIdx_[frontier_[pos]] = pos;
    }
    frontier_.pop_back();
    frontierIdx_.erase(it);
}

// ─── REMOVED in #56 (sign-bucketed frontier was for the annihilate move) ──
//   void addToSignBucket(VertexPtr)
//   void removeFromSignBucket(VertexPtr)

double InteractionSimulation::edgeLength(VertexPtr u, VertexPtr v) const {
    // d_VR(u, v) = -log(I(u:v) / iMax_), with I computed from the joint
    // ρ_u ⊗ ρ_v of the two endpoint states. The joint is product at
    // this layer (lineage rides on Σ vertices in the state graph, see
    // issue #56); when u or v carries Σ-like internal structure, KI
    // applied to the product recovers it.
    if (u == nullptr || v == nullptr) return 0.0;
    const auto& rhoU = u->quantumState().matrix();
    const auto& rhoV = v->quantumState().matrix();
    const int dU = static_cast<int>(rhoU.rows());
    const int dV = static_cast<int>(rhoV.rows());
    Eigen::MatrixXcd rhoUV(dU * dV, dU * dV);
    for (int i = 0; i < dU; ++i)
        for (int j = 0; j < dU; ++j)
            for (int a = 0; a < dV; ++a)
                for (int b = 0; b < dV; ++b)
                    rhoUV(i * dV + a, j * dV + b) = rhoU(i, j) * rhoV(a, b);
    const double I = tessera::quantum::mutualInformation(rhoUV, dU, dV);
    if (!(I > 0.0) || !(iMax_ > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::log(I / iMax_);
}

// ─────────────────────────────────────────────────────────────────────────
// Moves
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::interact() {
    ++interactAttempts_;
    if (frontier_.size() < 2) return false;
    if (config_.targetInteractions != 0
        && interactionCount_ >= config_.targetInteractions)
        return false;

    // Sample a uniform-random unordered pair of distinct frontier vertices.
    // REMOVED in #56: useCharges-aware re-sampling (same-sign-only constraint
    // belonged to the v0.1 annihilate move; reinstated against QuantumState
    // in charge-observables-v0.3).
    std::uniform_int_distribution<std::size_t> pick(
        0, frontier_.size() - 1);
    std::size_t i = pick(rng_);
    std::size_t j = pick(rng_);
    while (j == i) j = pick(rng_);
    VertexPtr x = frontier_[i];
    VertexPtr y = frontier_[j];

    const std::size_t nFrontier   = frontier_.size();
    const std::size_t nPlusBefore = nFrontier * (nFrontier - 1) / 2;

    // Build the joint \f$\rho_{XY} = \rho_X \otimes \rho_Y\f$ from the
    // two endpoint vertex states and KI-decompose it (issue #56). The
    // joint is a product at this layer — joint history is carried by
    // the Σ vertices already in the state graph, not by a separate
    // pairwise cache. When X or Y is itself a Σ vertex with internal
    // block-diagonal structure, KI sees that structure in the product
    // and propagates it forward.
    const Eigen::MatrixXcd& rhoX = x->quantumState().matrix();
    const Eigen::MatrixXcd& rhoY = y->quantumState().matrix();
    const int dX = static_cast<int>(rhoX.rows());
    const int dY = static_cast<int>(rhoY.rows());
    Eigen::MatrixXcd rhoXY(dX * dY, dX * dY);
    for (int a = 0; a < dX; ++a)
        for (int ap = 0; ap < dX; ++ap)
            for (int b = 0; b < dY; ++b)
                for (int bp = 0; bp < dY; ++bp)
                    rhoXY(a * dY + b, ap * dY + bp) =
                        rhoX(a, ap) * rhoY(b, bp);

    const tessera::quantum::KoashiImotoTolerances kiTol{
        config_.epsKiEigen, config_.epsKiCondState, config_.epsKiSvd};
    const auto ki = tessera::quantum::koashiImotoDecompose(
        rhoXY, dX, dY, kiTol);

    // Candidate child states. Label convention matches kCellEdges:
    //   0=X, 1=Y, 2=X' (aPrime), 3=Σ_{A,B} (sigma), 4=Y' (bPrime).
    const Eigen::MatrixXcd* rhoByLabel[5] = {
        &rhoX, &rhoY, &ki.aPrime, &ki.sigma, &ki.bPrime};

    // Edge length from MI between the two endpoint states. The MI is
    // computed on the product joint of the two states (consistent with
    // the simulation-wide "joint = product, lineage in Σ" rule). The
    // length stored on the edge is d_VR = -log(I / iMax_); when I = 0
    // we store IEEE +∞ (#56: dVrMax removed in favour of true infinity).
    auto miOf = [](const Eigen::MatrixXcd& r1,
                   const Eigen::MatrixXcd& r2) -> double {
        const int d1 = static_cast<int>(r1.rows());
        const int d2 = static_cast<int>(r2.rows());
        Eigen::MatrixXcd j(d1 * d2, d1 * d2);
        for (int a = 0; a < d1; ++a)
            for (int ap = 0; ap < d1; ++ap)
                for (int b = 0; b < d2; ++b)
                    for (int bp = 0; bp < d2; ++bp)
                        j(a * d2 + b, ap * d2 + bp) = r1(a, ap) * r2(b, bp);
        return tessera::quantum::mutualInformation(j, d1, d2);
    };
    auto dVrOf = [this](double mi) -> double {
        if (!(mi > 0.0) || !(iMax_ > 0.0))
            return std::numeric_limits<double>::infinity();
        return -std::log(mi / iMax_);
    };

    // The ten signed edge "lengths" of the proposed cell. Per #56 the
    // value stored in `Edge::squaredLength` is d_VR directly (the
    // field name keeps the legacy spelling — see issue #56 / CDT
    // follow-up). signedSquaredLength still applies the timelike-vs-
    // spacelike sign that the Regge action needs.
    double edgeSq[10];
    for (int e = 0; e < 10; ++e) {
        const CellEdge& ce = kCellEdges[e];
        const double mi  = miOf(*rhoByLabel[ce.u], *rhoByLabel[ce.v]);
        const double len = dVrOf(mi);
        edgeSq[e] = signedSquaredLength(len, ce.spacelike);
    }

    // ΔS evaluated in a throwaway complex — a rejected proposal never
    // touches the live geometry.
    const double deltaS = cellHingeAction(edgeSq);

    // Reverse-move denominator: number of leaf simplices AFTER this
    // interact. A frontier vertex's parent simplex (if any) is the
    // unique simplex it currently belongs to; once that vertex leaves
    // the frontier, the parent simplex is no longer a leaf. (#56
    // replaces producedByCell_ + consumedProductsOf_ + leafCellCount_
    // with the leafSimplices_ index queried directly.)
    auto parentOf = [](VertexPtr v) -> SimplexPtr {
        const auto& sims = v->getSimplices();
        return sims.empty() ? nullptr : sims.front();
    };
    const SimplexPtr sX = parentOf(x);
    const SimplexPtr sY = parentOf(y);
    std::size_t leavesAfter = leafSimplices_.size() + 1;  // the new cell
    if (sX != nullptr && leafSimplices_.count(sX)) --leavesAfter;
    if (sY != nullptr && sY != sX && leafSimplices_.count(sY))
        --leavesAfter;
    const std::size_t nMinusAfter = std::max<std::size_t>(leavesAfter, 1);
    const double logPrefactor =
        std::log(static_cast<double>(nPlusBefore))
        - std::log(static_cast<double>(nMinusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Accepted — build the (2,3) cell into the live complex. Products
    // are placed one time-step after the *later* of the two inputs so
    // every product comes strictly after both its parents (the
    // causal-set indexing of events).
    const double tNext = std::max(x->getTime(), y->getTime()) + 1.0;
    VertexPtr xp = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});
    VertexPtr ab = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});
    VertexPtr yp = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});

    // Install KI-decomposed states on the three children. The matrices
    // come out of KI already Hermitian, positive, and trace-1 (the
    // weight * core / weight * tail factorisation guarantees it), so
    // we skip the QuantumState validators that the public setMatrix
    // would re-run.
    xp->quantumState().setMatrixUnchecked(ki.aPrime);
    ab->quantumState().setMatrixUnchecked(ki.sigma);
    yp->quantumState().setMatrixUnchecked(ki.bPrime);

    VertexPtr label[5] = {x, y, xp, ab, yp};
    for (int e = 0; e < 10; ++e)
        (void)spacetime_->createEdge(label[kCellEdges[e].u],
                                     label[kCellEdges[e].v], edgeSq[e]);
    auto [cell, created] =
        spacetime_->createSimplex(VertexPtrs{x, y, xp, ab, yp});
    (void)created;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets())
            if (hinge->getVertices().size() == 3)
                hingeAction_[hinge] = hinge->area() * hinge->deficitAngle();

    // ─── REMOVED in #56 ──────────────────────────────────────────────
    //   stateOf_ / jointOf_ writes        — state lives on vertex.quantumState()
    //   producedByCell_, consumedByCell_  — derivable from simplex containment
    //   consumedProductsOf_, leafCellCount_ — replaced by leafSimplices_
    //   v0.2 qudit / Choi seeding (quditStateOf_, quditJointOf_, choiSigmaAbSet_)
    //   v0.1 charge inheritance (chargeOf_ updates)

    // Leaf-simplex bookkeeping: the new cell is a leaf (all three of
    // its children xp / ab / yp join the frontier in the next block);
    // sX and sY, if they were leaves, stop being leaves because x and
    // y are about to leave the frontier.
    leafSimplices_.insert(cell);
    if (sX != nullptr) leafSimplices_.erase(sX);
    if (sY != nullptr && sY != sX) leafSimplices_.erase(sY);

    // Incremental frontier update — O(1).
    removeFromFrontier(x);
    removeFromFrontier(y);
    addToFrontier(xp);
    addToFrontier(ab);
    addToFrontier(yp);

    ++interactionCount_;
    ++interactAccepted_;
    return true;
}

bool InteractionSimulation::unInteract() {
    ++unInteractAttempts_;
    // Eligible: only leaf simplices (#56 — the legacy "deep
    // un-interaction" of an arbitrary cell required the
    // producedByCell_ / consumedByCell_ DAG tables, which are removed
    // in this refactor along with the rest of the per-vertex
    // bookkeeping. Leaf-only is sufficient for ergodicity because
    // every interaction eventually becomes a leaf once its descendants
    // are themselves un-interacted, and the move pair (interact,
    // unInteract) is detail-balanced under the leaf-set Metropolis
    // prefactor below.)
    if (leafSimplices_.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(
        0, leafSimplices_.size() - 1);
    auto it = leafSimplices_.begin();
    std::advance(it, pick(rng_));
    const SimplexPtr cell = *it;

    // Classify the 5 vertices into 2 parents (earlier timestamp) and
    // 3 children (latest timestamp). The simplex stores its vertex
    // list in whatever order spacetime chose at creation; recovering
    // the parent/child split from timestamps is robust to that ordering.
    std::vector<VertexPtr> parents, children;
    double maxT = -std::numeric_limits<double>::infinity();
    for (VertexPtr v : cell->getVertices())
        if (v->getTime() > maxT) maxT = v->getTime();
    for (VertexPtr v : cell->getVertices()) {
        if (v->getTime() == maxT) children.push_back(v);
        else                      parents.push_back(v);
    }
    if (parents.size() != 2 || children.size() != 3) {
        // Defensive: a well-formed (2,3) cell always has 2-vs-3. A
        // mismatch indicates a corrupted simplex; refuse the move.
        return false;
    }

    // ΔS = -Σ A_h ε_h over this cell's hinges (no descendant cells
    // in the leaf-only design).
    double deltaS = 0.0;
    std::vector<SimplexPtr> hingesToErase;
    for (SimplexPtr facet : cell->getFacets()) {
        for (SimplexPtr h : facet->getFacets()) {
            if (h->getVertices().size() != 3) continue;
            auto hit = hingeAction_.find(h);
            if (hit != hingeAction_.end()) {
                deltaS -= hit->second;
                hingesToErase.push_back(h);
            }
        }
    }

    // Reverse-move Metropolis prefactor:
    //   N_- (eligible reverses now) = |leafSimplices_|
    //   N_+ (eligible forwards after) = C(|frontier| + 2 - 3, 2)
    //                                  = C(|frontier| - 1, 2)
    // (Lose 3 children from frontier, gain 2 parents back → net −1.)
    const std::size_t nMinusBefore   = leafSimplices_.size();
    const std::size_t nFrontierNow   = frontier_.size();
    const std::size_t nFrontierAfter =
        (nFrontierNow >= 1) ? nFrontierNow - 1 : 0;
    const std::size_t nPlusAfter =
        nFrontierAfter < 2
            ? 1
            : nFrontierAfter * (nFrontierAfter - 1) / 2;
    const double logPrefactor =
        std::log(static_cast<double>(nMinusBefore))
        - std::log(static_cast<double>(nPlusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Commit. Order matters: remove the simplex first so vertex /
    // edge removal in step 2 doesn't dangle simplex references.

    // 1. Drop the cell from the leaf set and from spacetime.
    leafSimplices_.erase(cell);
    spacetime_->removeSimplex(cell);

    // 2. Drop hingeAction entries for the now-removed hinges.
    for (SimplexPtr h : hingesToErase) hingeAction_.erase(h);

    // 3. Remove the 3 children from the frontier and from spacetime.
    //    Their QuantumStates are stored on the Vertex itself and go
    //    away when the Vertex is removed — no separate state cleanup.
    //    (#56: stateOf_, jointOf_, chargeOf_, producedByCell_ erasures
    //    are gone; per-vertex state lives on the vertex.)
    for (VertexPtr v : children) {
        removeFromFrontier(v);
        spacetime_->removeVertex(v);
    }

    // 4. Restore the 2 parents to the frontier. Their states were
    //    never mutated by interact (KI is decomposition, not
    //    transformation — issue #56), so nothing to restore.
    for (VertexPtr v : parents) addToFrontier(v);

    // 5. Leaf-set bookkeeping: each restored parent may have its own
    //    parent simplex S_v whose other two children are still on the
    //    frontier. If so, S_v is now a leaf again. Same parent/child
    //    classification via timestamp as in step 2.
    auto parentSimplex = [](VertexPtr v) -> SimplexPtr {
        const auto& sims = v->getSimplices();
        // After removeSimplex(cell), the cell isn't in v->simplices any
        // more, so the remaining simplex (if any) is v's birth simplex.
        return sims.empty() ? nullptr : sims.front();
    };
    auto isLeafSimplex = [this](SimplexPtr s) -> bool {
        if (s == nullptr) return false;
        double sMaxT = -std::numeric_limits<double>::infinity();
        for (VertexPtr v : s->getVertices())
            if (v->getTime() > sMaxT) sMaxT = v->getTime();
        int childCount = 0, onFront = 0;
        for (VertexPtr v : s->getVertices())
            if (v->getTime() == sMaxT) {
                ++childCount;
                if (onFrontier(v)) ++onFront;
            }
        return childCount == 3 && onFront == 3;
    };
    for (VertexPtr v : parents) {
        SimplexPtr sV = parentSimplex(v);
        if (sV != nullptr && isLeafSimplex(sV))
            leafSimplices_.insert(sV);
    }

    if (interactionCount_ > 0) --interactionCount_;
    ++unInteractAccepted_;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Driving loop
// ─────────────────────────────────────────────────────────────────────────

int InteractionSimulation::sweep() {
    // Move count scales with N₊ + N₋ — frontier-pair count plus the
    // total cell count (every cell is uninteractable under deep
    // truncation).
    const std::size_t nFront = frontier_.size();
    const std::size_t nPairs = nFront < 2 ? 0 : nFront * (nFront - 1) / 2;
    const std::size_t nMoves = std::max<std::size_t>(
        1, nPairs + interactionCount_);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    int accepted = 0;
    for (std::size_t k = 0; k < nMoves; ++k) {
        // Two-move sweep (post-#56): interact / unInteract with equal
        // proposal probability. The annihilate / pairCreate charge moves
        // are removed alongside the rest of the charge code paths and
        // will be reinstated against QuantumState in the
        // charge-observables-v0.3 follow-up.
        const bool doInteract =
            interactionCount_ == 0 ? true
            : nFront < 2           ? false
                                   : coin(rng_) < 0.5;
        if (doInteract ? interact() : unInteract()) ++accepted;
    }
    return accepted;
}

void InteractionSimulation::tune(std::function<void(int, int)> progress) {
    const std::size_t target = config_.targetInteractions;
    std::size_t guard = 0;
    const std::size_t guardMax = (target == 0) ? 0 : 100 * target;
    while (interactionCount_ < target && guard < guardMax) {
        interact();
        ++guard;
        if (progress)
            progress(static_cast<int>(interactionCount_),
                     static_cast<int>(target));
    }
}

void InteractionSimulation::thermalize() {
    if (config_.targetInteractions != 0) tune();

    double prevAction = computeAction();
    for (int s = 0; s < 1000; ++s) {
        sweep();
        const double action = computeAction();
        const double denom =
            std::abs(prevAction) > 1e-12 ? std::abs(prevAction) : 1.0;
        if (s > 5 && std::abs(action - prevAction) / denom < 0.01) break;
        prevAction = action;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────

double InteractionSimulation::computeAction() const {
    double s = 0.0;
    for (auto const& [hinge, contrib] : hingeAction_) {
        (void)hinge;
        s += contrib;
    }
    return s;
}

std::vector<double>
InteractionSimulation::getDeficitAngleDistribution() const {
    std::vector<double> out;
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 3) continue;
        if (s->getCofaces().empty()) continue;
        out.push_back(s->deficitAngle());
    }
    return out;
}

std::vector<int> InteractionSimulation::getVolumeProfile() const {
    std::vector<int> profile;
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 5) continue;
        double earliest = std::numeric_limits<double>::infinity();
        for (VertexPtr v : s->getVertices())
            earliest = std::min(earliest, v->getTime());
        const int slice = static_cast<int>(std::lround(earliest));
        if (static_cast<int>(profile.size()) <= slice)
            profile.resize(static_cast<std::size_t>(slice) + 1, 0);
        ++profile[static_cast<std::size_t>(slice)];
    }
    return profile;
}

std::map<std::string, double>
InteractionSimulation::getAcceptanceRates() const {
    auto rate = [](std::int64_t acc, std::int64_t att) {
        return att > 0 ? static_cast<double>(acc) / static_cast<double>(att)
                       : 0.0;
    };
    std::map<std::string, double> out{
        {"interact",   rate(interactAccepted_,   interactAttempts_)},
        {"unInteract", rate(unInteractAccepted_, unInteractAttempts_)},
    };
    // REMOVED in #56: annihilate / pairCreate counters no longer exist.
    return out;
}
bool InteractionSimulation::accept(double deltaS, double logPrefactor) {
    const double exponent = -config_.beta * deltaS + logPrefactor;
    if (exponent >= 0.0) return true;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < std::exp(exponent);
}

// ─────────────────────────────────────────────────────────────────────────
// Transactional proposers — the InteractionMove class is fleshed out
// alongside the incremental-action work; defined here (not just
// forward-declared) so unique_ptr can instantiate its deleter.
// ─────────────────────────────────────────────────────────────────────────

class InteractionMove {
  public:
    virtual ~InteractionMove() = default;
};

std::unique_ptr<InteractionMove> InteractionSimulation::proposeInteract() {
    return nullptr;  // TODO: transactional move object
}

std::unique_ptr<InteractionMove> InteractionSimulation::proposeUnInteract() {
    return nullptr;  // TODO: transactional move object
}

std::vector<double> InteractionSimulation::getSpectralDimension(
    const std::vector<double>& sigmas, int krylovDim) const {
    // The dimension we want is that of the simplicial complex of
    // completed (2,3) 4-simplices. We measure its 1-skeleton: systems
    // as nodes, the cells' edges MI-weighted. Bare initial-layer edges
    // that never joined a cell are part of the primal interaction
    // lattice and are excluded by the topK == 4 size filter.
    //
    // Delegates to the shared Spacetime → SpectralGraph path (issue #31)
    // so the holography and interaction-history pipelines share one
    // measurement code path. AllSimplexFilter matches the pre-#31
    // behavior (every 4-simplex passes); use of a stricter filter is
    // tracked in #38.
    return spacetime_->getSpectralDimensionOnSkeleton(
        sigmas, krylovDim, ::tessera::AllSimplexFilter{},
        /*topK=*/4, /*skeletonDim=*/1);
}

} // namespace tessera::simulations
