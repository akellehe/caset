// InteractionSimulation — implementation.
//
// This file holds the construction, the initial Poisson-Delaunay layer,
// the frontier / N₊ / N₋ bookkeeping, the simplicial side of the
// interact / un-interact moves, the Metropolis driving loop, and the
// action accounting. Every quantum-state touch is routed through the
// applyInteractionMPS / revertInteractionMPS / buildGroundState helpers;
// those carry the KAK / mediated-unitary / ITensor site-insertion work
// and are stubbed at the bottom of this file until that machinery lands.

#include "quantum/interaction_simulation.hpp"

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Metric.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tessera::quantum {

namespace {

// ℓ = -log(I / I_max), normalised so ℓ ≥ 0; +inf when I is below the floor.
// I_max = 2 ln 2 is the algebraic maximum MI between two single sites.
constexpr double kIMax = 2.0 * 0.6931471805599453;

double edgeLengthFromMI(double mi, double epsilon) {
    const double x = std::max(mi, 0.0) / kIMax;
    if (x < epsilon) return std::numeric_limits<double>::infinity();
    return -std::log(x);
}

// CDT disposition of a cell edge: same time slice → spacelike (ℓ² > 0),
// across slices → timelike (ℓ² < 0).
double signedSquaredLength(double length, bool spacelike) {
    const double sq = length * length;
    return spacelike ? sq : -sq;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────

InteractionSimulation::InteractionSimulation(InteractionConfig config)
    : config_(std::move(config)),
      sites_(),
      psi_(),
      rng_(config_.seed) {
    if (config_.params.N < 2)
        throw std::invalid_argument("InteractionConfig.params.N must be >= 2");
    if (config_.delaunayEdges.empty())
        throw std::invalid_argument(
            "InteractionConfig.delaunayEdges must be non-empty "
            "(the initial Poisson-Delaunay layer connectivity)");
    for (auto const& [i, j] : config_.delaunayEdges) {
        if (i < 0 || j < 0 || i >= config_.params.N || j >= config_.params.N
            || i == j)
            throw std::invalid_argument(
                "InteractionConfig.delaunayEdges has an out-of-range or "
                "degenerate site-index pair");
    }
    if (config_.epsilonI < 0.0)
        throw std::invalid_argument("InteractionConfig.epsilonI must be >= 0");

    // Coordinate-free Lorentzian Regge spacetime — edge lengths are set
    // explicitly from mutual information, not derived from coordinates.
    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Lorentzian));
    spacetime_ = std::make_shared<Spacetime>(
        metric, SpacetimeType::REGGE,
        /*alpha=*/1.0, /*a=*/1.0, Foliation::NONE,
        /*topology=*/std::nullopt);

    buildInitialLayer();
}

InteractionSimulation::~InteractionSimulation() = default;

void InteractionSimulation::buildInitialLayer() {
    const int n = config_.params.N;

    // 1. The N initial systems are vertices at time slice 0.
    vertexOfSite_.assign(static_cast<std::size_t>(n), nullptr);
    for (int s = 0; s < n; ++s) {
        VertexPtr v = spacetime_->createVertex(
            static_cast<std::uint64_t>(s), std::vector<double>{0.0});
        vertexOfSite_[static_cast<std::size_t>(s)] = v;
        siteOfVertex_[v] = s;
        frontier_.push_back(v);
    }

    // 2. The Schwinger DMRG ground state on those N sites.
    buildGroundState();

    // 3. Delaunay edges of the initial layer, lengthed by the ground-state
    //    site-site mutual information. All initial edges are frozen-once-
    //    consumed spatial edges; here they seed eligibleEdges_.
    for (auto const& [i, j] : config_.delaunayEdges) {
        VertexPtr a = vertexOfSite_[static_cast<std::size_t>(i)];
        VertexPtr b = vertexOfSite_[static_cast<std::size_t>(j)];
        // MI from the ground state is filled in once the MPS machinery is
        // wired in; until then the edge is created with a unit spacelike
        // length so the simplicial bookkeeping is exercisable.
        const double mi = 0.0;  // TODO: groundStateMI(i, j) from the MPS
        const double len = (mi > 0.0)
                               ? edgeLengthFromMI(mi, config_.epsilonI)
                               : 1.0;
        EdgePtr e = spacetime_->createEdge(
            a, b, signedSquaredLength(len, /*spacelike=*/true));
        (void)e;
        const auto key = std::minmax(a->getId(), b->getId());
        frozenEdgeLength_[{key.first, key.second}] = len;
    }

    rebuildMoveTables();
}

// ─────────────────────────────────────────────────────────────────────────
// Frontier / move-table bookkeeping
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::isFrontier(VertexPtr v) noexcept {
    return v != nullptr && v->getOutEdges().empty();
}

void InteractionSimulation::rebuildMoveTables() {
    // Frontier: vertices with no out-edges.
    frontier_.clear();
    for (auto const& [v, site] : siteOfVertex_) {
        (void)site;
        if (isFrontier(v)) frontier_.push_back(v);
    }

    // N₊: spatial edges with both endpoints on the frontier. A "spatial
    // edge" joins two systems on the same time slice. We scan the edge
    // list once.
    eligibleEdges_.clear();
    for (EdgePtr e : spacetime_->getEdgeList()->toVector()) {
        VertexPtr a = e->getSource();
        VertexPtr b = e->getTarget();
        if (!isFrontier(a) || !isFrontier(b)) continue;
        if (a->getTime() != b->getTime()) continue;  // spatial only
        eligibleEdges_.emplace_back(a, b);
    }

    // N₋: leaf cells — (2,3) top-simplices whose three later-slice
    // products are all still on the frontier.
    leafCells_.clear();
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 5) continue;  // (2,3) cell
        bool leaf = true;
        for (VertexPtr v : s->getVertices()) {
            // The three products sit on the later slice; the two parents
            // on the earlier one. A leaf cell is one whose products have
            // not themselves interacted.
            if (!v->getOutEdges().empty() && !isFrontier(v)) {
                // parent — fine
            }
        }
        // A cell is a leaf iff exactly its two parents are frozen and its
        // three products are on the frontier.
        int frontierProducts = 0;
        for (VertexPtr v : s->getVertices())
            if (isFrontier(v)) ++frontierProducts;
        leaf = (frontierProducts == 3);
        if (leaf) leafCells_.push_back(s);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Moves
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::interact() {
    ++interactAttempts_;
    if (eligibleEdges_.empty()) return false;
    if (config_.targetInteractions != 0
        && interactionCount_ >= config_.targetInteractions
        && leafCells_.empty())
        return false;

    // Propose: a uniformly-random eligible frontier spatial edge.
    std::uniform_int_distribution<std::size_t> pick(0, eligibleEdges_.size() - 1);
    auto [x, y] = eligibleEdges_[pick(rng_)];

    const std::size_t nPlusBefore = eligibleEdges_.size();

    // Tentatively apply the interaction to the MPS — creates the AB site,
    // evolves, and reports the ten new-edge mutual informations.
    InteractionMIs mis = applyInteractionMPS(x, y);

    // Build the (2,3) cell {X, Y, X', AB, Y'} on the next time slice.
    const double tNext = x->getTime() + 1.0;
    VertexPtr xp = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});
    VertexPtr ab = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});
    VertexPtr yp = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});

    // Edge lengths from the reported MIs. Local labels: 0=X 1=Y 2=X' 3=AB
    // 4=Y'. Same-slice pairs are spacelike, cross-slice timelike.
    auto vertexOfLabel = [&](int l) -> VertexPtr {
        switch (l) {
            case 0: return x;
            case 1: return y;
            case 2: return xp;
            case 3: return ab;
            default: return yp;
        }
    };
    auto sliceOfLabel = [](int l) { return l < 2 ? 0 : 1; };

    for (auto const& [pairKey, mi] : mis.edgeMI) {
        const int la = pairKey.first, lb = pairKey.second;
        VertexPtr va = vertexOfLabel(la);
        VertexPtr vb = vertexOfLabel(lb);
        const bool spacelike = sliceOfLabel(la) == sliceOfLabel(lb);
        const double len = edgeLengthFromMI(mi, config_.epsilonI);
        if (!std::isfinite(len)) {
            // Disconnected proposal — reject and undo the MPS step.
            revertInteractionMPS();
            return false;
        }
        EdgePtr e = spacetime_->createEdge(
            va, vb, signedSquaredLength(len, spacelike));
        (void)e;
    }

    auto [cell, created] =
        spacetime_->createSimplex(VertexPtrs{x, y, xp, ab, yp});
    (void)created;

    // ΔS: the new cell's hinge contributions. (Pre-existing hinges on the
    // {X,Y} edge would also shift, but in this construction X,Y freeze on
    // interaction so {X,Y} bounds only this one cell.)
    double deltaS = 0.0;
    std::vector<std::pair<SimplexPtr, double>> newHinges;
    for (SimplexPtr facet : cell->getFacets()) {
        for (SimplexPtr hinge : facet->getFacets()) {
            if (hinge->getVertices().size() != 3) continue;
            const double contrib = hinge->area() * hinge->deficitAngle();
            newHinges.emplace_back(hinge, contrib);
            deltaS += contrib;
        }
    }

    // Metropolis: after the move, N₋ gains this leaf cell; N₊ loses every
    // edge incident to X or Y and gains {X',AB},{AB,Y'}.
    const std::size_t nMinusAfter = leafCells_.size() + 1;
    const double logPrefactor =
        std::log(static_cast<double>(nPlusBefore))
        - std::log(static_cast<double>(nMinusAfter));

    if (!accept(deltaS, logPrefactor)) {
        spacetime_->removeSimplex(cell);
        revertInteractionMPS();
        return false;
    }

    // Commit: freeze X, Y; promote X', AB, Y'; update tables.
    for (auto const& [hinge, contrib] : newHinges)
        hingeAction_[hinge] = contrib;
    const int siteX = siteOfVertex_.count(x) ? siteOfVertex_[x] : -1;
    (void)siteX;
    siteOfVertex_[xp] = static_cast<int>(vertexOfSite_.size());
    vertexOfSite_.push_back(xp);
    siteOfVertex_[ab] = static_cast<int>(vertexOfSite_.size());
    vertexOfSite_.push_back(ab);
    siteOfVertex_[yp] = static_cast<int>(vertexOfSite_.size());
    vertexOfSite_.push_back(yp);

    ++interactionCount_;
    ++interactAccepted_;
    rebuildMoveTables();
    return true;
}

bool InteractionSimulation::unInteract() {
    ++unInteractAttempts_;
    if (leafCells_.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, leafCells_.size() - 1);
    SimplexPtr cell = leafCells_[pick(rng_)];

    const std::size_t nMinusBefore = leafCells_.size();

    // ΔS of removing the cell = -(its hinge contributions).
    double deltaS = 0.0;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets()) {
            if (hinge->getVertices().size() != 3) continue;
            auto it = hingeAction_.find(hinge);
            if (it != hingeAction_.end()) deltaS -= it->second;
        }

    // After removal, N₊ gains the parent edge; N₋ loses this cell.
    const std::size_t nPlusAfter = eligibleEdges_.size() + 1;
    const double logPrefactor =
        std::log(static_cast<double>(nMinusBefore))
        - std::log(static_cast<double>(nPlusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Commit: drop the cell's product vertices, invert the MPS step.
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets())
            hingeAction_.erase(hinge);
    spacetime_->removeSimplex(cell);
    revertInteractionMPS();

    --interactionCount_;
    ++unInteractAccepted_;
    rebuildMoveTables();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Driving loop
// ─────────────────────────────────────────────────────────────────────────

int InteractionSimulation::sweep() {
    const std::size_t nMoves =
        std::max<std::size_t>(1, eligibleEdges_.size() + leafCells_.size());
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    int accepted = 0;
    for (std::size_t k = 0; k < nMoves; ++k) {
        const bool doInteract =
            leafCells_.empty() ? true
            : eligibleEdges_.empty() ? false
            : coin(rng_) < 0.5;
        if (doInteract ? interact() : unInteract()) ++accepted;
    }
    return accepted;
}

void InteractionSimulation::tune(std::function<void(int, int)> progress) {
    // Initial-condition phase: grow toward targetInteractions with
    // interact moves only.
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
        const double denom = std::abs(prevAction) > 1e-12
                                 ? std::abs(prevAction) : 1.0;
        if (std::abs(action - prevAction) / denom < 0.01 && s > 5) break;
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

std::vector<double> InteractionSimulation::getDeficitAngleDistribution() const {
    std::vector<double> out;
    out.reserve(hingeAction_.size());
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
        int slice = 0;
        for (VertexPtr v : s->getVertices())
            slice = std::min(slice, static_cast<int>(v->getTime()));
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
    return {
        {"interact", rate(interactAccepted_, interactAttempts_)},
        {"unInteract", rate(unInteractAccepted_, unInteractAttempts_)},
    };
}

bool InteractionSimulation::accept(double deltaS, double logPrefactor) {
    const double exponent = -config_.beta * deltaS + logPrefactor;
    if (exponent >= 0.0) return true;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < std::exp(exponent);
}

// ─────────────────────────────────────────────────────────────────────────
// MPS-dependent helpers — stubbed until the KAK / mediated-unitary /
// ITensor site-insertion machinery is wired in. interact()/unInteract()
// above are written against this interface so they go live unchanged.
// ─────────────────────────────────────────────────────────────────────────

void InteractionSimulation::buildGroundState() {
    // DMRG the Schwinger ground state of the N-site initial layer. The
    // SiteSet is built without QN conservation: the interaction unitaries
    // applied later need not conserve total Sz, so the MPS must carry no
    // QN block structure. DMRG from a Néel state still lands in the
    // charge-neutral sector — the Schwinger H does not mix sectors.
    SchwingerHamiltonian ham(config_.params);
    auto sm = ham.mpo(/*conserveQns=*/false);
    sites_ = sm.sites;

    auto init = itensor::InitState(sites_);
    for (int i = 1; i <= config_.params.N; ++i)
        init.set(i, (i % 2 == 1) ? "Up" : "Dn");
    auto psi0 = itensor::MPS(init);

    auto sweeps = itensor::Sweeps(config_.dmrgNSweeps);
    const int b = config_.dmrgMaxBondDim;
    sweeps.maxdim() = std::min(20, b), std::min(40, b), std::min(80, b),
                      b, b;
    sweeps.cutoff() = config_.dmrgCutoff;
    sweeps.niter() = config_.dmrgKrylovDim;
    sweeps.noise() = 1e-7, 1e-8, 0.0;

    auto [energy, psi] = itensor::dmrg(
        sm.H, psi0, sweeps, itensor::Args("Silent", config_.quiet));
    (void)energy;
    psi_ = psi;
}

InteractionSimulation::InteractionMIs
InteractionSimulation::applyInteractionMPS(VertexPtr x, VertexPtr y) {
    (void)x;
    (void)y;
    // TODO: create the AB site in the MPS, build exp(-i H_XY dt),
    // KAK-decompose, route the Cartan core through AB, evolve psi_, and
    // read back the ten new-edge reduced-density-matrix MIs.
    InteractionMIs mis;
    // Placeholder unit MIs so the simplicial path is exercisable: ten
    // edges of the (2,3) cell, local labels 0..4.
    static const std::pair<int, int> kEdges[10] = {
        {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2},
        {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    for (auto const& e : kEdges) mis.edgeMI[e] = 0.5;
    return mis;
}

void InteractionSimulation::revertInteractionMPS() {
    // TODO: pop the most recent interaction off the MPS undo log
    // (inverse mediated unitary, drop the AB site).
}

// ─────────────────────────────────────────────────────────────────────────
// Transactional proposers — the InteractionMove class is fleshed out
// alongside the incremental-action work; it shares the undo-log machinery.
// Defined here (not just forward-declared) so unique_ptr can instantiate
// its deleter.
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
    (void)krylovDim;
    // TODO: build the MI-weighted graph Laplacian of the complex and read
    // off the heat-kernel spectral dimension.
    return std::vector<double>(sigmas.size(), 0.0);
}

} // namespace tessera::quantum
