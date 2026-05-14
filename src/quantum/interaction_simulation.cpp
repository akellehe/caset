// InteractionSimulation — implementation.
//
// See docs/source/interaction-history-monte-carlo.md for the charter.
// The construction is purely local: each interaction works on the
// participating systems' one-qubit density matrices, the joint state
// ρ_AB = U(ρ_X⊗ρ_Y)U†, and conservation-law bookkeeping. No MPS, no
// Choi state, no global wavefunction — the global correlation structure
// lives in the geometry (the accumulated edge lengths / Regge action).

#include "quantum/interaction_simulation.hpp"

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "quantum/holography.hpp"
#include "spacetime/Metric.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace tessera::quantum {

namespace {

using cd = std::complex<double>;
constexpr cd I_UNIT{0.0, 1.0};

// Algebraic maximum mutual information between two single qubits.
const double kIMax = 2.0 * std::log(2.0);

// Von Neumann entropy S(ρ) = -Tr ρ log ρ, in nats.
double vonNeumannEntropy(Eigen::MatrixXcd const& rho) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    double s = 0.0;
    for (double lambda : es.eigenvalues())
        if (lambda > 1e-13) s -= lambda * std::log(lambda);
    return s;
}

// A randomized correlated mixed state on n qubits: ρ = M M† / Tr(M M†)
// with M a complex-Gaussian 2ⁿ × 2ⁿ matrix. Generic — every pair of
// qubits shares genuine mutual information.
Eigen::MatrixXcd randomCorrelatedState(int n, std::mt19937& rng) {
    const int dim = 1 << n;
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::MatrixXcd m(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            m(i, j) = cd(g(rng), g(rng));
    Eigen::MatrixXcd rho = m * m.adjoint();
    return rho / rho.trace().real();
}

// Reduced density matrix of an n-qubit state, keeping the qubits in
// `keep` (0-based; qubit 0 is the most significant bit). The kept qubits
// keep their listed order, so keep = {i, j} returns the (qubit i ⊗
// qubit j) joint state in the same ordering as tensor2.
Eigen::MatrixXcd partialTrace(Eigen::MatrixXcd const& rho, int n,
                              std::vector<int> const& keep) {
    std::vector<int> traced;
    for (int b = 0; b < n; ++b)
        if (std::find(keep.begin(), keep.end(), b) == keep.end())
            traced.push_back(b);
    const int k = static_cast<int>(keep.size());
    const int t = static_cast<int>(traced.size());
    const int dimK = 1 << k;
    const int dimT = 1 << t;
    auto fullIndex = [&](int kv, int tv) {
        int idx = 0;
        for (int i = 0; i < k; ++i)             // keep[0] = most significant
            if (kv & (1 << (k - 1 - i))) idx |= 1 << (n - 1 - keep[i]);
        for (int i = 0; i < t; ++i)
            if (tv & (1 << (t - 1 - i))) idx |= 1 << (n - 1 - traced[i]);
        return idx;
    };
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(dimK, dimK);
    for (int r = 0; r < dimK; ++r)
        for (int c = 0; c < dimK; ++c)
            for (int tv = 0; tv < dimT; ++tv)
                out(r, c) += rho(fullIndex(r, tv), fullIndex(c, tv));
    return out;
}

// Sorted vertex-pointer pair, the canonical key for jointOf_.
std::pair<VertexPtr, VertexPtr> sortedPair(VertexPtr a, VertexPtr b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

// Swap the two qubits of a two-qubit operator: (X ⊗ Y) -> (Y ⊗ X).
Eigen::Matrix4cd swapQubits(Eigen::Matrix4cd const& m) {
    Eigen::Matrix4cd out;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                for (int d = 0; d < 2; ++d)
                    out(2 * b + a, 2 * d + c) = m(2 * a + b, 2 * c + d);
    return out;
}

// Kronecker product of two one-qubit states, ordering (X ⊗ Y).
Eigen::Matrix4cd tensor2(SystemState const& a, SystemState const& b) {
    Eigen::Matrix4cd out;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    out(2 * i + k, 2 * j + l) = a(i, j) * b(k, l);
    return out;
}

// Partial traces of a two-qubit joint state.
SystemState traceOutSecond(Eigen::Matrix4cd const& rho) {  // -> X marginal
    SystemState out;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            out(i, j) = rho(2 * i + 0, 2 * j + 0) + rho(2 * i + 1, 2 * j + 1);
    return out;
}
SystemState traceOutFirst(Eigen::Matrix4cd const& rho) {  // -> Y marginal
    SystemState out;
    for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l)
            out(k, l) = rho(0 + k, 0 + l) + rho(2 + k, 2 + l);
    return out;
}

// The Schwinger two-site interaction unitary U = exp(-i H_XY dt).
// H_XY is the local two-site Hamiltonian — hopping plus the staggered
// mass term — built as a 4×4 dense matrix and exponentiated through a
// Hermitian eigendecomposition. (The electric term L_n² is non-local in
// the staggered formulation and is not part of a two-site fragment.)
Eigen::Matrix4cd schwingerTwoSiteU(double a, double m, double dt) {
    // Pauli matrices.
    Eigen::Matrix2cd X, Y, Z, Id;
    X << 0, 1, 1, 0;
    Y << 0, -I_UNIT, I_UNIT, 0;
    Z << 1, 0, 0, -1;
    Id << 1, 0, 0, 1;
    auto kron = [](Eigen::Matrix2cd const& p, Eigen::Matrix2cd const& q) {
        Eigen::Matrix4cd r;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int l = 0; l < 2; ++l)
                        r(2 * i + k, 2 * j + l) = p(i, j) * q(k, l);
        return r;
    };
    // H_hop = (1/(4a)) (XX + YY); H_m = (m/2)(-Z⊗I + I⊗Z).
    Eigen::Matrix4cd H =
        (1.0 / (4.0 * a)) * (kron(X, X) + kron(Y, Y))
        + (m / 2.0) * (-kron(Z, Id) + kron(Id, Z));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> es(H);
    Eigen::Vector4cd phase;
    for (int k = 0; k < 4; ++k)
        phase(k) = std::exp(-I_UNIT * dt * es.eigenvalues()(k));
    return es.eigenvectors() * phase.asDiagonal()
           * es.eigenvectors().adjoint();
}

// Mutual information of a two-qubit joint state, in nats.
double jointMutualInformation(Eigen::Matrix4cd const& rho) {
    const double sX = vonNeumannEntropy(traceOutSecond(rho));
    const double sY = vonNeumannEntropy(traceOutFirst(rho));
    const double sXY = vonNeumannEntropy(rho);
    return std::max(sX + sY - sXY, 0.0);
}

// ℓ = -log(I / I_max), normalised so ℓ ≥ 0; floored when I is tiny so
// uncorrelated systems are far apart but not infinitely so.
double edgeLengthFromMI(double mi, double epsilon) {
    const double x = std::max(mi, 0.0) / kIMax;
    if (x < epsilon) return -std::log(epsilon);
    return -std::log(x);
}

double signedSquaredLength(double length, bool spacelike) {
    const double sq = length * length;
    return spacelike ? sq : -sq;
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
        /*coordinateFree=*/true, Signature(4, SignatureType::Lorentzian));
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
    if (config_.a <= 0.0)
        throw std::invalid_argument("InteractionConfig.a must be > 0");

    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Lorentzian));
    spacetime_ = std::make_shared<Spacetime>(
        metric, SpacetimeType::REGGE, 1.0, 1.0, Foliation::NONE,
        std::nullopt);

    interactionU_ = schwingerTwoSiteU(config_.a, config_.m, config_.dt);

    buildInitialLayer();
}

InteractionSimulation::~InteractionSimulation() = default;

void InteractionSimulation::buildInitialLayer() {
    const int n = config_.nSystems;

    // The initial layer is one randomized *correlated* mixed state, so
    // Delaunay-adjacent systems share genuine mutual information.
    const Eigen::MatrixXcd layer = randomCorrelatedState(n, rng_);

    std::vector<VertexPtr> verts(static_cast<std::size_t>(n));
    for (int s = 0; s < n; ++s) {
        VertexPtr v = spacetime_->createVertex(
            static_cast<std::uint64_t>(s), std::vector<double>{0.0});
        verts[static_cast<std::size_t>(s)] = v;
        stateOf_[v] = partialTrace(layer, n, {s});  // one-qubit marginal
        frontier_.push_back(v);
    }

    // Delaunay edges, lengthed by the genuine pairwise mutual information
    // of the correlated layer. The joint pair-states seed jointOf_.
    for (auto const& [i, j] : config_.delaunayEdges) {
        VertexPtr a = verts[static_cast<std::size_t>(i)];
        VertexPtr b = verts[static_cast<std::size_t>(j)];
        const Eigen::Matrix4cd rhoAB = partialTrace(layer, n, {i, j});
        // Stored in (key.first ⊗ key.second) qubit order.
        const auto key = sortedPair(a, b);
        jointOf_[key] = (key.first == a) ? rhoAB : swapQubits(rhoAB);
        const double mi = jointMutualInformation(rhoAB);
        const double len = edgeLengthFromMI(mi, config_.epsilonI);
        (void)spacetime_->createEdge(
            a, b, signedSquaredLength(len, /*spacelike=*/true));
    }

    rebuildMoveTables();
}

// The joint state ρ_XY in (X ⊗ Y) qubit order — the stored correlated
// pair if X, Y share one, the uncorrelated product otherwise.
Eigen::Matrix4cd
InteractionSimulation::jointStateFor(VertexPtr x, VertexPtr y) const {
    auto it = jointOf_.find(sortedPair(x, y));
    if (it == jointOf_.end())
        return tensor2(stateOf_.at(x), stateOf_.at(y));
    return (sortedPair(x, y).first == x) ? it->second
                                         : swapQubits(it->second);
}

// ─────────────────────────────────────────────────────────────────────────
// The interaction
// ─────────────────────────────────────────────────────────────────────────

InteractionSimulation::InteractionResult
InteractionSimulation::computeInteraction(VertexPtr x, VertexPtr y) const {
    // The genuine joint input state — correlated when X, Y are Delaunay
    // neighbours of the initial layer or co-products of one interaction.
    const Eigen::Matrix4cd rhoXY = jointStateFor(x, y);

    // ρ_AB = U ρ_XY U† — the genuine joint state after the interaction.
    const Eigen::Matrix4cd rhoAB =
        interactionU_ * rhoXY * interactionU_.adjoint();

    const SystemState primeX = traceOutSecond(rhoAB);  // X' = Tr_Y ρ_AB
    const SystemState primeY = traceOutFirst(rhoAB);   // Y' = Tr_X ρ_AB

    const double sX = vonNeumannEntropy(traceOutSecond(rhoXY));
    const double sY = vonNeumannEntropy(traceOutFirst(rhoXY));
    // I(X:AB): the genuine mutual information sitting in ρ_AB — how much
    // the interaction correlated the two systems. The primary temporal
    // quantity; I(X:X') = S(X) - I(X:AB) is the residual.
    const double iJoint = jointMutualInformation(rhoAB);
    const double iInput = jointMutualInformation(rhoXY);

    InteractionResult res;
    res.statePrimeX = primeX;
    res.statePrimeY = primeY;
    res.stateAB = primeX;  // AB carried forward via its X-marginal proxy
    res.jointAB = rhoAB;   // (X' ⊗ Y') joint, inherited by the products

    // Local labels: 0=X 1=Y 2=X' 3=AB 4=Y'.
    auto key = [](int u, int v) {
        return std::make_pair(std::min(u, v), std::max(u, v));
    };
    res.edgeMI[key(0, 1)] = iInput;                  // X-Y   input pair
    res.edgeMI[key(0, 2)] = std::max(sX - iJoint, 0.0);  // X-X' residual
    res.edgeMI[key(1, 4)] = std::max(sY - iJoint, 0.0);  // Y-Y' residual
    res.edgeMI[key(0, 3)] = iJoint;                  // X-AB  joint MI
    res.edgeMI[key(1, 3)] = iJoint;                  // Y-AB  joint MI
    res.edgeMI[key(2, 3)] = iJoint;                  // X'-AB
    res.edgeMI[key(3, 4)] = iJoint;                  // Y'-AB
    res.edgeMI[key(2, 4)] = iJoint;                  // X'-Y' joint correlation
    res.edgeMI[key(0, 4)] = iJoint;                  // X-Y'  closure
    res.edgeMI[key(1, 2)] = iJoint;                  // Y-X'  closure
    return res;
}

// ─────────────────────────────────────────────────────────────────────────
// Frontier / move-table bookkeeping
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::isFrontier(VertexPtr v) noexcept {
    if (v == nullptr) return false;
    // A system is frozen once it has interacted — i.e. once it has an
    // out-edge crossing to a later time slice. Same-slice (spatial /
    // Delaunay) out-edges do not freeze it.
    for (EdgePtr e : v->getOutEdges())
        if (e->getTarget()->getTime() > v->getTime()) return false;
    return true;
}

void InteractionSimulation::rebuildMoveTables() {
    frontier_.clear();
    for (auto const& [v, state] : stateOf_) {
        (void)state;
        if (isFrontier(v)) frontier_.push_back(v);
    }

    eligibleEdges_.clear();
    for (EdgePtr e : spacetime_->getEdgeList()->toVector()) {
        VertexPtr a = e->getSource();
        VertexPtr b = e->getTarget();
        if (!isFrontier(a) || !isFrontier(b)) continue;
        if (a->getTime() != b->getTime()) continue;  // spatial only
        eligibleEdges_.emplace_back(a, b);
    }

    leafCells_.clear();
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 5) continue;  // (2,3) cell
        int frontierProducts = 0;
        for (VertexPtr v : s->getVertices())
            if (isFrontier(v)) ++frontierProducts;
        if (frontierProducts == 3) leafCells_.push_back(s);
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

    std::uniform_int_distribution<std::size_t> pick(
        0, eligibleEdges_.size() - 1);
    auto [x, y] = eligibleEdges_[pick(rng_)];

    const std::size_t nPlusBefore = eligibleEdges_.size();
    const InteractionResult res = computeInteraction(x, y);

    // The ten signed squared edge lengths of the proposed cell.
    double edgeSq[10];
    for (int e = 0; e < 10; ++e) {
        CellEdge const& ce = kCellEdges[e];
        const double mi = res.edgeMI.at(
            {std::min(ce.u, ce.v), std::max(ce.u, ce.v)});
        const double len = edgeLengthFromMI(mi, config_.epsilonI);
        edgeSq[e] = signedSquaredLength(len, ce.spacelike);
    }

    // ΔS evaluated in a throwaway complex — a rejected proposal never
    // touches the live geometry.
    const double deltaS = cellHingeAction(edgeSq);

    const std::size_t nMinusAfter = leafCells_.size() + 1;
    const double logPrefactor =
        std::log(static_cast<double>(nPlusBefore))
        - std::log(static_cast<double>(nMinusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Accepted — build the (2,3) cell into the live complex.
    const double tNext = x->getTime() + 1.0;
    VertexPtr xp = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});
    VertexPtr ab = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});
    VertexPtr yp = spacetime_->createVertex(
        spacetime_->getVertexCount(), std::vector<double>{tNext});
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

    stateOf_[xp] = res.statePrimeX;
    stateOf_[ab] = res.stateAB;
    stateOf_[yp] = res.statePrimeY;
    // The two products inherit the joint state ρ_AB — they stay
    // correlated, so a later interaction between them is genuine.
    {
        const auto key = sortedPair(xp, yp);
        jointOf_[key] =
            (key.first == xp) ? res.jointAB : swapQubits(res.jointAB);
    }

    ++interactionCount_;
    ++interactAccepted_;
    rebuildMoveTables();
    return true;
}

bool InteractionSimulation::unInteract() {
    ++unInteractAttempts_;
    if (leafCells_.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(
        0, leafCells_.size() - 1);
    SimplexPtr cell = leafCells_[pick(rng_)];

    const std::size_t nMinusBefore = leafCells_.size();

    double deltaS = 0.0;
    std::vector<SimplexPtr> cellHinges;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets()) {
            if (hinge->getVertices().size() != 3) continue;
            cellHinges.push_back(hinge);
            auto it = hingeAction_.find(hinge);
            if (it != hingeAction_.end()) deltaS -= it->second;
        }

    const std::size_t nPlusAfter = eligibleEdges_.size() + 1;
    const double logPrefactor =
        std::log(static_cast<double>(nMinusBefore))
        - std::log(static_cast<double>(nPlusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Commit: drop the cell's three product vertices and their states.
    for (SimplexPtr hinge : cellHinges) hingeAction_.erase(hinge);
    for (VertexPtr v : cell->getVertices())
        if (v->getOutEdges().empty()) stateOf_.erase(v);  // products only
    spacetime_->removeSimplex(cell);

    if (interactionCount_ > 0) --interactionCount_;
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
            leafCells_.empty()      ? true
            : eligibleEdges_.empty() ? false
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
    // Index every vertex that appears in the complex.
    std::unordered_map<VertexPtr, int> idx;
    for (EdgePtr e : spacetime_->getEdgeList()->toVector())
        for (VertexPtr v : {e->getSource(), e->getTarget()})
            if (!idx.count(v)) idx[v] = static_cast<int>(idx.size());
    if (idx.empty())
        return std::vector<double>(sigmas.size(), 0.0);

    // Weighted graph with edge weight = the mutual information itself
    // (W = I, per holography §3.4) — recovered from the stored edge
    // length ℓ via I = I_max · exp(-ℓ).
    std::vector<std::tuple<int, int, double>> edges;
    edges.reserve(static_cast<std::size_t>(spacetime_->getEdgeList()
                                               ->toVector().size()));
    for (EdgePtr e : spacetime_->getEdgeList()->toVector()) {
        const double len = std::sqrt(std::abs(e->getSquaredLength()));
        const double mi = kIMax * std::exp(-len);
        edges.emplace_back(idx.at(e->getSource()),
                           idx.at(e->getTarget()), mi);
    }

    EmergentGraph graph = EmergentGraph::fromWeightedEdges(
        static_cast<int>(idx.size()), edges);
    const std::vector<double> p = graph.returnProbability(sigmas, krylovDim);
    return EmergentGraph::spectralDimension(sigmas, p);
}

} // namespace tessera::quantum
