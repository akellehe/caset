// InteractionSimulation — Metropolis Monte Carlo over interaction histories,
// weighted by the geometric Regge action on the dual lattice.
//
// See docs/source/interaction-history-monte-carlo.md for the full charter.
//
// ─── The construction ────────────────────────────────────────────────────
//
// The configuration is an interaction history: a simplicial complex grown
// from a Poisson-Delaunay initial layer of quantum systems by pairwise
// interaction events.
//
//   • Initial layer. N systems, each a known randomized *mixed* state
//     (S(ρ) > 0), Delaunay-triangulated in 2D.
//
//   • Interaction event. Two frontier systems X, Y interact through a
//     two-system unitary U; the interaction product is the genuine joint
//     state ρ_AB = U (ρ_X ⊗ ρ_Y) U†. The (2,3) cell {X, Y, X', AB, Y'} is
//     attached: X, Y leave the frontier, the marginals X', Y' and the
//     joint-state node AB join it.
//
//   • Edge lengths. ℓ = -log I. The spatial / co-existing edges are
//     ordinary mutual informations; the temporal edges close by a
//     conservation law — S(X) = I(X:X') + I(X:AB) with I(X:X') the
//     residual. Every quantity is a reduced-density-matrix computation on
//     a state that concretely exists: no MPS, no Choi, no freeze.
//
// ─── The ensemble ────────────────────────────────────────────────────────
//
// The geometric Regge action S = Σ_h A_h ε_h on the MI-lengthed complex,
// sampled at inverse temperature β by Metropolis-Hastings with the
// interact / un-interact move pair:
//
//     A(C→C') = min{ 1, (N₊/N₋)·(C_C/C_{C'})·e^{-β ΔS} }.
//
// Volume is controlled by capping the interaction count (the T-cap). The
// object of the search is the β at which the emergent spectral dimension
// reaches 4. The class mirrors the abstraction level of tessera::CDT —
// move primitives, propose* counterparts, sweep / thermalize / tune, and
// the computeAction / getAcceptanceRates / observable getters.

#pragma once

#include "simulations/Simulation.h"
#include "spacetime/Spacetime.h"

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tessera::quantum {

// A single system's quantum state — a one-qubit density matrix.
using SystemState = Eigen::Matrix2cd;

// Flat configuration for an interaction-history Monte Carlo run.
struct InteractionConfig {
    // ─── Initial layer ──────────────────────────────────────────────────
    int nSystems{0};   // system count = Poisson-point count

    // ─── Two-site interaction unitary U = exp(-i H_XY dt) ───────────────
    // H_XY is the Schwinger two-site Hamiltonian term.
    double a{1.0};
    double g{1.0};
    double m{0.5};
    double dt{0.25};

    // ─── Regge ensemble ─────────────────────────────────────────────────
    double beta{1.0};        // inverse temperature in e^{-β S}
    double epsilonI{1e-10};  // mutual-information floor for ℓ = -log I

    // ─── Volume control (the T-cap) ─────────────────────────────────────
    std::size_t targetInteractions{0};

    // ─── Initial Poisson-Delaunay layer connectivity ────────────────────
    // Delaunay edges among the N initial systems, 0-based index pairs,
    // supplied by the caller (scipy.spatial.Delaunay).
    std::vector<std::pair<int, int>> delaunayEdges;

    std::uint32_t seed{0};
    bool          quiet{true};
};

// Transactional interaction move — mirrors tessera::PachnerMove. Defined
// in the .cpp; forward-declared so propose* can hand one back.
class InteractionMove;

// Metropolis Monte Carlo over interaction histories.
class InteractionSimulation : public tessera::Simulation {
  public:
    // Build the Poisson-Delaunay initial layer of randomized mixed-state
    // systems and the frontier / N₊ / N₋ bookkeeping. Throws
    // std::invalid_argument on a malformed config.
    explicit InteractionSimulation(InteractionConfig config);

    ~InteractionSimulation() override;

    // ─── Move primitives (propose + Metropolis accept, like CDT::add) ───

    // Interact a uniformly-random eligible frontier spatial edge {X, Y}:
    // form ρ_AB = U(ρ_X⊗ρ_Y)U†, attach the (2,3) cell, length its edges.
    // Returns true if the Metropolis test accepted.
    bool interact();

    // Remove a uniformly-random leaf cell, restoring its parents to the
    // frontier. Returns true if accepted.
    bool unInteract();

    // ─── Transactional proposers (caller drives propose/apply/rollback) ──
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeInteract();
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeUnInteract();

    // ─── Driving loop (Simulation overrides + sweep) ────────────────────

    // One Monte Carlo sweep: propose ~N₊ + N₋ moves, accept/reject each.
    // Returns the number of accepted moves.
    int sweep();

    // Grow the complex toward targetInteractions — the initial-condition
    // phase, analogous to CDT::tune driving N₄.
    void tune(std::function<void(int, int)> progress = nullptr) override;

    // Run sweeps until the action stabilises (relative change < 1%).
    void thermalize() override;

    // ─── Diagnostics ────────────────────────────────────────────────────

    // The geometric Regge action S = Σ_h A_h ε_h of the current complex.
    [[nodiscard]] double computeAction() const;

    // Heat-kernel spectral dimension D_S(σ) of the MI-weighted complex
    // graph, on the supplied σ-grid.
    [[nodiscard]] std::vector<double>
    getSpectralDimension(const std::vector<double> &sigmas,
                         int krylovDim = 30) const;

    // Histogram of deficit angles over interior hinges.
    [[nodiscard]] std::vector<double> getDeficitAngleDistribution() const;

    // Interaction-count profile by generation depth.
    [[nodiscard]] std::vector<int> getVolumeProfile() const;

    // Accepted / attempted ratio per move type.
    [[nodiscard]] std::map<std::string, double> getAcceptanceRates() const;

    [[nodiscard]] const std::shared_ptr<tessera::Spacetime> &
    getSpacetime() const noexcept { return spacetime_; }

    [[nodiscard]] std::size_t interactionCount() const noexcept {
        return interactionCount_;
    }
    [[nodiscard]] double getBeta() const noexcept { return config_.beta; }

    void setBeta(double beta) noexcept { config_.beta = beta; }
    void setSeed(std::uint32_t s) noexcept { rng_.seed(s); }

  private:
    // ─── Configuration + owned state ────────────────────────────────────
    InteractionConfig config_;
    std::shared_ptr<tessera::Spacetime> spacetime_;
    std::mt19937 rng_;

    // The two-site interaction unitary U = exp(-i H_XY dt), 4×4.
    Eigen::Matrix4cd interactionU_;

    // Per-system quantum state — the one-qubit marginal. Every system ever
    // created keeps its state here; frozen systems retain theirs so
    // un-interact restores cleanly.
    std::unordered_map<tessera::VertexPtr, SystemState> stateOf_;

    // Joint states of correlated system pairs: the randomized initial
    // layer is one correlated mixed state, so Delaunay-adjacent systems
    // share genuine mutual information, and an interaction's two products
    // X', Y' inherit the joint state ρ_AB. Keyed by the sorted vertex
    // pointer pair. A pair absent here is treated as uncorrelated.
    std::map<std::pair<tessera::VertexPtr, tessera::VertexPtr>,
             Eigen::Matrix4cd>
        jointOf_;

    // ─── Frontier / move bookkeeping (the DP tables) ────────────────────
    // The frontier — systems that have not yet interacted (no temporal
    // out-edges). Maintained incrementally; O(1) membership.
    std::unordered_set<tessera::VertexPtr> frontier_;

    // N₊: eligible interact candidates — frontier spatial edges {X, Y}.
    // Kept as a flat vector for O(1) uniform-random pick by index, plus
    // the back-reference indices below for O(degree) incremental
    // add/remove.
    std::vector<std::pair<tessera::VertexPtr, tessera::VertexPtr>>
        eligibleEdges_;
    // edge (sorted pair) -> its position in eligibleEdges_.
    std::map<std::pair<tessera::VertexPtr, tessera::VertexPtr>, std::size_t>
        edgePos_;
    // vertex -> the eligible-edges incident to it. Used to drop all of v's
    // edges in O(degree) when v freezes.
    std::unordered_map<tessera::VertexPtr,
                       std::set<std::pair<tessera::VertexPtr,
                                          tessera::VertexPtr>>>
        vertexEdges_;

    // Per-hinge action contribution A_h·ε_h. Updated locally per move so
    // ΔS is O(affected hinges).
    std::unordered_map<tessera::SimplexPtr, double,
                       tessera::SimplexPtrHash, tessera::SimplexPtrEq>
        hingeAction_;

    // Dependency tracking for deep un-interactions.
    //   producedByCell_[v]  = the cell whose interaction created v (a
    //                         product vertex X', AB, or Y').
    //   consumedByCell_[v]  = the cell that took v as a parent in its
    //                         interaction (if any). Each vertex is
    //                         consumed by at most one cell.
    // Together these give the dependency tree: un-interacting a cell
    // truncates its future cone via BFS through producedByCell_ →
    // consumedByCell_.
    std::unordered_map<tessera::VertexPtr, tessera::SimplexPtr>
        producedByCell_;
    std::unordered_map<tessera::VertexPtr, tessera::SimplexPtr>
        consumedByCell_;

    // Per-cell count of its products that have been consumed by a child
    // interaction. A cell is a *leaf* (its deep un-interact returns to
    // exactly the state where this cell didn't exist) iff this count is
    // zero. The Metropolis prefactor for interact uses leafCellCount_,
    // bounded as the lattice grows. Maintained incrementally.
    std::unordered_map<tessera::SimplexPtr, std::uint8_t,
                       tessera::SimplexPtrHash, tessera::SimplexPtrEq>
        consumedProductsOf_;
    std::size_t leafCellCount_{0};

    std::size_t   interactionCount_{0};
    std::uint64_t nextVertexId_{0};   // monotone; never reused after removeVertex

    // ─── Helpers ────────────────────────────────────────────────────────
    // Build the Poisson-Delaunay initial layer: randomized mixed-state
    // systems as t=0 vertices, the Delaunay edges, the initial tables.
    void buildInitialLayer();

    // Initial population of frontier_, eligibleEdges_, edgePos_,
    // vertexEdges_ from the Poisson-Delaunay layer. Called once.
    void initMoveTables();

    // Incremental updates — O(degree) per call.
    void addEligibleEdge(tessera::VertexPtr a, tessera::VertexPtr b);
    void removeEligibleEdge(
        std::pair<tessera::VertexPtr, tessera::VertexPtr> key);
    // Freeze v: drop it from the frontier and drop all its eligible
    // edges. The mirror unfreeze (used by un-interact) re-adds v to the
    // frontier and reinstates eligible edges to its still-frontier
    // neighbours.
    void freezeFromFrontier(tessera::VertexPtr v);
    void unfreezeIntoFrontier(tessera::VertexPtr v);

    // Sorted-pair key constructor for the edge tables.
    static std::pair<tessera::VertexPtr, tessera::VertexPtr>
    edgeKey(tessera::VertexPtr a, tessera::VertexPtr b) noexcept {
        return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    }

    // The ten edge mutual informations of an interaction X,Y → X',AB,Y'.
    // Keyed by the cell's local-label pairs (0=X 1=Y 2=X' 3=AB 4=Y').
    // Also returns the marginal states X' = Tr_Y ρ_AB, Y' = Tr_X ρ_AB.
    struct InteractionResult {
        std::map<std::pair<int, int>, double> edgeMI;
        SystemState statePrimeX;     // X' = Tr_Y ρ_AB
        SystemState statePrimeY;     // Y' = Tr_X ρ_AB
        SystemState stateAB;         // AB carried as its X-marginal proxy
        Eigen::Matrix4cd jointAB;    // ρ_AB in (X' ⊗ Y') order
    };
    [[nodiscard]] InteractionResult
    computeInteraction(tessera::VertexPtr x, tessera::VertexPtr y) const;

    // The joint state ρ_XY in (X ⊗ Y) order: the stored correlated pair
    // if X, Y share one (initial-layer Delaunay neighbours, or the two
    // products of one interaction), the uncorrelated product otherwise.
    [[nodiscard]] Eigen::Matrix4cd
    jointStateFor(tessera::VertexPtr x, tessera::VertexPtr y) const;

    // Metropolis-Hastings acceptance: accepts when -βΔS + logPrefactor ≥ 0,
    // else with probability e^{-βΔS + logPrefactor}.
    [[nodiscard]] bool accept(double deltaS, double logPrefactor = 0.0);

    // ─── Acceptance counters ────────────────────────────────────────────
    std::int64_t interactAttempts_{0},   interactAccepted_{0};
    std::int64_t unInteractAttempts_{0}, unInteractAccepted_{0};
};

} // namespace tessera::quantum
