// InteractionSimulation — Metropolis Monte Carlo over interaction histories,
// weighted by the geometric Regge action on the dual lattice.
//
// ─── The construction ────────────────────────────────────────────────────
//
// The configuration is an *interaction history*: a simplicial complex grown
// from a Poisson-Delaunay initial layer of Schwinger sites by pairwise
// interaction events, together with the evolving MPS quantum state the
// interactions act on.
//
//   • Initial layer.  N staggered Schwinger sites, Delaunay-triangulated in
//     2D (the triangulation is supplied by the caller — scipy.spatial). The
//     joint state is the Schwinger DMRG ground state.
//
//   • Interaction event.  Two frontier systems X, Y sharing a frontier
//     spatial edge interact: a fresh staggered site AB is created (pair
//     creation — "a new worldline"), the mediated interaction unitary is
//     applied to the MPS, and the (2,3) cell {X, Y, X', AB, Y'} is attached.
//     X, Y leave the frontier; X', AB, Y' join it.
//
//   • Frontier rule.  A system may interact only while it has no out-edges.
//     un-interact may remove only a *leaf* cell — one whose three products
//     are all still on the frontier.
//
//   • Edge lengths.  ℓ = -log I, with I the ordinary reduced-density-matrix
//     mutual information between co-existing subsystems of the MPS. No Choi
//     states: the interaction event makes a system and its successors
//     co-exist in one pure state.
//
// ─── The action and the ensemble ─────────────────────────────────────────
//
// The geometric Regge action on the MI-lengthed complex,
//
//     S = Σ_{h ∈ hinges} A_h ε_h,
//
// with A_h the Heron hinge area and ε_h the deficit angle, evaluated (not
// solved) via the ReggeSolver primitives. The partition function is
//
//     Z = Σ_C (1 / C_C) e^{-β S[C]}
//
// over interaction histories reachable from the initial layer. β is the
// inverse-temperature coupling: varying β (and the Schwinger m/g) maps
// out the phase structure, and the object of the search is the point
// where the emergent heat-kernel spectral dimension reaches 4 — the
// 3+1-dimensional phase. The equilibrium ensemble is sampled by
// Metropolis-Hastings with the interact / un-interact move pair:
//
//     A(C→C') = min{ 1, (N₊/N₋)·(C_C/C_{C'})·e^{-β ΔS} }.
//
// Volume is controlled by capping the interaction count (the T-cap).
//
// The class mirrors the abstraction level of tessera::CDT — move
// primitives, propose* counterparts, sweep / thermalize / tune, and the
// computeAction / getAcceptanceRates / observable getters — but with the
// interaction move set, the β coupling, and an MPS state carried
// internally. The MPS never crosses the language boundary.

#pragma once

#include "matter/MatterConfiguration.h"
#include "quantum/schwinger_model.hpp"
#include "simulations/Simulation.h"
#include "spacetime/Spacetime.h"

#include <itensor/all.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tessera::quantum {

// Flat configuration for an interaction-history Monte Carlo run.
struct InteractionConfig {
    // ─── Schwinger Hamiltonian (initial ground state + interaction unitary) ─
    SchwingerParams params{};   // N is the initial-layer site count

    // ─── DMRG ground state of the initial layer ─────────────────────────
    int    dmrgMaxBondDim{64};
    int    dmrgNSweeps{12};
    int    dmrgKrylovDim{4};
    double dmrgCutoff{1e-12};

    // ─── Interaction unitary ────────────────────────────────────────────
    // The two-site Schwinger evolution exp(-i H_XY dt) is KAK-decomposed
    // and its Cartan core routed through the freshly-created AB site.
    double dt{0.25};

    // ─── MPS bookkeeping for gate application ───────────────────────────
    int    maxBondDim{200};
    double cutoff{1e-10};

    // ─── Regge ensemble ─────────────────────────────────────────────────
    double beta{1.0};            // inverse temperature in e^{-β S}
    double epsilonI{1e-10};      // MI floor for ℓ = -log I

    // ─── Volume control (the T-cap) ─────────────────────────────────────
    // The complex is grown to this many interaction events, then the MC
    // equilibrates with interact / un-interact moves holding ~this size.
    std::size_t targetInteractions{0};

    // ─── Initial Poisson-Delaunay layer connectivity ────────────────────
    // Delaunay edges among the N initial sites, as 0-based site-index
    // pairs. Supplied by the caller (scipy.spatial.Delaunay).
    std::vector<std::pair<int, int>> delaunayEdges;

    bool         quiet{true};
    std::uint32_t seed{0};
};

// Transactional interaction move — propose() tentatively applies the
// interaction (or its inverse) to the MPS and the complex; apply() commits;
// rollback() replays the undo log. Mirrors tessera::PachnerMove. Defined in
// the .cpp; forward-declared here so propose* can hand one back.
class InteractionMove;

// Metropolis Monte Carlo over interaction histories.
class InteractionSimulation : public tessera::Simulation {
  public:
    // Build the initial Poisson-Delaunay layer, solve its Schwinger DMRG
    // ground state, and stand up the (vertex, frontier, table) bookkeeping.
    // Throws std::invalid_argument on a malformed config (N < 2, empty
    // delaunayEdges, out-of-range edge indices, ...).
    explicit InteractionSimulation(InteractionConfig config);

    ~InteractionSimulation() override;

    // ─── Move primitives (propose + Metropolis accept, like CDT::add) ───

    // Propose interacting a uniformly-random eligible frontier spatial
    // edge {X, Y}; attach the (2,3) cell, create AB, apply the mediated
    // interaction unitary. Returns true if accepted.
    bool interact();

    // Propose removing a uniformly-random leaf cell; invert its unitary,
    // restore the parents to the frontier. Returns true if accepted.
    bool unInteract();

    // ─── Transactional proposers (caller drives propose/apply/rollback) ──
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeInteract();
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeUnInteract();

    // ─── Driving loop (Simulation overrides + sweep) ────────────────────

    // One Monte Carlo sweep: propose ~N₊ + N₋ moves (uniformly chosen
    // between interact and un-interact) and accept/reject each. Returns
    // the number of accepted moves.
    int sweep();

    // Grow to the T-capped size, then run sweeps until the action
    // stabilises (relative change below 1% between sweeps).
    void thermalize() override;

    // Grow the complex toward targetInteractions by biased interact moves;
    // the initial-condition phase, analogous to CDT::tune driving N₄.
    void tune(std::function<void(int, int)> progress = nullptr) override;

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

    // Interaction-count profile by generation depth — the analogue of
    // CDT::getVolumeProfile.
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

    // The evolving quantum state. sites_ grows by one each interaction
    // (AB creation); psi_ is rebuilt onto the longer SiteSet.
    itensor::SpinHalf sites_;
    itensor::MPS      psi_;

    std::mt19937 rng_;

    // ─── Frontier / move bookkeeping (the DP tables) ────────────────────
    // System ↔ MPS-site index. A "system" is a Spacetime vertex.
    std::unordered_map<tessera::VertexPtr, int> siteOfVertex_;
    std::vector<tessera::VertexPtr>             vertexOfSite_;

    // Systems with no out-edges. Maintained incrementally.
    std::vector<tessera::VertexPtr> frontier_;

    // N₊: eligible interact candidates — frontier spatial edges {X, Y}
    // with both endpoints on the frontier.
    std::vector<std::pair<tessera::VertexPtr, tessera::VertexPtr>>
        eligibleEdges_;

    // N₋: leaf cells — (2,3) cells whose three products are all still on
    // the frontier.
    std::vector<tessera::SimplexPtr> leafCells_;

    // Per-hinge action contribution A_h·ε_h, keyed by simplex pointer
    // (fingerprint-deduplicated by the Spacetime). Updated locally per
    // move so ΔS is O(affected hinges).
    std::unordered_map<tessera::SimplexPtr, double,
                       tessera::SimplexPtrHash, tessera::SimplexPtrEq>
        hingeAction_;

    // Frozen edge lengths ℓ = -log I — immutable once a system leaves the
    // frontier. The memoization boundary. Keyed by the sorted endpoint
    // id pair.
    std::map<std::pair<tessera::IdType, tessera::IdType>, double>
        frozenEdgeLength_;

    std::size_t interactionCount_{0};

    // Undo stack for Metropolis rollback and the un-interact move: the
    // MPS state and the frontier-position bookkeeping, snapshotted
    // immediately before each applyInteractionMPS. revertInteractionMPS
    // pops the most recent frame back.
    struct UndoFrame {
        itensor::MPS psi;
        std::unordered_map<tessera::VertexPtr, int> siteOfVertex;
        std::vector<tessera::VertexPtr> vertexOfSite;
    };
    std::vector<UndoFrame> undoStack_;

    // Matter config kept null/empty: matter is in the MIs, not a separate
    // worldline term (see writeup — avoids double-counting).
    tessera::MatterConfiguration matter_{};

    // ─── Metropolis-Hastings acceptance test ────────────────────────────
    // Accepts when -βΔS + logPrefactor ≥ 0, else with probability
    // e^{-βΔS + logPrefactor}. logPrefactor carries log[(N₊/N₋)(C_C/C_C')].
    [[nodiscard]] bool accept(double deltaS, double logPrefactor = 0.0);

    // Build the initial layer: vertices, Delaunay edges, ground-state MPS,
    // and the initial frontier / eligible-edge tables.
    void buildInitialLayer();

    // DMRG the Schwinger ground state of the N-site initial layer onto
    // sites_ / psi_.
    void buildGroundState();

    // Predicate: a system is on the frontier iff it has no out-edges.
    [[nodiscard]] static bool isFrontier(tessera::VertexPtr v) noexcept;

    // Recompute eligibleEdges_ (N₊) and leafCells_ (N₋) from the current
    // complex. Used at construction; per-move updates are incremental.
    void rebuildMoveTables();

    // MPS-dependent helpers. interact() / unInteract() route every
    // quantum-state touch through these — the simplicial bookkeeping is
    // written against this interface and goes live unchanged once the
    // mediated-unitary / site-insertion machinery is wired in.

    // Result of tentatively applying an interaction to the MPS: the
    // mutual informations of the ten edges of the new (2,3) cell, keyed
    // by the cell's local-label pairs (0=X 1=Y 2=X' 3=AB 4=Y'), plus the
    // MPS positions the three products X', AB, Y' now occupy.
    struct InteractionMIs {
        std::map<std::pair<int, int>, double> edgeMI;
        int posXp{0};
        int posAB{0};
        int posYp{0};
    };

    // Apply the mediated KAK interaction unitary for frontier systems
    // X, Y: create the AB site in the MPS, evolve, and return the
    // new-edge MIs.
    [[nodiscard]] InteractionMIs applyInteractionMPS(tessera::VertexPtr x,
                                                     tessera::VertexPtr y);

    // Invert the most recent interaction on the MPS (the un-interact
    // move / a Metropolis rollback).
    void revertInteractionMPS();

    // ─── Acceptance counters ────────────────────────────────────────────
    std::int64_t interactAttempts_{0},   interactAccepted_{0};
    std::int64_t unInteractAttempts_{0}, unInteractAccepted_{0};
};

} // namespace tessera::quantum
