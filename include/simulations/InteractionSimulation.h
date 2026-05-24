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
// reaches 4. The class mirrors the abstraction level of tessera::simulations::CDT —
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

// A single system's quantum state — a one-qubit density matrix.
// LEGACY: pre-#56 v0.1 path used this; the live state-bearer is now
// `tessera::quantum::QuantumState` carried by every Vertex. Kept as an
// alias only because the removed `computeInteraction` helper returned
// a struct full of these; remove together with the helper once the
// rewrite of `interact()` is complete.
using SystemState = Eigen::Matrix2cd;

// LEGACY (removed in #56): v0.1 charge mode is gone with the rest of the
// charge observables. Kept here as a comment so anyone bisecting through
// git history can quickly identify what was removed. Will be reinstated
// against QuantumState in the charge-observables-v0.3 follow-up.
//
//   enum class InitialChargeMode { ALTERNATING, RANDOM };

// Flat configuration for an interaction-history Monte Carlo run.
// Post-#56 design: no time-evolution parameters, no charge flags —
// every interaction is a Koashi-Imoto decomposition of the joint state
// at the moment of interaction (see docs/source/design — captured in
// the issue body since the design doc was inlined there).
struct InteractionConfig {
    // ─── Initial layer ──────────────────────────────────────────────────
    int nSystems{0};       // system count = Poisson-point count
    int dimPerVertex{4};   // per-vertex physical Hilbert dimension

    // ─── Regge ensemble ─────────────────────────────────────────────────
    double beta{1.0};        // inverse temperature in e^{-β S}
    double epsilonI{1e-10};  // mutual-information floor (legacy, retained
                             // for edge-length numerical safety; the d_VR
                             // formula uses I_max-relative MI but still
                             // needs a positivity floor near I = 0)

    // ─── Volume control (the T-cap) ─────────────────────────────────────
    std::size_t targetInteractions{0};

    // ─── Initial Poisson-Delaunay layer connectivity ────────────────────
    std::vector<std::pair<int, int>> delaunayEdges;

    // ─── Information budget and KI tolerances (issue #56) ──────────────
    // I_max for the whole simulation is N · epsInit; per-vertex
    // unique-information contribution at initialisation is epsInit nats.
    double epsInit{1.0};
    double epsLocalPure{1e-10};
    double epsKiEigen{1e-10};
    double epsKiCondState{1e-10};
    double epsKiSvd{1e-10};

    // ─── REMOVED in #56 (kept here as a comment for git-bisect breadcrumbs) ──
    //
    // Schwinger two-site H_XY:        a, g, m, dt
    // v0.1 charge moves:              featureCharges, featureDeactivateOnAnnihilate,
    //                                 featurePhotonOnAnnihilate, useCharges,
    //                                 cpBias, initialChargeMode
    // v0.2 qudit-basis:               featureQuditBasis, featureChoiSigmaAB,
    //                                 j_chargeCharge, j_spinSpin, massShift,
    //                                 gammaCpViolation, dtPair
    //
    // All of the above are reinstated in the charge-observables-v0.3
    // follow-up against the new QuantumState representation.

    std::uint32_t seed{0};
    bool          quiet{true};
};

// Transactional interaction move — mirrors tessera::spacetime::PachnerMove. Defined
// in the .cpp; forward-declared so propose* can hand one back.
class InteractionMove;

// Metropolis Monte Carlo over interaction histories.
class InteractionSimulation : public tessera::simulations::Simulation {
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

    // ─── REMOVED in #56 ─────────────────────────────────────────────────
    // bool annihilate();   — v0.1 (+,−) partial annihilation move.
    // bool pairCreate();   — v0.1 (+,−) pair creation move.
    // Both depended on per-vertex charges and are reinstated against
    // QuantumState in the charge-observables-v0.3 follow-up.

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

    // ─── REMOVED in #56 ─────────────────────────────────────────────────
    // double                    getGlobalCharge()         — v0.1.
    // vector<array<double, 4>>  getChargeProfile()        — v0.1.
    // vector<double>            getChargeCorrelation(int) — v0.1.
    // Reinstated against QuantumState in charge-observables-v0.3.

    [[nodiscard]] const std::shared_ptr<tessera::spacetime::Spacetime> &
    getSpacetime() const noexcept { return spacetime_; }

    // ─── REMOVED in #56 ─────────────────────────────────────────────────
    // double           quditChargeOf(VertexPtr)               — v0.2.
    // MatrixXcd        quditJointStateFor(VertexPtr, VertexPtr) — v0.2.
    // map<VertexPtr,M> quditStateOfMap()                       — v0.2.
    //
    // Replacements: per-vertex state is now `vertex.quantumState()`
    // (mesh::Vertex); pairwise joint state is reconstructed on demand
    // from endpoint vertex states. Charge observables are reinstated
    // against QuantumState in charge-observables-v0.3.

    [[nodiscard]] std::size_t interactionCount() const noexcept {
        return interactionCount_;
    }
    [[nodiscard]] std::size_t frontierSize() const noexcept {
        return frontier_.size();
    }
    // The cardinality of the eligible un-interact set. Replaces the
    // pre-#56 `leafCellCount_` counter; the value is now `leafSimplices_.size()`.
    [[nodiscard]] std::size_t leafSimplexCount() const noexcept {
        return leafSimplices_.size();
    }
    // Global information ceiling for the simulation: N · epsInit. Set
    // once at construction from the initial layer; used by edge-length
    // computation (d_VR = -log(I / I_max)).
    [[nodiscard]] double getIMax() const noexcept { return iMax_; }
    [[nodiscard]] double getBeta() const noexcept { return config_.beta; }

    void setBeta(double beta) noexcept { config_.beta = beta; }
    void setSeed(std::uint32_t s) noexcept { rng_.seed(s); }

  private:
    // ─── Configuration + owned state ────────────────────────────────────
    InteractionConfig                              config_;
    std::shared_ptr<tessera::spacetime::Spacetime> spacetime_;
    std::mt19937                                   rng_;

    // ─── Frontier (uninteracted vertices) ───────────────────────────────
    // Flat for O(1) sampling, index map for O(1) swap-and-pop removal.
    // N_+ = |frontier|·(|frontier|−1)/2; a uniform-random pair is
    // (frontier_[i], frontier_[j]) for random i ≠ j.
    std::vector<tessera::mesh::VertexPtr>                     frontier_;
    std::unordered_map<tessera::mesh::VertexPtr, std::size_t> frontierIdx_;

    // ─── Leaf simplices (eligible un-interact targets) ──────────────────
    // A simplex S is a "leaf" iff all three of its child vertices are
    // currently on the frontier. Maintained incrementally on interact /
    // unInteract; `leafSimplices_.size()` replaces the old
    // `leafCellCount_` counter (the Metropolis prefactor reads the
    // size directly).
    //
    // REMOVED in #56 — derivable from leafSimplices_ + simplex containment:
    //   producedByCell_     — a vertex's parent simplex is found by
    //                         walking its `simplices_` set and selecting
    //                         the one where the vertex is a child.
    //   consumedByCell_     — symmetric; a vertex's child simplex (if any)
    //                         is the one where it's a parent.
    //   consumedProductsOf_ — per-cell "kids still on frontier?" counter
    //                         replaced by leafSimplices_ membership.
    //   leafCellCount_      — replaced by leafSimplices_.size().
    std::unordered_set<tessera::mesh::SimplexPtr,
                       tessera::mesh::SimplexPtrHash,
                       tessera::mesh::SimplexPtrEq>           leafSimplices_;

    // ─── REMOVED in #56 (state now lives on mesh::Vertex) ───────────────
    //
    //   Eigen::Matrix4cd                                     interactionU_;
    //   unordered_map<VertexPtr, SystemState>                stateOf_;
    //   map<pair<VertexPtr, VertexPtr>, Eigen::Matrix4cd>    jointOf_;
    //   unordered_map<VertexPtr, double>                     chargeOf_;
    //   unordered_map<VertexPtr, Eigen::Matrix4cd>           quditStateOf_;
    //   map<pair<VertexPtr, VertexPtr>, Eigen::MatrixXcd>    quditJointOf_;
    //   Eigen::MatrixXcd                                     quditInteractionU_;
    //   Eigen::MatrixXcd                                     quditChoiU_;
    //   unordered_set<VertexPtr>                             choiSigmaAbSet_;
    //   vector<VertexPtr>          frontierPos_, frontierNeg_;
    //   unordered_map<VertexPtr, size_t> frontierPosIdx_, frontierNegIdx_;
    //   int64_t annihilateAttempts_, annihilateAccepted_;
    //   int64_t pairCreateAttempts_, pairCreateAccepted_;
    //
    // All replaced by `vertex.quantumState()` (per-vertex marginal,
    // joints reconstructed on demand) plus KI-based interact / unInteract.

    // Per-hinge action contribution A_h·ε_h. Updated locally per move so
    // ΔS is O(affected hinges).
    std::unordered_map<tessera::mesh::SimplexPtr, double,
                       tessera::mesh::SimplexPtrHash,
                       tessera::mesh::SimplexPtrEq>            hingeAction_;

    double        iMax_{0.0};            // N · epsInit, set in ctor
    std::size_t   interactionCount_{0};
    std::uint64_t nextVertexId_{0};   // monotone; never reused after removeVertex

    // ─── Helpers ────────────────────────────────────────────────────────
    // Build the Poisson-Delaunay initial layer: randomized mixed-state
    // systems as t=0 vertices, the Delaunay edges, the initial tables.
    void buildInitialLayer();

    // O(1) frontier mutations via swap-and-pop on the flat vector.
    void addToFrontier(tessera::mesh::VertexPtr v);
    void removeFromFrontier(tessera::mesh::VertexPtr v);
    [[nodiscard]] bool onFrontier(tessera::mesh::VertexPtr v) const noexcept {
        return frontierIdx_.find(v) != frontierIdx_.end();
    }

    // d_VR(u, v) = -log(I(u:v) / iMax_), with I computed from the
    // two vertex states' joint (product joint ρ_u ⊗ ρ_v as per the
    // #56 design — joint history rides on Σ vertices in the state graph,
    // not on a separate lookup table). Returns +∞ if I == 0.
    [[nodiscard]] double edgeLength(tessera::mesh::VertexPtr u,
                                    tessera::mesh::VertexPtr v) const;

    // ─── REMOVED in #56 ─────────────────────────────────────────────────
    //   void               addToSignBucket(VertexPtr);
    //   void               removeFromSignBucket(VertexPtr);
    //   InteractionResult  computeInteraction(VertexPtr, VertexPtr);     — Schwinger.
    //   InteractionResultQudit computeInteractionQudit(VertexPtr, VertexPtr);
    //   Matrix4cd          jointStateFor(VertexPtr, VertexPtr);
    //
    // The Schwinger-based `computeInteraction*` is replaced inline in
    // `interact()` by a Koashi-Imoto decomposition via
    // `tessera::quantum::koashiImotoDecompose`. The 10 edge MIs are
    // recovered by querying the resulting per-vertex states pairwise.

    // Metropolis-Hastings acceptance: accepts when -βΔS + logPrefactor ≥ 0,
    // else with probability e^{-βΔS + logPrefactor}.
    [[nodiscard]] bool accept(double deltaS, double logPrefactor = 0.0);

    // ─── Acceptance counters ────────────────────────────────────────────
    std::int64_t interactAttempts_{0},   interactAccepted_{0};
    std::int64_t unInteractAttempts_{0}, unInteractAccepted_{0};
};

} // namespace tessera::simulations
