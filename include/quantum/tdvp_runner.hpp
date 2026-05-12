// SchwingerQuench — coarse-grained q-qbar quench + TDVP real-time
// evolution pipeline for the Schwinger model.
//
// PLAN.md §5 Phase 4: a self-contained DMRG → quench → TDVP driver
// that takes a flat config struct and returns a list of per-snapshot
// scalar diagnostics. PLAN.md §5 Phase 5 / §1: the same model can also
// run the causal-order comparison pipeline (DMRG → quench → TDVP →
// build the three orders → compare).
//
// ─── End-to-end pipeline (SchwingerQuench::evolve) ────────────────────────
//
//   1. Build the Schwinger MPO (Phase 1) and run DMRG to the GS (Phase 2).
//   2. Apply the σ⁻_{i0} · σ⁺_{i0+d} quench to flip the spins at the
//      ends of the q-qbar pair (Phase 4 quench, see quench.hpp).
//   3. Record the post-quench observables (snapshot at t = 0).
//   4. Step TDVP forward by Δt for n_steps = T/Δt steps. After every
//      `snapshotEvery` steps record ⟨L_n⟩(t), ⟨σ^z_n⟩(t), bondDim,
//      and the energy ⟨ψ(t)|H|ψ(t)⟩.
//   5. Optionally compute Schmidt spectra + majorization poset per
//      snapshot — expensive (O(N²) SVDs each), so off by default.
//
// ─── Causal-order comparison (SchwingerQuench::compareCausalOrders) ──────
//
// Phase 5: build three partial orders on the (cut, time) labels
// produced by `evolve()` (with recordSpectra forced on) and report
// pairwise agreement statistics. See causal_compare.hpp for the
// definitions of the three orders.

#pragma once

#include "quantum/causal_compare.hpp"   // CausalComparisonReport
#include "quantum/dmrg_runner.hpp"      // GroundStateResult
#include "quantum/majorization.hpp"     // MajorizationPredicate, Poset
#include "quantum/schmidt.hpp"          // SchmidtSpectra

#include <vector>

namespace tessera::quantum {

// Configuration for a complete DMRG-then-TDVP run. Hamiltonian fields
// mirror QuantumConfig; the rest configure the quench, the TDVP loop,
// and per-step observable recording.
struct TDVPConfig {
    // ─── Hamiltonian (passed to SchwingerParams) ────────────────────────
    int    N{0};
    double a{1.0};
    double m{0.0};
    double g{1.0};
    double L0{0.0};

    // ─── DMRG ground-state setup ────────────────────────────────────────
    int    dmrgMaxBondDim{100};
    int    dmrgNSweeps{12};
    int    dmrgKrylovDim{4};
    double dmrgCutoff{1e-12};

    // ─── q-qbar quench: σ⁻_{i0} · σ⁺_{i0+d} ─────────────────────────────
    int  i0{0};                // first site of the pair, 1-based
    int  d{0};                 // separation, must be odd for the heavy-
                               // quark Néel parity to align (see quench.hpp)
    bool quenchEnforceParity{true};

    // ─── TDVP loop ─────────────────────────────────────────────────────
    double dt{0.05};           // real-time step
    double T{1.0};             // total evolution time (sets n_steps = T/dt)
    int    maxBondDim{200};   // bond-dim cap during TDVP sweeps
    int    krylovDim{12};     // Krylov / Lanczos dimension per local solve
    double cutoff{1e-10};      // SVD truncation per local solve
    int    snapshotEvery{1};  // record observables every k steps (≥ 1)
    bool   quiet{true};
    bool   conserveQns{true};

    // ─── Observable recording ──────────────────────────────────────────
    // recordSpectra = true is needed to have any spectra in snapshots;
    // recordPoset additionally builds the majorization poset on those
    // spectra. recordPoset implies recordSpectra.
    bool recordSpectra{false};
    bool recordPoset{false};
};

// Per-step diagnostics. Schmidt spectra and poset are populated only
// when the corresponding TDVPConfig flags are set.
struct TDVPSnapshot {
    double time{0.0};
    double energy{0.0};               // ⟨ψ|H|ψ⟩ + sm.constant
    int    bondDim{0};               // maxLinkDim(ψ) at this time
    std::vector<double> zProfile;    // ⟨σ^z_n⟩ for n = 1..N
    std::vector<double> lProfile;    // ⟨L_n⟩  for n = 1..N-1
    SchmidtSpectra      spectra;      // populated if cfg.recordSpectra
    Poset               poset;        // populated if cfg.recordPoset
};

// Result bundle from SchwingerQuench::evolve: GS diagnostics, then a
// vector of snapshots starting at t=0 (post-quench, before any TDVP
// steps).
struct QuenchResult {
    GroundStateResult         groundState;
    std::vector<TDVPSnapshot> snapshots;
};

// Coarse-grained Schwinger-model quench + dynamics pipeline.
//
// One instance binds a TDVPConfig; the methods run the full DMRG →
// quench → TDVP loop (`evolve`) or that loop followed by a Phase 5
// causal-order comparison (`compareCausalOrders`). The model is
// stateless beyond its config — every method runs the underlying
// pipeline from scratch.
class SchwingerQuench {
public:
    explicit SchwingerQuench(TDVPConfig config) noexcept;

    [[nodiscard]] TDVPConfig const& config() const noexcept { return config_; }

    // Run the full DMRG → quench → TDVP pipeline. Throws
    // std::invalid_argument on bad config (out-of-range i0, parity
    // mismatch with quenchEnforceParity, etc.).
    [[nodiscard]] QuenchResult evolve() const;

    // End-to-end Phase 5 pipeline: evolve(), then build three partial
    // orders on the (cut, time) labels and compare. Forces
    // `cfg.recordSpectra = true` regardless of the input.
    //
    // `vLr` is the Lieb-Robinson velocity in lattice units (sites /
    // time). Default 1.0 corresponds to the free-fermion group velocity
    // for our hopping coefficient.
    //
    // `predicate` selects the majorization variant for ≼_maj. nullptr
    // means classical {N1999} majorization (StandardMajorization{1e-12}).
    [[nodiscard]] CausalComparisonReport compareCausalOrders(
        double vLr = 1.0,
        MajorizationPredicate const* predicate = nullptr) const;

private:
    TDVPConfig config_;
};

} // namespace tessera::quantum
