// Real-time evolution of a quenched Schwinger-model state via 2-site
// TDVP. PLAN.md §5 Phase 4: thin wrapper around ITensor's tdvp() (from
// the `third_party/itensor_tdvp` submodule), with a per-step observables
// callback that records ⟨L_n⟩(t), ⟨σ^z_n⟩(t), and optionally the full
// Schmidt-spectrum / majorization-poset data the methodology page
// (docs/source/quantum-methodology.md) targets.
//
// ─── End-to-end pipeline ──────────────────────────────────────────────────
//
//   1. Build the Schwinger MPO (Phase 1) and run DMRG to the GS (Phase 2).
//   2. Apply the σ⁻_{i0} · σ⁺_{i0+d} quench to flip the spins at the
//      ends of the q-qbar pair (Phase 4 quench, see quench.hpp).
//   3. Record the post-quench observables (snapshot at t = 0).
//   4. Step TDVP forward by Δt for n_steps = T/Δt steps. After every
//      `snapshotEvery` steps record ⟨L_n⟩(t), ⟨σ^z_n⟩(t), bondDim,
//      and the energy ⟨ψ(t)|H|ψ(t)⟩ (constant if H is time-independent
//      modulo Trotter / truncation error).
//   5. Optionally compute Schmidt spectra + majorization poset per
//      snapshot — expensive (O(N^2) SVDs each), so off by default.
//
// ─── Acceptance (PLAN.md §5 Phase 4) ──────────────────────────────────────
//
//   * Heavy-quark m/g ≫ 1, d odd (e.g. 5), T = d·a:
//     ⟨L_n⟩(t) is approximately +1 (above vacuum) inside the q-qbar
//     interval and 0 outside, to within 0.05 after t = T/2.
//   * Energy conservation: |ΔE|/|E| < 1e-3 over the run.

#pragma once

#include "quantum/dmrg_runner.hpp"
#include "quantum/majorization.hpp"
#include "quantum/schmidt.hpp"

namespace caset::quantum {

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
    int    maxBondDim{200};  // bond-dim cap during TDVP sweeps
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

// Per-step diagnostics. Schmidt spectra and poset are populated only when
// the corresponding TDVPConfig flags are set.
struct TDVPSnapshot {
    double time{0.0};
    double energy{0.0};               // ⟨ψ|H|ψ⟩ + sm.constant
    int    bondDim{0};               // maxLinkDim(ψ) at this time
    std::vector<double> zProfile;    // ⟨σ^z_n⟩ for n = 1..N
    std::vector<double> lProfile;    // ⟨L_n⟩  for n = 1..N-1
    SchmidtSpectra      spectra;      // populated if cfg.recordSpectra
    Poset               poset;        // populated if cfg.recordPoset
};

// Result bundle: GS diagnostics, then a vector of snapshots starting at
// t=0 (post-quench, before any TDVP steps).
struct QuenchResult {
    GroundStateResult         groundState;
    std::vector<TDVPSnapshot> snapshots;
};

// End-to-end pipeline. Throws std::invalid_argument on bad config (out-
// of-range i0, parity mismatch with quenchEnforceParity, etc.).
QuenchResult runQqbarQuench(TDVPConfig const& config);

} // namespace caset::quantum
