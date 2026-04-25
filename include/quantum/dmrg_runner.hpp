// Thin wrapper around ITensor's dmrg() for the Schwinger model.
//
// PLAN.md §5 Phase 2: a self-contained ground-state driver that takes a
// flat config struct and returns the bare numbers we want to expose to
// Python — no MPS, no MPO, no ITensor types cross the API boundary. This
// keeps the Python side a pure result viewer in line with the architectural
// principle in PLAN.md §1 ("minimize Python/C++ crossings").
//
// What this wrapper does:
//   1. Builds the SchwingerMPO from the dimensional parameters in
//      QuantumConfig (delegating to schwinger_model.cpp).
//   2. Sets up a Néel |↑↓↑↓…⟩ initial MPS in the Sz=0 sector so DMRG stays
//      charge-neutral when QN conservation is enabled.
//   3. Runs ITensor's dmrg() with a sweep schedule built from QuantumConfig
//      (max bond dim, number of sweeps, truncation cutoff, Krylov dim).
//   4. Reads back the final operator energy ⟨H⟩, the c-number constant
//      from L_n² expansion, the achieved bond dim, and the largest
//      truncation error reported by the last sweep.

#pragma once

#include "quantum/schwinger_model.hpp"

namespace caset::quantum {

// Flat, Python-friendly configuration. Holds Hamiltonian parameters plus
// DMRG sweep settings. Defaults are tuned for the small / moderate N runs
// we use in tests; production callers should override maxBondDim and
// nSweeps for tighter convergence.
struct QuantumConfig {
    // ─── Hamiltonian (passed straight to SchwingerParams) ───────────────
    int    N{0};       // staggered sites, 1-based; must be ≥ 2
    double a{1.0};     // lattice spacing
    double m{0.0};     // bare fermion mass
    double g{1.0};     // gauge coupling
    double L0{0.0};    // background electric field on the link left of site 1

    // ─── DMRG ──────────────────────────────────────────────────────────
    int    maxBondDim{100};  // cap on MPS bond dim during sweeps
    int    nSweeps{12};       // total sweep count
    double cutoff{1e-12};      // SVD truncation threshold per local solve
    int    krylovDim{4};      // Lanczos / Krylov dimension per local solve
    bool   quiet{true};        // suppress ITensor's per-sweep diagnostics
    bool   conserveQns{true}; // U(1) total-Sz conservation on the SiteSet

    // ─── Phase 4 (TDVP / quench) parameters; unused by computeGroundState.
    // Carried in this struct to match PLAN.md §6's exposed API surface.
    double dt{0.01};   // real-time step size
    double T{1.0};     // total evolution time
};

// What computeGroundState returns. PLAN.md §6 specifies just
// (energy, bondDim, truncationErr); we additionally expose `constant`
// and `operatorEnergy` because callers comparing against published
// numerics need to know whether the value they have is the operator-only
// part or the full physical energy.
struct GroundStateResult {
    double energy{0.0};          // ⟨H⟩ + constant — the full physical energy
    double operatorEnergy{0.0}; // ⟨H⟩ alone — what ITensor's dmrg() returned
    double constant{0.0};        // c-number shift from L_n² expansion
    int    bondDim{0};          // achieved max bond dim of the optimized MPS
    double truncationErr{0.0};  // largest truncation error in the final sweep
};

// Run DMRG to the ground state at the parameters in `config`. Throws
// std::invalid_argument for invalid Hamiltonian parameters (forwarded
// from buildSchwingerMpo).
GroundStateResult computeGroundState(QuantumConfig const& config);

} // namespace caset::quantum
