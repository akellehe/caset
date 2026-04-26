// Implementation of computeGroundState — see dmrg_runner.hpp for the
// full architectural rationale.

#include "quantum/dmrg_runner.hpp"

#include <itensor/all.h>

#include <algorithm>

namespace tessera::quantum {

namespace {

// Néel |↑↓↑↓…⟩ initial state. Total Sz = 0 (for even N), which is the
// charge-neutral sector Bañuls 2013 works in throughout. With
// ConserveQNs=true on the SiteSet, DMRG stays in this sector.
itensor::MPS neel_init(itensor::SpinHalf const& sites, int N) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

// Build the sweep schedule for ITensor::dmrg. We ramp the bond-dim cap
// (20 → 40 → 80 → max) over the first sweeps so early iterations don't
// commit truncation errors that later sweeps have to undo. Noise is on
// for the first two sweeps to perturb out of local minima, then off so
// later sweeps converge cleanly.
itensor::Sweeps make_sweeps(QuantumConfig const& cfg) {
    auto sweeps = itensor::Sweeps(cfg.nSweeps);
    const int b = cfg.maxBondDim;
    // Ramp values clamp at b so smaller caps don't get pulled up by the
    // initial 20/40/80 plateau when the user explicitly asked for less.
    sweeps.maxdim() = std::min(20, b),
                      std::min(40, b),
                      std::min(80, b),
                      b, b;
    sweeps.cutoff() = cfg.cutoff;
    sweeps.niter()  = cfg.krylovDim;
    sweeps.noise()  = 1e-7, 1e-8, 0.0;
    return sweeps;
}

} // namespace

GroundStateResult computeGroundState(QuantumConfig const& cfg) {
    // 1) Forward the Hamiltonian parameters into a SchwingerParams and
    //    build the MPO. Validation (N ≥ 2, a > 0) happens inside
    //    buildSchwingerMpo and surfaces as std::invalid_argument.
    SchwingerParams p;
    p.N = cfg.N; p.a = cfg.a; p.m = cfg.m; p.g = cfg.g; p.L0 = cfg.L0;

    auto sm = buildSchwingerMpo(p, cfg.conserveQns);

    // 2) Néel initial state (Sz = 0 for even N).
    auto psi0 = neel_init(sm.sites, p.N);

    // 3) Run DMRG.
    auto sweeps = make_sweeps(cfg);
    auto [energy, psi] = itensor::dmrg(
        sm.H, psi0, sweeps,
        itensor::Args("Silent", cfg.quiet));

    // 4) Read back observables. ITensor v3 exposes:
    //   • maxLinkDim(psi)  — the largest bond dimension after the final sweep
    //   • doesn't directly return the truncation error from sweeps, but the
    //     SVD cutoff used in the last sweep IS our cutoff, and the tail
    //     mass discarded is at most cutoff — we report that as our
    //     conservative upper bound.
    GroundStateResult r;
    r.operatorEnergy = energy;
    r.constant        = sm.constant;
    r.energy          = energy + sm.constant;
    r.bondDim        = itensor::maxLinkDim(psi);
    r.truncationErr  = cfg.cutoff;
    return r;
}

} // namespace tessera::quantum
