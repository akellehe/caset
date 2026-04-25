// Implementation of computeGroundStateMajorization (Phase 3 end-to-end
// pipeline). See include/quantum/pipeline.hpp for the architectural
// rationale.
//
// The DMRG-driver fragment here intentionally duplicates the small piece
// of code in src/quantum/dmrg_runner.cpp's computeGroundState(): the two
// functions could share via a `detail::run_dmrg(...)` helper, but keeping
// each function self-contained makes them easier to audit independently
// (and the duplicated piece is ~15 lines of MPS setup).

#include "quantum/pipeline.hpp"
#include "quantum/schwinger_model.hpp"

#include <itensor/all.h>

#include <algorithm>

namespace caset::quantum {

namespace {

itensor::MPS neel_init(itensor::SpinHalf const& sites, int N) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

itensor::Sweeps make_sweeps(QuantumConfig const& cfg) {
    auto sweeps = itensor::Sweeps(cfg.nSweeps);
    const int b = cfg.maxBondDim;
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

GroundStateMajorizationResult
computeGroundStateMajorization(QuantumConfig const& cfg,
                                  double tol) {
    // (1) DMRG ground state — same setup as computeGroundState().
    SchwingerParams p;
    p.N = cfg.N; p.a = cfg.a; p.m = cfg.m; p.g = cfg.g; p.L0 = cfg.L0;

    auto sm    = buildSchwingerMpo(p, cfg.conserveQns);
    auto psi0  = neel_init(sm.sites, p.N);
    auto sweeps = make_sweeps(cfg);
    auto [energy, psi] = itensor::dmrg(
        sm.H, psi0, sweeps,
        itensor::Args("Silent", cfg.quiet));

    GroundStateMajorizationResult out;
    out.groundState.operatorEnergy = energy;
    out.groundState.constant        = sm.constant;
    out.groundState.energy          = energy + sm.constant;
    out.groundState.bondDim        = itensor::maxLinkDim(psi);
    out.groundState.truncationErr  = cfg.cutoff;

    // (2) All contiguous-cut Schmidt spectra. This is N(N+1)/2 - 1 SVDs;
    // for the sizes we test (N ≤ 20) it's well under a second on top of
    // the DMRG run itself.
    out.spectra = allContiguousSpectra(psi);

    // (3) Majorization poset of those spectra (Hasse cover edges only).
    out.poset = majorizationPoset(out.spectra.spectra, tol);

    return out;
}

} // namespace caset::quantum
