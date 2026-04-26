// Implementation of runQqbarQuench (Phase 4 end-to-end pipeline).
// See include/quantum/tdvp_runner.hpp for the architectural narrative.

#include "quantum/tdvp_runner.hpp"
#include "quantum/quench.hpp"
#include "quantum/schwinger_model.hpp"

#include "tdvp.h"  // ITensor TDVP add-on, vendored under third_party/itensor_tdvp

#include <itensor/all.h>

#include <algorithm>
#include <cmath>

namespace tessera::quantum {

namespace {

// ─── Observable helpers ───────────────────────────────────────────────────

// ⟨σ^z_n⟩ for every site n = 1..N. Uses the standard ITensor pattern of
// orthogonalising at site n and contracting bra × Sz × ket; multiplies
// the result by 2 to convert from ITensor's "Sz" (= ½ σ^z) to bare σ^z.
// Result is real because σ^z is Hermitian and the wavefunction stays
// normalised under TDVP (we drop the imaginary noise via .real()).
std::vector<double> sigma_z_profile(itensor::MPS const& psi_in,
                                    itensor::SpinHalf const& sites) {
    using namespace itensor;
    auto psi = psi_in;
    const int N = length(psi);
    std::vector<double> out(static_cast<std::size_t>(N), 0.0);
    for (int n = 1; n <= N; ++n) {
        psi.position(n);
        auto Sz  = op(sites, "Sz", n);
        auto bra = dag(psi(n));
        bra.prime("Site");
        const auto val = eltC(bra * Sz * psi(n));
        out[static_cast<std::size_t>(n - 1)] = 2.0 * val.real();
    }
    return out;
}

// ⟨L_n⟩ for every link n = 1..N-1 from the σ^z profile. Closed form:
//   ⟨L_n⟩  =  c_n  -  ½ Σ_{k=1..n} ⟨σ^z_k⟩
// with c_n = L0 + ((-1)^n - 1)/4 (matches schwinger_model.cpp).
std::vector<double> L_profile_from_sz(std::vector<double> const& sz,
                                      double L0) {
    const int N = static_cast<int>(sz.size());
    std::vector<double> L(static_cast<std::size_t>(std::max(N - 1, 0)), 0.0);
    double cum = 0.0;
    for (int n = 1; n <= N - 1; ++n) {
        cum += sz[static_cast<std::size_t>(n - 1)];
        const double c_n = L0 + ((n % 2 == 0) ? 0.0 : -0.5);
        L[static_cast<std::size_t>(n - 1)] = c_n - 0.5 * cum;
    }
    return L;
}

double compute_energy(itensor::MPS const& psi, itensor::MPO const& H,
                      double constant_shift) {
    return std::real(itensor::innerC(psi, H, psi)) + constant_shift;
}

TDVPSnapshot make_snapshot(double t,
                           itensor::MPS const& psi,
                           SchwingerMPO const& sm,
                           SchwingerParams const& p,
                           bool recordSpectra,
                           bool recordPoset) {
    TDVPSnapshot snap;
    snap.time      = t;
    snap.energy    = compute_energy(psi, sm.H, sm.constant);
    snap.bondDim  = itensor::maxLinkDim(psi);
    snap.zProfile = sigma_z_profile(psi, sm.sites);
    snap.lProfile = L_profile_from_sz(snap.zProfile, p.L0);
    if (recordSpectra || recordPoset) {
        snap.spectra = allContiguousSpectra(psi);
        if (recordPoset) {
            snap.poset = majorizationPoset(snap.spectra.spectra);
        }
    }
    return snap;
}

// ─── DMRG / TDVP sweep schedules ──────────────────────────────────────────

itensor::Sweeps make_dmrg_sweeps(int max_bond, int nSweeps,
                                 int krylov, double cutoff) {
    auto sweeps = itensor::Sweeps(nSweeps);
    sweeps.maxdim() = std::min(20, max_bond),
                      std::min(40, max_bond),
                      std::min(80, max_bond),
                      max_bond, max_bond;
    sweeps.cutoff() = cutoff;
    sweeps.niter()  = krylov;
    sweeps.noise()  = 1e-7, 1e-8, 0.0;
    return sweeps;
}

itensor::Sweeps make_tdvp_sweeps(int max_bond, int krylov, double cutoff) {
    // 1 sweep per call to tdvp() — we drive the schedule externally by
    // calling tdvp() once per Δt step.
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = max_bond;
    sweeps.cutoff() = cutoff;
    sweeps.niter()  = krylov;
    return sweeps;
}

itensor::MPS neel_init(itensor::SpinHalf const& sites, int N) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

} // namespace

QuenchResult runQqbarQuench(TDVPConfig const& cfg) {
    // (1) Build the Schwinger MPO and run DMRG to the GS — same setup
    // as computeGroundState(), reproduced here so the runner is self-
    // contained.
    SchwingerParams p;
    p.N = cfg.N; p.a = cfg.a; p.m = cfg.m; p.g = cfg.g; p.L0 = cfg.L0;

    auto sm = buildSchwingerMpo(p, cfg.conserveQns);
    auto psi0 = neel_init(sm.sites, cfg.N);
    auto sweeps_dmrg = make_dmrg_sweeps(
        cfg.dmrgMaxBondDim, cfg.dmrgNSweeps,
        cfg.dmrgKrylovDim, cfg.dmrgCutoff);
    auto [E_gs, psi_gs] = itensor::dmrg(
        sm.H, psi0, sweeps_dmrg,
        itensor::Args("Silent", cfg.quiet));

    QuenchResult result;
    result.groundState.operatorEnergy = E_gs;
    result.groundState.constant        = sm.constant;
    result.groundState.energy          = E_gs + sm.constant;
    result.groundState.bondDim        = itensor::maxLinkDim(psi_gs);
    result.groundState.truncationErr  = cfg.dmrgCutoff;

    // (2) Apply the q-qbar quench (Phase 4 ≈ Buyens 2014 string state).
    auto psi = applyQqbarQuench(psi_gs, sm.sites, cfg.i0, cfg.d,
                                  cfg.quenchEnforceParity);

    // (3) Initial snapshot: t = 0 means "right after the quench, before
    // any TDVP step". Energy here is generally above the GS energy by
    // the quench excitation cost.
    result.snapshots.push_back(make_snapshot(
        /*t=*/0.0, psi, sm, p,
        cfg.recordSpectra, cfg.recordPoset));

    // (4) TDVP loop. Real-time evolution e^{-i H Δt} corresponds to
    // ITensor's tdvp(...) with the time argument t = -i Δt.
    auto sweeps_tdvp = make_tdvp_sweeps(
        cfg.maxBondDim, cfg.krylovDim, cfg.cutoff);
    const itensor::Cplx t_step{0.0, -cfg.dt};
    const auto tdvp_args = itensor::Args(
        "Truncate",     true,
        "DoNormalize",  true,
        // "Silent" implies Quiet + PrintEigs=false in TDVPWorker (see
        // third_party/itensor_tdvp/tdvp.h) — needed to suppress the
        // per-sweep "vN Entropy" diagnostic that "Quiet" alone leaves on.
        "Silent",       cfg.quiet,
        "NumCenter",    2,        // 2-site TDVP — grows bond dim adaptively
        "ErrGoal",      1e-7);

    const int n_steps = static_cast<int>(std::round(cfg.T / cfg.dt));
    for (int step = 1; step <= n_steps; ++step) {
        itensor::tdvp(psi, sm.H, t_step, sweeps_tdvp, tdvp_args);
        // Take a snapshot every `snapshotEvery` steps and always at the
        // very last step so callers see the final state.
        if (step % cfg.snapshotEvery == 0 || step == n_steps) {
            const double current_t = step * cfg.dt;
            result.snapshots.push_back(make_snapshot(
                current_t, psi, sm, p,
                cfg.recordSpectra, cfg.recordPoset));
        }
    }

    return result;
}

} // namespace tessera::quantum
