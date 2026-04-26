// Phase 4 acceptance: heavy-quark q-qbar quench yields a flux tube of
// value +1 above the vacuum on the d links between i0 and i0+d, with the
// surrounding links unchanged. Energy is conserved by TDVP to better
// than 0.1% of the post-quench energy. PLAN.md §5 Phase 4 specifies
// this benchmark verbatim.
//
// We use d = 5 (odd) instead of the plan's d = 4: the σ⁻ σ⁺ quench
// only acts non-trivially on the heavy-quark Néel when the two ends
// land on opposite sublattices, which requires d odd. Five lattice
// spacings is the closest match to the plan's d = 4 that satisfies
// the parity constraint — see include/quantum/quench.hpp.

#include "quantum/quench.hpp"
#include "quantum/schwinger_model.hpp"
#include "quantum/tdvp_runner.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace tessera::quantum;

namespace {

// Pretty-print a per-link / per-site profile next to the expected
// reference values. Helps diagnose flux-tube failures.
void print_profile(const char* label,
                   std::vector<double> const& values,
                   std::vector<double> const& reference) {
    std::cout << "  " << label << ":\n   ";
    for (std::size_t k = 0; k < values.size(); ++k) {
        std::cout << std::setw(7) << std::fixed << std::setprecision(3)
                  << values[k];
    }
    std::cout << "\n   ref:";
    for (std::size_t k = 0; k < reference.size(); ++k) {
        std::cout << std::setw(7) << std::fixed << std::setprecision(3)
                  << reference[k];
    }
    std::cout << "\n";
}

// Vacuum L_n profile in the heavy-quark limit (|↑↓↑↓ … ⟩ Néel) with
// L0 = 0: alternating −1 (odd link), 0 (even link). The first index
// of the returned vector is link n=1.
std::vector<double> heavy_quark_vacuum_L_profile(int N) {
    std::vector<double> v;
    v.reserve(static_cast<std::size_t>(std::max(N - 1, 0)));
    for (int n = 1; n <= N - 1; ++n) {
        v.push_back((n % 2 == 1) ? -1.0 : 0.0);
    }
    return v;
}

// Expected post-quench L_n profile for a +1 flux tube on links
// [i0, i0 + d - 1]: vacuum value plus 1 on those links, vacuum elsewhere.
std::vector<double> expected_flux_tube_L_profile(int N, int i0, int d) {
    auto v = heavy_quark_vacuum_L_profile(N);
    for (int n = i0; n <= i0 + d - 1; ++n) {
        v[static_cast<std::size_t>(n - 1)] += 1.0;
    }
    return v;
}

bool flux_tube_test() {
    std::cout << "Heavy-quark flux-tube test (m/g ≫ 1, d = 5)\n";
    std::cout << "--------------------------------------------\n";

    TDVPConfig cfg;
    cfg.N    = 14;
    cfg.a    = 1.0; cfg.g = 1.0;
    cfg.m    = 20.0;       // heavy-quark limit
    cfg.L0   = 0.0;
    cfg.dmrgMaxBondDim = 64;
    cfg.dmrgNSweeps     = 12;
    cfg.dmrgKrylovDim   = 4;
    cfg.dmrgCutoff       = 1e-12;

    cfg.i0 = 5;            // odd → Up site in the heavy-quark vacuum
    cfg.d  = 5;            // odd → i0 + d = 10 is even → Dn site

    cfg.dt              = 0.05;
    cfg.T               = static_cast<double>(cfg.d) * cfg.a;  // T = d·a = 5
    cfg.maxBondDim    = 100;
    cfg.cutoff          = 1e-10;
    cfg.krylovDim      = 12;
    cfg.snapshotEvery  = 5;
    cfg.quiet           = true;
    cfg.conserveQns    = true;
    // Spectrum / poset recording is expensive and not needed for the
    // flux-tube test; leave them off.
    cfg.recordSpectra  = false;
    cfg.recordPoset    = false;

    auto result = runQqbarQuench(cfg);

    // Sanity: we always get an initial snapshot (t = 0) and at least
    // one mid-run snapshot before the final.
    if (result.snapshots.size() < 3) {
        std::cout << "FAIL: too few snapshots ("
                  << result.snapshots.size() << ")\n";
        return false;
    }

    auto const& snap0    = result.snapshots.front();    // post-quench, t=0
    auto const& snap_mid = result.snapshots[result.snapshots.size() / 2];
    auto const& snap_end = result.snapshots.back();

    const auto reference =
        expected_flux_tube_L_profile(cfg.N, cfg.i0, cfg.d);

    // ── (1) Initial post-quench profile must already be the flux tube.
    // The σ⁻ σ⁺ operator is exact at finite m, so this is a tight
    // sanity check (tolerance set by GS-vs-Néel admixture, ~ O((1/m)²)).
    bool initial_ok = true;
    constexpr double initial_tol = 0.05;
    for (int n = 1; n <= cfg.N - 1; ++n) {
        const double diff = std::abs(
            snap0.lProfile[static_cast<std::size_t>(n - 1)]
            - reference[static_cast<std::size_t>(n - 1)]);
        if (diff > initial_tol) initial_ok = false;
    }
    std::cout << "Initial post-quench ⟨L_n⟩ vs heavy-quark flux-tube reference\n";
    print_profile("L(t=0)", snap0.lProfile, reference);
    std::cout << "  " << (initial_ok ? "PASS" : "FAIL")
              << "  (tol " << initial_tol << ")\n\n";

    // ── (2) Mid-run profile must still match the flux tube to within
    // 0.05 (PLAN.md §5 Phase 4 spec) at t = T/2.
    bool mid_ok = true;
    constexpr double mid_tol = 0.05;
    for (int n = 1; n <= cfg.N - 1; ++n) {
        const double diff = std::abs(
            snap_mid.lProfile[static_cast<std::size_t>(n - 1)]
            - reference[static_cast<std::size_t>(n - 1)]);
        if (diff > mid_tol) mid_ok = false;
    }
    std::cout << "Mid-run ⟨L_n⟩ at t = " << snap_mid.time
              << " (≈ T/2 = " << cfg.T / 2 << ")\n";
    print_profile("L(t=T/2)", snap_mid.lProfile, reference);
    std::cout << "  " << (mid_ok ? "PASS" : "FAIL")
              << "  (tol " << mid_tol << ")\n\n";

    // ── (3) Energy conservation: |E(T) - E(0)| / |E(0)| < 1e-3.
    const double E0   = snap0.energy;
    const double Eend = snap_end.energy;
    const double rel  = std::abs((Eend - E0) / E0);
    constexpr double energy_tol = 1e-3;
    const bool energy_ok = rel < energy_tol;
    std::cout << "Energy conservation: E(0)=" << E0
              << "  E(T)=" << Eend
              << "  |ΔE|/|E0|=" << rel
              << "  " << (energy_ok ? "PASS" : "FAIL")
              << "  (tol " << energy_tol << ")\n";

    return initial_ok && mid_ok && energy_ok;
}

} // namespace

int main() {
    std::cout << std::setprecision(8);
    bool ok = true;
    ok &= flux_tube_test();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
