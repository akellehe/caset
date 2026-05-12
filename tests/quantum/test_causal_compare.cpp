// Causal-order comparison acceptance test (PLAN.md §5):
//
// Run a TDVP q-qbar quench, build the three partial orders (≼_maj from
// Schmidt-spectrum majorization across (cut, time), ≼_LR from a
// Lieb-Robinson cone with prescribed vLr, ≼_cs from the trivial
// time-only causet on a regular chain), and verify that the comparison
// pipeline produces a sensible report:
//
//   * non-trivial label set,
//   * Kendall-τ in [-1, 1] for every pairwise comparison,
//   * fractions in [0, 1],
//   * vLr-tightening monotonicity:
//     larger vLr ⇒ ≼_LR has more cover edges (subset relation in the
//     opposite direction; agreement with ≼_cs strictly increases or
//     stays the same).
//
// PLAN.md §5 calls for "agreement rate quoted with uncertainty
// from bootstrap over Trotter seeds". We don't bootstrap here (TDVP is
// deterministic given a fixed schedule); the test is a smoke-test of
// the full pipeline plus a vLr-monotonicity sanity check that exposes
// any cone-direction sign error.

#include "quantum/causal_compare.hpp"
#include "quantum/tdvp_runner.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace tessera::quantum;

namespace {

bool in_unit_interval(double x) { return x >= 0.0 && x <= 1.0; }
bool in_kendall_range(double x) { return x >= -1.0 && x <= 1.0; }

// Print one OrderAgreement struct.
void print_agreement(const char* label, OrderAgreement const& a) {
    std::cout << "  " << label
              << "  τ=" << std::setw(8) << a.kendallTau
              << "  disc=" << std::setw(7) << a.discordantFraction
              << "  edit=" << std::setw(7) << a.hasseEditDistance
              << "  (n_both=" << a.nComparableBoth
              << ", concord=" << a.nConcordant
              << ", discord=" << a.nDiscordant << ")\n";
}

bool sanity_check(CausalComparisonReport const& r, const char* tag) {
    std::cout << "  " << tag << ":  nLabels=" << r.nLabels
              << "  nSnapshots=" << r.nSnapshots
              << "  vLr=" << r.vLr << "\n";
    print_agreement("majVsLr", r.majVsLr);
    print_agreement("majVsCs", r.majVsCs);
    print_agreement("lrVsCs ", r.lrVsCs);

    bool ok = r.nLabels > 0 && r.nSnapshots > 1;
    auto check = [&](OrderAgreement const& a, const char* nm) {
        if (!in_kendall_range(a.kendallTau)) {
            std::cout << "    " << nm << " kendallTau out of range\n";
            ok = false;
        }
        if (!in_unit_interval(a.discordantFraction)) {
            std::cout << "    " << nm << " discordantFraction out of range\n";
            ok = false;
        }
        if (!in_unit_interval(a.hasseEditDistance)) {
            std::cout << "    " << nm << " hasseEditDistance out of range\n";
            ok = false;
        }
        if (std::isnan(a.kendallTau) || std::isnan(a.discordantFraction) ||
            std::isnan(a.hasseEditDistance)) {
            std::cout << "    " << nm << " has NaN value\n";
            ok = false;
        }
    };
    check(r.majVsLr, "majVsLr");
    check(r.majVsCs, "majVsCs");
    check(r.lrVsCs,  "lrVsCs");
    return ok;
}

bool monotonicity_test() {
    // Regenerate the same TDVP run and rebuild ≼_LR at multiple vLr
    // values. Larger vLr ⇒ more pairs satisfy distance ≤ vLr · Δt
    // ⇒ ≼_LR has at least as many transitive-closure relations.
    // Operationally: lrVsCs's nComparableBoth should be NON-DECREASING
    // in vLr, since ≼_cs is fixed (time-only).
    std::cout << "\nv_LR-monotonicity: lrVsCs.nComparableBoth ↑ in vLr\n";

    TDVPConfig cfg;
    cfg.N = 10; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0;
    cfg.dmrgMaxBondDim = 32; cfg.dmrgNSweeps = 10;
    cfg.i0 = 3; cfg.d = 3;
    cfg.dt = 0.2; cfg.T = 1.0;
    cfg.snapshotEvery = 1;
    cfg.maxBondDim = 60;
    cfg.cutoff = 1e-10; cfg.krylovDim = 12;
    cfg.quiet = true; cfg.conserveQns = true;

    bool ok = true;
    int prev_n = -1;
    for (double v : {0.5, 1.0, 2.0, 4.0, 16.0}) {
        auto r = SchwingerQuench{cfg}.compareCausalOrders(v);
        std::cout << "  vLr=" << std::setw(5) << v
                  << "  lrVsCs.n_both=" << r.lrVsCs.nComparableBoth
                  << "  edit=" << r.lrVsCs.hasseEditDistance
                  << "  τ=" << r.lrVsCs.kendallTau << "\n";
        if (prev_n >= 0 && r.lrVsCs.nComparableBoth < prev_n) {
            std::cout << "    FAIL: comparability decreased on vLr ↑\n";
            ok = false;
        }
        prev_n = r.lrVsCs.nComparableBoth;
    }
    return ok;
}

bool acceptance() {
    std::cout << "Causal-order comparison acceptance — full pipeline\n";
    std::cout << "----------------------------------------------------\n";

    // Light-quark, modest evolution time so the spectra evolve non-
    // trivially across snapshots and the maj poset has both within-time
    // and cross-time edges.
    TDVPConfig cfg;
    cfg.N = 12; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0;
    cfg.dmrgMaxBondDim = 64; cfg.dmrgNSweeps = 12;
    cfg.i0 = 5; cfg.d = 3;
    cfg.dt = 0.1; cfg.T = 1.0;
    cfg.snapshotEvery = 1;
    cfg.maxBondDim = 80;
    cfg.cutoff = 1e-10; cfg.krylovDim = 12;
    cfg.quiet = true; cfg.conserveQns = true;

    const double vLr = 1.0;       // free-fermion group velocity scale
    auto report = SchwingerQuench{cfg}.compareCausalOrders(vLr);

    bool ok = sanity_check(report, "Light-quark m=0.5, N=12, T=1, dt=0.1");

    if (report.lrVsCs.nComparableBoth <= 0) {
        std::cout << "  FAIL: ≼_LR has no comparable pairs (cone too tight?)\n";
        ok = false;
    }
    if (report.majVsCs.nComparableBoth <= 0) {
        std::cout << "  FAIL: ≼_maj has no edges that ≼_cs sees\n";
        ok = false;
    }
    return ok;
}

} // namespace

int main() {
    std::cout << std::setprecision(6);
    bool ok = true;
    ok &= acceptance();
    ok &= monotonicity_test();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
