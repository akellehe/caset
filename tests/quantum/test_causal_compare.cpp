// Phase 5 acceptance test (PLAN.md §5):
//
// Run a TDVP q-qbar quench, build the three partial orders (≼_maj from
// Schmidt-spectrum majorization across (cut, time), ≼_LR from a
// Lieb-Robinson cone with prescribed v_LR, ≼_cs from the trivial
// time-only causet on a regular chain), and verify that the comparison
// pipeline produces a sensible report:
//
//   * non-trivial label set,
//   * Kendall-τ in [-1, 1] for every pairwise comparison,
//   * fractions in [0, 1],
//   * v_LR-tightening monotonicity:
//     larger v_LR ⇒ ≼_LR has more cover edges (subset relation in the
//     opposite direction; agreement with ≼_cs strictly increases or
//     stays the same).
//
// PLAN.md §5 Phase 5 calls for "agreement rate quoted with uncertainty
// from bootstrap over Trotter seeds". We don't bootstrap here (TDVP is
// deterministic given a fixed schedule); the test is a smoke-test of
// the full pipeline plus a v_LR-monotonicity sanity check that exposes
// any cone-direction sign error.

#include "quantum/causal_compare.hpp"
#include "quantum/tdvp_runner.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace caset::quantum;

namespace {

bool in_unit_interval(double x) { return x >= 0.0 && x <= 1.0; }
bool in_kendall_range(double x) { return x >= -1.0 && x <= 1.0; }

// Print one OrderAgreement struct.
void print_agreement(const char* label, OrderAgreement const& a) {
    std::cout << "  " << label
              << "  τ=" << std::setw(8) << a.kendall_tau
              << "  disc=" << std::setw(7) << a.discordant_fraction
              << "  edit=" << std::setw(7) << a.hasse_edit_distance
              << "  (n_both=" << a.n_comparable_both
              << ", concord=" << a.n_concordant
              << ", discord=" << a.n_discordant << ")\n";
}

bool sanity_check(CausalComparisonReport const& r, const char* tag) {
    std::cout << "  " << tag << ":  n_labels=" << r.n_labels
              << "  n_snapshots=" << r.n_snapshots
              << "  v_LR=" << r.v_LR << "\n";
    print_agreement("maj_vs_lr", r.maj_vs_lr);
    print_agreement("maj_vs_cs", r.maj_vs_cs);
    print_agreement("lr_vs_cs ", r.lr_vs_cs);

    bool ok = r.n_labels > 0 && r.n_snapshots > 1;
    auto check = [&](OrderAgreement const& a, const char* nm) {
        if (!in_kendall_range(a.kendall_tau)) {
            std::cout << "    " << nm << " kendall_tau out of range\n";
            ok = false;
        }
        if (!in_unit_interval(a.discordant_fraction)) {
            std::cout << "    " << nm << " discordant_fraction out of range\n";
            ok = false;
        }
        if (!in_unit_interval(a.hasse_edit_distance)) {
            std::cout << "    " << nm << " hasse_edit_distance out of range\n";
            ok = false;
        }
        if (std::isnan(a.kendall_tau) || std::isnan(a.discordant_fraction) ||
            std::isnan(a.hasse_edit_distance)) {
            std::cout << "    " << nm << " has NaN value\n";
            ok = false;
        }
    };
    check(r.maj_vs_lr, "maj_vs_lr");
    check(r.maj_vs_cs, "maj_vs_cs");
    check(r.lr_vs_cs,  "lr_vs_cs");
    return ok;
}

bool monotonicity_test() {
    // Regenerate the same TDVP run and rebuild ≼_LR at multiple v_LR
    // values. Larger v_LR ⇒ more pairs satisfy distance ≤ v_LR · Δt
    // ⇒ ≼_LR has at least as many transitive-closure relations.
    // Operationally: lr_vs_cs's n_comparable_both should be NON-DECREASING
    // in v_LR, since ≼_cs is fixed (time-only).
    std::cout << "\nv_LR-monotonicity: lr_vs_cs.n_comparable_both ↑ in v_LR\n";

    TDVPConfig cfg;
    cfg.N = 10; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0;
    cfg.dmrg_max_bond_dim = 32; cfg.dmrg_n_sweeps = 10;
    cfg.i0 = 3; cfg.d = 3;
    cfg.dt = 0.2; cfg.T = 1.0;
    cfg.snapshot_every = 1;
    cfg.max_bond_dim = 60;
    cfg.cutoff = 1e-10; cfg.krylov_dim = 12;
    cfg.quiet = true; cfg.conserve_qns = true;

    bool ok = true;
    int prev_n = -1;
    for (double v : {0.5, 1.0, 2.0, 4.0, 16.0}) {
        auto r = compute_causal_comparison(cfg, v);
        std::cout << "  v_LR=" << std::setw(5) << v
                  << "  lr_vs_cs.n_both=" << r.lr_vs_cs.n_comparable_both
                  << "  edit=" << r.lr_vs_cs.hasse_edit_distance
                  << "  τ=" << r.lr_vs_cs.kendall_tau << "\n";
        if (prev_n >= 0 && r.lr_vs_cs.n_comparable_both < prev_n) {
            std::cout << "    FAIL: comparability decreased on v_LR ↑\n";
            ok = false;
        }
        prev_n = r.lr_vs_cs.n_comparable_both;
    }
    return ok;
}

bool acceptance() {
    std::cout << "Phase 5 acceptance — full causal-comparison pipeline\n";
    std::cout << "----------------------------------------------------\n";

    // Light-quark, modest evolution time so the spectra evolve non-
    // trivially across snapshots and the maj poset has both within-time
    // and cross-time edges.
    TDVPConfig cfg;
    cfg.N = 12; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0;
    cfg.dmrg_max_bond_dim = 64; cfg.dmrg_n_sweeps = 12;
    cfg.i0 = 5; cfg.d = 3;
    cfg.dt = 0.1; cfg.T = 1.0;
    cfg.snapshot_every = 1;
    cfg.max_bond_dim = 80;
    cfg.cutoff = 1e-10; cfg.krylov_dim = 12;
    cfg.quiet = true; cfg.conserve_qns = true;

    const double v_LR = 1.0;       // free-fermion group velocity scale
    auto report = compute_causal_comparison(cfg, v_LR);

    bool ok = sanity_check(report, "Light-quark m=0.5, N=12, T=1, dt=0.1");

    if (report.lr_vs_cs.n_comparable_both <= 0) {
        std::cout << "  FAIL: ≼_LR has no comparable pairs (cone too tight?)\n";
        ok = false;
    }
    if (report.maj_vs_cs.n_comparable_both <= 0) {
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
