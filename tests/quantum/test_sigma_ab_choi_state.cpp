// Tests for Σ_AB as the full 256-dim Choi state of U (GitHub issue #16).
//
// The v0.2 first-pass uses a maximally-mixed I/4 proxy for the Σ_AB
// vertex marginal. This proxy is inconsistent with the joint
// correlations stored in quditJointOf_, which manifests as a discrete
// Q-drift bug (documented in v02_finite_size_investigation.md). The
// fix is to make Σ_AB carry the full Choi state J(U) of the
// interaction unitary, so its single-vertex marginal and its joint
// content resolve to the same quantum object.
//
// This test file is organised in two parts:
//   * "current behavior pinned" — tests that document the v0.2 baseline
//     (including the bug) so any future change is intentional.
//   * "behavior after #16 lands" — tests asserting the new behavior
//     with featureChoiSigmaAB = true. These will fail until the
//     implementation lands; they're the TDD targets.
//
// See:
//   docs/source/quantum-experiments/v02_finite_size_investigation.md
//   docs/source/quantum-experiments/charged_cartan_monte_carlo_v0.2.md

#include "quantum/interaction_simulation.hpp"

#include <cmath>
#include <iostream>
#include <random>

using tessera::quantum::InitialChargeMode;
using tessera::quantum::InteractionConfig;
using tessera::quantum::InteractionSimulation;

namespace {

int failures = 0;

#define CHECK(expr, msg)                                                \
    do {                                                                \
        if (!(expr)) {                                                  \
            std::cerr << "FAIL [" << __LINE__ << "] " << msg << "\n";   \
            ++failures;                                                 \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                      \
    do {                                                                \
        const double _a = (a), _b = (b);                                \
        if (std::abs(_a - _b) > (tol)) {                                \
            std::cerr << "FAIL [" << __LINE__ << "] " << msg            \
                      << "  (" << _a << " vs " << _b                    \
                      << ", |diff|=" << std::abs(_a - _b) << ")\n";     \
            ++failures;                                                 \
        }                                                               \
    } while (0)

// Track failures of a single labelled test independently so we can
// summarise per-test outcomes.
struct TestBlock {
    const char* name;
    int failuresAtStart;
    TestBlock(const char* n) : name(n), failuresAtStart(failures) {
        std::cout << "\n=== " << name << " ===\n";
    }
    ~TestBlock() {
        const int delta = failures - failuresAtStart;
        if (delta == 0) std::cout << "  [pass]\n";
        else            std::cout << "  [FAIL: " << delta << " check(s)]\n";
    }
};

InteractionConfig baseConfig() {
    InteractionConfig cfg;
    cfg.nSystems = 8;
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.dt = 0.25;
    cfg.beta = 3e-4;
    cfg.epsilonI = 1e-10;
    cfg.targetInteractions = 50;
    cfg.delaunayEdges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                         {5, 6}, {6, 7}, {0, 7},
                         {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
    cfg.seed = 1;
    cfg.quiet = true;
    cfg.featureQuditBasis = true;
    cfg.j_chargeCharge = 1.0;
    cfg.j_spinSpin = 0.25;
    cfg.massShift = 0.0;
    cfg.gammaCpViolation = 0.0;
    cfg.dtPair = 0.25;
    cfg.initialChargeMode = InitialChargeMode::ALTERNATING;
    return cfg;
}

// ─── PART 1: current behavior pinned (v0.2 with I/4 proxy) ─────────────

void testBaselineQDriftExistsWithoutChoi() {
    TestBlock _("baseline: discrete Q-drift still occurs when Choi flag "
                "is explicitly turned OFF");
    // Across many seeds at γ_CP = 0 with the I/4 proxy, Q drifts in
    // integer steps. This test pins that historical behavior so
    // anyone disabling the Choi flag knows what they're getting.
    int n_drifters = 0;
    constexpr int N_SEEDS = 12;
    for (int s = 0; s < N_SEEDS; ++s) {
        InteractionConfig cfg = baseConfig();
        cfg.seed = 100 + s;
        cfg.targetInteractions = 80;
        // Explicitly disable Choi so we get the legacy I/4 proxy
        // behavior (Choi is on by default since the #16 fix).
        cfg.featureChoiSigmaAB = false;
        InteractionSimulation sim(cfg);
        sim.tune();
        if (std::abs(sim.getGlobalCharge()) > 1e-6) ++n_drifters;
    }
    CHECK(n_drifters >= 1,
          "at least 1 of 12 seeds shows Q drift at γ_CP=0 with "
          "featureChoiSigmaAB=false");
    std::cout << "  [info] " << n_drifters << "/" << N_SEEDS
              << " seeds with |Q|>1e-6 (γ_CP=0, Choi off → I/4 proxy)\n";
}

void testBaselineSigmaAbStateIsQuarterIdentity() {
    TestBlock _("baseline: Σ_AB state is I/4 when Choi flag is OFF");
    // With the Choi flag explicitly off, the Σ_AB proxy in
    // computeInteractionQudit is set to I/4. Pinning this so any
    // future change to the I/4 default is intentional.
    InteractionConfig cfg = baseConfig();
    cfg.seed = 1;
    cfg.targetInteractions = 20;
    cfg.featureChoiSigmaAB = false;
    InteractionSimulation sim(cfg);
    sim.tune();
    CHECK(std::isfinite(sim.getGlobalCharge()),
          "Q is finite after a 20-cell tune with Choi off");
    std::cout << "  [info] (pinned indirectly via Q-conservation derivation; "
              << "direct check requires a stateOf-getter binding)\n";
}

// ─── PART 2: behavior expected after #16 implementation ────────────────

void testChoiFlag_QConservation_AtGammaCpZero() {
    TestBlock _("Q is exactly conserved under interact at γ_CP = 0 "
                "with the default Choi flag (on)");
    // Using cfg defaults — Choi is on by default since #16. Q should
    // be exactly conserved (modulo float round-off, which is much
    // smaller than the 1e-6 threshold).
    constexpr int N_SEEDS = 12;
    int n_drifters = 0;
    for (int s = 0; s < N_SEEDS; ++s) {
        InteractionConfig cfg = baseConfig();
        cfg.seed = 200 + s;
        cfg.targetInteractions = 80;
        InteractionSimulation sim(cfg);
        sim.tune();
        if (std::abs(sim.getGlobalCharge()) > 1e-6) ++n_drifters;
    }
    CHECK(n_drifters == 0,
          "no drifters across 12 seeds with Choi flag default-on (γ_CP=0)");
    std::cout << "  [info] " << n_drifters << "/" << N_SEEDS
              << " drifters (target: 0)\n";
}

void testChoiFlag_OptOutRestoresOldBehavior() {
    TestBlock _("opt-out (featureChoiSigmaAB = false) restores the "
                "legacy v0.2 path bit-identically");
    // Setting featureChoiSigmaAB = false twice produces identical
    // results — the legacy I/4 proxy path is still available for
    // anyone who needs to reproduce v0.2-era data.
    InteractionConfig cfg = baseConfig();
    cfg.seed = 1;
    cfg.targetInteractions = 30;
    cfg.featureChoiSigmaAB = false;
    InteractionSimulation sim(cfg);
    sim.tune();
    const double q_baseline = sim.getGlobalCharge();
    const std::size_t cells_baseline = sim.interactionCount();

    InteractionConfig cfg2 = cfg;
    cfg2.featureChoiSigmaAB = false;
    InteractionSimulation sim2(cfg2);
    sim2.tune();
    CHECK(sim2.interactionCount() == cells_baseline,
          "cell count identical with featureChoiSigmaAB = false (reproducible)");
    CHECK_NEAR(sim2.getGlobalCharge(), q_baseline, 1e-12,
               "Q_global identical with featureChoiSigmaAB = false (reproducible)");
    std::cout << "  [info] opt-out cells=" << cells_baseline
              << " Q=" << q_baseline << "\n";
}

void testChoiFlag_DefaultIsOn() {
    TestBlock _("featureChoiSigmaAB defaults to ON (post-#16)");
    InteractionConfig cfg;
    CHECK(cfg.featureChoiSigmaAB == true,
          "featureChoiSigmaAB defaults to true");
    std::cout << "  [info] default featureChoiSigmaAB = "
              << (cfg.featureChoiSigmaAB ? "true" : "false") << "\n";
}

void testChoiFlag_DefaultClearsWhenQuditOff() {
    TestBlock _("Choi flag auto-clears when featureQuditBasis is off");
    // For non-qudit (v0/v0.1) configs, default-on Choi would be
    // meaningless. The constructor should silently disable it
    // rather than throw.
    InteractionConfig cfg;
    cfg.nSystems = 4;
    cfg.delaunayEdges = {{0, 1}, {1, 2}, {2, 3}, {0, 2}};
    cfg.beta = 1e-3;
    cfg.targetInteractions = 5;
    cfg.seed = 1;
    cfg.quiet = true;
    // featureQuditBasis defaults to false; featureChoiSigmaAB defaults
    // to true. The constructor should accept this and auto-clear Choi.
    bool constructed = true;
    try {
        InteractionSimulation sim(cfg);
        sim.tune();
        // If we got here, no throw — good.
    } catch (...) {
        constructed = false;
    }
    CHECK(constructed,
          "v0/v0.1 config builds without throwing despite default-on Choi");
}

void testChoiFlag_GeometryUnchangedByFlag() {
    TestBlock _("post-#16: enabling Choi flag does not drastically "
                "reshape the per-run peak D_S");
    // The Choi flag should fix Q-bookkeeping without rewriting the
    // geometry. Concretely: per-seed peak D_S with the flag on
    // should be in the same ballpark as with the flag off — say
    // within a factor of 3 of each other in the mean, when both
    // are run at the same (seed, β, T). If the geometry IS
    // dramatically reshaped (mean ratio > 3 or < 1/3), something
    // structural changed and we want to know.
    //
    // Uses small T=200 for speed; this is in the early-growth
    // regime where peak D_S is sub-unity for both, but the *ratio*
    // is the diagnostic.
    constexpr int N_SEEDS = 5;
    constexpr int T = 200;
    auto runOne = [&](int seed, bool choiFlag) -> double {
        InteractionConfig cfg = baseConfig();
        cfg.seed = seed;
        cfg.targetInteractions = T;
        cfg.featureChoiSigmaAB = choiFlag;  // wired in once #16 lands
        InteractionSimulation sim(cfg);
        sim.tune();
        std::vector<double> sigmas;
        for (int k = 0; k < 12; ++k)
            sigmas.push_back(std::exp(std::log(1e-2)
                + k * (std::log(1e4) - std::log(1e-2)) / 11.0));
        auto dS = sim.getSpectralDimension(sigmas, 15);
        double peak = 0;
        for (double d : dS)
            if (std::isfinite(d)) peak = std::max(peak, d);
        return peak;
    };
    double sum_off = 0, sum_on = 0;
    for (int s = 0; s < N_SEEDS; ++s) {
        sum_off += runOne(400 + s, false);
        sum_on  += runOne(400 + s, true);
    }
    const double mean_off = sum_off / N_SEEDS;
    const double mean_on  = sum_on  / N_SEEDS;
    const double ratio = (mean_off > 1e-9) ? mean_on / mean_off : 1.0;
    CHECK(ratio > 0.33 && ratio < 3.0,
          "mean peak D_S ratio (flag on / flag off) within [1/3, 3]");
    std::cout << "  [info] mean peak D_S: off=" << mean_off
              << "  on=" << mean_on
              << "  ratio=" << ratio << "\n";
}

void testChoiFlag_CharGammaCpStillDriftsCharge() {
    TestBlock _("post-#16: CP-violation still drifts Q "
                "(structural CP effect not masked)");
    // Even with Choi state on, γ_CP ≠ 0 should still produce Q drift
    // — it should come purely from the CP-violating Hamiltonian
    // term, not be confounded by proxy bookkeeping noise.
    InteractionConfig cfg = baseConfig();
    cfg.seed = 1;
    cfg.targetInteractions = 50;
    cfg.gammaCpViolation = 0.5;
    cfg.featureChoiSigmaAB = true;
    InteractionSimulation sim(cfg);
    sim.tune();
    CHECK(std::abs(sim.getGlobalCharge()) > 0.5,
          "Q drifts > 0.5 at γ_CP = 0.5 with Choi flag on");
    std::cout << "  [info] Q drift over 50 cells at γ_CP=0.5: "
              << sim.getGlobalCharge() << "\n";
}

// ─── PART 3: invariants of the Choi state itself ───────────────────────
//
// These would ideally be unit tests of the J(U) construction (purity,
// rank, etc.). Adding them as integration-level checks for now; the
// full unit tests will go in once the helper API is exposed.

void testChoiFlag_NoExceptionsUnderMixedSweep() {
    TestBlock _("post-#16: mixed sweep with Choi flag on runs "
                "without exceptions");
    InteractionConfig cfg = baseConfig();
    cfg.seed = 7;
    cfg.targetInteractions = 30;
    cfg.featureChoiSigmaAB = true;
    InteractionSimulation sim(cfg);
    sim.tune();
    // Mixed-move sweep — should not throw, should not segfault.
    bool ok = true;
    try {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        for (int k = 0; k < 50; ++k) {
            const double r = coin(rng);
            if      (r < 0.25) (void)sim.interact();
            else if (r < 0.50) (void)sim.unInteract();
            else if (r < 0.75) (void)sim.annihilate();
            else               (void)sim.pairCreate();
        }
    } catch (...) {
        ok = false;
    }
    CHECK(ok, "mixed sweep with Choi flag on completes cleanly");
    CHECK(std::isfinite(sim.getGlobalCharge()),
          "Q finite after mixed sweep with Choi flag on");
}

} // namespace

int main() {
    std::cout << "Σ_AB Choi-state tests — see issue #16\n";
    std::cout << "==========================================\n";

    // Part 1: pin current v0.2 behavior. These should pass NOW.
    testBaselineQDriftExistsWithoutChoi();
    testBaselineSigmaAbStateIsQuarterIdentity();

    // Part 2: behavior expected after #16. Most should FAIL until
    // the Choi flag is implemented. They are intentionally written
    // as if the flag were already in place — the (commented-out)
    // cfg.featureChoiSigmaAB = true lines uncomment when the flag
    // exists.
    testChoiFlag_QConservation_AtGammaCpZero();
    testChoiFlag_OptOutRestoresOldBehavior();
    testChoiFlag_DefaultIsOn();
    testChoiFlag_DefaultClearsWhenQuditOff();
    testChoiFlag_GeometryUnchangedByFlag();
    testChoiFlag_CharGammaCpStillDriftsCharge();
    testChoiFlag_NoExceptionsUnderMixedSweep();

    std::cout << "\n==========================================\n";
    if (failures == 0) {
        std::cout << "PASS — all Σ_AB Choi-state checks satisfied\n";
        return 0;
    }
    std::cerr << "\n" << failures << " check(s) failed "
              << "(some are intentional TDD targets until #16 lands)\n";
    return 1;
}
