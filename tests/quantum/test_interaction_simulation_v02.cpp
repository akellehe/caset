// Test suite for the v0.2 qudit-basis additions to InteractionSimulation.
// See docs/source/quantum-experiments/charged_cartan_monte_carlo_v0.2.md.

#include "simulations/InteractionSimulation.h"

#include <cmath>
#include <iostream>

using tessera::simulations::InitialChargeMode;
using tessera::simulations::InteractionConfig;
using tessera::simulations::InteractionSimulation;

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

InteractionConfig baseConfig() {
    InteractionConfig cfg;
    cfg.nSystems = 8;
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.dt = 0.25;
    cfg.beta = 1e-3;
    cfg.epsilonI = 1e-10;
    cfg.targetInteractions = 12;
    cfg.delaunayEdges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                         {5, 6}, {6, 7}, {0, 7},
                         {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
    cfg.seed = 1;
    cfg.quiet = true;
    // v0.2 defaults: charge-conserving Hamiltonian (γ_CP = 0).
    cfg.featureQuditBasis = true;
    cfg.j_chargeCharge = 1.0;
    cfg.j_spinSpin = 0.25;
    cfg.massShift = 0.0;
    cfg.gammaCpViolation = 0.0;
    cfg.dtPair = 0.25;
    cfg.initialChargeMode = InitialChargeMode::ALTERNATING;
    return cfg;
}

void testBackwardCompat() {
    std::cout << "\n=== featureQuditBasis = false leaves v0.1 unchanged ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.featureQuditBasis = false;
    cfg.useCharges = false;
    InteractionSimulation sim(cfg);
    sim.tune();
    CHECK(sim.interactionCount() == cfg.targetInteractions,
          "v0 path still tunes to target");
    std::cout << "  [ok] v0 unchanged when featureQuditBasis = false\n";
}

void testInitialChargesAreSectorProjected() {
    std::cout << "\n=== v0.2 initial layer: ALTERNATING ± sector projection ===\n";
    InteractionConfig cfg = baseConfig();
    InteractionSimulation sim(cfg);
    // N=8 with ALTERNATING: 4 positive sector (q=+1) + 4 negative
    // sector (q=-1). Q_global = 0 exactly.
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-12,
               "ALTERNATING start: Q_global = 0 in 4-dim qudit basis");
}

void testQConservationWhenCPOff() {
    std::cout << "\n=== Q is exactly conserved when γ_CP = 0 ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.gammaCpViolation = 0.0;
    InteractionSimulation sim(cfg);
    const double q0 = sim.getGlobalCharge();
    sim.tune();
    const double q1 = sim.getGlobalCharge();
    CHECK_NEAR(q0, q1, 1e-9,
               "Q conserved through tune at γ_CP = 0");
    std::cout << "  [info] Q before/after tune: " << q0 << " / " << q1
              << "\n";
}

void testQDriftsWhenCPOn() {
    std::cout << "\n=== Q drifts when γ_CP ≠ 0 (CP-violating Hamiltonian) ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.targetInteractions = 50;
    cfg.gammaCpViolation = 0.5;
    InteractionSimulation sim(cfg);
    const double q0 = sim.getGlobalCharge();
    sim.tune();
    const double q1 = sim.getGlobalCharge();
    // We don't require any particular direction or magnitude — just
    // that Q has moved appreciably from 0.
    CHECK(std::abs(q1 - q0) > 0.5,
          "Q drifts > 0.5 when CP-violation term is on");
    std::cout << "  [info] Q drift over 50 cells at γ_CP=0.5: "
              << (q1 - q0) << "\n";
}

void testNoQ1DriftUnderUnInteract() {
    std::cout << "\n=== v0.2 fixes the v0.1 un-interact Q-drift bug ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.gammaCpViolation = 0.0;
    cfg.targetInteractions = 12;
    InteractionSimulation sim(cfg);
    sim.tune();
    const double q_before = sim.getGlobalCharge();
    // Un-interact the entire complex.
    for (int k = 0; k < 12; ++k) sim.unInteract();
    const double q_after = sim.getGlobalCharge();
    CHECK_NEAR(q_before, q_after, 1e-9,
               "Q stays put under un-interact at γ_CP=0 in qudit basis");
    std::cout << "  [info] Q before/after full un-interact: "
              << q_before << " / " << q_after << "\n";
}

void testSpectralDimensionFinite() {
    std::cout << "\n=== v0.2: spectral dimension is finite and non-saturated ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.targetInteractions = 30;
    InteractionSimulation sim(cfg);
    sim.tune();
    std::vector<double> sigmas;
    for (int s = 0; s < 16; ++s)
        sigmas.push_back(std::exp(std::log(1e-2)
            + s * (std::log(1e4) - std::log(1e-2)) / 15.0));
    auto dS = sim.getSpectralDimension(sigmas, 15);
    double peak = 0.0;
    int nFinite = 0;
    for (double d : dS)
        if (std::isfinite(d)) { ++nFinite; peak = std::max(peak, d); }
    CHECK(nFinite == static_cast<int>(dS.size()),
          "all D_S values finite");
    CHECK(peak > 0.0 && peak < 50.0,
          "peak D_S is a reasonable physical number "
          "(not zero, not pathologically huge)");
    std::cout << "  [info] peak D_S = " << peak << "\n";
}

} // namespace

int main() {
    testBackwardCompat();
    testInitialChargesAreSectorProjected();
    testQConservationWhenCPOff();
    testQDriftsWhenCPOn();
    testNoQ1DriftUnderUnInteract();
    testSpectralDimensionFinite();

    if (failures == 0) {
        std::cout << "\nPASS — all charged-Cartan v0.2 checks satisfied\n";
        return 0;
    }
    std::cerr << "\n" << failures << " v0.2 check(s) failed\n";
    return 1;
}
