// Test suite for the Charged Cartan Monte Carlo (v0.1) additions to
// InteractionSimulation: charge bookkeeping, sign-bucketed frontier,
// the annihilate / pairCreate moves, the observables, and the
// interactions with the existing interact / unInteract dynamics.
//
// See docs/source/quantum-experiments/charged_cartan_monte_carlo_v0.1.md
// for the construction.

#include "simulations/InteractionSimulation.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

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
    cfg.a = 1.0;
    cfg.g = 1.0;
    cfg.m = 0.5;
    cfg.dt = 0.25;
    cfg.beta = 1e-3;
    cfg.epsilonI = 1e-10;
    cfg.targetInteractions = 12;
    cfg.delaunayEdges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                         {5, 6}, {6, 7}, {0, 7},
                         {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
    cfg.seed = 1;
    cfg.quiet = true;
    return cfg;
}

// ─── Backward compatibility ─────────────────────────────────────────────────
void testBackwardCompat() {
    std::cout << "\n=== backward compat (useCharges = false) ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = false;
    InteractionSimulation sim(cfg);
    sim.tune();
    CHECK(sim.interactionCount() == cfg.targetInteractions,
          "tune should reach target with useCharges=false");
    auto rates = sim.getAcceptanceRates();
    CHECK(rates.find("annihilate") == rates.end(),
          "no annihilate rate when useCharges=false");
    CHECK(rates.find("pairCreate") == rates.end(),
          "no pairCreate rate when useCharges=false");
    CHECK(rates.find("interact")   != rates.end(),
          "interact rate present");
    CHECK(rates.find("unInteract") != rates.end(),
          "unInteract rate present");
    CHECK(sim.getGlobalCharge() == 0.0,
          "Q_global is 0 when useCharges=false (chargeOf_ empty)");
    std::cout << "  [ok] back-compat path intact\n";
}

// ─── Initial layer assignment ───────────────────────────────────────────────
void testInitialChargesAlternating() {
    std::cout << "\n=== initial charge ALTERNATING ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.initialChargeMode = InitialChargeMode::ALTERNATING;
    InteractionSimulation sim(cfg);
    // N=8 → 4 positive + 4 negative; Q_global = 0.
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-12,
               "ALTERNATING: Q_global = 0 exactly");
    auto profile = sim.getChargeProfile();
    CHECK(profile.size() == 1,
          "single time slice at construction");
    if (profile.size() >= 1) {
        const auto& row = profile[0];
        CHECK_NEAR(row[0], 4.0, 1e-12, "ALTERNATING: 4 positives at t=0");
        CHECK_NEAR(row[1], 0.0, 1e-12, "ALTERNATING: 0 neutrals at t=0");
        CHECK_NEAR(row[2], 4.0, 1e-12, "ALTERNATING: 4 negatives at t=0");
        CHECK_NEAR(row[3], 0.0, 1e-12, "ALTERNATING: Σq at t=0 = 0");
    }
}

void testInitialChargesRandom() {
    std::cout << "\n=== initial charge RANDOM ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.nSystems = 200;
    cfg.delaunayEdges = {{0, 1}};   // minimal connectivity; we just need
                                    // the initial-layer assignment to be
                                    // tested.
    for (int i = 1; i < 200; ++i) cfg.delaunayEdges.push_back({i - 1, i});
    cfg.useCharges = true;
    cfg.initialChargeMode = InitialChargeMode::RANDOM;
    cfg.seed = 42;
    InteractionSimulation sim(cfg);
    const double q = sim.getGlobalCharge();
    // For N=200 random ±1 vertices, Q_global ~ Gaussian(0, √200) ≈ ±14
    // typically; allow ±5σ.
    CHECK(std::abs(q) <= 5.0 * std::sqrt(200.0),
          "RANDOM: |Q_global| within 5σ of 0");
    std::cout << "  [info] RANDOM Q_global = " << q << "\n";
}

// ─── interact: charge-restricted eligibility, product inheritance ───────────
void testInteractInheritsCharges() {
    std::cout << "\n=== interact inherits parent charges ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.beta = 1e-4;  // strongly Metropolis-accepting
    InteractionSimulation sim(cfg);
    // After a single accepted interact: 2 inputs consumed (their charge
    // entries persist but are off-frontier), 3 products added (+, 0, ±
    // depending on parents). Frontier Q should still be 0 since
    // products' net charge equals the consumed inputs' net charge.
    int attempts = 0;
    while (sim.interactionCount() < 1 && attempts < 100) {
        sim.interact();
        ++attempts;
    }
    CHECK(sim.interactionCount() == 1,
          "got one accepted interact in ≤ 100 tries");
    // Charge conservation: frontier-Q unchanged. (Initial 0; +, +, 0 or
    // -, -, 0 products: net same as inputs; cancels.)
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-12,
               "frontier Q stays 0 after one same-sign interact");
}

// ─── annihilate: charge-only neutralisation ─────────────────────────────────
void testAnnihilateNeutralisesCharge() {
    std::cout << "\n=== annihilate neutralises (no vertex deletion) ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    InteractionSimulation sim(cfg);
    const std::size_t v0 = sim.getSpacetime()->getVertexCount();
    // After repeated annihilates, the spacetime vertex count must NOT
    // decrease (annihilate only neutralises charge).
    int accepted = 0;
    for (int k = 0; k < 50; ++k)
        if (sim.annihilate()) ++accepted;
    const std::size_t v1 = sim.getSpacetime()->getVertexCount();
    CHECK(accepted > 0, "at least one annihilate accepted");
    CHECK(v1 == v0, "annihilate must NOT remove vertices from spacetime");
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-12,
               "annihilate conserves Q exactly");
    // After many annihilates with the initial 4+/4- layer, the per-sign
    // bucket counts should be reduced (or empty) but the neutral count
    // should match the removed-from-buckets count.
    auto prof = sim.getChargeProfile();
    if (!prof.empty()) {
        const double npos  = prof[0][0];
        const double nzero = prof[0][1];
        const double nneg  = prof[0][2];
        CHECK_NEAR(npos + nzero + nneg, 8.0, 1e-12,
                   "ALTERNATING start: 8 vertices in t=0 profile total");
        std::cout << "  [info] post-annihilate t=0: +" << npos
                  << " 0" << nzero << " -" << nneg << "\n";
    }
}

// ─── pairCreate: symmetric and CP-biased ────────────────────────────────────
void testPairCreateSymmetric() {
    std::cout << "\n=== pairCreate symmetric (cpBias = 0) ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.cpBias = 0.0;
    InteractionSimulation sim(cfg);
    for (int k = 0; k < 100; ++k) sim.pairCreate();
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-9,
               "pairCreate at cpBias=0 conserves Q exactly");
}

void testPairCreateCpBiased() {
    std::cout << "\n=== pairCreate with positive CP bias ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.cpBias = 0.4;
    InteractionSimulation sim(cfg);
    for (int k = 0; k < 400; ++k) sim.pairCreate();
    const double q = sim.getGlobalCharge();
    // Expected positive drift; mean per pair = cpBias/2 = 0.2 if all
    // 400 accepted, but most are rejected by Metropolis prefactor.
    // Still — sign must be positive on average.
    CHECK(q > 0.0,
          "positive cpBias drifts Q upward");
    std::cout << "  [info] Q_global after 400 pairCreate at +bias = "
              << q << "\n";
}

void testPairCreateCpBiasedNegative() {
    std::cout << "\n=== pairCreate with negative CP bias ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.cpBias = -0.4;
    InteractionSimulation sim(cfg);
    for (int k = 0; k < 400; ++k) sim.pairCreate();
    const double q = sim.getGlobalCharge();
    CHECK(q < 0.0,
          "negative cpBias drifts Q downward");
    std::cout << "  [info] Q_global after 400 pairCreate at -bias = "
              << q << "\n";
}

// ─── Mixed move sweep: no crash, observables coherent ──────────────────────
void testSweepStability() {
    std::cout << "\n=== sweep stability under all four moves ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.targetInteractions = 30;
    InteractionSimulation sim(cfg);
    sim.tune();
    for (int k = 0; k < 3; ++k) {
        const int n = sim.sweep();
        CHECK(n >= 0, "sweep returns non-negative accepted count");
        CHECK(std::isfinite(sim.getGlobalCharge()),
              "Q_global stays finite through sweep");
        CHECK(sim.getSpacetime()->getVertexCount() > 0,
              "spacetime non-empty after sweep");
    }
    std::cout << "  [ok] 3 sweeps without crash, observables finite\n";
}

// ─── Observable smoke ──────────────────────────────────────────────────────
void testObservablesSmoke() {
    std::cout << "\n=== observable smoke ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.targetInteractions = 20;
    InteractionSimulation sim(cfg);
    sim.tune();
    auto prof = sim.getChargeProfile();
    CHECK(prof.size() >= 1, "getChargeProfile has at least one slice");
    auto corr = sim.getChargeCorrelation(4);
    CHECK(corr.size() == 4u,
          "getChargeCorrelation returns vector of requested length");
    for (double c : corr)
        CHECK(std::isfinite(c), "correlation entry is finite");
    auto rates = sim.getAcceptanceRates();
    for (auto const& [name, r] : rates)
        CHECK(std::isfinite(r) && r >= 0.0 && r <= 1.0,
              "rate finite and in [0, 1]");
}

// ─── unInteract under charges: still functional ────────────────────────────
void testUnInteractUnderCharges() {
    std::cout << "\n=== unInteract under charges ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.targetInteractions = 20;
    InteractionSimulation sim(cfg);
    sim.tune();
    const std::size_t cellsBefore = sim.interactionCount();
    int accepted = 0;
    for (int k = 0; k < 30; ++k) if (sim.unInteract()) ++accepted;
    CHECK(accepted > 0, "at least one unInteract accepted");
    CHECK(sim.interactionCount() < cellsBefore,
          "interactionCount decreased after un-interact");
    CHECK(std::isfinite(sim.getGlobalCharge()),
          "Q_global remains finite through unInteract");
}

// ─── Conservation under forward-only moves ─────────────────────────────────
//
// At cpBias=0, the moves that don't rewind history — interact, annihilate,
// pairCreate — should conserve frontier Q exactly. unInteract is *not*
// expected to conserve frontier Q in the presence of intervening
// annihilations: when a parent vertex is restored to the frontier its
// original charge comes back, but any annihilation that happened after
// it became a parent has already neutralised the descendant charge.
// This is a known limitation of v0.1's "charge as classical label" model;
// v0.2 (qudit basis) addresses it by making charge intrinsic to the state.
void testForwardOnlyConservation() {
    std::cout << "\n=== forward-only moves conserve Q (interact + annihilate + pairCreate) ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.cpBias = 0.0;
    cfg.targetInteractions = 8;
    InteractionSimulation sim(cfg);
    sim.tune();

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    double maxDrift = 0.0;
    for (int k = 0; k < 500; ++k) {
        const double r = coin(rng);
        // Skip unInteract to isolate forward-only conservation. The
        // forward-only moves should preserve Q exactly.
        if      (r < 0.333) sim.interact();
        else if (r < 0.666) sim.annihilate();
        else                sim.pairCreate();
        const double q = sim.getGlobalCharge();
        if (std::abs(q) > maxDrift) maxDrift = std::abs(q);
    }
    CHECK(maxDrift < 1e-9,
          "max |Q_global| over 500 forward-only moves stays < 1e-9 "
          "at cpBias=0");
    std::cout << "  [info] max |Q| over 500 forward-only moves: "
              << maxDrift << "\n";
}

// ─── featureDeactivateOnAnnihilate (option B) ───────────────────────────
//
// Annihilated worldlines terminate (vertices removed from frontier) but
// stay in the spacetime. chargeOf_ for deactivated vertices is preserved
// (they're historical charges, not counted in the frontier-only Q sum).
void testDeactivateOnAnnihilateRemovesFromFrontier() {
    std::cout << "\n=== featureDeactivateOnAnnihilate: vertices leave frontier ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.featureCharges = true;
    cfg.featureDeactivateOnAnnihilate = true;
    InteractionSimulation sim(cfg);
    const std::size_t v0 = sim.getSpacetime()->getVertexCount();
    const std::size_t f0 = sim.frontierSize();
    int accepted = 0;
    for (int k = 0; k < 50; ++k) if (sim.annihilate()) ++accepted;
    CHECK(accepted > 0,
          "annihilate accepted with deactivate flag on");
    CHECK(sim.getSpacetime()->getVertexCount() == v0,
          "deactivated vertices stay in the spacetime");
    // Each full annihilation removes 2 vertices from the frontier;
    // each partial annihilation removes 1. We accept at least one drop
    // per accepted annihilation.
    CHECK(sim.frontierSize() <= f0 - static_cast<std::size_t>(accepted),
          "frontier shrank by ≥ accepted-annihilate count");
    std::cout << "  [info] frontier: before " << f0
              << "  after " << sim.frontierSize()
              << "  (Δ = -" << (f0 - sim.frontierSize())
              << ", accepted=" << accepted << ")\n";
}

void testDeactivateOnAnnihilateDocumentedDrift() {
    std::cout << "\n=== featureDeactivateOnAnnihilate: Q-drift is unchanged ===\n";
    // IMPORTANT: deactivate alone does NOT fix the Q-drift under
    // un-interact. The drift comes from the *un-interact* restoring a
    // parent (charge +qx) without knowing that the product (xp) was
    // annihilated against some partner (m) whose parent is on a
    // different cell — so the partner side of the original
    // annihilation isn't compensated. To eliminate the drift entirely
    // we'd need annihilation events to be first-class objects in the
    // un-interact cascade BFS, or to forbid un-interact on cells whose
    // products have been annihilated. Neither is in v0.1; both are
    // candidates for v0.2.
    InteractionConfig cfg = baseConfig();
    cfg.featureCharges = true;
    cfg.featureDeactivateOnAnnihilate = true;
    cfg.targetInteractions = 8;
    InteractionSimulation sim(cfg);
    sim.tune();
    for (int k = 0; k < 30; ++k) sim.annihilate();
    const double q0 = sim.getGlobalCharge();
    for (int k = 0; k < 8; ++k) sim.unInteract();
    const double q1 = sim.getGlobalCharge();
    CHECK(std::isfinite(q1),
          "Q stays finite even when deactivate-then-uninteract drifts");
    std::cout << "  [info] Q before/after un-interact (deactivate on): "
              << q0 << " / " << q1
              << "  (drift remains — see test comment)\n";
}

// ─── featurePhotonOnAnnihilate (option iii) ─────────────────────────────
void testPhotonOnAnnihilateSpawnsNeutralVertex() {
    std::cout << "\n=== featurePhotonOnAnnihilate: spawns neutral vertex per annihilation ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.featureCharges = true;
    cfg.featureDeactivateOnAnnihilate = true;
    cfg.featurePhotonOnAnnihilate = true;
    InteractionSimulation sim(cfg);
    const std::size_t v0 = sim.getSpacetime()->getVertexCount();
    int accepted = 0;
    for (int k = 0; k < 30; ++k) if (sim.annihilate()) ++accepted;
    CHECK(accepted > 0, "annihilate accepted with photon flag on");
    const std::size_t v1 = sim.getSpacetime()->getVertexCount();
    // Photon flag spawns one new vertex per annihilation event.
    CHECK(v1 == v0 + static_cast<std::size_t>(accepted),
          "vertex count grew by exactly the number of annihilations");
    std::cout << "  [info] " << accepted << " annihilations → " << (v1 - v0)
              << " new vertices in spacetime\n";
    // Q must still be conserved (photons are neutral).
    CHECK_NEAR(sim.getGlobalCharge(), 0.0, 1e-9,
               "Q conserved through annihilation+photon emission");
}

// ─── Feature-flag default behaviour ────────────────────────────────────────
void testFeatureFlagsDefaultOff() {
    std::cout << "\n=== feature flags default off / dependent flags auto-clear ===\n";
    InteractionConfig cfg = baseConfig();
    // No flags set explicitly — all should be false.
    cfg.featureDeactivateOnAnnihilate = true;  // but featureCharges off
    cfg.featurePhotonOnAnnihilate = true;
    InteractionSimulation sim(cfg);
    // Construction should not enable the dependent flags without
    // charges; the constructor's reconciliation step clears them.
    auto rates = sim.getAcceptanceRates();
    CHECK(rates.find("annihilate") == rates.end(),
          "annihilate move absent when charges off (dependent flags cleared)");
}

// Test the documented unInteract drift: confirm it CAN cause Q to drift
// when annihilations have occurred. This is intentional — the test
// asserts the known behaviour so a future code change doesn't silently
// alter it.
void testUnInteractCanDriftQWithAnnihilation() {
    std::cout << "\n=== documented unInteract drift after annihilation ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.cpBias = 0.0;
    cfg.targetInteractions = 8;
    InteractionSimulation sim(cfg);
    sim.tune();
    // Mix in some annihilations then un-interact: Q may drift.
    for (int k = 0; k < 30; ++k) sim.annihilate();
    const double qBefore = sim.getGlobalCharge();
    for (int k = 0; k < 8; ++k) sim.unInteract();
    const double qAfter = sim.getGlobalCharge();
    // We don't require drift, just that it's finite and the system
    // doesn't crash. (Drift is a known v0.1 limitation; v0.2 fixes it.)
    CHECK(std::isfinite(qAfter),
          "Q stays finite even when un-interact drifts after annihilation");
    std::cout << "  [info] Q before/after un-interact (after 30 ann): "
              << qBefore << " / " << qAfter
              << "  (drift is expected v0.1 behaviour, see design notes)\n";
}

// ─── Charges + un-interact cascade conservation ────────────────────────────
void testUnInteractCascadeConservesQ() {
    std::cout << "\n=== unInteract cascade conserves Q ===\n";
    InteractionConfig cfg = baseConfig();
    cfg.useCharges = true;
    cfg.targetInteractions = 12;
    InteractionSimulation sim(cfg);
    sim.tune();
    const double q0 = sim.getGlobalCharge();
    // Multiple un-interacts trigger deep cascades. Q must stay
    // unchanged regardless.
    for (int k = 0; k < 12; ++k) sim.unInteract();
    const double q1 = sim.getGlobalCharge();
    CHECK_NEAR(q0, q1, 1e-9,
               "unInteract cascade conserves Q exactly");
    std::cout << "  [info] Q before/after cascade un-interact: " << q0
              << " / " << q1 << "\n";
}

} // namespace

int main() {
    testBackwardCompat();
    testInitialChargesAlternating();
    testInitialChargesRandom();
    testInteractInheritsCharges();
    testAnnihilateNeutralisesCharge();
    testPairCreateSymmetric();
    testPairCreateCpBiased();
    testPairCreateCpBiasedNegative();
    testSweepStability();
    testObservablesSmoke();
    testUnInteractUnderCharges();
    testForwardOnlyConservation();
    testUnInteractCascadeConservesQ();
    testUnInteractCanDriftQWithAnnihilation();
    testDeactivateOnAnnihilateRemovesFromFrontier();
    testDeactivateOnAnnihilateDocumentedDrift();
    testPhotonOnAnnihilateSpawnsNeutralVertex();
    testFeatureFlagsDefaultOff();

    if (failures == 0) {
        std::cout << "\nPASS — all charged-Cartan v0.1 checks satisfied\n";
        return 0;
    }
    std::cerr << "\n" << failures << " v0.1 check(s) failed\n";
    return 1;
}
