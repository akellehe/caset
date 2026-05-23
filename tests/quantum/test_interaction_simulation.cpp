// End-to-end smoke test for InteractionSimulation: construct the
// randomized mixed-state initial layer, run interaction moves, and check
// the simplicial + density-matrix machinery produces finite, sane
// numbers. If this fails the bug is in the simulation plumbing — the
// physics interpretation is checked separately.

#include "simulations/InteractionSimulation.h"

#include <cmath>
#include <iostream>

using tessera::InteractionConfig;
using tessera::InteractionSimulation;

int main() {
    int failures = 0;

    InteractionConfig cfg;
    cfg.nSystems = 6;
    cfg.a = 1.0;
    cfg.g = 1.0;
    cfg.m = 0.5;
    cfg.dt = 0.25;
    cfg.beta = 1e-3;  // ~1/<ΔS>; the Metropolis-active regime
    cfg.epsilonI = 1e-10;
    cfg.targetInteractions = 4;
    cfg.seed = 1;
    // A connected Delaunay-like edge set on the six initial systems.
    cfg.delaunayEdges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                         {0, 2}, {1, 3}, {2, 4}, {3, 5}};

    InteractionSimulation sim(cfg);
    const std::size_t v0 = sim.getSpacetime()->getVertexCount();
    std::cout << "[construct] initial vertices = " << v0 << std::endl;
    if (v0 != 6) {
        std::cerr << "FAIL: expected 6 initial vertices\n";
        ++failures;
    }

    // Run interaction moves directly.
    int accepted = 0;
    for (int k = 0; k < 40; ++k)
        if (sim.interact()) ++accepted;
    std::cout << "[interact] " << accepted << "/40 accepted, "
              << "interactionCount = " << sim.interactionCount()
              << std::endl;
    if (sim.interactionCount() == 0) {
        std::cerr << "FAIL: no interactions accepted in 40 attempts\n";
        ++failures;
    }

    const double action = sim.computeAction();
    std::cout << "[action] S = " << action << std::endl;
    if (!std::isfinite(action)) {
        std::cerr << "FAIL: Regge action is not finite\n";
        ++failures;
    }

    const std::size_t v1 = sim.getSpacetime()->getVertexCount();
    if (v1 <= v0) {
        std::cerr << "FAIL: complex did not grow under interactions\n";
        ++failures;
    }
    std::cout << "[grow] vertices " << v0 << " -> " << v1 << std::endl;

    // Check observables on the *grown* complex, before any un-interactions
    // can truncate it.
    {
        const auto profile = sim.getVolumeProfile();
        std::cout << "[volume] profile slices = " << profile.size() << "  [";
        for (int c : profile) std::cout << c << " ";
        std::cout << "]" << std::endl;
        if (profile.size() < 2) {
            std::cerr << "FAIL: complex did not advance past one time slice\n";
            ++failures;
        }
        std::vector<double> sigmas;
        for (int s = 0; s < 24; ++s)
            sigmas.push_back(std::exp(std::log(1e-2)
                + s * (std::log(1e3) - std::log(1e-2)) / 23.0));
        const auto dS = sim.getSpectralDimension(sigmas);
        double peakDS = 0.0;
        int nFinite = 0;
        for (double d : dS)
            if (std::isfinite(d)) { ++nFinite; peakDS = std::max(peakDS, d); }
        std::cout << "[spectral] " << nFinite << "/" << dS.size()
                  << " finite, peak D_S = " << peakDS << std::endl;
        if (nFinite == 0) {
            std::cerr << "FAIL: spectral dimension produced no finite values\n";
            ++failures;
        }
    }

    // A Monte Carlo sweep, then deep un-interactions (each accepted
    // un-interact truncates a cell's whole future cone).
    const int swept = sim.sweep();
    std::cout << "[sweep] " << swept << " moves accepted" << std::endl;

    const std::size_t countBeforeUninteract = sim.interactionCount();
    int unInteracted = 0;
    for (int k = 0; k < 20; ++k)
        if (sim.unInteract()) ++unInteracted;
    std::cout << "[unInteract] " << unInteracted << "/20 accepted, "
              << "interactionCount = " << sim.interactionCount()
              << "  (was " << countBeforeUninteract << ")" << std::endl;
    if (unInteracted > 0 && sim.interactionCount() >= countBeforeUninteract) {
        std::cerr << "FAIL: un-interactions accepted but interactionCount "
                  << "did not decrease\n";
        ++failures;
    }

    // Acceptance rates must be in range.
    for (auto const& [name, rate] : sim.getAcceptanceRates()) {
        std::cout << "[rate] " << name << " = " << rate << std::endl;
        if (!std::isfinite(rate) || rate < 0.0 || rate > 1.0) {
            std::cerr << "FAIL: acceptance rate out of range\n";
            ++failures;
        }
    }

    // tune() should grow toward the target without crashing.
    InteractionSimulation sim2(cfg);
    sim2.tune();
    std::cout << "[tune] interactionCount = " << sim2.interactionCount()
              << " (target " << cfg.targetInteractions << ")" << std::endl;
    if (!std::isfinite(sim2.computeAction())) {
        std::cerr << "FAIL: action not finite after tune()\n";
        ++failures;
    }

    if (failures == 0) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
