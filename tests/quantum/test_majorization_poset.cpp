// Phase 3 acceptance tests (PLAN.md §5):
//
//   (1) Product state |0⟩^⊗N: every contiguous-cut Schmidt spectrum is
//       (1, 0, …); the Hasse diagram has no edges (all spectra are
//       equal — pairwise majorization in both directions, no STRICT
//       relations).
//
//   (2) N-qubit GHZ: every single-site spectrum is (½, ½); pairwise
//       majorization relations all equal — no strict edges.
//
//   (3) Bell state across the center cut of N=2: spectrum (½, ½);
//       compared against a synthetic product-state spectrum (1, 0)
//       we should see (1, 0) ≻ (½, ½) as a strict cover edge.
//
// These exercise the full pipeline: schmidt_spectrum() → vector of
// spectra → majorization_poset() → Hasse cover edges.

#include "quantum/schmidt.hpp"
#include "quantum/majorization.hpp"
#include "test_mps_helpers.hpp"

#include <itensor/all.h>

#include <iostream>
#include <vector>

using namespace caset::quantum;
using namespace caset::test_helpers;

namespace {

bool acceptance_product_state() {
    std::cout << "Acceptance #1 — product |↑↑↑↑⟩ Hasse has no edges\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = product_up(sites);

    auto cuts = all_contiguous_spectra(psi);
    auto poset = majorization_poset(cuts.spectra);

    const bool ok = poset.covers.empty();
    std::cout << "  cuts=" << cuts.spectra.size()
              << "  Hasse edges=" << poset.covers.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    if (!ok) {
        for (auto [a, b] : poset.covers) {
            std::cout << "    unexpected edge: " << a << " ≻ " << b << "\n";
        }
    }
    return ok;
}

bool acceptance_ghz_no_strict_edges() {
    std::cout << "\nAcceptance #2 — N=4 GHZ Hasse has no strict edges\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = ghz(sites);

    // Every single-site spectrum and every contiguous-cut spectrum is
    // exactly (½, ½), so all pairwise comparisons are equality and the
    // strict-majorization graph is empty.
    auto cuts = all_contiguous_spectra(psi);
    auto poset = majorization_poset(cuts.spectra);

    const bool ok = poset.covers.empty();
    std::cout << "  cuts=" << cuts.spectra.size()
              << "  Hasse edges=" << poset.covers.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    if (!ok) {
        for (auto [a, b] : poset.covers) {
            std::cout << "    unexpected edge: " << a << " ≻ " << b << "\n";
        }
    }
    return ok;
}

bool acceptance_bell_vs_product() {
    std::cout
        << "\nAcceptance #3 — Bell (½, ½) vs product (1, 0): (1, 0) ≻ (½, ½)\n";
    auto sites = itensor::SpinHalf(2, {"ConserveQNs=", false});
    auto psi = bell_phi_plus(sites);
    auto bell_spec = schmidt_spectrum(psi, 1, 1);

    // Extra synthetic node injects a perfectly product-state spectrum
    // into the comparison set, so the poset is non-degenerate.
    std::vector<std::vector<double>> spectra = {
        bell_spec,        // node 0 — Bell cut, expected (½, ½)
        {1.0, 0.0},       // node 1 — synthetic product cut
    };
    auto poset = majorization_poset(spectra);

    bool ok = true;
    if (poset.covers.size() != 1) {
        std::cout << "  expected exactly 1 Hasse edge, got "
                  << poset.covers.size() << "  FAIL\n";
        ok = false;
    } else {
        auto [a, b] = poset.covers.front();
        if (a == 1 && b == 0) {
            std::cout << "  edge (1, 0) → (½, ½) present  PASS\n";
        } else {
            std::cout << "  unexpected cover: " << a << " ≻ " << b
                      << "  FAIL\n";
            ok = false;
        }
    }
    // Sanity: the Bell spectrum should actually be (½, ½) to TOL.
    if (bell_spec.size() != 2 ||
        std::abs(bell_spec[0] - 0.5) > 1e-10 ||
        std::abs(bell_spec[1] - 0.5) > 1e-10) {
        std::cout << "  Bell spectrum check FAIL: ";
        for (double s : bell_spec) std::cout << s << " ";
        std::cout << "\n";
        ok = false;
    }
    return ok;
}

bool ghz_full_chain_includes_extreme() {
    // Cross-check: a GHZ-cut chain compared against a product node gives
    // the expected (1, 0) → (½, ½) → … structure when we mix in spectra
    // from both states.
    std::cout
        << "\nMixed product+GHZ: Hasse edges follow concentration order\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi_p = product_up(sites);
    auto psi_g = ghz(sites);

    auto cuts_p = all_contiguous_spectra(psi_p);  // all (1, 0, 0)
    auto cuts_g = all_contiguous_spectra(psi_g);  // all (½, ½, 0)

    std::vector<std::vector<double>> spectra;
    spectra.insert(spectra.end(), cuts_p.spectra.begin(), cuts_p.spectra.end());
    const std::size_t n_product = cuts_p.spectra.size();
    spectra.insert(spectra.end(), cuts_g.spectra.begin(), cuts_g.spectra.end());

    auto poset = majorization_poset(spectra);

    // We expect: every GHZ node (index ≥ n_product) should be covered by
    // every product node (index < n_product), but transitive reduction
    // collapses to one edge per (product, GHZ) pair *only if* product
    // nodes don't dominate each other (they're equal). Equality means no
    // cover edges among product nodes themselves, and likewise among GHZ
    // nodes — just the cross-class edges remain.
    const std::size_t expected = n_product * cuts_g.spectra.size();

    std::cout << "  product cuts=" << n_product
              << "  GHZ cuts="     << cuts_g.spectra.size()
              << "  expected edges=" << expected
              << "  got=" << poset.covers.size()
              << "  " << (poset.covers.size() == expected ? "PASS" : "FAIL")
              << "\n";

    // Spot-check: pick one cover edge and verify it points from product to GHZ.
    bool direction_ok = true;
    for (auto [a, b] : poset.covers) {
        // a is the strict majorizer, should be a product node;
        // b is the majorized, should be a GHZ node.
        if (a >= static_cast<int>(n_product) ||
            b <  static_cast<int>(n_product)) {
            direction_ok = false;
            std::cout << "  unexpected cover (" << a << ", " << b
                      << ")  FAIL\n";
            break;
        }
    }
    return poset.covers.size() == expected && direction_ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_product_state();
    ok &= acceptance_ghz_no_strict_edges();
    ok &= acceptance_bell_vs_product();
    ok &= ghz_full_chain_includes_extreme();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
