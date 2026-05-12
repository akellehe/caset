// Pure-majorization tests on synthetic spectra (no MPS involved).
//
// Verifies the basic algebraic properties of the partial order — reflexive,
// transitive, antisymmetric on padded-sorted equivalence classes — plus
// the canonical reference comparison (1, 0) ≻ (½, ½).
//
// The Phase 3 *physical* acceptance tests (product / GHZ / Bell against
// the Schwinger model code) live in tests/quantum/test_majorization_poset.cpp;
// this file is the unit-test layer for the majorizes() predicate itself.

#include "quantum/majorization.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace tessera::quantum;

namespace {

// Tests target the classical Nielsen-1999 majorization throughout, so we
// thread one StandardMajorization through every call site. Local
// shorthand keeps the assertion lines below readable.
const StandardMajorization classical{1e-12};
auto majorizes = [](std::vector<double> const& mu,
                    std::vector<double> const& lambda) {
    return classical.majorizes(mu, lambda);
};
auto strictlyMajorizes = [](std::vector<double> const& mu,
                            std::vector<double> const& lambda) {
    return classical.strictlyMajorizes(mu, lambda);
};

bool expect_true(bool cond, const char* desc) {
    std::cout << "  " << desc << " ... " << (cond ? "PASS" : "FAIL") << "\n";
    return cond;
}

bool expect_false(bool cond, const char* desc) {
    std::cout << "  " << desc << " ... " << (cond ? "FAIL" : "PASS") << "\n";
    return !cond;
}

bool reflexivity() {
    std::cout << "Reflexivity (μ ≼ μ for any μ)\n";
    bool ok = true;
    ok &= expect_true(majorizes({1.0}, {1.0}),
                      "μ = (1) majorizes itself");
    ok &= expect_true(majorizes({0.5, 0.5}, {0.5, 0.5}),
                      "μ = (½, ½) majorizes itself");
    ok &= expect_true(majorizes({0.7, 0.2, 0.1}, {0.7, 0.2, 0.1}),
                      "μ = (0.7, 0.2, 0.1) majorizes itself");
    ok &= expect_false(strictlyMajorizes({0.5, 0.5}, {0.5, 0.5}),
                       "(½,½) does NOT strictly majorize itself");
    return ok;
}

bool the_canonical_strict_pair() {
    std::cout << "\nCanonical strict pair (1, 0) ≻ (½, ½)\n";
    bool ok = true;
    ok &= expect_true(majorizes({1.0, 0.0}, {0.5, 0.5}),
                      "(1, 0) majorizes (½, ½)");
    ok &= expect_false(majorizes({0.5, 0.5}, {1.0, 0.0}),
                       "(½, ½) does NOT majorize (1, 0)");
    ok &= expect_true(strictlyMajorizes({1.0, 0.0}, {0.5, 0.5}),
                      "(1, 0) STRICTLY majorizes (½, ½)");
    return ok;
}

bool zero_padding_invariance() {
    std::cout << "\nZero-padding invariance — extra trailing zeros are ignored\n";
    bool ok = true;
    ok &= expect_true(majorizes({0.5, 0.5, 0.0, 0.0, 0.0}, {0.5, 0.5}),
                      "padded (½, ½, 0, 0, 0) ~ (½, ½)");
    ok &= expect_true(majorizes({0.5, 0.5}, {0.5, 0.5, 0.0, 0.0}),
                      "(½, ½) ~ padded (½, ½, 0, 0)");
    ok &= expect_false(strictlyMajorizes({0.5, 0.5, 0.0}, {0.5, 0.5}),
                       "extra zeros do NOT introduce strict majorization");
    return ok;
}

bool sort_invariance() {
    std::cout << "\nSort invariance — input order doesn't matter\n";
    bool ok = true;
    ok &= expect_true(majorizes({0.1, 0.2, 0.7}, {0.7, 0.2, 0.1}),
                      "permuted inputs compare equal");
    ok &= expect_true(majorizes({0.7, 0.1, 0.2}, {0.2, 0.7, 0.1}),
                      "different permutations compare equal");
    return ok;
}

bool transitivity() {
    std::cout << "\nTransitivity (a ≻ b, b ≻ c ⇒ a ≻ c)\n";
    // Standard chain: (1, 0, 0) ≻ (½, ½, 0) ≻ (⅓, ⅓, ⅓)
    const std::vector<double> a = {1.0, 0.0, 0.0};
    const std::vector<double> b = {0.5, 0.5, 0.0};
    const std::vector<double> c = {1.0/3, 1.0/3, 1.0/3};
    bool ok = true;
    ok &= expect_true(strictlyMajorizes(a, b), "(1,0,0) ≻ (½,½,0)");
    ok &= expect_true(strictlyMajorizes(b, c), "(½,½,0) ≻ (⅓,⅓,⅓)");
    ok &= expect_true(strictlyMajorizes(a, c),
                      "(1,0,0) ≻ (⅓,⅓,⅓) (transitive consequence)");
    return ok;
}

bool unequal_total_mass_rejected() {
    std::cout << "\nUnequal total mass — never majorizes\n";
    bool ok = true;
    ok &= expect_false(majorizes({1.0, 0.0}, {0.5, 0.5, 0.5}),
                       "totals 1.0 vs 1.5 mismatch — not majorizes");
    ok &= expect_false(majorizes({0.5, 0.5, 0.5}, {1.0, 0.0}),
                       "(reverse) totals 1.5 vs 1.0 — not majorizes");
    return ok;
}

bool incomparable_pairs() {
    std::cout << "\nIncomparable pairs exist (partial — not total — order)\n";
    // Classic incomparable pair on the 3-simplex:
    //   a = (0.5, 0.4, 0.1)
    //   b = (0.6, 0.2, 0.2)
    // a₁ < b₁ but a₁ + a₂ > b₁ + b₂  ⇒ neither majorizes the other.
    const std::vector<double> a = {0.5, 0.4, 0.1};
    const std::vector<double> b = {0.6, 0.2, 0.2};
    bool ok = true;
    ok &= expect_false(majorizes(a, b), "a does not majorize b");
    ok &= expect_false(majorizes(b, a), "b does not majorize a");
    return ok;
}

bool small_poset_construction() {
    std::cout << "\nPoset construction on the (1,0) — (½,½) — (⅓,⅓,⅓) chain\n";
    std::vector<std::vector<double>> spectra = {
        {1.0/3, 1.0/3, 1.0/3},  // node 0  — most uniform
        {0.5, 0.5},             // node 1  — middle
        {1.0},                  // node 2  — most concentrated (= (1, 0, 0))
    };
    auto poset = Majorization::posetOf(spectra);
    // Hasse cover relations should be 2 → 1 and 1 → 0; the direct edge
    // 2 → 0 should be REMOVED by transitive reduction.
    bool ok = true;
    ok &= expect_true(poset.getNodeCount() == 3, "getNodeCount = 3");
    ok &= expect_true(poset.covers().size() == 2, "exactly 2 cover edges");
    bool has_2_to_1 = false;
    bool has_1_to_0 = false;
    bool has_2_to_0 = false;
    for (auto [a, b] : poset.covers()) {
        if (a == 2 && b == 1) has_2_to_1 = true;
        if (a == 1 && b == 0) has_1_to_0 = true;
        if (a == 2 && b == 0) has_2_to_0 = true;
    }
    ok &= expect_true(has_2_to_1, "(1,0) ≻ (½,½) is a cover edge");
    ok &= expect_true(has_1_to_0, "(½,½) ≻ (⅓,⅓,⅓) is a cover edge");
    ok &= expect_false(has_2_to_0,
                       "(1,0) ≻ (⅓,⅓,⅓) edge was correctly transitively reduced");
    return ok;
}

bool empty_input() {
    std::cout << "\nEdge cases — empty / single-node input\n";
    auto p_empty = Majorization::posetOf({});
    auto p_one   = Majorization::posetOf({{1.0}});
    bool ok = true;
    ok &= expect_true(p_empty.getNodeCount() == 0,    "empty input → 0 nodes");
    ok &= expect_true(p_empty.covers().empty(),  "empty input → 0 edges");
    ok &= expect_true(p_one.getNodeCount() == 1,      "single-node input → 1 node");
    ok &= expect_true(p_one.covers().empty(),    "single-node input → 0 edges");
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= reflexivity();
    ok &= the_canonical_strict_pair();
    ok &= zero_padding_invariance();
    ok &= sort_invariance();
    ok &= transitivity();
    ok &= unequal_total_mass_rejected();
    ok &= incomparable_pairs();
    ok &= small_poset_construction();
    ok &= empty_input();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
