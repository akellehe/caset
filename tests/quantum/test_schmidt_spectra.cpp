// Schmidt-spectrum extraction tests on simple, hand-checkable MPSes.
// Verifies that Schmidt::of and Schmidt::allOf return
// the textbook values for product, GHZ, and Bell states — the inputs we
// also feed into the Phase 3 majorization-poset acceptance test, but
// here we check only the spectra in isolation, before the poset layer
// gets involved.

#include "quantum/schmidt.hpp"
#include "test_mps_helpers.hpp"

#include <itensor/all.h>

#include <cmath>
#include <iostream>

using namespace tessera::quantum;
using namespace tessera::test_helpers;

namespace {

constexpr double TOL = 1e-10;

bool spectrum_is(std::vector<double> const& spec,
                 std::vector<double> const& expected,
                 double tol = TOL) {
    if (spec.size() < expected.size()) return false;
    for (std::size_t k = 0; k < expected.size(); ++k) {
        if (std::abs(spec[k] - expected[k]) > tol) return false;
    }
    // Anything beyond `expected` must be (numerically) zero — otherwise the
    // helper would be silently accepting more entanglement than expected.
    for (std::size_t k = expected.size(); k < spec.size(); ++k) {
        if (std::abs(spec[k]) > tol) return false;
    }
    return true;
}

bool product_state_all_trivial() {
    std::cout << "Product |↑↑↑↑⟩ — every contiguous spectrum is (1)\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = product_up(sites);

    auto cuts = Schmidt::allOf(psi);
    bool ok = true;
    int n_checked = 0;
    for (std::size_t k = 0; k < cuts.spectra.size(); ++k) {
        const auto& spec = cuts.spectra[k];
        const auto& iv   = cuts.intervals[k];
        const bool match = spectrum_is(spec, {1.0});
        if (!match) {
            std::cout << "  [" << iv.i << "," << iv.j << "]  spec=";
            for (double s : spec) std::cout << s << " ";
            std::cout << " — FAIL\n";
            ok = false;
        }
        ++n_checked;
    }
    std::cout << "  checked " << n_checked << " contiguous cuts on N=" << N
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool ghz_single_site_half_half() {
    std::cout << "\nGHZ on N=4 — every single-site spectrum is (½, ½)\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = ghz(sites);

    bool ok = true;
    for (int k = 1; k <= N; ++k) {
        auto spec = Schmidt::of(psi, k, k);
        const bool match = spectrum_is(spec, {0.5, 0.5});
        if (!match) {
            std::cout << "  site " << k << " spec=";
            for (double s : spec) std::cout << s << " ";
            std::cout << " — FAIL\n";
            ok = false;
        } else {
            std::cout << "  site " << k << " spec=(½, ½)  PASS\n";
        }
    }
    return ok;
}

bool ghz_all_cuts_half_half() {
    std::cout << "\nGHZ on N=6 — every contiguous cut spectrum is (½, ½)\n";
    constexpr int N = 6;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = ghz(sites);

    auto cuts = Schmidt::allOf(psi);
    bool ok = true;
    for (std::size_t k = 0; k < cuts.spectra.size(); ++k) {
        const auto& spec = cuts.spectra[k];
        if (!spectrum_is(spec, {0.5, 0.5})) {
            const auto& iv = cuts.intervals[k];
            std::cout << "  [" << iv.i << "," << iv.j << "]  spec=";
            for (double s : spec) std::cout << s << " ";
            std::cout << " — FAIL\n";
            ok = false;
        }
    }
    std::cout << "  checked " << cuts.spectra.size()
              << " contiguous cuts  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool bell_phi_plus_center() {
    std::cout << "\nBell |Φ⁺⟩ on N=2 — center cut spectrum is (½, ½)\n";
    auto sites = itensor::SpinHalf(2, {"ConserveQNs=", false});
    auto psi = bell_phi_plus(sites);
    auto spec = Schmidt::of(psi, 1, 1);
    const bool ok = spectrum_is(spec, {0.5, 0.5});
    std::cout << "  spec[1,1]=";
    for (double s : spec) std::cout << s << " ";
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool bell_singlet_center() {
    std::cout << "\nSinglet (|↑↓⟩−|↓↑⟩)/√2 on N=2 — same (½, ½) spectrum\n";
    auto sites = itensor::SpinHalf(2, {"ConserveQNs=", false});
    auto psi = bell_singlet(sites);
    auto spec = Schmidt::of(psi, 1, 1);
    const bool ok = spectrum_is(spec, {0.5, 0.5});
    std::cout << "  spec[1,1]=";
    for (double s : spec) std::cout << s << " ";
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool spectra_normalize_to_one() {
    std::cout << "\nAll spectra sum to 1 (probability-distribution sanity)\n";
    constexpr int N = 5;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi = ghz(sites);
    auto cuts = Schmidt::allOf(psi);
    bool ok = true;
    for (std::size_t k = 0; k < cuts.spectra.size(); ++k) {
        double total = 0.0;
        for (double s : cuts.spectra[k]) total += s;
        if (std::abs(total - 1.0) > TOL) {
            std::cout << "  cut [" << cuts.intervals[k].i << ","
                      << cuts.intervals[k].j << "] sums to " << total
                      << " — FAIL\n";
            ok = false;
        }
    }
    std::cout << "  checked " << cuts.spectra.size() << " spectra  "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool single_site_equals_left_bond() {
    // Cross-check: for a left-edge cut [1, 1], the entanglement spectrum
    // is exactly the bond-1 SVD of the MPS. For a 4-site GHZ that's
    // (½, ½) — but the more useful sanity check is that Schmidt::of
    // and ITensor's built-in bond singular values agree.
    std::cout << "\nLeft-edge cut [1,1] vs ITensor bond SVD on N=4 GHZ\n";
    constexpr int N = 4;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});
    auto psi_in = ghz(sites);
    auto psi = psi_in;
    psi.position(1);

    auto our_spec = Schmidt::of(psi, 1, 1);

    // ITensor's bond-1 SVD: split MPS into A_1 and the rest.
    using namespace itensor;
    ITensor A1 = psi(1);
    ITensor rest = psi(2);
    for (int k = 3; k <= N; ++k) rest *= psi(k);
    auto right_inds = std::vector<Index>();
    for (int k = 2; k <= N; ++k) right_inds.push_back(siteIndex(psi, k));
    // SVD of (A1 * rest) reshaped as (site_1, bond_1) × (sites 2..N).
    ITensor M = A1 * rest;
    auto args = Args("Cutoff", 0.0, "MaxDim", 1 << 24);
    auto [U, S, V] = svd(M, IndexSet({siteIndex(psi, 1)}), args);

    std::vector<double> ref;
    auto inds = S.inds();
    int rank = std::min(inds(1).dim(), inds(2).dim());
    for (int k = 1; k <= rank; ++k) {
        const double sv = elt(S, k, k);
        ref.push_back(sv * sv);
    }
    std::sort(ref.begin(), ref.end(), std::greater<double>{});

    bool ok = our_spec.size() == ref.size();
    if (ok) {
        for (std::size_t k = 0; k < ref.size(); ++k) {
            if (std::abs(our_spec[k] - ref[k]) > TOL) ok = false;
        }
    }
    std::cout << "  our=";
    for (double s : our_spec) std::cout << s << " ";
    std::cout << " ref=";
    for (double s : ref) std::cout << s << " ";
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= product_state_all_trivial();
    ok &= ghz_single_site_half_half();
    ok &= ghz_all_cuts_half_half();
    ok &= bell_phi_plus_center();
    ok &= bell_singlet_center();
    ok &= spectra_normalize_to_one();
    ok &= single_site_equals_left_bond();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
