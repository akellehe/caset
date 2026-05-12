// Shared MPS-construction helpers for the Schmidt, TDVP, and causal-comparison acceptance
// tests. Each helper returns a normalized MPS in canonical form so the
// test code can drop straight into Schmidt-spectrum / majorization
// pipelines without further preparation.

#pragma once

#include <itensor/all.h>

#include <cmath>
#include <stdexcept>

namespace tessera::test_helpers {

// |↑↑ … ↑⟩ product state on a SpinHalf SiteSet. Schmidt spectrum across
// any contiguous bipartition is (1) — this is the trivial product-state
// case for PLAN.md §5 majorization-poset acceptance #1.
inline itensor::MPS product_up(itensor::SpinHalf const& sites) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= itensor::length(sites); ++i) {
        state.set(i, "Up");
    }
    return itensor::MPS(state);
}

// Néel |↑↓↑↓ … ⟩. Like the product-up state but with alternating spins —
// still a product state with Schmidt spectrum (1) at every cut, but it
// lives in the Sz=0 sector.
inline itensor::MPS neel(itensor::SpinHalf const& sites) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= itensor::length(sites); ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

// GHZ state (|↑↑…↑⟩ + |↓↓…↓⟩) / √2 as an MPS with bond dim 2.
//
// Construction: write the GHZ amplitude as a 1 × 2 row vector at site 1,
// 2 × 2 diagonal matrices at intermediate sites, and a 2 × 1 column
// vector at site N. Each tensor has site index s and bond indices α (in)
// and β (out); we set entries so that contracting the chain reproduces
// (|↑↑…↑⟩ + |↓↓…↓⟩) / √2.
//
// Schmidt spectrum at every cut is (½, ½) — majorization-poset acceptance #2.
inline itensor::MPS ghz(itensor::SpinHalf const& sites) {
    using namespace itensor;
    const int N = length(sites);
    if (N < 2) throw std::invalid_argument("ghz expects N ≥ 2");

    // Build by superposing two product MPSs (|↑↑…↑⟩ and |↓↓…↓⟩) and
    // truncating with a small SVD pass; this keeps the construction
    // self-contained without tensor-by-tensor assembly.
    auto state_up = InitState(sites);
    auto state_dn = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state_up.set(i, "Up");
        state_dn.set(i, "Dn");
    }
    MPS psi_up(state_up), psi_dn(state_dn);
    // sum() coerces to a single MPS. Default args give an exact result
    // (no truncation) for products; bond dim ends up at most 2.
    auto psi = sum(psi_up, psi_dn);
    psi.normalize();
    return psi;
}

// 2-qubit Bell state (|↑↑⟩ + |↓↓⟩) / √2 — equivalent to the N=2 GHZ.
// Schmidt spectrum across the only nontrivial cut [1, 1] | [2, 2] is
// (½, ½) — majorization-poset acceptance #3 input.
inline itensor::MPS bell_phi_plus(itensor::SpinHalf const& sites) {
    if (itensor::length(sites) != 2) {
        throw std::invalid_argument("bell_phi_plus expects N = 2");
    }
    return ghz(sites);
}

// 2-qubit singlet (|↑↓⟩ − |↓↑⟩) / √2. Sz = 0 sector with Schmidt
// spectrum (½, ½). Useful as a second Bell-flavoured test point that
// crosses Sz parity.
inline itensor::MPS bell_singlet(itensor::SpinHalf const& sites) {
    using namespace itensor;
    if (length(sites) != 2) {
        throw std::invalid_argument("bell_singlet expects N = 2");
    }
    auto state_a = InitState(sites);
    state_a.set(1, "Up"); state_a.set(2, "Dn");
    auto state_b = InitState(sites);
    state_b.set(1, "Dn"); state_b.set(2, "Up");
    MPS psi_a(state_a), psi_b(state_b);
    // |↑↓⟩ − |↓↑⟩: combine with opposite signs by scaling one before sum.
    psi_b *= -1.0;
    auto psi = sum(psi_a, psi_b);
    psi.normalize();
    return psi;
}

} // namespace tessera::test_helpers
