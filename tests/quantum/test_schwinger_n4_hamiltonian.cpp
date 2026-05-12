// PLAN.md §7 explicitly calls out:
//
//     "Gauss's law: after elimination, the long-range Z_m Z_{m'} term has
//      coefficients that depend on the parity of sites between m and m'.
//      Verify by independent sum on N=4 before trusting the MPO."
//
// This file is that independent verification. We compute ⟨ψ|H|ψ⟩ on
// every computational-basis state of an N=4 chain by evaluating the
// PLAN.md §4 Hamiltonian formula symbolically (term-by-term, with no
// MPO machinery) and confirm that SchwingerHamiltonian::denseMatrix agrees to
// machine precision. We also check a representative off-diagonal hopping
// matrix element. If the L_n² expansion in src/quantum/schwinger_model.cpp
// got a sign or coefficient wrong, this test catches it independently of
// any AutoMPO / DMRG machinery.
//
// Why N=4: small enough to enumerate every basis state in tens of
// microseconds, large enough (N - 1 = 3 link links) that all three
// pieces of L_n² (constant, linear σ^z, pair σ^z σ^z) are non-trivially
// exercised.

#include "quantum/schwinger_model.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

using namespace tessera::quantum;

namespace {

// σ^z eigenvalue at 1-based site n in basis state s, MSB-first bit
// convention matching SchwingerHamiltonian::denseMatrix.
inline double sigma_z(std::uint64_t s, int n, int N) {
    return ((s >> (N - n)) & 1ull) == 0 ? +1.0 : -1.0;
}

// L_n on a computational basis state, evaluated DIRECTLY from PLAN.md §4:
//
//   L_n = L0 + Σ_{k=1..n} [(1 - σ^z_k)/2  -  (1 - (-1)^k)/2]
//
// No simplification, no operator expansion — this is the literal formula.
// If our schwinger_model.cpp's c_n - (1/2)Σσ^z_k closed form deviates
// from this, the diagonal H_E elements will disagree.
double L_n_direct(int n, std::uint64_t s, int N, double L0) {
    double sum = 0.0;
    for (int k = 1; k <= n; ++k) {
        const double zk = sigma_z(s, k, N);
        const double sign_k = (k % 2 == 0) ? +1.0 : -1.0;  // (-1)^k
        sum += (1.0 - zk) / 2.0 - (1.0 - sign_k) / 2.0;
    }
    return L0 + sum;
}

// ⟨ψ|H_diag|ψ⟩ on a basis state, computed directly from PLAN.md §4.
// H_hop is purely off-diagonal so we only sum H_m + H_E here.
double H_diagonal_direct(std::uint64_t s, SchwingerParams const& p) {
    // H_m = (m/2) Σ_n (-1)^n σ^z_n
    double H_m_val = 0.0;
    for (int n = 1; n <= p.N; ++n) {
        const double sign_n = (n % 2 == 0) ? +1.0 : -1.0;
        H_m_val += sign_n * sigma_z(s, n, p.N);
    }
    H_m_val *= 0.5 * p.m;

    // H_E = (g²a/2) Σ_{n=1..N-1} L_n²
    double H_E_val = 0.0;
    for (int n = 1; n <= p.N - 1; ++n) {
        const double Ln = L_n_direct(n, s, p.N, p.L0);
        H_E_val += Ln * Ln;
    }
    H_E_val *= 0.5 * p.g * p.g * p.a;

    return H_m_val + H_E_val;
}

bool diagonal_test(SchwingerParams const& p, double tol) {
    std::cout << "  ⟨ψ|H|ψ⟩ on every basis state for N=" << p.N
              << ", m=" << p.m << ", L0=" << p.L0 << "\n";
    auto sd = SchwingerHamiltonian{p}.denseMatrix();
    const std::size_t dim = static_cast<std::size_t>(sd.H.rows());

    bool ok = true;
    for (std::size_t s = 0; s < dim; ++s) {
        const double direct  = H_diagonal_direct(s, p);
        const double from_sd = sd.H(static_cast<Eigen::Index>(s),
                                    static_cast<Eigen::Index>(s)) + sd.constant;
        if (std::abs(direct - from_sd) > tol) {
            std::cout << "    FAIL state s=" << s
                      << " direct=" << direct
                      << " dense=" << from_sd
                      << " Δ=" << (direct - from_sd) << "\n";
            ok = false;
        }
    }
    std::cout << "    " << dim << " states checked, "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool off_diagonal_test(SchwingerParams const& p, double tol) {
    // Pick a basis state |s⟩ with adjacent (Up, Dn) at sites n, n+1; the
    // hopping σ⁺_n σ⁻_{n+1} + σ⁻_n σ⁺_{n+1} flips this to a sister state
    // |s'⟩ with the (Up, Dn) swapped. The expected matrix element is
    //   (1/(4a)) · ⟨s'|XX + YY|s⟩ = (1/(4a)) · 2 = 1/(2a)
    // for any such adjacent flip pair.
    std::cout << "  ⟨s'|H_hop|s⟩ off-diagonal coefficient at N=" << p.N << "\n";
    auto sd = SchwingerHamiltonian{p}.denseMatrix();

    // s = |↑↓↑↑⟩ with bits 0010 at MSB-first → site 3 is Dn (bit 1).
    // After flipping the (n=2, n=3) pair: |↑↑↓↑⟩, which has bit 0100.
    // Let's just pick by bits: s_in = 0b0010 (= 2), s_out = 0b0100 (= 4).
    const std::uint64_t s_in  = 0b0010ull;  // site 3 = Dn, sites 1,2,4 = Up
    const std::uint64_t s_out = 0b0100ull;  // site 2 = Dn, sites 1,3,4 = Up

    const double element = sd.H(static_cast<Eigen::Index>(s_out),
                                static_cast<Eigen::Index>(s_in));
    const double expected = 1.0 / (2.0 * p.a);
    const double diff = std::abs(element - expected);
    const bool ok = diff < tol;
    std::cout << "    element=" << element
              << " expected=" << expected
              << " Δ=" << diff
              << " " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << std::setprecision(12);
    bool ok = true;

    std::cout << "PLAN.md §7 trap: independent N=4 verification of L_n² expansion\n";
    std::cout << "----------------------------------------------------------------\n";

    // Diagonal sweep across multiple parameter combos: each combo
    // exercises a different cross-section of the L_n² expansion (the
    // c_n contribution depends on L0; the linear σ^z piece depends on
    // m and L0; the σ^z σ^z pair piece depends on g²a).
    for (double m  : {0.0, 0.125, 0.5, 1.0}) {
    for (double L0 : {0.0, 0.5}) {
        SchwingerParams p;
        p.N = 4; p.a = 1.0; p.g = 1.0; p.m = m; p.L0 = L0;
        ok &= diagonal_test(p, 1e-12);
    }
    }

    // Off-diagonal hopping coefficient — same for any (m, L0).
    {
        SchwingerParams p;
        p.N = 4; p.a = 1.0; p.g = 1.0; p.m = 0.0; p.L0 = 0.0;
        ok &= off_diagonal_test(p, 1e-12);
    }

    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
