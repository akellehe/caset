// Physics-limit checks on the Schwinger MPO. These tests compare DMRG output
// against ANALYTIC reference values (not against the dense matrix), so they
// catch sign / convention bugs that would slip past an MPO-vs-dense
// agreement check — both sides could share the same systematic error and
// silently agree.
//
// ─── Limit 1: free-fermion (g=0, m=0) ────────────────────────────────────
//
// At g = 0 the H_E prefactor (g²a/2) vanishes, so the entire electric term
// drops out. With m = 0 the mass term also drops, leaving only:
//
//   H = (1/(4a)) Σ (X X + Y Y) = (1/(2a)) Σ (σ⁺_n σ⁻_{n+1} + σ⁻_n σ⁺_{n+1})
//
// After Jordan-Wigner this is non-interacting fermions on an N-site OBC
// chain with hopping coefficient +1/(2a) on (c†_n c_{n+1} + h.c.). The
// single-particle eigenmodes are sin(k_j x) on [0, N] with quantized k_j =
// πj/(N+1), j = 1..N, and energies ε_j = (1/a) cos(k_j) (the sign comes
// from our positive hopping coefficient: with H = +t Σ (c†c + h.c.) we get
// ε_j = +2t cos(k_j) = (1/a) cos(k_j)).
//
// At half-filling (Sz=0 sector ↔ N/2 fermions), the GS fills the N/2
// most-negative single-particle levels. cos(πj/(N+1)) is most negative at
// j → N (where k_j → π), so the filled levels are j = N/2+1..N:
//
//   E_GS_free  =  (1/a) Σ_{j=N/2+1..N} cos(πj/(N+1))
//
// This is testable to machine precision against our DMRG output at g=0,m=0.
//
// ─── Limit 2: strong-coupling vacuum (m → ∞, L₀=0, g=a=1) ─────────────────
//
// For very large m, the mass term dominates. With our convention
// H_m = (m/2) Σ (-1)^n σ^z_n (1-based n: sign − at odd n, + at even n),
// the energy-minimizing classical state has σ^z_n = +1 on odd n, −1 on
// even n — i.e. |↑↓↑↓ ... ⟩.
//
// On this product state:
//   • Hopping ⟨σ⁺σ⁻ + σ⁻σ⁺⟩ = 0 (hop maps |UD⟩ → |DU⟩, orthogonal).
//   • Mass    ⟨H_m⟩ = (m/2) Σ_n (-1)^n · (-1)^(n+1) = -mN/2.
//   • Electric: L_n on |↑↓↑↓…⟩ evaluates to −1 on odd links and 0 on even
//     links (from the alternating σ^z sum collapsing against c_n). So
//     L_n² = 1 on odd links, 0 on even, and:
//       ⟨H_E⟩ = (g²a/2) · (number of odd n in [1, N-1]) · 1 = (g²a/2)(N/2)
//             = g²aN/4   for even N.
//
// So at L₀=0, even N, m → ∞ the asymptotic GS energy is
//
//   E_∞  =  -mN/2  +  g²aN/4  +  O(t²/m)
//
// where t = 1/(2a) is the hopping scale and the leading correction is
// second-order perturbation theory through |↑↓↑↓…⟩'s connected hopping
// neighbours. We test at m=100 with N=4,6,8; the absolute |E_dmrg − E_∞|
// should be ≲ (N−1)/(m·a²) · O(1) per the perturbative scale.
//
// Together these two limits independently exercise:
//
//   • the hopping-term coefficient and JW mapping  (free-fermion limit)
//   • the (-1)^n staggering sign of H_m,
//     the L_n² expansion in H_E,
//     the c_n / A_k tail-sum bookkeeping       (strong-coupling limit)
//
// Either limit failing while MPO-vs-dense passes would localize the bug
// to the build_schwinger_*() functions (both paths share the same formula
// constants), not to AutoMPO or DMRG.

#include "quantum/SchwingerModel.hpp"

#include <itensor/all.h>

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace tessera::quantum;
using itensor::dmrg;
using itensor::Sweeps;
using itensor::MPS;
using itensor::InitState;

namespace {

double dmrg_groundstate_sz0(SchwingerMPO const& sm, int maxBondDim, int nSweeps) {
    auto state = InitState(sm.sites);
    for (int i = 1; i <= sm.params.N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    auto psi0 = MPS(state);

    auto sweeps = Sweeps(nSweeps);
    sweeps.maxdim() = 20, 40, 80, maxBondDim, maxBondDim;
    sweeps.cutoff() = 1e-12;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-7, 1e-8, 0.0;

    auto [energy, psi] = dmrg(sm.H, psi0, sweeps, {"Silent=", true});
    return energy;
}

// Free-fermion half-filled OBC chain with hopping coefficient +1/(2a) on
// (c† c + h.c.) terms gives ε_j = (1/a) cos(π j/(N+1)). Half-filling sums
// the N/2 most-negative levels (j > (N+1)/2).
double free_fermion_half_filled_energy(int N, double a) {
    if (N % 2 != 0 || N < 2) {
        throw std::invalid_argument("free_fermion test expects even N >= 2");
    }
    double sum = 0.0;
    for (int j = N / 2 + 1; j <= N; ++j) {
        sum += std::cos(std::numbers::pi * j / (N + 1));
    }
    return sum / a;
}

bool check_free_fermion(int N) {
    SchwingerParams p;
    p.N = N; p.a = 1.0; p.g = 0.0; p.m = 0.0; p.L0 = 0.0;

    auto mpo = SchwingerHamiltonian{p}.mpo();
    const double e_dmrg = dmrg_groundstate_sz0(mpo, 64, 10);
    const double e_total = e_dmrg + mpo.constant;
    const double e_analytic = free_fermion_half_filled_energy(N, p.a);
    const double diff = std::abs(e_total - e_analytic);

    constexpr double tol = 1e-8;
    const bool pass = diff < tol;
    std::cout
        << "  N=" << N
        << "  E_dmrg+const=" << e_total
        << "  E_analytic="   << e_analytic
        << "  |Δ|="          << diff
        << (pass ? "  PASS" : "  FAIL") << "\n";
    return pass;
}

bool check_strong_coupling(int N, double m) {
    SchwingerParams p;
    p.N = N; p.a = 1.0; p.g = 1.0; p.m = m; p.L0 = 0.0;

    auto mpo = SchwingerHamiltonian{p}.mpo();
    const double e_dmrg = dmrg_groundstate_sz0(mpo, 100, 14);
    const double e_total = e_dmrg + mpo.constant;
    // Leading-order asymptotic GS energy on |↑↓↑↓...⟩ for L0=0:
    //   mass term:     -mN/2
    //   electric term: g²aN/4   (every odd link in [1,N-1] gives L_n^2 = 1)
    const double e_asymptotic = -m * N / 2.0 + p.g * p.g * p.a * N / 4.0;
    const double diff = std::abs(e_total - e_asymptotic);

    // Leading correction is second-order perturbation theory through
    // hopping: each adjacent (Up, Dn) bond can virtually flip with energy
    // denominator ~ m/a², contributing O(t²/m) per bond with t = 1/(2a).
    // (N − 1) bonds, so the total deviation scales as ~ (N−1)/(m·a²).
    // Allowing 2× that gives generous slack against bond-dim truncation.
    const double allowed = (N - 1) / (m * p.a * p.a) * 2.0;
    const bool pass = diff < allowed;
    std::cout
        << "  N=" << N << " m=" << m
        << "  E_dmrg+const=" << e_total
        << "  E_asympt="      << e_asymptotic
        << "  |Δ|="           << diff
        << "  (allowed " << allowed << ")"
        << (pass ? "  PASS" : "  FAIL") << "\n";
    return pass;
}

} // namespace

int main() {
    bool ok = true;

    std::cout << "Free-fermion analytic limit (g=0, m=0)\n";
    std::cout << "--------------------------------------\n";
    for (int N : {4, 6, 8, 12}) {
        if (!check_free_fermion(N)) ok = false;
    }

    std::cout << "\nStrong-coupling vacuum limit (m large, g=a=1, L0=0)\n";
    std::cout << "----------------------------------------------------\n";
    for (int N : {4, 6, 8}) {
        if (!check_strong_coupling(N, /*m=*/100.0)) ok = false;
    }

    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
