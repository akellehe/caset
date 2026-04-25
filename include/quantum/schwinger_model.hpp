// Schwinger-model Hamiltonian on a Jordan-Wigner'd staggered-fermion chain
// with Gauss's law eliminated. Constructs an ITensor MPO via AutoMPO, plus a
// dense Eigen reference matrix for small-N cross-checks.
//
// ─── Physics ──────────────────────────────────────────────────────────────
//
// We implement the dimensional spin Hamiltonian from PLAN.md §4 (1-based
// site indexing n = 1..N), which is unitarily equivalent (via a global spin
// flip σ^z → −σ^z combined with an index relabeling) to Bañuls et al., JHEP
// 11, 158 (2013) eq. (2.6) at L₀ = 0:
//
//     H = H_hop + H_m + H_E
//
//     H_hop = (1/(4a)) Σ_{n=1..N-1}  (X_n X_{n+1} + Y_n Y_{n+1})
//           = (1/(2a)) Σ_{n=1..N-1}  (σ⁺_n σ⁻_{n+1} + σ⁻_n σ⁺_{n+1})
//
//     H_m   = (m/2) Σ_{n=1..N}       (-1)^n σ^z_n
//
//     H_E   = (g²a/2) Σ_{n=1..N-1}   L_n²
//
//     L_n   = L₀ + Σ_{k=1..n} [(1 - σ^z_k)/2  -  (1 - (-1)^k)/2]
//           = c_n - (1/2) Σ_{k=1..n} σ^z_k         (after simplifying)
//
//     c_n   = L₀ + ((-1)^n - 1)/4                  (a c-number per link n)
//
// Bañuls' dimensionless parameters are x = 1/(g²a²) and μ = 2m/(g²a). Our
// dimensional energy E_dim relates to their dimensionless eigenvalue E_W as
// E_W = (2/(ag²)) E_dim.
//
// ─── What this header exposes ─────────────────────────────────────────────
//
// • SchwingerParams — dimensional inputs (N, a, m, g, L₀).
// • SchwingerMPO    — ITensor MPO + the SiteSet it lives on, plus an
//                     additive c-number `constant` that is the part of L_n²
//                     that's pure-identity. The MPO encodes only operator-
//                     valued terms; full physical energy = ⟨H_MPO⟩ + constant.
// • SchwingerDense  — same Hamiltonian as a 2^N×2^N real-symmetric matrix
//                     (no symmetry reduction). Used for small-N cross-checks
//                     against the MPO/DMRG path.
//
// Why split off `constant`: AutoMPO encodes only operator-valued terms; the
// pure-identity part of L_n² is a c-number (E_const, see schwinger_model.cpp)
// that a callers needs to add back to compare against any reference value.
// Returning it as a separate field lets callers compute either the operator
// spectrum (matches the MPO eigenvalues directly) or the full physical
// energy (matches Bañuls' published values after the dimensional rescaling).

#pragma once

#include <itensor/all.h>

#include <Eigen/Dense>

#include <cstddef>

namespace caset::quantum {

// Dimensional inputs to the Schwinger Hamiltonian. All in lattice units —
// users wanting Bañuls' dimensionless x, μ should set a = 1 and choose
// g = 1/√x and m = (μ/2) g² a accordingly.
struct SchwingerParams {
    int    N{0};   // staggered sites, 1-based; must be ≥ 2 (≥ 4 for nontrivial H_E)
    double a{1.0}; // lattice spacing  (must be > 0)
    double m{0.0}; // bare fermion mass
    double g{1.0}; // gauge coupling   (g = 0 is the free-Dirac limit; allowed)
    double L0{0.0};// background electric field on the link to the left of site 1
};

// Operator-valued Schwinger Hamiltonian as an ITensor MPO.
//
// Total physical energy on a state |ψ⟩ is ⟨ψ|H|ψ⟩ + constant.
struct SchwingerMPO {
    SchwingerParams params;
    itensor::SpinHalf sites;  // SpinHalf SiteSet; carries QN structure if enabled
    itensor::MPO H;           // operator-valued part of the Hamiltonian
    double constant{0.0};     // c-number shift from expanding L_n² (see .cpp)
};

// Build the Schwinger MPO via ITensor's AutoMPO.
//
// `conserve_qns = true` (default) makes the bond indices carry total Sz, i.e.
// total electric charge after JW. This blocks the MPO/MPS into U(1) sectors,
// makes DMRG converge faster in the charge-neutral sector, and lets us pin
// the GS sector via a Néel initial state.
//
// Pass `conserve_qns = false` for measurements with operators that don't
// preserve Sz (e.g. σ^x, the building block of the lattice charge-conjugation
// S_R = σ^x_odd · T⁽¹⁾ on Bañuls page 8). With QNs enabled, applying σ^x to
// an MPS would require ITensor to handle indefinite-flux tensors, which it
// doesn't cleanly support; the path of least resistance is to rebuild H
// without symmetry tracking for that one measurement.
SchwingerMPO build_schwinger_mpo(SchwingerParams const& p, bool conserve_qns = true);

// Dense 2^N×2^N reference Hamiltonian. No symmetry reduction; bit n of the
// row index corresponds to spin n in the convention documented in the .cpp
// file. Hard-capped at N=16 (= 2^16 × 2^16 = 32 GB of doubles, already too
// big — practical use is N ≤ 12).
struct SchwingerDense {
    SchwingerParams params;
    Eigen::MatrixXd H;     // 2^N × 2^N, real symmetric
    double constant{0.0};  // same c-number shift as in SchwingerMPO
};
SchwingerDense build_schwinger_dense(SchwingerParams const& p);

// The c-number part of L_n² accumulated over n = 1..N-1, multiplied by the
// (g²a/2) prefactor in front of H_E. Returned alongside the MPO/dense H so
// callers can recover the full physical energy by adding it to ⟨H⟩.
//
//   constant = (g²a/2) Σ_{n=1..N-1} (c_n² + n/4)
//
// (Derivation in the .cpp file.)
double schwinger_energy_constant(SchwingerParams const& p);

} // namespace caset::quantum
