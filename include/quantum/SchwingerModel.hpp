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
//           = c_n - (1/2) Σ_{k=1..n} σ^z_k
//
//     c_n   = L₀ + ((-1)^n - 1)/4
//
// Bañuls' dimensionless parameters are x = 1/(g²a²) and μ = 2m/(g²a). The
// dimensional energy E_dim relates to their dimensionless eigenvalue E_W as
// E_W = (2/(ag²)) E_dim.
//
// ─── What this header exposes ─────────────────────────────────────────────
//
// • SchwingerParams — dimensional inputs (data class).
// • SchwingerMPO    — ITensor MPO + the SiteSet plus the c-number constant
//                     (data class).
// • SchwingerDense  — same Hamiltonian as a 2^N×2^N real-symmetric matrix
//                     (data class).
// • SchwingerHamiltonian — coarse-grained builder that bundles a
//                     SchwingerParams with the various Hamiltonian
//                     representations downstream code consumes (MPO for
//                     DMRG / TDVP, dense matrix for small-N cross-checks).
//
// Why split off `constant`: AutoMPO encodes only operator-valued terms; the
// pure-identity part of L_n² is a c-number (E_const, see schwinger_model.cpp)
// that callers need to add back to compare against any reference value.
// SchwingerMPO and SchwingerDense expose it as a member so callers can
// recover the full physical energy (= ⟨H⟩ + constant) without rebuilding.

#pragma once

#include <itensor/all.h>

#include <Eigen/Dense>

#include <cstddef>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

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

// Dense 2^N × 2^N reference Hamiltonian. No symmetry reduction; bit n of the
// row index corresponds to spin n in the convention documented in the .cpp
// file. Hard-capped at N=16; practical use is N ≤ 12.
struct SchwingerDense {
    SchwingerParams params;
    Eigen::MatrixXd H;     // 2^N × 2^N, real symmetric
    double constant{0.0};  // same c-number shift as in SchwingerMPO
};

// Coarse-grained interface for Schwinger-Hamiltonian construction.
//
// One instance binds a SchwingerParams; the methods build the various
// Hamiltonian representations downstream code consumes (MPO for DMRG /
// TDVP, dense matrix for small-N cross-checks). The builder is stateless
// beyond its parameters — every method returns a freshly assembled
// representation.
//
// `conserveQns = true` (default on `mpo()` / `mpoChain()`) makes the bond
// indices carry total Sz, i.e. total electric charge after JW. This blocks
// the MPO/MPS into U(1) sectors, makes DMRG converge faster in the
// charge-neutral sector, and lets us pin the GS sector via a Néel initial
// state. Pass false for measurements with operators that don't preserve
// Sz (e.g. σ^x, the building block of the lattice charge-conjugation
// operator on Bañuls page 8).
class SchwingerHamiltonian {
public:
    explicit SchwingerHamiltonian(SchwingerParams params) noexcept;

    [[nodiscard]] SchwingerParams const& params() const noexcept { return params_; }

    // Build the operator-valued MPO via ITensor's AutoMPO, with the
    // standard 1D nearest-neighbour hopping graph.
    [[nodiscard]] SchwingerMPO mpo(bool conserveQns = true) const;

    // Chain-causet variant. The hopping graph for H_hop is
    // supplied externally as a list of (site_i, site_j) pairs in 0-based
    // flat lattice indexing. Mass and electric-field terms remain the
    // standard 1D formulas — they're well-defined as long as the lattice
    // has a linear ordering, which is guaranteed when the underlying
    // causet is a chain (one vertex per time slice).
    //
    // For a chain causet (every antichain has one vertex), `hoppingPairs`
    // is exactly `[(0,1), (1,2), …, (N-2, N-1)]` and the resulting MPO
    // is bit-for-bit identical to `mpo(conserveQns)` — that's the
    // chain-causet sanity equality.
    //
    // Sites in `hoppingPairs` are 0-based flat indices and must be in
    // [0, p.N - 1]. ITensor's site indexing is 1-based internally; we
    // add 1 when feeding AutoMPO.
    [[nodiscard]] SchwingerMPO mpoChain(
        std::vector<std::pair<int, int>> const& hoppingPairs,
        bool conserveQns = true) const;

    // Dense 2^N × 2^N Hamiltonian, no symmetry reduction. Throws
    // std::invalid_argument when N < 2 or N > 16.
    [[nodiscard]] SchwingerDense denseMatrix() const;

    // The c-number part of L_n² accumulated over n = 1..N-1, multiplied
    // by the (g²a/2) prefactor in front of H_E. Returned alongside the
    // MPO/dense H so callers can recover the full physical energy by
    // adding it to ⟨H⟩.
    //
    //   constant = (g²a/2) Σ_{n=1..N-1} (c_n² + n/4)
    //
    // (Derivation in the .cpp file.)
    [[nodiscard]] double constant() const;

private:
    SchwingerParams params_;
};

} // namespace tessera::quantum
