// Mutual information on a Schwinger MPS state.
//
// Two-site reduced density matrices for arbitrary site pairs (i, j),
// von Neumann entropy of a small Hermitian matrix, and the resulting
// site-site mutual information I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
//
// The implementation uses ITensor canonical form: with orthogonality
// center brought to site i, sites 1..i-1 are left-canonical and sites
// j+1..N are right-canonical, so the partial trace over everything
// except {i, j} reduces to a single tensor contraction
// ρ_{ij} = T · dag(T) over the contracted block T = A_i ⊗ A_{i+1} ⊗
// … ⊗ A_j, with site indices at i and j primed on the bra side to keep
// them open (Hauschild–Pollmann §III.A).
//
// References:
//   Hauschild, Pollmann, "Efficient numerical simulations with Tensor
//   Networks", SciPost Phys. Lect. Notes 5 (2018), §III.

#pragma once

#include <Eigen/Dense>
#include <itensor/all.h>

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

// Static utility class. Not instantiable.
class MutualInformation {
public:
    MutualInformation() = delete;
    MutualInformation(MutualInformation const&) = delete;
    MutualInformation& operator=(MutualInformation const&) = delete;

    // Single-site reduced density matrix at 1-based site i, returned as
    // a 2x2 Hermitian complex matrix. Bringing the orthogonality center
    // to i and contracting with dag yields ρ_i in one step.
    [[nodiscard]] static Eigen::Matrix2cd
    oneSiteReducedDensity(itensor::MPS const& psi, int i);

    // Two-site reduced density matrix on 1-based sites (i, j) with i ≤ j.
    // Returned as a 4x4 Hermitian complex matrix in the basis order
    // (|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩) — i.e. row index = 2*s_i + s_j with
    // |↑⟩ ↔ 0, |↓⟩ ↔ 1 (matching ITensor SpinHalf indexing).
    //
    // Throws std::invalid_argument when i, j fall outside [1, N] or
    // when i > j.
    [[nodiscard]] static Eigen::Matrix4cd
    twoSiteReducedDensity(itensor::MPS const& psi, int i, int j);

    // Von Neumann entropy of a Hermitian density matrix, in nats.
    // Eigenvalues below ``tol`` (default 1e-12) contribute zero,
    // avoiding the log(0) singularity. The dynamic-size overload
    // makes the Python binding trivial; the fixed-size overloads exist
    // for callers (C++) that want allocation-free fast paths.
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::Matrix2cd const& rho, double tol = 1e-12);
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::Matrix4cd const& rho, double tol = 1e-12);
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::MatrixXcd const& rho, double tol = 1e-12);

    // Site-site mutual information I({i} : {j}) on the MPS, in nats.
    // = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
    [[nodiscard]] static double
    siteSite(itensor::MPS const& psi, int i, int j);

    // All-pairs site-site mutual information as a symmetric N×N matrix
    // (zero diagonal). Computed in one pass over the MPS — the canonical
    // form is maintained between pairs so each pair costs O(N χ³).
    [[nodiscard]] static Eigen::MatrixXd
    allPairs(itensor::MPS const& psi);

    // Edge length ℓ = -log(I) with infinity floor at -log(epsilon).
    // Returns +inf when I < epsilon. Used by EmergentGraph to convert
    // mutual information to a metric weight.
    [[nodiscard]] static double
    edgeLength(double I, double epsilon = 1e-10) noexcept;

    // ── Dual / bond-cut observables (van Raamsdonk graph) ──────────────
    //
    // Contiguous-interval entropy and bond-cut tripartite information
    // power the bond-cut spectral-dimension pipeline that uses
    // bipartitions of the chain as graph vertices (one per bond, one
    // per snapshot). See
    // ``docs/source/quantum-experiments/earlier-work/emergent-spectral-dimension-schwinger-tdvp.md``
    // §3 (Dual lattice / bond-cut graph) for the construction.

    // Bipartite entanglement entropy at bond ``k`` (1-based, between
    // sites k and k+1, 1 ≤ k ≤ N-1) of an MPS. Computed from the
    // Schmidt spectrum of the bipartition |1..k] / [k+1..N|.
    [[nodiscard]] static double
    bondEntropy(itensor::MPS const& psi, int k);

    // Entropy of the reduced density matrix on the contiguous interval
    // [i, j] (1-based, inclusive, 1 ≤ i ≤ j ≤ N). Uses a χ⁴ transfer-
    // matrix sweep — never materialises ρ in the (d^L, d^L) basis, so
    // memory stays bounded at O(χ²) and runtime at O((j-i) χ⁵). The
    // returned entropy is in nats.
    [[nodiscard]] static double
    regionEntropy(itensor::MPS const& psi, int i, int j);

    // Tripartite information between bonds n < m (both 1-based,
    // 1 ≤ n < m ≤ N-1):
    //   I(A : C) = S(A) + S(C) - S(B), with
    //     A = [1..n], B = [n+1..m], C = [m+1..N].
    // For a pure state |ψ⟩, S(A ∪ C) = S(B) so this equals the
    // standard mutual information between the two outer regions and
    // captures how strongly the two cuts are entangled through the
    // bulk middle (the van Raamsdonk picture: bond ↔ minimal surface,
    // tripartite info ↔ mutual connectivity).
    [[nodiscard]] static double
    tripartiteInformation(itensor::MPS const& psi, int n, int m);

    // All-pairs bond-cut tripartite information as a symmetric
    // (N-1) × (N-1) matrix with zero diagonal. Caches the per-bond
    // ``bondEntropy`` once and shares the interval-entropy sweep
    // structure across pairs so the per-pair cost is O(N χ⁵).
    [[nodiscard]] static Eigen::MatrixXd
    allBondPairs(itensor::MPS const& psi);
};

} // namespace tessera::quantum
