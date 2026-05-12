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

namespace tessera::quantum {

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
};

} // namespace tessera::quantum
