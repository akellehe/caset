// Implementation of MutualInformation — see mutual_information.hpp for
// the architectural overview and the canonical-form derivation.

#include "quantum/mutual_information.hpp"

#include <itensor/all.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tessera::quantum {

namespace {

// SpinHalf basis convention: ITensor's "Up" index value is 1 and "Dn"
// is 2 (1-based). We pack (s_i, s_j) into a 4×4 matrix row index
// using row = 2*(s_i - 1) + (s_j - 1) so the basis order matches the
// header docstring: (|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩) at rows 0..3.
inline int packTwoSite(int s_i, int s_j) noexcept {
    return 2 * (s_i - 1) + (s_j - 1);
}

// Symmetrise a candidate Hermitian matrix by averaging with its
// adjoint. Numerical noise breaks exact Hermiticity by O(ε); the fix
// is cheap and keeps SelfAdjointEigenSolver happy downstream.
template <typename MatT>
MatT symmetrise(MatT const& m) {
    return 0.5 * (m + m.adjoint());
}

double entropyFromEigenvalues(Eigen::ArrayXd const& eigs, double tol) {
    double S = 0.0;
    for (Eigen::Index k = 0; k < eigs.size(); ++k) {
        const double p = eigs[k];
        if (p > tol) S -= p * std::log(p);
    }
    return S;
}

} // namespace

Eigen::Matrix2cd
MutualInformation::oneSiteReducedDensity(itensor::MPS const& psi_in, int i) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i < 1 || i > N) {
        throw std::invalid_argument(
            "MutualInformation::oneSiteReducedDensity: i out of range [1, N]");
    }

    MPS psi = psi_in;
    psi.position(i);

    // Contract psi(i) with dag(psi(i)) but with the site index primed
    // on the bra side. The orthogonality center handles both bond
    // sums implicitly — left bond is trivially a δ from the canonical
    // chain to its left, and right bond is δ from the right-canonical
    // chain to its right.
    auto site = siteIndex(psi, i);
    auto bra = dag(psi(i));
    bra.prime(site);

    ITensor rho = psi(i) * bra;
    // rho now has indices (site_unprimed, site_primed) — a rank-2
    // tensor that IS the 2×2 reduced density matrix.

    Eigen::Matrix2cd out;
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 2; ++b) {
            // a is the unprimed site value (ket); b is the primed
            // value (bra) — so out(a-1, b-1) = ⟨a | ρ | b⟩.
            out(a - 1, b - 1) = eltC(rho, site = a, prime(site) = b);
        }
    }
    return symmetrise(out);
}

Eigen::Matrix4cd
MutualInformation::twoSiteReducedDensity(itensor::MPS const& psi_in,
                                          int i, int j) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i < 1 || j < 1 || i > N || j > N) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: site index out of [1, N]");
    }
    if (i > j) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: require i <= j");
    }
    if (i == j) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: require i < j; "
            "use oneSiteReducedDensity for the i == j case");
    }

    MPS psi = psi_in;
    psi.position(i);

    // Contract sites i..j into a single tensor T. Its indices are:
    //   left_bond (between i-1 and i), site_i, site_{i+1}, …, site_j,
    //   right_bond (between j and j+1).
    ITensor T = psi(i);
    for (int k = i + 1; k <= j; ++k) {
        T *= psi(k);
    }

    auto site_i = siteIndex(psi, i);
    auto site_j = siteIndex(psi, j);

    // The bra is T daggered, with site indices at i and j primed so
    // they stay open in the contraction. Site indices for k ∈ (i, j)
    // are left unprimed → they auto-contract with the ket (tracing
    // out those interior sites). Bond indices are unprimed on both
    // sides → they also auto-contract; the canonical form makes the
    // left and right bond contractions equal to identity.
    ITensor Td = dag(T);
    Td.prime(site_i);
    Td.prime(site_j);

    ITensor rho = T * Td;
    // rho's open indices are (site_i, site_i', site_j, site_j') — a
    // rank-4 tensor whose flat 4×4 representation is the joint
    // reduced density matrix ρ_{ij}.

    Eigen::Matrix4cd out;
    out.setZero();
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 2; ++b) {
            for (int c = 1; c <= 2; ++c) {
                for (int d = 1; d <= 2; ++d) {
                    // Ket indices: site_i = a, site_j = b
                    // Bra indices: site_i' = c, site_j' = d
                    const auto v = eltC(rho,
                                          site_i = a, prime(site_i) = c,
                                          site_j = b, prime(site_j) = d);
                    out(packTwoSite(a, b), packTwoSite(c, d)) = v;
                }
            }
        }
    }
    return symmetrise(out);
}

double
MutualInformation::vonNeumannEntropy(Eigen::Matrix2cd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::vonNeumannEntropy(Eigen::Matrix4cd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::vonNeumannEntropy(Eigen::MatrixXcd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::siteSite(itensor::MPS const& psi, int i, int j) {
    if (i == j) return 0.0;
    if (i > j) std::swap(i, j);

    auto rho_i  = oneSiteReducedDensity(psi, i);
    auto rho_j  = oneSiteReducedDensity(psi, j);
    auto rho_ij = twoSiteReducedDensity(psi, i, j);

    const double Si  = vonNeumannEntropy(rho_i);
    const double Sj  = vonNeumannEntropy(rho_j);
    const double Sij = vonNeumannEntropy(rho_ij);
    return Si + Sj - Sij;
}

Eigen::MatrixXd
MutualInformation::allPairs(itensor::MPS const& psi) {
    using namespace itensor;
    const int N = length(psi);
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(N, N);
    if (N < 2) return out;

    // Cache single-site entropies — each pair would otherwise recompute
    // S(ρ_i) and S(ρ_j) redundantly.
    std::vector<double> S_single(N + 1, 0.0);  // 1-based
    for (int i = 1; i <= N; ++i) {
        S_single[i] = vonNeumannEntropy(oneSiteReducedDensity(psi, i));
    }

    for (int i = 1; i <= N; ++i) {
        for (int j = i + 1; j <= N; ++j) {
            const auto rho_ij = twoSiteReducedDensity(psi, i, j);
            const double Sij  = vonNeumannEntropy(rho_ij);
            const double I_ij = S_single[i] + S_single[j] - Sij;
            // Symmetric, 0-based output matrix.
            out(i - 1, j - 1) = I_ij;
            out(j - 1, i - 1) = I_ij;
        }
    }
    return out;
}

double
MutualInformation::edgeLength(double I, double epsilon) noexcept {
    if (I < epsilon) return std::numeric_limits<double>::infinity();
    return -std::log(I);
}

} // namespace tessera::quantum
