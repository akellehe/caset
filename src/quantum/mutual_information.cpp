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

    auto site_i = siteIndex(psi, i);
    auto site_j = siteIndex(psi, j);

    // Transfer-matrix sweep from site i to site j. At every step the
    // intermediate tensor T carries four open indices — (site_i,
    // site_i', current_right_bond_ket, current_right_bond_bra) — so
    // its size is O(χ²) regardless of (j - i). Interior site indices
    // are unprimed on the bra side and auto-trace against the ket.
    // The left bond at site i and the right bond at site j auto-
    // contract via the orth-canonical / right-canonical conditions
    // (psi.position(i) places sites 1..i-1 in left-canonical form and
    // sites i+1..N in right-canonical form), giving identity
    // environments at both boundaries.
    //
    // The previous implementation contracted sites i..j into a single
    // dense tensor before forming rho; that accumulates 2^(j-i+1)
    // physical-site elements and is intractable on long doubled
    // chains (e.g. the Choi state's (in_1, out_N) pair).

    auto primeBondAt = [&](ITensor& tensor, int bond) {
        // bond ∈ [1, N-1]: link index between sites `bond` and bond+1.
        if (bond < 1 || bond > N - 1) return;
        auto link = commonIndex(psi(bond), psi(bond + 1));
        if (link) tensor.prime(link);
    };

    // Site i: prime site_i on the bra (keep it open) and the right
    // bond on the bra (keep ket/bra bonds distinct as we sweep).
    ITensor bra = dag(psi(i));
    bra.prime(site_i);
    primeBondAt(bra, i);
    ITensor T = psi(i);
    T *= bra;

    // Interior sites i+1 .. j-1: prime left+right bonds on the bra so
    // its chain stays distinct; site_k is unprimed → auto-traces.
    for (int k = i + 1; k <= j - 1; ++k) {
        ITensor braK = dag(psi(k));
        primeBondAt(braK, k - 1);
        primeBondAt(braK, k);
        T *= psi(k);
        T *= braK;
    }

    // Site j: prime site_j on the bra and the left bond on the bra
    // (continuing the primed chain from j-1). The right bond is left
    // unprimed so it auto-traces against the right-canonical
    // environment past j.
    ITensor braJ = dag(psi(j));
    braJ.prime(site_j);
    primeBondAt(braJ, j - 1);
    T *= psi(j);
    T *= braJ;

    // T now has only (site_i, site_i', site_j, site_j') open.
    Eigen::Matrix4cd out;
    out.setZero();
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 2; ++b) {
            for (int c = 1; c <= 2; ++c) {
                for (int d = 1; d <= 2; ++d) {
                    const auto v = eltC(T,
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
