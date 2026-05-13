// MutualInformation implementation. See include/quantum/cuda/mutual_information.hpp
// for the architectural overview.

#include "quantum/cuda/mutual_information.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tessera::quantum::cuda {

double MutualInformation::vonNeumannEntropy(torch::Tensor const& rho,
                                              double tol) {
    if (rho.dim() != 2 || rho.size(0) != rho.size(1)) {
        throw std::invalid_argument(
            "vonNeumannEntropy: rho must be a square matrix");
    }
    auto sym = 0.5 * (rho + rho.conj().transpose(0, 1));
    auto eigs = torch::linalg_eigvalsh(sym, "L").to(torch::kFloat64);
    auto eigs_cpu = eigs.cpu().contiguous();
    auto const* data = eigs_cpu.data_ptr<double>();
    double S = 0.0;
    for (int64_t k = 0; k < eigs_cpu.numel(); ++k) {
        double p = data[k];
        if (p > tol) S -= p * std::log(p);
    }
    return S;
}

double MutualInformation::edgeLength(double I, double epsilon) {
    if (I < epsilon) return std::numeric_limits<double>::infinity();
    return -std::log(I);
}

double MutualInformation::siteSite(MPS& psi, int64_t i, int64_t j) {
    if (i == j) return 0.0;
    if (i > j) std::swap(i, j);
    auto rho_i  = psi.oneSiteReducedDensity(i);
    auto rho_j  = psi.oneSiteReducedDensity(j);
    auto rho_ij = psi.twoSiteReducedDensity(i, j);
    const double S_i  = vonNeumannEntropy(rho_i);
    const double S_j  = vonNeumannEntropy(rho_j);
    const double S_ij = vonNeumannEntropy(rho_ij);
    return S_i + S_j - S_ij;
}

torch::Tensor MutualInformation::allPairs(MPS& psi) {
    auto N = psi.length();
    auto opts = torch::TensorOptions()
                  .device(psi.device()).dtype(torch::kFloat64);
    auto out = torch::zeros({N, N}, opts);
    if (N < 2) return out;

    std::vector<double> S_single(static_cast<std::size_t>(N), 0.0);
    for (int64_t i = 0; i < N; ++i) {
        S_single[static_cast<std::size_t>(i)] =
            vonNeumannEntropy(psi.oneSiteReducedDensity(i));
    }
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = i + 1; j < N; ++j) {
            auto rho_ij = psi.twoSiteReducedDensity(i, j);
            const double S_ij = vonNeumannEntropy(rho_ij);
            const double I_ij = S_single[static_cast<std::size_t>(i)]
                                 + S_single[static_cast<std::size_t>(j)]
                                 - S_ij;
            out.index_put_({i, j}, I_ij);
            out.index_put_({j, i}, I_ij);
        }
    }
    return out;
}

} // namespace tessera::quantum::cuda
