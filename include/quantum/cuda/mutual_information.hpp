// Mutual information utilities, mirroring src/quantum/mutual_information.cpp
// but acting on torch::Tensor density matrices and tessera::quantum::cuda::MPS
// states.

#pragma once

#include "quantum/cuda/mps.hpp"

#include <torch/torch.h>

namespace tessera::quantum::cuda {

class MutualInformation {
public:
    // S(rho) = -Tr(rho log rho) for the eigenvalues > tol of rho.
    // Symmetrises rho before diagonalising so numerical asymmetry
    // doesn't break the Hermitian eigensolver.
    static double vonNeumannEntropy(torch::Tensor const& rho,
                                      double tol = 1e-12);

    // ℓ = -log(I) with infinity floor below `epsilon`.
    static double edgeLength(double I, double epsilon = 1e-6);

    // I(i:j) = S(rho_i) + S(rho_j) - S(rho_{ij}). Triggers canonical-
    // form moves on `psi` (so psi is taken by non-const reference).
    static double siteSite(MPS& psi, int64_t i, int64_t j);

    // Symmetric (N, N) MI matrix returned as a real float64 tensor.
    // Diagonal is zero by convention; single-site entropies are cached
    // so each pair only triggers one fresh two-site RDM extraction.
    static torch::Tensor allPairs(MPS& psi);
};

} // namespace tessera::quantum::cuda
