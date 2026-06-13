// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include <complex>

namespace tessera {
namespace cuda {

/// GPU (cuSOLVER) dense self-adjoint eigensolvers -- an OPT-IN accelerator for
/// HodgeLaplacian's CPU (Eigen `SelfAdjointEigenSolver`) eigendecomposition.
/// Same result to machine precision: ascending eigenvalues, orthonormal
/// eigenvectors (unique up to sign/phase, and -- within a degenerate eigenspace
/// such as ker L_k -- up to an orthonormal basis choice, exactly as on the CPU).
///
/// Both take the matrix in COLUMN-MAJOR layout (Eigen's native `.data()`),
/// `a[i + j*n] = A(i, j)`, and write:
///   - `evals` (length n): the n eigenvalues in ascending order;
///   - `evecs` (length n*n): `evecs[i*n + j]` = the i-th component of the j-th
///     eigenvector -- matching the row-major storage HodgeLaplacian already uses
///     (`sp.evecs[i*nk + j] = V(i, j)`).
/// Each call allocates, uploads, solves, and downloads independently (no shared
/// device state), so they are safe to call from anywhere the CPU path is called.

/// Real symmetric eigendecomposition via `cusolverDnDsyevd`.
/// @throws std::runtime_error on any CUDA/cuSOLVER failure (caller falls back).
void symmetric_eigh(const double *a, int n, double *evals, double *evecs);

/// Hermitian eigendecomposition via `cusolverDnZheevd`.
/// @throws std::runtime_error on any CUDA/cuSOLVER failure (caller falls back).
void hermitian_eigh(const std::complex<double> *a, int n, double *evals,
                    std::complex<double> *evecs);

}  // namespace cuda
}  // namespace tessera
