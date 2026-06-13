// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// GPU (cuSOLVER) dense self-adjoint eigensolvers backing the opt-in GPU path of
// HodgeLaplacian. Host-callable wrappers around cusolverDnDsyevd (real
// symmetric) and cusolverDnZheevd (Hermitian); cuSOLVER's dense divide-and-
// conquer drivers return ascending eigenvalues and orthonormal eigenvectors,
// matching Eigen's SelfAdjointEigenSolver convention.

#include "cuda/hodge_eigensolver_cuda.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace tessera {
namespace cuda {

namespace {

void cudaCheck(cudaError_t e, const char *what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                             cudaGetErrorString(e));
}

void solverCheck(cusolverStatus_t s, const char *what) {
  if (s != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuSOLVER error in ") + what +
                             " (status=" + std::to_string(static_cast<int>(s)) +
                             ")");
}

// RAII for a cuSOLVER handle + a set of device pointers, so every throw path
// frees cleanly.
struct DeviceScope {
  cusolverDnHandle_t handle = nullptr;
  std::vector<void *> ptrs;
  DeviceScope() { solverCheck(cusolverDnCreate(&handle), "cusolverDnCreate"); }
  ~DeviceScope() {
    for (void *p : ptrs)
      if (p) cudaFree(p);
    if (handle) cusolverDnDestroy(handle);
  }
  template <typename T>
  T *alloc(std::size_t count) {
    void *p = nullptr;
    cudaCheck(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
    ptrs.push_back(p);
    return static_cast<T *>(p);
  }
};

}  // namespace

void symmetric_eigh(const double *a, int n, double *evals, double *evecs) {
  if (n <= 0) return;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  DeviceScope dev;
  double *dA = dev.alloc<double>(nn);
  double *dW = dev.alloc<double>(static_cast<std::size_t>(n));
  int *dInfo = dev.alloc<int>(1);
  // `a` is column-major; for a symmetric matrix column-major == row-major, so it
  // uploads directly as cuSOLVER's column-major A.
  cudaCheck(cudaMemcpy(dA, a, nn * sizeof(double), cudaMemcpyHostToDevice),
            "memcpy A");

  int lwork = 0;
  solverCheck(
      cusolverDnDsyevd_bufferSize(dev.handle, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, &lwork),
      "Dsyevd_bufferSize");
  double *dWork = dev.alloc<double>(static_cast<std::size_t>(lwork));
  solverCheck(cusolverDnDsyevd(dev.handle, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, dWork,
                               lwork, dInfo),
              "Dsyevd");
  int info = 0;
  cudaCheck(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost),
            "memcpy info");
  if (info != 0)
    throw std::runtime_error("cusolverDnDsyevd: eigensolve did not converge "
                             "(info=" + std::to_string(info) + ")");

  cudaCheck(cudaMemcpy(evals, dW, static_cast<std::size_t>(n) * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "memcpy W");
  std::vector<double> V(nn);
  cudaCheck(cudaMemcpy(V.data(), dA, nn * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "memcpy V");
  // dA now holds eigenvectors as columns (column-major): V[i + j*n] = i-th
  // component of the j-th eigenvector. Store as evecs[i*n + j].
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      evecs[static_cast<std::size_t>(i) * n + j] =
          V[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
}

void hermitian_eigh(const std::complex<double> *a, int n, double *evals,
                    std::complex<double> *evecs) {
  if (n <= 0) return;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  DeviceScope dev;
  cuDoubleComplex *dA = dev.alloc<cuDoubleComplex>(nn);
  double *dW = dev.alloc<double>(static_cast<std::size_t>(n));
  int *dInfo = dev.alloc<int>(1);
  // `a` is column-major Hermitian; std::complex<double> and cuDoubleComplex are
  // layout-compatible (two contiguous doubles), so a reinterpret copy is exact.
  cudaCheck(cudaMemcpy(dA, a, nn * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice),
            "memcpy A");

  int lwork = 0;
  solverCheck(
      cusolverDnZheevd_bufferSize(dev.handle, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, &lwork),
      "Zheevd_bufferSize");
  cuDoubleComplex *dWork = dev.alloc<cuDoubleComplex>(
      static_cast<std::size_t>(lwork));
  solverCheck(cusolverDnZheevd(dev.handle, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, dWork,
                               lwork, dInfo),
              "Zheevd");
  int info = 0;
  cudaCheck(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost),
            "memcpy info");
  if (info != 0)
    throw std::runtime_error("cusolverDnZheevd: eigensolve did not converge "
                             "(info=" + std::to_string(info) + ")");

  cudaCheck(cudaMemcpy(evals, dW, static_cast<std::size_t>(n) * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "memcpy W");
  std::vector<cuDoubleComplex> V(nn);
  cudaCheck(cudaMemcpy(V.data(), dA, nn * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost),
            "memcpy V");
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const cuDoubleComplex z =
          V[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
      evecs[static_cast<std::size_t>(i) * n + j] =
          std::complex<double>(cuCreal(z), cuCimag(z));
    }
}

}  // namespace cuda
}  // namespace tessera
