// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

namespace tessera {
namespace cuda {

/// FP32 cuBLAS (SGEMM) accelerator for the per-edge analytic r_U-gradient loop
/// of `EigenstateSynthesis::residualForPeriodsGradient` (#348) — the level-2
/// relaxation bottleneck (~82 s/call at n1=2724 edges on 16-core CPU FP64).
///
/// The loop-invariant matrices (the non-null / null eigenvector blocks of the
/// metric Laplacian M = L1, M itself, and the unit carried representative p) are
/// uploaded once as single precision; each edge then streams only its low-rank
/// dM/dl² factors `fa`, `fb`, and the heavy GEMMs run on the GPU in FP32. The
/// dense eigensolve and the cheap small-dimension per-edge algebra stay on the
/// CPU in FP64; FP32 in these GEMMs is the ONLY approximation (pre-approved:
/// ~1e-5 relative vs FP64 at level-2, identical descent direction). The FP64
/// CPU path remains the default and the correctness oracle.
///
/// All matrices are COLUMN-MAJOR (matching Eigen's default storage and cuBLAS),
/// so the caller passes `Eigen::MatrixXf::data()` directly. Complex N-vectors
/// are packed as an N×2 column-major block `[Re | Im]`.
class RuGradientGpu {
 public:
  /// Upload the loop-invariant matrices (column-major, FP32):
  ///   `Unn`  (N×nnd) — non-null eigenvectors of M;
  ///   `UnnS` (N×nnd) — `Unn` with column r scaled by invlam[r] = −1/λ_nn[r]
  ///                    (so dUn = UnnS·core hoists the diagonal, by associativity);
  ///   `Un`   (N×nd)  — null/harmonic eigenvectors of M;
  ///   `M`    (N×N)   — the real metric Laplacian L1;
  ///   `p2`   (N×2)   — `[Re p | Im p]`, the unit carried representative.
  /// `rmax` is the largest per-edge rank (columns of `fa`/`fb`) — sizes scratch.
  RuGradientGpu(int N, int nnd, int nd, int rmax,
                const float* Unn, const float* UnnS, const float* Un,
                const float* M, const float* p2);
  ~RuGradientGpu();
  RuGradientGpu(const RuGradientGpu&) = delete;
  RuGradientGpu& operator=(const RuGradientGpu&) = delete;

  /// Per-edge stage 1 (the dominant GEMMs). Given `fa`, `fb` (both N×r,
  /// column-major FP32), compute on the GPU and download:
  ///   `dUn`  (N×nd)  = UnnS · ((Unnᵀ·fa)·(fbᵀ·Un));
  ///   `dMp2` (N×2)   = fa · (fbᵀ · p2)            — the complex dM·p, packed [Re|Im].
  void edgeStage1(const float* fa, const float* fb, int r,
                  float* dUn, float* dMp2);

  /// Per-edge stage 2: `Mdpsi2` (N×2) = M · `dpsi2` (N×2), the dense GEMV against
  /// the post-leak perturbed cochain (packed [Re|Im]).
  void edgeStage2(const float* dpsi2, float* Mdpsi2);

 private:
  int N_, nnd_, nd_, rmax_;
  void* handle_;  // cublasHandle_t, opaque so the header stays cuBLAS-free
  // Loop-invariant device constants.
  float *dUnn_, *dUnnS_, *dUn_, *dM_, *dP2_;
  // Per-edge device scratch (sized by N, nnd, nd, rmax in the ctor).
  float *dFa_, *dFb_, *dL_, *dR_, *dCore_, *dDUn_, *dT2_, *dDMp2_;
  float *dDpsi2_, *dMdpsi2_;
};

}  // namespace cuda
}  // namespace tessera
