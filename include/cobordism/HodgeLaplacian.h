// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TESSERA_COBORDISM_HODGELAPLACIAN_H
#define TESSERA_COBORDISM_HODGELAPLACIAN_H

#include <complex>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # HodgeLaplacian
///
/// The Hodge Laplacian on a `Spacetime`, degree-parameterized by `int k`.
///
/// At \f$ k = 0 \f$ it is the **Hermitian U(1)-weighted graph Laplacian**
/// \f$ L = D - A \f$ on the 1-skeleton, assembled directly from the complex edge
/// weights \f$ \text{squaredLength}\cdot e^{i\,\text{phase}} \f$ carried by each
/// `Edge` (the magnitude convention on the degree keeps \f$ L \f$ Hermitian and
/// \f$ e^{-iLt} \f$ unitary). This path is unchanged.
///
/// At \f$ k \geq 1 \f$ it is the **metric Hodge Laplacian**, assembled from the
/// integer boundary maps \f$ \partial_k,\partial_{k+1} \f$ (`ChainComplex`) and
/// the diagonal inner-product weights \f$ W_k \f$ — the per-\f$ k \f$-simplex
/// Euclidean volumes (`Simplex::volume`), with \f$ W_0 = I \f$. With the metric
/// adjoint \f$ \partial_k^* = W_k^{-1}\partial_k^{\top} W_{k-1} \f$ the Laplacian
/// is \f$ L_k = \partial_k^*\partial_k + \partial_{k+1}\partial_{k+1}^* \f$,
/// returned in its **symmetric** (\f$ W_k \f$-orthonormal) representation
/// \f$ W_k^{1/2} L_k W_k^{-1/2} = B_k^{\top}B_k + B_{k+1}B_{k+1}^{\top} \f$ with
/// \f$ B_k = W_{k-1}^{1/2}\partial_k W_k^{-1/2} \f$ — symmetric positive
/// semidefinite, so `SelfAdjointEigenSolver` applies and the spectrum/kernel are
/// those of \f$ L_k \f$ (a similarity). By the discrete Hodge theorem
/// \f$ \ker L_k \cong H_k \f$, so \f$ \dim\ker L_k = b_k \f$ for **any** positive
/// weights; passing `metric = false` uses unit weights — the combinatorial
/// \f$ \partial_k^{\top}\partial_k + \partial_{k+1}\partial_{k+1}^{\top} \f$ — as
/// a same-kernel cross-check. A negative \f$ k \f$ throws; a \f$ k \f$ beyond the
/// top dimension has no \f$ k \f$-cells and yields empty results.
///
/// For \f$ k = 0 \f$ vertices are indexed by a stable order (sorted vertex id
/// \f$ \to 0..N-1 \f$); for \f$ k \geq 1 \f$ the \f$ k \f$-cells follow the
/// canonical `ChainComplex` column order (sorted vertex-id tuples), so the
/// returned flat row-major matrices are reproducible and align with
/// `boundaryMatrix(k)` and `weights(k)`.
///
/// ## Assembly (k = 0)
///
/// - Adjacency \f$ A_{ij} = \sum_{(i,j)} \text{squaredLength}\cdot e^{i\,\text{phase}} \f$,
///   summed over edges between \f$ i \f$ and \f$ j \f$; the stored source→target
///   orientation carries \f$ +\text{phase} \f$, the reverse carries
///   \f$ -\text{phase} \f$, so \f$ A = A^\dagger \f$ (Hermitian).
/// - Degree \f$ D_{ii} = \sum |\text{squaredLength}| \f$ over incident edges
///   (the magnitude convention — keeps \f$ L \f$ Hermitian and \f$ e^{-iLt} \f$
///   unitary for complex weights).
/// - Laplacian \f$ L = D - A \f$.
///
/// This class is the *operator* only: it does not compute fluxes, cycle bases,
/// or Betti numbers (those are `WilsonLoop` / `ChainComplex`'s job), and it does
/// not gauge-transform the mesh (gauge invariance is exercised by rephasing the
/// edges and rebuilding). The Hermitian eigendecomposition is lazily computed
/// (Eigen `SelfAdjointEigenSolver<MatrixXcd>`) and cached.
class HodgeLaplacian {
  public:
    /// Construct the operator over a triangulation. Edge weights/phases are read
    /// lazily (at the first matrix/spectrum query), so the spacetime must
    /// outlive the operator; the held `shared_ptr` keeps it alive.
    explicit HodgeLaplacian(std::shared_ptr<Spacetime> st);

    /// Weighted adjacency \f$ A \f$ as a flat row-major \f$ N\times N \f$ array
    /// of complex entries. Hermitian by construction.
    [[nodiscard]] std::vector<std::complex<double>> adjacency() const;

    /// Degree vector \f$ (D_{00},\dots,D_{N-1,N-1}) \f$, real, length \f$ N \f$
    /// (magnitude convention \f$ D_{ii} = \sum |\text{squaredLength}| \f$).
    [[nodiscard]] std::vector<double> degree() const;

    /// Laplacian \f$ L_k \f$ as a flat row-major matrix of complex entries:
    /// \f$ N\times N \f$ for \f$ k = 0 \f$ (\f$ L = D - A \f$), else
    /// \f$ |C_k|\times|C_k| \f$ (the symmetric metric Laplacian above; imaginary
    /// parts are zero). For \f$ k \geq 1 \f$, `metric = false` selects unit
    /// weights (the combinatorial Laplacian); `metric` is ignored at \f$ k = 0 \f$.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty for \f$ k \f$ above the
    ///   top dimension.
    [[nodiscard]] std::vector<std::complex<double>> laplacian(int k = 0,
                                                             bool metric = true) const;

    /// Diagonal inner-product weights \f$ W_k \f$ (length \f$ |C_k| \f$) in the
    /// canonical `ChainComplex` column order: the per-\f$ k \f$-simplex Euclidean
    /// volume (`Simplex::volume`, magnitude; degenerate cells fall back to 1 to
    /// keep \f$ W_k \f$ positive). \f$ W_0 = I \f$ (all ones). Empty for \f$ k < 0 \f$
    /// or \f$ k \f$ above the top dimension.
    [[nodiscard]] std::vector<double> weights(int k) const;

    /// Whether \f$ \| L - L^\dagger \| \le \text{tol} \f$ (Frobenius norm) for
    /// the \f$ k = 0 \f$ Laplacian. True by construction.
    [[nodiscard]] bool isHermitian(double tol = 1e-12) const;

    /// Unitarity residual of the time-evolution operator
    /// \f$ U = e^{-iLt} = V\,\mathrm{diag}(e^{-i\lambda t})\,V^\dagger \f$ formed
    /// from the eigendecomposition: returns \f$ \| U U^\dagger - I \| \f$
    /// (Frobenius). ~0 for the Hermitian \f$ L \f$.
    [[nodiscard]] double unitarityResidual(double t = 1.0) const;

    /// Eigenvalues of \f$ L_k \f$ (real, ascending). For \f$ k \geq 1 \f$,
    /// `metric` selects volume vs. unit weights (ignored at \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<double> eigenvalues(int k = 0, bool metric = true) const;

    /// Eigenvectors of \f$ L_k \f$ as a flat row-major \f$ M\times M \f$ array
    /// (\f$ M = N \f$ for \f$ k = 0 \f$, else \f$ |C_k| \f$); column \f$ j \f$
    /// (entries at indices \f$ iM + j \f$) is the eigenvector for the
    /// \f$ j \f$-th ascending eigenvalue. For \f$ k \geq 1 \f$, `metric` selects
    /// volume vs. unit weights (ignored at \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvectors(int k = 0,
                                                               bool metric = true) const;

    /// Harmonic representatives: the eigenvectors with \f$ |\lambda| < \text{tol} \f$
    /// (a basis for \f$ \ker L_k \cong H_k \f$), as a flat row-major
    /// \f$ M\times H \f$ array whose \f$ H \f$ columns are the harmonics (so
    /// \f$ H = \f$ size/\f$ M \f$, the harmonic dimension \f$ = b_k \f$). For
    /// \f$ k \geq 1 \f$, `metric` selects volume vs. unit weights (ignored at
    /// \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> harmonics(int k = 0,
                                                              double tol = 1e-9,
                                                              bool metric = true) const;

  private:
    std::shared_ptr<Spacetime> st_;

    // Stable vertex order: ids_[idx] = vertex id, idToIndex_[id] = idx. Built
    // once in the constructor (the vertex set is fixed for the operator's life;
    // only the edge weights/phases are read lazily).
    std::vector<std::uint64_t> ids_{};
    std::unordered_map<std::uint64_t, std::size_t> idToIndex_{};
    std::size_t order_{0};  // N = |V|

    // Lazy, cached Hermitian eigendecomposition of the k=0 Laplacian.
    mutable bool decomposed_{false};
    mutable std::vector<double> evals_{};               // ascending, length N
    mutable std::vector<std::complex<double>> evecs_{};  // flat N*N, columns

    // Real symmetric eigendecomposition of the k>=1 metric Laplacian L_k^sym,
    // cached per (k, metric). evecs is flat |C_k|*|C_k| (columns = ascending
    // eigenvectors), stored complex (imag 0) to share the public return type.
    struct MetricSpectrum {
      int dim{0};
      std::vector<double> evals{};                       // ascending, length |C_k|
      std::vector<std::complex<double>> evecs{};         // flat |C_k|*|C_k|, columns
    };
    mutable std::unordered_map<long long, MetricSpectrum> metricCache_{};

    // Throw for k < 0 (no negative-degree chains).
    static void requireNonNegativeDegree(int k);

    // Build/fetch the cached symmetric spectrum of L_k^sym (k >= 1). Key folds in
    // `metric` so the metric and combinatorial spectra are cached separately.
    const MetricSpectrum &ensureMetricSpectrum(int k, bool metric) const;

    // Assemble the adjacency (flat row-major N*N) and degree (length N) from the
    // current edge weights/phases, using the stable vertex order. Kept Eigen-free
    // in its signature so the public header carries no Eigen dependency.
    void assemble(std::vector<std::complex<double>> &A, std::vector<double> &D) const;

    // Build and cache the eigendecomposition of L (k=0) if not already done.
    void ensureDecomposition() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HODGELAPLACIAN_H
