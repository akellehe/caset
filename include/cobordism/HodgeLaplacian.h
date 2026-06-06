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
/// The Hermitian-weighted Hodge Laplacian on a `Spacetime`. **Stage 1**
/// implements the degree-zero case — the U(1)-weighted graph Laplacian
/// \f$ L = D - A \f$ on the 1-skeleton — assembled directly from the complex
/// edge weights \f$ \text{squaredLength}\cdot e^{i\,\text{phase}} \f$ carried by
/// each `Edge`. The API is **degree-parameterized** (`int k`) so Stage 2 can add
/// the cochain Laplacians \f$ L_k = d_{k-1}d_{k-1}^\dagger + d_k^\dagger d_k \f$
/// without changing call sites; any \f$ k \neq 0 \f$ currently throws.
///
/// Vertices are indexed by a stable order: sorted vertex id \f$ \to 0..N-1 \f$,
/// so the returned flat \f$ N\times N \f$ matrices (row-major) are reproducible.
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

    /// Laplacian \f$ L = D - A \f$ for \f$ k = 0 \f$, flat row-major
    /// \f$ N\times N \f$ complex.
    /// @throws std::runtime_error for \f$ k \neq 0 \f$ (Stage 2: \f$ L_k \f$ not
    ///   yet implemented).
    [[nodiscard]] std::vector<std::complex<double>> laplacian(int k = 0) const;

    /// Whether \f$ \| L - L^\dagger \| \le \text{tol} \f$ (Frobenius norm) for
    /// the \f$ k = 0 \f$ Laplacian. True by construction.
    [[nodiscard]] bool isHermitian(double tol = 1e-12) const;

    /// Unitarity residual of the time-evolution operator
    /// \f$ U = e^{-iLt} = V\,\mathrm{diag}(e^{-i\lambda t})\,V^\dagger \f$ formed
    /// from the eigendecomposition: returns \f$ \| U U^\dagger - I \| \f$
    /// (Frobenius). ~0 for the Hermitian \f$ L \f$.
    [[nodiscard]] double unitarityResidual(double t = 1.0) const;

    /// Eigenvalues of \f$ L_k \f$ (real, ascending).
    /// @throws std::runtime_error for \f$ k \neq 0 \f$.
    [[nodiscard]] std::vector<double> eigenvalues(int k = 0) const;

    /// Eigenvectors of \f$ L_k \f$ as a flat row-major \f$ N\times N \f$ array;
    /// column \f$ j \f$ (entries at indices \f$ iN + j \f$) is the eigenvector
    /// for the \f$ j \f$-th ascending eigenvalue.
    /// @throws std::runtime_error for \f$ k \neq 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> eigenvectors(int k = 0) const;

    /// Harmonic representatives: the eigenvectors with \f$ |\lambda| < \text{tol} \f$
    /// (a basis for \f$ \ker L_k \f$), as a flat row-major \f$ N\times M \f$
    /// array whose \f$ M \f$ columns are the harmonics (so \f$ M = \f$
    /// size/\f$ N \f$). \f$ M \f$ is the harmonic dimension.
    /// @throws std::runtime_error for \f$ k \neq 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> harmonics(int k = 0,
                                                              double tol = 1e-9) const;

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

    // Throw unless k == 0 (Stage 1 only implements the graph Laplacian).
    static void requireStageOne(int k);

    // Assemble the adjacency (flat row-major N*N) and degree (length N) from the
    // current edge weights/phases, using the stable vertex order. Kept Eigen-free
    // in its signature so the public header carries no Eigen dependency.
    void assemble(std::vector<std::complex<double>> &A, std::vector<double> &D) const;

    // Build and cache the eigendecomposition of L (k=0) if not already done.
    void ensureDecomposition() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HODGELAPLACIAN_H
