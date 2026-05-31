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

#ifndef TESSERA_COBORDISM_CHAINCOMPLEX_H
#define TESSERA_COBORDISM_CHAINCOMPLEX_H

#include <cstdint>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # ChainComplex
///
/// The simplicial chain complex of a triangulation: the boundary maps
/// ∂_k : C_k → C_{k-1} over ℤ, plus the homology invariants derived from them
/// (Betti numbers over ℤ and ℤ/2, torsion coefficients) and the ∂²=0 sanity
/// check. Purely combinatorial — built from the vertex sets of the complex,
/// independent of any geometry.
///
/// Simplices are enumerated as the full face-closure of the complex's simplices
/// (every sub-face of every registered simplex). Each k-simplex is identified
/// by its sorted vertex-id tuple and assigned an index; its reference
/// orientation is the increasing-vertex-id ordering, so
/// ∂[v_0 < … < v_k] = Σ_i (−1)^i [v_0,…,v̂_i,…,v_k].
class ChainComplex {
  public:
    /// Build the chain complex from a triangulation (a Spacetime). Reads vertex
    /// sets only; no coordinates/geometry required.
    [[nodiscard]] static ChainComplex fromSpacetime(const Spacetime &K);

    /// Top dimension n (largest k with a k-simplex), or -1 if empty.
    [[nodiscard]] int dimension() const noexcept { return dimension_; }

    /// |C_k|, the number of k-simplices (0 if k out of range).
    [[nodiscard]] std::size_t numSimplices(int k) const noexcept;

    /// f-vector (|C_0|, …, |C_n|).
    [[nodiscard]] const std::vector<std::size_t> &fVector() const noexcept { return counts_; }

    /// Euler characteristic χ = Σ_k (−1)^k |C_k|.
    [[nodiscard]] int eulerCharacteristic() const noexcept;

    /// The boundary matrix ∂_k (rows = |C_{k-1}|, cols = |C_k|), flat row-major.
    /// Entries in {−1, 0, +1}. ∂_0 is empty. Out-of-range k returns an empty matrix.
    [[nodiscard]] const std::vector<long> &boundaryMatrix(int k) const;

    /// Check ∂_{k-1} ∘ ∂_k = 0 for all k (chain-complex axiom / V3 sanity check).
    [[nodiscard]] bool boundaryComposesToZero() const;

    /// Betti numbers b_0..b_n over ℚ (free ranks of H_k):
    /// b_k = |C_k| − rank ∂_k − rank ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbers() const;

    /// Betti numbers over GF(2): b_k = |C_k| − rank₂ ∂_k − rank₂ ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbersGF2() const;

    /// Torsion coefficients of \f$ H_k \f$: the invariant factors \f$ > 1 \f$
    /// of \f$ \partial_{k+1} \f$. (E.g. \f$ \mathbb{RP}^2 \f$ has
    /// \f$ \mathrm{torsion}(1) = \{2\} \f$.)
    [[nodiscard]] std::vector<long> torsion(int k) const;

    /// The symmetric intersection form \f$ Q_{ij} = \langle \alpha_i \cup
    /// \alpha_j, [K] \rangle \f$ on a basis \f$ \{\alpha_i\} \f$ of the free
    /// part of \f$ H^2 \f$, as a flat row-major \f$ b_2 \times b_2 \f$ matrix.
    /// Defined for a closed oriented 4-manifold (\f$ n = 4 \f$); the cup product
    /// is the Alexander–Whitney product evaluated on the fundamental class
    /// \f$ [K] \f$ (the generator of \f$ \ker \partial_4 \f$). Empty when
    /// \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    /// @throws std::runtime_error if \f$ n = 4 \f$, \f$ b_2 > 0 \f$, but the
    ///   complex is not closed-orientable (no fundamental class).
    [[nodiscard]] std::vector<double> intersectionForm() const;

    /// Signature \f$ \sigma = b_+ - b_- \f$ of the intersection form
    /// (Sylvester inertia). 0 when \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    [[nodiscard]] int signature() const;

  private:
    int dimension_{-1};
    std::vector<std::size_t> counts_{};                 // |C_k|
    std::vector<std::vector<long>> boundary_{};         // boundary_[k] = ∂_k
    // faceVerts_[k][j] = sorted vertex ids of the j-th k-simplex (column j of
    // ∂_{k+1} / row j of ∂_k). Needed by the cup product (front/back faces).
    std::vector<std::vector<std::vector<std::uint64_t>>> faceVerts_{};
    [[nodiscard]] int rankOfBoundary(int k) const;      // rank ∂_k over ℚ (0 if out of range)
    [[nodiscard]] int gf2RankOfBoundary(int k) const;   // rank ∂_k over GF(2)
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHAINCOMPLEX_H
