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

    /// Torsion coefficients of H_k: the invariant factors > 1 of ∂_{k+1}.
    /// (E.g. ℝP² has torsion(1) = {2}.)
    [[nodiscard]] std::vector<long> torsion(int k) const;

  private:
    int dimension_{-1};
    std::vector<std::size_t> counts_{};                 // |C_k|
    std::vector<std::vector<long>> boundary_{};         // boundary_[k] = ∂_k
    [[nodiscard]] int rankOfBoundary(int k) const;      // rank ∂_k over ℚ (0 if out of range)
    [[nodiscard]] int gf2RankOfBoundary(int k) const;   // rank ∂_k over GF(2)
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHAINCOMPLEX_H
