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

#ifndef TESSERA_COBORDISM_INTEGERLINALG_H
#define TESSERA_COBORDISM_INTEGERLINALG_H

#include <cstdint>
#include <vector>

// Linear algebra for simplicial homology. The exact-integer (Smith Normal Form
// over ℤ) and GF(2) routines are hand-rolled — no numeric LA library (Eigen,
// LAPACK, ITensor) provides exact-integer or mod-2 arithmetic. The signature
// (Sylvester inertia) instead reuses Eigen's symmetric eigensolver, since the
// intersection form is a small real-symmetric matrix. Matrices are flat
// row-major; the matrices involved are tiny, so dense routines are fine.
namespace tessera::cobordism {

/// Smith Normal Form data for an integer matrix: its rank and the positive
/// invariant factors d_1 | d_2 | ... | d_rank (each > 0, each dividing the next).
struct SmithNormalForm {
  int rank{0};
  std::vector<long> invariantFactors{};
};

/// Smith Normal Form of an integer matrix M (rows x cols, flat row-major).
/// Returns the rank and invariant factors. M is taken by value (mutated
/// internally).
[[nodiscard]] SmithNormalForm smithNormalForm(std::vector<long> M, int rows, int cols);

/// Rank over Q of an integer matrix (== number of SNF pivots).
[[nodiscard]] int integerRank(const std::vector<long> &M, int rows, int cols);

/// Rank over GF(2) of a 0/1 matrix (flat row-major). Entries are read mod 2.
[[nodiscard]] int gf2Rank(std::vector<int> M, int rows, int cols);

/// Inertia of a symmetric integer matrix Q (n x n): the counts of positive,
/// negative, and zero eigenvalues by Sylvester's law of inertia. The signature
/// is nPos - nNeg.
struct Inertia {
  int nPos{0};
  int nNeg{0};
  int nZero{0};
  [[nodiscard]] int signature() const noexcept { return nPos - nNeg; }
};

/// Compute the inertia of a symmetric integer matrix Q (n x n, flat row-major)
/// via Eigen's self-adjoint eigensolver, counting eigenvalue signs. Q must be
/// symmetric; behaviour is undefined otherwise. Eigenvalues within `tol` of
/// zero count as zero — robust for the nondegenerate (unimodular) intersection
/// forms this is used on; a degenerate form near the tolerance is the only
/// place the sign could be ambiguous.
[[nodiscard]] Inertia symmetricInertia(std::vector<long> Q, int n,
                                       double tol = 1e-9);

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_INTEGERLINALG_H
