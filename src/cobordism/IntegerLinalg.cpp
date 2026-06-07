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

#include "cobordism/IntegerLinalg.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tessera::cobordism {

namespace {

// Index helper for a flat row-major (rows x cols) matrix.
inline long &at(std::vector<long> &M, int cols, int i, int j) {
  return M[static_cast<std::size_t>(i) * cols + j];
}

}  // namespace

SmithNormalForm smithNormalForm(std::vector<long> M, int rows, int cols) {
  SmithNormalForm out;
  const int dim = std::min(rows, cols);
  int t = 0;  // current pivot index along the diagonal

  for (; t < dim; ++t) {
    // Reduce the trailing submatrix [t.., t..] until row t and column t are
    // cleared off the pivot and the pivot divides the whole submatrix.
    for (;;) {
      // Find the nonzero entry of smallest absolute value in the submatrix.
      int pi = -1, pj = -1;
      long best = 0;
      for (int i = t; i < rows; ++i) {
        for (int j = t; j < cols; ++j) {
          long v = at(M, cols, i, j);
          if (v != 0 && (pi < 0 || std::llabs(v) < std::llabs(best))) {
            best = v;
            pi = i;
            pj = j;
          }
        }
      }
      if (pi < 0) goto done;  // submatrix is all zero — no more pivots

      // Move the pivot to (t, t) with row/column swaps.
      if (pi != t)
        for (int j = 0; j < cols; ++j) std::swap(at(M, cols, t, j), at(M, cols, pi, j));
      if (pj != t)
        for (int i = 0; i < rows; ++i) std::swap(at(M, cols, i, t), at(M, cols, i, pj));

      long piv = at(M, cols, t, t);

      // Clear the rest of column t using row t.
      bool changed = false;
      for (int i = t + 1; i < rows; ++i) {
        long a = at(M, cols, i, t);
        if (a == 0) continue;
        long q = a / piv;
        for (int j = t; j < cols; ++j) at(M, cols, i, j) -= q * at(M, cols, t, j);
        if (at(M, cols, i, t) != 0) changed = true;  // remainder → re-pivot
      }
      // Clear the rest of row t using column t.
      for (int j = t + 1; j < cols; ++j) {
        long a = at(M, cols, t, j);
        if (a == 0) continue;
        long q = a / piv;
        for (int i = t; i < rows; ++i) at(M, cols, i, j) -= q * at(M, cols, i, t);
        if (at(M, cols, t, j) != 0) changed = true;
      }
      if (changed) continue;  // a remainder survived; re-find a smaller pivot

      // Row t and column t are clear off the pivot. Enforce the divisibility
      // d_t | (every submatrix entry): if some entry isn't divisible, fold its
      // row into row t and re-pivot.
      bool divisible = true;
      for (int i = t + 1; i < rows && divisible; ++i)
        for (int j = t + 1; j < cols; ++j)
          if (at(M, cols, i, j) % piv != 0) {
            for (int k = t; k < cols; ++k) at(M, cols, t, k) += at(M, cols, i, k);
            divisible = false;
            break;
          }
      if (divisible) break;  // pivot t finalized
    }
  }

done:
  out.rank = t;
  out.invariantFactors.reserve(static_cast<std::size_t>(t));
  for (int i = 0; i < t; ++i) out.invariantFactors.push_back(std::llabs(at(M, cols, i, i)));
  return out;
}

int integerRank(const std::vector<long> &M, int rows, int cols) {
  return smithNormalForm(M, rows, cols).rank;
}

int gf2Rank(std::vector<int> M, int rows, int cols) {
  auto idx = [cols](int i, int j) { return static_cast<std::size_t>(i) * cols + j; };
  for (auto &v : M) v &= 1;
  int rank = 0;
  for (int col = 0; col < cols && rank < rows; ++col) {
    // Find a pivot row with a 1 in this column, at or below `rank`.
    int piv = -1;
    for (int i = rank; i < rows; ++i)
      if (M[idx(i, col)] & 1) { piv = i; break; }
    if (piv < 0) continue;
    if (piv != rank)
      for (int j = 0; j < cols; ++j) std::swap(M[idx(rank, j)], M[idx(piv, j)]);
    // Eliminate this column from every other row.
    for (int i = 0; i < rows; ++i) {
      if (i != rank && (M[idx(i, col)] & 1))
        for (int j = col; j < cols; ++j) M[idx(i, j)] ^= M[idx(rank, j)];
    }
    ++rank;
  }
  return rank;
}

std::vector<std::vector<int>> gf2Nullspace(std::vector<int> M, int rows, int cols) {
  auto idx = [cols](int i, int j) { return static_cast<std::size_t>(i) * cols + j; };
  for (auto &v : M) v &= 1;
  // Reduce to RREF (mirroring gf2Rank), recording each pivot's column.
  std::vector<int> pivotCol;  // pivotCol[r] == pivot column of reduced row r
  int rank = 0;
  for (int col = 0; col < cols && rank < rows; ++col) {
    int piv = -1;
    for (int i = rank; i < rows; ++i)
      if (M[idx(i, col)] & 1) { piv = i; break; }
    if (piv < 0) continue;
    if (piv != rank)
      for (int j = 0; j < cols; ++j) std::swap(M[idx(rank, j)], M[idx(piv, j)]);
    // Eliminate this column from every other row (above and below).
    for (int i = 0; i < rows; ++i) {
      if (i != rank && (M[idx(i, col)] & 1))
        for (int j = col; j < cols; ++j) M[idx(i, j)] ^= M[idx(rank, j)];
    }
    pivotCol.push_back(col);
    ++rank;
  }

  // Free columns are the non-pivot columns; one kernel vector per free column.
  std::vector<char> isPivot(static_cast<std::size_t>(cols), 0);
  for (int c : pivotCol) isPivot[static_cast<std::size_t>(c)] = 1;

  std::vector<std::vector<int>> basis;
  basis.reserve(static_cast<std::size_t>(cols - rank));
  for (int f = 0; f < cols; ++f) {
    if (isPivot[static_cast<std::size_t>(f)]) continue;
    // Set the free variable to 1; back-substitute each pivot variable to the
    // (already reduced) coefficient tying it to this free column.
    std::vector<int> x(static_cast<std::size_t>(cols), 0);
    x[static_cast<std::size_t>(f)] = 1;
    for (int r = 0; r < rank; ++r)
      x[static_cast<std::size_t>(pivotCol[r])] = M[idx(r, f)] & 1;
    basis.push_back(std::move(x));
  }
  return basis;
}

std::vector<std::vector<int>> gf2Span(const std::vector<std::vector<int>> &basis,
                                      int cols) {
  const int k = static_cast<int>(basis.size());
  // 2^k vectors are materialized; refuse a count that cannot fit in memory (and
  // would also overflow the shift below).
  if (k > 24)
    throw std::invalid_argument(
        "gf2Span: basis too large to enumerate (2^" + std::to_string(k) +
        " combinations)");
  const std::size_t count = std::size_t{1} << k;
  std::vector<std::vector<int>> out;
  out.reserve(count);
  for (std::size_t mask = 0; mask < count; ++mask) {
    std::vector<int> v(static_cast<std::size_t>(cols), 0);
    for (int b = 0; b < k; ++b)
      if (mask & (std::size_t{1} << b))
        for (int j = 0; j < cols; ++j) v[static_cast<std::size_t>(j)] ^=
            basis[static_cast<std::size_t>(b)][static_cast<std::size_t>(j)] & 1;
    out.push_back(std::move(v));
  }
  return out;
}

Inertia symmetricInertia(std::vector<long> Q, int n, double tol) {
  Inertia out;
  if (n == 0) return out;
  Eigen::MatrixXd A(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      A(i, j) = static_cast<double>(Q[static_cast<std::size_t>(i) * n + j]);
  // Symmetrize defensively against caller asymmetry / rounding.
  A = 0.5 * (A + A.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A, Eigen::EigenvaluesOnly);
  for (int i = 0; i < n; ++i) {
    double lam = es.eigenvalues()[i];
    if (lam > tol) ++out.nPos;
    else if (lam < -tol) ++out.nNeg;
    else ++out.nZero;
  }
  return out;
}

}  // namespace tessera::cobordism
