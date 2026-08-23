// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

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

/// Basis of the GF(2) kernel (nullspace) of a 0/1 matrix M (rows x cols, flat
/// row-major; entries read mod 2). Each returned vector x has length `cols` and
/// satisfies M·x ≡ 0 (mod 2); the vectors are linearly independent over GF(2)
/// and there are exactly nullity = cols - gf2Rank(M) of them, so the whole
/// kernel is their GF(2) span. The basis is returned as a list of length-`cols`
/// vectors (one per free column of the reduced row echelon form), the form that
/// reads naturally for a collection of independent vectors; the count is just
/// the size of the returned list. Empty when M has full column rank.
///
/// This is the cocycle space Z¹ = ker(∂₂ᵀ mod 2) of flat ℤ₂ gauge fields in the
/// Dijkgraaf–Witten state sum; pair with gf2Span to enumerate the connections.
[[nodiscard]] std::vector<std::vector<int>> gf2Nullspace(std::vector<int> M,
                                                         int rows, int cols);

/// Basis of the rational kernel of an integer matrix M (rows x cols, flat
/// row-major), returned as INTEGER vectors: exact Gauss-Jordan over Q (the
/// integer sibling of gf2Nullspace; smithNormalForm reports only the rank and
/// invariant factors, not the transforms a kernel basis needs). Each returned
/// vector x has length `cols`, satisfies M·x = 0 exactly over Z, has coprime
/// entries, and there are exactly nullity = cols - rank(M) of them: over Q
/// they span the whole kernel. Deterministic pivoting (first nonzero per
/// column), so a relabeled input yields the correspondingly mapped basis.
/// @throws std::overflow_error when the exact rational elimination would
///   overflow 64-bit intermediates — the topological claim is exact or
///   absent, never rounded.
[[nodiscard]] std::vector<std::vector<long>> integerNullspace(
    const std::vector<long> &M, int rows, int cols);

/// All 2^k GF(2) linear combinations of a `basis` of k length-`cols` vectors,
/// each combination a length-`cols` vector (entries read mod 2). The first
/// element is always the zero vector (empty combination); `cols` is taken
/// explicitly so the trivial connection has the right length even when the
/// basis is empty (k = 0 → a single zero vector). For a cocycle basis from
/// gf2Nullspace this enumerates the 2^nullity flat ℤ₂ connections. Enumeration
/// is exponential in k; intended for the small nullities of tiny complexes, it
/// rejects k large enough to be unmaterializable.
[[nodiscard]] std::vector<std::vector<int>> gf2Span(
    const std::vector<std::vector<int>> &basis, int cols);

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
