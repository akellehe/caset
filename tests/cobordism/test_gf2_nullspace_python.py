# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GF(2) nullspace and span (#106): cocycles Z1 = ker(d2^T mod 2) and the flat
Z2 connections enumerated from them, validated against an independent numpy
GF(2) oracle on random binary matrices plus a handful of hand fixtures.
"""

import itertools
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


def _flat(M):
    """Flatten a 2D array-like to a flat row-major list of python ints."""
    return [int(v) for v in np.asarray(M, dtype=np.int64).reshape(-1)]


def _gf2_rank_np(M):
    """Independent GF(2) rank oracle (Gaussian elimination, pure numpy)."""
    A = (np.asarray(M, dtype=np.int64) & 1).copy()
    if A.ndim == 1:
        A = A.reshape(1, -1) if A.size else A.reshape(0, 0)
    rows, cols = A.shape
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        piv = next((i for i in range(rank, rows) if A[i, col] & 1), None)
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        for i in range(rows):
            if i != rank and (A[i, col] & 1):
                A[i] ^= A[rank]
        rank += 1
    return rank


def _assert_kernel_basis(test, A, rows, cols):
    """Run every nullspace acceptance property for matrix A (rows x cols)."""
    A = np.asarray(A, dtype=np.int64).reshape(rows, cols) if rows else \
        np.zeros((0, cols), dtype=np.int64)
    basis = cob.gf2_nullspace(_flat(A) if rows else [], rows, cols)
    basis = [list(v) for v in basis]
    nullity = len(basis)

    # rank-nullity over GF(2): gf2_rank + nullity == cols.
    rank = cob.gf2_rank(_flat(A) if rows else [], rows, cols)
    test.assertEqual(rank, _gf2_rank_np(A))
    test.assertEqual(rank + nullity, cols)

    for x in basis:
        test.assertEqual(len(x), cols)
        test.assertTrue(set(x) <= {0, 1})
        # A . x == 0 (mod 2).
        if rows:
            prod = (A @ np.asarray(x, dtype=np.int64)) % 2
            test.assertTrue(np.all(prod == 0), f"A.x != 0 for x={x}")

    # The basis vectors are linearly independent over GF(2).
    test.assertEqual(_gf2_rank_np(basis) if nullity else 0, nullity)
    return basis


class TestGf2NullspaceFixtures(unittest.TestCase):

    def test_full_rank_has_empty_kernel(self):
        # 2x2 identity: trivial kernel.
        self.assertEqual(cob.gf2_nullspace([1, 0, 0, 1], 2, 2), [])

    def test_single_relation(self):
        # [1 1] x = 0  ->  kernel spanned by (1, 1).
        basis = cob.gf2_nullspace([1, 1], 1, 2)
        self.assertEqual([list(v) for v in basis], [[1, 1]])

    def test_two_relations_three_cols(self):
        # [[1,1,0],[0,1,1]] x = 0  ->  kernel spanned by (1, 1, 1).
        basis = cob.gf2_nullspace([1, 1, 0, 0, 1, 1], 2, 3)
        self.assertEqual([list(v) for v in basis], [[1, 1, 1]])

    def test_zero_matrix_is_whole_space(self):
        # 2x2 zero matrix: nullity 2, kernel == all of GF(2)^2.
        basis = [list(v) for v in cob.gf2_nullspace([0, 0, 0, 0], 2, 2)]
        self.assertEqual(len(basis), 2)
        self.assertEqual(_gf2_rank_np(basis), 2)

    def test_rank_one_all_ones(self):
        # 3x3 all-ones: rank 1, nullity 2; every kernel vector has even weight.
        basis = _assert_kernel_basis(self, np.ones((3, 3), dtype=int), 3, 3)
        self.assertEqual(len(basis), 2)
        for x in basis:
            self.assertEqual(sum(x) % 2, 0)

    def test_entries_reduced_mod_two(self):
        # Odd entries read as 1: [3, 3] behaves like [1, 1].
        self.assertEqual([list(v) for v in cob.gf2_nullspace([3, 3], 1, 2)],
                         [[1, 1]])


class TestGf2NullspaceOracle(unittest.TestCase):

    def test_random_binary_matrices(self):
        rng = np.random.default_rng(20251106)
        shapes = [(1, 1), (2, 3), (3, 2), (4, 4), (5, 8), (8, 5),
                  (6, 6), (7, 10), (10, 7), (3, 12)]
        for rows, cols in shapes:
            for trial in range(8):
                with self.subTest(rows=rows, cols=cols, trial=trial):
                    A = rng.integers(0, 2, size=(rows, cols))
                    _assert_kernel_basis(self, A, rows, cols)

    def test_random_low_rank_matrices(self):
        # Force a large kernel by building rank-deficient matrices (outer-product
        # sums of r < cols binary vectors), stressing the free-column logic.
        rng = np.random.default_rng(424242)
        for cols in (6, 9, 12):
            for r in range(0, 4):
                with self.subTest(cols=cols, r=r):
                    rows = cols
                    A = np.zeros((rows, cols), dtype=np.int64)
                    for _ in range(r):
                        u = rng.integers(0, 2, size=(rows, 1))
                        v = rng.integers(0, 2, size=(1, cols))
                        A = (A + u @ v) % 2
                    _assert_kernel_basis(self, A, rows, cols)

    def test_empty_matrix_zero_rows(self):
        # No equations: the kernel is the whole space (nullity == cols).
        basis = [list(v) for v in cob.gf2_nullspace([], 0, 4)]
        self.assertEqual(len(basis), 4)
        self.assertEqual(_gf2_rank_np(basis), 4)


class TestGf2Span(unittest.TestCase):

    def _expected_span(self, basis, cols):
        """All GF(2) combinations of `basis`, computed independently in numpy."""
        out = set()
        k = len(basis)
        for mask in range(1 << k):
            v = np.zeros(cols, dtype=np.int64)
            for b in range(k):
                if mask & (1 << b):
                    v ^= np.asarray(basis[b], dtype=np.int64) & 1
            out.add(tuple(int(x) for x in v))
        return out

    def test_empty_basis_yields_single_zero_vector(self):
        # k == 0: the only combination is the zero vector, of length cols.
        self.assertEqual([list(v) for v in cob.gf2_span([], 3)], [[0, 0, 0]])

    def test_span_matches_numpy_oracle(self):
        rng = np.random.default_rng(98765)
        for rows, cols in [(2, 4), (3, 6), (1, 5), (4, 7), (5, 5)]:
            A = rng.integers(0, 2, size=(rows, cols))
            basis = [list(v) for v in cob.gf2_nullspace(_flat(A), rows, cols)]
            span = [list(v) for v in cob.gf2_span(basis, cols)]
            with self.subTest(rows=rows, cols=cols):
                # 2^nullity combinations, all distinct, first is the zero vector.
                self.assertEqual(len(span), 2 ** len(basis))
                self.assertEqual(span[0], [0] * cols)
                self.assertEqual(len(set(map(tuple, span))), len(span))
                # Set equality with the independent oracle.
                self.assertEqual(set(map(tuple, span)),
                                 self._expected_span(basis, cols))
                # Every connection in the span is itself a cocycle (A.v == 0).
                for v in span:
                    prod = (A @ np.asarray(v, dtype=np.int64)) % 2
                    self.assertTrue(np.all(prod == 0))

    def test_span_rejects_unmaterializable_basis(self):
        # 1x25 zero matrix -> nullity 25; enumerating 2^25 is refused.
        basis = cob.gf2_nullspace([0] * 25, 1, 25)
        self.assertEqual(len(basis), 25)
        with self.assertRaises(ValueError):
            cob.gf2_span(basis, 25)


if __name__ == "__main__":
    unittest.main()
