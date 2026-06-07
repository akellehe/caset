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

"""Signature-aware Simplex geometry: volume(), gramMatrix(), and the extracted
cayleyMengerMatrix().

The geometry is honest (signature-respecting) by default: a timelike edge keeps
its negative squared length, so a Lorentzian cell reports a signed content. The
Wick-rotated (|l^2|) behaviour the CDT/Regge path relies on is still available
via ``wickRotate=True`` and is exercised here too.
"""

import math
import unittest

import numpy as np

from tessera import Spacetime


def _make_simplex(st, vertex_ids, squared_by_pair):
    """Build a simplex on ``st`` with full control of vertex order and of every
    edge's signed squared length.

    ``vertex_ids`` fixes the vertex order (so vertex 0 -- the Gram origin -- is
    ``vertex_ids[0]``); ``squared_by_pair`` maps ``frozenset({i, j})`` to the
    signed l^2 of that edge. Returns ``(simplex, verts, edges)``.
    """
    verts = {i: st.createVertex(i) for i in vertex_ids}
    edges = {}
    for pair, l2 in squared_by_pair.items():
        a, b = tuple(pair)
        edges[pair] = st.createEdge(verts[a], verts[b], float(l2))
    simplex, _ = st.createSimplex([verts[i] for i in vertex_ids],
                                  list(edges.values()))
    return simplex, verts, edges


def _expected_gram(simplex, squared_by_pair, wick):
    """Independent reconstruction of the (d x d) Gram matrix from the signed (or
    |l^2| when ``wick``) squared lengths, in the simplex's own vertex order."""
    vids = [v.getId() for v in simplex.getVertices()]
    d = len(vids) - 1

    def s(i, j):
        if i == j:
            return 0.0
        l2 = squared_by_pair[frozenset({vids[i], vids[j]})]
        return abs(l2) if wick else float(l2)

    G = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            G[i, j] = 0.5 * (s(0, i + 1) + s(0, j + 1) - s(i + 1, j + 1))
    return G


def _cofactor(M):
    """Cofactor matrix, mirroring Simplex::cofactorMatrix exactly."""
    n = M.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(M, i, axis=0), j, axis=1)
            C[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return C


def _dihedral_from_cm(B, bi, bj):
    """Dihedral angle from a Cayley-Menger matrix, mirroring the C++ cofactor
    formula (including the abs under the sqrt and the cos clamp)."""
    C = _cofactor(B)
    denom = math.sqrt(abs(C[bi, bi] * C[bj, bj]))
    if denom < 1e-15:
        return 0.0
    cos = max(-1.0, min(1.0, -C[bi, bj] / denom))
    return math.acos(cos)


class TestSimplexVolume(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def test_euclidean_triangle_volume_matches_hand_value(self):
        # Right isosceles triangle, legs 1 and 1, hypotenuse^2 = 2. Area = 1/2.
        sq = {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 1.0,
            frozenset({1, 2}): 2.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        self.assertAlmostEqual(simplex.volume(), 0.5, places=9)

    def test_euclidean_tetrahedron_volume_matches_hand_value(self):
        # Corner tetrahedron with three unit legs along orthogonal axes:
        # vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1). Volume = 1/6.
        sq = {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 1.0,
            frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 2.0,
            frozenset({1, 3}): 2.0,
            frozenset({2, 3}): 2.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        self.assertAlmostEqual(simplex.volume(), 1.0 / 6.0, places=9)

    def test_lorentzian_triangle_volume_is_signed(self):
        # One timelike edge (l^2 < 0). In coordinates (t, x):
        # v0=(0,0), v1=(1,0) timelike from v0 -> l^2 = -1,
        # v2=(0,2) spacelike -> l^2 = 4, edge v1-v2 -> l^2 = -1 + 4 = 3.
        # Honest Gram = [[-1, 0], [0, 4]], det = -4, signed content = -1.
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)

        vol = simplex.volume()
        # The honest content is signed (negative), recording the signature.
        self.assertLess(vol, 0.0)
        self.assertAlmostEqual(vol, -1.0, places=9)

        # It must NOT be the |l^2| (Wick-rotated) value, which is positive.
        gram_wick = _expected_gram(simplex, sq, wick=True)
        wick_content = math.sqrt(np.linalg.det(gram_wick)) / 2.0
        self.assertGreater(wick_content, 0.0)
        self.assertFalse(math.isclose(vol, wick_content, abs_tol=1e-6))

    def test_volume_matches_signed_gram_determinant(self):
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        gram = _expected_gram(simplex, sq, wick=False)
        det = np.linalg.det(gram)
        expected = math.copysign(math.sqrt(abs(det)), det) / math.factorial(2)
        self.assertAlmostEqual(simplex.volume(), expected, places=9)


class TestSimplexGramSignatureAware(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def test_gram_honest_keeps_timelike_sign(self):
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        honest = np.array(simplex.gramMatrix()).reshape(2, 2)          # default
        honest_explicit = np.array(
            simplex.gramMatrix(wickRotate=False)).reshape(2, 2)
        np.testing.assert_allclose(honest, honest_explicit)
        np.testing.assert_allclose(honest, _expected_gram(simplex, sq, False))
        # Honest != Wick-rotated for a Lorentzian cell.
        wick = np.array(simplex.gramMatrix(wickRotate=True)).reshape(2, 2)
        np.testing.assert_allclose(wick, _expected_gram(simplex, sq, True))
        self.assertFalse(np.allclose(honest, wick))

    def test_gram_euclidean_toggle_is_noop(self):
        # All-spacelike (l^2 > 0): dropping the abs changes nothing.
        sq = {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 1.0,
            frozenset({1, 2}): 2.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        honest = np.array(simplex.gramMatrix(wickRotate=False)).reshape(2, 2)
        wick = np.array(simplex.gramMatrix(wickRotate=True)).reshape(2, 2)
        np.testing.assert_allclose(honest, wick)


class TestCayleyMengerMatrix(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def _tetra(self):
        # Unit-edge tetra under Wick rotation, but with edge (0,1) timelike so
        # honest and |l^2| geometries differ.
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 1.0,
            frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 1.0,
            frozenset({1, 3}): 1.0,
            frozenset({2, 3}): 1.0,
        }
        simplex, verts, edges = _make_simplex(self.st, [0, 1, 2, 3], sq)
        return simplex, verts, edges, sq

    def _expected_inner(self, simplex, sq, wick):
        vids = [v.getId() for v in simplex.getVertices()]
        n = len(vids)
        inner = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                l2 = sq[frozenset({vids[i], vids[j]})]
                inner[i, j] = abs(l2) if wick else float(l2)
        return inner

    def test_structure_border_and_inner_block(self):
        simplex, _, _, sq = self._tetra()
        n = len(simplex.getVertices()) + 1  # d + 2 == 5 for a tetrahedron
        for wick in (True, False):
            B = np.array(simplex.cayleyMengerMatrix(wickRotate=wick)).reshape(n, n)
            self.assertEqual(B[0, 0], 0.0)
            np.testing.assert_allclose(B[0, 1:], np.ones(n - 1))
            np.testing.assert_allclose(B[1:, 0], np.ones(n - 1))
            np.testing.assert_allclose(B[1:, 1:],
                                       self._expected_inner(simplex, sq, wick))

    def test_wick_and_honest_inner_blocks_differ(self):
        simplex, _, _, _ = self._tetra()
        n = len(simplex.getVertices()) + 1
        wick = np.array(simplex.cayleyMengerMatrix(wickRotate=True)).reshape(n, n)
        honest = np.array(
            simplex.cayleyMengerMatrix(wickRotate=False)).reshape(n, n)
        self.assertFalse(np.allclose(wick, honest))

    def test_reproduces_dihedral_angle(self):
        """The extracted matrix is exactly what dihedralAngle consumes: feeding
        cayleyMengerMatrix through the cofactor formula reproduces
        dihedralAngle(hinge) for both the Wick-rotated and honest geometries."""
        simplex, verts, edges, _ = self._tetra()
        # Hinge = edge (2,3); the two opposite vertices are 0 and 1.
        hinge, _ = self.st.createSimplex([verts[2], verts[3]],
                                         [edges[frozenset({2, 3})]])
        vids = [v.getId() for v in simplex.getVertices()]
        n = len(vids) + 1
        bi = vids.index(0) + 1
        bj = vids.index(1) + 1

        for wick in (True, False):
            B = np.array(
                simplex.cayleyMengerMatrix(wickRotate=wick)).reshape(n, n)
            reconstructed = _dihedral_from_cm(B, bi, bj)
            actual = simplex.dihedralAngle(hinge, wickRotate=wick)
            self.assertAlmostEqual(actual, reconstructed, places=7)


if __name__ == "__main__":
    unittest.main()
