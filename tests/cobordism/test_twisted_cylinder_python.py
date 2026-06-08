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

"""Non-identity DW boundary maps via finite-order twisted cylinders (#192).

The realizable Dijkgraaf–Witten 2-boundary map image had collapsed to a single
point, ``{I₄}``: the only buildable ``T²→T²`` cobordism was the trivial product
cylinder (``→`` identity). ``Cobordism.twistedCylinder(Σ, φ)`` breaks that
degeneracy by identifying the two boundary copies of the surface ``Σ`` through a
finite-order **simplicial automorphism** ``φ`` instead of the order-preserving
identity. The interior monodromy from the bottom boundary to the top boundary is
then ``φ``, so ``DijkgraafWitten.map()`` returns the permutation ``φ`` induces on
the holonomy classes ``Z(Σ) = ℂ[H¹(Σ; ℤ₂)]``.

On the symmetric square-product torus ``T² = S¹×S¹`` (``SimplicialProduct(∂Δ²,
∂Δ²)``, 9 vertices) the coordinate-swap ``φ: (x,y) ↦ (y,x)`` (``φ mod 2`` is the
modular ``S = [[0,1],[1,0]]``) transposes the holonomy classes ``[a]↔[b]`` while
fixing ``[0]`` and ``[a+b]`` — the non-identity ``4×4`` permutation

    [[1 0 0 0]
     [0 0 1 0]
     [0 1 0 0]
     [0 0 0 1]]

asserted here against an independent numpy GF(2) holonomy oracle. The realizable
image is now ``{I₄} ∪ {this swap}`` — strictly larger than a point.

Functoriality: ``φ`` of order ``n`` ⇒ a DW map of order ``n`` (the swap is an
involution, ``map² = I₄``), and composing twists composes the permutations
(``twistedCylinder(φ∘φ) = map(φ)²``).

The Dehn twist ``T = [[1,1],[0,1]]`` is *not* realizable this way — it is
infinite-order, hence not a finite-order simplicial automorphism of any fixed
triangulation. That is exactly why the map image was stuck at ``{I₄}``: only the
finite-order modular elements (orders 2, 3, 4, 6) are simplicially realizable.
"""

import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle
Cobordism = cobordism.Cobordism


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    # S¹ = ∂Δ², the minimal triangle-boundary circle (3 vertices).
    return tessera.SimplexBoundarySphere(1)


def _torus_topology():
    # T² = S¹ × S¹ via the simplicial (Eilenberg–Zilber) product: 9 vertices,
    # productId(u, v) = 3·u + v, and the coordinate swap (u, v) ↦ (v, u) is a
    # simplicial automorphism (the staircase is swap-symmetric).
    return tessera.SimplicialProduct(_circle(), _circle())


def _torus():
    return _build(_torus_topology())


# The two factors of T² = S¹×S¹ each have 3 vertices, so productId(u, v) = 3u+v.
# The coordinate swap φ(u, v) = (v, u) reads 3u+v ↦ 3v+u; it fixes the diagonal
# (0, 4, 8) and pairs (1,3), (2,6), (5,7).
_SWAP = [v * 3 + u for u in range(3) for v in range(3)]
_IDENTITY = list(range(9))


def _top_triangles(spacetime):
    """Top-cell (triangle) vertex-id tuples of a built surface, sorted."""
    by_size = {}
    for simplex in spacetime.getSimplices():
        cell = tuple(sorted(vertex.getId() for vertex in simplex.getVertices()))
        by_size.setdefault(len(cell), []).append(cell)
    return sorted(by_size[max(by_size)])


# --------------------------------------------------------------------------- #
# Independent numpy GF(2) holonomy oracle for the induced permutation.
#
# Reimplements, in pure numpy, the same per-component class indexing
# DijkgraafWitten.computeBoundary uses — H¹(Σ) = Z¹/B¹ representatives ordered by
# gf2Span(reps) — and reads off the permutation φ induces by pulling each class
# back along φ. Convention-matched to the C++ so the *exact* matrix can be
# asserted, but computed through a fully separate code path.
# --------------------------------------------------------------------------- #
def _gf2_nullspace(matrix, cols):
    if matrix.size == 0:
        return [np.eye(cols, dtype=np.int64)[i] for i in range(cols)]
    a = (np.asarray(matrix, dtype=np.int64) & 1).copy()
    rows, _ = a.shape
    pivots, r = [], 0
    for col in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if a[i, col] & 1), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and (a[i, col] & 1):
                a[i] ^= a[r]
        pivots.append(col)
        r += 1
    is_pivot = [c in pivots for c in range(cols)]
    basis = []
    for free in range(cols):
        if is_pivot[free]:
            continue
        x = np.zeros(cols, dtype=np.int64)
        x[free] = 1
        for t, pc in enumerate(pivots):
            x[pc] = a[t, free] & 1
        basis.append(x)
    return basis


def _gf2_span(basis, cols):
    out = []
    for mask in range(1 << len(basis)):
        x = np.zeros(cols, dtype=np.int64)
        for b in range(len(basis)):
            if (mask >> b) & 1:
                x ^= basis[b]
        out.append(x)
    return out


def _gf2_residue(vector, span_rows):
    v = (np.asarray(vector, dtype=np.int64) & 1).copy()
    for row in span_rows:
        piv = int(np.argmax(row)) if row.any() else -1
        if piv >= 0 and (v[piv] & 1):
            v ^= row
    return v & 1


def _gf2_echelon(generators, cols):
    rows = []
    for gen in generators:
        v = _gf2_residue(gen, rows)
        if v.any():
            rows.append(v)
            rows.sort(key=lambda row: int(np.argmax(row)))
    return rows


def _cohomology_reps(cocycles, coboundaries, cols):
    span_rows = _gf2_echelon(coboundaries, cols)
    reps = []
    for z in cocycles:
        if _gf2_residue(z, span_rows).any():
            reps.append(np.asarray(z, dtype=np.int64) & 1)
            span_rows = _gf2_echelon(
                list(span_rows) + [np.asarray(z, dtype=np.int64) & 1], cols)
    return reps


def _holonomy_permutation(triangles, phi):
    """The permutation φ induces on Z(Σ), in computeBoundary's class order.

    Returns (sigma, b1): sigma[c] is the class index that holonomy class c maps
    to under φ (a connection's class is its gf2Span(reps) index), and b1 is the
    first Betti number of Σ (so 2**b1 = dim Z(Σ)).
    """
    edges = sorted({pair for tri in triangles
                    for pair in ((tri[0], tri[1]), (tri[0], tri[2]),
                                 (tri[1], tri[2]))})
    edge_index = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)
    verts = sorted({v for tri in triangles for v in tri})

    # d₁ (triangles × edges) incidence; Z¹ = ker d₁, B¹ = im d₀ (vertex stars).
    d1 = np.zeros((len(triangles), n_edges), dtype=np.int64)
    for t, tri in enumerate(triangles):
        for pair in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            d1[t, edge_index[pair]] = 1
    cocycles = _gf2_nullspace(d1, n_edges)
    coboundaries = [np.array([1 if v in e else 0 for e in edges], dtype=np.int64)
                    for v in verts]
    reps = _cohomology_reps(cocycles, coboundaries, n_edges)
    classes = _gf2_span(reps, n_edges)
    cob_rows = _gf2_echelon(coboundaries, n_edges)
    class_of_residue = {tuple(_gf2_residue(c, cob_rows)): i
                        for i, c in enumerate(classes)}

    # φ pulls a connection (edge-cochain) back: (φ* g)[e] = g[sorted(φ e)].
    def pullback(g):
        out = np.zeros(n_edges, dtype=np.int64)
        for e, i in edge_index.items():
            out[i] = g[edge_index[tuple(sorted((phi[e[0]], phi[e[1]])))]]
        return out

    sigma = {c: class_of_residue[tuple(_gf2_residue(pullback(cls), cob_rows))]
             for c, cls in enumerate(classes)}
    return sigma, len(reps)


def _oracle_matrix(triangles, phi):
    """The expected DW map: the top boundary class is φ* of the bottom's."""
    sigma, b1 = _holonomy_permutation(triangles, phi)
    dim = 1 << b1
    matrix = np.zeros((dim, dim))
    for bottom, top in sigma.items():
        matrix[bottom][top] = 1.0
    return matrix


def _map(phi, cocycle=Cocycle.Trivial):
    """The DW map of the φ-twisted T² cylinder, as a numpy array."""
    return np.asarray(DijkgraafWitten(
        Cobordism.twistedCylinder(_torus(), phi), cocycle).map())


# --------------------------------------------------------------------------- #
# The headline: the swap gives a verified non-identity permutation.
# --------------------------------------------------------------------------- #
class TestSwapIsNonIdentityPermutation(unittest.TestCase):

    EXPECTED = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=float)

    def test_map_is_not_the_identity(self):
        # The whole point of #192: the realizable image is no longer {I₄}.
        matrix = _map(_SWAP)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertFalse(np.allclose(matrix, np.eye(4)),
                         "the twisted cylinder must NOT be the identity")

    def test_map_equals_the_expected_swap_permutation(self):
        # Fixes the trivial class [0] (index 0) and [a+b] (index 3); transposes
        # [a]↔[b] (indices 1, 2) — φ mod 2 = S = [[0,1],[1,0]].
        np.testing.assert_allclose(_map(_SWAP), self.EXPECTED, atol=1e-9)

    def test_map_matches_independent_numpy_oracle(self):
        oracle = _oracle_matrix(_top_triangles(_torus()), _SWAP)
        np.testing.assert_allclose(_map(_SWAP), oracle, atol=1e-9)
        np.testing.assert_allclose(oracle, self.EXPECTED, atol=1e-9)

    def test_map_is_unitary(self):
        matrix = _map(_SWAP)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=1e-9)

    def test_map_is_a_genuine_zero_one_permutation(self):
        matrix = _map(_SWAP).real
        # Entries are 0/1, with exactly one 1 per row and per column.
        self.assertTrue(np.all(np.isclose(matrix, 0) | np.isclose(matrix, 1)))
        np.testing.assert_array_equal(np.round(matrix.sum(axis=0)).astype(int),
                                      np.ones(4, dtype=int))
        np.testing.assert_array_equal(np.round(matrix.sum(axis=1)).astype(int),
                                      np.ones(4, dtype=int))

    def test_map_fixes_the_trivial_holonomy_class(self):
        # The trivial connection (index 0) is φ-invariant for any φ.
        self.assertAlmostEqual(_map(_SWAP)[0, 0], 1.0, places=9)

    def test_map_is_a_single_transposition(self):
        # The swap fixes two classes ([0], [a+b]) and exchanges the other two
        # ([a]↔[b]): an involution with exactly one 2-cycle.
        matrix = _map(_SWAP).real
        np.testing.assert_allclose(matrix @ matrix, np.eye(4), atol=1e-9)  # P²=I
        fixed = int(np.round(np.trace(matrix)))
        self.assertEqual(fixed, 2, "a transposition fixes exactly two classes")
        off_diagonal_ones = int(np.round(matrix.sum() - np.trace(matrix)))
        self.assertEqual(off_diagonal_ones, 2, "exactly one 2-cycle")

    def test_sign_cocycle_gives_the_same_permutation(self):
        # The mod-2 cup cube vanishes on the cylinder, so the Sign twist agrees.
        np.testing.assert_allclose(_map(_SWAP, Cocycle.Sign), self.EXPECTED,
                                   atol=1e-9)


# --------------------------------------------------------------------------- #
# The trivial twist (φ = identity) recovers the product cylinder → I₄.
# --------------------------------------------------------------------------- #
class TestTrivialTwistIsIdentity(unittest.TestCase):

    def test_identity_phi_is_the_identity_map(self):
        np.testing.assert_allclose(_map(_IDENTITY), np.eye(4), atol=1e-9)

    def test_identity_phi_matches_the_product_cylinder(self):
        # Cross-check against the independently-built product cylinder
        # T²×[0,T] = SimplicialProduct(T², interval), which #109 pins to I₄.
        interval = tessera.SolidSimplex(1)
        product = _build(tessera.SimplicialProduct(_torus_topology(), interval))
        product_map = np.asarray(
            DijkgraafWitten(product, Cocycle.Trivial).map())
        np.testing.assert_allclose(_map(_IDENTITY), product_map, atol=1e-9)

    def test_identity_phi_matches_oracle(self):
        oracle = _oracle_matrix(_top_triangles(_torus()), _IDENTITY)
        np.testing.assert_allclose(oracle, np.eye(4), atol=1e-9)


# --------------------------------------------------------------------------- #
# Functoriality of the DW functor.
# --------------------------------------------------------------------------- #
class TestFunctoriality(unittest.TestCase):

    def test_order_of_map_equals_order_of_phi(self):
        # φ (the swap) has order 2, so the DW map has order 2: map² = I₄, and
        # map ≠ I₄ (so the order is exactly 2, not 1).
        matrix = _map(_SWAP)
        self.assertFalse(np.allclose(matrix, np.eye(4)))
        np.testing.assert_allclose(matrix @ matrix, np.eye(4), atol=1e-9)

    def test_composing_twists_composes_permutations(self):
        # twistedCylinder(φ∘φ) realizes the composite permutation map(φ)·map(φ);
        # for the order-2 swap φ∘φ = id, so this is I₄ — the identity cylinder.
        swap_squared = [_SWAP[_SWAP[i]] for i in range(9)]
        self.assertEqual(swap_squared, _IDENTITY)
        composed = _map(swap_squared)
        np.testing.assert_allclose(composed, _map(_SWAP) @ _map(_SWAP), atol=1e-9)
        np.testing.assert_allclose(composed, np.eye(4), atol=1e-9)

    def test_realizable_image_is_strictly_larger_than_a_point(self):
        # The realizable image now contains {I₄, swap} — at least two distinct
        # unitaries, where before #192 it was the single point {I₄}.
        image = [_map(_IDENTITY), _map(_SWAP)]
        distinct = []
        for matrix in image:
            if not any(np.allclose(matrix, seen, atol=1e-9) for seen in distinct):
                distinct.append(matrix)
        self.assertGreaterEqual(len(distinct), 2)


# --------------------------------------------------------------------------- #
# Boundary / b₁ cross-checks against the existing machinery.
# --------------------------------------------------------------------------- #
class TestBoundaryCrossChecks(unittest.TestCase):

    def test_boundary_dimensions_are_two_tori(self):
        # ∂W = T² ⊔ T², each Z(T²) of dimension 2^{b₁(T²)} = 4.
        dims = DijkgraafWitten(Cobordism.twistedCylinder(_torus(), _SWAP),
                               Cocycle.Trivial).boundaryDimensions()
        self.assertEqual(list(dims), [4, 4])

    def test_is_a_genuine_cobordism_from_torus_to_torus(self):
        torus = _torus()
        twisted = Cobordism.twistedCylinder(torus, _SWAP)
        result = Cobordism.verify(twisted, torus, torus)
        self.assertTrue(result.ok, result.detail)

    def test_bulk_is_a_torus_times_interval(self):
        # W ≃ T²×I (φ is a homeomorphism), so b = (1, 2, 1, 0): the b₁ = 2 bulk
        # is what 2^{b₁} = 4 holonomy classes per boundary rests on.
        chain = cobordism.ChainComplex.fromSpacetime(
            Cobordism.twistedCylinder(_torus(), _SWAP))
        self.assertEqual(chain.bettiNumbers(), [1, 2, 1, 0])

    def test_boundary_has_two_torus_components(self):
        twisted = Cobordism.twistedCylinder(_torus(), _SWAP)
        faces = Cobordism.boundaryFaces(twisted)
        components = Cobordism.connectedComponents([list(f) for f in faces])
        self.assertEqual(len(components), 2)
        for component in components:  # each a closed T² (χ = 0)
            verts = {v for tri in component for v in tri}
            edges = {tuple(sorted(pair)) for tri in component
                     for pair in ((tri[0], tri[1]), (tri[0], tri[2]),
                                  (tri[1], tri[2]))}
            self.assertEqual(len(verts) - len(edges) + len(component), 0)


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
class TestGuards(unittest.TestCase):

    def test_phi_must_have_one_entry_per_vertex(self):
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.twistedCylinder(_torus(), _SWAP[:-1])

    def test_phi_must_be_a_permutation(self):
        not_a_permutation = list(_SWAP)
        not_a_permutation[0] = not_a_permutation[1]  # 1 repeated, 0 missing
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.twistedCylinder(_torus(), not_a_permutation)

    def test_phi_must_be_a_simplicial_automorphism(self):
        # A bare transposition of two non-equivalent vertices is a permutation
        # but not a simplicial automorphism of the torus.
        not_an_automorphism = list(_IDENTITY)
        not_an_automorphism[0], not_an_automorphism[1] = 1, 0
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.twistedCylinder(_torus(), not_an_automorphism)

    def test_sigma_must_be_a_surface(self):
        # A circle (1-manifold) has no triangles, so it is rejected.
        circle = _build(_circle())
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.twistedCylinder(circle, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
