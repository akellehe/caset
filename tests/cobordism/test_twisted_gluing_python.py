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

"""Twisted gluing + T² Dehn twists (#135, spec §5.0).

`Cobordism.glue`/`selfGlue` gained an overload that identifies the shared
boundary surface Σ_C by a *caller-supplied* vertex bijection instead of the
canonical order-preserving isomorphism. Self-gluing T²×[0,T] through such a
bijection realizes the **mapping torus** of the boundary self-map — the
building block for the modular S/T data (#136).

`TorusTwist` wraps the GL(2,ℤ) mapping classes of T² and realizes them on the
product-lattice torus `SimplicialProduct(S¹,S¹)`. The SL(2,ℤ) generators are
S = [[0,-1],[1,0]] and the Dehn twist T = [[1,1],[0,1]].

Acceptance (spec §5.0):

* **Identity-bijection glue reproduces #113.** Gluing the standard torus
  cylinders through the canonical (order-preserving) vertex correspondence gives
  the same composite as the auto-finding `glue`/`selfGlue` — the DW boundary map
  is `id_{Z(T²)}` and the self-glued T²×[0,T] is T³.

* **Twisted self-glue = mapping torus.** Self-gluing a thick T²×[0,T] through a
  non-trivial torus automorphism yields a valid closed 3-manifold that is a
  *non-trivial torus bundle* (its Betti numbers differ from T³'s).

* **Modular relations.** The S/T generators obey S⁴ = I and (ST)³ = S² (as exact
  integer matrices and as the induced vertex permutations).

The realizable simplicial twist is the coordinate flip (i,j)↦(j,i): the staircase
product triangulation is not vertex-transitive, so S and T are *not* simplicial
automorphisms of it (verified below) — the classical obstruction to realizing a
parabolic / order-4 mapping class on a fixed torus triangulation.
"""

import itertools
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
Cobordism = cobordism.Cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle
TorusTwist = cobordism.TorusTwist
ChainComplex = cobordism.ChainComplex

COCYCLES = ((Cocycle.Trivial, "trivial"), (Cocycle.Sign, "sign"))

CIRCLE_VERTS = 3          # S¹ = ∂Δ² has 3 vertices
TORUS_VERTS = CIRCLE_VERTS * CIRCLE_VERTS  # 9


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    return tessera.SimplexBoundarySphere(1)          # S¹ = ∂Δ²


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())  # T² = S¹ × S¹


def _interval():
    return tessera.SolidSimplex(1)                    # [0, 1]


def _torus_cylinder():
    # T²×I, two layers; SimplicialProduct(torus, interval) ⇒ vertex (u, v) has
    # id u*2 + v, so the bottom torus (v=0) is the even ids and the top (v=1)
    # the odd ids — a known layout we use to build explicit identifications.
    return _build(tessera.SimplicialProduct(_torus_topology(), _interval()))


def _from_simplices(num_vertices, simplices):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, tessera.Toroid())
    verts = [spacetime.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        spacetime.createSimplex([verts[i] for i in simplex])
    return spacetime


def _top_simplices(spacetime):
    tuples = [tuple(sorted(v.getId() for v in s.getVertices()))
              for s in spacetime.getSimplices()]
    top = max(len(t) for t in tuples)
    return [t for t in tuples if len(t) == top]


def _torus_triangles():
    """The 18 triangles of SimplicialProduct(S¹,S¹) on torus vertices 0..8."""
    return _top_simplices(_build(_torus_topology()))


def _prism_tets(triangle, t):
    # Staircase of (2-simplex × edge): triangle (s0<s1<s2) over the edge
    # (t, t+1) → three tetrahedra, vertices written as (torus_vertex, layer).
    s0, s1, s2 = triangle
    return [((s0, t), (s1, t), (s2, t), (s2, t + 1)),
            ((s0, t), (s1, t), (s1, t + 1), (s2, t + 1)),
            ((s0, t), (s0, t + 1), (s1, t + 1), (s2, t + 1))]


def _thick_torus_cylinder(layers):
    """T²×[0,layers] as an explicit triangulation; vertex (u,t)→u*(layers+1)+t.

    Bottom torus is layer 0, top torus is layer `layers`. Returns the Spacetime
    and the (u,t)→id indexer."""
    triangles = _torus_triangles()
    def pid(u, t):
        return u * (layers + 1) + t
    tets = []
    for tri in triangles:
        for t in range(layers):
            for tet in _prism_tets(tri, t):
                tets.append(tuple(sorted(pid(u, tt) for (u, tt) in tet)))
    spacetime = _from_simplices(TORUS_VERTS * (layers + 1), tets)
    return spacetime, pid


def _seam_bijection(twist_perm, layers, pid):
    """Identify the top torus (u, layers) with the bottom (φ(u), 0)."""
    return {pid(u, layers): pid(twist_perm[u], 0) for u in range(TORUS_VERTS)}


def _is_closed_manifold(spacetime):
    counts = {}
    for top in _top_simplices(spacetime):
        for drop in range(len(top)):
            facet = top[:drop] + top[drop + 1:]
            counts[facet] = counts.get(facet, 0) + 1
    return bool(counts) and all(c == 2 for c in counts.values())


def _perm_compose(p, q):       # p after q
    return {v: p[q[v]] for v in q}


def _perm_power(p, k):
    out = {v: v for v in p}
    for _ in range(k):
        out = _perm_compose(p, out)
    return out


# --------------------------------------------------------------------------- #
# Acceptance 1 — identity-bijection gluing reproduces #113.
# --------------------------------------------------------------------------- #
class TestIdentityBijectionReproducesAutoGlue(unittest.TestCase):

    def test_explicit_canonical_glue_matches_auto_glue(self):
        # Stack two torus cylinders by identifying W2's bottom torus (even ids
        # 2u) with W1's top torus (odd ids 2u+1) via the identity on u — exactly
        # the order-preserving correspondence the auto-glue uses.
        w1, w2 = _torus_cylinder(), _torus_cylinder()
        canonical = {2 * u: 2 * u + 1 for u in range(TORUS_VERTS)}
        glued_explicit = Cobordism.glue(w1, w2, canonical)
        glued_auto = Cobordism.glue(w1, w2)
        # Same size and same homology as the auto composite, and #113's result:
        # the DW boundary map is the 4×4 identity (a product cobordism).
        self.assertEqual(glued_explicit.getVertexCount(),
                         glued_auto.getVertexCount())
        self.assertEqual(glued_explicit.getVertexCount(),
                         2 * w1.getVertexCount() - TORUS_VERTS)
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                m_explicit = np.asarray(DijkgraafWitten(glued_explicit, cocycle).map())
                m_auto = np.asarray(DijkgraafWitten(glued_auto, cocycle).map())
                self.assertEqual(m_explicit.shape, (4, 4))
                np.testing.assert_allclose(m_explicit, np.eye(4), atol=1e-12)
                np.testing.assert_allclose(m_explicit, m_auto, atol=1e-12)

    def test_explicit_identity_selfglue_is_three_torus(self):
        # The canonical (identity-on-(i,j)) self-glue of a thick T²×[0,3] is T³,
        # the same closed manifold the auto selfGlue produces.
        cylinder, pid = _thick_torus_cylinder(3)
        identity = TorusTwist.identity().vertexPermutation(CIRCLE_VERTS)
        bijection = _seam_bijection(identity, 3, pid)
        closed_explicit = Cobordism.selfGlue(cylinder, bijection)
        closed_auto = Cobordism.selfGlue(cylinder)
        for closed in (closed_explicit, closed_auto):
            self.assertEqual(len(Cobordism.boundaryFaces(closed)), 0)
            self.assertTrue(_is_closed_manifold(closed))
            chain = ChainComplex.fromSpacetime(closed)
            self.assertEqual(chain.dimension(), 3)
            self.assertTrue(chain.boundaryComposesToZero())
            self.assertEqual(chain.bettiNumbers(), [1, 3, 3, 1])  # T³


# --------------------------------------------------------------------------- #
# Acceptance 2 — twisted self-glue is the mapping torus (a torus bundle).
# --------------------------------------------------------------------------- #
class TestTwistedMappingTorus(unittest.TestCase):

    def test_flip_twisted_selfglue_is_nontrivial_torus_bundle(self):
        # Self-gluing the thick cylinder through the coordinate flip (i,j)↦(j,i)
        # — a genuine simplicial automorphism of the product torus — builds the
        # mapping torus of that automorphism: a valid closed 3-manifold whose
        # homology differs from T³, i.e. a non-trivial torus bundle.
        cylinder, pid = _thick_torus_cylinder(3)
        flip = TorusTwist.flip().vertexPermutation(CIRCLE_VERTS)
        bundle = Cobordism.selfGlue(cylinder, _seam_bijection(flip, 3, pid))

        self.assertEqual(len(Cobordism.boundaryFaces(bundle)), 0)  # closed
        self.assertTrue(_is_closed_manifold(bundle))
        chain = ChainComplex.fromSpacetime(bundle)
        self.assertEqual(chain.dimension(), 3)
        self.assertTrue(chain.boundaryComposesToZero())
        self.assertEqual(chain.eulerCharacteristic(), 0)  # closed 3-manifold
        # The orientation-reversing flip [[0,1],[1,0]] gives the non-orientable
        # T²-bundle with H_* = (Z, Z², Z⊕Z₂, 0): Betti [1,2,1,0] over Q.
        betti = chain.bettiNumbers()
        self.assertEqual(betti, [1, 2, 1, 0])
        self.assertNotEqual(betti, [1, 3, 3, 1])  # ≠ T³ ⇒ non-trivial bundle

    def test_flip_is_a_simplicial_automorphism_but_S_and_T_are_not(self):
        # The flip preserves the staircase product triangulation; S and the Dehn
        # twist T do not (it is not vertex-transitive). This is why the mapping
        # torus above uses the flip — the classical obstruction to realizing a
        # parabolic / order-4 mapping class simplicially on a fixed triangulation.
        torus = _build(_torus_topology())
        flip = TorusTwist.flip().vertexPermutation(CIRCLE_VERTS)
        ident = TorusTwist.identity().vertexPermutation(CIRCLE_VERTS)
        s = TorusTwist.S().vertexPermutation(CIRCLE_VERTS)
        t = TorusTwist.T().vertexPermutation(CIRCLE_VERTS)
        self.assertTrue(TorusTwist.isSimplicialAutomorphism(torus, ident))
        self.assertTrue(TorusTwist.isSimplicialAutomorphism(torus, flip))
        self.assertFalse(TorusTwist.isSimplicialAutomorphism(torus, s))
        self.assertFalse(TorusTwist.isSimplicialAutomorphism(torus, t))

    def test_thin_collar_twisted_selfglue_is_rejected(self):
        # A two-layer cylinder is too thin: folding its ends collapses a
        # tetrahedron, so the twisted selfGlue must refuse it (same guard as the
        # canonical selfGlue).
        cylinder, pid = _thick_torus_cylinder(1)
        flip = TorusTwist.flip().vertexPermutation(CIRCLE_VERTS)
        with self.assertRaises(RuntimeError):
            Cobordism.selfGlue(cylinder, _seam_bijection(flip, 1, pid))


# --------------------------------------------------------------------------- #
# Acceptance 3 — the modular S/T relations.
# --------------------------------------------------------------------------- #
class TestModularRelations(unittest.TestCase):

    def test_relations_hold_as_exact_matrices(self):
        self.assertTrue(TorusTwist.satisfiesModularRelations())
        S, T, I = TorusTwist.S(), TorusTwist.T(), TorusTwist.identity()
        # S⁴ = I and (ST)³ = S² as integer matrices.
        self.assertTrue(S.power(4).equals(I))
        self.assertTrue(S.compose(T).power(3).equals(S.power(2)))
        # S² = −I is the central involution; S is order 4, not 2.
        self.assertTrue(S.power(2).equals(TorusTwist(-1, 0, 0, -1)))
        self.assertFalse(S.power(2).equals(I))
        # Orientation: S, T preserve it; the realizable flip reverses it.
        self.assertEqual(S.determinant(), 1)
        self.assertEqual(T.determinant(), 1)
        self.assertEqual(TorusTwist.flip().determinant(), -1)

    def test_relations_hold_as_vertex_permutations(self):
        # The same relations as the induced permutations of the torus vertices.
        s = TorusTwist.S().vertexPermutation(CIRCLE_VERTS)
        t = TorusTwist.T().vertexPermutation(CIRCLE_VERTS)
        ident = TorusTwist.identity().vertexPermutation(CIRCLE_VERTS)
        self.assertEqual(_perm_power(s, 4), ident)                 # S⁴ = id
        st = _perm_compose(s, t)
        self.assertEqual(_perm_power(st, 3), _perm_power(s, 2))    # (ST)³ = S²

    def test_T_matches_the_shear_and_S_the_rotation(self):
        # T is the shear (i,j)↦(i+j, j); S is (i,j)↦(−j, i). Spot-check the
        # vertex permutation against these closed forms (id = i*n + j).
        n = CIRCLE_VERTS
        t = TorusTwist.T().vertexPermutation(n)
        s = TorusTwist.S().vertexPermutation(n)
        for i in range(n):
            for j in range(n):
                self.assertEqual(t[i * n + j], ((i + j) % n) * n + j)
                self.assertEqual(s[i * n + j], ((-j) % n) * n + (i % n))


if __name__ == "__main__":
    unittest.main()
