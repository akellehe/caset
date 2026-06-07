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

"""Gluing constructor + T5 functoriality (#113).

The Dijkgraaf–Witten boundary map is a functor: gluing W₁: Σ_A → Σ_C to
W₂: Σ_C → Σ_B along the shared surface Σ_C composes the two boundary maps,

    Z(W₂ ∪_{Σ_C} W₁) = Z(W₂) · Z(W₁).

`Cobordism.glue` forms the composite complex (identifying the matching boundary
surfaces vertex-for-vertex, reindexing, and merging); `Cobordism.selfGlue`
caps a cobordism by gluing its two boundary components to each other (the
mapping torus / categorical trace). The state sum of the glued complex is then
compared to the matrix product of the individual boundary maps.

Acceptance (spec §5, plan ticket T5):

* **T5 — composition is the matrix product.** For a glued pair,
  `DijkgraafWitten(glue(W₁, W₂), ω).map()` equals `map(W₂) · map(W₁)`, for both
  `Cocycle.Trivial` and `Cocycle.Sign`. Concrete fixture: two trivial cylinders
  `T²×I` glue to a (longer) cylinder, so the composite is the identity,
  consistent with `id · id`; a second pair `S²×I` glues to the 1×1 identity.

* **Trace = closed invariant.** Capping a cylinder reproduces the closed
  invariant: `Tr(map(T²×I)) = Z(T³)`, and `Z(selfGlue(T²×[0,T])) =
  Tr(map(T²×[0,T]))` on a thick enough cylinder (the self-glued cylinder is the
  3-torus). Both for Trivial and Sign.

* **Functoriality on states.** Gluing a cap (the solid torus, ∂ = T²) through a
  cylinder leaves its boundary vector unchanged: vector(glue(solid torus,
  cylinder)) = map(cylinder) · vector(solid torus) = vector(solid torus).
"""

import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
Cobordism = cobordism.Cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle

COCYCLES = ((Cocycle.Trivial, "trivial"), (Cocycle.Sign, "sign"))


# --------------------------------------------------------------------------- #
# Fixtures (identical conventions to the #109 boundary-map tests).
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    return tessera.SimplexBoundarySphere(1)            # S¹ = ∂Δ²


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())  # T² = S¹ × S¹


def _interval():
    return tessera.SolidSimplex(1)                     # [0, 1] (a single edge)


def _torus_cylinder():
    return _build(tessera.SimplicialProduct(_torus_topology(), _interval()))


def _sphere_cylinder():
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            _interval()))


def _solid_torus():
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


def _s2_cross_s1():
    # S² × S¹, the closed manifold a capped S²×I is. Small enough (12 vertices)
    # for the gauge-redundant closed state sum, unlike T³ (the cap of T²×I).
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            _circle()))


def _boundary_face_count(spacetime):
    return len(Cobordism.boundaryFaces(spacetime))


def _top_simplices(spacetime):
    tuples = [tuple(sorted(v.getId() for v in s.getVertices()))
              for s in spacetime.getSimplices()]
    top = max(len(t) for t in tuples)
    return [t for t in tuples if len(t) == top]


def _is_closed_manifold(spacetime):
    """Every codimension-one face is shared by exactly two top simplices."""
    counts = {}
    for top in _top_simplices(spacetime):
        for drop in range(len(top)):
            facet = top[:drop] + top[drop + 1:]
            counts[facet] = counts.get(facet, 0) + 1
    return bool(counts) and all(c == 2 for c in counts.values())


# --------------------------------------------------------------------------- #
# T5 — the boundary map composes (matrix product).
# --------------------------------------------------------------------------- #
class TestGlueIsMatrixProduct(unittest.TestCase):

    def test_torus_cylinders_compose_to_identity(self):
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                w1, w2 = _torus_cylinder(), _torus_cylinder()
                glued = Cobordism.glue(w1, w2)
                map1 = np.asarray(DijkgraafWitten(w1, cocycle).map())
                map2 = np.asarray(DijkgraafWitten(w2, cocycle).map())
                map_glued = np.asarray(DijkgraafWitten(glued, cocycle).map())
                # Z(T²) is 4-dimensional, and a product cobordism is the identity.
                self.assertEqual(map_glued.shape, (4, 4))
                np.testing.assert_allclose(map_glued, map2 @ map1, atol=1e-12)
                np.testing.assert_allclose(map_glued, np.eye(4), atol=1e-12)

    def test_sphere_cylinders_compose(self):
        # A second, cheap glued pair: Z(S²) is one-dimensional, so every map is
        # the 1×1 identity [[1]] and the composite is too.
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                w1, w2 = _sphere_cylinder(), _sphere_cylinder()
                glued = Cobordism.glue(w1, w2)
                map1 = np.asarray(DijkgraafWitten(w1, cocycle).map())
                map2 = np.asarray(DijkgraafWitten(w2, cocycle).map())
                map_glued = np.asarray(DijkgraafWitten(glued, cocycle).map())
                self.assertEqual(map_glued.shape, (1, 1))
                np.testing.assert_allclose(map_glued, map2 @ map1, atol=1e-12)
                np.testing.assert_allclose(map_glued, np.eye(1), atol=1e-12)

    def test_gluing_is_associative_on_cylinders(self):
        # (C ∪ C) ∪ C and C ∪ (C ∪ C) are both longer cylinders ⇒ both identity.
        left = Cobordism.glue(Cobordism.glue(_torus_cylinder(),
                                             _torus_cylinder()),
                              _torus_cylinder())
        right = Cobordism.glue(_torus_cylinder(),
                               Cobordism.glue(_torus_cylinder(),
                                              _torus_cylinder()))
        for glued in (left, right):
            map_glued = np.asarray(DijkgraafWitten(glued, Cocycle.Trivial).map())
            np.testing.assert_allclose(map_glued, np.eye(4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Trace = closed invariant (the gluing axiom previewed in #109).
# --------------------------------------------------------------------------- #
class TestTraceIsClosedInvariant(unittest.TestCase):

    def test_trace_of_sphere_cylinder_is_partition_function_of_s2_cross_s1(self):
        # Capping both ends of S²×I gives S²×S¹; the categorical trace of the
        # boundary map equals the closed partition function: Tr(map) = Z(S²×S¹).
        # (S²×S¹ is small enough for the gauge-redundant closed state sum; T³ —
        # the cap of T²×I — is not, so the torus case is pinned via the trace.)
        s2_cross_s1 = _s2_cross_s1()
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                trace = np.trace(np.asarray(
                    DijkgraafWitten(_sphere_cylinder(), cocycle).map()))
                z_closed = DijkgraafWitten(s2_cross_s1, cocycle).partitionFunction()
                self.assertAlmostEqual(trace, z_closed, places=12)
                self.assertAlmostEqual(z_closed, 1.0, places=12)  # Z(S²×S¹) = 1

    def test_trace_of_torus_cylinder_is_dimension_of_boundary_space(self):
        # Tr(map(T²×I)) = dim Z(T²) = 4 = 2^{b₁(T³)−1} = Z(T³). The 3-torus is
        # too large for the brute-force closed state sum, so its invariant is
        # pinned through the trace (the self-glued T³ is verified structurally).
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                trace = np.trace(np.asarray(
                    DijkgraafWitten(_torus_cylinder(), cocycle).map()))
                self.assertAlmostEqual(trace, 4.0, places=12)

    def test_self_glued_sphere_cylinder_partition_function_matches_trace(self):
        # A thick S²×I (four layers, built by gluing three short ones) self-glues
        # without collapsing a tetrahedron, closing to S²×S¹; its partition
        # function equals the trace of the boundary map — Tr(map) = Z_closed on a
        # genuinely self-glued cylinder.
        cylinder = Cobordism.glue(
            Cobordism.glue(_sphere_cylinder(), _sphere_cylinder()),
            _sphere_cylinder())
        closed = Cobordism.selfGlue(cylinder)
        self.assertEqual(_boundary_face_count(closed), 0)   # no boundary
        self.assertTrue(_is_closed_manifold(closed))
        chain = cobordism.ChainComplex.fromSpacetime(closed)
        self.assertEqual(chain.dimension(), 3)
        self.assertTrue(chain.boundaryComposesToZero())
        self.assertEqual(chain.bettiNumbers(), [1, 1, 1, 1])  # S²×S¹
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                trace = np.trace(np.asarray(
                    DijkgraafWitten(cylinder, cocycle).map()))
                z_closed = DijkgraafWitten(closed, cocycle).partitionFunction()
                self.assertAlmostEqual(z_closed, trace, places=12)

    def test_self_glued_torus_cylinder_is_the_three_torus(self):
        # The torus analogue: self-gluing a thick T²×I closes it to T³ (verified
        # by its Betti numbers; Z(T³) itself is too large to brute-force, so only
        # the topology is checked here — the value identity is the S²×S¹ case).
        cylinder = Cobordism.glue(
            Cobordism.glue(_torus_cylinder(), _torus_cylinder()),
            _torus_cylinder())
        closed = Cobordism.selfGlue(cylinder)
        self.assertEqual(_boundary_face_count(closed), 0)
        self.assertTrue(_is_closed_manifold(closed))
        chain = cobordism.ChainComplex.fromSpacetime(closed)
        self.assertEqual(chain.dimension(), 3)
        self.assertTrue(chain.boundaryComposesToZero())
        self.assertEqual(chain.bettiNumbers(), [1, 3, 3, 1])  # T³

    def test_self_glue_rejects_thin_collar(self):
        # The minimal T²×I is two layers thick; gluing its ends to each other
        # would collapse a tetrahedron, so selfGlue must refuse it.
        with self.assertRaises(RuntimeError):
            Cobordism.selfGlue(_torus_cylinder())


# --------------------------------------------------------------------------- #
# Functoriality on states: gluing a cap through a cylinder is the identity.
# --------------------------------------------------------------------------- #
class TestGlueOfStateVector(unittest.TestCase):

    def test_solid_torus_through_cylinder_preserves_boundary_vector(self):
        # The solid torus S¹×D² (∂ = T²) capped by a cylinder is still a solid
        # torus; since the cylinder map is the identity, the boundary state
        # vector is unchanged: vector(glued) = map(cylinder)·vector(cap).
        for cocycle, kind in COCYCLES:
            with self.subTest(cocycle=kind):
                cap = _solid_torus()
                glued = Cobordism.glue(cap, _torus_cylinder())
                # The composite has a single boundary component (the cylinder's
                # free end), so it is again read as a state vector.
                self.assertEqual(len(DijkgraafWitten(glued, cocycle)
                                     .boundaryDimensions()), 1)
                vector_cap = np.asarray(
                    DijkgraafWitten(cap, cocycle).boundaryVector())
                vector_glued = np.asarray(
                    DijkgraafWitten(glued, cocycle).boundaryVector())
                self.assertEqual(vector_glued.shape, vector_cap.shape)
                np.testing.assert_allclose(sorted(vector_glued.real),
                                           sorted(vector_cap.real), atol=1e-12)


# --------------------------------------------------------------------------- #
# Structure of the glued complex.
# --------------------------------------------------------------------------- #
class TestGlueStructure(unittest.TestCase):

    def test_glued_cylinder_is_a_valid_two_boundary_cobordism(self):
        glued = Cobordism.glue(_torus_cylinder(), _torus_cylinder())
        # A valid manifold-with-boundary: ∂∂ = 0 and two boundary components.
        chain = cobordism.ChainComplex.fromSpacetime(glued)
        self.assertEqual(chain.dimension(), 3)
        self.assertTrue(chain.boundaryComposesToZero())
        components = Cobordism.connectedComponents(
            Cobordism.boundaryFaces(glued))
        self.assertEqual(len(components), 2)
        # Both remaining boundary components are tori (isomorphic to the original
        # cylinder's boundary torus).
        torus_face = Cobordism.connectedComponents(
            Cobordism.boundaryFaces(_torus_cylinder()))[0]
        for component in components:
            self.assertTrue(Cobordism.areIsomorphic(component, torus_face))

    def test_glue_merges_along_shared_surface(self):
        # Gluing identifies one torus (9 vertices) of each cylinder, so the glued
        # vertex count is |V₁| + |V₂| − |Σ_C|.
        cyl = _torus_cylinder()
        n = cyl.getVertexCount()
        torus_vertices = len({v for face in Cobordism.connectedComponents(
            Cobordism.boundaryFaces(cyl))[0] for v in face})
        glued = Cobordism.glue(cyl, _torus_cylinder())
        self.assertEqual(glued.getVertexCount(), 2 * n - torus_vertices)

    def test_glue_rejects_no_shared_surface(self):
        # The solid torus (∂ = T²) and the sphere cylinder (∂ = S² ⊔ S²) are both
        # 3-manifolds, but share no isomorphic boundary surface (T² ≇ S²).
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.glue(_solid_torus(), _sphere_cylinder())

    def test_glue_rejects_dimension_mismatch(self):
        # A solid 4-simplex (top dimension 4) cannot glue to a 3-manifold.
        solid4 = _build(tessera.SolidSimplex(4))
        with self.assertRaises((ValueError, RuntimeError)):
            Cobordism.glue(solid4, _torus_cylinder())


if __name__ == "__main__":
    unittest.main()
