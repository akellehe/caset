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

"""Capability C: cobordism verification, boundary-structure increment (#66).

A cobordism from M1 to M2 is a manifold W whose boundary is exactly M1 and M2
side by side. We verify the boundary structure: that the boundary of W splits
into connected pieces matching M1 and M2 as triangulations, and that the
boundary is itself closed. Fixtures are built from solid simplices (balls) and
simplicial products with an interval (cylinders M x I) / a ball (null-bordisms
M x D^k). The PL-manifold link check and the orientation check are follow-ups.
"""

import unittest

import tessera

cobordism = tessera.cobordism


def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _empty():
    return tessera.Spacetime()


def _sphere(n):
    return _build(tessera.SimplexBoundarySphere(n))


def _interval():
    return tessera.SolidSimplex(1)  # a single edge = the interval [0, 1]


def _cylinder(sphere_dim):
    # M x I, a cobordism from M to M.
    return _build(tessera.SimplicialProduct(
        tessera.SimplexBoundarySphere(sphere_dim), _interval()))


def _product(*topologies):
    """Left-nested SimplicialProduct of two or more factor topologies."""
    result = topologies[0]
    for factor in topologies[1:]:
        result = tessera.SimplicialProduct(result, factor)
    return result


def _from_simplices(num_vertices, simplices):
    """Build a Spacetime directly from explicit top-simplex vertex tuples
    (vertices 0..num_vertices-1), for hand-crafted / pathological complexes."""
    # Signature dimension must match the hand-built top cells (d = max cell
    # vertex count - 1) so they register as top simplices and getBoundary sees
    # them. These complexes are not topology-driven, so derive d from the data.
    dimension = max(len(simplex) for simplex in simplices) - 1
    signature = tessera.Signature(dimension, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, tessera.Toroid())
    verts = [spacetime.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        spacetime.createSimplex([verts[i] for i in simplex])
    return spacetime


Ok = cobordism.CobordismCheck.Ok


class TestBoundaryFaces(unittest.TestCase):

    def test_ball_boundary_is_a_sphere(self):
        # The boundary of the solid n-simplex is its n+1 facets = S^{n-1}.
        for n in range(2, 6):
            with self.subTest(ball=n):
                faces = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(n)))
                self.assertEqual(len(faces), n + 1)

    def test_closed_manifold_has_no_boundary(self):
        for topology in (tessera.SimplexBoundarySphere(2),
                         tessera.SimplexBoundarySphere(4),
                         tessera.RealProjectivePlane()):
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(cobordism.Cobordism.boundaryFaces(_build(topology)), [])

    def test_cylinder_boundary_has_two_components(self):
        faces = cobordism.Cobordism.boundaryFaces(_cylinder(2))
        comps = cobordism.Cobordism.connectedComponents(faces)
        self.assertEqual(len(comps), 2)


class TestSpacetimeGetBoundary(unittest.TestCase):
    """Spacetime.getBoundary() is the canonical, side-effect-free boundary
    derivation that Cobordism.boundaryFaces() now delegates to (#162). It must
    agree with boundaryFaces() and reproduce the known boundary on the fixtures:
    SolidSimplex -> sphere boundary, cylinder -> two components, closed
    manifolds -> empty."""

    @staticmethod
    def _as_set(faces):
        return {tuple(f) for f in faces}

    def test_agrees_with_boundary_faces(self):
        for topology in (tessera.SolidSimplex(2), tessera.SolidSimplex(3),
                         tessera.SolidSimplex(4), tessera.SimplexBoundarySphere(3),
                         tessera.RealProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                st = _build(topology)
                self.assertEqual(
                    self._as_set(st.getBoundary()),
                    self._as_set(cobordism.Cobordism.boundaryFaces(st)))

    def test_solid_simplex_boundary_is_a_sphere(self):
        # The boundary of Delta^n is its n+1 facets = S^{n-1} (each on n verts).
        for n in range(2, 6):
            with self.subTest(ball=n):
                boundary = _build(tessera.SolidSimplex(n)).getBoundary()
                self.assertEqual(len(boundary), n + 1)
                self.assertTrue(all(len(f) == n for f in boundary))

    def test_closed_manifold_has_empty_boundary(self):
        for topology in (tessera.SimplexBoundarySphere(2),
                         tessera.SimplexBoundarySphere(4),
                         tessera.RealProjectivePlane()):
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(_build(topology).getBoundary(), [])

    def test_cylinder_boundary_has_two_components(self):
        comps = cobordism.Cobordism.connectedComponents(_cylinder(2).getBoundary())
        self.assertEqual(len(comps), 2)

    def test_get_boundary_is_side_effect_free(self):
        # Unlike getExternalSimplices, getBoundary() must not materialize facets:
        # the registered-simplex set is unchanged by the call.
        st = _build(tessera.SolidSimplex(4))
        before = len(st.getSimplices())
        st.getBoundary()
        self.assertEqual(len(st.getSimplices()), before)


class TestIsomorphism(unittest.TestCase):

    def test_same_sphere_is_isomorphic(self):
        s2a = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(3)))  # S^2
        s2b = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(3)))
        self.assertTrue(cobordism.Cobordism.areIsomorphic(s2a, s2b))

    def test_relabeled_is_isomorphic(self):
        a = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]   # boundary of a tetra
        b = [[10, 11, 12], [10, 11, 13], [10, 12, 13], [11, 12, 13]]
        self.assertTrue(cobordism.Cobordism.areIsomorphic(a, b))

    def test_different_spheres_not_isomorphic(self):
        s2 = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(3)))   # S^2
        s3 = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(4)))   # S^3
        self.assertFalse(cobordism.Cobordism.areIsomorphic(s2, s3))

    def test_empty_lists_are_isomorphic(self):
        self.assertTrue(cobordism.Cobordism.areIsomorphic([], []))


class TestVerifyValidCobordisms(unittest.TestCase):

    def test_ball_caps_sphere(self):
        # Solid n-simplex is a cobordism S^{n-1} -> empty (T0.2 / T3.1).
        for n in range(2, 5):
            with self.subTest(ball=n):
                result = cobordism.Cobordism.verify(
                    _build(tessera.SolidSimplex(n)),
                    _sphere(n - 1), _empty())
                self.assertTrue(result.ok, result.detail)
                self.assertEqual(result.code, Ok)

    def test_cylinders(self):
        # M x I is a cobordism M -> M (T0.1, T0.4).
        for dim in (1, 2):
            with self.subTest(sphere=dim):
                result = cobordism.Cobordism.verify(
                    _cylinder(dim), _sphere(dim), _sphere(dim))
                self.assertTrue(result.ok, result.detail)

    def test_s1_cross_s2_null_bordism(self):
        # S^1 x D^3 is a manifold whose boundary is S^1 x S^2: a cobordism
        # S^1 x S^2 -> empty (T3.2 — S^1 x S^2 bounds).
        w = _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                             tessera.SolidSimplex(3)))
        boundary = tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                             tessera.SimplexBoundarySphere(2))
        result = cobordism.Cobordism.verify(w, _build(boundary), _empty())
        self.assertTrue(result.ok, result.detail)

    def test_empty_to_empty(self):
        result = cobordism.Cobordism.verify(_empty(), _empty(), _empty())
        self.assertTrue(result.ok, result.detail)


class TestVerifyRejections(unittest.TestCase):

    def test_wrong_boundary_manifold(self):
        # Delta^4's boundary is S^3, not S^2.
        result = cobordism.Cobordism.verify(
            _build(tessera.SolidSimplex(4)), _sphere(2), _empty())
        self.assertFalse(result.ok)
        self.assertEqual(result.code, cobordism.CobordismCheck.BoundaryNotIsomorphic)

    def test_wrong_number_of_components(self):
        # A cylinder has two boundary circles/spheres, not one.
        result = cobordism.Cobordism.verify(_cylinder(2), _sphere(2), _empty())
        self.assertFalse(result.ok)
        self.assertEqual(result.code,
                         cobordism.CobordismCheck.WrongNumberOfBoundaryComponents)

    def test_closed_manifold_is_not_a_cobordism_with_boundary(self):
        # A closed 4-sphere has empty boundary; claiming it bounds S^3 fails.
        result = cobordism.Cobordism.verify(_sphere(4), _sphere(3), _empty())
        self.assertFalse(result.ok)
        self.assertEqual(result.code,
                         cobordism.CobordismCheck.WrongNumberOfBoundaryComponents)

    def test_book_of_three_triangles_has_non_closed_boundary(self):
        # Three triangles sharing the edge {1,2} — a non-manifold "book". Its
        # boundary edges meet 3-to-a-vertex at 1 and 2, so the boundary is not a
        # closed manifold.
        book = _from_simplices(5, [(0, 1, 2), (1, 2, 3), (1, 2, 4)])
        result = cobordism.Cobordism.verify(book, _empty(), _empty())
        self.assertFalse(result.ok)
        self.assertEqual(result.code,
                         cobordism.CobordismCheck.BoundaryChainNotClosed)


# A solid disk's boundary edges DO form a closed circle, by contrast.
class TestBallCapsSweep(unittest.TestCase):

    def test_solid_simplex_caps_its_boundary_sphere(self):
        # Delta^n is a cobordism S^{n-1} -> empty for a range of n.
        for n in range(2, 7):
            with self.subTest(ball=n):
                result = cobordism.Cobordism.verify(
                    _build(tessera.SolidSimplex(n)), _sphere(n - 1), _empty())
                self.assertTrue(result.ok, f"Delta^{n}: {result.detail}")
                self.assertEqual(result.code, Ok)


class TestCylinderSweep(unittest.TestCase):

    def test_sphere_cylinders(self):
        # S^n x I is a cobordism S^n -> S^n.
        for n in (1, 2, 3):
            with self.subTest(sphere=n):
                result = cobordism.Cobordism.verify(
                    _cylinder(n), _sphere(n), _sphere(n))
                self.assertTrue(result.ok, result.detail)

    def test_real_projective_plane_cylinder(self):
        # RP^2 x I is a (non-orientable) cobordism RP^2 -> RP^2.
        rp2 = tessera.RealProjectivePlane()
        w = _build(tessera.SimplicialProduct(rp2, _interval()))
        result = cobordism.Cobordism.verify(w, _build(rp2), _build(rp2))
        self.assertTrue(result.ok, result.detail)


class TestNullBordisms(unittest.TestCase):
    """W = M x D^k has boundary M x S^{k-1}, exhibiting M x S^{k-1} as bounding."""

    def test_solid_torus_bounds_torus(self):
        # S^1 x D^2 (solid torus) -> T^2 = S^1 x S^1.
        w = _build(_product(tessera.SimplexBoundarySphere(1), tessera.SolidSimplex(2)))
        torus = _product(tessera.SimplexBoundarySphere(1), tessera.SimplexBoundarySphere(1))
        result = cobordism.Cobordism.verify(w, _build(torus), _empty())
        self.assertTrue(result.ok, result.detail)

    def test_s1_cross_s2_bounds(self):
        # S^1 x D^3 -> S^1 x S^2 (T3.2).
        w = _build(_product(tessera.SimplexBoundarySphere(1), tessera.SolidSimplex(3)))
        boundary = _product(tessera.SimplexBoundarySphere(1), tessera.SimplexBoundarySphere(2))
        result = cobordism.Cobordism.verify(w, _build(boundary), _empty())
        self.assertTrue(result.ok, result.detail)

    def test_s2_cross_s1_bounds(self):
        # S^2 x D^2 -> S^2 x S^1.
        w = _build(_product(tessera.SimplexBoundarySphere(2), tessera.SolidSimplex(2)))
        boundary = _product(tessera.SimplexBoundarySphere(2), tessera.SimplexBoundarySphere(1))
        result = cobordism.Cobordism.verify(w, _build(boundary), _empty())
        self.assertTrue(result.ok, result.detail)

    def test_t3_bounds(self):
        # T^2 x D^2 -> T^3 = T^2 x S^1 (T3.3 — the 3-torus bounds).
        torus = _product(tessera.SimplexBoundarySphere(1), tessera.SimplexBoundarySphere(1))
        w = _build(tessera.SimplicialProduct(torus, tessera.SolidSimplex(2)))
        t3 = tessera.SimplicialProduct(torus, tessera.SimplexBoundarySphere(1))
        result = cobordism.Cobordism.verify(w, _build(t3), _empty())
        self.assertTrue(result.ok, result.detail)


class TestM1M2Order(unittest.TestCase):
    """verify() matches the boundary to the *set* {M1, M2}, so the order is
    irrelevant and either may be empty."""

    def test_cap_order_irrelevant(self):
        d4 = tessera.SolidSimplex(4)
        forward = cobordism.Cobordism.verify(_build(d4), _sphere(3), _empty())
        reversed_ = cobordism.Cobordism.verify(_build(d4), _empty(), _sphere(3))
        self.assertTrue(forward.ok and reversed_.ok)

    def test_cylinder_symmetric(self):
        cyl = _cylinder(2)
        self.assertTrue(cobordism.Cobordism.verify(cyl, _sphere(2), _sphere(2)).ok)


class TestConnectedComponentsUnit(unittest.TestCase):

    def test_single_sphere_is_connected(self):
        faces = cobordism.Cobordism.boundaryFaces(_build(tessera.SolidSimplex(4)))
        self.assertEqual(len(cobordism.Cobordism.connectedComponents(faces)), 1)

    def test_two_disjoint_triangles_are_two_components(self):
        comps = cobordism.Cobordism.connectedComponents(
            [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(len(comps), 2)

    def test_two_triangles_sharing_an_edge_are_one_component(self):
        comps = cobordism.Cobordism.connectedComponents(
            [[0, 1, 2], [1, 2, 3]])
        self.assertEqual(len(comps), 1)

    def test_empty_has_no_components(self):
        self.assertEqual(cobordism.Cobordism.connectedComponents([]), [])


class TestIsomorphismExtended(unittest.TestCase):

    def _boundary_of(self, topology):
        return cobordism.Cobordism.boundaryFaces(_build(topology))

    def test_relabeling_preserves_isomorphism(self):
        a = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]            # boundary tetra
        b = [[5, 6, 7], [5, 6, 8], [5, 7, 8], [6, 7, 8]]            # relabeled
        self.assertTrue(cobordism.Cobordism.areIsomorphic(a, b))

    def test_different_vertex_counts_not_isomorphic(self):
        triangle_cycle = [[0, 1], [1, 2], [0, 2]]                   # S^1 on 3 verts
        square_cycle = [[0, 1], [1, 2], [2, 3], [0, 3]]             # S^1 on 4 verts
        self.assertFalse(cobordism.Cobordism.areIsomorphic(triangle_cycle, square_cycle))

    def test_sphere_not_isomorphic_to_torus(self):
        s2 = self._boundary_of(tessera.SolidSimplex(3))             # S^2
        torus = _product(tessera.SimplexBoundarySphere(1),
                         tessera.SimplexBoundarySphere(1))
        # Hold the Spacetime in a variable: the Simplex handles from
        # getSimplices() point into its storage, so it must outlive their use.
        torus_st = _build(torus)
        torus_tops = [list(sorted(v.getId() for v in s.getVertices()))
                      for s in torus_st.getSimplices()
                      if len(s.getVertices()) == 3]
        self.assertFalse(cobordism.Cobordism.areIsomorphic(s2, torus_tops))

    def test_same_manifold_two_constructions_isomorphic(self):
        # The boundary of Delta^4 and the minimal S^3 are both the 5-vertex S^3.
        boundary_of_ball = self._boundary_of(tessera.SolidSimplex(4))
        minimal_s3 = [list(t) for t in (
            (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4))]
        self.assertTrue(cobordism.Cobordism.areIsomorphic(boundary_of_ball, minimal_s3))

    def test_one_empty_one_nonempty_not_isomorphic(self):
        self.assertFalse(cobordism.Cobordism.areIsomorphic([], [[0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
