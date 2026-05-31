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
    signature = tessera.Signature(4, tessera.Lorentzian)
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


if __name__ == "__main__":
    unittest.main()
