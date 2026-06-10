"""Circumcentric dual cell volumes on Simplex (#246).

Intrinsic, signature-aware circumcenters / dual volumes from the edge lengths
(Cayley-Menger / Gram), verified on the equilateral triangle where everything
is known in closed form:
  * circumradius²  R² = a²/3   (here a² = 1 → 1/3);
  * circumcenter = centroid (barycentric 1/3, 1/3, 1/3);
  * edge circumradius² = a²/4;
  * each vertex's circumcentric dual area = area/3, and the three sum to the
    triangle area (the circumcentric subdivision partitions the cell);
  * top cell's dual is a point (content 1); Hodge star ⋆ = |★σ|/|σ|.

Continuous/spectral only; Lorentzian (signed) content, not Euclidean.
"""

from __future__ import annotations

import math
import unittest

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _triangle(st):
    for s in st.getSimplices():
        if len(s.getEdges()) == 3:
            return s
    raise AssertionError("no triangle simplex in spacetime")


def _set_equilateral(st, a2=1.0):
    for e in _edge_map(st).values():
        e.setSquaredLength(a2)
        e.setPhase(0.0)


class TestCircumcenter(unittest.TestCase):
    def test_equilateral_circumradius_squared(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        self.assertAlmostEqual(_triangle(st).circumradiusSquared(),
                               1.0 / 3.0, places=10)

    def test_equilateral_circumcenter_is_centroid(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        for b in _triangle(st).circumcenterBarycentric():
            self.assertAlmostEqual(b, 1.0 / 3.0, places=10)

    def test_edge_circumradius_squared(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        edge_simplex = _triangle(st).getFacets()[0]  # a 1-simplex
        self.assertAlmostEqual(edge_simplex.circumradiusSquared(), 0.25, places=10)

    def test_timelike_edge_carries_signed_content(self):
        # A timelike edge (squaredLength < 0) yields a signed circumradius² —
        # no crash, no Euclidean |.|.
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        em = _edge_map(st)
        em[(1, 2)].setSquaredLength(-1.0)
        edge_simplex = None
        for e in _triangle(st).getFacets():
            ids = sorted(v.getId() for v in e.getVertices())
            if ids == [1, 2]:
                edge_simplex = e
        self.assertIsNotNone(edge_simplex)
        self.assertAlmostEqual(edge_simplex.circumradiusSquared(), -0.25, places=10)


class TestDualVolume(unittest.TestCase):
    def test_top_cell_dual_is_a_point(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        self.assertAlmostEqual(_triangle(st).dualVolume(), 1.0, places=10)

    def test_hodge_star_top_cell(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        tri = _triangle(st)
        self.assertAlmostEqual(tri.hodgeStar(), 1.0 / tri.volume(), places=10)

    def test_vertex_dual_partitions_the_area(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        tri = _triangle(st)
        area = tri.volume()  # sqrt(3)/4 for a = 1
        # Materialize the facet/coface skeleton: triangle -> edges -> vertices.
        verts = {}
        for e in tri.getFacets():
            for v in e.getFacets():
                key = tuple(x.getId() for x in v.getVertices())
                verts[key] = v
        self.assertEqual(len(verts), 3)
        duals = [v.dualVolume() for v in verts.values()]
        for d in duals:
            self.assertAlmostEqual(d, area / 3.0, places=8)
        self.assertAlmostEqual(sum(duals), area, places=8)


if __name__ == "__main__":
    unittest.main()
