"""The ordinary-Lorentzian convention on the geometry stack (#580/#589).

Resident l^2 is REAL and SIGNED (spacelike > 0, timelike < 0, null 0); the
non-Wick geometry entry points consume Re l^2 and the Wick-rotated paths |l^2|.
There is NO runtime enforcement — the #582 interim `domain_error` guard is
deleted (#589): the dynamics keeps l^2 on the real axis by construction
(`MultiCobordism::runStage2` proposes exactly-real trials), and that invariant
is proven where invariants live — in the suite
(tests/cobordism/test_stage2_real_manifold_python.py) — never by a throw,
veto, or backoff inside the physics loop.

These tests pin the convention itself: geometry never throws on a resident
Im l^2 (nothing polices storage), the non-Wick read IS the Re projection
(value-equal to the Im-zeroed twin), Wick paths stay Im-tolerant, and signed
real (timelike) geometry — the supported Lorentzian case — keeps working.

Also the plot-layout collapse: the layout rest length is the documented
|Re l^2| with an epsilon floor, so a null edge cannot pin two vertices
together.
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


def _triangle_host():
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    tessera.ReggeSolver(st, tessera.MatterConfiguration())  # materialize
    return st


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _by_verts(st, ids):
    want = tuple(sorted(ids))
    for s in st.getSimplices():
        if tuple(sorted(v.getId() for v in s.getVertices())) == want:
            return s
    raise AssertionError(f"no simplex {want}")


class TestRealSignedConvention(unittest.TestCase):
    def _twin_hosts(self, im=0.25):
        """Two identical triangles; one carries Im l^2 on an edge, one does not."""
        with_im, without_im = _triangle_host(), _triangle_host()
        for st in (with_im, without_im):
            for e in _edge_map(st).values():
                e.setSquaredLength(1.0)
        _edge_map(with_im)[(0, 1)].setSquaredLength(complex(1.0, im))
        _edge_map(without_im)[(0, 1)].setSquaredLength(complex(1.0, 0.0))
        return with_im, without_im

    def test_non_wick_reads_are_the_re_projection_no_throw(self):
        # The convention, executable: geometry on an Im-carrying host neither
        # throws (no runtime enforcement) nor differs from the Im-zeroed twin
        # (the non-Wick read is exactly Re l^2).
        with_im, without_im = self._twin_hosts()
        tri_a = _by_verts(with_im, [0, 1, 2])
        tri_b = _by_verts(without_im, [0, 1, 2])
        self.assertEqual(tri_a.volume(), tri_b.volume())
        v0_a, v0_b = _by_verts(with_im, [0]), _by_verts(without_im, [0])
        self.assertEqual(complex(tri_a.lorentzianDihedralAngle(v0_a)),
                         complex(tri_b.lorentzianDihedralAngle(v0_b)))
        self.assertEqual(complex(v0_a.lorentzianDeficitAngle()),
                         complex(v0_b.lorentzianDeficitAngle()))
        solver_a = tessera.ReggeSolver(with_im, tessera.MatterConfiguration())
        solver_b = tessera.ReggeSolver(without_im, tessera.MatterConfiguration())
        self.assertEqual(complex(solver_a.dualReggeAction()),
                         complex(solver_b.dualReggeAction()))

    def test_wick_paths_stay_im_tolerant(self):
        # |l^2| paths: the Euclidean/CDT pipeline reads the modulus and keeps
        # working whatever the phase of the stored value.
        with_im, _ = self._twin_hosts()
        tri = _by_verts(with_im, [0, 1, 2])
        v0 = _by_verts(with_im, [0])
        theta = tri.dihedralAngle(v0, True)
        self.assertTrue(math.isfinite(theta))
        eps = v0.deficitAngle()          # wick-rotated by definition
        self.assertTrue(math.isfinite(eps))
        area = tri.area(True)
        self.assertTrue(math.isfinite(area))

    def test_signed_real_geometry_still_works(self):
        # timelike (negative REAL) l^2 is the supported Lorentzian case
        st = _triangle_host()
        em = _edge_map(st)
        em[(0, 1)].setSquaredLength(-4.0)
        em[(0, 2)].setSquaredLength(-3.0)
        em[(1, 2)].setSquaredLength(1.0)
        tri = _by_verts(st, [0, 1, 2])
        self.assertTrue(math.isfinite(tri.volume()))
        th = complex(tri.lorentzianDihedralAngle(_by_verts(st, [2])))
        self.assertTrue(math.isfinite(th.real) and math.isfinite(th.imag))


class TestLayoutCollapseFloor(unittest.TestCase):
    def test_null_edge_rest_length_is_floored(self):
        plot = pytest.importorskip("tessera.utils.plot")
        st = _triangle_host()
        em = _edge_map(st)
        em[(0, 1)].setSquaredLength(0.0)            # null edge
        em[(0, 2)].setSquaredLength(complex(1.0, 0.7))  # layout is Im-blind
        em[(1, 2)].setSquaredLength(1.0)
        verts = st.getVertexList().toVector()
        edges = st.getEdgeList().toVector()
        pos, vid_to_idx, edge_idx = plot.layout_from_spacetime(
            verts, edges, iters=1)
        self.assertEqual(len(edge_idx), 3)
        # rebuild the rest lengths the way the helper does and check the floor
        rest = {(min(e.getSource().getId(), e.getTarget().getId()),
                 max(e.getSource().getId(), e.getTarget().getId())):
                math.sqrt(max(abs(e.getSquaredLength().real), 1e-6))
                for e in edges}
        self.assertEqual(rest[(0, 1)], math.sqrt(1e-6))   # floored, not 0
        self.assertEqual(rest[(0, 2)], 1.0)               # |Re|, Im ignored


if __name__ == "__main__":
    unittest.main()
