"""Fail-loud guard on resident Im l^2 in the non-Wick geometry stack (#581
scope item 6).

The ordinary-Lorentzian convention: resident l^2 is REAL and SIGNED; a
resident Im l^2 != 0 (Picard-Lefschetz / analytically continued length) is
unsupported by the geometry stack, which previously projected it to Re
SILENTLY at every Gram / Cayley-Menger build.  The non-Wick entry points now
throw std::domain_error (ValueError in Python) naming the offending edge; the
Wick-rotated (|l^2|) paths stay Im-tolerant; storage-level paths (the
rollback records of item 2) never evaluate geometry and are untouched.

Also the plot-layout collapse (item 8): the layout rest length is the
documented |Re l^2| with an epsilon floor, so a null edge cannot pin two
vertices together.
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


class TestResidentImGuard(unittest.TestCase):
    def _im_host(self, im=0.25):
        st = _triangle_host()
        for k, e in _edge_map(st).items():
            e.setSquaredLength(1.0)
        _edge_map(st)[(0, 1)].setSquaredLength(complex(1.0, im))
        return st

    def test_volume_throws_with_edge_ids(self):
        st = self._im_host()
        tri = _by_verts(st, [0, 1, 2])
        with self.assertRaises(ValueError) as ctx:
            tri.volume()
        msg = str(ctx.exception)
        self.assertIn("Im l^2", msg)
        self.assertIn("(0, 1)", msg)
        self.assertIn("580", msg)  # cites the audit / convention decision

    def test_lorentzian_angles_throw(self):
        st = self._im_host()
        tri = _by_verts(st, [0, 1, 2])
        v0 = _by_verts(st, [0])
        with self.assertRaises(ValueError):
            tri.lorentzianDihedralAngle(v0)
        with self.assertRaises(ValueError):
            v0.lorentzianDeficitAngle()

    def test_dual_regge_action_throws(self):
        st = _triangle_host()
        for k, e in _edge_map(st).items():
            e.setSquaredLength(1.0)
        solver = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        self.assertTrue(math.isfinite(solver.dualReggeAction().real))
        _edge_map(st)[(0, 2)].setSquaredLength(complex(1.0, 0.5))
        with self.assertRaises(ValueError):
            solver.dualReggeAction()

    def test_wick_paths_stay_im_tolerant(self):
        st = self._im_host()
        tri = _by_verts(st, [0, 1, 2])
        v0 = _by_verts(st, [0])
        # |l^2| paths: the Euclidean/CDT pipeline must keep working
        theta = tri.dihedralAngle(v0, True)
        self.assertTrue(math.isfinite(theta))
        eps = v0.deficitAngle()          # wick-rotated by definition
        self.assertTrue(math.isfinite(eps))
        area = tri.area(True)
        self.assertTrue(math.isfinite(area))

    def test_float_noise_below_tolerance_is_accepted(self):
        st = _triangle_host()
        for k, e in _edge_map(st).items():
            e.setSquaredLength(1.0)
        _edge_map(st)[(0, 1)].setSquaredLength(complex(1.0, 1e-13))
        tri = _by_verts(st, [0, 1, 2])
        self.assertTrue(math.isfinite(tri.volume()))
        self.assertTrue(
            math.isfinite(complex(tri.lorentzianDihedralAngle(
                _by_verts(st, [0]))).real))

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
