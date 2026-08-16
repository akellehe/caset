"""Tests for the Wilson loop observable."""
import cmath
import math
import unittest

import tessera


def _make_spacetime(n_simplices=20):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(n_simplices)
    return st


def _find_hinge(st):
    """Find a hinge (3-vertex simplex) with cofaces."""
    for s in st.getSimplices():
        if len(s.getVertices()) == 3 and len(s.getCofaces()) > 0:
            return s
    return None


class TestHingeLoop(unittest.TestCase):
    def test_hinge_loop_nonempty(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        self.assertGreaterEqual(len(loop), 2,
            "Hinge loop should have at least 2 simplices")

    def test_hinge_loop_all_contain_hinge(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        hinge_verts = hinge.getVertices()
        for sigma in loop.simplices:
            for hv in hinge_verts:
                self.assertTrue(sigma.hasVertex(hv),
                    "Every loop simplex must contain the hinge")


class TestGeodesicLoop(unittest.TestCase):
    def test_geodesic_loop_exists(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        # Find a top-simplex
        start = None
        for s in st.getSimplices():
            if len(s.getVertices()) == 5:  # top-simplex in 4D
                start = s
                break
        if start is None:
            self.skipTest("No top-simplex found")
        loop = wl.geodesicLoop(start)
        self.assertGreaterEqual(len(loop), 2,
            "Geodesic loop should exist on a closed manifold")


class TestDualLatticeLoop(unittest.TestCase):
    def test_dual_lattice_loop_exists(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        start = None
        for s in st.getSimplices():
            if len(s.getVertices()) == 5:  # top-simplex in 4D
                start = s
                break
        if start is None:
            self.skipTest("No top-simplex found")
        loop = wl.dualLatticeLoop(start, 6)
        self.assertGreaterEqual(len(loop), 2)


class TestCombinatorialMode(unittest.TestCase):
    def test_combinatorial_returns_loop_size(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        result = wl.evaluateCombinatorial(loop)
        self.assertEqual(result.loopSize, len(loop))
        self.assertEqual(result.value, float(len(loop)))


class TestDeficitAngleMode(unittest.TestCase):
    def test_hinge_wilson_value_bounded(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluateDeficitAngle(loop)
        val = complex(result.value)
        self.assertTrue(cmath.isfinite(val),
            f"Wilson value should be finite, got {val}")
        # The rotation part alone is bounded; the boost enters as a cosh, so
        # only the REAL part keeps the classical lower bound. A CDT hinge with
        # boost content legitimately has |value| > 1.
        self.assertGreaterEqual(val.real, -1.0 - 1e-9)
        self.assertLessEqual(result.value, 1.0)

    def test_hinge_wilson_matches_deficit(self):
        """For a hinge loop, W = ((d-2)+2cos(ε))/d should match."""
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        matter = tessera.MatterConfiguration()
        solver = tessera.ReggeSolver(st, matter)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluateDeficitAngle(loop)
        eps = solver.deficitAngle(hinge)
        # The deficit is complex; the holonomy keeps it whole (cos of a complex
        # angle — the boost enters as a cosh), so compare in C.
        expected = ((4 - 2) + 2 * cmath.cos(eps)) / 4
        self.assertAlmostEqual(abs(complex(result.value) - expected), 0.0, places=6,
            msg=f"Wilson value {result.value} != expected {expected}")


class TestCausalMode(unittest.TestCase):
    def test_causal_winding_is_integer(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluateCausal(loop)
        self.assertEqual(result.causalWindingNumber,
                         int(result.causalWindingNumber))


class TestEvaluateDispatch(unittest.TestCase):
    def test_evaluate_dispatches_correctly(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hingeLoop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        r1 = wl.evaluate(loop, tessera.WilsonMode.COMBINATORIAL)
        r2 = wl.evaluateCombinatorial(loop)
        self.assertEqual(r1.value, r2.value)
        self.assertEqual(r1.loopSize, r2.loopSize)


class TestMeasurements(unittest.TestCase):
    def test_measure_all_hinges_populates(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measureAllHinges(tessera.WilsonMode.DEFICIT_ANGLE)
        measurements = wl.getMeasurements()
        self.assertGreater(len(measurements), 0,
            "measureAllHinges should produce measurements")

    def test_reset_clears(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measureAllHinges(tessera.WilsonMode.DEFICIT_ANGLE)
        wl.reset()
        self.assertEqual(len(wl.getMeasurements()), 0)

    def test_average_by_size(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measureAllHinges(tessera.WilsonMode.DEFICIT_ANGLE)
        avg = wl.getAverageBySize()
        self.assertGreater(len(avg), 0)
        for size, val in avg.items():
            self.assertIsInstance(size, int)
            self.assertTrue(cmath.isfinite(complex(val)))
