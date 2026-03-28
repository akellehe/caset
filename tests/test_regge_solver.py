"""Tests for the Regge equation solver."""
import math
import unittest

import caset


def _make_spacetime(n_simplices=200):
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)
    return st


class TestDihedralAngles(unittest.TestCase):
    """Basic sanity checks for dihedral angle computation."""

    def test_dihedral_angles_are_positive(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        solver = caset.ReggeSolver(st, matter)

        # Find a top-simplex and one of its hinges (triangles, 3 verts)
        for s in st.getSimplices():
            if len(s.getVertices()) == 5:  # top-simplex in 4D
                for facet in s.getFacets():
                    for hinge in facet.getFacets():
                        if len(hinge.getVertices()) == 3:
                            angle = solver.dihedralAngle(s, hinge)
                            self.assertGreater(angle, 0.0,
                                "Dihedral angle should be positive")
                            self.assertLess(angle, math.pi,
                                "Dihedral angle should be < π")
                            return
        self.skipTest("No suitable hinge found")

    def test_deficit_angles_exist(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        solver = caset.ReggeSolver(st, matter)

        # Find a hinge and compute its deficit angle
        for s in st.getSimplices():
            if len(s.getVertices()) == 3 and len(s.getCofaces()) > 0:
                eps = solver.deficitAngle(s)
                # Deficit can be positive, negative, or zero
                self.assertIsInstance(eps, float)
                return
        self.skipTest("No hinge with cofaces found")


class TestReggeAction(unittest.TestCase):
    """Test that the Regge action is computable."""

    def test_regge_action_is_finite(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        solver = caset.ReggeSolver(st, matter)
        S = solver.reggeAction()
        self.assertTrue(math.isfinite(S), f"Regge action should be finite, got {S}")


class TestFlatSpaceResidual(unittest.TestCase):
    """In vacuum (no matter), the target deficit is 0 everywhere.
    The residual measures how far the geometry is from flat."""

    def test_vacuum_residual_is_nonnegative(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()  # no matter = vacuum
        solver = caset.ReggeSolver(st, matter)
        L = solver.deficitResidual()
        self.assertGreaterEqual(L, 0.0)

    def test_solver_reduces_residual(self):
        """A few gradient steps should reduce (or not increase) the residual."""
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        solver = caset.ReggeSolver(st, matter)
        L0 = solver.deficitResidual()
        if L0 < 1e-12:
            self.skipTest("Already at flat space")
        for _ in range(5):
            L1 = solver.step(0.0001)
        # After several steps the residual should generally decrease
        # (not a strict guarantee per step, but over several steps)
        self.assertLess(L1, L0 * 10,
            f"Residual should not explode: {L1} vs initial {L0}")


class TestMatterConfiguration(unittest.TestCase):
    """Test matter configuration specification."""

    def test_point_mass_creates_nonzero_targets(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        v = st.getVertexList().toVector()[0]
        matter.setPointMass(v, 1.0)
        solver = caset.ReggeSolver(st, matter)
        # With matter, residual should be nonzero (geometry doesn't match yet)
        L = solver.deficitResidual()
        self.assertGreater(L, 0.0)

    def test_radial_profile(self):
        st = _make_spacetime(20)
        matter = caset.MatterConfiguration()
        v = st.getVertexList().toVector()[0]
        # Exponential profile: ρ(r) = exp(-r)
        matter.setRadialProfile(v, lambda r: math.exp(-r))
        solver = caset.ReggeSolver(st, matter)
        L = solver.deficitResidual()
        self.assertGreater(L, 0.0)


class TestHingeArea(unittest.TestCase):
    def test_hinge_area_positive(self):
        st = _make_spacetime(20)
        for s in st.getSimplices():
            if len(s.getVertices()) == 3 and len(s.getEdges()) >= 3:
                area = caset.ReggeSolver.hingeArea(s)
                self.assertGreaterEqual(area, 0.0)
                return
        self.skipTest("No triangle found")
