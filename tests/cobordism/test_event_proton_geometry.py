"""Falsifiable geometric proton check on the Experiment-B emergent interior (#451).

Pins the geometry read off the RELAXED emergent worldtube (not the frozen Dirichlet
top slice) at the carriable depth nL = 2. The bands below are PRE-REGISTERED from the
findings report (`docs/design/451-proton-geometry-0a1a5aa.md`) and are NEVER loosened
to manufacture green: the relaxation is fully deterministic (symmetric A4 windows +
uniform seed), so every measured number is identical across seeds and the bands are
tight tolerances around the reported values, not wide guesses.

What is asserted (the robust, definition-independent claims):
  * the interior set is exactly the 264 closed-coface-fan hinges (a topological count);
  * it is the proton sector (top-slice singlet -> 1, color sigma -> 0);
  * the relaxation is at the nL=2 carriable floor (residual < 100, matching A's ~71);
  * the dual radius is GENUINELY EMERGENT (~5, an order above the frozen ~1.29);
  * the curvature is a LOCALIZED, NET-POSITIVE lump (mean Re(deficit) > 0;
    participation ratio well below the round sphere's 1.0; ~all of it near the quarks);
  * the task's literal r*m is the same O(1) as the physical 4.0.

r*m itself is deliberately bounded LOOSELY (it is definition-sensitive at nL=2, per the
report); the sharp guards are the localization + positive curvature + emergent radius.
"""

import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples", "cobordism"))
import event_proton_geometry as G  # noqa: E402


@pytest.mark.slow
class EventProtonGeometryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.o = G.measure(n_layers=2, max_iters=200)

    # -- it is the proton sector, at the carriable floor --
    def test_proton_sector(self):
        self.assertGreaterEqual(self.o["singlet"], 0.95)
        self.assertLessEqual(self.o["sigma"], 0.05)

    def test_carriable_floor(self):
        # nL=2 ||grad S||^2 plateau, matching Experiment A's ~71 (extensive in depth).
        self.assertLess(self.o["residual"], 100.0)

    def test_interior_hinge_count(self):
        # Topological: the closed-coface-fan edges off the two frozen Dirichlet slices.
        self.assertEqual(self.o["n_interior_hinges"], 264)

    # -- the dual radius is genuinely emergent (not the frozen ~1.29) --
    def test_emergent_dual_radius(self):
        r = self.o["radius"]["r_Vdual"]
        self.assertGreater(r, 4.0)   # an order above the frozen prior (1.29)
        self.assertLess(r, 6.5)
        # primal/dual agreement (the dual volume is not an artefact)
        self.assertLess(abs(self.o["radius"]["r_V3"] - r) / r, 0.25)

    # -- the curvature is a localized, net-positive (sphere-like-sign) lump --
    def test_curvature_is_positive_lump(self):
        c = self.o["curvature"]
        self.assertGreater(c["mean_re"], 0.0)        # positive curvature
        self.assertLess(c["PR"], 0.55)               # well below the round sphere's 1.0
        self.assertGreater(c["PR"], 0.30)            # not a single spike either
        self.assertGreater(c["concentration_ratio"], 1.8)  # vs equal-volume sphere

    def test_curvature_localized_near_quarks(self):
        loc = self.o["localization"]
        self.assertGreater(loc["fraction_within_shell1"], 0.90)
        self.assertLess(loc["rms_shell_radius"], 1.3)

    # -- r*m: same order as the physical 4.0, below the boundary-polluted prior 8.8 --
    def test_rm_order_of_magnitude(self):
        # LOOSE on purpose: r*m is definition-sensitive at nL=2 (see the report). The
        # claim is order-of-magnitude agreement, not a hit on 4.0.
        self.assertGreater(self.o["r_m"], 0.5)
        self.assertLess(self.o["r_m"], 10.0)
        # the boundary-pollution correction must move it BELOW the prior whole-tube 8.8
        self.assertLess(self.o["r_m"], self.o["r_m_prior"])


if __name__ == "__main__":
    unittest.main()
