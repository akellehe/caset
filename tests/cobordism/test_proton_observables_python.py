# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The proton's emergent observables off the relaxed W_ABC singlet (#400).

Exercises ``examples/cobordism/proton_observables.py``: feed the natural
color-symmetric (omega-representation) quark input to the symmetric junction, relax,
and read the proton's observables OFF the relaxed geometry (emergent-first):

  * color charge sigma -> 0 (confinement: the bound state is color-neutral) while the
    singlet component -> 1 (the proton EMERGES; it is never pinned);
  * a finite positive radius r = sqrt(mean(l^2 > 0));
  * two mass handles -- A = |dual_action| (cross-check), B = shell-summed Re-deficit
    (the #352 method, the r*m anchor) -- and the dimensionless r*m for each.
"""

import importlib.util
import pathlib
import unittest

import pytest

_EXAMPLE = (pathlib.Path(__file__).resolve().parents[2]
            / "examples" / "cobordism" / "proton_observables.py")
_spec = importlib.util.spec_from_file_location("proton_observables", _EXAMPLE)
_obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_obs)


@pytest.mark.slow
class ProtonObservablesTest(unittest.TestCase):
    """All observables are read off ONE relaxed geometry; a short relax keeps the
    test quick (the reading interface is what is under test, not the relax depth)."""

    @classmethod
    def setUpClass(cls):
        cls.o = _obs.measure(max_iters=25)

    def test_color_charge_is_confined(self):
        # sigma = projection onto the g-invariant (+1) charge mode; singlet => 0.
        self.assertLess(self.o["sigma"], 0.15)

    def test_singlet_emerges_not_pinned(self):
        # The proton: the omega-mode component of the emergent result. The result is
        # never pinned -- it EMERGES from the symmetric quark input via the transport.
        self.assertGreater(self.o["singlet"], 0.99)

    def test_radius_is_finite_positive(self):
        r = self.o["radius"]
        self.assertGreater(r, 0.0)
        self.assertTrue(r == r and r < float("inf"))  # not NaN/inf
        self.assertEqual(self.o["n_timelike"], 0)     # all-spacelike (Riemannian seed)

    def test_both_mass_handles_are_finite(self):
        self.assertGreater(self.o["mass_a"], 0.0)     # |dual_action| (cross-check)
        self.assertGreater(self.o["mass_b"], 0.0)     # shell-deficit (anchor)

    def test_rm_anchored_on_B_is_order_unity_scale(self):
        # r*m on the B (shell-deficit) anchor -- the figure compared to the ~4.0
        # target. A loose bound guards against regressions while allowing the value
        # to move with the relax (prior crude figure was ~73; here it is single digits).
        rm_b = self.o["rm_b"]
        self.assertGreater(rm_b, 0.0)
        self.assertLess(rm_b, 50.0)
        self.assertAlmostEqual(rm_b, self.o["radius"] * self.o["mass_b"], places=9)

    def test_a_is_a_distinct_cross_check(self):
        # The two mass handles are genuinely different scales (A global, B shell-local).
        self.assertGreater(self.o["rm_a"], self.o["rm_b"])

    def test_relaxation_ran(self):
        self.assertGreater(self.o["relax_iters"], 0)


if __name__ == "__main__":
    unittest.main()
