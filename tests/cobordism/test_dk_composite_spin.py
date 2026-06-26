# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Composite proton spin — building blocks (#485).

Fast: the spinor holonomy element `exp(ε·Σ)` is a genuine `Spin` group element whose
eigenvalue phases are `±ε/2` (the spin-½ double cover, not the vector `±ε`). This is the
transport building block for the inter-hole Wilson line; the full composite `J²` readout
(per-hole spin operators + the line + the spin–spin correlations) is the rest of #485.
"""
import importlib.util
import math
import os
import sys
import unittest

import numpy as np

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class CompositeSpinTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.sr = _load("dk_spin_readout")
        cls.cs = _load("dk_composite_spin")

    def test_spinor_holonomy_is_the_spin_half_double_cover(self):
        host = self.eo.build_closed_s4(n_refine=12, seed=0)
        dk = self.eo.cob.DiracKahler(host)
        sigma = self.sr.spin_generators(dk)[(1, 2)]      # a Σ_ij spin generator
        for eps in (0.3, 1.0, 2.0, math.pi / 2):
            phases = self.cs.holonomy_phases(eps, sigma)
            # eigenvalue phases are ±ε/2 (half-angle), not ±ε
            self.assertTrue(np.allclose(sorted(phases), sorted([-eps / 2, eps / 2]),
                                        atol=1e-9))
            self.assertTrue(self.cs.is_double_cover(eps, sigma))


if __name__ == "__main__":
    unittest.main()
