# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Register-specific spin via the deficit-angle (spin-connection) Wilson loop (#477).

* fast — the **spin-½ double cover**: the spinor Wilson loop is exactly `√` the vector-rep
  deficit-angle Wilson loop (`cos(ε/2)` vs `cos²(ε/2)`), the `ε/2` half-angle, on any host.
* `@slow` — the register-specific read: the spin holonomy is well-defined on the
  register-boundary hinges and distinct from the bulk, frame-independently.
"""
import cmath
import importlib.util
import math
import os
import sys
import unittest

import pytest

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")
_W = cmath.exp(2j * math.pi / 3)


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SpinWilsonHalfAngleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.sw = _load("dk_spin_wilson")

    def test_spinor_wilson_is_sqrt_of_vector_the_double_cover(self):
        # the spin-½ double cover holds on any host: spinor Wilson loop = √(vector)
        host = self.eo.build_closed_s4(n_refine=14, seed=0)
        self.assertLess(self.sw.half_angle_residual(host), 1e-12)


@pytest.mark.slow
class RegisterSpinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.sw = _load("dk_spin_wilson")

    def test_register_carries_a_well_defined_spin_holonomy(self):
        eo, sw = self.eo, self.sw
        for s in range(3, 40):
            host = eo.build_closed_s4(n_refine=20, seed=s % 997)
            opt = eo.EmergentOptimizer(
                host, [[1.0, _W, _W * _W], [1.0, _W * _W, _W]], [1.0, _W, _W * _W],
                degrees=[3], gamma=1.0, seed=s)
            seeds = [v.getId() for v in host.getVertexList().toVector()][:2]
            opt.construct_inputs(seeds, rounds=12)
            opt.run_stage1(max_steps=30, n_candidates=8, patience=8)
            holes = eo.emergent_holes(opt.st, 3)
            if len(holes) >= 3:
                break

        rep = sw.register_spin(opt.st, holes)
        # the register boundary has hinges, and the spin-½ Wilson loop is well-defined
        # (a magnitude in [0,1]) on both register and bulk
        self.assertGreater(rep["register_hinges"], 0)
        self.assertGreater(rep["bulk_hinges"], 0)
        for key in ("register_spinor_W", "bulk_spinor_W"):
            self.assertGreaterEqual(rep[key], 0.0)
            self.assertLessEqual(rep[key], 1.0 + 1e-9)
        # frame-independent register-specific signal: the register holonomy is distinct
        # from the bulk (it concentrates curvature, so the ratio departs from 1)
        self.assertNotAlmostEqual(rep["ratio"], 1.0, places=1)
        # and the double cover still holds exactly on the converged geometry
        self.assertLess(sw.half_angle_residual(opt.st), 1e-12)


if __name__ == "__main__":
    unittest.main()
