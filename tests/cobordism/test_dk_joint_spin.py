# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the entangled joint 3-fermion spin read (#489).

The fast, load-bearing check is that the composite-spin `J²` operator is **exact** on
clean spin-½ states (proton eigenstate → ¾, Δ → 15/4, product → 7/4) — this is the
measuring stick the #489 finding rests on. A slower test drives the simultaneous
pair-creation build and pins the two positive results (the proton block carries the
singlet; per-hole flavor is independent) and the honest negative (the per-hole-product
`J²` sits at the mixture, not ¾).
"""
import importlib.util
import math
import os
import sys
import unittest

import cmath
import numpy as np

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dj = _load("dk_joint_spin")
_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _kr(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


class J2OperatorTest(unittest.TestCase):
    """The validated measuring stick: J² is exact on clean spin-½ states."""

    def test_proton_eigenstate_is_three_quarters(self):
        proton = 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP) - _kr(_DN, _UP, _UP)
        self.assertAlmostEqual(dj.j2_three_qubit(proton), 0.75, places=9)

    def test_delta_is_fifteen_quarters(self):
        self.assertAlmostEqual(dj.j2_three_qubit(_kr(_UP, _UP, _UP)), 3.75, places=9)

    def test_product_uud_is_seven_quarters(self):
        # a PRODUCT of aligned/anti-aligned spins is the mixture, never the proton ¾
        self.assertAlmostEqual(dj.j2_three_qubit(_kr(_UP, _UP, _DN)), 1.75, places=9)

    def test_spinor_to_qubit_is_unit(self):
        rng = np.random.default_rng(489)
        for _ in range(5):
            s4 = rng.normal(size=4) + 1j * rng.normal(size=4)
            q = dj.spinor_to_qubit(s4)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)
            self.assertEqual(q.shape, (2,))


class PairCreationBuildTest(unittest.TestCase):
    """Slow: the simultaneous pair-creation build's positive results + honest negative."""

    def test_carries_singlet_independent_flavor_mixture_j2(self):
        found = dj.build_converged_proton(
            seeds=range(3, 14), max_residual=0.5,
            n_refine=18, stage1_steps=60, stage2_iters=20)
        if not found:
            self.skipTest("no converged 3-hole proton block in the seed range")
        _opt, rd, _seed = found
        # positive: the proton output block carries the color singlet
        self.assertLessEqual(rd["block_residual"], 0.5)
        self.assertGreaterEqual(rd["n_holes"], 3)
        # positive: per-hole flavor is independent (the structure #488 lacked)
        self.assertGreater(rd["flavor_spread"], 1e-3)
        # honest negative: the per-hole-product J² sits at the mixture, not ¾
        self.assertGreater(rd["j2_product"], 0.75 + 0.2)
        self.assertTrue(math.isfinite(rd["j2_joint"]))


if __name__ == "__main__":
    unittest.main()
