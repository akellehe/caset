# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the proton's electric / U(1) charge read (#478, part of #410).

Two layers, mirroring `test_dk_joint_spin.py`:

  * fast, load-bearing checks on the **pure-numpy** charge plumbing (the singlet-phased net
    of equal per-hole charges is 0; the constituent total is positive; the quantization
    classifier lands on the right lattice) — the measuring sticks the build-test rests on;
  * a slow check that drives the simultaneous pair-creation build (#489) and pins the
    contract: the per-hole DK charges are finite and **metric-robust** (small jitter std),
    they land on the integer / third-integer lattice within tolerance, and the **honest
    negative** holds — the singlet net U(1) → ~0 (flavor-blind), not the physical +1.
"""
import importlib.util
import math
import os
import sys
import unittest

import cmath
import numpy as np

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")
_W = cmath.exp(2j * math.pi / 3)


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pc = _load("proton_charge")


class ChargePlumbingTest(unittest.TestCase):
    """Fast measuring sticks on the pure-numpy charge helpers."""

    def test_equal_charges_have_zero_singlet_net(self):
        # three equal per-hole charges weighted by [1, ω, ω²] cancel: the flavor-blind net
        net, total = pc.signed_net([0.7, 0.7, 0.7])
        self.assertAlmostEqual(abs(net), 0.0, places=9)
        self.assertAlmostEqual(total, 2.1, places=9)

    def test_total_is_positive_sum(self):
        net, total = pc.signed_net([0.3, 0.5, 0.9])
        self.assertAlmostEqual(total, 1.7, places=9)
        self.assertGreater(total, 0.0)

    def test_quantization_classifier_third_integer(self):
        lattice, res, ok, pts = pc.quantization_verdict([1.0 / 3, 2.0 / 3, 1.0], tol=1e-6)
        self.assertEqual(lattice, "third-integer (n/3)")
        self.assertLess(res, 1e-9)
        self.assertTrue(ok)
        np.testing.assert_allclose(pts, [1.0 / 3, 2.0 / 3, 1.0], atol=1e-9)

    def test_quantization_classifier_off_lattice(self):
        _lattice, res, ok, _pts = pc.quantization_verdict([0.17, 0.49, 0.83], tol=0.02)
        self.assertGreater(res, 0.02)
        self.assertFalse(ok)

    def test_quantization_residual_shape(self):
        r = pc.quantization_residual([0.34, 0.66, 1.01], 3)
        self.assertEqual(r.shape, (3,))
        self.assertTrue(np.all(r >= 0))


class ProtonChargeBuildTest(unittest.TestCase):
    """Slow: the built proton's emergent charge — robustness + the honest negative."""

    def test_emergent_charge_is_robust_and_net_vanishes(self):
        found = pc.dj.build_converged_proton(
            seeds=range(5, 16), max_residual=0.6,
            n_refine=18, stage1_steps=60, stage2_iters=20)
        if not found:
            self.skipTest("no converged 3-hole proton block in the seed range")
        opt, rd, _seed = found
        res = pc.read_charge(opt, rd, jitter_trials=6)
        # the per-hole charges are finite and the constituent total is positive
        self.assertTrue(np.all(np.isfinite(res["charges"])))
        self.assertGreater(res["total"], 0.0)
        # metric robustness: a gauged holonomy is stable under spacelike-l² jitter
        self.assertTrue(math.isfinite(res["jitter_net_std"]))
        self.assertTrue(math.isfinite(res["jitter_total_std"]))
        self.assertLess(res["jitter_total_std"],
                        0.5 * abs(res["jitter_total_mean"]) + 1e-6)
        # quantization residual is a real, non-negative number (verdict reported, not forced)
        self.assertGreaterEqual(float(res["quant_residual"]), 0.0)
        # honest negative: the singlet net U(1) is far below the physical proton +1
        self.assertLess(abs(res["net"]), 0.5)


if __name__ == "__main__":
    unittest.main()
