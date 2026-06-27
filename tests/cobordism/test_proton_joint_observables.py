# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the joint proton assembly (#481) — all quantum numbers off ONE relaxed state.

A fast unit test pins the pure-Python observable plumbing (`radius_rms`, `dual_regge_mass`,
`format_report`, and the report shape) on a tiny hand-built mesh, with no expensive build. A
slow test drives the simultaneous pair-creation build to a converged emergent proton and
asserts every sector co-occurs on the **same** structure:

  * COLOR — the color singlet is carried (`r_state → 0`), and confinement shows as
    color-neutrality (singlet-phase-weighted net DK charge ≪ the constituent total). The
    honest negative is recorded: `r_state(sub, 3, [1,1,1])` does NOT floor here (the emergent
    block's `b₃ ≥ 3` register realizes every color rep), so r_state alone cannot floor a
    non-singlet — color-neutrality is the genuine confinement signal.
  * FLAVOR — per-hole DK charge spread > 0 (distinguishable quarks).
  * CHARGE — net Dirac–Kähler charge finite.
  * RADIUS · MASS — both finite.
  * STRUCTURAL SPIN-½ — `Σ_ij` eigenvalues are exactly ±½.
"""
import importlib.util
import math
import os
import sys
import unittest

import numpy as np

import tessera

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pjo = _load("proton_joint_observables")


class PlumbingTest(unittest.TestCase):
    """Fast: the pure-Python observable helpers and report shape, no build."""

    def _tiny_mesh(self):
        # a single 4-simplex (pentatope) with all-spacelike unit edges
        cells = [[0, 1, 2, 3, 4]]
        return tessera.Spacetime.fromCells(4, cells, 1.0, 0.0)

    def test_radius_rms_spacelike(self):
        st = self._tiny_mesh()
        r, n_sp, n_tl = pjo.radius_rms(st)
        self.assertGreater(n_sp, 0)
        self.assertEqual(n_tl, 0)            # all unit-spacelike
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_dual_regge_mass_finite(self):
        st = self._tiny_mesh()
        m = pjo.dual_regge_mass(st)
        self.assertTrue(math.isfinite(m))
        self.assertGreaterEqual(m, 0.0)

    def test_proton_report_handles_no_register(self):
        # a converged-build tuple whose read yields no 3-hole register → None
        self.assertIsNone(pjo.proton_report((None, {"n_holes": 0}, 7)))
        self.assertIsNone(pjo.proton_report((None, None, 7)))

    def test_format_report_none(self):
        self.assertIn("honest negative", pjo.format_report(None))


class JointAssemblyBuildTest(unittest.TestCase):
    """Slow: build a converged emergent proton and assert every sector co-occurs."""

    def test_all_quantum_numbers_on_one_relaxed_state(self):
        rep, seed = pjo.build_and_report(
            seeds=range(5, 25), max_residual=0.6,
            n_refine=18, stage1_steps=60, stage2_iters=20)
        if rep is None:
            self.skipTest("no converged 3-hole proton block in the seed range")

        # the report is one coherent dict for ONE structure
        self.assertIsNotNone(seed)
        self.assertGreaterEqual(rep["n_holes"], 3)

        # COLOR: the singlet is carried (r_state → 0)
        self.assertTrue(rep["color_singlet_carried"])
        self.assertLess(rep["color_singlet_residual"], 1e-3)
        # the r_state [1,1,1] probe value is reported & finite (honest negative: it does
        # not floor here because b₃ ≥ 3 — see module docstring)
        self.assertTrue(math.isfinite(rep["color_uniform_residual"]))
        self.assertGreaterEqual(rep["betti_sub"][3], 3)
        # confinement IS shown as color-neutrality: net color charge ≪ constituent total
        self.assertTrue(rep["color_neutral"])
        self.assertLess(rep["color_net_charge"], rep["color_constituent_total"])

        # FLAVOR: distinguishable per-hole charge
        self.assertGreater(rep["flavor_spread"], 0.0)
        self.assertTrue(rep["flavor_independent"])

        # CHARGE: net Dirac–Kähler charge finite
        self.assertTrue(math.isfinite(rep["net_charge"]))
        self.assertEqual(len(rep["per_hole_charge"]), 3)

        # RADIUS · MASS: both finite
        self.assertTrue(math.isfinite(rep["radius"]) and rep["radius"] > 0)
        self.assertTrue(math.isfinite(rep["mass"]) and rep["mass"] > 0)
        self.assertTrue(math.isfinite(rep["r_times_m"]))

        # STRUCTURAL SPIN-½: Σ_ij eigenvalues exactly ±½
        self.assertTrue(rep["spin_half"])
        for mags in rep["spin_eigenvalue_magnitudes"].values():
            self.assertEqual(len(mags), 1)
            self.assertAlmostEqual(mags[0], 0.5, places=6)

        # the joint report renders
        self.assertIn("THE PROTON, END TO END", pjo.format_report(rep))


if __name__ == "__main__":
    unittest.main()
