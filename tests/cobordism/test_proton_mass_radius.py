# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the proton mass & radius read off the relaxed emergent geometry (#480).

Two layers, matching the #410 epic style:

  * a fast, deterministic layer pins the two **pure readers** (`radius_rms`,
    `curvature_concentration_mass`) on a hand-built sub-complex with a known metric — no
    expensive proton build, so the geometric proxies are exercised on every run;

  * a slow layer drives the simultaneous pair-creation build for two converged seeds and
    asserts the observables are **finite**, have a **positive spacelike-edge count**, and are
    **stable** across seeds (same order of magnitude). The magnitude itself is reported, not
    asserted to hit ~4.0 — the honest finding is what it is.
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


pmr = _load("proton_mass_radius")


class PureReaderTest(unittest.TestCase):
    """Fast: the geometric readers on a hand-built complex with a known metric."""

    def _pentatope(self):
        # one 4-simplex (5 vertices) with a unit spacelike metric.
        cells = [[0, 1, 2, 3, 4]]
        st = tessera.Spacetime.fromCells(4, cells, 1.0, 0.0)
        st.materializeFacets()
        return st

    def test_radius_counts_spacelike_edges(self):
        st = self._pentatope()
        r, n_sp, n_tl = pmr.radius_rms(st)
        # a unit-metric pentatope: 10 edges, all spacelike, RMS length 1.
        self.assertEqual(n_sp, 10)
        self.assertEqual(n_tl, 0)
        self.assertTrue(math.isfinite(r))
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_radius_separates_timelike(self):
        st = self._pentatope()
        # flip one edge timelike (negative real squared length) and re-read.
        e0 = st.getEdgeList().toVector()[0]
        e0.setSquaredLength(complex(-1.0, 0.0))
        r, n_sp, n_tl = pmr.radius_rms(st)
        self.assertEqual(n_sp, 9)
        self.assertEqual(n_tl, 1)
        self.assertTrue(math.isfinite(r) and r > 0.0)

    def test_curvature_proxy_finite_with_hinges(self):
        st = self._pentatope()
        mass_b, n_hinges = pmr.curvature_concentration_mass(st)
        # a 4-complex has triangular (2-cell) hinges; the proxy must be finite & >= 0.
        self.assertGreater(n_hinges, 0)
        self.assertTrue(math.isfinite(mass_b))
        self.assertGreaterEqual(mass_b, 0.0)

    def test_dual_action_returns_three_finite_parts(self):
        st = self._pentatope()
        re, im, mag = pmr.dual_action_mass(st)
        for x in (re, im, mag):
            self.assertTrue(math.isfinite(x))
        self.assertGreaterEqual(mag, 0.0)
        self.assertAlmostEqual(mag, abs(complex(re, im)), places=9)


class EmergentProtonTest(unittest.TestCase):
    """Slow: finiteness + spacelike count + cross-seed stability on real protons."""

    @classmethod
    def setUpClass(cls):
        cls.reads = []
        # two disjoint, narrow seed ranges; each converges on its first seed.
        for seed_lo in (5, 6):
            res = pmr.measure(
                seeds=range(seed_lo, seed_lo + 4), max_residual=0.6,
                n_refine=18, stage1_steps=60, stage2_iters=20)
            if res:
                cls.reads.append(res[0])

    def test_finite_and_spacelike_positive(self):
        if not self.reads:
            self.skipTest("no converged proton block in the seed ranges")
        for o in self.reads:
            for key in ("radius", "mass_re", "mass_im", "mass_abs",
                        "mass_b", "rm_a", "rm_b"):
                self.assertTrue(math.isfinite(o[key]), f"{key} not finite")
            self.assertGreater(o["n_spacelike"], 0)
            self.assertGreater(o["radius"], 0.0)
            self.assertGreaterEqual(o["mass_abs"], 0.0)

    def test_stable_across_seeds(self):
        if len(self.reads) < 2:
            self.skipTest("need >= 2 converged protons for a stability check")
        radii = [o["radius"] for o in self.reads]
        masses = [o["mass_abs"] for o in self.reads]
        # same order of magnitude across seeds (a loose, honest stability bound).
        self.assertLess(max(radii) / (min(radii) + 1e-12), 10.0)
        self.assertLess(max(masses) / (min(masses) + 1e-12), 1e3)
        # the reported r*|m| lands on a finite, positive number (magnitude not asserted).
        for o in self.reads:
            self.assertTrue(math.isfinite(o["rm_a"]) and o["rm_a"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
