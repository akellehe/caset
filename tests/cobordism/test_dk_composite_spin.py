# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Composite proton spin (#485). See docs/design/composite_proton_spin_findings.md.

Fast building block: the spinor holonomy `exp(ε·Σ)` is a genuine `Spin` element whose
eigenphases are `±ε/2` (the spin-½ double cover, not the vector `±ε`).

Readout (on controlled-b₃ fixtures, `tests/fixtures/composite_spin/`):
* **Robust, every b₃:** each constituent is spin-½ (`|⟨S⟩|=½`) and the composite `J²` lies in
  the three-spin-½ baryon range `[3/2, 15/4]` — the n=3 fingerprint.
* **Generic geometry:** the readout is frame-invariant — GAUGE (per-cell SO(4) of the
  embedding) and RELABEL (vertex-id permutation) both leave `J²` unchanged.
* The composite ½-vs-3/2 channel is only an honest negative (near-symmetric cells have no
  canonical rest frame; a product of per-hole spinors can't carry the entanglement) — the
  findings doc, not asserted here.
"""
import importlib.util
import json
import math
import os
import random
import sys
import unittest

import numpy as np
import scipy.linalg as sla

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")
_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "composite_spin")


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
            self.assertTrue(np.allclose(sorted(phases), sorted([-eps / 2, eps / 2]),
                                        atol=1e-9))
            self.assertTrue(self.cs.is_double_cover(eps, sigma))


class CompositeSpinReadoutTest(unittest.TestCase):
    """The composite-J² readout on controlled-b₃ fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.cs = _load("dk_composite_spin")
        cls.T = cls.eo.T

    # ----- fixture helpers -----
    def _rebuild(self, cells, edges, perm=None):
        T = self.T
        if perm is not None:
            cells = [[perm[v] for v in c] for c in cells]
            edges = {tuple(sorted((perm[a], perm[b]))): z for (a, b), z in edges.items()}
        st = T.Spacetime.fromCells(4, cells, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            e.setSquaredLength(edges[(a, b) if a < b else (b, a)])
        T.ReggeSolver(st, T.MatterConfiguration())
        return st

    def _load_fixture(self, name):
        d = json.load(open(os.path.join(_FIX, name)))
        cells = [list(c) for c in d["cells"]]
        edges = {}
        for k, (re, im) in d["edges"].items():
            a, b = (int(x) for x in k.split(","))
            edges[(a, b)] = complex(re, im)
        st = self._rebuild(cells, edges)
        return d, cells, edges, st, self.eo.emergent_holes(st, 3)

    def _all_fixtures(self):
        return sorted(f for f in os.listdir(_FIX) if f.endswith(".json"))

    # ----- robust, every b₃ -----
    def test_constituents_are_spin_half_for_all_betti(self):
        for name in self._all_fixtures():
            d, _c, _e, st, holes = self._load_fixture(name)
            with self.subTest(fixture=name, b3=d["b3"]):
                self.assertGreaterEqual(len(holes), 3)
                for s in self.cs.emergent_spinors(st, holes, self.eo._top_tuple):
                    s = s / np.linalg.norm(s)
                    pol = [float(np.real(s.conj() @ self.cs._SG[a] @ s)) for a in range(3)]
                    self.assertAlmostEqual(np.linalg.norm(pol), 0.5, places=6)

    def test_j2_in_three_spin_half_baryon_range(self):
        for name in self._all_fixtures():
            d, _c, _e, st, holes = self._load_fixture(name)
            with self.subTest(fixture=name, b3=d["b3"]):
                j2 = self.cs.emergent_j2(st, holes, self.eo._top_tuple)
                self.assertGreaterEqual(j2, 1.5 - 1e-6)      # n=3 product floor (3/2)
                self.assertLessEqual(j2, 3.75 + 1e-6)        # n=3 product ceiling (15/4)

    # ----- frame-invariance on geometrically generic structures -----
    def _gates(self, cells, edges, st, holes):
        cs, eo = self.cs, self.eo
        base = cs.emergent_j2(st, holes, eo._top_tuple)
        rng = np.random.default_rng(7)
        rmap = {}
        orig = cs.embed_cell

        def patched(cell):
            c = orig(cell)
            key = tuple(sorted(c))
            if key not in rmap:
                a = rng.standard_normal((4, 4))
                rmap[key] = sla.expm(a - a.T)
            r = rmap[key]
            return {v: r @ x for v, x in c.items()}

        cs.embed_cell = patched
        try:
            gauged = cs.emergent_j2(st, holes, eo._top_tuple)
        finally:
            cs.embed_cell = orig
        allv = sorted({v for s in st.getTopSimplices() for v in eo._top_tuple(s)})
        shuf = allv[:]
        random.Random(3).shuffle(shuf)
        perm = dict(zip(allv, shuf))
        st2 = self._rebuild(cells, edges, perm=perm)
        holes2 = [tuple(sorted(perm[v] for v in h)) for h in holes[:3]]
        relabeled = cs.emergent_j2(st2, holes2, eo._top_tuple)
        return base, abs(gauged - base), abs(relabeled - base)

    def test_gauge_and_relabel_invariance_on_generic_geometry(self):
        # Generic (non-degenerate) structures across a range of b₃; the readout is a genuine
        # frame-free observable here (near-symmetric cells are the documented obstruction).
        for name in ("synthetic_b3_3.json", "synthetic_b3_5.json",
                     "synthetic_b3_6.json", "synthetic_b3_7.json"):
            d, cells, edges, st, holes = self._load_fixture(name)
            with self.subTest(fixture=name, b3=d["b3"]):
                base, dg, dr = self._gates(cells, edges, st, holes)
                self.assertLess(dg, 1e-5)
                self.assertLess(dr, 1e-5)

    # ----- the exploratory joint-state channel read runs and is well-formed -----
    def test_joint_channel_weights_well_formed(self):
        d, _c, _e, st, holes = self._load_fixture("synthetic_b3_5.json")
        tt = self.eo._top_tuple
        spn = self.cs.joint_spinors(st, holes, tt)
        wd, wp = self.cs.spin_channel_weights(st, holes, tt, spn)
        self.assertAlmostEqual(wd + wp, 1.0, places=6)
        self.assertGreaterEqual(wd, -1e-6)
        self.assertGreaterEqual(wp, -1e-6)


if __name__ == "__main__":
    unittest.main()
