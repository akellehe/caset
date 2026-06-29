# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The connected-correlator escape-hatch test (#512, part of #410).

See `docs/theory/cobordism/proton-spin/cartan_weyl_gluon.tex` §7 and
`docs/design/connected_correlator_escape_hatch_findings.md`.

Two layers:

* **Instrument (fast, load-bearing).** `J² = 9/4 + 2·Σ ⟨S_i·S_j⟩` reproduces the validated
  measuring stick (proton ¾, product 7/4, Δ 15/4), and the connected `C_ij` is exactly the
  entangling content: it is non-zero (summing to −¾) only for the entangled proton, and zero
  for both product `|uud⟩` and Δ `|uuu⟩`.

* **Mesh (the escape-hatch result).** On a b₃=3 fixture, neither the vertical (reconstructed
  joint state) nor the horizontal (holonomy-invariants-only) route reaches the proton ¾; the
  vertical route's `C_ij` is zero (separable by construction), and the horizontal `J²` is a
  genuine frame-free observable (GAUGE + RELABEL invariant). The escape hatch is closed.
"""
import importlib.util
import json
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


cc = _load("dk_connected_correlator")
dj = _load("dk_joint_spin")          # the independently-validated J² instrument
_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _kr(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


_PROTON = 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP) - _kr(_DN, _UP, _UP)
_PRODUCT = _kr(_UP, _UP, _DN)
_DELTA = _kr(_UP, _UP, _UP)


class CorrelatorInstrumentTest(unittest.TestCase):
    """J² via the two-hole correlators reproduces the validated instrument, and C_ij is the
    entangling content."""

    def test_j2_from_pairs_matches_validated_instrument(self):
        for psi, want in ((_PROTON, 0.75), (_PRODUCT, 1.75), (_DELTA, 3.75)):
            self.assertAlmostEqual(cc.j2_from_pairs(psi), want, places=9)
            # identical to the independently-validated operator read
            self.assertAlmostEqual(cc.j2_from_pairs(psi), dj.j2_three_qubit(psi), places=9)

    def test_connected_correlator_is_the_entangling_content(self):
        # the proton carries the entanglement: Σ C_ij = −¾, all of the J² shift
        pr = cc.correlator_report(_PROTON)
        self.assertAlmostEqual(pr["sum_connected"], -0.75, places=9)
        self.assertGreater(abs(pr["sum_connected"]), 1e-6)
        # both product |uud⟩ and Δ |uuu⟩ are product states: C_ij = 0 for every pair
        for psi in (_PRODUCT, _DELTA):
            rep = cc.correlator_report(psi)
            self.assertAlmostEqual(rep["sum_connected"], 0.0, places=9)
            for pair in [(0, 1), (0, 2), (1, 2)]:
                self.assertAlmostEqual(rep["connected"][pair], 0.0, places=9)

    def test_connected_correlator_zero_on_random_products(self):
        rng = np.random.default_rng(512)
        for _ in range(5):
            qs = [rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(3)]
            qs = [q / np.linalg.norm(q) for q in qs]
            psi = _kr(*qs)
            for pair in [(0, 1), (0, 2), (1, 2)]:
                self.assertAlmostEqual(cc.connected_correlator(psi, *pair), 0.0, places=9)


class EscapeHatchMeshTest(unittest.TestCase):
    """Both routes on a b₃=3 fixture: neither reaches ¾ (the hatch is closed)."""

    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.T = cls.eo.T

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

    def test_neither_route_reaches_three_quarters(self):
        _d, _c, _e, st, holes = self._load_fixture("synthetic_b3_3.json")
        tt = self.eo._top_tuple
        vert = cc.cij_vertical(st, holes, tt, joint=True)
        horiz = cc.cij_horizontal(st, holes, tt)
        self.assertIsNotNone(vert)
        self.assertIsNotNone(horiz)
        # the escape hatch is closed: neither route lands on the proton ¾
        self.assertFalse(cc.reaches_proton(vert))
        self.assertFalse(cc.reaches_proton(horiz))
        self.assertGreater(vert["j2"], 0.75 + 0.2)
        self.assertGreater(horiz["j2"], 0.75 + 0.2)

    def test_vertical_connected_correlator_is_zero(self):
        # a reconstruction from per-hole spinors is separable — C_ij vanishes (the floor)
        _d, _c, _e, st, holes = self._load_fixture("synthetic_b3_3.json")
        vert = cc.cij_vertical(st, holes, self.eo._top_tuple, joint=True)
        self.assertAlmostEqual(vert["sum_connected"], 0.0, places=6)

    def test_horizontal_j2_is_frame_invariant(self):
        # the holonomy-only J² is a genuine frame-free observable (GAUGE + RELABEL)
        _d, cells, edges, st, holes = self._load_fixture("synthetic_b3_3.json")
        tt = self.eo._top_tuple
        base = cc.cij_horizontal(st, holes, tt)["j2"]

        rng = np.random.default_rng(7)
        rmap = {}
        orig = cc.cs.embed_cell

        def patched(cell):
            c = orig(cell)
            key = tuple(sorted(c))
            if key not in rmap:
                a = rng.standard_normal((4, 4))
                rmap[key] = sla.expm(a - a.T)
            return {v: rmap[key] @ x for v, x in c.items()}

        cc.cs.embed_cell = patched
        try:
            gauged = cc.cij_horizontal(st, holes, tt)["j2"]
        finally:
            cc.cs.embed_cell = orig

        allv = sorted({v for s in st.getTopSimplices() for v in tt(s)})
        shuf = allv[:]
        random.Random(3).shuffle(shuf)
        perm = dict(zip(allv, shuf))
        st2 = self._rebuild(cells, edges, perm=perm)
        holes2 = [tuple(sorted(perm[v] for v in h)) for h in holes[:3]]
        relabeled = cc.cij_horizontal(st2, holes2, tt)["j2"]

        self.assertLess(abs(gauged - base), 1e-5)
        self.assertLess(abs(relabeled - base), 1e-5)


if __name__ == "__main__":
    unittest.main()
