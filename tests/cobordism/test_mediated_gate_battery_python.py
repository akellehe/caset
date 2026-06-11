"""Regge-mediated gate-battery β-sweep (#249).

Locks in the headline claims of examples/cobordism/mediated_gate_battery.py:
  * β=0 reproduces the base-layer 13-gate set bit-for-bit, with H3 (the holonomy-
    charge leak) at machine precision;
  * H2 (the input boundary data) is byte-fixed across the sweep;
  * mediation contracts the realizable set as β grows;
  * H3 holds (the explicit amplitude matches) for every realized gate at every β.

The sweep is cheap (per-gate residual projections + 4 precomputed |S_Regge| values),
so the whole battery runs in-process.
"""

from __future__ import annotations

import os
import sys
import unittest

import pytest

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "examples", "cobordism")
sys.path.insert(0, _EX)

try:
    import tessera  # noqa: F401
    import spectral_gate_realizability as base
    import mediated_gate_battery as mgb
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


class MediatedGateBattery(unittest.TestCase):
    def test_beta0_reproduces_13_gate_base_layer(self):
        rows, h2 = mgb.sweep([0.0])
        s = mgb.summarize(rows, [0.0])[0]
        self.assertEqual(s["n_realizable"], 13)
        self.assertEqual(set(s["realized"]), set(base.CANONICAL_SET))
        self.assertTrue(h2["input_boundary_byte_fixed"])          # H2
        self.assertLess(s["max_charge_leak_realized"], 1e-9)      # H3 at β=0

    def test_mediation_contracts_the_realizable_set(self):
        rows, _ = mgb.sweep([0.0, 10.0])
        by_beta = {s["beta"]: s["n_realizable"]
                   for s in mgb.summarize(rows, [0.0, 10.0])}
        self.assertEqual(by_beta[0.0], 13)
        self.assertLess(by_beta[10.0], by_beta[0.0])  # a strong gravitational prior shrinks it

    def test_h3_holds_for_every_realized_gate(self):
        rows, _ = mgb.sweep([0.0, 1.0, 3.0])
        for r in rows:
            if r["realizable"]:
                # the explicit amplitude (charge leak) matches at machine precision
                self.assertLess(r["charge_leak"], 1e-9)
                self.assertTrue(r["h3_holds"])

    def test_realized_gates_pick_all_three_holes_at_beta0(self):
        # at β=0 the search pays nothing for holes, so every realized gate opens the
        # full register (k=3, b_1=2) — the base-layer topology.
        rows, _ = mgb.sweep([0.0])
        for r in rows:
            if r["realizable"]:
                self.assertEqual(r["k_star"], 3)
                self.assertEqual(r["b1"], 2)


if __name__ == "__main__":
    unittest.main()
