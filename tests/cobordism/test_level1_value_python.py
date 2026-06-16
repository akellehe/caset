# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The level-1 value reading
(``examples/cobordism/level1_value_reading.py``).

H3 one level up. The level-1 value of a transport at a carried pair is the
anchored cross-end pairing Z_T(q, alpha) = s1 <g_out(q), g_in(alpha)>, and
these tests pin the structure the example establishes:

  1. **The equivariant prism is the level-1 isometric chart.** The staircase
     prism's diagonal choices break the end's C_3 (its chart is measurably
     anisotropic, ~1e-2); the equivariant prism (a center per wall quad and
     per prism cell, no diagonal choices) carries every end automorphism,
     so its chart is isometric at machine precision and Z_T equals the hand
     amplitude <q|u'|alpha> on every pair.
  2. **The exact deviation law, one level up.** On every anisotropic fill,
     Z_T - amp = w~^dag (G1 - I) alpha~ to machine precision, with w~ read
     off the output-indexed chart (no transport inverse needed) -- the
     level-0 lemma verbatim, including on the 4-dimensional S_3 family and
     its transposition (CNOT on the rho twist).
  3. **Composition.** Stacked fills multiply: the 2-layer equivariant
     gamma-twisted prism's value equals the matrix product of the single
     layer's transports. And across levels: the straight fill's value of
     trivial evolution reduces to the level-0 anchored amplitude on the end
     register.
  4. **No value for non-transports**: the carried-pair leak certificate.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "level1_value_reading.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("level1_value_reading",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["level1_value_reading"] = module
    spec.loader.exec_module(module)
    return module


VR = _load_example()

# Shared deterministic fixtures.
_EQ_STRAIGHT = VR.CellsFill(VR._equivariant_prism_cells(layers=1), layers=1)
_EQ_GAMMA = VR.CellsFill(
    VR._equivariant_prism_cells(layers=1, twist=VR.L1._GAMMA), layers=1)
_STAIRCASE = VR.L1.Level1Fill(layers=1)


class EquivariantChartTest(unittest.TestCase):
    """1: the equivariant prism restores the isometric chart."""

    def test_construction(self):
        cells = VR._equivariant_prism_cells(layers=1)
        self.assertEqual(len(cells), 238)
        self.assertEqual(len({v for c in cells for v in c}), 71)
        self.assertTrue(_EQ_STRAIGHT.dual_valid, _EQ_STRAIGHT.dual_reason)
        self.assertEqual(_EQ_STRAIGHT.dim, 2)

    def test_charts_are_isometric_at_machine_precision(self):
        for fill in (_EQ_STRAIGHT, _EQ_GAMMA):
            _G, dev = VR.ValueReader(fill).gram()
            self.assertLess(dev, 1e-9)

    def test_transports_are_the_mapping_classes(self):
        self.assertEqual(VR.L1.match_gate(_EQ_STRAIGHT.emergent_gate()),
                         "Identity")
        self.assertEqual(VR.L1.match_gate(_EQ_GAMMA.emergent_gate()),
                         "3-cycle (0312)")

    def test_value_equals_amplitude_on_every_pair(self):
        for fill, label in ((_EQ_STRAIGHT, "straight"), (_EQ_GAMMA, "gamma")):
            s = VR._survey(fill, label)
            self.assertLess(s["worst_dev"], 1e-9, msg=label)

    def test_staircase_chart_is_anisotropic(self):
        # the constructive finding: the staircase diagonals break the end
        # symmetry, so its chart cannot be isometric
        _G, dev = VR.ValueReader(_STAIRCASE).gram()
        self.assertGreater(dev, 1e-4)


class ExactDeviationLawTest(unittest.TestCase):
    """2: the level-0 lemma, one level up, on every anisotropic fill."""

    def test_staircase_obeys_the_law_exactly(self):
        s = VR._survey(_STAIRCASE, "staircase")
        self.assertLess(s["worst_law_residual"], 1e-9)
        self.assertGreater(s["worst_dev"], 1e-4)   # and the raw dev is real

    def test_cut_variant_obeys_the_law_exactly(self):
        s = VR._survey(VR._cut_variant(11), "cut")
        self.assertLess(s["worst_law_residual"], 1e-9)

    def test_4d_transposition_obeys_the_law_exactly(self):
        fill = VR.LAW.Level1FillS3(twist=VR.LAW._RHO)
        s = VR._survey(fill, "4d rho", n_states=3)
        self.assertEqual(s["emergent"], "CNOT")
        self.assertLess(s["worst_law_residual"], 1e-9)


class CompositionTest(unittest.TestCase):
    """3: stacked fills multiply; levels reduce."""

    def test_stacked_equivariant_fills_multiply(self):
        stack = VR.CellsFill(
            VR._equivariant_prism_cells(layers=2, twist=VR.L1._GAMMA),
            layers=2)
        u_single = VR._u3(_EQ_GAMMA)
        reader = VR.ValueReader(stack)
        worst = 0.0
        for q in VR._states(2, 5):
            for a in (reader.beta0,):
                z = reader.transport_value(q, a)
                amp = complex(np.vdot(q, u_single @ (u_single @ a)))
                worst = max(worst, abs(z - amp))
        self.assertLess(worst, 1e-9)
        u_stack = VR._u3(stack)
        self.assertLess(float(np.max(np.abs(
            VR._E @ (u_stack - u_single @ u_single) @ VR._E.T))), 1e-9)

    def test_level1_trivial_evolution_reduces_to_level0(self):
        reg = VR.BASE.Register()
        reader = VR.ValueReader(_EQ_STRAIGHT)
        h_b = reg.harmonic_form(reg.sign * reader.beta0)
        s0 = 1.0 / float(np.vdot(h_b, h_b).real)
        for q in VR._states(2, 9):
            h_q = reg.harmonic_form(reg.sign * q)
            a0 = s0 * complex(np.vdot(h_q, h_b))
            z1 = reader.transport_value(q, reader.beta0)
            flat = complex(np.vdot(q, reader.beta0))
            self.assertLess(abs(a0 - flat), 1e-9)
            self.assertLess(abs(z1 - flat), 1e-9)


class NoValueTest(unittest.TestCase):
    """4: a non-transport has no carried pair."""

    def test_swap_leaks_on_the_straight_fill(self):
        swap_u = next(u for n, u in VR.L1._v_candidates() if n == "SWAP")
        a = VR._CP_IN.astype(complex)
        fill = _EQ_STRAIGHT
        pair = np.concatenate([fill.sign0 * a, fill.sign1 * (swap_u @ a)])
        coeffs, *_ = np.linalg.lstsq(fill.P6.T, pair, rcond=None)
        leak = float(np.linalg.norm(pair - coeffs @ fill.P6))
        self.assertGreater(leak, 1e-6)


if __name__ == "__main__":
    unittest.main()
