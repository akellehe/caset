# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Stationary phase over the level-1 fill ensemble
(``examples/cobordism/level1_stationary_phase.py``).

The optimizer consumes |S|; a sum over geometries interferes with
e^{i lambda S}. These tests pin what the complex Sorkin action establishes
on the level-1 ensemble:

  1. **Phases see geometry.** Isometry classes are bit-exact phase
     degenerates: the twisted prisms carry the same complex action as their
     straight twins, and the gated single-cut draws collapse to one class
     (the interior tets are a symmetry orbit; the interiority guard forbids
     a second cut).
  2. **The honest negative.** With uniform weights over classes, the
     discrete phase sum does NOT localize on the minimal action -- the tail
     phase velocity sits in the family's bulk. Continuum stationary-phase
     intuition does not transfer to a uniform discrete measure, and the
     example asserts that rather than hiding it.
  3. **Where stationarity is well-posed, the optimizer sits on it.** The
     interior action dip in the growth direction at fixed thickness is
     exactly the fixed-thickness F_beta winner; globally, every positive
     beta selects a member of the minimal-action class.
  4. **Sorkin damping localizes the Lorentzian family on the F_beta
     winner.** The layered-time fills' actions are genuinely complex with
     |Im S| extensive in thickness, so the damped branch suppresses thicker
     fills exponentially and the thinnest fill -- the global optimizer's
     winner -- carries essentially all the weight.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "level1_stationary_phase.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("level1_stationary_phase",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["level1_stationary_phase"] = module
    spec.loader.exec_module(module)
    return module


SP = _load_example()

# One deterministic analysis pass shared by the tests below.
_RES = SP.analyze(n_variants=8, seed=12345)
_BY = {r["label"]: r for r in _RES["rows"]}


class PhasesSeeGeometryTest(unittest.TestCase):
    """1: isometry classes are exact phase degeneracies."""

    def test_twists_carry_the_same_complex_action(self):
        self.assertTrue(_RES["twist_degenerate"])

    def test_cut_draws_collapse_to_one_class(self):
        cut_classes = [c for c in _RES["classes"]
                       if any("cut" in m for m in c["members"])]
        self.assertEqual(len(cut_classes), 1)
        self.assertGreaterEqual(len(cut_classes[0]["members"]), 2)

    def test_unit_pin_actions_are_real(self):
        self.assertLess(_RES["max_im"], 1e-9)


class UniformSumDoesNotLocalizeTest(unittest.TestCase):
    """2: the asserted honest negative."""

    def test_phase_velocity_sits_in_the_bulk(self):
        self.assertTrue(_RES["non_localized"])
        self.assertGreater(_RES["slope"], 1.05 * _RES["s_min"])
        self.assertLess(_RES["slope"], _RES["s_max"])


class OptimizerSitsOnStationarityTest(unittest.TestCase):
    """3: the beta bridge, globally and at fixed thickness."""

    def test_global_winners_are_the_minimal_class(self):
        self.assertTrue(_RES["global_agree"])
        minimal = _RES["classes"][0]
        self.assertIn("straight L=1", minimal["members"])

    def test_interior_dip_exists_and_is_the_fixed_thickness_winner(self):
        self.assertTrue(_RES["interior_dip"])
        self.assertTrue(_RES["fixed_agree"])
        self.assertEqual(_RES["dip_label"], _RES["fixed_winner"])


class SorkinDampingLocalizesTest(unittest.TestCase):
    """4: the Lorentzian-native family."""

    @classmethod
    def setUpClass(cls):
        cls.actions = [SP._regge_complex(SP.SEL._layered_time_bulk(layers))
                       for layers in (1, 2, 3)]

    def test_actions_are_genuinely_complex(self):
        for s in self.actions:
            self.assertGreater(abs(s.imag), 1e-6)

    def test_damping_is_extensive_in_thickness(self):
        ims = [abs(s.imag) for s in self.actions]
        self.assertLess(ims[0], ims[1])
        self.assertLess(ims[1], ims[2])

    def test_damped_sum_localizes_on_the_thinnest_fill(self):
        ims = np.array([abs(s.imag) for s in self.actions])
        weights = np.exp(-ims)
        share = float(weights[0] / weights.sum())
        self.assertGreater(share, 1.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
