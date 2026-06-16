# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Mediated selection over the level-1 fill ensemble
(``examples/cobordism/level1_mediated_selection.py``).

Among fills that all realize the identity transport at machine zero -- the
spectral tie the mediation objective was built to break -- these tests pin
what the dual Lorentzian Regge action selects:

  1. Twists are action-neutral: gluing through a symmetry is an isometry of
     the complex, so |S_Regge| is bit-identical to the straight prism's.
  2. Gated interior cuts RAISE the action above the straight fill of the
     same thickness; stellar growth is scored, not assumed.
  3. For every beta > 0, F_beta selects the minimal-action member -- the
     thinnest straight prism, the minimal interpolating geometry between
     the two interaction events.
  4. The Lorentzian-native fills (vertex times = layer, timelike inter-layer
     edges by the tracked metric rule) have finite actions with the same
     thinner-is-smaller ordering, and keep the 2-dimensional ker L_1 (the
     transport space is topological, so the metric cannot move it).
"""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "level1_mediated_selection.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("level1_mediated_selection",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["level1_mediated_selection"] = module
    spec.loader.exec_module(module)
    return module


SEL = _load_example()

# One small deterministic ensemble shared by the tests below.
_MEMBERS = SEL.build_ensemble(n_variants=4, base_seed=12345)
_ROWS = SEL.score_ensemble(_MEMBERS)
_BY = {r["label"]: r for r in _ROWS}


class EnsembleIsDegenerateTest(unittest.TestCase):
    """The family F_beta breaks ties in: every untwisted member realizes the
    identity at machine zero with a valid dual complex."""

    def test_every_member_keeps_a_valid_dual(self):
        for r in _ROWS:
            self.assertTrue(r["dual_valid"], msg=r["label"])

    def test_untwisted_members_realize_the_identity(self):
        for r in _ROWS:
            if r["twist"] is None:
                self.assertTrue(r["realizes_identity"], msg=r["label"])
                self.assertLess(r["identity_residual"], SEL.REALIZE,
                                msg=r["label"])


class ActionStructureTest(unittest.TestCase):
    """1 + 2: what the action sees."""

    def test_twists_are_action_neutral(self):
        s1 = _BY["straight L=1"]["S"]
        self.assertLess(abs(_BY["gamma twist L=1"]["S"] - s1), 1e-9)
        self.assertLess(abs(_BY["gamma^2 twist L=1"]["S"] - s1), 1e-9)

    def test_cuts_raise_the_action(self):
        s3 = _BY["straight L=3"]["S"]
        cuts = [r for r in _ROWS if r["n_cut"] > 0]
        self.assertGreaterEqual(len(cuts), 1)
        for r in cuts:
            self.assertGreater(r["S"], s3, msg=r["label"])

    def test_thinner_straight_prisms_have_smaller_action(self):
        self.assertLess(_BY["straight L=1"]["S"], _BY["straight L=2"]["S"])
        self.assertLess(_BY["straight L=2"]["S"], _BY["straight L=3"]["S"])


class FBetaSelectionTest(unittest.TestCase):
    """3: the selection verdict."""

    def test_positive_beta_selects_the_thinnest_straight_prism(self):
        table = SEL.f_beta_table(_ROWS)
        winners = {t["winner"] for t in table if t["beta"] > 0}
        self.assertEqual(winners, {"straight L=1"})

    def test_the_winner_is_the_minimal_action_member(self):
        family = [r for r in _ROWS if r["realizes_identity"]]
        min_s = min(r["S"] for r in family)
        self.assertLess(abs(_BY["straight L=1"]["S"] - min_s), 1e-12)


class LorentzianNativeTest(unittest.TestCase):
    """4: the CDT-natural fills."""

    @classmethod
    def setUpClass(cls):
        cls.fills = {layers: SEL._layered_time_bulk(layers)
                     for layers in (1, 2, 3)}
        cls.actions = {layers: SEL._regge_magnitude(st)
                       for layers, st in cls.fills.items()}

    def test_actions_are_finite_and_ordered(self):
        for layers, s in self.actions.items():
            self.assertTrue(SEL.np.isfinite(s), msg=f"L={layers}")
        self.assertLess(self.actions[1], self.actions[2])
        self.assertLess(self.actions[2], self.actions[3])

    def test_transport_space_survives_the_lorentzian_metric(self):
        cob = SEL.tessera.cobordism
        harm = cob.HodgeLaplacian(self.fills[1]).harmonics(1)
        self.assertEqual(len(harm), 2)


if __name__ == "__main__":
    unittest.main()
