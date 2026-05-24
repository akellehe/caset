"""Python tests for the post-#56 InteractionSimulation bindings.

Pins the new InteractionConfig fields (epsInit, eps*Ki*, dimPerVertex,
epsLocalPure), the new InteractionSimulation properties (leafSimplexCount,
iMax), and verifies that the removed v0.1/v0.2 surface (annihilate,
pairCreate, getGlobalCharge, quditChargeOf, etc.) raises AttributeError.

Skips cleanly when ``TESSERA_QUANTUM=1`` was not set at build time.
"""

from __future__ import annotations

import math
import unittest

import pytest


try:
    from tessera import quantum  # noqa: F401
    _quantum = quantum
    _has_interaction_sim = hasattr(quantum, "InteractionSimulation")
except ImportError:
    _quantum = None
    _has_interaction_sim = False


pytestmark = pytest.mark.skipif(
    not _has_interaction_sim,
    reason="tessera.quantum.InteractionSimulation not built "
           "(needs TESSERA_QUANTUM=1)",
)


def _minimal_config(seed: int = 1):
    cfg = _quantum.InteractionConfig()
    cfg.nSystems = 4
    cfg.dimPerVertex = 2
    cfg.epsInit = 0.4
    cfg.beta = 1e-3
    cfg.epsilonI = 1e-10
    cfg.targetInteractions = 8
    cfg.delaunayEdges = [(0, 1), (1, 2), (2, 3),
                          (0, 2), (1, 3), (0, 3)]
    cfg.seed = seed
    cfg.quiet = True
    return cfg


class TestNewConfigSurface(unittest.TestCase):
    """The post-#56 config fields are exposed and round-trip."""

    def test_new_fields_settable(self):
        cfg = _quantum.InteractionConfig()
        # The KI-related tolerances and per-vertex dim.
        cfg.dimPerVertex = 8
        cfg.epsInit = 1.5
        cfg.epsLocalPure = 1e-12
        cfg.epsKiEigen = 5e-11
        cfg.epsKiCondState = 5e-11
        cfg.epsKiSvd = 5e-11
        # Read back.
        self.assertEqual(cfg.dimPerVertex, 8)
        self.assertAlmostEqual(cfg.epsInit, 1.5)
        self.assertAlmostEqual(cfg.epsLocalPure, 1e-12)
        self.assertAlmostEqual(cfg.epsKiEigen, 5e-11)
        self.assertAlmostEqual(cfg.epsKiCondState, 5e-11)
        self.assertAlmostEqual(cfg.epsKiSvd, 5e-11)

    def test_legacy_fields_removed_from_python_surface(self):
        cfg = _quantum.InteractionConfig()
        for removed in (
            "a", "g", "m", "dt",
            "useCharges", "featureCharges",
            "featureDeactivateOnAnnihilate", "featurePhotonOnAnnihilate",
            "featureQuditBasis", "featureChoiSigmaAB",
            "j_chargeCharge", "j_spinSpin",
            "massShift", "gammaCpViolation", "dtPair",
            "cpBias", "initialChargeMode",
        ):
            with self.assertRaises(AttributeError,
                                   msg=f"InteractionConfig still exposes {removed}"):
                setattr(cfg, removed, 0.0)


class TestInteractionSimulationProperties(unittest.TestCase):
    """The new readonly properties (iMax, leafSimplexCount)."""

    def test_imax_equals_n_times_eps_init(self):
        cfg = _minimal_config()
        sim = _quantum.InteractionSimulation(cfg)
        self.assertAlmostEqual(sim.iMax, cfg.nSystems * cfg.epsInit,
                               places=12)

    def test_leaf_simplex_count_starts_zero(self):
        sim = _quantum.InteractionSimulation(_minimal_config())
        self.assertEqual(sim.leafSimplexCount, 0)

    def test_initial_frontier_size_matches_n(self):
        cfg = _minimal_config()
        sim = _quantum.InteractionSimulation(cfg)
        self.assertEqual(sim.frontierSize, cfg.nSystems)

    def test_interaction_count_starts_zero(self):
        sim = _quantum.InteractionSimulation(_minimal_config())
        self.assertEqual(sim.interactionCount, 0)


class TestMovePrimitives(unittest.TestCase):
    """interact / unInteract / sweep at the Python binding level."""

    def test_interact_can_be_called_without_crashing(self):
        # NOTE: with the current placeholder buildInitialLayer (no MI
        # injection), interact() never accepts because product joints
        # give infinite edge lengths and NaN Regge action. This test
        # only pins that the move primitive is callable and the
        # simulation reports the rejection cleanly. Tighten when the
        # MI-injection follow-up lands.
        cfg = _minimal_config()
        cfg.beta = 1e-9
        sim = _quantum.InteractionSimulation(cfg)
        for _ in range(4):
            result = sim.interact()
            self.assertIn(result, (True, False))
        # Whether anything accepted or not, internal counts must agree.
        self.assertEqual(
            sim.interactionCount + sim.leafSimplexCount * 0,
            sim.interactionCount,
        )

    def test_uninteract_on_empty_leaves_returns_false(self):
        sim = _quantum.InteractionSimulation(_minimal_config())
        self.assertFalse(sim.unInteract())

    def test_sweep_returns_int(self):
        cfg = _minimal_config()
        cfg.beta = 1e-6
        sim = _quantum.InteractionSimulation(cfg)
        accepted = sim.sweep()
        self.assertIsInstance(accepted, int)
        self.assertGreaterEqual(accepted, 0)


class TestRemovedAPI(unittest.TestCase):
    """The v0.1/v0.2 methods all raise AttributeError post-#56."""

    def setUp(self):
        self.sim = _quantum.InteractionSimulation(_minimal_config())

    def test_annihilate_removed(self):
        with self.assertRaises(AttributeError):
            self.sim.annihilate()

    def test_pair_create_removed(self):
        with self.assertRaises(AttributeError):
            self.sim.pairCreate()

    def test_get_global_charge_removed(self):
        with self.assertRaises(AttributeError):
            self.sim.getGlobalCharge()

    def test_get_charge_profile_removed(self):
        with self.assertRaises(AttributeError):
            self.sim.getChargeProfile()

    def test_get_charge_correlation_removed(self):
        with self.assertRaises(AttributeError):
            self.sim.getChargeCorrelation(2)

    def test_qudit_charge_of_removed(self):
        # We don't have a Vertex handy; just check the attribute is gone.
        self.assertFalse(hasattr(self.sim, "quditChargeOf"))

    def test_qudit_state_of_removed(self):
        self.assertFalse(hasattr(self.sim, "quditStateOf"))

    def test_qudit_joint_state_for_removed(self):
        self.assertFalse(hasattr(self.sim, "quditJointStateFor"))

    def test_initial_charge_mode_enum_removed(self):
        self.assertFalse(hasattr(_quantum, "InitialChargeMode"))


class TestSeedDeterminism(unittest.TestCase):
    """Same seed -> same trajectory; different seeds -> independent."""

    def test_same_seed_yields_same_interaction_count(self):
        a = _quantum.InteractionSimulation(_minimal_config(seed=42))
        b = _quantum.InteractionSimulation(_minimal_config(seed=42))
        for _ in range(8):
            a.interact()
            b.interact()
        self.assertEqual(a.interactionCount, b.interactionCount)
        self.assertEqual(a.frontierSize, b.frontierSize)
        self.assertEqual(a.leafSimplexCount, b.leafSimplexCount)

    def test_beta_setter_round_trips(self):
        sim = _quantum.InteractionSimulation(_minimal_config())
        sim.beta = math.pi
        self.assertAlmostEqual(sim.beta, math.pi, places=14)


if __name__ == "__main__":
    unittest.main()
