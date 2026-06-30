# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the cobordism RL objective-search environment (#537).

Confirm the Gym-style contract of `examples/cobordism/rl/objective_env.py`: the
``reset``/``step`` shapes and types, the reward equals the (monotone) drop in the true
objective F, the engine's ``dualComplexValid`` gate is respected after every macro-action
(the env never leaves an invalid complex behind), the action-input forms are parsed, and
``reset(seed)`` is deterministic. The env drives the real C++ engine, so the budgets here
are deliberately tiny to keep the suite fast.
"""
import math
import os
import sys
import unittest

import numpy as np

_EX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "examples", "cobordism"))
if _EX not in sys.path:
    sys.path.insert(0, _EX)

# `rl` is a package under examples/cobordism with relative imports, so import it normally
# (the sys.path entry makes it resolvable); spec-loading a single file would break the
# `from .objective_env import ...` relative imports its siblings use.
import rl.objective_env as oe  # noqa: E402


def _tiny_env(**kw):
    """A minimal-budget formation env so the engine work per step stays small."""
    params = dict(max_actions=3, grow_steps=(2, 3), evolve_steps=(2, 3),
                  relax_iters=(1, 2), n_candidate_moves=3)
    params.update(kw)
    return oe.make_formation_env(**params)


class EnvContractFastTest(unittest.TestCase):
    """Cheap checks that touch the engine only through a single fast reset."""

    def test_space_metadata(self):
        env = _tiny_env()
        self.assertEqual(env.obs_dim, oe.OBS_DIM)
        self.assertEqual(env.n_moves, oe.N_MOVES)
        self.assertEqual(env.param_dim, oe.PARAM_DIM)
        self.assertEqual(env.observation_space.shape, (oe.OBS_DIM,))
        self.assertEqual(env.action_space.n_moves, oe.N_MOVES)
        self.assertEqual(env.action_space.param_box.shape, (oe.PARAM_DIM,))

    def test_reset_shape_and_finite(self):
        env = _tiny_env()
        obs = env.reset(seed=1)
        self.assertEqual(obs.shape, (oe.OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        # A bare Δ⁴ seed: exactly one top cell (one pentatope, a contractible 4-ball). Its
        # single boundary 5-tuple is one emergent k=3 hole — the register grows from there.
        self.assertEqual(env.metrics["n_top_cells"], 1)
        self.assertEqual(env.metrics["betti"][0], 1)
        self.assertEqual(env.metrics["holes"], 1)

    def test_reconstructed_F_matches_engine_objective(self):
        # The env reconstructs F = ‖∇S‖² + γ·r_U from the two published components (to skip
        # a third eigensolve); it must equal the engine's own objective() exactly, or the
        # reward (a function of F) would be measuring the wrong thing.
        env = _tiny_env()
        env.reset(seed=3)
        self.assertAlmostEqual(env.metrics["F"], env.node.objective(), places=6)

    def test_reset_is_deterministic_in_seed(self):
        # The starting complex is the fixed Δ⁴ seed, so the same seed gives a bit-identical
        # initial observation (the seed only drives the engine's MOVE rng thereafter).
        a = _tiny_env().reset(seed=7)
        b = _tiny_env().reset(seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_split_action_forms(self):
        # Tuple, dict, and flat-array action encodings all parse to (move:int, params).
        move, params = oe.CobordismObjectiveEnv._split_action((oe.RELAX, [0.3, 0.7]))
        self.assertEqual(move, oe.RELAX)
        self.assertTrue(np.allclose(params, [0.3, 0.7]))
        move, params = oe.CobordismObjectiveEnv._split_action(
            {"move": oe.GROW, "params": [0.1, 0.2]})
        self.assertEqual(move, oe.GROW)
        move, params = oe.CobordismObjectiveEnv._split_action([2.0, 0.5, 0.9])
        self.assertEqual(move, 2)
        self.assertEqual(params.shape[0], oe.PARAM_DIM)

    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            _tiny_env().step((oe.GROW, [0.5, 0.5]))


class EnvStepTest(unittest.TestCase):
    """One short episode shared across the step-level assertions, to amortize engine cost."""

    @classmethod
    def setUpClass(cls):
        # A slightly larger grow budget than the fast tests so the first grow reliably does
        # real surgery (grows the complex), while the episode still runs in a few seconds.
        cls.env = oe.make_formation_env(max_actions=3, grow_steps=(5, 8),
                                        evolve_steps=(3, 5), relax_iters=(1, 2),
                                        n_candidate_moves=4)
        cls.obs0 = cls.env.reset(seed=2)
        cls.F0 = cls.env.metrics["F"]
        cls.cells0 = cls.env.metrics["n_top_cells"]
        # A grow, then an evolve, then a relax — one of each macro-move.
        cls.script = [(oe.GROW, [1.0, 0.5]), (oe.EVOLVE, [0.5, 0.5]), (oe.RELAX, [1.0, 0.4])]
        cls.records = []
        for action in cls.script:
            obs, reward, done, info = cls.env.step(action)
            gate_ok = cls.env.dual_complex_valid()[0]
            cls.records.append((obs, reward, done, info, gate_ok))

    def test_step_contract(self):
        for obs, reward, done, info, _gate in self.records:
            self.assertEqual(obs.shape, (oe.OBS_DIM,))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            for key in ("move", "F", "F_before", "delta_F", "rU", "rstate", "holes",
                        "carried", "terminated", "truncated"):
                self.assertIn(key, info)

    def test_reward_equals_monotone_drop_in_F(self):
        # reward == reward_scale * (slog(F_before) - slog(F_after)) when no carry/engine
        # fault — so its SIGN is exactly the sign of the true objective's drop.
        for _obs, reward, _done, info, _gate in self.records:
            if info["carried"] or info["engine_error"] is not None:
                continue
            expected = (math.log1p(abs(info["F_before"])) - math.log1p(abs(info["F"])))
            self.assertAlmostEqual(reward, expected, places=5)
            if info["F"] < info["F_before"]:
                self.assertGreater(reward, 0.0)
            elif info["F"] > info["F_before"]:
                self.assertLess(reward, 0.0)

    def test_first_grow_grows_the_complex(self):
        # GROW from the bare seed must add topology — the defining effect of a stage-1
        # surgery pass (whether the bounded budget also lowers F is up to the engine, so
        # the robust invariant is the cell count growing, not a guaranteed F drop).
        first = self.records[0]
        self.assertEqual(first[3]["move_name"], "grow")
        self.assertGreater(first[3]["n_top_cells"], self.cells0)

    def test_gating_respected_after_every_move(self):
        # Every macro-action is gated inside the engine; the resulting complex must stay a
        # valid manifold-with-boundary (dualComplexValid) after each one.
        for _obs, _reward, _done, _info, gate_ok in self.records:
            self.assertTrue(gate_ok)

    def test_done_truncates_on_budget(self):
        # max_actions=3: only the third (last) step reports done (truncated, not terminated).
        self.assertFalse(self.records[0][2])
        self.assertFalse(self.records[1][2])
        self.assertTrue(self.records[2][2])
        self.assertTrue(self.records[2][3]["truncated"])

    def test_episode_return_telescopes_to_log_reduction(self):
        # With no carry bonus, the summed reward equals the total log-reduction of F over
        # the episode (the per-step rewards telescope) — the benchmark's headline metric.
        total = sum(r for _o, r, _d, info, _g in self.records
                    if not info["carried"])
        log_red = math.log1p(abs(self.F0)) - math.log1p(abs(self.records[-1][3]["F"]))
        self.assertAlmostEqual(total, log_red, places=5)


if __name__ == "__main__":
    unittest.main()
