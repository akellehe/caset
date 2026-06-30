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


class RewardShapingTest(unittest.TestCase):
    """The #546 dense proton-shaping reward terms (hole-progress + r_state descent). These
    are a TRAINING SIGNAL only — they never touch F, the engine, or the carry verdict. The
    formula checks validate the terms against the engine's own holes/r_state regardless of
    whether the tiny budget here actually forms a hole, so they stay fast + robust."""

    @staticmethod
    def _slog(x):
        return math.copysign(math.log1p(abs(x)), x)

    def test_weights_default_off(self):
        env = _tiny_env()
        self.assertEqual(env.hole_reward_weight, 0.0)
        self.assertEqual(env.rstate_reward_weight, 0.0)

    def test_shaping_off_zeroes_those_terms(self):
        # Default weights ⇒ the hole/r_state terms are exactly 0, so the reward is the
        # foundation's −ΔF (+ carry) and the breakdown still sums to the reward.
        env = _tiny_env()
        env.reset(seed=5)
        _o, reward, _d, info = env.step((oe.GROW, [1.0, 0.5]))
        terms = info["reward_terms"]
        self.assertEqual(terms["hole"], 0.0)
        self.assertEqual(terms["rstate"], 0.0)
        self.assertAlmostEqual(sum(terms.values()), reward, places=6)

    def test_terms_sum_to_reward_with_shaping_on(self):
        env = _tiny_env(hole_reward_weight=2.0, rstate_reward_weight=1.5, carry_bonus=7.0)
        env.reset(seed=6)
        for action in [(oe.GROW, [1.0, 0.5]), (oe.RELAX, [0.5, 0.4])]:
            _o, reward, _d, info = env.step(action)
            self.assertAlmostEqual(sum(info["reward_terms"].values()), reward, places=6)

    def test_hole_term_matches_capped_progress(self):
        # hole term == w·(min(holes, T) − min(prev_holes, T)); tracking the holes sequence
        # validates the formula AND its cap (no reward past T holes) vs the engine's count.
        W, T = 3.0, 3
        env = _tiny_env(hole_reward_weight=W, target_holes=T, rstate_reward_weight=0.0)
        env.reset(seed=7)
        prev_h = env.metrics["holes"]
        for action in [(oe.GROW, [1.0, 0.5]), (oe.GROW, [1.0, 0.5]), (oe.RELAX, [0.5, 0.4])]:
            _o, _r, _d, info = env.step(action)
            cur_h = info["holes"]
            expected = W * (min(cur_h, T) - min(prev_h, T))
            self.assertAlmostEqual(info["reward_terms"]["hole"], expected, places=6)
            prev_h = cur_h

    def test_rstate_term_matches_slog_descent(self):
        W = 1.25
        env = _tiny_env(rstate_reward_weight=W, hole_reward_weight=0.0)
        env.reset(seed=8)
        prev_r = env.metrics["rstate"]
        for action in [(oe.GROW, [1.0, 0.5]), (oe.RELAX, [0.5, 0.4])]:
            _o, _r, _d, info = env.step(action)
            cur_r = info["rstate"]
            if math.isfinite(prev_r) and math.isfinite(cur_r):
                expected = W * (self._slog(prev_r) - self._slog(cur_r))
                self.assertAlmostEqual(info["reward_terms"]["rstate"], expected, places=5)
            prev_r = cur_r

    def test_rstate_term_off_without_whole_target(self):
        # The recombination node has no whole-cobordism target, so r_state shaping is a
        # no-op even with a positive weight (there is nothing to fit the singlet against).
        env = oe.make_recombination_env(max_actions=2, grow_steps=(2, 3),
                                        evolve_steps=(2, 3), relax_iters=(1, 2),
                                        n_candidate_moves=3, rstate_reward_weight=2.0)
        env.reset(seed=9)
        _o, _r, _d, info = env.step((oe.GROW, [1.0, 0.5]))
        self.assertEqual(info["reward_terms"]["rstate"], 0.0)


class TerminationFlagTest(unittest.TestCase):
    """`terminate_on_carry` gates whether forming the proton ENDS the episode. The carry
    happens at the GROW stage, but the full proton arc is grow → evolve → relax, so the
    carry profile keeps going past the carry to relax. White-box: force the carry verdict so
    the logic is tested without a real (slow) carry."""

    def test_default_terminates_on_carry(self):
        env = _tiny_env()  # default terminate_on_carry=True (#539 behavior)
        self.assertTrue(env.terminate_on_carry)
        env.reset(seed=11)
        env._is_carried = lambda metrics: True  # force the carry verdict
        _o, _r, done, info = env.step((oe.RELAX, [0.5, 0.5]))
        self.assertTrue(info["carried"])
        self.assertTrue(info["terminated"])
        self.assertTrue(done)

    def test_flag_off_keeps_episode_running_after_carry(self):
        # With the flag off, carrying does NOT end the episode (only the budget does), so the
        # policy can relax after forming the register. The one-time carry bonus still fires.
        env = _tiny_env(terminate_on_carry=False, carry_bonus=5.0)
        self.assertFalse(env.terminate_on_carry)
        env.reset(seed=12)
        env._is_carried = lambda metrics: True
        _o, _r, done, info = env.step((oe.RELAX, [0.5, 0.5]))
        self.assertTrue(info["carried"])
        self.assertFalse(info["terminated"])   # flag off → no termination
        self.assertFalse(done)                 # step 1 of max_actions=3, so not truncated
        self.assertEqual(info["reward_terms"]["carry"], 5.0)  # bonus still fires once


class DirectedConeOutTest(unittest.TestCase):
    """The #546 directed cone-out probe — deliberate, gated hole creation (SurgicalCone)
    instead of run_stage1's random cone draws."""

    def test_config_stored(self):
        env = _tiny_env(directed_grow=True, cone_strategy="bfs", cone_overshoot=2)
        self.assertTrue(env.directed_grow)
        self.assertEqual(env.cone_strategy, "bfs")
        self.assertEqual(env.cone_overshoot, 2)
        self.assertFalse(_tiny_env().directed_grow)  # default off (pure #539 growth)

    def test_no_cone_on_bare_seed(self):
        # The bare Δ⁴ seed has a single top cell; coneOut refuses to remove the last cell, so
        # the probe opens nothing and leaves the complex untouched + valid.
        env = _tiny_env()
        env.reset(seed=4)
        self.assertEqual(env.metrics["n_top_cells"], 1)
        self.assertEqual(env.directed_cone_out("greedy"), 0)
        self.assertTrue(env.dual_complex_valid()[0])

    def test_probe_opens_holes_and_keeps_complex_valid(self):
        # On a grown complex the directed cone-out probe opens holes (or no-ops), never
        # closes them, stops by `target_holes + cone_overshoot`, always leaves a valid
        # manifold-with-boundary, and never strands a pinned input vertex. Both strategies.
        for strategy in ("greedy", "bfs"):
            env = oe.make_formation_env(max_actions=2, grow_steps=(8, 12),
                                        evolve_steps=(3, 5), relax_iters=(1, 2),
                                        n_candidate_moves=4, target_holes=3, cone_overshoot=2)
            env.reset(seed=2)
            env.step((oe.GROW, [1.0, 0.5]))            # build some bulk (random draws)
            pinned = env._pinned_vertex_ids()
            holes_before = env.metrics["holes"]
            opened = env.directed_cone_out(strategy)
            self.assertGreaterEqual(opened, 0)
            holes_after = len(oe.cob.MultiCobordism.emergent_holes(env.node.st, env.k))
            self.assertGreaterEqual(holes_after, holes_before)   # never closes holes
            self.assertLessEqual(holes_after, env.target_holes + env.cone_overshoot)
            self.assertTrue(env.dual_complex_valid()[0])         # still a valid manifold
            current = {int(v.getId()) for v in env.node.st.getVertexList().toVector()}
            self.assertTrue(pinned.issubset(current))            # no pinned vertex stranded

    def test_cone_in_noop_at_or_below_target(self):
        # directed_cone_in (register selection) only trims when MORE than target_holes exist;
        # the bare seed has 1 hole ≤ 3, so it is a no-op and leaves a valid complex.
        env = _tiny_env()
        env.reset(seed=4)
        self.assertEqual(env.directed_cone_in("greedy"), 0)
        self.assertTrue(env.dual_complex_valid()[0])

    def test_directed_cone_in_caps_to_target_and_stays_valid(self):
        # On a grown complex, over-open with cone-out (target + overshoot) then cone-in
        # SELECTS back toward target_holes: it never leaves more holes than it started with,
        # never drops below target while holes remain above it, and keeps a valid manifold.
        env = oe.make_formation_env(max_actions=3, grow_steps=(8, 12), evolve_steps=(3, 5),
                                    relax_iters=(1, 2), n_candidate_moves=4, target_holes=3,
                                    cone_overshoot=2)
        env.reset(seed=2)
        env.step((oe.GROW, [1.0, 0.5]))
        env.directed_cone_out("greedy")
        holes_open = len(oe.cob.MultiCobordism.emergent_holes(env.node.st, env.k))
        closed = env.directed_cone_in("greedy")
        holes_sel = len(oe.cob.MultiCobordism.emergent_holes(env.node.st, env.k))
        self.assertGreaterEqual(closed, 0)
        self.assertLessEqual(holes_sel, holes_open)              # cone-in only closes
        self.assertGreaterEqual(holes_sel, min(holes_open, env.target_holes))
        self.assertTrue(env.dual_complex_valid()[0])


if __name__ == "__main__":
    unittest.main()
