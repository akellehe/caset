# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the cobordism RL PPO agent + training loop (#537).

The network/PPO-update checks run on synthetic data (no engine), so they are instant; a
single end-to-end ``benchmark`` smoke run drives the real engine with a deliberately tiny
budget (a couple of macro-actions, one training iteration) to confirm the train → evaluate
→ compare pipeline is wired up. PyTorch is installed via the ``dev`` extra (which pulls in
``rl``), so these run as part of the normal suite.
"""
import os
import sys
import unittest

import numpy as np

_EX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "examples", "cobordism"))
if _EX not in sys.path:
    sys.path.insert(0, _EX)

import pytest  # noqa: E402

# The PPO agent (and rl.train) require PyTorch, which lives in the optional `rl` extra and
# is deliberately NOT in `dev`/CI — skip this whole module when torch isn't installed.
torch = pytest.importorskip("torch")

import rl.objective_env as oe  # noqa: E402
import rl.ppo_agent as agent  # noqa: E402
import rl.train as train  # noqa: E402


class ActorCriticTest(unittest.TestCase):
    """Synthetic-data network checks (no engine, instant)."""

    def setUp(self):
        agent.set_seed(0)
        self.net = agent.HybridActorCritic(obs_dim=oe.OBS_DIM, n_moves=oe.N_MOVES,
                                           param_dim=oe.PARAM_DIM, hidden=16)

    def test_forward_shapes(self):
        obs = torch.randn(5, oe.OBS_DIM)
        move_logits, param_mean, param_std, value = self.net(obs)
        self.assertEqual(tuple(move_logits.shape), (5, oe.N_MOVES))
        self.assertEqual(tuple(param_mean.shape), (5, oe.PARAM_DIM))
        self.assertEqual(tuple(param_std.shape), (5, oe.PARAM_DIM))
        self.assertEqual(tuple(value.shape), (5,))
        self.assertTrue(torch.all(param_std > 0))

    def test_act_produces_valid_action(self):
        obs = torch.randn(1, oe.OBS_DIM)
        a = self.net.act(obs)
        self.assertIn(a["move"], range(oe.N_MOVES))
        self.assertEqual(a["params"].shape, (oe.PARAM_DIM,))
        self.assertIsInstance(a["logp"], float)
        self.assertIsInstance(a["value"], float)
        # Deterministic action = the mode (argmax move, mean params), reproducible.
        d1 = self.net.act(obs, deterministic=True)
        d2 = self.net.act(obs, deterministic=True)
        self.assertEqual(d1["move"], d2["move"])
        self.assertTrue(np.allclose(d1["params"], d2["params"]))

    def test_evaluate_actions_shapes_and_grad(self):
        obs = torch.randn(8, oe.OBS_DIM)
        moves = torch.randint(0, oe.N_MOVES, (8,))
        params = torch.randn(8, oe.PARAM_DIM)
        logp, entropy, value = self.net.evaluate_actions(obs, moves, params)
        self.assertEqual(tuple(logp.shape), (8,))
        self.assertEqual(tuple(entropy.shape), (8,))
        self.assertEqual(tuple(value.shape), (8,))
        logp.sum().backward()  # the joint log-prob is differentiable end-to-end
        self.assertIsNotNone(self.net.move_logits.weight.grad)


class PPOUpdateTest(unittest.TestCase):
    """The PPO update + GAE on synthetic transitions (no engine, instant)."""

    def _fake_transitions(self, n=40):
        rng = np.random.default_rng(0)
        ts = []
        for _ in range(n):
            ts.append(agent.Transition(
                obs=rng.standard_normal(oe.OBS_DIM).astype(np.float32),
                move=int(rng.integers(oe.N_MOVES)),
                params=rng.standard_normal(oe.PARAM_DIM).astype(np.float32),
                logp=float(rng.standard_normal()), value=float(rng.standard_normal()),
                reward=float(rng.standard_normal())))
        return ts

    def test_gae_fills_advantage_and_return(self):
        ppo = agent.PPO(oe.OBS_DIM, oe.N_MOVES, oe.PARAM_DIM, hidden=16)
        ts = self._fake_transitions(10)
        ppo._finish_gae(ts, last_value=0.0)
        self.assertTrue(all(np.isfinite(t.advantage) for t in ts))
        self.assertTrue(all(np.isfinite(t.ret) for t in ts))
        # return = advantage + value, by construction.
        for t in ts:
            self.assertAlmostEqual(t.ret, t.advantage + t.value, places=5)

    def test_update_runs_and_changes_params(self):
        ppo = agent.PPO(oe.OBS_DIM, oe.N_MOVES, oe.PARAM_DIM, hidden=16, update_epochs=2)
        ts = self._fake_transitions(40)
        ppo._finish_gae(ts, last_value=0.0)
        before = [p.detach().clone() for p in ppo.policy.parameters()]
        stats = ppo.update(ts)
        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        after = list(ppo.policy.parameters())
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)),
                        "PPO update did not change any parameters")


class RandomPolicyTest(unittest.TestCase):
    def test_random_policy_in_range(self):
        np.random.seed(0)
        for _ in range(20):
            move, params = train.random_policy(np.zeros(oe.OBS_DIM, np.float32))
            self.assertIn(move, range(oe.N_MOVES))
            self.assertEqual(params.shape, (oe.PARAM_DIM,))
            self.assertTrue(np.all((params >= 0) & (params <= 1)))


class TrainSmokeTest(unittest.TestCase):
    """A tiny end-to-end run: the real engine, but a couple of macro-actions and one
    training iteration, so the train → evaluate → compare pipeline is exercised fast."""

    def test_benchmark_smoke(self):
        result = train.benchmark(
            target="formation", iterations=1, episodes_per_iter=1, eval_seeds=1,
            max_actions=2, hidden=16, lr=1e-2, agent_seed=0, verbose=False,
            env_kwargs=dict(grow_steps=(2, 3), evolve_steps=(2, 3),
                            relax_iters=(1, 1), n_candidate_moves=3))
        # The pipeline produced both evaluations and a training history.
        self.assertEqual(len(result["history"]), 1)
        for side in ("rl", "random"):
            self.assertIn("mean_F_final", result[side])
            self.assertIn("mean_log_reduction", result[side])
            self.assertEqual(len(result[side]["episodes"]), 1)
        # The seed Δ⁴ objective is ~3150; any rollout leaves F finite and non-negative.
        self.assertGreaterEqual(result["rl"]["mean_F_final"], 0.0)
        self.assertTrue(np.isfinite(result["random"]["mean_F_final"]))


if __name__ == "__main__":
    unittest.main()
