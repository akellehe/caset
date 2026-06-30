# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A PyTorch PPO actor-critic for the cobordism objective-search policy (#537).

The policy is **hybrid / parameterized-action**: at each state it picks a discrete
macro-move (GROW / EVOLVE / RELAX) *and* a continuous parameter vector (intensity, β
knob). The network is a shared MLP trunk feeding four heads:

  * a **categorical** head over the macro-moves,
  * a **Gaussian** head (mean + state-independent log-σ) for the continuous params,
  * a **value** head (the critic).

Training is Proximal Policy Optimization with Generalized Advantage Estimation — the
joint log-prob is ``logπ(move) + Σ logπ(param_i)`` and the joint entropy adds likewise, so
both heads are optimized under one clipped surrogate. The continuous samples are kept raw
for the log-prob; the environment clips them to its valid range (so the squashing lives in
the env dynamics, keeping the policy a clean diagonal Gaussian).

This module needs PyTorch (``pip install -e ".[rl]"``). It is engine-agnostic: it only
sees observation vectors and the ``(obs, reward, done, info)`` contract of
:class:`objective_env.CobordismObjectiveEnv`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch for reproducible agent behavior (the env's engine RNG is
    seeded separately, per `reset(seed)`)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class HybridActorCritic(nn.Module):
    """Shared-trunk actor-critic with a categorical move head, a Gaussian parameter head,
    and a value head."""

    def __init__(self, obs_dim: int, n_moves: int, param_dim: int, hidden: int = 64):
        super().__init__()
        self.n_moves = n_moves
        self.param_dim = param_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.move_logits = nn.Linear(hidden, n_moves)
        self.param_mean = nn.Linear(hidden, param_dim)
        # State-independent log-σ for the continuous params (standard PPO parameterization).
        self.param_log_std = nn.Parameter(torch.full((param_dim,), -0.5))
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        move_logits = self.move_logits(h)
        param_mean = self.param_mean(h)
        param_std = torch.exp(self.param_log_std).expand_as(param_mean)
        value = self.value(h).squeeze(-1)
        return move_logits, param_mean, param_std, value

    def _distributions(self, obs: torch.Tensor):
        move_logits, param_mean, param_std, value = self.forward(obs)
        return Categorical(logits=move_logits), Normal(param_mean, param_std), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> dict:
        """Sample (or, if ``deterministic``, take the mode of) an action for one obs."""
        move_dist, param_dist, value = self._distributions(obs)
        if deterministic:
            move = torch.argmax(move_dist.probs, dim=-1)
            params = param_dist.mean
        else:
            move = move_dist.sample()
            params = param_dist.sample()
        logp = move_dist.log_prob(move) + param_dist.log_prob(params).sum(-1)
        # `act` scores ONE observation (the rollout passes a batch of 1); squeeze the batch
        # dim so `params` is the flat (param_dim,) vector the env expects.
        return {
            "move": int(move.reshape(-1)[0].item()),
            "params": params.reshape(-1)[:self.param_dim].cpu().numpy().astype(np.float32),
            "logp": float(logp.reshape(-1)[0].item()),
            "value": float(value.reshape(-1)[0].item()),
        }

    @torch.no_grad()
    def value_of(self, obs: torch.Tensor) -> float:
        return float(self.forward(obs)[3].item())

    def evaluate_actions(self, obs: torch.Tensor, moves: torch.Tensor,
                         params: torch.Tensor):
        """Joint log-prob, joint entropy, and value for a batch — the PPO update path."""
        move_dist, param_dist, value = self._distributions(obs)
        logp = move_dist.log_prob(moves) + param_dist.log_prob(params).sum(-1)
        entropy = move_dist.entropy() + param_dist.entropy().sum(-1)
        return logp, entropy, value


@dataclass
class Transition:
    obs: np.ndarray
    move: int
    params: np.ndarray
    logp: float
    value: float
    reward: float
    advantage: float = 0.0
    ret: float = 0.0


class PPO:
    """Proximal Policy Optimization for the hybrid policy."""

    def __init__(self, obs_dim: int, n_moves: int, param_dim: int, hidden: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
                 update_epochs: int = 6, minibatch_size: int = 64,
                 max_grad_norm: float = 0.5, device: Optional[str] = None):
        self.device = torch.device(device or "cpu")
        self.policy = HybridActorCritic(obs_dim, n_moves, param_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.update_epochs, self.minibatch_size = update_epochs, minibatch_size
        self.max_grad_norm = max_grad_norm

    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, np.float32), device=self.device)

    # ------------------------------------------------------------------ rollout
    def collect_episode(self, env, seed: int) -> Tuple[List[Transition], dict]:
        """Run one full episode (reset → done) under the current policy, returning its
        transitions with GAE advantages + returns already filled in, plus the final
        ``info`` dict (for benchmark metrics)."""
        obs = env.reset(seed)
        transitions: List[Transition] = []
        done = False
        info: dict = {}
        while not done:
            action = self.policy.act(self._tensor(obs).unsqueeze(0))
            next_obs, reward, done, info = env.step((action["move"], action["params"]))
            transitions.append(Transition(
                obs=np.asarray(obs, np.float32), move=action["move"],
                params=np.asarray(action["params"], np.float32),
                logp=action["logp"], value=action["value"], reward=reward))
            obs = next_obs
        # Bootstrap: 0 if the episode truly terminated (target carried), else V(final state)
        # for a time-limit truncation — the standard bias fix for fixed-horizon episodes.
        terminated = bool(info.get("terminated", False))
        last_value = 0.0 if terminated else self.policy.value_of(
            self._tensor(obs).unsqueeze(0))
        self._finish_gae(transitions, last_value)
        return transitions, info

    def _finish_gae(self, transitions: List[Transition], last_value: float) -> None:
        gae = 0.0
        next_value = last_value
        for t in reversed(transitions):
            delta = t.reward + self.gamma * next_value - t.value
            gae = delta + self.gamma * self.lam * gae
            t.advantage = gae
            t.ret = gae + t.value
            next_value = t.value

    # ------------------------------------------------------------------ update
    def update(self, transitions: List[Transition]) -> dict:
        """One PPO update over a batch of transitions (several epochs of minibatch SGD)."""
        obs = self._tensor(np.stack([t.obs for t in transitions]))
        moves = torch.as_tensor([t.move for t in transitions], device=self.device)
        params = self._tensor(np.stack([t.params for t in transitions]))
        old_logp = self._tensor([t.logp for t in transitions])
        returns = self._tensor([t.ret for t in transitions])
        advantages = self._tensor([t.advantage for t in transitions])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(transitions)
        idx = np.arange(n)
        last = {}
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, device=self.device)
                logp, entropy, value = self.policy.evaluate_actions(
                    obs[mb_t], moves[mb_t], params[mb_t])
                ratio = torch.exp(logp - old_logp[mb_t])
                adv = advantages[mb_t]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (returns[mb_t] - value).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (policy_loss + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last = {"policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.mean().item())}
        return last

    # ------------------------------------------------------------------ eval
    def select_action(self, obs, deterministic: bool = True):
        """The greedy/sampled action for evaluation — ``(move, params)`` for `env.step`."""
        a = self.policy.act(self._tensor(obs).unsqueeze(0), deterministic=deterministic)
        return a["move"], a["params"]
