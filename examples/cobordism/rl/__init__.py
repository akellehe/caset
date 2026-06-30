# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Reinforcement-learning driver for the cobordism objective search (#537).

A scoped foundation for learning the SEARCH POLICY that drives the emergent
`MultiCobordism` optimizer, instead of the random+greedy orchestration in
`Proton.build()`. The objective ``F = ‖∇S_Regge‖² + Γ·r_U``, the physics, and the
C++ engine are untouched: the policy only chooses WHICH gated surgery / relaxation
macro-action (and its parameters) to apply at each step, through the existing public
bindings, with every topology move still gated on ``dualComplexValid`` inside the
engine.

  * :mod:`objective_env` — a Gym-style environment wrapping one `MultiCobordism` node.
  * :mod:`ppo_agent`     — a PyTorch PPO actor-critic (hybrid discrete+continuous).
  * :mod:`train`         — a training loop + a random-baseline benchmark.

`objective_env` depends only on `tessera` + `numpy`; `ppo_agent`/`train` additionally
need PyTorch (``pip install -e ".[rl]"``).
"""

from .objective_env import (
    CobordismObjectiveEnv,
    formation_node_factory,
    recombination_node_factory,
    make_formation_env,
    make_recombination_env,
)

__all__ = [
    "CobordismObjectiveEnv",
    "formation_node_factory",
    "recombination_node_factory",
    "make_formation_env",
    "make_recombination_env",
]
