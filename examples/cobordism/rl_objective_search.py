# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Driver for the RL cobordism objective-search foundation (#537).

A runnable entry point for the `examples/cobordism/rl/` package: it learns the SEARCH
POLICY that drives the emergent `MultiCobordism` optimizer (the random+greedy orchestration
in `Proton.build()`), with a PyTorch PPO actor-critic over BOTH engine stages — discrete
topology surgery (`run_stage1`) and continuous geometric relaxation (`run_stage2`). The
objective ``F = ‖∇S_Regge‖² + Γ·r_U``, the physics, and the C++ engine are untouched: the
policy only chooses WHICH gated macro-action (and its parameters) to apply, and every
topology move stays gated on ``dualComplexValid`` inside the engine.

This thin launcher just puts `examples/cobordism` on `sys.path` (so the `rl` package — which
uses relative imports — resolves) and re-exports the pieces, then defers to
`rl.train.main()`. The actual code lives in:

  * ``rl/objective_env.py`` — the Gym-style environment over one `MultiCobordism` node.
  * ``rl/ppo_agent.py``     — the PyTorch PPO actor-critic (hybrid discrete + continuous).
  * ``rl/train.py``         — the training loop + the random-baseline benchmark.

Examples::

    # quick smoke (seconds): tiny env + 1 short training iteration
    python examples/cobordism/rl_objective_search.py --smoke

    # the proton-carry benchmark (#546, ~30-45 min): long horizon + proton shaping, so the
    # learned policy actually FORMS the singlet (≥3 colour holes, r_state below tol) — the
    # default profile. Compares carry rate / holes / r_state vs random and grow-only.
    python examples/cobordism/rl_objective_search.py --target formation

    # the #539 fast F-reduction benchmark (~2-4 min): short horizon, shaping off
    python examples/cobordism/rl_objective_search.py --profile fast

Requires the ``rl`` extra (PyTorch): ``pip install -e ".[rl]"`` (or ``".[dev]"``, which
includes it).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Re-export the public surface so callers can `import rl_objective_search as rl` and reach
# everything from one module, exactly like the other flat cobordism example drivers.
from rl.objective_env import (  # noqa: E402,F401
    CobordismObjectiveEnv, make_formation_env, make_recombination_env,
    formation_node_factory, recombination_node_factory,
    GROW, EVOLVE, RELAX, MOVE_NAMES, N_MOVES, PARAM_DIM, OBS_DIM,
)
from rl.ppo_agent import PPO, HybridActorCritic, set_seed  # noqa: E402,F401
from rl.train import (  # noqa: E402,F401
    benchmark, train, evaluate, run_episode, random_policy, grow_only_policy,
    CARRY_PROFILE, main,
)


if __name__ == "__main__":
    main()
