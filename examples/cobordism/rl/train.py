# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Train the PPO policy and benchmark it against the random + grow-only baselines (#537,
extended for proton carry in #546).

Two baselines bracket the learned policy: the **random** macro-level policy (uniform move +
params — the engine still draws and greedily keeps the gated moves inside `run_stage1`) and
the **grow-only** policy (always GROW at full intensity — what #539's learned policy
collapsed to). All run under the identical environment + budget, so the only difference is
the policy. The #539 headline was how far each drives the true objective F down; the #546
headline is the PROTON CRITERION — the **carry rate** (fraction of eval seeds that carry the
singlet over ≥3 emergent colour holes with `r_state` below tol), plus mean holes / mean
`r_state` / final F.

    # quick smoke (seconds): tiny env + 1 training iteration
    python examples/cobordism/rl_objective_search.py --smoke

    # the proton-carry benchmark (#546, ~30-45 min) — the default profile
    python examples/cobordism/rl_objective_search.py --target formation

    # the #539 fast F-reduction benchmark (~2-4 min)
    python examples/cobordism/rl_objective_search.py --profile fast

The flat launcher ``examples/cobordism/rl_objective_search.py`` puts ``examples/cobordism``
on ``sys.path`` so this package's relative imports resolve; ``python -m rl.train`` also
works from inside ``examples/cobordism``.
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Callable, List

import numpy as np

from .objective_env import (
    CobordismObjectiveEnv, make_formation_env, make_recombination_env,
    GROW, N_MOVES, PARAM_DIM)
from .ppo_agent import PPO, set_seed


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(float(x))), float(x))


def random_policy(obs):
    """The macro-level random baseline: a uniform move + uniform [0, 1] parameters."""
    move = int(np.random.randint(N_MOVES))
    params = np.random.rand(PARAM_DIM).astype(np.float32)
    return move, params


def grow_only_policy(obs):
    """The #539 grow-dominant baseline: always GROW at full intensity. #539's learned
    policy collapsed to all-GROW (60/0/0 over its eval), so this is the strongest
    *non-adaptive* policy and the right "did learning add anything" control for the carry
    benchmark — with a large grow budget it forms the register, but it never relaxes the
    geometry or adapts the budget per seed."""
    return GROW, np.array([1.0, 0.5], np.float32)


def run_episode(env: CobordismObjectiveEnv, policy: Callable, seed: int) -> dict:
    """Run one episode under ``policy`` (``obs -> (move, params)``) and return its stats:
    initial/final F, the log-reduction of F, the episode reward, and the carry verdict."""
    obs = env.reset(seed)
    F0 = env.metrics["F"]
    total_reward = 0.0
    done = False
    info: dict = {}
    moves: List[int] = []
    carried_ever = False
    while not done:
        move, params = policy(obs)
        obs, reward, done, info = env.step((move, params))
        total_reward += reward
        moves.append(int(move))
        carried_ever = carried_ever or bool(info["carried"])
    return {
        "seed": seed, "F0": F0, "F_final": info["F"],
        "log_reduction": _slog(F0) - _slog(info["F"]),
        "reward": total_reward, "rstate": info["rstate"], "holes": info["holes"],
        # `carried` = the proton verdict on the FINAL state (what `Proton.build()` reads);
        # `carried_ever` = carried at any step, so the two agreeing confirms a later relax
        # never un-carries the register.
        "carried": bool(info["carried"]), "carried_ever": carried_ever,
        "n_actions": len(moves),
        "move_counts": [moves.count(m) for m in range(N_MOVES)],
    }


def evaluate(env: CobordismObjectiveEnv, policy: Callable, seeds: List[int]) -> dict:
    """Aggregate ``run_episode`` stats over ``seeds`` into mean/std summaries."""
    episodes = [run_episode(env, policy, s) for s in seeds]
    arr = lambda key: np.array([e[key] for e in episodes], dtype=float)
    return {
        "episodes": episodes,
        "mean_F_final": float(arr("F_final").mean()),
        "median_F_final": float(np.median(arr("F_final"))),
        "std_F_final": float(arr("F_final").std()),
        "mean_log_reduction": float(arr("log_reduction").mean()),
        "median_log_reduction": float(np.median(arr("log_reduction"))),
        "std_log_reduction": float(arr("log_reduction").std()),
        "mean_reward": float(arr("reward").mean()),
        "mean_rstate": float(np.nanmean(arr("rstate"))),
        "mean_holes": float(arr("holes").mean()),
        "carry_rate": float(arr("carried").mean()),
        "carry_rate_ever": float(arr("carried_ever").mean()),
    }


def train(env: CobordismObjectiveEnv, ppo: PPO, n_iterations: int,
          episodes_per_iter: int, train_seeds: List[int], verbose: bool = True,
          entropy_coef_final: float = None) -> List[dict]:
    """Collect ``episodes_per_iter`` episodes per iteration and run one PPO update on the
    pooled transitions. Returns the per-iteration training history.

    If ``entropy_coef_final`` is given, the entropy bonus is linearly annealed from
    ``ppo.entropy_coef`` down to it across the run — explore the grow/evolve/relax mix
    early, then commit to the high-reward sequence (grow the register, then relax). This
    is the fix for the policy either collapsing to a single move (too little exploration)
    or never committing (too much)."""
    history = []
    seed_cycle = iter(_cycle(train_seeds))
    entropy_coef_start = ppo.entropy_coef
    for iteration in range(n_iterations):
        if entropy_coef_final is not None and n_iterations > 1:
            frac = iteration / (n_iterations - 1)
            ppo.entropy_coef = (entropy_coef_start
                                + frac * (entropy_coef_final - entropy_coef_start))
        batch = []
        returns = []
        for _ in range(episodes_per_iter):
            transitions, info = ppo.collect_episode(env, next(seed_cycle))
            batch.extend(transitions)
            returns.append(sum(t.reward for t in transitions))
        stats = ppo.update(batch)
        rec = {"iteration": iteration, "mean_return": float(np.mean(returns)),
               "n_transitions": len(batch), "entropy_coef": ppo.entropy_coef, **stats}
        history.append(rec)
        if verbose:
            print(f"  iter {iteration:3d}  mean_return={rec['mean_return']:+.3f}  "
                  f"policy_loss={rec.get('policy_loss', float('nan')):+.4f}  "
                  f"value_loss={rec.get('value_loss', float('nan')):.4f}  "
                  f"entropy={rec.get('entropy', float('nan')):.3f}  "
                  f"ent_coef={ppo.entropy_coef:.3f}", flush=True)
    return history


def _cycle(items):
    while True:
        for x in items:
            yield x


def benchmark(target: str = "formation", iterations: int = 18, episodes_per_iter: int = 6,
              eval_seeds: int = 12, max_actions: int = 5, hidden: int = 64, lr: float = 7e-4,
              update_epochs: int = 8, entropy_coef: float = 0.01,
              entropy_coef_final: float = None, eval_deterministic: bool = True,
              agent_seed: int = 0, hole_reward_weight: float = 0.0,
              rstate_reward_weight: float = 0.0, carry_bonus: float = None,
              eval_grow_only: bool = True, env_kwargs: dict = None,
              verbose: bool = True) -> dict:
    """Train PPO and compare it to the random AND grow-only baselines on the chosen target.

    Returns a dict with the RL / random / grow-only evaluation summaries and the training
    history. The same env (same config) and the same held-out evaluation seeds are used for
    every policy, so the only difference is the policy.

    The proton-carry knobs (#546): pass ``hole_reward_weight`` / ``rstate_reward_weight`` to
    turn on the dense shaping that points the policy at the carry outcome, ``carry_bonus`` to
    weight the terminal carry reward, and big ``grow_steps`` in ``env_kwargs`` so a single
    GROW has enough budget to actually form the 3-quark register (the foundation's tiny
    ``grow_steps`` is exactly why no hole ever formed). Defaults keep the original
    F-reduction config (shaping off), so existing callers are unchanged."""
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.setdefault("max_actions", max_actions)
    env_kwargs.setdefault("grow_steps", (3, 6))
    env_kwargs.setdefault("evolve_steps", (3, 6))
    env_kwargs.setdefault("relax_iters", (2, 4))
    env_kwargs.setdefault("n_candidate_moves", 4)
    env_kwargs.setdefault("hole_reward_weight", hole_reward_weight)
    env_kwargs.setdefault("rstate_reward_weight", rstate_reward_weight)
    if carry_bonus is not None:
        env_kwargs.setdefault("carry_bonus", carry_bonus)
    builder = make_recombination_env if target == "recombination" else make_formation_env

    set_seed(agent_seed)
    env = builder(**env_kwargs)
    ppo = PPO(env.obs_dim, env.n_moves, env.param_dim, hidden=hidden, lr=lr,
              update_epochs=update_epochs, entropy_coef=entropy_coef)

    # Train and eval on disjoint seed sets so the benchmark measures generalization, not
    # memorization of a single trajectory.
    train_seeds = list(range(100, 100 + max(8, episodes_per_iter * 2)))
    held_out = list(range(eval_seeds))

    if verbose:
        print(f"== training PPO on the {target} target "
              f"({iterations} iters x {episodes_per_iter} eps, max_actions={max_actions}) ==",
              flush=True)
    t0 = time.time()
    history = train(env, ppo, iterations, episodes_per_iter, train_seeds, verbose=verbose,
                    entropy_coef_final=entropy_coef_final)
    train_time = time.time() - t0

    if verbose:
        print(f"== evaluating on held-out seeds {held_out} ==", flush=True)
    rl_eval = evaluate(
        env, lambda o: ppo.select_action(o, deterministic=eval_deterministic), held_out)
    set_seed(agent_seed + 1)  # independent randomness for the baseline
    rand_eval = evaluate(env, random_policy, held_out)
    grow_eval = evaluate(env, grow_only_policy, held_out) if eval_grow_only else None

    result = {"target": target, "train_time_s": train_time, "history": history,
              "rl": rl_eval, "random": rand_eval, "grow_only": grow_eval,
              "eval_seeds": held_out}
    if verbose:
        _print_comparison(result)
    return result


def _print_comparison(result: dict) -> None:
    rl, rnd, grow = result["rl"], result["random"], result.get("grow_only")
    cols = [("RL (learned)", rl), ("random", rnd)]
    if grow is not None:
        cols.append(("grow-only", grow))
    print("\n=== carry benchmark "
          f"({result['target']}, {len(result['eval_seeds'])} held-out seeds, "
          f"trained in {result['train_time_s']:.1f}s) ===")
    header = f"{'metric':<32}" + "".join(f"{name:>16}" for name, _ in cols)
    print(header)
    # The PROTON criterion (#546) leads; the F-proxy metrics follow as context.
    rows = [
        ("carry rate (proton criterion)", "carry_rate", "{:.2f}"),
        ("mean emergent holes (->3)", "mean_holes", "{:.2f}"),
        ("mean r_state (->0)", "mean_rstate", "{:.3f}"),
        ("median final F (lower=better)", "median_F_final", "{:.1f}"),
        ("mean final F (lower=better)", "mean_F_final", "{:.1f}"),
        ("median log-reduction of F", "median_log_reduction", "{:+.3f}"),
        ("mean episode reward", "mean_reward", "{:+.3f}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:<32}" + "".join(f"{fmt.format(ev[key]):>16}" for _, ev in cols)
        print(line)
    print(f"\nPROTON carry rate -- RL {rl['carry_rate']:.2f} vs random "
          f"{rnd['carry_rate']:.2f}" +
          (f" vs grow-only {grow['carry_rate']:.2f}" if grow is not None else "") +
          f"  (mean holes RL {rl['mean_holes']:.2f}, r_state RL {rl['mean_rstate']:.3f})")


# The proton-carry training profile (#546): the long-horizon, proton-shaped config under
# which the policy actually forms the singlet. A single GROW gets a big `run_stage1` budget
# (the foundation's tiny `grow_steps` never formed a hole — `run_stage1`'s grow-burst
# recovery + patience only act WITHIN one call); dense hole + r_state shaping plus a strong
# terminal carry bonus point the policy at the carry outcome F alone is too flat to find; a
# short action horizon (grow, then relax) keeps each episode affordable. The engine knobs
# (patience 15, n_candidate_moves 8) match `Proton.build()`'s init/evolve drive, so the RL
# arc and the reference build see the identical engine.
CARRY_PROFILE = dict(
    max_actions=4, hole_reward_weight=2.0, rstate_reward_weight=1.0, carry_bonus=10.0,
    entropy_coef=0.03, entropy_coef_final=0.005,
    # grow_steps caps a single GROW: high enough to form the register (carriers converge in
    # ~50-120 engine steps, then run_stage1 breaks early once carried -- a carrying GROW
    # costs ~40-60s, and any further GROW on a carried state is a ~1s no-op), but bounded so
    # a NON-carrying seed -- which runs the full budget on an ever-growing complex -- does
    # not blow up the wall-clock. terminate_on_carry=False lets the policy keep going after
    # the carry to relax the geometry (the grow -> evolve -> relax arc); since post-carry
    # GROW/EVOLVE early-break, the full arc is nearly free on carrying episodes.
    # directed_grow on: a GROW finishes the register the random draws left short by a gated
    # DIRECTED cone-out probe (open the missing hole deliberately), and an EVOLVE SELECTS the
    # best target_holes by a gated cone-in probe (cap the worst over-opened hole). This
    # rescues a fraction of seeds whose random growth stalls below 3 holes (carrier seeds are
    # unaffected -- post-carry cone-out/in are no-ops), raising the carry rate.
    env_kwargs=dict(grow_steps=(50, 130), evolve_steps=(10, 40), relax_iters=(3, 8),
                    n_candidate_moves=8, patience=15, terminate_on_carry=False,
                    directed_grow=True, cone_strategy="greedy", cone_overshoot=2),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["formation", "recombination"], default="formation")
    ap.add_argument("--profile", choices=["carry", "fast"], default="carry",
                    help="carry (#546): long-horizon + proton shaping that forms the singlet "
                         "(~30-45 min); fast (#539): short-horizon F-reduction (~2-4 min)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--episodes-per-iter", type=int, default=None)
    ap.add_argument("--eval-seeds", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="a few-second sanity run: tiny env + 1 short training iteration")
    args = ap.parse_args()

    if args.smoke:
        benchmark(target=args.target, iterations=1, episodes_per_iter=1, eval_seeds=1,
                  max_actions=2, hidden=16, hole_reward_weight=2.0, rstate_reward_weight=1.0,
                  env_kwargs=dict(grow_steps=(2, 4), evolve_steps=(2, 4), relax_iters=(1, 2)))
        return

    if args.profile == "carry":
        cfg = dict(CARRY_PROFILE)
        iterations = args.iterations or 8
        episodes_per_iter = args.episodes_per_iter or 3
        eval_seeds = args.eval_seeds or 6
    else:  # fast: the #539 F-reduction config (shaping off, short horizon)
        cfg = dict(max_actions=5)
        iterations = args.iterations or 18
        episodes_per_iter = args.episodes_per_iter or 6
        eval_seeds = args.eval_seeds or 12

    benchmark(target=args.target, iterations=iterations,
              episodes_per_iter=episodes_per_iter, eval_seeds=eval_seeds,
              hidden=args.hidden, lr=args.lr, agent_seed=args.agent_seed, **cfg)


if __name__ == "__main__":
    main()
