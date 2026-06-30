# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Train the PPO policy and benchmark it against the random baseline (#537).

The baseline is the *current* search policy at the macro level: choose each surgery /
relaxation macro-action (and its parameters) uniformly at random — the engine still draws
and greedily keeps the actual gated moves inside `run_stage1`. The RL policy instead
LEARNS which macro-action to take. Both run under the identical environment and action
budget, so the comparison is apples-to-apples; the headline metric is how far each drives
the true objective F down from the (fixed) seed state, plus how often the target color
state is carried.

    # quick smoke (seconds): tiny env + 1 training iteration
    python examples/cobordism/rl_objective_search.py --smoke

    # a real (minutes) benchmark on the small formation target
    python examples/cobordism/rl_objective_search.py --target formation \
        --iterations 22 --episodes-per-iter 6 --eval-seeds 12

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
    CobordismObjectiveEnv, make_formation_env, make_recombination_env, N_MOVES, PARAM_DIM)
from .ppo_agent import PPO, set_seed


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(float(x))), float(x))


def random_policy(obs):
    """The macro-level random baseline: a uniform move + uniform [0, 1] parameters."""
    move = int(np.random.randint(N_MOVES))
    params = np.random.rand(PARAM_DIM).astype(np.float32)
    return move, params


def run_episode(env: CobordismObjectiveEnv, policy: Callable, seed: int) -> dict:
    """Run one episode under ``policy`` (``obs -> (move, params)``) and return its stats:
    initial/final F, the log-reduction of F, the episode reward, and the carry verdict."""
    obs = env.reset(seed)
    F0 = env.metrics["F"]
    total_reward = 0.0
    done = False
    info: dict = {}
    moves: List[int] = []
    while not done:
        move, params = policy(obs)
        obs, reward, done, info = env.step((move, params))
        total_reward += reward
        moves.append(int(move))
    return {
        "seed": seed, "F0": F0, "F_final": info["F"],
        "log_reduction": _slog(F0) - _slog(info["F"]),
        "reward": total_reward, "rstate": info["rstate"], "holes": info["holes"],
        "carried": bool(info["carried"]), "n_actions": len(moves),
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
              agent_seed: int = 0, env_kwargs: dict = None, verbose: bool = True) -> dict:
    """Train PPO and compare it to the random baseline on the chosen small target.

    Returns a dict with the RL-vs-random evaluation summaries and the training history. The
    same env (same config) and the same held-out evaluation seeds are used for both
    policies, so the only difference is the policy.

    The default knobs are tuned so the comparison is decisive: a low entropy bonus lets the
    policy COMMIT to the high-reward macro-move rather than stay near-uniform, and the
    surgery/relaxation step ranges put the optimizer in the regime where the learnable
    strategy (front-load the register-growing surgery) clearly out-reduces F vs spending
    the same budget on random macro-moves."""
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.setdefault("max_actions", max_actions)
    env_kwargs.setdefault("grow_steps", (3, 6))
    env_kwargs.setdefault("evolve_steps", (3, 6))
    env_kwargs.setdefault("relax_iters", (2, 4))
    env_kwargs.setdefault("n_candidate_moves", 4)
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

    result = {"target": target, "train_time_s": train_time, "history": history,
              "rl": rl_eval, "random": rand_eval, "eval_seeds": held_out}
    if verbose:
        _print_comparison(result)
    return result


def _print_comparison(result: dict) -> None:
    rl, rnd = result["rl"], result["random"]
    print("\n=== RL vs random baseline "
          f"({result['target']}, {len(result['eval_seeds'])} held-out seeds, "
          f"trained in {result['train_time_s']:.1f}s) ===")
    print(f"{'metric':<26}{'RL (learned)':>16}{'random baseline':>18}")
    rows = [
        ("median final F (lower=better)", "median_F_final", "{:.2f}"),
        ("mean final F (lower=better)", "mean_F_final", "{:.2f}"),
        ("std final F", "std_F_final", "{:.2f}"),
        ("median log-reduction of F", "median_log_reduction", "{:+.3f}"),
        ("mean log-reduction of F", "mean_log_reduction", "{:+.3f}"),
        ("mean episode reward", "mean_reward", "{:+.3f}"),
        ("mean r_state (target)", "mean_rstate", "{:.3f}"),
        ("mean emergent holes", "mean_holes", "{:.2f}"),
        ("carry rate", "carry_rate", "{:.2f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<30}{fmt.format(rl[key]):>16}{fmt.format(rnd[key]):>18}")
    # Median final F is the robust headline (the final-F distribution is heavy-tailed).
    better = rl["median_F_final"] < rnd["median_F_final"]
    print(f"\nRL drives F lower than random (median): {better} "
          f"(RL {rl['median_F_final']:.2f} vs random {rnd['median_F_final']:.2f}, "
          f"Δ={rnd['median_F_final'] - rl['median_F_final']:+.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["formation", "recombination"], default="formation")
    ap.add_argument("--iterations", type=int, default=18)
    ap.add_argument("--episodes-per-iter", type=int, default=6)
    ap.add_argument("--eval-seeds", type=int, default=12)
    ap.add_argument("--max-actions", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="a few-second sanity run: tiny env + 1 short training iteration")
    args = ap.parse_args()

    if args.smoke:
        benchmark(target=args.target, iterations=1, episodes_per_iter=1, eval_seeds=1,
                  max_actions=2, hidden=16,
                  env_kwargs=dict(grow_steps=(2, 4), evolve_steps=(2, 4), relax_iters=(1, 2)))
        return
    benchmark(target=args.target, iterations=args.iterations,
              episodes_per_iter=args.episodes_per_iter, eval_seeds=args.eval_seeds,
              max_actions=args.max_actions, hidden=args.hidden, lr=args.lr,
              agent_seed=args.agent_seed)


if __name__ == "__main__":
    main()
