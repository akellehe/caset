#!/usr/bin/env python
# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Search the MultiCobordism objective with the libtorch RL — train a PPO policy to assemble
the **proton** (carry the colorless singlet {1, ω, ω²}) and benchmark it against the random
and grow-only baselines on the proton carry criterion.

The RL is a HARNESS: every macro-action is one ``MultiCobordism.buildStep`` (plus the
canonical directed cone-out/in probe), and the reward reads the engine's own published
quantities. It never reimplements proton construction — ``MultiCobordism`` + ``Proton`` stay
the source of truth. What the policy *learns* is the SEARCH: which grow / evolve / relax
macro-move, at what intensity, drives the objective ``F = ||∇S||² + Γ·r_U`` toward its floor
while the three-quark colour register emerges and the whole cobordism comes to carry the
singlet.

The hopeful result: a learned policy reaches carry rate ~1.0 (three emergent colour holes,
``r_state → 0``) — the emergent proton assembled by a policy that only ever drives the
canonical engine.

    # the proton-carry benchmark (~20-40 min): a policy that carries the singlet
    python examples/cobordism/rl_objective_search.py

    # quick smoke (seconds): a tiny env + one short iteration
    python examples/cobordism/rl_objective_search.py --smoke

    # save the trained policy (reused by the animation)
    python examples/cobordism/rl_objective_search.py --checkpoint proton_policy.pt

Requires the ``rl`` extra (libtorch): ``pip install -e ".[rl]"``.
"""
import argparse

import tessera.rl as rl


def _smoke_configs():
    """A few-second sanity config (tiny env, one short iteration)."""
    env = rl.EnvConfig()
    env.max_actions = 2
    env.grow_steps = (4, 8)
    env.evolve_steps = (2, 4)
    env.relax_iters = (1, 2)
    env.hole_reward_weight = 2.0
    env.rstate_reward_weight = 1.0
    train = rl.TrainConfig()
    train.iterations = 1
    train.episodes_per_iter = 1
    train.eval_seeds = 1
    train.hidden = 16
    train.entropy_coef_final = -1.0
    return env, train


def _print_benchmark(res):
    cols = [("RL (learned)", res.rl), ("random", res.random), ("grow-only", res.grow_only)]
    print(f"\n=== proton carry benchmark (trained in {res.train_time_s:.0f}s) ===")
    print(f"{'metric':<30}" + "".join(f"{name:>14}" for name, _ in cols))
    rows = [
        ("carry rate (-> 1.0)", "carry_rate", "{:.2f}"),
        ("mean emergent holes (-> 3)", "mean_holes", "{:.2f}"),
        ("mean r_state (-> 0)", "mean_rstate", "{:.3f}"),
        ("mean final F (lower=better)", "mean_final_F", "{:.1f}"),
        ("mean episode reward", "mean_reward", "{:+.2f}"),
    ]
    for label, attr, fmt in rows:
        print(f"{label:<30}"
              + "".join(f"{fmt.format(getattr(ev, attr)):>14}" for _, ev in cols))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["formation", "recombination"], default="formation",
                    help="formation (2->1, carry the singlet) or recombination (2->2 diquarks)")
    ap.add_argument("--smoke", action="store_true", help="a few-second sanity run")
    ap.add_argument("--iterations", type=int, default=None, help="override training iterations")
    ap.add_argument("--checkpoint", default="", help="save the trained policy to this path")
    ap.add_argument("--agent-seed", type=int, default=0)
    args = ap.parse_args()

    env, train = _smoke_configs() if args.smoke else (rl.carry_profile_env(),
                                                      rl.carry_profile_train())
    if args.iterations is not None:
        train.iterations = args.iterations
    train.agent_seed = args.agent_seed

    print(f"== training the libtorch PPO on the {args.target} target "
          f"({train.iterations} iters x {train.episodes_per_iter} eps) ==", flush=True)
    res = rl.benchmark(env, train, formation=(args.target == "formation"),
                       checkpoint_path=args.checkpoint)
    _print_benchmark(res)
    if args.checkpoint:
        print(f"\ntrained policy saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
