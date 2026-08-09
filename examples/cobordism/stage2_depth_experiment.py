# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""How deep does the geometric relaxation actually go? (#630)

`Proton.build` drives `runStage2(beta, stage2MaxIters)` with `stage2MaxIters = 10`, and
the proton animation defaults to the same. That budget is nowhere near the basin floor:
on a formation node every one of the first 10 iterations is *accepted*, and
`lastStage2Stationary` is **false** — the loop exits on the iteration cap while `F` is
still falling, not because the line search ran out of descent.

This script measures the difference. It drives both nodes exactly the way
`Proton::build` does — one INIT pass (`grow_boundaries=True`), one EVOLUTION pass
(`grow_boundaries=False`), then a SINGLE `run_stage2` call — and reports the full `F`
trace, the stationarity verdict, and each term of the objective before and after.

Driving stage 2 as one call matters. `run_stage2(max_iters=N)` is NOT the same as N
calls of `run_stage2(max_iters=1)`: the backtracking line search adapts its step scale
across iterations and resets it on every fresh call. The animation advances one
iteration per frame (so it can draw), which is a different trajectory from what
`Proton.build` runs.

Measured on a formation node at 300 iterations: `F` fell 669.6 -> 62.15 (10.8x) with 292
of 300 steps accepted and `stationary = True` — so the floor is ~292 iterations out, and
roughly three quarters of the total descent happens past the default cap of 10.

Caveat worth keeping in view: stage 1's surgery is stochastic and a single node often
fails to grow the 3-hole register at all (which is exactly why `Proton.build` restarts
across up to 16 seeds). A run that ends with `holes=1` and `r_U` pinned on its zero-fill
floor is the ordinary failure mode, not a regression. Sample several seeds before
concluding anything from one.

Run:
    python examples/cobordism/stage2_depth_experiment.py
    python examples/cobordism/stage2_depth_experiment.py --iters 300 --seed 3
"""

import argparse
import json
import math
import time

import tessera as T

cob = T.cobordism


def _snapshot(node, tag):
    """Every term of `F = ||grad S||^2 + gamma * r_U`, plus the register state."""
    st = node.st
    return {
        "tag": tag,
        "cells": len(st.getTopSimplices()),
        "edges": len(st.getEdgeList().toVector()),
        "timelike": sum(1 for e in st.getEdgeList().toVector() if e.isTimelike()),
        "holes": len(cob.MultiCobordism.emergent_holes(st, 3)),
        "b3": cob.MultiCobordism.betti(st)[3],
        "grad_S_sq": cob.MultiCobordism.regge_action_gradient(st),
        "r_U": node.r_u(st),
        "F": node.objective(),
        "singlet": cob.MultiCobordism.r_state(st, 3, cob.Proton.singlet()),
        "grad_r_U_max": max((abs(x) for x in node.r_u_gradient(st)), default=0.0),
    }


def run_node(label, node, iters, init_steps, evolve_steps, beta):
    started = time.time()
    node.run_stage1(init_steps, 8, 15, True)     # INIT: grow the carrying regions
    node.run_stage1(evolve_steps, 8, 15, False)  # EVOLVE: boundary frozen
    before = _snapshot(node, "after_stage1")
    stage1_secs = time.time() - started

    started = time.time()
    trace = list(node.run_stage2(beta=beta, max_iters=iters))   # ONE call, not `iters`
    stage2_secs = time.time() - started
    after = _snapshot(node, "after_stage2")

    print(f"\n===== {label} =====")
    print(f"  after stage 1: cells={before['cells']} edges={before['edges']} "
          f"timelike={before['timelike']} holes={before['holes']} b3={before['b3']}")
    print(f"  ||grad S||^2  {before['grad_S_sq']:.6e} -> {after['grad_S_sq']:.6e}")
    print(f"  r_U           {before['r_U']:.6e} -> {after['r_U']:.6e}")
    print(f"  |grad r_U|max {before['grad_r_U_max']:.3e} -> {after['grad_r_U_max']:.3e}")
    print(f"  singlet       {before['singlet']:.6e} -> {after['singlet']:.6e}")
    print(f"  F             {trace[0]:.6e} -> {trace[-1]:.6e}")
    print(f"  accepted {len(trace) - 1}/{iters}   STATIONARY={node.last_stage2_stationary}")
    if len(trace) > 11:
        print(f"  F at the default cap of 10 iterations: {trace[10]:.6e}   "
              f"remaining descent past it: "
              f"{(trace[10] - trace[-1]) / max(abs(trace[0] - trace[-1]), 1e-300):.1%}")
    print(f"  ({stage1_secs:.1f}s stage 1, {stage2_secs:.1f}s stage 2)")

    return {"before": before, "after": after, "accepted": len(trace) - 1,
            "stationary": bool(node.last_stage2_stationary),
            "F_first": trace[0], "F_last": trace[-1], "trace": trace,
            "secs_stage1": round(stage1_secs, 1), "secs_stage2": round(stage2_secs, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300,
                    help="stage-2 iteration budget (Proton.build uses 10)")
    ap.add_argument("--seed", type=int, default=3, help="base seed")
    ap.add_argument("--init", type=int, default=180, help="INIT-pass stage-1 steps")
    ap.add_argument("--evolve", type=int, default=60, help="EVOLUTION-pass stage-1 steps")
    ap.add_argument("--beta", type=float, default=1.0, help="beta in beta*||grad S||^2 + gamma*r_U")
    ap.add_argument("--json", help="also write the full traces here")
    args = ap.parse_args()

    # The same seed convention Proton.build uses: A-seed 2i, B-seed 2i+1.
    proton = cob.Proton(args.seed, 3, 50.0, 20.0, 0, False)
    nodes = (("A_recombination", proton.recombination_node(2 * args.seed)),
             ("B_formation", proton.formation_node(2 * args.seed + 1)))

    results = {}
    for label, node in nodes:
        results[label] = run_node(label, node, args.iters, args.init, args.evolve,
                                  args.beta)

    capped = [k for k, v in results.items() if not v["stationary"]]
    print("\n" + "=" * 60)
    if capped:
        print(f"HIT THE ITERATION CAP (still descending): {', '.join(capped)}")
        print("  -> the budget is the binding constraint, not the basin floor")
    else:
        print("Both nodes reached stationarity: this budget does reach the floor.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"traces -> {args.json}")


if __name__ == "__main__":
    main()
