#!/usr/bin/env python3
"""Benchmark the exact Regge-Hodge joint objective on proton host geometry.

Examples:
    OMP_NUM_THREADS=8 python scripts/benchmark_joint_objective.py
    OMP_NUM_THREADS=8 python scripts/benchmark_joint_objective.py --drive

The default seed and precone match the scale-sensitive proton-animation run
that motivated issue #756. Timings are medians over identical evaluations on
one immutable geometry. ``--drive`` additionally times one topology pass and
one geometric update, each on its own freshly built copy because they mutate
the node.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "cobordism"))

import tessera as T  # noqa: E402
from proton_animation import build_proton_nodes  # noqa: E402


cob = T.cobordism


def build_node(args):
    cob.HodgeLaplacian.setDefaultWeightConvention(
        cob.HodgeWeightConvention.SquaredContent)
    return build_proton_nodes(
        seed=args.seed,
        precone=args.precone,
        balanced_edges=True,
        degree=args.degree,
        objective_mode="joint-stationarity",
    )[0][0]


def median_timing(function, repeats):
    durations = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precone", type=int, default=50)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drive", action="store_true",
                        help="also time one stage-one pass and stage-two update")
    args = parser.parse_args()
    if args.precone < 0 or args.seed < 0 or args.degree < 1 or args.repeats < 1:
        parser.error("precone and seed must be non-negative; degree and repeats positive")

    node = build_node(args)
    hodge_seconds, hodge_value = median_timing(
        node.hodge_entropy_stationarity, args.repeats)
    objective_seconds, objective_value = median_timing(
        node.objective, args.repeats)
    result = {
        "seed": args.seed,
        "precone": args.precone,
        "degree": args.degree,
        "cells": len(node.st.getTopSimplices()),
        "edges": len(node.st.getEdgeList().toVector()),
        "repeats": args.repeats,
        "hodge_stationarity": float(hodge_value),
        "hodge_stationarity_median_s": hodge_seconds,
        "objective": float(objective_value),
        "objective_median_s": objective_seconds,
    }

    if args.drive:
        stage_one = build_node(args)
        started = time.perf_counter()
        stage_one_trace = stage_one.run_stage1(
            max_steps=1, n_candidate_moves=12, grow_boundaries=True,
            max_lookahead=5)
        result["stage1_s"] = time.perf_counter() - started
        result["stage1_lookahead"] = int(stage_one.last_stage1_lookahead)
        result["stage1_objective"] = float(stage_one_trace[-1])

        stage_two = build_node(args)
        started = time.perf_counter()
        stage_two_trace = stage_two.run_stage2(
            beta=1.0, max_iters=1, alpha0=0.05, tolerance=1e-6)
        result["stage2_s"] = time.perf_counter() - started
        result["stage2_objective"] = float(stage_two_trace[-1])

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
