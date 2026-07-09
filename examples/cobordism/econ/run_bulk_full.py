"""Bounded interactive bulk run on the full economy — tessera#602.

Three year pairs (calm / financial crisis / pandemic) through the
tessera-native cylinder protocol: anchor + observed + random verdicts
per pair, then a bounded surgery loop on the crisis pair. Budgets are
deliberately small (the proton "fixed-budget single-node" regime, not
the overnight campaign): every stage prints wall time so the run can be
watched interactively via the log.
"""

from __future__ import annotations

import time

import numpy as np

from econ_register import load_year, build_money_flows, build_register
from econ_cobordism_tessera import (
    build_cylinder_spacetime, chart_verdict, register_periods, surgery_step,
)
from leak_experiment import edge_vector

PAIRS = [(2005, 2006, "calm"), (2007, 2008, "crisis"), (2019, 2020, "pandemic")]
SURGERY_PAIR = (2007, 2008)
SURGERY_STEPS = 3
SURGERY_CANDIDATES = 6


def main() -> None:
    rng = np.random.default_rng(0)
    flows = {}
    for y in sorted({y for a, b, _ in PAIRS for y in (a, b)}):
        flows[y] = build_money_flows(load_year("data", y))[0]

    surgery_ctx = None
    for t0y, t1y, label in PAIRS:
        t_start = time.time()
        reg = build_register(flows[t0y], 0.0, "conductance")
        bulk = build_cylinder_spacetime(reg)
        p_t = register_periods(reg, reg.net)
        p_t1 = register_periods(reg, edge_vector(reg, flows[t1y])[0])
        dp = float(np.linalg.norm(p_t1 - p_t) ** 2
                   / (np.linalg.norm(p_t) ** 2 + np.linalg.norm(p_t1) ** 2))
        print(f"[{label}] {t0y}->{t1y}: V={len(reg.vertices)} "
              f"E={len(reg.edges)} T={len(reg.triangles)} b1={reg.b1} "
              f"(build {time.time()-t_start:.0f}s)", flush=True)

        cache: dict = {}
        t0 = time.time()
        anchor = chart_verdict(bulk, p_t, p_t, cache)
        print(f"  anchor   r = {anchor['r']:.3e}  "
              f"harmonic_dim = {anchor['harmonic_dim']}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        obs = chart_verdict(bulk, p_t, p_t1, cache)
        print(f"  observed r = {obs['r']:.3e}   (dp scale {dp:.3e})", flush=True)
        noise = rng.normal(size=len(p_t)) * np.linalg.norm(p_t) / len(p_t) ** 0.5
        rand = chart_verdict(bulk, p_t, noise, cache)
        print(f"  random   r = {rand['r']:.3e}", flush=True)
        print(f"  separation: observed/anchor = {obs['r']/max(anchor['r'],1e-300):.1f}x,"
              f" random/observed = {rand['r']/max(obs['r'],1e-300):.1f}x", flush=True)

        if (t0y, t1y) == SURGERY_PAIR:
            surgery_ctx = (bulk, p_t, p_t1, obs["r"])

    if surgery_ctx is not None:
        bulk, p_t, p_t1, r0 = surgery_ctx
        print(f"\nsurgery on {SURGERY_PAIR[0]}->{SURGERY_PAIR[1]} "
              f"({SURGERY_STEPS} steps x {SURGERY_CANDIDATES} candidates):",
              flush=True)
        hist = [r0]
        for step in range(SURGERY_STEPS):
            t0 = time.time()
            out = surgery_step(bulk, p_t, p_t1,
                               n_candidates=SURGERY_CANDIDATES, seed=step)
            b = out["best"]
            names = None
            if b is not None:
                names = tuple(bulk.reg.vertices[v] for v in b["edge"])
                hist.append(b["after"]["r"] if out["accepted"] else hist[-1])
            print(f"  step {step}: edge {names}  gain {b['gain'] if b else 0:+.3e}"
                  f"  accepted={out['accepted']}  r -> {hist[-1]:.3e}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        print("r trajectory:", " -> ".join(f"{r:.3e}" for r in hist), flush=True)


if __name__ == "__main__":
    main()
