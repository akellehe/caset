"""Interaction-history Monte Carlo: locating the spectral dimension = 4 point.

See docs/source/interaction-history-monte-carlo.md for the charter.

A set of randomized correlated quantum systems on a Poisson-Delaunay
initial layer interact pairwise; each interaction attaches a (2,3) cell
to a simplicial complex, with edge lengths from mutual information
(d = -log I) and the conservation-law bookkeeping. Which interactions
occur is sampled by Metropolis-Hastings from the geometric Regge action
e^{-beta S}.

The object of the search is the inverse temperature beta at which the
emergent heat-kernel spectral dimension of the interaction-history
complex reaches 4 -- the 3+1-dimensional phase. At beta -> 0 the complex
grows freely (the action is irrelevant); at large beta growth is
suppressed; the emergent-dimension transition is in between.

This script scans beta, builds the complex at each, measures the peak
spectral dimension, and reports where D_S crosses 4.

Run::

    OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \\
    MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \\
        python examples/quantum/interaction_history_spectral_dimension.py \\
            --N 9 --target-interactions 40 --out-json /tmp/interaction-history/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial import Delaunay

from tessera.quantum import InteractionConfig, InteractionSimulation


def poisson_delaunay_edges(n_systems: int, rng: np.random.Generator):
    """Poisson-distribute n_systems points in the unit square and return
    the Delaunay edge set as 0-based index pairs."""
    points = rng.uniform(0.0, 1.0, size=(n_systems, 2))
    edges = set()
    for simplex in Delaunay(points).simplices:
        i, j, k = (int(x) for x in simplex)
        for a, b in ((i, j), (j, k), (i, k)):
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def peak_spectral_dimension(sim: InteractionSimulation,
                            sigmas: list) -> float:
    """Peak finite D_S(sigma) of the simulation's current complex."""
    d_s = sim.getSpectralDimension(sigmas)
    finite = [d for d in d_s if math.isfinite(d)]
    return max(finite) if finite else float("nan")


def run_beta_point(n_systems: int, delaunay_edges, m_over_g: float,
                   dt: float, beta: float, target_interactions: int,
                   sweeps: int, sigmas: list, seed: int) -> dict:
    """Build the complex at one beta, equilibrate, measure peak D_S."""
    cfg = InteractionConfig()
    cfg.nSystems = n_systems
    cfg.a = 1.0
    cfg.g = 1.0
    cfg.m = m_over_g
    cfg.dt = dt
    cfg.beta = beta
    cfg.epsilonI = 1e-10
    cfg.targetInteractions = target_interactions
    cfg.delaunayEdges = delaunay_edges
    cfg.seed = seed
    cfg.quiet = True

    sim = InteractionSimulation(cfg)
    sim.thermalize()                 # tune to the target volume, then
                                     # sweep to equilibrium
    for _ in range(sweeps):          # extra equilibration sweeps
        sim.sweep()

    return {
        "beta": beta,
        "seed": seed,
        "interaction_count": sim.interactionCount,
        "action": sim.computeAction(),
        "acceptance": dict(sim.getAcceptanceRates()),
        "peak_dS": peak_spectral_dimension(sim, sigmas),
        "volume_profile": list(sim.getVolumeProfile()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=9,
                   help="initial-layer system count (capped by the 2^N "
                        "correlated-state build; keep <= 11)")
    p.add_argument("--m-over-g", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--target-interactions", type=int, default=40)
    p.add_argument("--sweeps", type=int, default=10,
                   help="equilibration sweeps after tune()")
    p.add_argument("--beta-min", type=float, default=1e-5)
    p.add_argument("--beta-max", type=float, default=1e-1)
    p.add_argument("--beta-count", type=int, default=16)
    p.add_argument("--layouts", type=int, default=4,
                   help="independent Poisson layouts averaged per beta")
    p.add_argument("--sigma-min", type=float, default=1e-2)
    p.add_argument("--sigma-max", type=float, default=1e3)
    p.add_argument("--sigma-count", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json",
                   default="/tmp/interaction-history/result.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = np.logspace(math.log10(args.beta_min),
                        math.log10(args.beta_max), args.beta_count)
    sigmas = list(np.logspace(math.log10(args.sigma_min),
                              math.log10(args.sigma_max),
                              args.sigma_count))

    print(f"[setup] N={args.N}, m/g={args.m_over_g}, dt={args.dt}, "
          f"target_interactions={args.target_interactions}, "
          f"{args.layouts} Poisson layouts/beta, "
          f"beta in [{args.beta_min:.0e}, {args.beta_max:.0e}]", flush=True)

    # Fixed set of Poisson layouts, reused across the beta grid so the
    # beta dependence is not confounded with layout noise.
    layouts = [poisson_delaunay_edges(args.N, rng)
               for _ in range(args.layouts)]

    records = []
    for beta in betas:
        runs = []
        for li, edges in enumerate(layouts):
            runs.append(run_beta_point(
                args.N, edges, args.m_over_g, args.dt, beta,
                args.target_interactions, args.sweeps, sigmas,
                seed=args.seed + li))
        peaks = np.array([r["peak_dS"] for r in runs
                          if math.isfinite(r["peak_dS"])])
        rec = {
            "beta": float(beta),
            "peak_dS_mean": float(peaks.mean()) if peaks.size else float("nan"),
            "peak_dS_std": float(peaks.std()) if peaks.size else float("nan"),
            "mean_interaction_count": float(np.mean(
                [r["interaction_count"] for r in runs])),
            "runs": runs,
        }
        records.append(rec)
        print(f"   beta={beta:.2e}  peak D_S = {rec['peak_dS_mean']:.3f} "
              f"+/- {rec['peak_dS_std']:.3f}  "
              f"(interactions ~ {rec['mean_interaction_count']:.0f})",
              flush=True)

    # Locate where peak D_S crosses 4, by linear interpolation in log-beta.
    crossing = None
    for a, b in zip(records[:-1], records[1:]):
        da, db = a["peak_dS_mean"] - 4.0, b["peak_dS_mean"] - 4.0
        if math.isfinite(da) and math.isfinite(db) and da * db < 0.0:
            la, lb = math.log10(a["beta"]), math.log10(b["beta"])
            crossing = 10.0 ** (la + (lb - la) * (-da) / (db - da))
            break

    if crossing is not None:
        print(f"[result] peak D_S crosses 4 at beta ~= {crossing:.3e}")
    else:
        peak = max((r["peak_dS_mean"] for r in records
                    if math.isfinite(r["peak_dS_mean"])), default=float("nan"))
        print(f"[result] peak D_S does not reach 4 in this beta range "
              f"(max peak D_S = {peak:.3f})")

    out = {
        "config": vars(args),
        "betas": [float(b) for b in betas],
        "records": records,
        "dS_eq_4_beta": crossing,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
