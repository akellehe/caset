"""Histograms of spatial vs temporal bond mutual information.

Runs a TDVP quench at a chosen (N, m/g) cell, collects all spatial
bond-MI values (within-snapshot, upper triangle) and all temporal
endpoint-averaged values (across snapshot pairs up to a stride cap),
and renders log-log histograms with candidate epsilon thresholds
overlaid.

The threshold ``epsilon_I`` in
``temporally_connected_entangled_spacetime.py`` drops edges with
``MI < epsilon_I``. This plot shows where various candidate thresholds
sit on the actual MI distribution — the right tail.

Default cell is a single representative ``(N, m/g)``; pass --N and
--m-over-g (multi-valued) to stack distributions across cells.
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tessera.quantum import SchwingerQuench, TDVPConfig
from examples.quantum.temporally_connected_entangled_spacetime import (
    _bond_matrices,
)


DEFAULT_THRESHOLDS = [1e-8, 1e-6, 1e-4]
DEFAULT_OUT_PNG = ("/home/andrew/tessera/docs/source/quantum-experiments/"
                    "figures/temporally_connected_mi_histograms.png")


def _config(N, mg, T=1.0, dt=0.25, max_bond=80):
    cfg = TDVPConfig()
    cfg.N = N
    cfg.a = 1.0
    cfg.g = 1.0
    cfg.m = mg * 1.0
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 64
    cfg.dmrgNSweeps    = 12
    cfg.dmrgKrylovDim  = 4
    cfg.dmrgCutoff     = 1e-12
    cfg.i0 = 3
    cfg.d  = 3
    cfg.quenchEnforceParity = True
    cfg.dt = dt
    cfg.T  = T
    cfg.snapshotEvery = 1
    cfg.maxBondDim = max_bond
    cfg.cutoff     = 1e-10
    cfg.krylovDim  = 12
    cfg.quiet      = True
    cfg.conserveQns = True
    cfg.recordBondMutualInformation = True
    return cfg


def _harvest(snapshots, max_stride):
    """Return (spatial_values, temporal_values) flat arrays."""
    bond_mi = _bond_matrices(snapshots)
    K, B, _ = bond_mi.shape

    # Spatial: per-snapshot upper-triangle of bond MI.
    sp = []
    iu, ju = np.triu_indices(B, k=1)
    for t in range(K):
        sp.append(bond_mi[t][iu, ju])
    spatial = np.concatenate(sp) if sp else np.empty(0)

    # Temporal: endpoint average over snapshot pairs (t, t+s) with
    # 1 <= s <= max_stride. All bond pairs (n, m) including diagonal.
    if max_stride <= 0:
        max_stride = K - 1
    tm = []
    for t in range(K - 1):
        s_max = min(max_stride, K - 1 - t)
        for s in range(1, s_max + 1):
            avg = 0.5 * (bond_mi[t] + bond_mi[t + s])
            tm.append(avg.ravel())
    temporal = np.concatenate(tm) if tm else np.empty(0)

    return spatial, temporal


def _hist_panel(ax, values, title, color, thresholds):
    values = values[values > 0]
    if values.size == 0:
        ax.text(0.5, 0.5, "no positive MI values",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    lo = max(1e-16, float(values.min()))
    hi = max(lo * 10, float(values.max()))
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    ax.hist(values, bins=bins, color=color, alpha=0.7,
            edgecolor="black", linewidth=0.3)
    for eps in thresholds:
        ax.axvline(eps, color="red", linestyle="--", linewidth=1)
        frac = float((values >= eps).mean())
        ax.text(eps, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 1 else 1,
                f"  $\\varepsilon$={eps:g}\n  kept {frac*100:.1f}%",
                rotation=90, fontsize=8, color="red",
                va="top", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MI value (nats)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"{title}\n n={values.size}, "
                  f"range=[{lo:.2g}, {hi:.2g}]")
    ax.grid(True, alpha=0.3, which="both")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N",  type=int,   default=20)
    p.add_argument("--m-over-g", type=float, default=0.5)
    p.add_argument("--T",  type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--max-bond-dim", type=int, default=80)
    p.add_argument("--max-temporal-stride", type=int, default=10)
    p.add_argument("--thresholds", type=float, nargs="+",
                    default=DEFAULT_THRESHOLDS)
    p.add_argument("--out-png", default=DEFAULT_OUT_PNG)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    cfg = _config(args.N, args.m_over_g, args.T, args.dt, args.max_bond_dim)
    print(f"[setup] N={args.N}, m/g={args.m_over_g}, "
          f"T={args.T}, dt={args.dt}, stride={args.max_temporal_stride}",
          flush=True)
    t0 = time.perf_counter()
    res = SchwingerQuench(cfg).evolve()
    print(f"[tdvp] {len(res.snapshots)} snapshots in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    spatial, temporal = _harvest(res.snapshots, args.max_temporal_stride)
    print(f"[hist] spatial:  n={spatial.size}  "
          f"min={spatial.min():.2e}  max={spatial.max():.2e}  "
          f"median={np.median(spatial):.2e}", flush=True)
    print(f"[hist] temporal: n={temporal.size}  "
          f"min={temporal.min():.2e}  max={temporal.max():.2e}  "
          f"median={np.median(temporal):.2e}", flush=True)
    for eps in args.thresholds:
        f_sp = (spatial  >= eps).mean()
        f_tm = (temporal >= eps).mean()
        print(f"  eps={eps:.0e}  spatial-kept={f_sp*100:5.1f}%  "
              f"temporal-kept={f_tm*100:5.1f}%", flush=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    _hist_panel(axL, spatial,  "Spatial bond MI (within snapshot)",
                "tab:blue",   args.thresholds)
    _hist_panel(axR, temporal, "Temporal endpoint-averaged MI "
                                "(across snapshot pairs)",
                "tab:orange", args.thresholds)
    fig.suptitle(f"Bond-MI distributions at N={args.N}, "
                  f"m/g={args.m_over_g}, T={args.T}, dt={args.dt}, "
                  f"stride={args.max_temporal_stride}",
                  y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"[wrote] {args.out_png}", flush=True)


if __name__ == "__main__":
    main()
