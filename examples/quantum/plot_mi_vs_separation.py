"""Plot van Raamsdonk distance d = -log I(n,m) vs bond separation |n-m|.

Runs a single TDVP cell, extracts the per-snapshot bond-MI matrix, and
plots:

  • spatial sector: d = -log I(n,m) vs |n-m|, multiple snapshots
    overlaid so we can see the t=0 "static" structure vs evolved.
  • temporal sector: d vs (|n-m|, |t-t'|), color-coded by snapshot
    stride s.

QCD-style confinement prediction (see writeup): for the Schwinger
ground state d(r) should grow ~linearly with |r|, slope set by the
meson mass. A flat tail at long |r| would indicate quench-driven
homogenisation has wiped out the natural decay.
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


def _config(N, mg, T, dt, max_bond):
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


def _spatial_dvr(bond_mi, eps_floor=1e-300):
    """For each snapshot t, return (|n-m|, d=-log MI) arrays."""
    K, B, _ = bond_mi.shape
    per_t = []
    for t in range(K):
        ii, jj = np.triu_indices(B, k=1)
        seps   = (jj - ii).astype(np.int64)
        mi     = bond_mi[t][ii, jj]
        d      = -np.log(np.clip(mi, eps_floor, None))
        per_t.append((seps, d, mi))
    return per_t


def _bin_mean(xs, ys, n_bins):
    """Mean y per integer x bin in xs. Returns (centers, mean, std, n)."""
    xs = np.asarray(xs); ys = np.asarray(ys)
    centers = np.arange(1, n_bins + 1)
    mean = np.full(n_bins, np.nan, dtype=float)
    std  = np.full(n_bins, np.nan, dtype=float)
    n    = np.zeros(n_bins, dtype=int)
    for k, c in enumerate(centers):
        mask = (xs == c)
        if mask.any():
            mean[k] = ys[mask].mean()
            std[k]  = ys[mask].std()
            n[k]    = int(mask.sum())
    return centers, mean, std, n


def _fit_line(x, y):
    """Linear fit y = a*x + b. Returns (a, b)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    return tuple(np.polyfit(x[finite], y[finite], 1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N",        type=int,   default=60)
    p.add_argument("--m-over-g", type=float, default=0.125)
    p.add_argument("--T",        type=float, default=2.0)
    p.add_argument("--dt",       type=float, default=0.25)
    p.add_argument("--max-bond-dim", type=int, default=80)
    p.add_argument("--out-png",
                    default=("/home/andrew/tessera/docs/source/"
                             "quantum-experiments/figures/"
                             "mi_vs_separation.png"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    cfg = _config(args.N, args.m_over_g, args.T, args.dt, args.max_bond_dim)
    print(f"[setup] N={args.N} mg={args.m_over_g} T={args.T} dt={args.dt}",
          flush=True)
    t0 = time.perf_counter()
    res = SchwingerQuench(cfg).evolve()
    print(f"[tdvp]  {len(res.snapshots)} snapshots in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    bond_mi = _bond_matrices(res.snapshots)
    K, B, _ = bond_mi.shape
    per_t = _spatial_dvr(bond_mi)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (axTL, axTR), (axBL, axBR) = axes

    # ── Top-left: spatial d vs |n-m|, overlay all snapshots ────────────
    cmap = plt.get_cmap("viridis")
    for t, (seps, d, mi) in enumerate(per_t):
        color = cmap(t / max(1, K - 1))
        centers, mean, std, n = _bin_mean(seps, d, n_bins=B - 1)
        valid = n > 0
        axTL.errorbar(centers[valid], mean[valid],
                       yerr=std[valid] / np.sqrt(np.maximum(n[valid], 1)),
                       color=color, fmt="o-", markersize=4, linewidth=1.0,
                       alpha=0.85, label=f"t={t * args.dt:.2f}")
    axTL.set_xlabel(r"bond separation $|n-m|$")
    axTL.set_ylabel(r"$d = -\log I(n,m)$  (mean ± SE)")
    axTL.set_title(f"Spatial sector at N={args.N}, m/g={args.m_over_g}")
    axTL.grid(True, alpha=0.3)
    axTL.legend(loc="best", fontsize=7, ncol=2)

    # ── Top-right: same, but only t=0 and t=T_max, with a linear fit ───
    seps0, d0, _ = per_t[0]
    sepsT, dT, _ = per_t[-1]
    axTR.scatter(seps0, d0, color="C0", alpha=0.25, s=8, label=f"t=0 raw")
    axTR.scatter(sepsT, dT, color="C3", alpha=0.25, s=8,
                  label=f"t={(K-1)*args.dt:.2f} raw")
    centers0, mean0, std0, n0 = _bin_mean(seps0, d0, n_bins=B - 1)
    centersT, meanT, stdT, nT = _bin_mean(sepsT, dT, n_bins=B - 1)
    a0, b0 = _fit_line(centers0[n0 > 0], mean0[n0 > 0])
    aT, bT = _fit_line(centersT[nT > 0], meanT[nT > 0])
    xs = np.linspace(1, B - 1, 50)
    axTR.plot(xs, a0 * xs + b0, color="C0", linewidth=2,
                label=f"t=0 fit: slope={a0:.3f} nats/site")
    axTR.plot(xs, aT * xs + bT, color="C3", linewidth=2,
                label=f"t={(K-1)*args.dt:.2f} fit: slope={aT:.3f} nats/site")
    axTR.set_xlabel(r"bond separation $|n-m|$")
    axTR.set_ylabel(r"$d = -\log I(n,m)$")
    axTR.set_title(r"Linear-fit test: $d \propto |r|$ (QCD confinement)")
    axTR.grid(True, alpha=0.3)
    axTR.legend(loc="best", fontsize=8)

    # ── Bottom-left: 2D heatmap d vs (|n-m|, t) ────────────────────────
    mean_mat = np.full((K, B - 1), np.nan, dtype=float)
    for t, (seps, d, _) in enumerate(per_t):
        for k in range(B - 1):
            mask = (seps == k + 1)
            if mask.any():
                mean_mat[t, k] = d[mask].mean()
    im = axBL.imshow(mean_mat, aspect="auto", origin="lower", cmap="magma",
                      extent=[0.5, B - 0.5, -0.5, K - 0.5])
    axBL.set_xlabel(r"bond separation $|n-m|$")
    axBL.set_ylabel("snapshot index $t$")
    axBL.set_title(r"Mean $d = -\log I$ vs ($|r|$, $t$)")
    plt.colorbar(im, ax=axBL, label="d (nats)")

    # ── Bottom-right: temporal sector d vs stride for fixed |n-m| ──────
    strides = np.arange(1, K)
    spatial_sample_seps = [1, 5, 10, 20, max(1, B - 2)]
    cmap2 = plt.get_cmap("plasma")
    for j, r in enumerate(spatial_sample_seps):
        if r >= B:
            continue
        means_per_stride = []
        for s in strides:
            vals = []
            for t in range(K - s):
                avg = 0.5 * (bond_mi[t] + bond_mi[t + s])
                ii, jj = np.indices(avg.shape)
                mask = (np.abs(ii - jj) == r)
                vals.extend(avg[mask].tolist())
            if vals:
                vals = np.array(vals)
                d_mean = -np.log(np.clip(vals.mean(), 1e-300, None))
                means_per_stride.append(d_mean)
            else:
                means_per_stride.append(np.nan)
        color = cmap2(j / max(1, len(spatial_sample_seps) - 1))
        axBR.plot(strides, means_per_stride, "o-", color=color,
                   markersize=6, linewidth=1.4,
                   label=f"$|n-m|$={r}")
    axBR.set_xlabel(r"snapshot stride $|t' - t|$")
    axBR.set_ylabel(r"$d = -\log I_{\rm temporal}$  (mean)")
    axBR.set_title("Temporal sector: how does d grow with stride?")
    axBR.grid(True, alpha=0.3)
    axBR.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"MI-derived distance vs separation — N={args.N}, m/g={args.m_over_g}",
        y=1.00)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"[wrote] {args.out_png}", flush=True)

    # Print summary stats.
    print()
    print(f"[fit] t=0     slope = {a0:.4f} nats/site,  intercept = {b0:.4f}")
    print(f"[fit] t=T_max slope = {aT:.4f} nats/site,  intercept = {bT:.4f}")


if __name__ == "__main__":
    main()
