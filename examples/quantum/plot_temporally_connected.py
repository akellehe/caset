"""Plot the m/g x N scan from temporally_connected_entangled_spacetime.py.

Reads ``--scan-dir/N*_mg_*.json`` and renders a four-panel figure:

  • D_S(sigma) for every cell with the Ambjorn-Loll fit explicitly
    overlaid on the same axes.
  • peak D_S vs N per m/g.
  • Spatial bond-MI histogram (log-log).
  • Temporal endpoint-averaged MI histogram (log-log).

The MI histograms come from the ``mi_distributions`` block written by
the experiment script. Older JSON without that block degrades
gracefully — the histogram panels show "no data".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_scan(scan_dir):
    pat = re.compile(r"N(\d+)_mg_([\d.]+)\.json$")
    rows = []
    for path in sorted(glob.glob(os.path.join(scan_dir, "N*_mg_*.json"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        N  = int(m.group(1))
        mg = float(m.group(2))
        data = json.load(open(path))
        rows.append({"N": N, "mg": mg, "path": path, **data})
    rows.sort(key=lambda r: (r["N"], r["mg"]))
    return rows


def _al(sigma, D_inf, C, B):
    return D_inf - C / (B + np.asarray(sigma))


def _plot_dS(ax, rows, mg_color, N_style):
    """Spectral dimension panel with explicit AL overlay."""
    for r in rows:
        sigmas = np.array(r["sigmas"])
        dSs    = np.array(r["dS_smoothed"])
        al     = r["ambjorn_loll"]
        finite = np.isfinite(dSs)
        label  = f"data N={r['N']}, m/g={r['mg']}"
        ax.plot(sigmas[finite], dSs[finite],
                color=mg_color[r["mg"]], linestyle=N_style[r["N"]],
                linewidth=1.5, alpha=0.85, label=label)
        fit = _al(sigmas, al["D_infinity"], al["C"], al["B"])
        ax.plot(sigmas, fit,
                color=mg_color[r["mg"]], linestyle=N_style[r["N"]],
                linewidth=1.0, alpha=0.55, marker="x", markersize=3,
                markevery=max(1, len(sigmas) // 12),
                label=f"AL fit  N={r['N']}, m/g={r['mg']}"
                      f"  ($D_\\infty$={al['D_infinity']:+.2f})")
    ax.axhline(2.0, color="grey", linestyle=":", linewidth=1)
    ax.text(1e-2, 2.05, r"$D_S = 2$ (lattice)",
            color="grey", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$D_S(\sigma)$  (smoothed)")
    ax.set_title("Spectral dimension with Ambjorn-Loll fit overlay")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=6.5, ncol=2, frameon=False)


def _plot_peak(ax, rows, mgs, Ns, mg_color):
    for mg in mgs:
        xs = [r["N"]        for r in rows if r["mg"] == mg]
        ys = [r["peak_dS"]  for r in rows if r["mg"] == mg]
        ax.plot(xs, ys, "o-", color=mg_color[mg],
                markersize=7, linewidth=1.5, label=f"m/g={mg}")
    ax.axhline(2.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"peak $D_S$")
    ax.set_title(r"Peak $D_S$ vs $N$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    if len(Ns) > 1:
        ax.set_xticks(Ns)


def _plot_mi_hist(ax, rows, kind, mg_color, N_style):
    """Stacked histogram of MI values across all cells.

    ``kind`` is ``"spatial"`` or ``"temporal"``. Each cell contributes
    its log-binned counts; cells share an axis but their distributions
    are drawn as step plots so they don't occlude.
    """
    any_drawn = False
    for r in rows:
        mid = r.get("mi_distributions")
        if not mid:
            continue
        block = mid.get(kind, {})
        edges  = block.get("edges",  [])
        counts = block.get("counts", [])
        if not edges or not counts:
            continue
        edges  = np.array(edges,  dtype=np.float64)
        counts = np.array(counts, dtype=np.float64)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.step(centers, counts, where="mid",
                color=mg_color[r["mg"]], linestyle=N_style[r["N"]],
                linewidth=1.3, alpha=0.85,
                label=f"N={r['N']}, m/g={r['mg']}  "
                      f"med={block['median']:.1e}")
        any_drawn = True

    if not any_drawn:
        ax.text(0.5, 0.5,
                f"no {kind} MI data in JSON\n"
                "(rerun experiment script after the histogram patch)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey")
        ax.set_title(f"{kind.title()} MI distribution (missing)")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MI value (nats)")
    ax.set_ylabel("count")
    ax.set_title(f"{kind.title()} MI distribution")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-dir", default="/tmp/temporal-entangled/scan")
    p.add_argument("--out-png",
                    default=("/home/andrew/tessera/docs/source/"
                             "quantum-experiments/figures/"
                             "temporally_connected_entangled_spacetime.png"))
    args = p.parse_args()

    rows = _load_scan(args.scan_dir)
    if not rows:
        raise SystemExit(f"no JSON found under {args.scan_dir}")

    mgs = sorted({r["mg"] for r in rows})
    Ns  = sorted({r["N"]  for r in rows})

    mg_color = {mg: c for mg, c in
                 zip(mgs, ["C0", "C1", "C2", "C3", "C4"])}
    N_style  = {N:  s for N,  s in
                 zip(Ns,  ["-", "--", "-.", ":", (0, (3, 1, 1, 1))])}

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (axTL, axTR), (axBL, axBR) = axes

    _plot_dS(axTL,       rows, mg_color, N_style)
    _plot_peak(axTR,     rows, mgs, Ns, mg_color)
    _plot_mi_hist(axBL,  rows, "spatial",  mg_color, N_style)
    _plot_mi_hist(axBR,  rows, "temporal", mg_color, N_style)

    fig.suptitle(
        f"Temporally-connected entangled spacetime — scan over "
        f"{args.scan_dir}",
        y=0.99)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
