"""Plot Test 3 — N-only scan at fixed K=5.

Reads /tmp/temporal-entangled/n_scan/N*_K5_mg_*.json and renders:

  • Top row: peak D_S vs N, hop diameter vs N, mean degree vs N.
  • Middle row: D_S(σ) curves with AL fit overlay across N values.
  • Bottom rows: six MI histograms (3 spatial + 3 temporal) at
    t=1·dt, t=T_max/2, t=T_max — representative cell at largest N.

H_4D passes iff peak D_S plateaus near 4 from below as N grows.

Output: docs/source/quantum-experiments/figures/n_scan.png.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_common import load_scan, draw_six_histograms  # noqa: E402


def _annotate_fields(rows):
    """Read N, T, m/g from each JSON's config block (filename-agnostic)."""
    for r in rows:
        cfg = r.get("config", {})
        r["N"]  = int(cfg.get("N", 0))
        r["T"]  = float(cfg.get("T", 0.0))
        r["mg"] = float(cfg.get("m_over_g", 0.0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-dir", default="/tmp/temporal-entangled/n_scan")
    p.add_argument("--out-png",
                    default=("/home/andrew/tessera/docs/source/"
                             "quantum-experiments/figures/n_scan.png"))
    args = p.parse_args()

    rows = load_scan(args.scan_dir)
    _annotate_fields(rows)
    rows = [r for r in rows if "N" in r]
    if not rows:
        raise SystemExit(f"no JSON in {args.scan_dir}")
    rows.sort(key=lambda r: (r["mg"], r["N"]))

    mgs = sorted({r["mg"] for r in rows})
    Ns  = sorted({r["N"]  for r in rows})

    mg_color = dict(zip(mgs, ["C0", "C1", "C2", "C3", "C4"]))

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    for mg in mgs:
        cell_rows = sorted((r for r in rows if r["mg"] == mg),
                            key=lambda r: r["N"])
        xs   = [r["N"]        for r in cell_rows]
        peak = [r["peak_dS"]  for r in cell_rows]
        diam = [r["graph"]["diameter"]["hop_diameter"]
                  for r in cell_rows]
        meandeg = [r["graph"]["degree"]["mean"] for r in cell_rows]
        axA.plot(xs, peak, "o-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
        axB.plot(xs, diam, "s-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
        axC.plot(xs, meandeg, "^-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")

    axA.axhline(4.0, color="black", linestyle=":", linewidth=1, label="$D_S=4$")
    axA.axhline(2.0, color="grey", linestyle=":", linewidth=1, label="$D_S=2$")
    axA.set_xlabel(r"$N$")
    axA.set_ylabel(r"peak $D_S$")
    axA.set_title(r"Peak $D_S$ vs $N$  (T=1.0, dt=0.25)")
    if Ns:
        axA.set_xticks(Ns)
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=8)

    axB.set_xlabel(r"$N$")
    axB.set_ylabel("hop diameter")
    axB.set_title("Hop diameter vs $N$")
    if Ns:
        axB.set_xticks(Ns)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=8)

    axC.set_xlabel(r"$N$")
    axC.set_ylabel("mean degree")
    axC.set_title("Mean degree vs $N$")
    if Ns:
        axC.set_xticks(Ns)
    axC.grid(True, alpha=0.3)
    axC.legend(loc="best", fontsize=8)

    # ── Middle row: D_S(σ) curves across N
    axD = fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("viridis")
    for j, N in enumerate(Ns):
        cell = next((r for r in rows if r["N"] == N), None)
        if cell is None:
            continue
        sigmas = np.array(cell["sigmas"])
        dSs    = np.array(cell["dS_smoothed"])
        al     = cell["ambjorn_loll"]
        finite = np.isfinite(dSs)
        color = cmap(j / max(1, len(Ns) - 1))
        axD.plot(sigmas[finite], dSs[finite],
                  color=color, linewidth=1.5, label=f"data N={N}")
        fit = al["D_infinity"] - al["C"] / (al["B"] + sigmas)
        axD.plot(sigmas, fit, color=color, linewidth=0.9,
                  linestyle="--", alpha=0.6,
                  label=f"AL N={N} ($D_\\infty$={al['D_infinity']:+.2f})")
    axD.axhline(4.0, color="black", linestyle=":", linewidth=1.5,
                 label="$D_S=4$")
    axD.set_xscale("log")
    axD.set_xlabel(r"$\sigma$")
    axD.set_ylabel(r"$D_S(\sigma)$")
    axD.set_title("$D_S(\\sigma)$ across N (T=1.0)")
    axD.grid(True, alpha=0.3, which="both")
    axD.legend(loc="best", fontsize=7, ncol=2)

    # ── Bottom rows: 6 histograms for representative largest-N cell
    Nmax = max(Ns)
    rep = next((r for r in rows if r["N"] == Nmax), None)
    if rep is None:
        rep = rows[-1]
    hax = np.array([[fig.add_subplot(gs[2, c]) for c in range(3)],
                     [fig.add_subplot(gs[3, c]) for c in range(3)]])
    draw_six_histograms(hax, rep)
    fig.text(0.5, 0.485,
              f"MI histograms by time slice — N={rep['config']['N']}, "
              f"K={rep['graph']['n_snapshots']}, "
              f"m/g={rep['config']['m_over_g']}, "
              f"ε={rep['config']['epsilon_I']:.0e}",
              ha="center", fontsize=12)

    fig.suptitle("Test 3 — N-scan at T=1.0 (H_4D asymptotic check)",
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
