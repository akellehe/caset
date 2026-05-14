"""Plot Test 1 — T-scan (evolution-time scan) at fixed N=40.

Reads /tmp/temporal-entangled/t_scan/N40_T*_mg_*.json (or the legacy
/tmp/temporal-entangled/k_scan/N40_K*_mg_*.json from earlier runs) and
renders a figure with:

  • Top row: peak D_S vs T, hop diameter vs T, D_S(σ) at largest T.
  • Middle row: peak D_S trajectory across T per m/g.
  • Bottom rows: six MI histograms (3 spatial + 3 temporal) at
    t=1·dt, t=T_max/2, t=T_max — for the representative cell at the
    largest T and m/g=0.5.

T is the controlled variable (physical evolution time). The snapshot
count K = T/dt + 1 is shown as a secondary annotation on the x-axis
only — it isn't an independent variable here (dt is held fixed).

Output: docs/source/quantum-experiments/figures/t_scan.png.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_common import load_scan, draw_six_histograms  # noqa: E402


def _annotate_fields(rows):
    """Read T, N, m/g from each JSON's config block (filename-agnostic)."""
    for r in rows:
        cfg = r.get("config", {})
        r["N"]  = int(cfg.get("N", 0))
        r["T"]  = float(cfg.get("T", 0.0))
        r["mg"] = float(cfg.get("m_over_g", 0.0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-dir", default="/tmp/temporal-entangled/t_scan")
    p.add_argument("--out-png",
                    default=("/home/andrew/tessera/docs/source/"
                             "quantum-experiments/figures/t_scan.png"))
    args = p.parse_args()

    rows = load_scan(args.scan_dir)
    # Backward-compat: fall back to the legacy k_scan dir.
    if not rows:
        legacy = "/tmp/temporal-entangled/k_scan"
        rows = load_scan(legacy)
        if rows:
            print(f"(falling back to legacy scan dir: {legacy})")
    if not rows:
        raise SystemExit(f"no JSON in {args.scan_dir} or legacy k_scan/")
    _annotate_fields(rows)
    rows.sort(key=lambda r: (r["T"], r["mg"]))

    mgs = sorted({r["mg"] for r in rows})
    Ts  = sorted({r["T"]  for r in rows})

    mg_color = dict(zip(mgs, ["C0", "C1", "C2", "C3", "C4"]))

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    for mg in mgs:
        cell_rows = [r for r in rows if r["mg"] == mg]
        xs = [r["T"]        for r in cell_rows]
        ys = [r["peak_dS"]  for r in cell_rows]
        axA.plot(xs, ys, "o-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
        diam = [r["graph"]["diameter"]["hop_diameter"]
                  for r in cell_rows]
        axB.plot(xs, diam, "s-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
    axA.axhline(4.0, color="black", linestyle=":", linewidth=1, label="$D_S=4$")
    axA.axhline(2.0, color="grey", linestyle=":", linewidth=1, label="$D_S=2$")
    axA.set_xlabel(r"evolution time $T$")
    axA.set_ylabel(r"peak $D_S$")
    axA.set_title("Peak spectral dimension vs $T$ (N=40)")
    axA.set_xticks(Ts)
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=8)

    axB.set_xlabel(r"evolution time $T$")
    axB.set_ylabel("hop diameter")
    axB.set_title("Graph hop diameter vs $T$")
    axB.set_xticks(Ts)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=8)

    Tmax = max(Ts)
    largeT_rows = [r for r in rows if r["T"] == Tmax]
    for r in largeT_rows:
        sigmas = np.array(r["sigmas"])
        dSs    = np.array(r["dS_smoothed"])
        al     = r["ambjorn_loll"]
        finite = np.isfinite(dSs)
        color  = mg_color[r["mg"]]
        axC.plot(sigmas[finite], dSs[finite],
                  color=color, linewidth=1.5,
                  label=f"data m/g={r['mg']}")
        fit = al["D_infinity"] - al["C"] / (al["B"] + sigmas)
        axC.plot(sigmas, fit,
                  color=color, linewidth=1.0, linestyle="--",
                  alpha=0.6,
                  label=f"AL m/g={r['mg']} ($D_\\infty$={al['D_infinity']:+.2f})")
    axC.axhline(4.0, color="black", linestyle=":", linewidth=1)
    axC.axhline(2.0, color="grey", linestyle=":", linewidth=1)
    axC.set_xscale("log")
    axC.set_xlabel(r"$\sigma$")
    axC.set_ylabel(r"$D_S(\sigma)$")
    axC.set_title(f"$D_S(\\sigma)$ at T={Tmax} (largest)")
    axC.grid(True, alpha=0.3, which="both")
    axC.legend(loc="best", fontsize=7)

    axD = fig.add_subplot(gs[1, :])
    for mg in mgs:
        cell_rows = [r for r in rows if r["mg"] == mg]
        xs = [r["T"]        for r in cell_rows]
        ys = [r["peak_dS"]  for r in cell_rows]
        axD.plot(xs, ys, "o-", color=mg_color[mg],
                  markersize=10, linewidth=2.0, label=f"m/g={mg}")
    axD.axhline(4.0, color="black", linestyle=":", linewidth=1.5,
                 label="$D_S=4$ (H_4D target)")
    axD.set_xlabel(r"evolution time $T$")
    axD.set_ylabel(r"peak $D_S$")
    axD.set_title("T-scan trajectory: does the peak plateau at 4?")
    axD.set_xticks(Ts)
    axD.grid(True, alpha=0.3)
    axD.legend(loc="best", fontsize=10)

    rep = next((r for r in rows
                 if r["T"] == Tmax and r["mg"] == max(mgs)), None)
    if rep is None:
        rep = rows[-1]
    hax = np.array([[fig.add_subplot(gs[2, c]) for c in range(3)],
                     [fig.add_subplot(gs[3, c]) for c in range(3)]])
    draw_six_histograms(hax, rep)
    fig.text(0.5, 0.485,
             f"MI histograms by time slice — representative cell "
             f"(N={rep['config']['N']}, T={rep['config']['T']}, "
             f"m/g={rep['config']['m_over_g']}, "
             f"ε={rep['config']['epsilon_I']:.0e})",
             ha="center", fontsize=12)

    fig.suptitle("Test 1 — T-scan at N=40 (H_4D evolution-time check)",
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
