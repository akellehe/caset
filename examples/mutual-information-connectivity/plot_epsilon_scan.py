"""Plot Test 2 — ε-scan at fixed (N=60, K=9).

Reads /tmp/temporal-entangled/epsilon_scan/N60_K9_mg_*_eps*.json and
renders:

  • Top row: peak D_S vs ε, hop diameter vs ε, fraction of MI candidates
    kept vs ε (one curve per m/g).
  • Middle row: D_S(σ) for one m/g with all ε overlaid.
  • Bottom rows: 6 MI histograms (3 spatial + 3 temporal) at
    t=1·dt, t=T_max/2, t=T_max — representative cell at the densest
    m/g, intermediate ε.

H_4D passes iff there is an ε window where peak D_S sits near 4.

Output: docs/source/quantum-experiments/figures/epsilon_scan.png.
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
    pat = re.compile(r"N(\d+)_K(\d+)_mg_([\d.]+)_eps([\d.e\+\-]+)\.json$")
    for r in rows:
        m = pat.search(os.path.basename(r["__path"]))
        if not m:
            continue
        r["N"]   = int(m.group(1))
        r["K"]   = int(m.group(2))
        r["mg"]  = float(m.group(3))
        r["eps"] = float(m.group(4))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-dir",
                    default="/tmp/temporal-entangled/epsilon_scan")
    p.add_argument("--out-png",
                    default=("/home/andrew/tessera/docs/source/"
                             "quantum-experiments/figures/epsilon_scan.png"))
    args = p.parse_args()

    rows = load_scan(args.scan_dir)
    _annotate_fields(rows)
    rows = [r for r in rows if "eps" in r]
    if not rows:
        raise SystemExit(f"no eps-split JSON in {args.scan_dir}")
    rows.sort(key=lambda r: (r["mg"], r["eps"]))

    mgs = sorted({r["mg"] for r in rows})
    epsilons = sorted({r["eps"] for r in rows})

    mg_color = dict(zip(mgs, ["C0", "C1", "C2", "C3", "C4"]))

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    for mg in mgs:
        cell_rows = sorted((r for r in rows if r["mg"] == mg),
                            key=lambda r: r["eps"])
        xs   = [r["eps"]      for r in cell_rows]
        ys   = [r["peak_dS"]  for r in cell_rows]
        diam = [r["graph"]["diameter"]["hop_diameter"]
                  for r in cell_rows]
        nE_sp = [r["graph"]["n_edges_spatial"]   for r in cell_rows]
        nE_tm = [r["graph"]["n_edges_temporal"]  for r in cell_rows]
        n_cand_sp = [r["mi_distributions"]["spatial"]["n_total"]
                       for r in cell_rows]
        n_cand_tm = [r["mi_distributions"]["temporal"]["n_total"]
                       for r in cell_rows]
        frac_sp = [a / b if b > 0 else 0.0 for a, b in zip(nE_sp, n_cand_sp)]
        frac_tm = [a / b if b > 0 else 0.0
                     for a, b in zip(nE_tm, n_cand_tm)]
        axA.plot(xs, ys, "o-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
        axB.plot(xs, diam, "s-", color=mg_color[mg],
                  markersize=8, linewidth=1.6, label=f"m/g={mg}")
        axC.plot(xs, frac_sp, "o-", color=mg_color[mg],
                  markersize=6, linewidth=1.4, label=f"spatial m/g={mg}")
        axC.plot(xs, frac_tm, "x--", color=mg_color[mg],
                  markersize=8, linewidth=1.4, label=f"temporal m/g={mg}")

    axA.axhline(4.0, color="black", linestyle=":", linewidth=1, label="$D_S=4$")
    axA.axhline(2.0, color="grey", linestyle=":", linewidth=1, label="$D_S=2$")
    axA.set_xscale("log")
    axA.set_xlabel(r"$\varepsilon_I$")
    axA.set_ylabel(r"peak $D_S$")
    axA.set_title(r"Peak $D_S$ vs $\varepsilon_I$  (Goldilocks window?)")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=8)

    axB.set_xscale("log")
    axB.set_xlabel(r"$\varepsilon_I$")
    axB.set_ylabel("hop diameter")
    axB.set_title(r"Hop diameter vs $\varepsilon_I$")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=8)

    axC.set_xscale("log")
    axC.set_xlabel(r"$\varepsilon_I$")
    axC.set_ylabel("fraction of MI candidates above $\\varepsilon$")
    axC.set_title("Edge survival fraction")
    axC.grid(True, alpha=0.3)
    axC.legend(loc="best", fontsize=7)

    # ── Middle row: D_S(σ) for one m/g, all ε overlaid
    target_mg = max(mgs)  # densest cell where ε matters most
    axD = fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("plasma")
    for j, eps in enumerate(epsilons):
        cell = next((r for r in rows
                      if r["mg"] == target_mg and r["eps"] == eps), None)
        if cell is None:
            continue
        sigmas = np.array(cell["sigmas"])
        dSs    = np.array(cell["dS_smoothed"])
        al     = cell["ambjorn_loll"]
        finite = np.isfinite(dSs)
        color = cmap(j / max(1, len(epsilons) - 1))
        axD.plot(sigmas[finite], dSs[finite],
                  color=color, linewidth=1.5,
                  label=f"data ε={eps:.0e}")
        fit = al["D_infinity"] - al["C"] / (al["B"] + sigmas)
        axD.plot(sigmas, fit, color=color, linewidth=0.9,
                  linestyle="--", alpha=0.6,
                  label=f"AL ε={eps:.0e}")
    axD.axhline(4.0, color="black", linestyle=":", linewidth=1.5,
                 label="$D_S=4$")
    axD.set_xscale("log")
    axD.set_xlabel(r"$\sigma$")
    axD.set_ylabel(r"$D_S(\sigma)$")
    axD.set_title(f"$D_S(\\sigma)$ at m/g={target_mg}, all $\\varepsilon_I$")
    axD.grid(True, alpha=0.3, which="both")
    axD.legend(loc="best", fontsize=7, ncol=2)

    # ── Bottom rows: 6 histograms for representative cell
    eps_mid = epsilons[len(epsilons) // 2]
    rep = next((r for r in rows
                 if r["mg"] == target_mg and r["eps"] == eps_mid), None)
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

    fig.suptitle("Test 2 — ε-scan at (N=60, K=9) (H_4D Goldilocks check)",
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
