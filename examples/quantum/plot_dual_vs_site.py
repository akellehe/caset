"""Plot dual-lattice vs site-graph spectral dimension at N=16.

Reads pre-computed JSON output from run_dual_spectral_dimension.py
(dual graph) and from the existing N=16 bootstrap (site graph) and
overlays them in a single comparison figure.
"""
from __future__ import annotations

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    dual_paths = {
        0.25: "/tmp/dual-holography/N16_mg_0.25.json",
        0.5:  "/tmp/dual-holography/N16_mg_0.5.json",
        5.0:  "/tmp/dual-holography/N16_mg_5.0.json",
    }
    site_aggregate = "/tmp/holography-N16-bootstrap/aggregate.json"
    out_png = ("/home/andrew/tessera/docs/source/quantum-experiments/"
                "figures/dual_vs_site_spectral_dimension.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    have_site = os.path.exists(site_aggregate)
    site_data = json.load(open(site_aggregate)) if have_site else None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {0.25: "C0", 0.5: "C1", 5.0: "C2"}
    for mg, path in dual_paths.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        sigs = np.array(data["sigmas"])
        ds = np.array(data["dS_smoothed"])
        axL.plot(sigs, ds, color=colors.get(mg, "C3"),
                  linewidth=1.6,
                  label=f"$m/g = {mg}$")

    axL.axhline(2.0, color="grey", linestyle=":", linewidth=1)
    axL.text(1e-2, 2.04, "$D_S = 2$ (lattice)", color="grey",
             fontsize=9, ha="left", va="bottom")
    axL.set_xscale("log")
    axL.set_xlabel(r"$\sigma$")
    axL.set_ylabel("$D_S(\\sigma)$")
    axL.set_title("Dual-lattice (bond-cut) $D_S(\\sigma)$ at $N=16$")
    axL.set_ylim(0.0, 2.9)
    axL.grid(True, alpha=0.3, which="both")
    axL.legend(loc="upper right", fontsize=9)

    if have_site:
        mgs = sorted(float(k) for k in site_data.keys())
        site_means = [site_data[str(mg)
                                   if str(mg) in site_data else f"{mg:g}"]
                                   ["mean_peak"] for mg in mgs]
        site_stds  = [site_data[str(mg)
                                   if str(mg) in site_data else f"{mg:g}"]
                                   ["std_peak"]  for mg in mgs]
    dual_peaks = []
    for mg in [0.25, 0.5, 5.0]:
        path = dual_paths[mg]
        if not os.path.exists(path):
            dual_peaks.append(None)
            continue
        with open(path) as f:
            data = json.load(f)
        dual_peaks.append(data["peak_dS"])

    mgs_x = [0.25, 0.5, 5.0]
    if have_site:
        axR.errorbar(mgs_x, site_means, yerr=site_stds, fmt="o",
                      color="C3", capsize=4, markersize=8, linewidth=1.5,
                      label="site graph (mean ± std over $i_0$)")
    valid = [(mg, d) for mg, d in zip(mgs_x, dual_peaks) if d is not None]
    if valid:
        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        axR.plot(xs, ys, "s", color="C0", markersize=10,
                  label="dual graph (single $i_0 = 3$)")
    axR.axhline(2.0, color="grey", linestyle=":", linewidth=1)
    axR.set_xscale("log")
    axR.set_xlabel("$m / g$")
    axR.set_ylabel("peak $D_S$")
    axR.set_title("peak $D_S$: site vs dual graph at $N=16$")
    axR.set_ylim(0.0, 2.9)
    axR.grid(True, alpha=0.3, which="both")
    axR.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
