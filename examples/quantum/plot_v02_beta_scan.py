"""Plot the v0.2 β-scan: compares v0.1 baseline (charges off),
v0.2 γ_CP=0 (charge-conserving Hamiltonian), and v0.2 γ_CP=0.5
(CP-violating Hamiltonian). Loads
/tmp/interaction-history/v02_beta_scan.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SRC = Path("/tmp/interaction-history/v02_beta_scan.json")
REPO = Path(__file__).resolve().parents[2]
OUT = (REPO / "docs/source/quantum-experiments/figures"
       / "v02_beta_scan.png")


def per_beta_series(records, label):
    rs = [r for r in records if r["label"] == label]
    betas = sorted({r["beta"] for r in rs})
    out = []
    for b in betas:
        bbatch = [r for r in rs if r["beta"] == b]
        out.append({
            "beta": b,
            "peaks":     np.array([r["peak_dS"]    for r in bbatch], float),
            "sigs":      np.array([r["peak_sigma"] for r in bbatch], float),
            "cells":     np.array([r["cells"]     for r in bbatch], float),
            "qglobal":   np.array([r["q_global"]   for r in bbatch], float),
            "n_vert":    np.array([r["n_vertices"] for r in bbatch], float),
            "n_frontier":np.array([r["n_frontier"] for r in bbatch], float),
        })
    return out


def main():
    with open(SRC) as f:
        data = json.load(f)
    series = {
        "v01_baseline": per_beta_series(data["records"], "v01_baseline"),
        "v02_qcons":    per_beta_series(data["records"], "v02_qcons"),
        "v02_qcp":      per_beta_series(data["records"], "v02_qcp"),
    }

    style = {
        "v01_baseline": ("#1f77b4", "v0.1 baseline (charges off)"),
        "v02_qcons":    ("#2ca02c", "v0.2 (γ_CP = 0)"),
        "v02_qcp":      ("#d62728", "v0.2 (γ_CP = 0.5)"),
    }

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax_d, ax_sigma, ax_q, ax_v = axes

    # peak D_S
    for key, color_label in style.items():
        color, label = color_label
        s = series[key]
        if not s: continue
        bs = np.array([r["beta"] for r in s])
        mean = np.array([r["peaks"].mean() for r in s])
        std  = np.array([r["peaks"].std()  for r in s])
        ax_d.plot(bs, mean, color=color, lw=1.8, label=label)
        ax_d.fill_between(bs, mean - std, mean + std,
                          color=color, alpha=0.15)
        for r in s:
            ax_d.scatter([r["beta"]] * len(r["peaks"]), r["peaks"],
                         s=10, alpha=0.3, color=color, edgecolor="none")
    ax_d.axhline(4.0, color="gray", lw=0.8, ls="--", label="D_S = 4 target")
    ax_d.set_yscale("log")
    ax_d.set_xscale("log")
    ax_d.set_ylabel("peak $D_S$  (mean ± std, 10 seeds)")
    ax_d.set_title("Charged-Cartan v0.2 β-scan: qudit basis vs v0.1 baseline")
    ax_d.grid(True, which="both", ls=":", alpha=0.4)
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # σ at peak D_S
    for key, color_label in style.items():
        color, label = color_label
        s = series[key]
        if not s: continue
        for r in s:
            ax_sigma.scatter([r["beta"]] * len(r["sigs"]), r["sigs"],
                             s=10, alpha=0.4, color=color,
                             edgecolor="none")
        bs = np.array([r["beta"] for r in s])
        mean = np.array([r["sigs"].mean() for r in s])
        ax_sigma.plot(bs, mean, color=color, lw=1.2, label=label)
    ax_sigma.set_xscale("log")
    ax_sigma.set_yscale("log")
    ax_sigma.set_ylabel("σ at peak $D_S$ (diffusion time)")
    ax_sigma.grid(True, which="both", ls=":", alpha=0.4)
    ax_sigma.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Q_global (vs β, by config)
    for key, color_label in style.items():
        color, label = color_label
        s = series[key]
        if not s: continue
        bs = np.array([r["beta"] for r in s])
        mean = np.array([r["qglobal"].mean() for r in s])
        std  = np.array([r["qglobal"].std()  for r in s])
        ax_q.plot(bs, mean, color=color, lw=1.6, label=label)
        ax_q.fill_between(bs, mean - std, mean + std,
                          color=color, alpha=0.15)
        for r in s:
            ax_q.scatter([r["beta"]] * len(r["qglobal"]), r["qglobal"],
                         s=10, alpha=0.35, color=color, edgecolor="none")
    ax_q.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax_q.set_xscale("log")
    ax_q.set_ylabel("Q_global at end of tune (frontier-summed)")
    ax_q.grid(True, which="both", ls=":", alpha=0.4)
    ax_q.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Vertex / cell counts (sanity)
    for key, color_label in style.items():
        color, label = color_label
        s = series[key]
        if not s: continue
        bs = np.array([r["beta"] for r in s])
        nv = np.array([r["n_vert"].mean() for r in s])
        nc = np.array([r["cells"].mean() for r in s])
        ax_v.plot(bs, nv, color=color, lw=1.5,
                  label=f"{label}: vertices")
        ax_v.plot(bs, nc, color=color, lw=0.7, ls=":",
                  label=f"{label}: cells")
    ax_v.set_xscale("log")
    ax_v.set_yscale("log")
    ax_v.set_xlabel("β (inverse temperature)")
    ax_v.set_ylabel("count (mean across 10 seeds)")
    ax_v.grid(True, which="both", ls=":", alpha=0.4)
    ax_v.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[wrote] {OUT}")

    # Console table
    print("\nPer-β mean ± std peak D_S, three configs:")
    print(f"{'β':>10}  "
          f"{'v0.1':>20}  {'v0.2 γ_CP=0':>20}  {'v0.2 γ_CP=0.5':>20}")
    base = series["v01_baseline"]; cons = series["v02_qcons"]; cp = series["v02_qcp"]
    for rb, rc, rp in zip(base, cons, cp):
        assert abs(rb["beta"] - rc["beta"]) < 1e-15
        assert abs(rb["beta"] - rp["beta"]) < 1e-15
        b_str = f"{rb['peaks'].mean():7.2f} ± {rb['peaks'].std():6.2f}"
        c_str = f"{rc['peaks'].mean():7.2f} ± {rc['peaks'].std():6.2f}"
        p_str = f"{rp['peaks'].mean():7.2f} ± {rp['peaks'].std():6.2f}"
        print(f"  {rb['beta']:.3e}  {b_str:>20}  {c_str:>20}  {p_str:>20}")


if __name__ == "__main__":
    main()
