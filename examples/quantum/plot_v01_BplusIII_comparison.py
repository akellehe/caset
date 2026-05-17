"""Plot baseline vs (B + iii) comparison from the paired scan.

Loads /tmp/interaction-history/v01_compare_BplusIII.json (which has
records labelled 'baseline' or 'BplusIII' on identical seeds) and
produces a four-panel comparison.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SRC  = Path("/tmp/interaction-history/v01_compare_BplusIII.json")
REPO = Path(__file__).resolve().parents[2]
OUT  = (REPO / "docs/source/quantum-experiments/figures"
        / "v01_BplusIII_comparison.png")


def per_beta_series(records, label):
    rs = [r for r in records if r["label"] == label]
    betas = sorted({r["beta"] for r in rs})
    out = []
    for b in betas:
        bbatch = [r for r in rs if r["beta"] == b]
        out.append({
            "beta": b,
            "peaks":     np.array([r["peak_dS"]   for r in bbatch], float),
            "sigs":      np.array([r["peak_sigma"]for r in bbatch], float),
            "cells":     np.array([r["cells"]    for r in bbatch], float),
            "n_pos":     np.array([r["n_pos"]    for r in bbatch], float),
            "n_zero":    np.array([r["n_zero"]   for r in bbatch], float),
            "n_neg":     np.array([r["n_neg"]    for r in bbatch], float),
            "n_vert":    np.array([r["n_vertices"]for r in bbatch], float),
            "n_frontier":np.array([r["n_frontier"]for r in bbatch], float),
        })
    return out


def main():
    with open(SRC) as f:
        data = json.load(f)
    base = per_beta_series(data["records"], "baseline")
    new  = per_beta_series(data["records"], "BplusIII")

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax_d, ax_sigma, ax_n, ax_q = axes

    style = {
        "baseline": ("#1f77b4", "v0.1 baseline (charges only)"),
        "BplusIII": ("#d62728", "v0.1 + B + iii"),
    }

    # peak D_S
    for series, key in [(base, "baseline"), (new, "BplusIII")]:
        color, label = style[key]
        bs   = np.array([r["beta"] for r in series])
        mean = np.array([r["peaks"].mean() for r in series])
        std  = np.array([r["peaks"].std()  for r in series])
        ax_d.plot(bs, mean, color=color, lw=1.8, label=label)
        ax_d.fill_between(bs, mean - std, mean + std,
                          color=color, alpha=0.15)
        for r in series:
            ax_d.scatter([r["beta"]] * len(r["peaks"]), r["peaks"],
                         s=10, alpha=0.3, color=color, edgecolor="none")
    ax_d.axhline(4.0, color="gray", lw=0.8, ls="--",
                 label="D_S = 4 target")
    ax_d.set_yscale("log")
    ax_d.set_xscale("log")
    ax_d.set_ylabel("peak $D_S$  (mean ± std, 10 seeds)")
    ax_d.set_title("Charged-Cartan v0.1 — baseline vs (deactivate on annihilate + photon emission)")
    ax_d.grid(True, which="both", ls=":", alpha=0.4)
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # σ at peak D_S
    for series, key in [(base, "baseline"), (new, "BplusIII")]:
        color, label = style[key]
        for r in series:
            ax_sigma.scatter([r["beta"]] * len(r["sigs"]), r["sigs"],
                             s=10, alpha=0.4, color=color,
                             edgecolor="none")
        bs   = np.array([r["beta"] for r in series])
        mean = np.array([r["sigs"].mean() for r in series])
        ax_sigma.plot(bs, mean, color=color, lw=1.2, label=label)
    ax_sigma.set_xscale("log")
    ax_sigma.set_yscale("log")
    ax_sigma.set_ylabel("σ at peak $D_S$ (diffusion time)")
    ax_sigma.grid(True, which="both", ls=":", alpha=0.4)
    ax_sigma.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # vertex / frontier / cell counts
    for series, key in [(base, "baseline"), (new, "BplusIII")]:
        color, label = style[key]
        bs = np.array([r["beta"] for r in series])
        nv = np.array([r["n_vert"].mean() for r in series])
        nf = np.array([r["n_frontier"].mean() for r in series])
        nc = np.array([r["cells"].mean() for r in series])
        ax_n.plot(bs, nv, color=color, lw=1.7,
                  label=f"{label}: total vertices")
        ax_n.plot(bs, nf, color=color, lw=1.0, ls="--",
                  label=f"{label}: frontier")
        ax_n.plot(bs, nc, color=color, lw=0.7, ls=":",
                  label=f"{label}: cells")
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.set_ylabel("count (mean across 10 seeds)")
    ax_n.grid(True, which="both", ls=":", alpha=0.4)
    ax_n.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    # charge composition
    for series, key in [(base, "baseline"), (new, "BplusIII")]:
        color, label = style[key]
        bs = np.array([r["beta"] for r in series])
        nP = np.array([r["n_pos"].mean()  for r in series])
        nZ = np.array([r["n_zero"].mean() for r in series])
        nN = np.array([r["n_neg"].mean()  for r in series])
        ax_q.plot(bs, nP, color=color, lw=1.4, label=f"{label} +")
        ax_q.plot(bs, nN, color=color, lw=1.4, ls="--", label=f"{label} −")
        ax_q.plot(bs, nZ, color=color, lw=0.8, ls=":",
                  label=f"{label} neutrals")
    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    ax_q.set_xlabel("β (inverse temperature)")
    ax_q.set_ylabel("vertex count by charge sign (mean)")
    ax_q.grid(True, which="both", ls=":", alpha=0.4)
    ax_q.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[wrote] {OUT}")

    # console summary
    print(f"\nPer-β peak D_S (mean ± std, 10 seeds):")
    print(f"{'β':>10}  {'baseline':>20}  {'B+iii':>20}  {'ΔV':>8}")
    for rb, rn in zip(base, new):
        assert abs(rb["beta"] - rn["beta"]) < 1e-15
        b_str = f"{rb['peaks'].mean():7.2f} ± {rb['peaks'].std():6.2f}"
        n_str = f"{rn['peaks'].mean():7.2f} ± {rn['peaks'].std():6.2f}"
        dv = int(rn["n_vert"].mean() - rb["n_vert"].mean())
        print(f"  {rb['beta']:.3e}  {b_str:>20}  {n_str:>20}  {dv:+8d}")


if __name__ == "__main__":
    main()
