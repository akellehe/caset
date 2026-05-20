"""Plot the v0.2 plateau finite-size investigation (GitHub issue #10).

Two panels:
  • A. peak D_S vs N at T = 2500 (three β values overlaid).
  • B. peak D_S vs T at N = 8, β = 3×10⁻⁴ (log-x; power-law fit
       D_S(T) = D_∞ + A·T^(-p) with the T→∞ asymptote marked).

Loads /tmp/interaction-history/issue10_finite_size.json and the
deep-T extension /tmp/interaction-history/issue10_T100k.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SRC = Path("/tmp/interaction-history/issue10_finite_size.json")
SRC_T100K = Path("/tmp/interaction-history/issue10_T100k.json")
REPO = Path(__file__).resolve().parents[2]
OUT = (REPO / "docs/source/quantum-experiments/figures"
       / "v02_finite_size_investigation.png")


def main():
    with open(SRC) as f:
        data = json.load(f)
    rs = data["records"]
    A = [r for r in rs if r["scan"] == "A_N_scan"]
    B = [r for r in rs if r["scan"] == "B_T_scan"]

    # Deep-T extension (T = 100000) lives in a separate scan file.
    if SRC_T100K.exists():
        with open(SRC_T100K) as f:
            B += json.load(f)["records"]

    fig, (axN, axT) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: D_S vs N, three β overlaid.
    betas = sorted({r["beta"] for r in A})
    Ns    = sorted({r["N"]    for r in A})
    colors = {b: c for b, c in zip(betas, ["#1f77b4", "#2ca02c", "#d62728"])}
    for b in betas:
        means = []
        stds = []
        for N in Ns:
            ps = [r["peak_dS"] for r in A if r["N"] == N and r["beta"] == b]
            means.append(np.mean(ps))
            stds.append(np.std(ps))
            axN.scatter([N] * len(ps), ps,
                        s=18, alpha=0.4, color=colors[b],
                        edgecolor="none")
        axN.errorbar(Ns, means, yerr=stds,
                     fmt="o-", color=colors[b], lw=1.6,
                     capsize=4, markersize=7,
                     label=f"β = {b:.0e}")
    axN.axhline(4.0, color="gray", lw=0.8, ls="--", label="target D_S = 4")
    axN.set_xlabel("N (initial-layer vertex count)")
    axN.set_ylabel("peak D_S  (mean ± std, 10 seeds)")
    axN.set_title("Plateau vs initial-layer size N  (T = 2500)")
    axN.set_xticks(Ns)
    axN.grid(True, ls=":", alpha=0.4)
    axN.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # Panel B: D_S vs T at fixed N = 8, β = 3e-4.
    Ts = sorted({r["T"] for r in B})
    means = []; stds = []
    for T in Ts:
        ps = [r["peak_dS"] for r in B if r["T"] == T]
        means.append(np.mean(ps))
        stds.append(np.std(ps))
        axT.scatter([T] * len(ps), ps,
                    s=18, alpha=0.4, color="#d62728", edgecolor="none")
    axT.errorbar(Ts, means, yerr=stds,
                 fmt="o-", color="#d62728", lw=1.6,
                 capsize=4, markersize=7,
                 label="N = 8, β = 3×10⁻⁴")
    axT.axhline(4.0, color="gray", lw=0.8, ls="--", label="target D_S = 4")
    # Power-law fit: D_S(T) = D_∞ + A·T^(-p), SEM-weighted.
    Ts_arr    = np.array(Ts, float)
    means_arr = np.array(means, float)
    sem_arr   = np.array(stds, float) / np.sqrt(
        [sum(1 for r in B if r["T"] == T) for T in Ts])
    (Dinf, A_fit, p_fit), _ = curve_fit(
        lambda T, D, A, p: D + A * T ** (-p),
        Ts_arr, means_arr, p0=[4.1, 50.0, 0.6],
        sigma=sem_arr, absolute_sigma=True, maxfev=20000)
    T_grid = np.logspace(np.log10(Ts[0]), np.log10(Ts[-1] * 1.3), 300)
    pred = Dinf + A_fit * T_grid ** (-p_fit)
    axT.plot(T_grid, pred, color="#7f7f7f", lw=1.2, ls=":",
             label=f"fit: {Dinf:.3f} + {A_fit:.0f}·T$^{{-{p_fit:.2f}}}$")
    axT.axhline(Dinf, color="#7f7f7f", lw=0.8, ls="-.",
                label=f"D_S(T→∞) = {Dinf:.3f}")
    axT.set_xscale("log")
    axT.set_xlabel("T (cells)")
    axT.set_ylabel("peak D_S  (mean ± std, 10 seeds)")
    axT.set_title("Plateau vs lattice size T  (N = 8, β = 3×10⁻⁴)")
    axT.grid(True, which="both", ls=":", alpha=0.4)
    axT.legend(loc="upper right", fontsize=9, framealpha=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[wrote] {OUT}")

    # Console summary.
    print("\n=== A. N-scan at T = 2500 ===")
    print(f"  {'β':>9}  {'N':>3}  {'mean ± std':>16}  {'count':>5}")
    for b in betas:
        for N in Ns:
            ps = np.array([r["peak_dS"] for r in A
                           if r["N"] == N and r["beta"] == b])
            print(f"  {b:9.2e}  {N:>3}  "
                  f"{ps.mean():6.3f} ± {ps.std():5.3f}  {len(ps):>5}")

    print("\n=== B. T-scan at N = 8, β = 3e-4 ===")
    print(f"  {'T':>6}  {'mean ± std':>16}  {'count':>5}")
    for T in Ts:
        ps = np.array([r["peak_dS"] for r in B if r["T"] == T])
        print(f"  {T:>6}  {ps.mean():6.3f} ± {ps.std():5.3f}  {len(ps):>5}")
    print(f"  power-law fit: D_S(T) = {Dinf:.3f} + {A_fit:.1f}·T^(-{p_fit:.3f})")
    print(f"  asymptote:     D_S(T→∞) = {Dinf:.3f}")

    # Q-drift summary.
    n_drift = sum(1 for r in rs if abs(r["q_global"]) >= 1e-6)
    n_total = len(rs)
    print(f"\nQ-drift: {n_drift}/{n_total} runs with |Q| ≥ 1e-6 at γ_CP = 0.")
    drifters = [r for r in rs if abs(r["q_global"]) >= 1e-6]
    if drifters:
        qs = sorted({round(r["q_global"]) for r in drifters})
        print(f"  Drift values seen: {qs}")


if __name__ == "__main__":
    main()
