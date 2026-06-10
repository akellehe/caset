#!/usr/bin/env python3
"""Scan agreement between the Schmidt-majorization order ≼_maj and the
Lieb-Robinson cone ≼_LR across (m/g, T, N) × vLr on a Schwinger TDVP run.

Sweeps a small parameter grid using the existing
SchwingerQuench.compareCausalOrders harness; for each point, prints
Kendall-τ, discordant fraction, and Hasse edit distance for the three
pairwise comparisons (maj↔LR, maj↔cs, LR↔cs). The maj↔LR row is the
substrate for the strong-falsification criterion in
emergent-causal-order-from-majorization.md §1; the cone-overflow companion script reports
the explicit n_only metric.

Output: a single tabulated report. No claims, just numbers.
"""
from __future__ import annotations
import time
from tessera.quantum import TDVPConfig, SchwingerQuench


def make_cfg(N: int, m_over_g: float, T: float, dt: float = 0.1,
             snapshotEvery: int = 5) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 64
    cfg.dmrgNSweeps     = 12
    cfg.dmrgKrylovDim   = 4
    cfg.dmrgCutoff       = 1e-12
    cfg.i0 = 3; cfg.d = 3
    cfg.dt = dt; cfg.T = T
    cfg.maxBondDim = 80
    cfg.cutoff = 1e-10
    cfg.krylovDim = 12
    cfg.snapshotEvery = snapshotEvery
    cfg.quiet = True
    cfg.conserveQns = True
    return cfg


def scan_vlr(label: str, cfg: TDVPConfig, vlr_values: list[float]) -> None:
    print(f"\n{label}")
    print(f"  N={cfg.N}  m/g={cfg.m/cfg.g}  d={cfg.d}  "
          f"T={cfg.T}  dt={cfg.dt}  snapshotEvery={cfg.snapshotEvery}")
    print(f"  {'vLr':>5}  {'τ(maj,LR)':>10}  {'discord':>9}  "
          f"{'edit':>6}  {'τ(maj,cs)':>10}  {'n_comp(maj,LR)':>15}  "
          f"{'n_comp(LR,cs)':>14}")
    for v in vlr_values:
        t0 = time.time()
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=v)
        dt_run = time.time() - t0
        print(f"  {v:>5.2f}  "
              f"{r.majVsLr.kendallTau:>10.4f}  "
              f"{r.majVsLr.discordantFraction:>9.4f}  "
              f"{r.majVsLr.hasseEditDistance:>6.3f}  "
              f"{r.majVsCs.kendallTau:>10.4f}  "
              f"{r.majVsLr.nComparableBoth:>15}  "
              f"{r.lrVsCs.nComparableBoth:>14}"
              f"   ({dt_run:.1f}s)")


if __name__ == "__main__":
    print("Lightcone vs. majorization — causal-order agreement scan")
    print("=========================================================")
    print("≼_maj — Schmidt-spectrum majorization order")
    print("≼_LR  — Lieb–Robinson cone with prescribed vLr")
    print("≼_cs  — causet order (time-only on regular 1D chain)")
    print()
    print("Sanity invariant: τ(LR, cs) = 1.0 exactly (≼_LR ⊆ ≼_cs) — should "
          "appear as 1.0 in every row at the right column (we collapse this "
          "into n_comp(LR,cs) here).")

    # With snapshotEvery=5 and dt=0.1, effective inter-snapshot Δt = 0.5;
    # interval gap d is integer 0..N-2. So vLr=0.5 → d ≤ 0.25 (only overlap),
    # vLr=2.0 → d ≤ 1, vLr=4.0 → d ≤ 2, vLr=8.0 → d ≤ 4. The scan should
    # show n_comp(LR,cs) growing monotonically in vLr.
    vlrs = [0.5, 2.0, 4.0, 8.0, 16.0]

    scan_vlr("Regime A — light quark (m/g=0.5), short evolution (T=1.0)",
             make_cfg(N=10, m_over_g=0.5, T=1.0), vlrs)

    scan_vlr("Regime B — heavy quark (m/g=5.0), short evolution (T=1.0)",
             make_cfg(N=10, m_over_g=5.0, T=1.0), vlrs)

    scan_vlr("Regime C — light quark (m/g=0.5), longer evolution (T=2.0)",
             make_cfg(N=10, m_over_g=0.5, T=2.0), vlrs)

    scan_vlr("Regime D — light quark, larger lattice (N=14, T=1.0)",
             make_cfg(N=14, m_over_g=0.5, T=1.0), vlrs)
