#!/usr/bin/env python3
"""N-scaling extension of the lightcone-vs-majorization scan.

Adds N=20 (Bañuls' workhorse size) at light-quark m/g=0.5 to compare
against the N=10 and N=14 baselines at the same m/g and T, testing
whether the τ(maj, LR) decrease with N continues into the Bañuls
regime.
"""
from __future__ import annotations
import time
from tessera.quantum import TDVPConfig, SchwingerQuench


def make_cfg(N: int, m_over_g: float, T: float,
             max_bond_dim: int = 80,
             dt: float = 0.1, snapshot_every: int = 5) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = max_bond_dim
    cfg.dmrgNSweeps    = 12
    cfg.dmrgKrylovDim  = 4
    cfg.dmrgCutoff     = 1e-12
    cfg.i0 = 3; cfg.d = 3
    cfg.dt = dt; cfg.T = T
    cfg.maxBondDim = max_bond_dim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 12
    cfg.snapshotEvery = snapshot_every
    cfg.quiet = True
    cfg.conserveQns = True
    return cfg


def scan_vlr(label: str, cfg: TDVPConfig, vlr_values: list[float]) -> None:
    print(f"\n{label}")
    print(f"  N={cfg.N}  m/g={cfg.m/cfg.g}  d={cfg.d}  "
          f"T={cfg.T}  max_bond={cfg.maxBondDim}  "
          f"snapshot_every={cfg.snapshotEvery}")
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
    print("Lightcone vs. majorization — N-scaling extension")
    print("=================================================")

    vlrs = [0.5, 2.0, 4.0, 8.0, 16.0]

    # Re-run the N=10 and N=14 light-quark T=1 points first as the
    # control baseline (lets us check that scaling N is the only
    # variable changing). They're fast.
    scan_vlr("N=10 baseline (light quark, T=1.0)",
             make_cfg(N=10, m_over_g=0.5, T=1.0), vlrs)
    scan_vlr("N=14 baseline (light quark, T=1.0)",
             make_cfg(N=14, m_over_g=0.5, T=1.0), vlrs)

    # The new data point.
    scan_vlr("N=20 (light quark, T=1.0, max_bond=120)",
             make_cfg(N=20, m_over_g=0.5, T=1.0, max_bond_dim=120), vlrs)
