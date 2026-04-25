#!/usr/bin/env python3
"""Lightcone-vs-majorization scan with explicit cone-overflow metric.

Re-runs the (m/g, T, N) × vLr scan from lightcone_vs_majorization.py
with the ``nOnlyA / nOnlyB`` fields on OrderAgreement. For
``(a, b) = (≼_maj, ≼_LR)``, ``nOnlyA`` is the count of majorization-
related pairs whose endpoints lie OUTSIDE the Lieb-Robinson cone — the
explicit criterion-1 (strong-falsification) metric of
quantum-methodology.md §1.2.
"""
from __future__ import annotations
import time
from caset.quantum import TDVPConfig, computeCausalComparison


def make_cfg(N: int, m_over_g: float, T: float,
             maxBondDim: int = 80,
             dt: float = 0.1, snapshotEvery: int = 5) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = maxBondDim
    cfg.dmrgNSweeps     = 12
    cfg.dmrgKrylovDim   = 4
    cfg.dmrgCutoff       = 1e-12
    cfg.i0 = 3; cfg.d = 3
    cfg.dt = dt; cfg.T = T
    cfg.maxBondDim = maxBondDim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 12
    cfg.snapshotEvery = snapshotEvery
    cfg.quiet = True
    cfg.conserveQns = True
    return cfg


def scan_vlr(label: str, cfg: TDVPConfig, vlr_values: list[float]) -> None:
    """Print the agreement table including strong-falsification counts.

    Columns:
      vLr             — Lieb-Robinson velocity
      τ(maj,LR)        — Kendall-τ on the both-comparable subset
      n_maj∉LR         — pairs ≼_maj relates that ≼_LR does NOT (criterion 1)
      n_maj∉LR_frac    — n_maj∉LR / n_maj_pairs ; ≼_maj-fraction outside cone
      n_LR∉maj         — pairs ≼_LR relates that ≼_maj does NOT
      total ≼_maj      — nConcordant + nDiscordant + nOnlyA (size of |maj|)
    """
    print(f"\n{label}")
    print(f"  N={cfg.N}  m/g={cfg.m/cfg.g}  d={cfg.d}  "
          f"T={cfg.T}  max_bond={cfg.maxBondDim}  "
          f"snapshotEvery={cfg.snapshotEvery}")
    print(f"  {'vLr':>5}  {'τ(maj,LR)':>10}  "
          f"{'n_maj∉LR':>10}  {'n_maj∉LR/|maj|':>14}  "
          f"{'n_LR∉maj':>10}  {'|≼_maj|':>10}")
    for v in vlr_values:
        t0 = time.time()
        r = computeCausalComparison(cfg, vLr=v)
        dt_run = time.time() - t0
        a = r.majVsLr
        n_maj_total = a.nConcordant + a.nDiscordant + a.nOnlyA
        frac_maj_out = (a.nOnlyA / n_maj_total) if n_maj_total > 0 else 0.0
        print(f"  {v:>5.2f}  "
              f"{a.kendallTau:>10.4f}  "
              f"{a.nOnlyA:>10}  "
              f"{frac_maj_out:>14.4f}  "
              f"{a.nOnlyB:>10}  "
              f"{n_maj_total:>10}"
              f"   ({dt_run:.1f}s)")


if __name__ == "__main__":
    print("Lightcone vs. majorization — cone-overflow scan (methodology §1.2 criterion 1)")
    print("================================================")
    print("n_maj∉LR : count of ≼_maj-related pairs OUTSIDE the LR cone.")
    print("         If > 0 at vLr ≥ 1.0 (free-fermion bound), criterion 1")
    print("         is engaged — ≼_maj sees super-LR order on these pairs.")
    print()

    vlrs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    scan_vlr("Regime A — light quark (m/g=0.5), N=10, T=1.0",
             make_cfg(N=10, m_over_g=0.5, T=1.0), vlrs)
    scan_vlr("Regime B — heavy quark (m/g=5.0), N=10, T=1.0",
             make_cfg(N=10, m_over_g=5.0, T=1.0), vlrs)
    scan_vlr("Regime D — light quark, N=14, T=1.0",
             make_cfg(N=14, m_over_g=0.5, T=1.0), vlrs)
    scan_vlr("Regime E — light quark, N=20, T=1.0",
             make_cfg(N=20, m_over_g=0.5, T=1.0, maxBondDim=120), vlrs)
