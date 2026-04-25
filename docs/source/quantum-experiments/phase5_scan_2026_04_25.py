#!/usr/bin/env python3
"""Phase 5 hypothesis-test scan.

Sweeps a small (m/g, T) × v_LR grid using the existing
caset.quantum.compute_causal_comparison harness, plus a strong-
falsification side probe via CausalOrders so we can count maj-pairs
outside the LR cone — the criterion 1 in quantum-methodology.md §1.

Output: a single tabulated report. No claims, just numbers.
"""
from __future__ import annotations
import time
from caset.quantum import (
    TDVPConfig, compute_causal_comparison, run_qqbar_quench,
)


def make_cfg(N: int, m_over_g: float, T: float, dt: float = 0.1,
             snapshot_every: int = 5) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrg_max_bond_dim = 64
    cfg.dmrg_n_sweeps     = 12
    cfg.dmrg_krylov_dim   = 4
    cfg.dmrg_cutoff       = 1e-12
    cfg.i0 = 3; cfg.d = 3
    cfg.dt = dt; cfg.T = T
    cfg.max_bond_dim = 80
    cfg.cutoff = 1e-10
    cfg.krylov_dim = 12
    cfg.snapshot_every = snapshot_every
    cfg.quiet = True
    cfg.conserve_qns = True
    return cfg


def scan_vlr(label: str, cfg: TDVPConfig, vlr_values: list[float]) -> None:
    print(f"\n{label}")
    print(f"  N={cfg.N}  m/g={cfg.m/cfg.g}  d={cfg.d}  "
          f"T={cfg.T}  dt={cfg.dt}  snapshot_every={cfg.snapshot_every}")
    print(f"  {'v_LR':>5}  {'τ(maj,LR)':>10}  {'discord':>9}  "
          f"{'edit':>6}  {'τ(maj,cs)':>10}  {'n_comp(maj,LR)':>15}  "
          f"{'n_comp(LR,cs)':>14}")
    for v in vlr_values:
        t0 = time.time()
        r = compute_causal_comparison(cfg, v_LR=v)
        dt_run = time.time() - t0
        print(f"  {v:>5.2f}  "
              f"{r.maj_vs_lr.kendall_tau:>10.4f}  "
              f"{r.maj_vs_lr.discordant_fraction:>9.4f}  "
              f"{r.maj_vs_lr.hasse_edit_distance:>6.3f}  "
              f"{r.maj_vs_cs.kendall_tau:>10.4f}  "
              f"{r.maj_vs_lr.n_comparable_both:>15}  "
              f"{r.lr_vs_cs.n_comparable_both:>14}"
              f"   ({dt_run:.1f}s)")


if __name__ == "__main__":
    print("Phase 5 hypothesis-test scan")
    print("============================")
    print("≼_maj — Schmidt-spectrum majorization order (Phase 3)")
    print("≼_LR  — Lieb–Robinson cone with prescribed v_LR")
    print("≼_cs  — causet order (time-only on regular 1D chain)")
    print()
    print("Sanity invariant: τ(LR, cs) = 1.0 exactly (≼_LR ⊆ ≼_cs) — should "
          "appear as 1.0 in every row at the right column (we collapse this "
          "into n_comp(LR,cs) here).")

    # With snapshot_every=5 and dt=0.1, effective inter-snapshot Δt = 0.5;
    # interval gap d is integer 0..N-2. So v_LR=0.5 → d ≤ 0.25 (only overlap),
    # v_LR=2.0 → d ≤ 1, v_LR=4.0 → d ≤ 2, v_LR=8.0 → d ≤ 4. The scan should
    # show n_comp(LR,cs) growing monotonically in v_LR.
    vlrs = [0.5, 2.0, 4.0, 8.0, 16.0]

    scan_vlr("Regime A — light quark (m/g=0.5), short evolution (T=1.0)",
             make_cfg(N=10, m_over_g=0.5, T=1.0), vlrs)

    scan_vlr("Regime B — heavy quark (m/g=5.0), short evolution (T=1.0)",
             make_cfg(N=10, m_over_g=5.0, T=1.0), vlrs)

    scan_vlr("Regime C — light quark (m/g=0.5), longer evolution (T=2.0)",
             make_cfg(N=10, m_over_g=0.5, T=2.0), vlrs)

    scan_vlr("Regime D — light quark, larger lattice (N=14, T=1.0)",
             make_cfg(N=14, m_over_g=0.5, T=1.0), vlrs)
