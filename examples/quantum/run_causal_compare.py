#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Phase 5: causal-order comparison on a TDVP q-qbar quench.

Runs the full tessera.quantum Phase 5 pipeline:

  1. DMRG ground state of the Schwinger Hamiltonian
  2. q-qbar quench (creates a flux tube on d links)
  3. 2-site TDVP with per-step Schmidt-spectra recording
  4. Build three partial orders on the (cut, time) labels:
        ≼_maj  — strict majorization on the spectra (Phase 3, across time)
        ≼_LR   — Lieb-Robinson cone with prescribed vLr
        ≼_cs   — causet (time-only on regular chain; Phase 6 makes
                 this informative within a time slice too)
  5. Compute pairwise agreement statistics (Kendall-τ, discordant
     fraction, Hasse-graph edit distance).

This is the experimental harness for the methodology page
(``docs/source/quantum-methodology.md``):

    Hypothesis (H): the three partial orders coincide on the cut family
    × time-sample restriction.

The script reports the statistics, not a verdict on the hypothesis —
interpretation requires running across multiple parameter combos and
bootstrapping over Trotter seeds (out of scope for v1).

References
----------

  Lieb, Robinson, *The finite group velocity of quantum spin systems*,
  Comm. Math. Phys. 28, 251 (1972) — the cone bound underlying ≼_LR.

  Hastings, Koma, *Spectral gap and exponential decay of correlations*,
  Comm. Math. Phys. 265, 781 (2006), arXiv:math-ph/0507008 — sharper
  decay rates relevant for OTOC-based vLr extraction.

  Bañuls et al., JHEP 11, 158 (2013), arXiv:1305.3765 — Schwinger MPS
  benchmarks underlying Phases 1-2.

  Buyens et al., Phys. Rev. Lett. 113, 091601 (2014), arXiv:1312.6654 —
  string-state quench prescription (Phase 4).

Examples
--------

Default — light-quark on N=10::

    python examples/quantum/run_causal_compare.py

Heavy-quark, longer evolution::

    python examples/quantum/run_causal_compare.py --m-over-g 5 --T 2.0

Scan over Lieb-Robinson velocity::

    python examples/quantum/run_causal_compare.py --scan-vlr 0.5 1.0 2.0 4.0
"""
from __future__ import annotations

import argparse
import math
import sys

try:
    from tessera.quantum import TDVPConfig, computeCausalComparison
except ImportError as e:
    print(f"tessera.quantum unavailable: {e}", file=sys.stderr)
    print("\nRebuild with: TESSERA_QUANTUM=1 pip install -e .", file=sys.stderr)
    sys.exit(1)


def _print_report(label: str, r) -> None:
    print(f"\n{label}")
    print(f"  nLabels    = {r.nLabels}")
    print(f"  nSnapshots = {r.nSnapshots}")
    print(f"  vLr        = {r.vLr}")
    print(f"  {'pair':<12}  {'Kendall-τ':>10}  {'discord':>10}  "
          f"{'edit-dist':>10}  {'comparable':>11}")
    for name, agr in (("majVsLr", r.majVsLr),
                      ("majVsCs", r.majVsCs),
                      ("lrVsCs",  r.lrVsCs)):
        print(f"  {name:<12}  "
              f"{agr.kendallTau:>10.4f}  "
              f"{agr.discordantFraction:>10.4f}  "
              f"{agr.hasseEditDistance:>10.4f}  "
              f"{agr.nComparableBoth:>11}")


def _build_config(args) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = args.N
    cfg.a = 1.0; cfg.g = args.g
    cfg.m = args.m_over_g * args.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = args.dmrg_bond
    cfg.dmrgNSweeps     = 12
    cfg.dmrgKrylovDim   = 4
    cfg.dmrgCutoff       = 1e-12
    cfg.i0 = args.i0
    cfg.d  = args.d
    cfg.dt = args.dt
    cfg.T  = args.T
    cfg.maxBondDim = args.max_bond_dim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 12
    cfg.snapshotEvery = args.snapshot_every
    cfg.quiet = True
    cfg.conserveQns = True
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--N",            type=int,   default=10)
    p.add_argument("--m-over-g",     type=float, default=0.5)
    p.add_argument("--g",            type=float, default=1.0)
    p.add_argument("--i0",           type=int,   default=3,
                   help="First site of the q-qbar pair (1-based, odd).")
    p.add_argument("--d",            type=int,   default=3,
                   help="Pair separation (odd).")
    p.add_argument("--dt",           type=float, default=0.1)
    p.add_argument("--T",            type=float, default=1.0)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--dmrg-bond",    type=int,   default=64)
    p.add_argument("--max-bond-dim", type=int,   default=80)
    p.add_argument("--v-LR",         type=float, default=1.0,
                   help="Lieb-Robinson velocity. Default 1.0 (free-fermion).")
    p.add_argument("--scan-vlr",     type=float, nargs="+", default=None,
                   metavar="V",
                   help="Scan over vLr values, printing one report per value.")
    args = p.parse_args()

    cfg = _build_config(args)
    print(f"Causal-order comparison — N={cfg.N}, m/g={args.m_over_g}, "
          f"d={cfg.d}, T={cfg.T}, dt={cfg.dt}")

    if args.scan_vlr is not None:
        for v in args.scan_vlr:
            r = computeCausalComparison(cfg, vLr=v)
            _print_report(f"=== vLr = {v} ===", r)
    else:
        r = computeCausalComparison(cfg, vLr=args.v_LR)
        _print_report(f"=== vLr = {args.v_LR} ===", r)


if __name__ == "__main__":
    main()
