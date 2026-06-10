#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Schmidt spectra and majorization poset of the Schwinger ground state.

Computes the DMRG ground state of the Schwinger model, extracts every
contiguous-interval Schmidt spectrum, and prints the majorization poset
on those spectra. Runs the Schmidt / majorization-poset pipeline of
``docs/source/quantum-plan.md`` end-to-end through the Python API:

    DMRG ground state  →  contiguous Schmidt spectra  →  Hasse cover edges

The output is the foundation for the comparison with the Lieb-Robinson
order and the causet order described in
``docs/source/quantum-experiments/earlier-work/emergent-causal-order-from-majorization.md``.

Theory
------

For each contiguous interval A = [i, j] the Schmidt spectrum λ_A is the
list of eigenvalues of ρ_A, sorted non-increasingly. Majorization
(``λ_A ≼ λ_B`` iff B is "at least as concentrated" as A) defines a
partial order; the Hasse diagram is its transitive reduction.

References
----------

  Bañuls, Cichy, Cirac, Jansen,
  "The mass spectrum of the Schwinger model with Matrix Product States",
  JHEP 11, 158 (2013), arXiv:1305.3765.

  Nielsen, "Conditions for a class of entanglement transformations",
  Phys. Rev. Lett. 83, 436 (1999), quant-ph/9811053.

Examples
--------

Default — N=10 massless Schwinger ground state with x = 1::

    python examples/quantum/run_majorization.py

Quantitative scan over m/g — strong-mass limit collapses spectra to
(1,0)-equivalent so the Hasse becomes empty::

    python examples/quantum/run_majorization.py --N 8 --m-over-g 5.0

Print only the highest-entropy intervals::

    python examples/quantum/run_majorization.py --top 5
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable

try:
    from tessera.quantum import QuantumConfig, SchwingerModel
except ImportError as e:
    print(f"tessera.quantum unavailable: {e}", file=sys.stderr)
    print("\nRebuild with: TESSERA_QUANTUM=1 pip install -e .", file=sys.stderr)
    sys.exit(1)


def _entropy(spectrum: list[float]) -> float:
    """Shannon (von Neumann) entropy of a Schmidt spectrum, base e."""
    s = 0.0
    for p in spectrum:
        if p > 1e-15:
            s -= p * math.log(p)
    return s


def _print_intervals_table(intervals, spectra, top_k: int | None) -> None:
    rows = []
    for iv, spec in zip(intervals, spectra):
        rows.append((iv.i, iv.j, _entropy(spec), len(spec), spec))
    # Sort by entropy descending so the most-entangled intervals are
    # surfaced first.
    rows.sort(key=lambda r: -r[2])
    if top_k is not None:
        rows = rows[:top_k]

    print(f"\n{'i':>3} {'j':>3} {'entropy':>10} {'rank':>5}  spectrum (top 4)")
    print("-" * 70)
    for (i, j, S, rank, spec) in rows:
        top = " ".join(f"{x:.4f}" for x in spec[:4])
        print(f"{i:>3} {j:>3} {S:>10.6f} {rank:>5}  {top}")


def _print_hasse(poset, intervals, max_edges: int) -> None:
    print(f"\nHasse cover edges (≻ means strictly majorizes; transitive "
          f"reduction)\n"
          f"{'a':>10} ≻ {'b':<10}     entropy(a) → entropy(b)\n"
          f"{'-' * 60}")
    if not poset.covers:
        print("  (no strict majorizations — all spectra are equivalent)")
        return
    n_shown = 0
    for a, b in poset.covers:
        ia, ib = intervals[a], intervals[b]
        print(f"  [{ia.i:>2},{ia.j:>2}]  ≻  [{ib.i:>2},{ib.j:>2}]")
        n_shown += 1
        if n_shown >= max_edges:
            remaining = len(poset.covers) - n_shown
            if remaining > 0:
                print(f"  … {remaining} more edges (use --max-edges to show)")
            break


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--N",            type=int,   default=10,
                   help="Number of staggered sites. Default 10.")
    p.add_argument("--m-over-g",     type=float, default=0.0,
                   help="Bañuls' m/g ratio. Default 0.0 (massless).")
    p.add_argument("--x",            type=float, default=1.0,
                   help="Bañuls' x = 1/(g²a²). Default 1.0.")
    p.add_argument("--L0",           type=float, default=0.0,
                   help="Background electric field. Default 0.0.")
    p.add_argument("--max-bond-dim", type=int,   default=64,
                   help="DMRG bond-dim cap. Default 64.")
    p.add_argument("--n-sweeps",     type=int,   default=10,
                   help="DMRG sweep count. Default 10.")
    p.add_argument("--tol",          type=float, default=1e-12,
                   help="Tolerance for majorization comparisons. Default 1e-12.")
    p.add_argument("--top",          type=int,   default=10,
                   metavar="K",
                   help="Show the top-K intervals by entropy. Use 0 to show all.")
    p.add_argument("--max-edges",    type=int,   default=20,
                   help="Cap on Hasse edges printed. Default 20.")
    args = p.parse_args()

    cfg = QuantumConfig()
    cfg.N            = args.N
    cfg.a            = 1.0
    cfg.g            = 1.0 / math.sqrt(args.x)
    cfg.m            = args.m_over_g * cfg.g
    cfg.L0           = args.L0
    cfg.maxBondDim = args.max_bond_dim
    cfg.nSweeps     = args.n_sweeps
    cfg.cutoff       = 1e-12
    cfg.krylovDim   = 4
    cfg.quiet        = True

    r = SchwingerModel(cfg).solveWithMajorization(tol=args.tol)

    print(f"Schwinger ground state — N={args.N}, m/g={args.m_over_g}, "
          f"x={args.x}, L0={args.L0}")
    print(f"  E_total       = {r.groundState.energy:.8f}")
    print(f"  E_op          = {r.groundState.operatorEnergy:.8f}")
    print(f"  E_const       = {r.groundState.constant:.8f}")
    print(f"  bondDim      = {r.groundState.bondDim}")
    print(f"  n_intervals   = {len(r.spectra.intervals)}")
    print(f"  n_cover_edges = {len(r.poset.covers)}")

    _print_intervals_table(r.spectra.intervals, r.spectra.spectra,
                           top_k=args.top if args.top > 0 else None)
    _print_hasse(r.poset, r.spectra.intervals, args.max_edges)


if __name__ == "__main__":
    main()
