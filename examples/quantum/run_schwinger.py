#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Schwinger-model ground state via DMRG.

Computes the ground-state energy of the 1+1D massive Schwinger model on a
staggered-fermion lattice with N sites, varying mass-to-coupling ratio
m/g, lattice coupling x = 1/(g²a²), and bond-dim cap. Reports both the
dimensional energy (units of 1/a) and Bañuls' dimensionless density
ω₀ = E_total · a / N for direct comparison with their published numerics.

Implements the Hamiltonian from:

  Bañuls, Cichy, Cirac, Jansen,
  "The mass spectrum of the Schwinger model with Matrix Product States",
  JHEP 11, 158 (2013), arXiv:1305.3765 -- equation (2.6).

Convergence diagnostics: the DMRG result's bondDim field reports the
achieved MPS bond dimension. If it hits ``--max-bond-dim``, the run is
bond-dim-limited; bump the cap and rerun for tighter convergence.

Reference values (m/g = 0, continuum limit):

  ω₀ → -1/π ≈ -0.31831      (Schwinger's exact result, m=0 continuum)

At small x and small N, the lattice value is significantly above this
floor; running with ``--scan-x`` shows the descent toward the continuum.

Examples
--------

Basic single-point run (Phase 1 PLAN.md spec parameters)::

    python examples/quantum/run_schwinger.py

Scan x = 1/(g²a²) at fixed m/g = 0 to see continuum approach::

    python examples/quantum/run_schwinger.py --scan-x 1 4 16 --N 80

Massive case at moderate x::

    python examples/quantum/run_schwinger.py \\
        --N 30 --m-over-g 0.25 --x 25 --max-bond-dim 60
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable

try:
    from caset.quantum import QuantumConfig, computeGroundState
except ImportError as e:
    print(f"caset.quantum unavailable: {e}", file=sys.stderr)
    print("\nRebuild with: CASET_QUANTUM=1 pip install -e .", file=sys.stderr)
    sys.exit(1)


def _config_for(N: int, m_over_g: float, x: float,
                L0: float, maxBondDim: int, nSweeps: int) -> QuantumConfig:
    """Build a QuantumConfig with a = 1, g = 1/√x, m = m_over_g · g.

    Bañuls' x = 1/(g² a²): with a = 1, x = 1/g² so g = 1/√x. The
    dimensional bare mass is then m = (m/g) · g.
    """
    cfg = QuantumConfig()
    cfg.N            = N
    cfg.a            = 1.0
    cfg.g            = 1.0 / math.sqrt(x)
    cfg.m            = m_over_g * cfg.g
    cfg.L0           = L0
    cfg.maxBondDim = maxBondDim
    cfg.nSweeps     = nSweeps
    cfg.cutoff       = 1e-12
    cfg.krylovDim   = 4
    cfg.quiet        = True
    return cfg


def _print_header() -> None:
    print(f"{'N':>4} {'m/g':>6} {'x':>6} {'L0':>5} "
          f"{'E_total':>14} {'omega_0':>14} "
          f"{'bondDim':>9} {'trunc_err':>11}")
    print("-" * 80)


def _print_row(N: int, m_over_g: float, x: float, L0: float,
               result) -> None:
    omega_0 = result.energy / N  # at a = 1: omega_0 = E_total · a / N
    print(f"{N:>4} {m_over_g:>6.3f} {x:>6.2f} {L0:>5.2f} "
          f"{result.energy:>14.8f} {omega_0:>14.8f} "
          f"{result.bondDim:>9d} {result.truncationErr:>11.2e}")


def _run_one(N: int, m_over_g: float, x: float, L0: float,
             maxBondDim: int, nSweeps: int) -> None:
    cfg = _config_for(N, m_over_g, x, L0, maxBondDim, nSweeps)
    result = computeGroundState(cfg)
    _print_row(N, m_over_g, x, L0, result)


def _scan_x(N: int, m_over_g: float, xs: Iterable[float], L0: float,
            maxBondDim: int, nSweeps: int) -> None:
    print(f"\nContinuum approach: ω₀ → -1/π ≈ {-1/math.pi:.6f} as x → ∞")
    _print_header()
    for x in xs:
        _run_one(N, m_over_g, x, L0, maxBondDim, nSweeps)


def _scan_bond_dim(N: int, m_over_g: float, x: float, L0: float,
                   bond_dims: Iterable[int], nSweeps: int) -> None:
    print(f"\nVariational descent: energy non-increasing in maxBondDim")
    _print_header()
    for D in bond_dims:
        _run_one(N, m_over_g, x, L0, D, nSweeps)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--N",            type=int,   default=20,
                   help="Number of staggered sites (1-based, even). Default 20.")
    p.add_argument("--m-over-g",     type=float, default=0.0,
                   help="Bañuls' m/g ratio. Default 0.0 (massless).")
    p.add_argument("--x",            type=float, default=1.0,
                   help="Bañuls' x = 1/(g²a²) (with a=1, g=1/√x). Default 1.0.")
    p.add_argument("--L0",           type=float, default=0.0,
                   help="Background electric field. Default 0.0.")
    p.add_argument("--max-bond-dim", type=int,   default=100,
                   help="DMRG bond-dim cap. Default 100.")
    p.add_argument("--n-sweeps",     type=int,   default=12,
                   help="DMRG sweep count. Default 12.")
    p.add_argument("--scan-x",       type=float, nargs="+", default=None,
                   metavar="X",
                   help="Scan over x values (overrides --x). E.g.: --scan-x 1 4 16")
    p.add_argument("--scan-bond-dim", type=int,  nargs="+", default=None,
                   metavar="D",
                   help="Scan over maxBondDim values (overrides --max-bond-dim).")
    args = p.parse_args()

    if args.scan_x is not None and args.scan_bond_dim is not None:
        p.error("--scan-x and --scan-bond-dim are mutually exclusive")

    if args.scan_x is not None:
        _scan_x(args.N, args.m_over_g, args.scan_x, args.L0,
                args.max_bond_dim, args.n_sweeps)
    elif args.scan_bond_dim is not None:
        _scan_bond_dim(args.N, args.m_over_g, args.x, args.L0,
                       args.scan_bond_dim, args.n_sweeps)
    else:
        _print_header()
        _run_one(args.N, args.m_over_g, args.x, args.L0,
                 args.max_bond_dim, args.n_sweeps)


if __name__ == "__main__":
    main()
