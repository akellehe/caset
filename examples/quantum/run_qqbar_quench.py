#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Real-time evolution of a Schwinger-model q-qbar quench via TDVP.

Computes the DMRG ground state of the Schwinger model, applies a flux-tube-
creating σ⁻ σ⁺ operator at sites (i₀, i₀ + d), and integrates the resulting
state forward in real time using two-site TDVP. Prints the electric-field
profile ⟨L_n⟩(t) and the charge density ⟨σ^z_n⟩(t) at sampled times so the
flux-tube dynamics are visible at a glance.

Runs the TDVP quench pipeline of ``docs/source/quantum-plan.md``. The
hypothesis under test is described in
``docs/source/quantum-experiments/earlier-work/emergent-causal-order-from-majorization.md`` — this script runs the simulation
that produces the entanglement / electric-field / charge data that the
downstream causal-order comparison mines for the majorization vs.
Lieb-Robinson vs. causet comparison.

Theory
------

In the staggered + Jordan-Wigner + Gauss-eliminated Schwinger formulation,
the σ⁻_{i0} σ⁺_{i0+d} operator (acting on the Néel-like vacuum) creates a
+1 flux tube on the d links between sites i0 and i0+d. In the heavy-quark
limit (m/g ≫ 1) the tube is approximately stable for short times because
the q and qbar are heavy and the Hamiltonian's hopping cost is small. As
m/g decreases, the tube spreads and eventually the q-qbar pair separates
or annihilates — the so-called "string breaking" regime explored in
Buyens et al. 2014.

Parity
------

For the σ⁻ σ⁺ pair to act non-trivially on |↑↓↑↓ … ⟩:
  * i0 must be odd (Up sublattice) so σ⁻ flips Up → Dn.
  * i0 + d must be even (Dn sublattice) so σ⁺ flips Dn → Up.
Therefore d must be odd. PLAN.md mentions d=4; for the heavy-quark
acceptance test we use d=5 (the closest odd value).

References
----------

  Buyens, Haegeman, Van Acoleyen, Verschelde, Verstraete,
  "Matrix product states for gauge field theories",
  Phys. Rev. Lett. 113, 091601 (2014), arXiv:1312.6654.

  Pichler, Dalmonte, Rico, Zoller, Montangero,
  "Real-time dynamics in U(1) lattice gauge theories with tensor networks",
  Phys. Rev. X 6, 011023 (2016), arXiv:1505.04440.

  Haegeman, Lubich, Oseledets, Vandereycken, Verstraete,
  "Unifying time evolution and optimization with matrix product states",
  Phys. Rev. B 94, 165116 (2016), arXiv:1408.5056.

Examples
--------

Heavy-quark limit (PLAN.md §5 acceptance setup)::

    python examples/quantum/runQqbarQuench.py

Light-quark limit — flux tube spreads / breaks::

    python examples/quantum/runQqbarQuench.py --m-over-g 0.5 --T 4.0

Track the bond dimension over time::

    python examples/quantum/runQqbarQuench.py --N 20 --d 5 --max-bond-dim 200
"""
from __future__ import annotations

import argparse
import math
import sys

try:
    from tessera.quantum import TDVPConfig, SchwingerQuench
except ImportError as e:
    print(f"tessera.quantum unavailable: {e}", file=sys.stderr)
    print("\nRebuild with: TESSERA_QUANTUM=1 pip install -e .", file=sys.stderr)
    sys.exit(1)


def _print_header(N: int) -> None:
    print(f"{'time':>8} {'energy':>14} {'bond':>5}  ⟨L_n⟩ profile (links 1..N-1)")
    print("-" * (29 + 7 * (N - 1)))


def _format_row(snap, N: int) -> str:
    L = snap.lProfile
    body = "".join(f"{x:>7.3f}" for x in L)
    return f"{snap.time:>8.3f} {snap.energy:>14.6f} {snap.bondDim:>5}  {body}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--N",            type=int,   default=14,
                   help="Number of staggered sites. Default 14.")
    p.add_argument("--m-over-g",     type=float, default=20.0,
                   help="Bañuls' m/g ratio. Default 20.0 (heavy-quark limit).")
    p.add_argument("--g",            type=float, default=1.0,
                   help="Gauge coupling. Default 1.0.")
    p.add_argument("--L0",           type=float, default=0.0,
                   help="Background electric field. Default 0.0.")
    p.add_argument("--i0",           type=int,   default=5,
                   help="First site of the q-qbar pair (1-based, odd). Default 5.")
    p.add_argument("--d",            type=int,   default=5,
                   help="Pair separation in lattice sites (odd for parity). "
                        "Default 5.")
    p.add_argument("--dt",           type=float, default=0.05,
                   help="TDVP real-time step. Default 0.05.")
    p.add_argument("--T",            type=float, default=None,
                   help="Total evolution time. Default = d (the plan's T = d·a).")
    p.add_argument("--dmrg-bond",    type=int,   default=64,
                   help="DMRG bond-dim cap. Default 64.")
    p.add_argument("--dmrg-sweeps",  type=int,   default=12,
                   help="DMRG sweep count. Default 12.")
    p.add_argument("--max-bond-dim", type=int,   default=100,
                   help="TDVP bond-dim cap. Default 100.")
    p.add_argument("--snapshot-every", type=int, default=5,
                   help="Record observables every k TDVP steps. Default 5.")
    p.add_argument("--bypass-parity", action="store_true",
                   help="Skip the i0/d parity check (useful for non-Néel "
                        "vacua at small m/g).")
    args = p.parse_args()

    cfg = TDVPConfig()
    cfg.N    = args.N
    cfg.a    = 1.0; cfg.g = args.g
    cfg.m    = args.m_over_g * args.g
    cfg.L0   = args.L0
    cfg.dmrgMaxBondDim = args.dmrg_bond
    cfg.dmrgNSweeps     = args.dmrg_sweeps
    cfg.dmrgKrylovDim   = 4
    cfg.dmrgCutoff       = 1e-12
    cfg.i0 = args.i0; cfg.d = args.d
    cfg.quenchEnforceParity = not args.bypass_parity
    cfg.dt = args.dt
    cfg.T  = args.T if args.T is not None else float(args.d)
    cfg.maxBondDim = args.max_bond_dim
    cfg.cutoff       = 1e-10
    cfg.krylovDim   = 12
    cfg.snapshotEvery = args.snapshot_every
    cfg.quiet = True
    cfg.conserveQns = True

    print(f"Schwinger q-qbar quench — N={cfg.N}, m/g={args.m_over_g}, "
          f"g={cfg.g}, L0={cfg.L0}")
    print(f"  q-qbar pair at sites ({cfg.i0}, {cfg.i0 + cfg.d}); d={cfg.d}")
    print(f"  TDVP: dt={cfg.dt}, T={cfg.T} ({int(round(cfg.T / cfg.dt))} steps), "
          f"bondDim≤{cfg.maxBondDim}")
    print()

    result = SchwingerQuench(cfg).evolve()
    print(f"DMRG ground state: E = {result.groundState.energy:.6f}, "
          f"bondDim = {result.groundState.bondDim}")
    print()

    _print_header(cfg.N)
    for snap in result.snapshots:
        print(_format_row(snap, cfg.N))


if __name__ == "__main__":
    main()
