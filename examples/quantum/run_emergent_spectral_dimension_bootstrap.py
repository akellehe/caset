"""i_0 bootstrap for the emergent spectral dimension experiment.

For each m/g, vary the quench location i_0 across the valid odd values
(spec §H4 parity: i_0 odd, d odd, i_0 + d <= N). Report mean ± std of
peak D_S, plus the full D_S(sigma) trace at each (m/g, i_0) point.

The single-trajectory writeup at
``docs/source/quantum-experiments/emergent_spectral_dimension_writeup.md``
flags the "no bootstrap" caveat; this script addresses it directly. The
output drives the bootstrap section of that writeup.

Usage::

    python examples/quantum/run_emergent_spectral_dimension_bootstrap.py \
        --N 8 --T 1.0 --dt 0.25 \
        --m-over-g 0.25 0.5 5.0 \
        --i0 1 3 5 \
        --out-dir /tmp/holography-bootstrap
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

from tessera.quantum import TDVPConfig
from tessera.quantum.holography import (
    EmergentSpectralDimension,
    HolographyConfig,
)


def make_config(args: argparse.Namespace,
                m_over_g: float,
                i0: int) -> HolographyConfig:
    cfg = HolographyConfig()
    cfg.tdvp = TDVPConfig()
    cfg.tdvp.N = args.N
    cfg.tdvp.a = 1.0
    cfg.tdvp.g = 1.0
    cfg.tdvp.m = m_over_g * cfg.tdvp.g
    cfg.tdvp.L0 = 0.0
    cfg.tdvp.dmrgMaxBondDim = 64
    cfg.tdvp.dmrgNSweeps = 12
    cfg.tdvp.dmrgKrylovDim = 4
    cfg.tdvp.dmrgCutoff = 1e-12
    cfg.tdvp.i0 = i0
    cfg.tdvp.d = args.d
    cfg.tdvp.quenchEnforceParity = True
    cfg.tdvp.dt = args.dt
    cfg.tdvp.T = args.T
    cfg.tdvp.snapshotEvery = 1
    cfg.tdvp.maxBondDim = 80
    cfg.tdvp.cutoff = 1e-10
    cfg.tdvp.krylovDim = 12
    cfg.tdvp.quiet = True
    cfg.tdvp.conserveQns = True
    cfg.sigmaMin = args.sigma_min
    cfg.sigmaMax = args.sigma_max
    cfg.sigmaCount = args.sigma_count
    cfg.epsilonI = args.epsilon_i
    cfg.krylovDim = 30
    cfg.includeTemporal = True
    cfg.maxTemporalStride = 0
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--d", type=int, default=3,
                   help="quench separation; spec requires odd")
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--m-over-g", type=float, nargs="+",
                   default=[0.25, 0.5, 5.0])
    p.add_argument("--i0", type=int, nargs="+", default=[1, 3, 5],
                   help="quench centres to sweep; each must satisfy "
                        "i0 odd and i0 + d <= N")
    p.add_argument("--sigma-min", type=float, default=1e-2)
    p.add_argument("--sigma-max", type=float, default=1e3)
    p.add_argument("--sigma-count", type=int, default=48)
    p.add_argument("--epsilon-i", type=float, default=1e-8)
    p.add_argument("--out-dir", default="/tmp/holography-bootstrap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    aggregate: dict[float, dict] = {}
    for mg in args.m_over_g:
        rows = []
        for i0 in args.i0:
            if i0 % 2 == 0 or i0 + args.d > args.N:
                raise ValueError(
                    f"invalid i0={i0}: parity constraint or "
                    f"i0+d={i0 + args.d} > N={args.N}")
            cfg = make_config(args, mg, i0)
            print(f"running m/g={mg}, i0={i0}...", flush=True)
            result = EmergentSpectralDimension(cfg).compute()
            peak = max(d for d in result.dSSmoothed if math.isfinite(d))
            rows.append({
                "m_over_g": mg,
                "i0": i0,
                "peak_dS": peak,
                "n_vertices": result.graphNVertices,
                "n_edges": result.graphNEdges,
                "D_infinity": result.dInfinity,
                "sigmas": list(result.sigmas),
                "dS_smoothed": list(result.dSSmoothed),
            })
            path = os.path.join(args.out_dir,
                                f"mg_{mg:g}_i0_{i0}.json")
            with open(path, "w") as f:
                f.write(result.toJson(cfg))
            print(f"  peak D_S = {peak:.4f}  |E| = {result.graphNEdges}",
                  flush=True)
        peaks = [r["peak_dS"] for r in rows]
        mean_peak = statistics.mean(peaks)
        std_peak = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
        aggregate[mg] = {
            "rows": rows,
            "peaks": peaks,
            "mean_peak": mean_peak,
            "std_peak": std_peak,
            "range_peak": [min(peaks), max(peaks)],
        }
        print(f"  m/g={mg}: peak D_S = {mean_peak:.4f} ± {std_peak:.4f}"
              f" (range {min(peaks):.3f}–{max(peaks):.3f})", flush=True)

    print()
    print("── i_0-bootstrap summary ────────────────────────────────")
    print(f"  {'m/g':>5}  {'mean peak D_S':>14}  {'std':>8}  "
          f"{'range':>14}  {'i_0 samples':>12}")
    for mg in args.m_over_g:
        a = aggregate[mg]
        rng = f"{a['range_peak'][0]:.3f}–{a['range_peak'][1]:.3f}"
        peaks_str = ", ".join(f"{p:.2f}" for p in a["peaks"])
        print(f"  {mg:>5.2f}  {a['mean_peak']:>14.4f}  "
              f"{a['std_peak']:>8.4f}  {rng:>14}  [{peaks_str}]")

    out = os.path.join(args.out_dir, "aggregate.json")
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in aggregate.items()}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
