# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Baseline benchmark for the per-edge stationary-action relaxation (#340/#343).

Times each component of a relaxation iteration on the geodesically-subdivided
merge substrate at ``--level N``, so the level-2 baseline is tracked as we
optimize. Run it on as many threads as you intend the relaxation to use
(``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``).

Measured level-2 baseline (n1=2724 edges, 16 threads, CPU):

    eigh(M) dense 2724^2              0.8 s     <- the bare eigensolve (NOT the bottleneck)
    actionGradientExact + dualRegge   ~0.5 s    <- action stationarity (cheap)
    harmonicMatrix(1)                15.3 s     <- ~19x the numpy eigensolve
    residualForPeriods (r_U value)   16.0 s     <- harmonicMatrix + projection
    residualForPeriodsGradient       82.4 s     <- the per-edge O(n1^3) loop (DOMINATES)

    => ~100 s / iteration (gradient ~82 s + the residualForPeriods guard ~16 s).

The lever is the per-edge gradient loop, not the eigensolve.

    python examples/cobordism/level2_benchmark.py [--level 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "deep_merge_baseline"))
sys.path.insert(0, _HERE)
import probe_deep as PD  # noqa: E402  (importable: no run-loop side effects)

tessera = PD.tessera
cob = tessera.cobordism


def _time(label, fn, reps=1):
    best = float("inf")
    out = None
    for _ in range(reps):
        t = time.time()
        out = fn()
        best = min(best, time.time() - t)
    print(f"  {label:32s} {best:7.2f}s", flush=True)
    return out, best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=2, help="geodesic subdivision level")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per timed op (min taken)")
    args = ap.parse_args()

    t0 = time.time()
    st, nreg, holes3, _hole_vs, _cells = PD.build_merge(args.level)
    st.materializeFacets()
    t_mesh = time.time() - t0
    es = cob.EigenstateSynthesis(st, 1)
    holes = [sorted(v + off for v in h) for off in (0, nreg, 2 * nreg) for h in holes3]
    P = np.asarray(es.cyclePeriods(holes), dtype=complex)
    m = len(holes)
    dim = len(P) // m
    target = [complex(z) for z in P.reshape(dim, m)[0]]
    n1 = len(es.cellSimplices())
    print(f"level {args.level}: n1={n1} edges, {st.getVertexList().size()} verts, "
          f"register dim={dim}  (threads: OMP={os.environ.get('OMP_NUM_THREADS','?')})")
    print(f"  {'build + materialize':32s} {t_mesh:7.2f}s")

    M = np.asarray(cob.HodgeLaplacian(st).laplacian(1, True, False),
                   dtype=complex).reshape(n1, n1).real
    _, t_eig = _time("eigh(M)  (bare eigensolve)", lambda: np.linalg.eigh(M), args.reps)
    rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
    _, t_S = _time("dualReggeAction (action S)", lambda: rs.dualReggeAction(), args.reps)
    _, t_dS = _time("actionGradientExact (dS)", lambda: rs.actionGradientExact(), args.reps)
    _, t_hm = _time("harmonicMatrix(1)", lambda: cob.HodgeLaplacian(st).harmonicMatrix(1, 1e-9, True), args.reps)
    _, t_rU = _time("residualForPeriods (r_U value)", lambda: es.residualForPeriods(holes, target), args.reps)
    _, t_drU = _time("residualForPeriodsGradient", lambda: es.residualForPeriodsGradient(holes, target), args.reps)

    # one relaxation iteration: the value eigensolve + action S + (g + 2 FD-Hessian) dS
    # + the r_U gradient, plus the asserted residualForPeriods guard.
    per_iter = t_eig + t_S + 3 * t_dS + t_drU
    print(f"\n  per relaxation iteration (objective eval) ~ {per_iter:.1f}s"
          f"  (+ {t_rU:.1f}s if the residualForPeriods guard is on)")
    print(f"    dominated by residualForPeriodsGradient ({t_drU:.1f}s = per-edge O(n1^3) loop);")
    print(f"    bare eigensolve is {t_eig:.1f}s — not the bottleneck.")


if __name__ == "__main__":
    main()
