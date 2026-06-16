# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""FP32 cuBLAS GPU port of the r_U gradient: precision + speedup vs FP64 CPU (#348).

``EigenstateSynthesis.residualForPeriodsGradientGpu`` runs the per-edge analytic
r_U-gradient loop as an FP32 cuBLAS (SGEMM) path on the GPU; the FP64 CPU
``residualForPeriodsGradient`` is the default and the correctness oracle. This
run verifies, on the geodesically-subdivided merge substrate at ``--level N``
PERTURBED so r_U > 0 (at the base geometry r_U ~ 0, so dr_U ~ 0 — a degenerate
test):

  1. FP32-GPU vs FP64-CPU gradient  -> relative L2 error (expect ~1e-5), cosine.
  2. Both vs a central finite difference of residualForPeriods on a few cells.
  3. Wall-clock speedup of the GPU path vs the ~82 s level-2 CPU baseline.

The level-2 substrate is ``probe_deep.build_merge(2)`` (n1=2724); ``--level 0``
is the small fast smoke over the same merge family.

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
        python examples/cobordism/level2_ru_gradient_fp32_gpu.py [--level 2]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=2, help="geodesic subdivision level")
    ap.add_argument("--cells", type=int, default=3, help="how many cells to FD-check")
    args = ap.parse_args()

    t0 = time.time()
    st, nreg, holes3, _hole_vs, _cells = PD.build_merge(args.level)
    st.materializeFacets()
    es = cob.EigenstateSynthesis(st, 1)
    circles = [sorted(v + off for v in h) for off in (0, nreg, 2 * nreg) for h in holes3]
    holes = [list(c) for c in circles]
    P = np.asarray(es.cyclePeriods(holes), dtype=complex)
    m = len(holes)
    dim = len(P) // m
    target = [complex(z) for z in P.reshape(dim, m)[0]]
    cells1 = [tuple(int(v) for v in c) for c in es.cellSimplices()]
    n1 = len(cells1)
    print(f"level {args.level}: n1={n1} edges, {st.getVertexList().size()}v, "
          f"register dim={dim}, build {time.time() - t0:.1f}s", flush=True)

    emap = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        emap[(min(a, b), max(a, b))] = e
    # perturb non-uniformly so the register is no longer exactly realized (r_U > 0)
    for i in range(0, n1, 7):
        ek = (min(cells1[i]), max(cells1[i]))
        emap[ek].setSquaredLength(emap[ek].getSquaredLength().real * 1.3)
    st.materializeFacets()
    print(f"r_U = {es.residualForPeriods(holes, target):.6e}  (perturbed; must be > 0)",
          flush=True)

    t1 = time.time()
    g_cpu = np.asarray(es.residualForPeriodsGradient(holes, target), float)
    t_cpu = time.time() - t1
    print(f"FP64 CPU residualForPeriodsGradient    {t_cpu:8.2f}s", flush=True)

    t2 = time.time()
    g_gpu = np.asarray(es.residualForPeriodsGradientGpu(holes, target), float)
    t_gpu = time.time() - t2
    print(f"FP32 GPU residualForPeriodsGradientGpu {t_gpu:8.2f}s", flush=True)

    # ---- precision: FP32-GPU vs FP64-CPU ----
    denom = np.linalg.norm(g_cpu)
    rel = np.linalg.norm(g_gpu - g_cpu) / denom if denom > 0 else float("nan")
    amax = np.max(np.abs(g_gpu - g_cpu))
    cos = float(g_cpu @ g_gpu / (np.linalg.norm(g_cpu) * np.linalg.norm(g_gpu)))
    print(f"\nFP32-GPU vs FP64-CPU: rel L2 = {rel:.3e}  max|d| = {amax:.3e}  "
          f"cos = {cos:.8f}  (||g_cpu|| = {denom:.3e})", flush=True)
    print(f"speedup (CPU/GPU)    = {t_cpu / t_gpu:.1f}x  ({t_cpu:.1f}s -> {t_gpu:.2f}s)",
          flush=True)

    # ---- both vs central finite difference of residualForPeriods ----
    h = 1e-6
    worst_cpu = worst_gpu = 0.0
    probe = sorted(set(int(i) for i in np.linspace(0, n1 - 1, args.cells)))
    print(f"\nfinite-difference cross-check (h={h:g}):", flush=True)
    for idx in probe:
        ek = (min(cells1[idx]), max(cells1[idx]))
        e = emap[ek]
        l0 = e.getSquaredLength().real
        e.setSquaredLength(l0 + h); st.materializeFacets()
        rp = es.residualForPeriods(holes, target)
        e.setSquaredLength(l0 - h); st.materializeFacets()
        rm = es.residualForPeriods(holes, target)
        e.setSquaredLength(l0); st.materializeFacets()
        fd = (rp - rm) / (2 * h)
        dc, dg = abs(g_cpu[idx] - fd), abs(g_gpu[idx] - fd)
        worst_cpu, worst_gpu = max(worst_cpu, dc), max(worst_gpu, dg)
        print(f"  cell {idx:5d}: CPU {g_cpu[idx]:+.4e}  GPU {g_gpu[idx]:+.4e}  "
              f"FD {fd:+.4e}  |CPU-FD|={dc:.2e}  |GPU-FD|={dg:.2e}", flush=True)
    print(f"\nworst |CPU-FD| = {worst_cpu:.2e}   worst |GPU-FD| = {worst_gpu:.2e}  "
          f"(n1={n1})", flush=True)

    ok = (rel < 1e-4) and (cos > 0.9999)
    print(f"\nVERDICT: FP32-GPU {'MATCHES' if ok else 'DOES NOT MATCH'} FP64-CPU "
          f"to the approved ~1e-5 tolerance (rel<1e-4, cos>0.9999).", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
