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

"""Level-2 feasibility for the C++ r_U gradient (#343).

The level-0 relaxation test pins correctness; this checks the C++
``EigenstateSynthesis.residualForPeriodsGradient`` stays correct — and the
relaxer scales — on the geodesically-subdivided level-2 merge substrate (the
~2724-edge deep merge of #313). It is a slow, manual feasibility run (a full
gradient call is ~100 s on CPU), NOT a suite test: verified n1=2724 gives
``|C++ - FD| ~ 1e-8``.

Full level-2 *convergence* is out of scope here — at this scale a gradient call
is dominated by the dense eigensolve (~17 s) AND the O(n1^3) per-edge
perturbation loop (~85 s). Speeding it up is the GPU eigensolve ticket (#342)
plus a faster per-edge sweep; this run only establishes correctness + feasibility.

    python examples/cobordism/level2_ru_gradient_feasibility.py [--level 2]
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

    t1 = time.time()
    g = np.asarray(es.residualForPeriodsGradient(holes, target), float)
    print(f"r_U={es.residualForPeriods(holes, target):.4e}; "
          f"C++ gradient call {time.time() - t1:.1f}s", flush=True)

    h = 1e-6
    worst = 0.0
    probe = sorted(set(int(i) for i in np.linspace(0, n1 - 1, args.cells)))
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
        d = abs(g[idx] - fd)
        worst = max(worst, d)
        print(f"  cell {idx:5d}: C++ {g[idx]:+.4e}  FD {fd:+.4e}  |d|={d:.2e}", flush=True)
    print(f"level-{args.level} worst |C++ - FD| = {worst:.2e}  (n1={n1})", flush=True)


if __name__ == "__main__":
    main()
