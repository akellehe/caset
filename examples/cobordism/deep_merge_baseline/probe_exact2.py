"""Single-kappa (the regulating kappa=1e4) full 7-DOF radial relaxation, exact
#312 physics, capped to ~80 min with periodic profile logging."""
import os, sys, time
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.argv = ["probe"]
# reuse the exact-physics setup + energy()/action()/curvature() defs verbatim
exec(open("/tmp/probe_exact.py").read().split("one = np.ones")[0])

KAPPA = 1e4
one = np.ones(len(DS))
set_profile(one)
print(f"level {LEVEL}, shells {DS}, DIM ker L1={DIM}, kappa={KAPPA:g}", flush=True)

neval = [0]; best = [np.inf, one.copy()]


def G(s):
    neval[0] += 1
    set_profile(s); S = action()
    g = S.real + KAPPA * energy() + LAM * abs(S.imag)
    if g < best[0]:
        best[0] = g; best[1] = np.asarray(s, dtype=float).copy()
    if neval[0] % 40 == 0:
        set_profile(best[1]); cur = curvature(); dd = sorted(cur); far = cur[dd[-1]]
        print(f"  [eval {neval[0]}] G*={best[0]:.1f}  "
              f"s*={' '.join(f'{v:.2f}' for v in best[1])}", flush=True)
        print(f"            excess(d)={' '.join(f'{cur[d]-far:+.3f}' for d in dd)}",
              flush=True)
    return g


t0 = time.time()
res = minimize(G, one, method="Nelder-Mead", bounds=[(0.25, 4.0)] * len(DS),
               options={"maxfev": 320, "xatol": 2e-3, "fatol": 1e-3, "adaptive": True})
set_profile(best[1]); cur = curvature(); dd = sorted(cur); far = cur[dd[-1]]
interior = all(0.26 < v < 3.99 for v in best[1])
print(f"\n=== DONE ({time.time()-t0:.0f}s, {neval[0]} evals, "
      f"{'CONVERGENT/interior' if interior else 'HIT A BOUND'}) ===", flush=True)
print("  s*(d)        =", " ".join(f"{v:6.3f}" for v in best[1]), flush=True)
print("  curvature(d) =", " ".join(f"{cur[d]:+6.3f}" for d in dd), flush=True)
print("  excess(d)    =", " ".join(f"{cur[d]-far:+6.3f}" for d in dd), flush=True)
H2 = np.abs(H) ** 2
cells_k1 = [tuple(int(v) for v in c) for c in es.cellSimplices()]
src = defaultdict(float); cnt = defaultdict(int)
for j, c in enumerate(cells_k1):
    d = min(dist.get(c[0], 99), dist.get(c[1], 99))
    src[d] += float(np.sum(H2[:, j])); cnt[d] += 1
print("  |h|^2(d)     =", " ".join(f"{src[d]/max(cnt[d],1):.5f}" for d in sorted(src)),
      flush=True)
