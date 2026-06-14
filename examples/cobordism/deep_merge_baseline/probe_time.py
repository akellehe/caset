"""Time each piece at level 2 to find the per-eval bottleneck."""
import os, sys, time
import numpy as np
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])


def t(label, fn):
    a = time.time(); r = fn(); print(f"  {label}: {time.time()-a:.3f}s", flush=True); return r


print("build + materialize (level 2):", flush=True)
t0 = time.time()
st, nreg, holes, hole_vs, cells = build_merge(2)
st.materializeFacets()
print(f"  build+materialize: {time.time()-t0:.3f}s ({st.getVertexList().size()}v)", flush=True)

es = cob.EigenstateSynthesis(st, 1)
HL = cob.HodgeLaplacian(st)
H = t("harmonicMatrix(1) [eigendecomp]",
      lambda: np.asarray(HL.harmonicMatrix(1, 1e-9, False), dtype=complex))
w1 = t("weights(1)", lambda: np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float))
circles = [list(tuple(sorted(v + off for v in h))) for off in (0, nreg, 2 * nreg) for h in holes]
P = t("cyclePeriods(9 circles)", lambda: np.asarray(es.cyclePeriods(circles), dtype=complex))
S = t("dualReggeAction", lambda: complex(tessera.ReggeSolver(
    st, tessera.MatterConfiguration()).dualReggeAction()))
print(f"\n  S = {S:.3f},  |w1|={w1.shape},  P->{P.shape}", flush=True)

# the per-eval cost of the CHEAP energy (fixed h, only w1 recomputed):
dim = H.reshape(-1, len(es.cellSimplices())).shape[0]
print(f"\n  ker L1 dim = {dim}", flush=True)
