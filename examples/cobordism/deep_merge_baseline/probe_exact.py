"""The real experiment, exact #312 physics: the mass regulates the conformal
runaway over a FULL 7-shell radial profile (no pinned boundary, no assumed form),
and we read curvature(d) of the convergent emergent dual.

Energy = EXACT #312: combinatorial harmonics H (harmonicMatrix(1,1e-9,False),
memoized — metric-independent, so value-identical every call), cyclePeriods
recomputed at EVERY geometry, c=lstsq(P,target), E=Re<h, w1 h>. Action, free
energy G=ReS+kappa E+lam|ImS|, and argmin selection all exactly #312. The only
broadening is the carrier family: a per-shell radial profile s(d) instead of two
collective scales."""
import os, sys, time
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])

LEVEL, LAM = 2, 1.0
st, nreg, holes, hole_vs, cells = build_merge(LEVEL)
st.materializeFacets()
R = 2 * nreg
es = cob.EigenstateSynthesis(st, 1)
circles = [list(tuple(sorted(v + off for v in h))) for off in (0, nreg, R) for h in holes]
NC = len(circles)
dist = shells(st, hole_vs)
emap = {}
for e in st.getEdgeList().toVector():
    a, b = e.getSource().getId(), e.getTarget().getId()
    emap[(min(a, b), max(a, b))] = e
tl = {k: min(dist.get(k[0], 99), dist.get(k[1], 99))
      for k in emap if (k[0] >= R) != (k[1] >= R)}
DS = sorted(set(tl.values()))


def set_profile(s):
    for k, e in emap.items():
        e.setPhase(0.0)
        e.setSquaredLength(-float(s[DS.index(tl[k])]) if k in tl else 1.0)
    st.materializeFacets()


set_profile(np.ones(len(DS)))
# EXACT #312: combinatorial harmonics, metric-independent -> memoize (value-identical)
H = np.asarray(cob.HodgeLaplacian(st).harmonicMatrix(1, 1e-9, False),
               dtype=complex).reshape(-1, len(es.cellSimplices()))
DIM = H.shape[0]
TARGET = np.asarray(es.cyclePeriods(circles), dtype=complex).reshape(DIM, NC)[0]


def energy():                                  # EXACT #312 energy()
    w1 = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
    P = np.asarray(es.cyclePeriods(circles), dtype=complex).reshape(DIM, NC)
    c, *_ = np.linalg.lstsq(P.T, TARGET, rcond=None)
    h = c @ H
    return float(np.real(np.vdot(h, w1 * h)))


def action():
    return complex(tessera.ReggeSolver(st, tessera.MatterConfiguration())
                   .dualReggeAction())


def curvature():
    bins = defaultdict(list)
    for sx in st.getSimplices():
        vs = [v.getId() for v in sx.getVertices()]
        if len(vs) == 2:
            bins[min(dist.get(vs[0], 99), dist.get(vs[1], 99))].append(
                sx.lorentzianDeficitAngle().real)
    return {d: float(np.mean(v)) for d, v in sorted(bins.items())}


one = np.ones(len(DS))
t0 = time.time(); set_profile(one); _ = energy(); _ = action()
print(f"shells {DS}; DIM ker L1={DIM}; one eval = {time.time()-t0:.1f}s", flush=True)

# --- conformal kappa scan (exact energy): which kappa makes the min interior? ---
print("\nconformal scan (exact #312 energy):", flush=True)
g = np.linspace(0.45, 2.6, 11); rows = []
for s in g:
    set_profile(np.full(len(DS), s)); S = action(); rows.append((s, S.real, abs(S.imag), energy()))
    print(f"  s={s:.2f}: ReS={S.real:9.1f} |ImS|={abs(S.imag):7.1f} E={rows[-1][3]:.5f}", flush=True)
arr = np.array(rows)
reg_kappa = None
for kappa in (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5):
    G = arr[:, 1] + kappa * arr[:, 3] + LAM * arr[:, 2]; i = int(np.argmin(G))
    interior = 0 < i < len(g) - 1
    print(f"  kappa={kappa:8g}: conformal s*={arr[i,0]:.2f} {'INTERIOR' if interior else 'boundary'}",
          flush=True)
    if interior and reg_kappa is None:
        reg_kappa = kappa

print(f"\nregulating kappa = {reg_kappa}", flush=True)

# --- full 7-DOF radial relaxation at the regulating kappa (and 3x stronger) ---
for kappa in [k for k in (reg_kappa, (reg_kappa or 1e3) * 3) if k]:
    def G(s):
        set_profile(s); S = action()
        return S.real + kappa * energy() + LAM * abs(S.imag)
    t1 = time.time()
    res = minimize(G, one, method="Nelder-Mead", bounds=[(0.25, 4.0)] * len(DS),
                   options={"maxiter": 240, "xatol": 2e-3, "fatol": 1e-3})
    set_profile(res.x); cur = curvature()
    interior = all(0.26 < v < 3.99 for v in res.x)
    dd = sorted(cur); far = cur[dd[-1]]
    print(f"\n=== kappa={kappa:g} ({time.time()-t1:.0f}s, nit={res.nit}, "
          f"{'CONVERGENT/interior' if interior else 'HIT BOUND'}) ===", flush=True)
    print("  s*(d)        =", " ".join(f"{v:6.3f}" for v in res.x), flush=True)
    print("  curvature(d) =", " ".join(f"{cur[d]:+6.3f}" for d in dd), flush=True)
    print("  excess(d)    =", " ".join(f"{cur[d]-far:+6.3f}" for d in dd), flush=True)
