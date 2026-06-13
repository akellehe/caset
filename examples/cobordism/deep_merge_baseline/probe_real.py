"""Probe v6 (the real one): the matter regulates the conformal runaway to a
CONVERGENT geometry over a radial profile, and we read the curvature(d) of that
emergent dual. No sidestep: self-consistent metric-harmonic matter energy (the
restoring force), the conformal runaway present and regulated by the mass."""
import os, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.optimize import minimize
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])

st, nreg, holes, hole_vs, cells = build_merge(2)
st.materializeFacets()
R = 2 * nreg
es = cob.EigenstateSynthesis(st, 1)
ncell = len(es.cellSimplices())
cidx = {tuple(int(v) for v in c): j for j, c in enumerate(es.cellSimplices())}
circles = [tuple(sorted(v + off for v in h)) for off in (0, nreg, R) for h in holes]
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


def metric_harm():
    L = np.asarray(cob.HodgeLaplacian(st).laplacian(1, True, False),
                   dtype=complex).reshape(ncell, ncell).real
    vals, vecs = spla.eigsh(sp.csr_matrix(L), k=8, sigma=0.0, which="LM")
    return vecs[:, np.abs(vals) < 1e-7].T.astype(complex)


def periods(H):
    P = np.zeros((H.shape[0], 9), dtype=complex)
    for ci, (a, b, c) in enumerate(circles):
        for (x, y) in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y)); sgn = 1.0 if x < y else -1.0
            if key in cidx: P[:, ci] += sgn * H[:, cidx[key]]
    return P


set_profile(np.ones(len(DS)))
TGT = periods(metric_harm())[0]
w1ref = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)


def energy():
    H = metric_harm(); P = periods(H)
    c, *_ = np.linalg.lstsq(P.T, TGT, rcond=None); h = c @ H
    w1 = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
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
print(f"shells {DS}; one G eval = {time.time()-t0:.2f}s", flush=True)

# (a) conformal scan: does the mass make G interior-minimal (convergent)?
print("\nconformal mode (all shells = s): ReS, |ImS|, E", flush=True)
g = np.linspace(0.4, 3.0, 14); rows = []
for s in g:
    set_profile(np.full(len(DS), s)); S = action(); rows.append((s, S.real, abs(S.imag), energy()))
arr = np.array(rows)
print(f"{'s':>5}{'ReS':>11}{'|ImS|':>9}{'E':>9}", flush=True)
for r in arr: print(f"{r[0]:5.2f}{r[1]:11.1f}{r[2]:9.1f}{r[3]:9.4f}", flush=True)
for kappa in (1e3, 1e4, 1e5, 1e6):
    G = arr[:, 1] + kappa * arr[:, 3] + arr[:, 2]; i = int(np.argmin(G))
    print(f"  kappa={kappa:8g}: conformal argmin s*={arr[i,0]:.2f} "
          f"{'INTERIOR (mass regulates!)' if 0 < i < len(g)-1 else 'boundary (runaway)'}",
          flush=True)

# (b) the radial relaxation at a regulating kappa → convergent dual → curvature
for kappa in (1e4, 1e5, 1e6):
    def G(s): set_profile(s); S = action(); return S.real + kappa*energy() + abs(S.imag)
    t1 = time.time()
    res = minimize(G, one, method="Nelder-Mead", bounds=[(0.3, 3.0)]*len(DS),
                   options={"maxiter": 800, "xatol": 2e-3, "fatol": 1e-3})
    set_profile(res.x); cur = curvature()
    interior = all(0.31 < v < 2.99 for v in res.x)
    print(f"\nkappa={kappa:g} ({time.time()-t1:.0f}s): {'CONVERGENT (interior)' if interior else 'hit bounds'}",
          flush=True)
    print("  s*(d) =", " ".join(f"{v:.3f}" for v in res.x), flush=True)
    print("  cur(d)=", " ".join(f"{cur[d]:+.3f}" for d in sorted(cur)), flush=True)
