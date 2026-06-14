"""Diagnostic: where does #312's matter restoring force come from? Is the
combinatorial register H metric-independent? Are cyclePeriods? Does the
self-consistent energy (recompute everything, as #312 does) have an interior
minimum in the worldtube scale at fixed bulk — the restoring force the user says
the mass must provide?"""
import os, sys
import numpy as np
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])

# Use the canonical #312 substrate (level 0) so we match its energy exactly.
st, nreg, holes, hole_vs, cells = build_merge(0)
st.materializeFacets()
R = 2 * nreg
es = cob.EigenstateSynthesis(st, 1)
circles = [list(tuple(sorted(v + off for v in h))) for off in (0, nreg, R) for h in holes]
emap = {}
for e in st.getEdgeList().toVector():
    a, b = e.getSource().getId(), e.getTarget().getId()
    emap[(min(a, b), max(a, b))] = e
holev = hole_vs
wt = [k for k in emap if ((k[0] >= R) != (k[1] >= R)) and (k[0] in holev or k[1] in holev)]
bulk = [k for k in emap if ((k[0] >= R) != (k[1] >= R)) and k not in set(wt)]


def set_scales(s_wt, s_bulk):
    for k, e in emap.items():
        e.setPhase(0.0)
        if (k[0] >= R) != (k[1] >= R):
            e.setSquaredLength(-(s_wt if k in set(wt) else s_bulk))
        else:
            e.setSquaredLength(1.0)
    st.materializeFacets()


def H_of():
    return np.asarray(cob.HodgeLaplacian(st).harmonicMatrix(1, 1e-9, False),
                      dtype=complex).reshape(-1, len(es.cellSimplices()))


def P_of():
    H = H_of()
    return np.asarray(es.cyclePeriods(circles), dtype=complex).reshape(H.shape[0], 9)


# (1) is the combinatorial register metric-independent?
set_scales(1.0, 1.0); Ha = H_of(); Pa = P_of(); wa = np.asarray(cob.HodgeLaplacian(st).weights(1))
set_scales(0.5, 2.0); Hb = H_of(); Pb = P_of(); wb = np.asarray(cob.HodgeLaplacian(st).weights(1))
# compare spans (H is a basis; compare projectors)
Qa, _ = np.linalg.qr(Ha.T); Qb, _ = np.linalg.qr(Hb.T)
dH = np.linalg.norm(Qa @ Qa.conj().T - Qb @ Qb.conj().T)
print(f"(1) register span change uniform->(0.5,2.0): ||Pa-Pb||_proj = {dH:.3e} "
      f"({'METRIC-INDEPENDENT' if dH < 1e-6 else 'metric-dependent'})")
print(f"    cyclePeriods change: {np.linalg.norm(Pa-Pb)/np.linalg.norm(Pa):.3e} "
      f"({'fixed' if np.linalg.norm(Pa-Pb) < 1e-6 else 'metric-dependent'})")
print(f"    weights(1) change:   {np.linalg.norm(wa-wb)/np.linalg.norm(wa):.3e} "
      f"({'fixed' if np.linalg.norm(wa-wb)<1e-6 else 'metric-dependent (volumes)'})")

# (2) the #312 self-consistent energy: recompute H, P, c each time
set_scales(1.0, 1.0)
target = P_of()[0]


def energy_312(s_wt, s_bulk):
    set_scales(s_wt, s_bulk)
    H = H_of(); P = P_of()
    c, *_ = np.linalg.lstsq(P.T, target, rcond=None)
    h = c @ H
    w1 = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
    return float(np.real(np.vdot(h, w1 * h)))


print("\n(2) self-consistent E(s_wt) at s_bulk=1.0 — restoring force?")
for s in (0.4, 0.7, 1.0, 1.4, 1.8, 2.4):
    print(f"    s_wt={s:.2f}: E = {energy_312(s, 1.0):.5f}")
print("\n(3) self-consistent E(s) along the CONFORMAL mode (s_wt=s_bulk=s):")
for s in (0.4, 0.7, 1.0, 1.4, 1.8, 2.4):
    print(f"    s={s:.2f}: E = {energy_312(s, s):.5f}")

# (4) FAST path: metric-harmonics via sparse null space of the metric L1.
import scipy.sparse as sp, scipy.sparse.linalg as spla, time
ncell = len(es.cellSimplices())
cidx = {tuple(int(v) for v in c): j for j, c in enumerate(es.cellSimplices())}

def metric_harm_sparse():
    L = np.asarray(cob.HodgeLaplacian(st).laplacian(1, True, False), dtype=complex)
    L = L.reshape(ncell, ncell).real
    vals, vecs = spla.eigsh(sp.csr_matrix(L), k=8, sigma=0.0, which="LM")
    return vecs[:, np.abs(vals) < 1e-7].T.astype(complex)

def my_periods(H):
    P = np.zeros((H.shape[0], 9), dtype=complex)
    for ci, (a, b, c) in enumerate(circles):
        for (x, y) in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y)); sgn = 1.0 if x < y else -1.0
            if key in cidx: P[:, ci] += sgn * H[:, cidx[key]]
    return P

set_scales(1.0, 1.0)
Hm0 = metric_harm_sparse(); tgt = my_periods(Hm0)[0]
def energy_fast(s_wt, s_bulk):
    set_scales(s_wt, s_bulk)
    Hm = metric_harm_sparse(); P = my_periods(Hm)
    c, *_ = np.linalg.lstsq(P.T, tgt, rcond=None); h = c @ Hm
    w1 = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
    return float(np.real(np.vdot(h, w1 * h)))

print("\n(4) FAST sparse metric-harmonic E(s_wt) at s_bulk=1.0 (vs #312 above):")
t0 = time.time()
for s in (0.4, 0.7, 1.0, 1.4, 1.8, 2.4):
    print(f"    s_wt={s:.2f}: E_fast = {energy_fast(s, 1.0):.5f}")
print(f"    ({(time.time()-t0)/6:.2f}s/eval — vs ~16s for cyclePeriods)")
