"""Probe v2: at level 2, the matter response Δ(d) = deficit(worldtube-sourced) −
deficit(reference) vs distance, and the matter-density source profile |h|²(d)."""
import importlib.util, os, sys
from collections import defaultdict, deque
import numpy as np
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])  # reuse helpers

LEVEL = 2


def build(level, s_wt, s_bulk):
    """Like build_merge but two timelike scales: worldtube (timelike edges
    incident to a hole-cycle vertex) vs bulk (the rest)."""
    holed, holes, nreg = deep_surface(level)
    A, B, R = 0, nreg, 2 * nreg
    cells = sorted(set(MC._staircase(holed, A, R) + MC._staircase(holed, B, R)))
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for c in cells for v in c})}
    for c in cells:
        t = sorted(c)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
    hole_vs = set(v + off for h in holes for off in (A, B, R) for v in h)
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        crossing = (a >= R) != (b >= R)
        if crossing:
            wt = (a in hole_vs) or (b in hole_vs)
            e.setSquaredLength(-(s_wt if wt else s_bulk))
        else:
            e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st, hole_vs


def deficit_by_shell(st, dist):
    st.materializeFacets()
    bins = defaultdict(list)
    for s in st.getSimplices():
        vs = [v.getId() for v in s.getVertices()]
        if len(vs) != 2: continue
        d = min(dist.get(vs[0], 10**9), dist.get(vs[1], 10**9))
        bins[d].append(s.lorentzianDeficitAngle().real)
    return {d: np.array(v) for d, v in bins.items()}


# reference + shells
st0, hole_vs = build(LEVEL, 1.0, 1.0)
dist = shells(st0, hole_vs)
ref = deficit_by_shell(st0, dist)

# matter source profile: |h|^2 of the carried register (ker L1), binned by distance
es = cob.EigenstateSynthesis(st0, 1)
cells_k1 = [tuple(int(v) for v in c) for c in es.cellSimplices()]
H = np.asarray(cob.HodgeLaplacian(st0).harmonicMatrix(1, 1e-9, False),
               dtype=complex).reshape(-1, len(cells_k1))
dens = defaultdict(float); cnt = defaultdict(int)
for j, c in enumerate(cells_k1):
    d = min(dist.get(c[0], 10**9), dist.get(c[1], 10**9))
    dens[d] += float(np.sum(np.abs(H[:, j])**2)); cnt[d] += 1
print(f"ker L1 dim = {H.shape[0]} (register survives = {H.shape[0] == 2})")
print("\nmatter-density source profile  |h|²(d)  (the charge's localization):")
for d in sorted(dens):
    print(f"  d={d}: mean|h|²/edge = {dens[d]/max(cnt[d],1):.5f}")

print("\nmatter curvature response  Δ(d) = defc(s_wt) − defc(ref)  [bulk=1.0]:")
for s_wt in (0.6, 0.8, 1.25, 1.6):
    stm, _ = build(LEVEL, s_wt, 1.0)
    mat = deficit_by_shell(stm, dist)
    print(f"\n  s_wt={s_wt}:")
    for d in sorted(ref):
        dd = mat[d].mean() - ref[d].mean()
        print(f"    d={d}: Δmean={dd:+.4f}  (|Δ|={abs(dd):.4f})")
