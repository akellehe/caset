"""Probe: deep merge over a subdivided holed register — shell count + curvature
profile feasibility, before committing the module."""
import importlib.util, os, sys, time
from collections import defaultdict, deque
import numpy as np

HERE = os.path.expanduser("~/deep-merge/examples/cobordism")
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location(
    "merge_cobordism", os.path.join(HERE, "merge_cobordism.py"))
MC = importlib.util.module_from_spec(spec); sys.modules["merge_cobordism"] = MC
spec.loader.exec_module(MC)
tessera, cob, BASE = MC.tessera, MC.cob, MC.BASE
_ICO, _CLASS_HOLES = BASE._ICO, BASE._CLASS_HOLES


def _subdivide_tracked(faces, holes):
    """One geodesic subdivision (matching BASE._subdivide's id scheme) that also
    follows each tracked hole onto its central child."""
    nxt = [max(v for f in faces for v in f) + 1]; mid = {}
    def m(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid: mid[k] = nxt[0]; nxt[0] += 1
        return mid[k]
    out, central = [], {}
    for (a, b, c) in faces:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        central[tuple(sorted((a, b, c)))] = tuple(sorted((ab, bc, ca)))
    return out, [central[tuple(sorted(h))] for h in holes]


def deep_surface(level):
    faces = [tuple(f) for f in _ICO]; holes = [tuple(sorted(h)) for h in _CLASS_HOLES]
    for _ in range(level):
        faces, holes = _subdivide_tracked(faces, holes)
    hs = set(tuple(sorted(h)) for h in holes)
    holed = [tuple(sorted(f)) for f in faces if tuple(sorted(f)) not in hs]
    return holed, sorted(hs), max(v for f in faces for v in f) + 1


def build_merge(level, timelike=1.0):
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
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        crossing = (a >= R) != (b >= R)
        e.setSquaredLength(-timelike if crossing else 1.0); e.setPhase(0.0)
    hole_vs = set(v + off for h in holes for off in (A, B, R) for v in h)
    return st, nreg, holes, hole_vs, cells


def shells(st, hole_vs):
    adj = defaultdict(set)
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        if a != b: adj[a].add(b); adj[b].add(a)
    dist = {v: 0 for v in hole_vs}; dq = deque(hole_vs)
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in dist: dist[w] = dist[u] + 1; dq.append(w)
    return dist


def curvature_profile(st, dist):
    st.materializeFacets()
    bins = defaultdict(list)
    for s in st.getSimplices():
        vs = [v.getId() for v in s.getVertices()]
        if len(vs) != 2: continue
        d = min(dist.get(vs[0], 10**9), dist.get(vs[1], 10**9))
        bins[d].append(s.lorentzianDeficitAngle().real)
    return bins


for level in (0, 1, 2):
    t0 = time.time()
    st, nreg, holes, hole_vs, cells = build_merge(level)
    nE = len(st.getEdgeList().toVector())
    dist = shells(st, hole_vs)
    maxd = max(dist.values())
    t_build = time.time() - t0
    t1 = time.time()
    bins = curvature_profile(st, dist)
    t_prof = time.time() - t1
    print(f"\n=== level {level}: register {nreg}v → merge {st.getVertexList().size()}v, "
          f"{nE}E, {len(cells)} tets | build {t_build:.1f}s prof {t_prof:.1f}s ===")
    print(f"  shells (BFS distance from {len(hole_vs)} hole-cycle verts): max d = {maxd}")
    for d in sorted(bins):
        v = np.array(bins[d])
        print(f"    d={d}: {len(v):4d} hinges  mean|defc|={np.mean(np.abs(v)):.4f}  "
              f"mean defc={np.mean(v):+.4f}")
