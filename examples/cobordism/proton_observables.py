# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The proton's emergent observables, read off the relaxed W_ABC singlet (#400).

Now that the symmetric junction produces the color singlet (#398), this reads the
proton's physical observables OFF the relaxed geometry --- emergent-first, nothing
fabricated:

  * color charge sigma -- the projection of the emergent result onto the g-invariant
    (+1) charge mode of the window-cycling color Z3; the singlet is the orthogonal
    omega-mode, so sigma -> 0 is confinement (the bound state is color-neutral);
  * radius r = sqrt(mean(l^2 > 0)) over the relaxed spacelike edges;
  * mass, two handles -- A: |dual_action| (the full complex dual Regge action), free
    on the object; B: the shell-summed Re-deficit (sum over BFS shells from the quark
    holes of the mean Re of the Lorentzian deficit angle), the #352 curvature method;
  * the dimensionless r*m, anchored on B (comparable to the prior ~73 trajectory and
    the ~4.0 target), with A reported as a free cross-check.

The proton is fed as the natural color-symmetric (omega-representation) quark input
--- the omega-eigenvector of the window-cycling symmetry g --- so the singlet EMERGES
on the result window; it is never pinned or hand-placed.

Run:  python examples/cobordism/proton_observables.py
"""

import cmath
from collections import defaultdict

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = np.array([1, _W, _W * _W], complex)


# --- junction symmetry helpers (self-contained, per the repo's test/example style) ---
def _windows(m):
    ih = [tuple(sorted(h)) for h in m.input_holes]
    return [ih[0:3], ih[3:6], ih[6:9], [tuple(sorted(h)) for h in m.result_holes]]


def _transport(m):
    """The input->result transport M (3x9): carry each unit input hole through the
    junction and read its raw periods on R (one build, nine carries)."""
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es.cellSimplices()) if len(c) == 2}
    holes = [h for w in _windows(m) for h in w]
    M = np.zeros((3, 9), complex)
    for col in range(9):
        psi = es.carriedRepresentative([list(holes[col])], [1.0])
        for k, (a, b, c) in enumerate(holes[9:12]):
            M[k, col] = psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]
    return M


def _window_cycle_rep(windows):
    """Signed-permutation reps (P_in, P_out) of the window-cycling symmetry g (the A4
    3-cycle fixing R, cycling A->B->C), from the icosahedral generators."""
    ico = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 5, 10),
           (1, 10, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4),
           (4, 8, 9), (4, 9, 5), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11),
           (9, 8, 11), (10, 9, 11)]
    mid, nxt = {}, [12]

    def mk(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = nxt[0]
            nxt[0] += 1
        return mid[key]

    for f in (tuple(sorted(t)) for t in ico):
        mk(f[0], f[1]); mk(f[1], f[2]); mk(f[0], f[2])
    gens = [[4, 3, 8, 9, 5, 0, 7, 11, 10, 1, 2, 6],
            [3, 4, 0, 2, 7, 8, 5, 1, 6, 11, 9, 10],
            [6, 10, 11, 7, 2, 1, 9, 8, 3, 0, 5, 4],
            [10, 6, 1, 5, 9, 11, 2, 0, 4, 8, 7, 3]]
    comp = lambda p, q: [p[q[i]] for i in range(len(q))]

    def lift(p):
        full = list(range(42))
        for i in range(12):
            full[i] = p[i]
        for (a, b), idx in mid.items():
            full[idx] = mk(p[a], p[b])
        return full

    group = {tuple(p): p for p in [list(range(12))] + [list(g) for g in gens]}
    changed = True
    while changed:
        changed = False
        for p in list(group.values()):
            for g in gens:
                r = comp(p, g)
                if tuple(r) not in group:
                    group[tuple(r)] = r
                    changed = True
    hs = [set(w) for w in windows]
    ah = lambda f, h: tuple(sorted(f[v] for v in h))

    def wp(f):
        perm = []
        for w in windows:
            img = {ah(f, h) for h in w}
            mm = [j for j in range(4) if img == hs[j]]
            if len(mm) != 1:
                return None
            perm.append(mm[0])
        return tuple(perm)

    gf = next(lift(p) for p in group.values() if wp(lift(p)) == (1, 2, 0, 3))
    holes = [h for w in windows for h in w]
    hi = {h: i for i, h in enumerate(holes)}
    sgn3 = lambda t: 1 if ((t[0] > t[1]) + (t[0] > t[2]) + (t[1] > t[2])) % 2 == 0 else -1
    P = np.zeros((12, 12), complex)
    for i, h in enumerate(holes):
        img = (gf[h[0]], gf[h[1]], gf[h[2]])
        P[hi[tuple(sorted(img))], i] = sgn3(img)
    return P[0:9, 0:9], P[9:12, 9:12]


def _omega_rep_input(windows):
    """The natural color-symmetric quark input: the omega-eigenvector of g (P_in),
    sliced into the three per-window color states A, B, C."""
    p_in, _ = _window_cycle_rep(windows)
    w, v = np.linalg.eig(p_in)
    phi = v[:, [k for k in range(9) if abs(w[k] - _W) < 1e-6][0]]
    return [list(phi[0:3]), list(phi[3:6]), list(phi[6:9])]


# --- observable readers (all emergent-first: read off the relaxed geometry) ---
def _proj(u, v):
    return abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)


def radius_rms(st):
    """r = sqrt(mean(l^2)) over the relaxed spacelike (l^2 > 0) edges."""
    l2 = [e.getSquaredLength().real for e in st.getEdgeList().toVector()]
    sp = [x for x in l2 if x > 0]
    return (float(np.mean(sp)) ** 0.5 if sp else 0.0, len(sp), len(l2) - len(sp))


def shell_deficit(st, seed_vertices):
    """mass B: sum over BFS shells (from the quark holes) of the mean Re of the
    Lorentzian deficit angle -- the matter's bending of the dual geometry (#352)."""
    st.materializeFacets()
    edges = [s for s in st.getSimplices() if len(s.getVertices()) == 2]
    adj = defaultdict(set)
    for s in edges:
        a, b = [v.getId() for v in s.getVertices()]
        adj[a].add(b); adj[b].add(a)
    dist = {v: 0 for v in seed_vertices}
    frontier = list(seed_vertices)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    bins = defaultdict(list)
    for s in edges:
        a, b = [v.getId() for v in s.getVertices()]
        bins[min(dist.get(a, 99), dist.get(b, 99))].append(s.lorentzianDeficitAngle())
    shells = {d: float(np.mean([z.real for z in v])) for d, v in sorted(bins.items())}
    return float(sum(shells.values())), shells


def measure(max_iters=80):
    """Build the symmetric junction, pin the omega-rep proton input, relax, and read
    the emergent observables off the relaxed geometry. Returns a dict."""
    seed = cob.TransportCobordism([[1, -1, 0], [1, 0, -1], [0, 1, -1]],
                                  max_iters=0, seed=0,
                                  topology=cob.TripartiteRegisterTopology())
    windows = _windows(seed)
    states = _omega_rep_input(windows)
    m = cob.TransportCobordism(states, max_iters=max_iters, seed=0,
                               topology=cob.TripartiteRegisterTopology())

    # color: decompose the emergent result over the color-Z3 eigenmodes of P_out.
    p_in, p_out = _window_cycle_rep(_windows(m))
    M = _transport(m)
    wo, vout = np.linalg.eig(p_out)
    v_charge = vout[:, int(np.argmin(np.abs(wo - 1.0)))]   # g-invariant: the charge
    v_singlet = vout[:, int(np.argmin(np.abs(wo - _W)))]   # the omega-mode: the proton
    wi, vin = np.linalg.eig(p_in)
    result = M @ vin[:, [k for k in range(9) if abs(wi[k] - _W) < 1e-6][0]]

    r, n_sp, n_tl = radius_rms(m.cobordism)
    mass_a = abs(m.stats.dual_action)
    quark_holes = set(v for w in _windows(m)[:3] for h in w for v in h)
    mass_b, shells = shell_deficit(m.cobordism, quark_holes)
    return {
        "sigma": _proj(v_charge, result),          # color charge (singlet => 0)
        "singlet": _proj(v_singlet, result),       # the proton (=> 1)
        "radius": r, "n_spacelike": n_sp, "n_timelike": n_tl,
        "mass_a": mass_a, "mass_b": mass_b, "shells": shells,
        "rm_a": r * mass_a, "rm_b": r * mass_b,    # anchor on B
        "relax_iters": m.stats.relax_iterations,
        "stat_residual": m.stats.stat_action_residual,
    }


def main():
    o = measure(max_iters=80)
    print("THE PROTON'S EMERGENT OBSERVABLES (#400), off the relaxed W_ABC singlet\n")
    print("color (the result emerges from the symmetric quark input, never pinned):")
    print(f"  color charge sigma = {o['sigma']:.3e}   (confinement: singlet => 0)")
    print(f"  singlet component  = {o['singlet']:.4f}   (the proton)")
    print(f"\nradius:")
    print(f"  r = sqrt(mean(l^2>0)) = {o['radius']:.4f}   "
          f"({o['n_spacelike']} spacelike, {o['n_timelike']} timelike/null edges)")
    print(f"\nmass (two handles):")
    print(f"  A  |dual_action|       = {o['mass_a']:.4f}   (free cross-check)")
    print(f"  B  shell-summed Re-def = {o['mass_b']:.4f}   (the #352 method; ANCHOR)")
    print(f"     shells (dist: mean Re-deficit): "
          + ", ".join(f"{d}:{o['shells'][d]:.3f}" for d in o['shells']))
    print(f"\ndimensionless r*m  (target ~4.0; prior crude ~73):")
    print(f"  r*m (B, anchored) = {o['rm_b']:.4f}      r*m (A, cross-check) = {o['rm_a']:.4f}")
    print(f"\nrelax: iters={o['relax_iters']}  stat_residual={o['stat_residual']:.3e}")


if __name__ == "__main__":
    main()
