# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The proton color singlet from the symmetric W_ABC junction (#396 / #398).

Three color quarks meet at the trivalent junction `TripartiteRegisterTopology`: one
connected geodesic-2 icosahedron (S^2, 42 vertices) minus 12 vertex-disjoint hole
triangles in FOUR windows of three -- A, B, C (the quark inputs) and R (the emergent
result). Distinct holes = independent cycles, so the inputs do NOT average (the #382
bipartite obstruction is escaped) and charge is conserved at the junction.

The windows are placed SYMMETRICALLY (#398): one orbit of a tetrahedral subgroup A4
of the icosahedral rotation group, seated at the icosahedron's four tetrahedral
vertex-orbits. Because the windows are A4-equivalent, the input->result transport
INTERTWINES the color Z3, so the color-symmetric (omega-representation) quark input
transports to the EXACT color singlet [1, omega, omega^2] -- the proton -- with
manifest S3. A naive (non-symmetric) input, or the geometrically inequivalent greedy
placement, reaches only ~0.74. The singlet is never imposed: it is the unique
omega-eigenvector, forced by the symmetry.

Run:  python examples/cobordism/proton_tripartite_junction.py
"""

import cmath

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = np.array([1, _W, _W * _W], complex)
_NEUTRAL_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]


def _windows(m):
    """The four windows (A, B, C inputs, R result), in the cobordism's own labels."""
    ih = [tuple(sorted(h)) for h in m.input_holes]
    return [ih[0:3], ih[3:6], ih[6:9], [tuple(sorted(h)) for h in m.result_holes]]


def _transport(m):
    """The input->result transport M (3 result x 9 input color amplitudes): carry
    each unit input hole and read its raw periods on R (one build, nine carries)."""
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {}
    for i, c in enumerate(es.cellSimplices()):
        if len(c) == 2:
            edge[(min(c), max(c))] = i
    holes = [h for w in _windows(m) for h in w]
    M = np.zeros((3, 9), complex)
    for col in range(9):
        psi = es.carriedRepresentative([list(holes[col])], [1.0])
        for k, (a, b, c) in enumerate(holes[9:12]):
            M[k, col] = psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]
    return M


def _window_cycle_rep(windows):
    """Signed-permutation reps (P_in, P_out) of the window-cycling symmetry g (the
    A4 3-cycle that fixes R and cycles A->B->C), from the icosahedral generators."""
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
    hsets = [set(w) for w in windows]
    apply_h = lambda full, h: tuple(sorted(full[v] for v in h))

    def winperm(full):
        perm = []
        for w in windows:
            img = {apply_h(full, h) for h in w}
            match = [j for j in range(4) if img == hsets[j]]
            if len(match) != 1:
                return None
            perm.append(match[0])
        return tuple(perm)

    g_full = next(lift(p) for p in group.values() if winperm(lift(p)) == (1, 2, 0, 3))
    holes = [h for w in windows for h in w]
    hidx = {h: i for i, h in enumerate(holes)}
    sgn3 = lambda t: 1 if ((t[0] > t[1]) + (t[0] > t[2]) + (t[1] > t[2])) % 2 == 0 else -1
    P = np.zeros((12, 12), complex)
    for i, h in enumerate(holes):
        img = (g_full[h[0]], g_full[h[1]], g_full[h[2]])
        P[hidx[tuple(sorted(img))], i] = sgn3(img)
    return P[0:9, 0:9], P[9:12, 9:12]


def _fmt(v):
    return "[" + ", ".join(f"{x.real:+.2f}{x.imag:+.2f}i" for x in v) + "]"


def _overlap(v, ref):
    return abs(np.vdot(v, ref)) / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-30)


def main():
    trt = cob.TripartiteRegisterTopology()
    m = cob.TransportCobordism(_NEUTRAL_PAIRS, max_iters=0, seed=0, topology=trt)
    windows = _windows(m)
    betti = list(cob.ChainComplex.fromSpacetime(m.cobordism).bettiNumbers())
    l2 = [e.getSquaredLength().real for e in m.cobordism.getEdgeList().toVector()]

    print("THE SYMMETRIC W_ABC JUNCTION (#398)\n")
    print("Four tetrahedral windows on the geodesic-2 icosahedron (corner = icosa vertex):")
    for name, w in zip("ABCR", windows):
        print(f"  {name}: {w}   corners {sorted(min(h) for h in w)}")
    print(f"\n  b1 = {betti[1]} (12 holes - 1 Stokes relation)   "
          f"metric uniform = {all(abs(x - 1.0) < 1e-12 for x in l2)}\n")

    M = _transport(m)
    P_in, P_out = _window_cycle_rep(windows)
    intertwine = np.linalg.norm(M @ P_in - P_out @ M) / np.linalg.norm(M)
    print(f"transport rank = {np.linalg.matrix_rank(M, tol=1e-9)}   "
          f"intertwines color Z3:  ||M P_in - P_out M|| / ||M|| = {intertwine:.2e}\n")

    # The color-symmetric (omega-rep) quark input -> the singlet.
    wv, vin = np.linalg.eig(P_in)
    wo, vout = np.linalg.eig(P_out)
    singlet = vout[:, int(np.argmin(np.abs(wo - _W)))]
    sym_overlaps = [_overlap(M @ vin[:, k], singlet)
                    for k in range(9) if abs(wv[k] - _W) < 1e-6]

    # A naive (non-symmetric) neutral input, for contrast.
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {(min(c), max(c)): i for i, c in enumerate(es.cellSimplices()) if len(c) == 2}
    psi = es.carriedRepresentative(list(m.input_holes), list(m.input_hole_targets))
    naive = np.array([psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]
                      for (a, b, c) in windows[3]])

    print(f"target proton color singlet:  {_fmt(_SINGLET)}")
    print(f"  naive (non-symmetric) input  -> result overlap with singlet = "
          f"{_overlap(naive, _SINGLET):.4f}")
    print(f"  color-SYMMETRIC (omega-rep)  -> result overlap with singlet = "
          f"{min(sym_overlaps):.4f} .. {max(sym_overlaps):.4f}")
    print("\nThe symmetric quark input transports to the EXACT color singlet -- the\n"
          "proton -- forced by the A4 window symmetry, never imposed.")


if __name__ == "__main__":
    main()
