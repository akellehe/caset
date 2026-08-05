"""Verify the dimension remark: the register theorems at n=3 on the hexagon join.

The join of two hexagons triangulates S^3 with 12 vertices, 48 edges, 72
triangles, 36 tetrahedra. Removing the three vertex-disjoint tetrahedra
{a0,a1,b0,b1}, {a2,a3,b2,b3}, {a4,a5,b4,b5} (a C_3-shift orbit) opens three
boundary 2-spheres. This script checks, with the unit cochain metric:

  * ker L_2 emerges 0 -> 2 on opening the holes (the register, one degree up),
  * harmonic periods over the three boundary spheres conserve total charge
    (the null covector of the period matrix is the all-ones covector), and
  * the anchor-normalized register Gram is the identity to machine precision,
    as the isometric-chart proposition demands -- here the setwise stabilizer
    of the hole triple (order 12 inside the 288-element automorphism group)
    realizes the full S_3, so Schur's lemma applies directly as well.
"""
import collections

import numpy as np

A = list(range(6))
B = list(range(6, 12))
ae = [(i, (i + 1) % 6) for i in range(6)]
be = [(6 + i, 6 + (i + 1) % 6) for i in range(6)]
tets = sorted(tuple(sorted((*ea, *eb))) for ea in ae for eb in be)
tris = sorted({tuple(sorted(t)) for tet in tets for t in
               [(tet[0], tet[1], tet[2]), (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]), (tet[1], tet[2], tet[3])]})
edges = sorted({tuple(sorted(e)) for t in tris for e in
                [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
assert (len(edges), len(tris), len(tets)) == (48, 72, 36)
assert 12 - len(edges) + len(tris) - len(tets) == 0  # chi(S^3)

ei = {e: k for k, e in enumerate(edges)}
ti = {t: k for k, t in enumerate(tris)}
d2 = np.zeros((len(edges), len(tris)))
for t, k in ti.items():
    a, b, c = t
    d2[ei[(b, c)], k] += 1
    d2[ei[(a, c)], k] -= 1
    d2[ei[(a, b)], k] += 1


def tet_col(tet):
    col = np.zeros(len(tris))
    for j in range(4):
        face = tuple(v for i, v in enumerate(tet) if i != j)
        col[ti[face]] += (-1) ** j
    return col


d3 = np.column_stack([tet_col(t) for t in tets])

HOLES = [tuple(sorted((0, 1, 6, 7))), tuple(sorted((2, 3, 8, 9))),
         tuple(sorted((4, 5, 10, 11)))]
assert all(not set(h1) & set(h2) for h1 in HOLES for h2 in HOLES if h1 != h2)
keep = [k for k, t in enumerate(tets) if t not in HOLES]

w, _ = np.linalg.eigh(d2.T @ d2 + d3 @ d3.T)
assert int((w < 1e-9).sum()) == 0, "closed S^3 must carry no register"
w, v = np.linalg.eigh(d2.T @ d2 + d3[:, keep] @ d3[:, keep].T)
assert int((w < 1e-9).sum()) == 2, "the holed bulk must carry dim ker L_2 = 2"
H = v[:, w < 1e-9]
print("ker L_2: 0 (closed) -> 2 (holed)")

P = np.array([[H[:, r] @ tet_col(h) for h in HOLES] for r in range(2)])
n = np.linalg.svd(P)[2][-1]
n = n / np.abs(n).max()
print(f"period null covector: {np.round(n, 12)}  (charge conservation)")
sign = np.sign(n.real)

e1 = np.array([1, -1, 0]) / np.sqrt(2)
e2 = np.array([1, 1, -2]) / np.sqrt(6)
forms = [np.linalg.lstsq(P.T, sign * e, rcond=None)[0] @ H.T for e in (e1, e2)]
G = np.array([[a @ b for b in forms] for a in forms])
G = G / G[0, 0]
print(f"max|G - I| = {np.abs(G - np.eye(2)).max():.2e}")
assert np.abs(G - np.eye(2)).max() < 1e-12

adj = {v_: set() for v_ in range(12)}
for a, b in edges:
    adj[a].add(b)
    adj[b].add(a)
tset = set(tets)
autos = []


def extend(m):
    if len(m) == 12:
        if all(tuple(sorted(m[v_] for v_ in t)) in tset for t in tets):
            autos.append(dict(m))
        return
    v_ = len(m)
    used = set(m.values())
    for w_ in range(12):
        if w_ in used:
            continue
        if all((u in adj[v_]) == (m[u] in adj[w_]) for u in m):
            m[v_] = w_
            extend(m)
            del m[v_]


extend({})
stab = [g for g in autos
        if {tuple(sorted(g[v_] for v_ in h)) for h in HOLES} == set(HOLES)]
perms = collections.Counter(
    tuple(HOLES.index(tuple(sorted(g[v_] for v_ in h))) for h in HOLES)
    for g in stab)
print(f"automorphisms: {len(autos)}; hole-triple stabilizer: {len(stab)}; "
      f"induced permutation group: S_3 = {sorted(perms)}")
assert len(autos) == 288 and len(stab) == 12 and len(perms) == 6
