"""Verify the symmetry hypothesis of the isometric-chart proposition.

Computes the full automorphism group of the icosahedral triangulation used by
examples/cobordism/spectral_gate_realizability.py (order 120), the setwise
stabilizer of the three canonical holonomy holes, and the induced action on
the hole triple. Output establishes:

  * the stabilizer is a C_3 acting by 3-cycles on the holes (no larger), and
  * an explicit generator: the order-3 rotation about the face axis
    {6,10,11} -- {0,3,4}, vertex cycles (0 3 4)(1 7 9)(2 8 5)(6 11 10).

That C_3, together with the reality of the unit cochain metric, is the
hypothesis under which the anchor-normalized register Gram is proven to be
the identity (the paper's isometric-chart proposition).
"""
from itertools import permutations

ICO = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1),
       (1, 5, 10), (1, 10, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8),
       (3, 8, 4), (4, 8, 9), (4, 9, 5), (5, 9, 10), (6, 10, 11), (7, 6, 11),
       (8, 7, 11), (9, 8, 11), (10, 9, 11)]
HOLES = [frozenset(h) for h in [(0, 1, 2), (3, 7, 8), (4, 9, 5)]]

faceset = {frozenset(f) for f in ICO}
adj = {v: set() for v in range(12)}
for a, b, c in ICO:
    adj[a] |= {b, c}
    adj[b] |= {a, c}
    adj[c] |= {a, b}

autos = []


def extend(m):
    if len(m) == 12:
        if all(frozenset(m[v] for v in f) in faceset for f in ICO):
            autos.append(dict(m))
        return
    v = len(m)
    used = set(m.values())
    for w in range(12):
        if w in used:
            continue
        if all((u in adj[v]) == (m[u] in adj[w]) for u in m):
            m[v] = w
            extend(m)
            del m[v]


extend({})
assert len(autos) == 120, len(autos)
print(f"automorphism group order: {len(autos)}")

stab = []
for g in autos:
    img = [frozenset(g[v] for v in h) for h in HOLES]
    if set(img) == set(HOLES):
        stab.append((tuple(HOLES.index(i) for i in img), g))
print(f"hole-triple setwise stabilizer order: {len(stab)}")
assert len(stab) == 3
perms = sorted(p for p, _ in stab)
print(f"induced permutations on the holes: {perms}")
assert perms == [(0, 1, 2), (1, 2, 0), (2, 0, 1)], "expected a C_3"

gen = next(g for p, g in stab if p != (0, 1, 2))
print("generator (vertex map):", {v: gen[v] for v in sorted(gen)})
fixed_faces = [tuple(f) for f in ICO if frozenset(gen[v] for v in f) == frozenset(f)]
print("axis faces (fixed setwise):", fixed_faces)
