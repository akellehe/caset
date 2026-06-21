# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Bipartite revisit (#416): the twisted antisymmetric-diquark projection and the
connected path to the color singlet.

#382 proves the singlet is absent from a single fusion 3 (x) 3 = 6 (+) 3bar -- but
NOT that a two-step bipartite construction fails. The gap is a merge that PROJECTS
onto the antisymmetric diquark 3bar (then 3bar (x) 3 = 1 (+) 8 contains the
singlet). This script measures two geometric levers, both THROUGH the existing
prism triangulation (the std::sort diagonal is measured, not removed):

(a) THE TWISTED BIPARTITE TUBE. The holed-icosahedron register tube extruded with
    an ORIENTATION-REVERSING twist (`RegisterTopology.set_twist` ->
    `Spacetime.prismCells`'s cumulative twist): reverse each color hole's induced
    orientation, so each carried color period flips sign. On the uniform metric the
    two input-transport blocks satisfy M_B = -M_A exactly, so the symmetric sextet
    6 cancels and the merge is the PURE antisymmetric diquark 3bar: the A<->B
    antisymmetric fraction goes from 0 (untwisted control, pure 6) to 1 (twisted,
    pure 3bar). The twist is an EXACT geometric antisymmetrizer.

(b) THE CONNECTED PATH. A connected distinct-windows geometry (the diquark stays an
    interior carried state on its own independent cycle -- no free diquark
    boundary, dW = the input/result windows only) reaches the singlet: the
    input->result transport is full rank, so the singlet [1,w,w^2] lies in its
    image (reachability 1.0). Contrast the WELDED/free-boundary bipartite SEQUENCE
    (read the colored diquark out as a free boundary, re-pin it): on the SHARED
    single register every state is over-determined (r_U ~ 16, the averaging that is
    the #382 obstruction), so the sequence floors. On DISTINCT windows even a
    colored state carries exactly (r_U ~ 3e-27): keeping the diquark interior on an
    independent cycle escapes the over-determination. Confinement-of-the-free-
    intermediate is the welded-sequence obstruction, and it is NOT fundamental.

The color read-outs are the EXACT period machinery (carriedRepresentative over the
removed-triangle holes, signed edge-sum periods), measured on the uniform l^2 = 1
metric -- the seed-independent geometric transport (the jittered relax is a noisy
proxy). Run:  python examples/cobordism/bipartite_twist.py
"""

from __future__ import annotations

import cmath

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = np.array([1, _W, _W * _W], complex)
_KSIGN = np.array([1, 1, -1])  # RegisterTopology.kColorSign (the #353 SIGN_BLOCK)

# Three color-neutral q-qbar pairs (Sigma = 0 each): the carriable inputs of #382.
_A = np.array([1, -1, 0], complex)
_B = np.array([1, 0, -1], complex)
_C = np.array([0, 1, -1], complex)
SEED = 416


# --------------------------------------------------------------------------- #
#  Shared read-out helpers (the exact #353 period machinery on a built complex)
# --------------------------------------------------------------------------- #
def _edge_index(es):
    return {(min(c), max(c)): i
            for i, c in enumerate(es.cellSimplices()) if len(c) == 2}


def _uniform(cobordism):
    """Reset every edge to l^2 = 1: the symmetric, seed-independent geometric metric
    on which the carried color transport is read (the jittered relax is a noisy
    proxy; the singlet/Stokes structure lives on the uniform point, cf. #398)."""
    for e in cobordism.getEdgeList().toVector():
        e.setSquaredLength(complex(1.0, 0.0))
    return cobordism


def _period(psi, edge, hole):
    a, b, c = sorted(hole)
    return psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]


def _transport(es, edge, in_holes, out_holes):
    """The transport block M (len(out) x len(in)): carry each UNIT input hole and
    read its signed period over each output hole (one build, many carries)."""
    M = np.zeros((len(out_holes), len(in_holes)), complex)
    for col, h in enumerate(in_holes):
        psi = es.carriedRepresentative([list(h)], [1.0])
        for k, oh in enumerate(out_holes):
            M[k, col] = _period(psi, edge, oh)
    return M


def _overlap(v, ref=_SINGLET):
    v = np.asarray(v, complex)
    r = np.asarray(ref, complex)
    nv, nr = np.linalg.norm(v), np.linalg.norm(r)
    return abs(np.vdot(v, r)) / (nv * nr) if nv * nr > 0 else 0.0


# --------------------------------------------------------------------------- #
#  (a) The twisted bipartite tube -> the antisymmetric diquark channel
# --------------------------------------------------------------------------- #
def _bipartite_tube(twist):
    """Build the holed-icosahedron register tube (RegisterTopology) with an optional
    twist, via the C++ TransportCobordism (max_iters=0 -- just the seed topology), and
    return the cobordism reset to the uniform metric plus its A/B/R color holes."""
    rt = cob.RegisterTopology()
    if twist is not None:
        rt.set_twist(twist)
    m = cob.TransportCobordism([list(_A), list(_B)], max_iters=0, seed=SEED, topology=rt)
    inp = [tuple(sorted(h)) for h in m.input_holes]
    return (_uniform(m.cobordism), inp[0:3], inp[3:6],
            [tuple(sorted(h)) for h in m.result_holes], m)


def _antisym_fraction(cobordism, a_holes, b_holes, r_holes):
    """The A<->B antisymmetric fraction of the emergent diquark: split the result
    r(A,B) = M_A.(sign.A) + M_B.(sign.B) into its parts under A<->B exchange. 0 for
    the symmetric sextet 6, 1 for the antisymmetric 3bar."""
    es = cob.EigenstateSynthesis(cobordism, 1)
    edge = _edge_index(es)
    m_a = _transport(es, edge, a_holes, r_holes)
    m_b = _transport(es, edge, b_holes, r_holes)
    sa, sb = _KSIGN * _A, _KSIGN * _B
    r_ab = m_a @ sa + m_b @ sb
    r_ba = m_a @ sb + m_b @ sa
    asym = 0.5 * (r_ab - r_ba)
    frac = np.linalg.norm(asym) / (np.linalg.norm(r_ab) + 1e-30)
    return float(frac), r_ab, float(np.linalg.norm(m_a + m_b))


def measure_twist():
    """Measure the emergent diquark channel of the untwisted control vs the
    orientation-reversing twisted tube. Returns a dict of the verdict numbers."""
    out = {}
    for label, twist in [("control", None),
                         ("twisted", cob.RegisterTopology.orientation_reversing_twist())]:
        co, ah, bh, rh, m = _bipartite_tube(twist)
        valid, _ = cob.EigenstateSynthesis(co, 1).dualComplexValid()
        frac, res, m_sum = _antisym_fraction(co, ah, bh, rh)
        out[label] = {
            "antisym_fraction": frac, "result": res, "valid": bool(valid),
            "b1": int(list(m.stats.betti_cobordism)[1]),
            "m_blocks_sum_norm": m_sum,            # ||M_A + M_B||: 0 iff pure 3bar
            "singlet_overlap": _overlap(res),
        }
    out["delta"] = out["twisted"]["antisym_fraction"] - out["control"]["antisym_fraction"]
    return out


# --------------------------------------------------------------------------- #
#  (b) The connected path -> the singlet, with the diquark kept interior
# --------------------------------------------------------------------------- #
_ICO = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 5, 10),
        (1, 10, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4),
        (4, 8, 9), (4, 9, 5), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11),
        (9, 8, 11), (10, 9, 11)]
_ICO = [tuple(sorted(f)) for f in _ICO]


def _geodesic2():
    """Frequency-2 geodesic icosahedron (42 vertices), the #398 base surface that
    hosts 12 vertex-disjoint hole triangles (the same edge-midpoint numbering as
    `TripartiteRegisterTopology`'s geodesicSphere)."""
    mid, nxt = {}, [12]

    def mk(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = nxt[0]
            nxt[0] += 1
        return mid[key]

    faces = []
    for a, b, c in _ICO:
        ab, bc, ca = mk(a, b), mk(b, c), mk(a, c)
        for tri in [(a, ab, ca), (b, ab, bc), (c, bc, ca), (ab, bc, ca)]:
            faces.append(tuple(sorted(tri)))
    return faces, mid


def _windows(mid):
    """The four A4-tetrahedral C3 windows (A, B, C, R) of three corner sub-triangles
    each, generated from the icosahedral symmetry (the #398 placement)."""
    gens = [[4, 3, 8, 9, 5, 0, 7, 11, 10, 1, 2, 6],
            [3, 4, 0, 2, 7, 8, 5, 1, 6, 11, 9, 10],
            [6, 10, 11, 7, 2, 1, 9, 8, 3, 0, 5, 4],
            [10, 6, 1, 5, 9, 11, 2, 0, 4, 8, 7, 3]]
    seed = [(2, 0, 1), (1, 6, 10), (0, 3, 4), (3, 2, 7)]
    rev = {v: k for k, v in mid.items()}

    def mk(x, y):
        return mid[(min(x, y), max(x, y))]

    def apply(p, h):
        out = []
        for v in h:
            if v < 12:
                out.append(p[v])
            else:
                x, y = rev[v]
                out.append(mk(p[x], p[y]))
        return tuple(sorted(out))

    wins = []
    for w in range(4):
        s = tuple(sorted((seed[w][0], mk(seed[w][0], seed[w][1]),
                          mk(seed[w][0], seed[w][2]))))
        h1 = apply(gens[w], s)
        h2 = apply(gens[w], h1)
        assert apply(gens[w], h2) == s, "window is not a C3 orbit"
        wins.append([s, h1, h2])
    return wins


def _connected_path():
    """The connected path geometry: the holed frequency-2 geodesic icosahedron (four
    distinct windows A, B, C, R) extruded x I. The diquark is the interior carried
    state where A, B's independent cycles meet C's before the result window -- it has
    NO free boundary (dW carries only the A, B, C, R window tubes and the two caps),
    so the colored intermediate is kept interior. Returns (cobordism, windows)."""
    faces, mid = _geodesic2()
    wins = _windows(mid)
    holes = {h for w in wins for h in w}
    holed = [list(f) for f in faces if f not in holes]
    cells = tessera.Spacetime.prismCells(holed, 2, None)
    co = tessera.Spacetime.fromCells(3, [list(c) for c in cells], 1.0, 0.0)
    # The per-hole induced-orientation signs (ChainComplex.endSignCovector), grouped
    # per window -- so the carried inputs and the emergent result are read in the same
    # global surface orientation (the relabeling-invariant Stokes charge).
    flat_holes = [h for w in wins for h in w]
    covec = list(cob.ChainComplex.endSignCovector([list(f) for f in faces],
                                                  [list(h) for h in flat_holes]))
    signs = [covec[3 * b:3 * b + 3] for b in range(4)]
    return co, wins, signs


def measure_path():
    """Measure the connected path's singlet reachability and the welded/free-boundary
    sequence + r_U confinement contrast. Returns a dict of the verdict numbers."""
    co, wins, signs = _connected_path()
    valid, _ = cob.EigenstateSynthesis(co, 1).dualComplexValid()
    es = cob.EigenstateSynthesis(co, 1)
    edge = _edge_index(es)
    in_holes = [h for w in wins[0:3] for h in w]    # A, B, C input windows
    r_holes = list(wins[3])                          # R result window
    M = _transport(es, edge, in_holes, r_holes)      # 3 x 9 input->result transport
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > 1e-9))
    U = np.linalg.svd(M)[0][:, :rank]
    reach = float(np.linalg.norm(U @ (U.conj().T @ _SINGLET)) / np.linalg.norm(_SINGLET))

    # G3 charge conservation (Stokes) on the connected path: carry three neutral
    # inputs (signed by each window's induced orientation) and read the signed
    # result -- sigma_R = -(sigma_A + sigma_B + sigma_C) = 0 for neutral inputs.
    in_targets = []
    for blk, state in enumerate((_A, _B, _C)):
        in_targets += [signs[blk][k] * state[k] for k in range(3)]
    psi = es.carriedRepresentative([list(h) for h in in_holes], in_targets)
    sigma_r = sum(signs[3][k] * _period(psi, edge, r_holes[k]) for k in range(3))

    # The welded/free-boundary bipartite SEQUENCE (read the diquark out, re-pin it):
    # the SHARED single register over-determines, so it floors short of the singlet.
    def merge(states):
        return cob.TransportCobordism([list(s) for s in states], max_iters=60,
                                      seed=SEED, topology=cob.RegisterTopology())
    ab = list(merge([_A, _B]).result)
    abc = list(merge([ab, _C]).result)

    # r_U: a colored state is over-determined on the SHARED bipartite register
    # (the welded sequence's free intermediate) but carries EXACTLY on DISTINCT
    # windows (the connected path's independent interior cycle).
    ru_shared = cob.TransportCobordism(
        [[1, 0, 0], list(_A)], max_iters=0, seed=SEED,
        topology=cob.RegisterTopology()).stats.state_residual
    ru_distinct = cob.TransportCobordism(
        [[1, 0, 0], list(_A), list(_B)], max_iters=0, seed=SEED,
        topology=cob.TripartiteRegisterTopology()).stats.state_residual

    betti = cob.ChainComplex.fromSpacetime(co).bettiNumbers()
    return {
        "valid": bool(valid),
        "b1": int(betti[1]) if len(betti) > 1 else 0,
        "transport_rank": rank,
        "reachability": reach,                 # singlet in the image (1.0 = reachable)
        "sigma_R_neutral": abs(complex(sigma_r)),      # Stokes: ~0 for neutral inputs
        "welded_singlet": _overlap(abc),       # the free-boundary sequence (floors)
        "welded_sigma": abs(sum(abc)),
        "ru_free_colored_shared": float(ru_shared),    # over-determined (~16)
        "ru_colored_distinct": float(ru_distinct),     # carries exactly (~3e-27)
    }


# --------------------------------------------------------------------------- #
def main():
    print("=== #416 bipartite revisit: twisted diquark + the connected path ===\n")
    tw = measure_twist()
    print("(a) TWISTED BIPARTITE TUBE -- the antisymmetric diquark channel "
          "(uniform metric)")
    for label in ("control", "twisted"):
        d = tw[label]
        print(f"    {label:8s}: A<->B antisym fraction = {d['antisym_fraction']:.4f}"
              f"   ||M_A+M_B|| = {d['m_blocks_sum_norm']:.2e}"
              f"   valid={d['valid']} b1={d['b1']}")
    print(f"    -> control is the PURE symmetric sextet 6 (antisym 0); the twist is "
          f"the EXACT antisymmetrizer onto 3bar (antisym 1).")
    print(f"    -> Delta|antisym| = {tw['delta']:.4f}\n")

    pa = measure_path()
    print("(b) CONNECTED PATH -- the singlet with the diquark kept interior")
    print(f"    transport rank = {pa['transport_rank']}  "
          f"singlet reachability = {pa['reachability']:.4f}   "
          f"valid={pa['valid']} b1={pa['b1']}")
    print(f"    charge conservation |sigma_R| (neutral inputs) = "
          f"{pa['sigma_R_neutral']:.2e}  (Stokes)")
    print(f"    welded/free-boundary sequence singlet overlap = "
          f"{pa['welded_singlet']:.4f}  (|sigma_ABC| = {pa['welded_sigma']:.4f})")
    print(f"    r_U(colored, SHARED register, free)   = "
          f"{pa['ru_free_colored_shared']:.3e}   (over-determined: averages)")
    print(f"    r_U(colored, DISTINCT window, interior) = "
          f"{pa['ru_colored_distinct']:.3e}   (carries exactly)\n")

    print("=== VERDICT ===")
    print("  Bipartite-WITH-PROJECTION produces the antisymmetric diquark 3bar: the")
    print("  orientation-reversing twist is an EXACT geometric antisymmetrizer")
    print(f"  (antisym 0 -> {tw['twisted']['antisym_fraction']:.2f}). The color singlet")
    print(f"  is REACHABLE through a connected path that keeps the diquark interior")
    print(f"  on an independent cycle (reachability {pa['reachability']:.2f}, full-rank")
    print("  transport), while the welded/free-boundary sequence floors")
    print(f"  ({pa['welded_singlet']:.2f}) because the colored diquark on a SHARED")
    print("  register is over-determined. Confinement-of-the-free-intermediate is the")
    print("  welded-sequence obstruction -- it is NOT fundamental: keeping the diquark")
    print("  interior on a distinct cycle (r_U ~ 0) escapes it.")


if __name__ == "__main__":
    main()
