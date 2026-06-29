# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Connected two-hole spin correlator C_ij, two ways — the escape-hatch test (#512, part of #410).

Companion experiment to `docs/theory/cobordism/proton-spin/cartan_weyl_gluon.tex` §7. The
composite proton spin obeys

    J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩,     ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij,

where the **connected** correlator C_ij ≡ ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩ is the entangling content a
product read zeroes. On (C²)³ this is exact and decisive (`j2_from_pairs`): the proton
`2|uud⟩−|udu⟩−|duu⟩` carries Σ C_ij = −¾ (so J²=¾), while the product `|uud⟩` (J²=7/4) AND the Δ
`|uuu⟩` (J²=15/4) are both *product* states with C_ij = 0 — C_ij is precisely what tells the
entangled proton apart.

The open question (`cartan_weyl_gluon.tex` §6 "the one escape hatch"): can C_ij be read from
inter-hole **holonomy** alone — sidestepping the fiber↔cell (Whitney/Kähler–Atiyah) point-fiber
assembly? We compute C_ij two ways on the same emergent geometry and compare both to the proton
target:

  (a) VERTICAL — reconstruct per-hole spin from the single joint color-correlated carried
      representative (`dk_composite_spin.joint_spinors`), transport to a common frame via the
      `Spin(4)` `wilson_line`, build the 3-qubit state and read C_ij. A reconstruction from
      independent per-hole spinors is separable, so C_ij = 0 by construction — the floor.

  (b) HORIZONTAL — predict ⟨S_i·S_j⟩ from the inter-hole holonomy invariants ALONE: the SO(3)
      rotation angle θ_ij of the `wilson_line` (`⟨S_i·S_j⟩_holo = ¼ cos θ_ij`), with no
      contraction against endpoint spinors. Optionally also the emergent color-Z₃ periods.

Agreement *with the proton target* (non-zero C_ij summing to −¾, J²→¾) would mean the escape
hatch is open. Agreement *only at the floor* (both J² ≥ 3/2, C_ij ≈ 0) means it is closed and
the quantum two-hole lift is still required. The result is reported as-is.

Validation gates (GAUGE, RELABEL) are run post-hoc, never as a loop condition.
"""
import importlib.util
import math
import os
import sys

import cmath
import numpy as np

import tessera as T

cob = T.cobordism

_HERE = os.path.dirname(os.path.abspath(__file__))
_W = cmath.exp(2j * math.pi / 3)


def _load(name):
    sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load("dk_composite_spin")     # transport + spinor extraction (reused, not reimplemented)

# Clean spin-½ operators on a single qubit (C²); _S = σ/2.
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]
_S = [p / 2 for p in _PAULI]
_I2 = np.eye(2, dtype=complex)
_PAIRS = [(0, 1), (0, 2), (1, 2)]


# ============================================================================
# Operator layer on (C²)³ — the instrument in "chamber" (correlator) coordinates
# ============================================================================
def _s_op(a, i):
    """Spin operator `S_a` on qubit `i` of three (an 8×8 operator on (C²)³)."""
    ops = [_I2, _I2, _I2]
    ops[i] = _S[a]
    return np.kron(np.kron(ops[0], ops[1]), ops[2])


def spin_vector(psi8, i):
    """The Bloch vector `⟨S_a^(i)⟩`, a real 3-vector, of qubit `i` in the 3-qubit state."""
    st = psi8 / np.linalg.norm(psi8)
    return np.array([float((st.conj() @ _s_op(a, i) @ st).real) for a in range(3)])


def spin_dot(psi8, i, j):
    """The full two-point function `⟨S_i·S_j⟩ = Σ_a ⟨S_a^(i) S_a^(j)⟩`."""
    st = psi8 / np.linalg.norm(psi8)
    return float(sum((st.conj() @ (_s_op(a, i) @ _s_op(a, j)) @ st).real for a in range(3)))


def connected_correlator(psi8, i, j):
    """`C_ij = ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩` — the connected (entangling) part. 0 for any product."""
    return spin_dot(psi8, i, j) - float(spin_vector(psi8, i) @ spin_vector(psi8, j))


def j2_from_pairs(psi8):
    """`J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩` — identical to the validated `dk_joint_spin.j2_three_qubit`,
    but expressed through the two-hole correlators this experiment reads."""
    return 2.25 + 2.0 * sum(spin_dot(psi8, i, j) for i, j in _PAIRS)


def correlator_report(psi8):
    """Per-pair `⟨S_i·S_j⟩`, `C_ij`, the marginal product, and the reconstructed `J²`."""
    return {
        "j2": j2_from_pairs(psi8),
        "sum_sdot": float(sum(spin_dot(psi8, i, j) for i, j in _PAIRS)),
        "sum_connected": float(sum(connected_correlator(psi8, i, j) for i, j in _PAIRS)),
        "sdot": {(i, j): spin_dot(psi8, i, j) for i, j in _PAIRS},
        "connected": {(i, j): connected_correlator(psi8, i, j) for i, j in _PAIRS},
    }


# ============================================================================
# Mesh helpers — transport between the emergent holes
# ============================================================================
def _spinor_to_qubit(s4):
    """Reduce a 4-component Dirac spinor to a clean spin-½ qubit along its Bloch vector
    (the spin direction read with the structural generators `cs._SG`)."""
    s4 = np.asarray(s4, complex)
    nrm = (s4.conj() @ s4).real
    if nrm < 1e-30:
        return np.array([1, 0], complex)
    n = np.array([(s4.conj() @ cs._SG[a] @ s4).real / nrm for a in range(3)])
    if np.linalg.norm(n) < 1e-9:
        return np.array([1, 0], complex)
    m = sum((n / np.linalg.norm(n))[a] * _PAULI[a] for a in range(3))
    w, v = np.linalg.eigh(m)
    return v[:, int(np.argmax(w.real))]


def _hole_cells_and_lines(st, holes, top_tuple):
    """The three hole cells and the `Spin(4)` `wilson_line`s `hole_j → hole_i` for all pairs.
    Returns `(cells, frames, hc, W)` where `W[(i,j)]` transports hole `j`'s frame to hole `i`'s."""
    cells = list(st.getTopSimplices())
    adj = cs._dual_adjacency(cells, top_tuple)
    frames = cs._frames(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    hc = [cell_of(h) for h in holes[:3]]
    W = {}
    for i, j in _PAIRS:
        W[(i, j)] = cs.wilson_line(cells, adj, hc[i], hc[j], frames)
    return cells, frames, hc, W


# ============================================================================
# (a) VERTICAL route — C_ij from the reconstructed joint state
# ============================================================================
def vertical_state(st, holes, top_tuple, joint=True):
    """The 3-qubit state reconstructed from the carried representative: per-hole spinors
    (`joint=True` → the single color-correlated `[1,ω,ω²]` read) transported to hole 0's frame
    and reduced to qubits. Separable by construction, so its `C_ij = 0` (the floor)."""
    spinors = (cs.joint_spinors(st, holes, top_tuple) if joint
               else cs.emergent_spinors(st, holes, top_tuple))
    _cells, _frames, _hc, W = _hole_cells_and_lines(st, holes, top_tuple)
    lines0 = [np.eye(4, dtype=complex) if j == 0 else W[(0, j)] for j in range(3)]
    if any(u is None for u in lines0):
        return None
    qubits = [_spinor_to_qubit(lines0[j] @ spinors[j]) for j in range(3)]
    state = np.kron(np.kron(qubits[0], qubits[1]), qubits[2])
    return state / np.linalg.norm(state)


def cij_vertical(st, holes, top_tuple, joint=True):
    """The vertical-route correlator report (reconstructed joint state). `None` if a Wilson
    line is missing."""
    state = vertical_state(st, holes, top_tuple, joint=joint)
    if state is None:
        return None
    rep = correlator_report(state)
    rep["route"] = "vertical(joint)" if joint else "vertical(product)"
    return rep


# ============================================================================
# (b) HORIZONTAL route — ⟨S_i·S_j⟩ from holonomy invariants only
# ============================================================================
def transport_so3(W):
    """The SO(3) rotation `R` that the `Spin(4)` transport `W` induces on the spin axes:
    `W S_a W† = Σ_b R_ba S_b`, read as `R_ba = Tr[S_b W S_a W†]` (with `Tr[S_a²]=1` for the
    structural generators `cs._SG`). Purely a function of the holonomy — no spinor is touched."""
    Wd = W.conj().T
    return np.array([[float((cs._SG[b] @ W @ cs._SG[a] @ Wd).trace().real)
                      for a in range(3)] for b in range(3)])


def transport_angle(W):
    """The rotation angle `θ` of the transport's SO(3) part: `cos θ = (Tr R − 1)/2`."""
    c = (np.trace(transport_so3(W)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def color_periods(st, holes):
    """The emergent color phases (arguments of the carried representative's periods over the
    holes), normalised so phase[0] = 0. Best-effort: returns `None` if the period API is absent."""
    es = cob.EigenstateSynthesis(st, 3)
    for getter in ("cyclePeriods", "periods"):
        fn = getattr(es, getter, None)
        if fn is None:
            continue
        try:
            per = np.asarray(fn([list(h) for h in holes[:3]]), dtype=complex).ravel()[:3]
            if per.size == 3 and np.all(np.abs(per) > 1e-12):
                ph = np.angle(per)
                return ph - ph[0]
        except Exception:
            continue
    return None


def cij_horizontal(st, holes, top_tuple):
    """The horizontal-route report: `⟨S_i·S_j⟩_holo = ¼ cos θ_ij` from the inter-hole Wilson-line
    SO(3) angles ALONE (and, if available, the color-Z₃ period estimate `¼ cos Δφ_ij`). `None`
    if a Wilson line is missing.

    NB this assigns each pair its own transport angle; unlike a realizable separable *state* it
    is not bound by the n=3 frustration floor, so it is the most generous holonomy-only read."""
    _cells, _frames, _hc, W = _hole_cells_and_lines(st, holes, top_tuple)
    if any(W[p] is None for p in _PAIRS):
        return None
    theta = {p: transport_angle(W[p]) for p in _PAIRS}
    sdot = {p: 0.25 * math.cos(theta[p]) for p in _PAIRS}
    sum_sdot = float(sum(sdot.values()))
    rep = {"route": "horizontal(spin-transport)",
           "theta_deg": {p: math.degrees(theta[p]) for p in _PAIRS},
           "sdot": sdot, "sum_sdot": sum_sdot,
           "j2": 2.25 + 2.0 * sum_sdot}
    ph = color_periods(st, holes)
    if ph is not None:
        csdot = {p: 0.25 * math.cos(float(ph[p[1]] - ph[p[0]])) for p in _PAIRS}
        rep["color_phase_deg"] = {p: math.degrees(float(ph[p[1]] - ph[p[0]])) for p in _PAIRS}
        rep["color_sdot"] = csdot
        rep["color_j2"] = 2.25 + 2.0 * float(sum(csdot.values()))
    return rep


# ============================================================================
# Comparison
# ============================================================================
_TARGETS = {"proton (J=½)": 0.75, "product floor (n=3)": 1.5,
            "product |uud⟩": 1.75, "mixture (3·¾)": 2.25, "Δ (J=3/2)": 3.75}


def compare(st, holes, top_tuple):
    """Run both routes on one geometry and report each vs the proton target and the floors."""
    vert = cij_vertical(st, holes, top_tuple, joint=True)
    vert_prod = cij_vertical(st, holes, top_tuple, joint=False)
    horiz = cij_horizontal(st, holes, top_tuple)
    return {"vertical_joint": vert, "vertical_product": vert_prod, "horizontal": horiz,
            "targets": _TARGETS}


def reaches_proton(rep, tol=0.1):
    """True iff a route's reconstructed `J²` lands on the proton ¾ (within `tol`) — the
    operational meaning of "the escape hatch is open" for that route."""
    return rep is not None and abs(rep["j2"] - 0.75) <= tol


# ============================================================================
# __main__ — instrument + a fixture, with the GAUGE / RELABEL gates
# ============================================================================
if __name__ == "__main__":
    import json

    _UP = np.array([1, 0], complex)
    _DN = np.array([0, 1], complex)

    def _kr(*a):
        out = a[0]
        for x in a[1:]:
            out = np.kron(out, x)
        return out

    print("== instrument: J² and Σ C_ij on hand-fed clean states ==")
    states = {
        "proton 2|uud⟩−|udu⟩−|duu⟩": 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP) - _kr(_DN, _UP, _UP),
        "product |uud⟩": _kr(_UP, _UP, _DN),
        "Δ |uuu⟩": _kr(_UP, _UP, _UP),
    }
    for name, psi in states.items():
        r = correlator_report(psi)
        print(f"  {name:28s}  J²={r['j2']:.4f}  ΣC_ij={r['sum_connected']:+.4f}  "
              f"(C_ij {'≠0 entangled' if abs(r['sum_connected'])>1e-9 else '=0 product'})")

    eo = _load("emergent_optimizer")
    _FIX = os.path.join(_HERE, "..", "..", "tests", "fixtures", "composite_spin")

    def _rebuild(cells, edges, perm=None):
        if perm is not None:
            cells = [[perm[v] for v in c] for c in cells]
            edges = {tuple(sorted((perm[a], perm[b]))): z for (a, b), z in edges.items()}
        st = T.Spacetime.fromCells(4, cells, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            e.setSquaredLength(edges[(a, b) if a < b else (b, a)])
        T.ReggeSolver(st, T.MatterConfiguration())
        return st

    def _load_fixture(name):
        d = json.load(open(os.path.join(_FIX, name)))
        cells = [list(c) for c in d["cells"]]
        edges = {}
        for k, (re, im) in d["edges"].items():
            a, b = (int(x) for x in k.split(","))
            edges[(a, b)] = complex(re, im)
        st = _rebuild(cells, edges)
        return d, cells, edges, st, eo.emergent_holes(st, 3)

    for fixture in ("synthetic_b3_3.json", "converged_b3_3.json"):
        print(f"\n== fixture {fixture} ==")
        d, cells, edges, st, holes = _load_fixture(fixture)
        tt = eo._top_tuple
        out = compare(st, holes, tt)
        for key in ("vertical_joint", "vertical_product", "horizontal"):
            rep = out[key]
            if rep is None:
                print(f"  {key:18s}: (Wilson line missing)")
                continue
            extra = ""
            if "theta_deg" in rep:
                extra = "  θ_ij=" + ", ".join(f"{v:.0f}°"
                                               for v in rep["theta_deg"].values())
            if "color_j2" in rep:
                extra += f"  color_J²={rep['color_j2']:.3f}"
            sc = rep.get("sum_connected")
            print(f"  {rep['route']:26s} J²={rep['j2']:.4f}"
                  + (f"  ΣC_ij={sc:+.4f}" if sc is not None else "") + extra)
        print(f"  proton target J²=0.75 | product floor 1.50 | mixture 2.25")
        print(f"  reaches ¾?  vertical={reaches_proton(out['vertical_joint'])}  "
              f"horizontal={reaches_proton(out['horizontal'])}")

    # ---- GAUGE / RELABEL gates on the horizontal read (the new observable) ----
    print("\n== GAUGE / RELABEL invariance of the horizontal J² (synthetic_b3_3) ==")
    d, cells, edges, st, holes = _load_fixture("synthetic_b3_3.json")
    tt = eo._top_tuple
    base = cij_horizontal(st, holes, tt)["j2"]

    import scipy.linalg as sla
    rng = np.random.default_rng(7)
    rmap = {}
    orig = cs.embed_cell

    def _patched(cell):
        c = orig(cell)
        key = tuple(sorted(c))
        if key not in rmap:
            aa = rng.standard_normal((4, 4))
            rmap[key] = sla.expm(aa - aa.T)
        return {v: rmap[key] @ x for v, x in c.items()}

    cs.embed_cell = _patched
    try:
        gauged = cij_horizontal(st, holes, tt)["j2"]
    finally:
        cs.embed_cell = orig

    import random
    allv = sorted({v for s in st.getTopSimplices() for v in tt(s)})
    shuf = allv[:]
    random.Random(3).shuffle(shuf)
    perm = dict(zip(allv, shuf))
    st2 = _rebuild(cells, edges, perm=perm)
    holes2 = [tuple(sorted(perm[v] for v in h)) for h in holes[:3]]
    relabeled = cij_horizontal(st2, holes2, tt)["j2"]
    print(f"  base J²={base:.4f}  |ΔGAUGE|={abs(gauged-base):.2e}  "
          f"|ΔRELABEL|={abs(relabeled-base):.2e}")
