# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The level-1 value reading: H3 one level up, with hierarchical composition.

The stabilizer law (#280/#284) settled WHAT a fill transports; this example
reads the VALUE of the transport and tests it against the hand amplitude --
the level-1 twin of the level-0 H3 validation, plus the genuinely new claim:
values compose across levels and across stacked fills.

The chart. A graph-like fill carries a 2-dimensional ker L_k whose harmonics
are indexed by either end alone: g(alpha) is the harmonic with end-0 periods
alpha (its end-1 periods are then u' alpha automatically -- the fill supplies
its own transport), and g_out(q) is the harmonic with end-1 periods q. One
scale is fixed by the T1 anchor on the input chart,
s1^-1 = <g(beta_0), g(beta_0)> for the unit reference; after that every
number is a prediction. The level-1 value of a transport at a carried pair:

    Z_T(q, alpha) = s1 <g_out(q), g(alpha)>,

and the level-1 H3 claim is Z_T(q, alpha) = <q| u' |alpha> -- exact whenever
the input chart is isometric (the anchor-normalized Gram G1 = I). WHICH
fills have isometric charts is the level-1 form of the level-0 question, and
the answer found here is constructive: the STAIRCASE prism's diagonal
choices follow the global vertex order, so the end's C_3 does NOT extend to
it and its chart is measurably anisotropic (~1e-2); the EQUIVARIANT prism
(a center per wall quad and per prism cell -- no diagonal choices) carries
every end automorphism, the cyclic-symmetry-plus-reality argument applies
one level up, and its chart is isometric at machine precision. On any
anisotropic fill the failure is not noise but the same exact law one level
up:

    Z_T - <q|u'|alpha> = w~^dag (G1 - I) alpha~,   w~ = the e-basis
    coordinates of g_out(q)'s end-0 periods (no inverse of u' needed: the
    output-indexed chart reads them off),

verified here on the staircase fills and on gated cut and growth variants.
Composition is tested two ways: STACKED fills (a 2-layer gamma-twisted prism transports gamma^2, and
its value equals the matrix product of the layers' transports -- several
interactions composed in sequence through geometry, values multiplying), and
ACROSS LEVELS (the straight fill's value of trivial evolution reduces to the
level-0 anchored amplitude on the end register -- the anchors compose
consistently). The 4-dimensional hexjoin-end fills extend the value table to
the full S_3, including a transposition (CNOT on the rho twist) that
icosahedron ends cannot transport at all.

Run:
    python examples/cobordism/level1_value_reading.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load("spectral_gate_realizability")
L2 = _load("l2_register_realizability")
L1 = _load("level1_fill_realizability")
LAW = _load("level1_stabilizer_law")
np = BASE.np

REALIZE = BASE.REALIZE
_CP_IN = BASE._CP_IN

_E = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
               [1.0, 1.0, -2.0] / np.sqrt(6.0)])      # flat-ON basis of V



# --------------------------------------------------------------------------- #
# The equivariant prism: a triangulation of W x I with NO diagonal choices --
# a center vertex per wall quad and per prism cell, every face coned. The
# staircase prism's diagonals follow the global vertex order, so the end's
# C_3 does NOT extend to it (the measured ~1e-2 chart anisotropy below); the
# equivariant prism has no order-dependent choices, so every automorphism of
# the end extends canonically and the cyclic-symmetry-plus-reality argument
# applies one level up: its chart MUST be isometric.
# --------------------------------------------------------------------------- #
def _equivariant_prism_cells(faces=L1._W_FACES, layers=1, twist=None):
    twist = twist or {v: v for v in range(12)}
    phi = [{v: v for v in range(12)}]
    for _ in range(layers):
        phi.append({v: twist[phi[-1][v]] for v in phi[-1]})
    edges = sorted({e for f in faces for e in BASE._cedges(tuple(sorted(f)))})
    nxt = [12 * (layers + 1)]
    fresh = {}

    def center(key):
        if key not in fresh:
            fresh[key] = nxt[0]
            nxt[0] += 1
        return fresh[key]

    cells = []
    for ell in range(layers):
        lo = {v: phi[ell][v] + 12 * ell for v in range(12)}
        hi = {v: phi[ell + 1][v] + 12 * (ell + 1) for v in range(12)}
        wall = {e: center(("w", e, ell)) for e in edges}
        for f in faces:
            a, b, c = sorted(f)
            p_c = center(("p", (a, b, c), ell))
            cells.append(tuple(sorted((lo[a], lo[b], lo[c], p_c))))
            cells.append(tuple(sorted((hi[a], hi[b], hi[c], p_c))))
            for (x, y) in BASE._cedges((a, b, c)):
                w = wall[(x, y)]
                quad = [(lo[x], lo[y]), (lo[y], hi[y]),
                        (hi[y], hi[x]), (hi[x], lo[x])]
                for (u, v) in quad:
                    cells.append(tuple(sorted((u, v, w, p_c))))
    return sorted(set(cells))


class CellsFill:
    """A level-1 fill read from an explicit 3-cell list (the equivariant
    prisms) -- the same spectral surface as Level1Fill (P6, H_full, signs,
    emergent gate), construction decoupled. The duplication with Level1Fill
    is deliberate and temporary: #286 absorbs both into the C++ carried
    register."""

    def __init__(self, cells, layers=1):
        self.circles0 = L1._hole_circles(0)
        self.circles1 = L1._hole_circles(12 * layers)
        # the end register surfaces (the holed icosahedron at each end,
        # shifted by 12*layers on the top) -- the faces _end_sign reads the
        # induced-orientation charge covector off, mirroring Level1Fill
        self.end_faces0 = [tuple(sorted(f)) for f in L1._W_FACES]
        self.end_faces1 = [tuple(sorted(v + 12 * layers for v in f))
                           for f in L1._W_FACES]
        self.reg_edges = [e for tri in (self.circles0 + self.circles1)
                          for e in BASE._cedges(tri)]
        self.eidx = {e: i for i, e in enumerate(self.reg_edges)}
        self.st = L2._bulk(cells)
        self.es = BASE.tessera.cobordism.EigenstateSynthesis(self.st, 1)
        ok, why = self.es.dualComplexValid()
        self.dual_valid, self.dual_reason = bool(ok), str(why)
        self.cells = [tuple(int(v) for v in c)
                      for c in self.es.cellSimplices()]
        harmonics = BASE.tessera.cobordism.HodgeLaplacian(self.st).harmonics(1)
        self.dim = len(harmonics)
        self.H_full = np.array(
            [[complex(h.amplitudeFor(list(c))) for c in self.cells]
             for h in harmonics]) if self.dim else \
            np.zeros((0, len(self.cells)), dtype=complex)
        self._reg_col = [self.cells.index(e) for e in self.reg_edges]
        h_reg = self.H_full[:, self._reg_col] if self.dim else self.H_full
        rows = []
        for r in range(self.dim):
            p0 = [self._period(h_reg[r], t) for t in self.circles0]
            p1 = [self._period(h_reg[r], t) for t in self.circles1]
            rows.append(p0 + p1)
        self.P6 = np.array(rows).reshape(self.dim, 6)
        self.sign0 = L1._end_sign(self.end_faces0, self.circles0)
        self.sign1 = L1._end_sign(self.end_faces1, self.circles1)

    def _period(self, vec, tri):
        a, b, c = sorted(tri)
        return (vec[self.eidx[(a, b)]] + vec[self.eidx[(b, c)]]
                - vec[self.eidx[(a, c)]])

    def emergent_gate(self):
        if self.dim != 2:
            return None
        A = (self.P6[:, 0:3] * self.sign0) @ _E.T
        B = (self.P6[:, 3:6] * self.sign1) @ _E.T
        if np.linalg.matrix_rank(A, tol=1e-9) < 2:
            return None
        return np.linalg.solve(A, B).T


# --------------------------------------------------------------------------- #
# The end-indexed charts and the anchored value.
# --------------------------------------------------------------------------- #
class ValueReader:
    """The level-1 value layer over a graph-like fill (any object exposing
    P6, H_full, sign0, sign1, dim -- the icosahedron-end Level1Fill and the
    hexjoin-end Level1FillS3 both do)."""

    def __init__(self, fill, tol=1e-9):
        if fill.dim != 2:
            raise ValueError("value reading needs a 2-dim carried register")
        self.fill = fill
        self.P0 = fill.P6[:, 0:3]
        self.P1 = fill.P6[:, 3:6]
        if (np.linalg.matrix_rank(self.P0, tol=tol) < 2
                or np.linalg.matrix_rank(self.P1, tol=tol) < 2):
            raise ValueError("an end block is rank-deficient (not a graph)")
        beta0 = (_CP_IN / np.linalg.norm(_CP_IN)).astype(complex)
        self.beta0 = beta0
        g0 = self.g_in(beta0)
        self.scale = 1.0 / float(np.vdot(g0, g0).real)   # the T1 anchor

    def _coeffs(self, block, sign, cp, tol=1e-9):
        target = np.asarray(sign, dtype=complex) * np.asarray(cp, dtype=complex)
        coeffs, *_ = np.linalg.lstsq(block.T, target, rcond=None)
        leak = float(np.linalg.norm(target - coeffs @ block))
        if leak > tol:
            raise ValueError(f"state is not carried by this end (leak {leak:.1e})")
        return coeffs

    def g_in(self, alpha):
        """The fill harmonic with end-0 periods *alpha* (cp convention)."""
        return self._coeffs(self.P0, self.fill.sign0, alpha) @ self.fill.H_full

    def g_out(self, q):
        """The fill harmonic with end-1 periods *q* (cp convention)."""
        return self._coeffs(self.P1, self.fill.sign1, q) @ self.fill.H_full

    def out_input_coords(self, q):
        """The e-basis coordinates of g_out(q)'s END-0 periods -- the w~ of
        the deviation law, read off the output-indexed chart (no transport
        inverse needed)."""
        coeffs = self._coeffs(self.P1, self.fill.sign1, q)
        raw0 = coeffs @ self.P0
        return _E @ (np.asarray(self.fill.sign0) * raw0)

    def gram(self):
        """The anchor-normalized input-chart Gram on the flat-ON basis of V."""
        forms = [self.g_in(_E[k]) for k in range(2)]
        G = self.scale * np.array([[complex(np.vdot(a, b)) for b in forms]
                                   for a in forms])
        return G, float(np.max(np.abs(G - np.eye(2))))

    def transport_value(self, q, alpha):
        """Z_T(q, alpha) = s1 <g_out(q), g_in(alpha)>."""
        return self.scale * complex(np.vdot(self.g_out(q), self.g_in(alpha)))

    def value_rows(self, u3, states_q, states_a):
        """Z_T vs the hand amplitude <q|u'|alpha> for every pair, plus the
        exact deviation-law residual |(Z_T - amp) - w~^dag (G1 - I) alpha~|."""
        G, _dev = self.gram()
        rows = []
        for q in states_q:
            w_t = self.out_input_coords(q)
            for a in states_a:
                amp = complex(np.vdot(q, u3 @ a))
                z = self.transport_value(q, a)
                predicted = complex(np.conj(w_t) @ (G - np.eye(2)) @ (_E @ a))
                rows.append({"dev": abs(z - amp),
                             "law_residual": abs((z - amp) - predicted)})
        return rows


def _u3(fill):
    """The fill's own transport as a 3x3 block acting on V, from the measured
    emergent gate (e-basis 2x2 mapped back through the chart)."""
    u2 = fill.emergent_gate()
    if u2 is None:
        return None
    return _E.T @ u2 @ _E


def _states(n, seed):
    return BASE.random_carried_states(n, seed)


def _survey(fill, label, n_states=4, seed=2026):
    """The full value survey of one fill: Gram, value table vs its own
    transport, worst deviation, worst law residual."""
    vr = ValueReader(fill)
    u3 = _u3(fill)
    _G, gram_dev = vr.gram()
    qs = _states(n_states, seed)
    alphas = [vr.beta0] + _states(n_states - 1, seed + 1)
    rows = vr.value_rows(u3, qs, alphas)
    return {"label": label,
            "emergent": L1.match_gate(fill.emergent_gate()),
            "gram_dev": gram_dev,
            "n_pairs": len(rows),
            "worst_dev": max(r["dev"] for r in rows),
            "worst_law_residual": max(r["law_residual"] for r in rows)}


# --------------------------------------------------------------------------- #
# Anisotropic controls: gated cut / growth variants of the 3-layer fill.
# --------------------------------------------------------------------------- #
def _cut_variant(seed, n_cut=2):
    import random
    fill = L1.Level1Fill(layers=3)
    rng = random.Random(seed)
    sites = sorted(tuple(sorted(int(v) for v in c))
                   for c in fill.es.interiorTopCells())
    rng.shuffle(sites)
    for cell in sites[:n_cut]:
        fill.es.removeInteriorCellChecked(list(cell))   # gated cut, dual-validity
    fill.read_spectral()
    return fill


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("The level-1 value reading (H3 one level up, with hierarchical "
          "composition)\n  (one anchor per fill on the input chart; then "
          "every value is a prediction:\n  Z_T(q, alpha) = s1 <g_out(q), "
          "g_in(alpha)> vs the hand amplitude <q|u'|alpha>)\n")
    prog = BASE._progress()

    # ---- 3d fills: the equivariant C_3 family ------------------------------ #
    prog.phase("equivariant fills (icosahedron ends)")
    gamma2 = L1._compose(L1._GAMMA, L1._GAMMA)
    fills_3d = [
        ("equivariant straight",
         CellsFill(_equivariant_prism_cells(layers=1), layers=1)),
        ("equivariant gamma",
         CellsFill(_equivariant_prism_cells(layers=1, twist=L1._GAMMA),
                   layers=1)),
        ("equivariant gamma^2",
         CellsFill(_equivariant_prism_cells(layers=1, twist=gamma2),
                   layers=1)),
    ]
    surveys = [_survey(f, lbl) for lbl, f in fills_3d]
    prog.finish("surveyed")
    print(f"  {'fill':22} {'transport':16} {'max|G1-I|':>10} "
          f"{'worst |Z_T - amp|':>18} {'law residual':>13} {'pairs':>6}")
    for s in surveys:
        print(f"  {s['label']:22} {s['emergent']:16} {s['gram_dev']:>10.1e} "
              f"{s['worst_dev']:>18.2e} {s['worst_law_residual']:>13.1e} "
              f"{s['n_pairs']:>6}")
    check("every equivariant fill keeps a valid dual complex",
          all(f.dual_valid for _l, f in fills_3d))
    check("the equivariant charts are isometric (G1 = I: the end symmetry "
          "extends, so the level-0 proposition applies one level up)",
          all(s["gram_dev"] < 1e-9 for s in surveys))
    check("H3 at level 1 on the C_3 family: Z_T equals the hand amplitude "
          "on every pair", all(s["worst_dev"] < 1e-9 for s in surveys))

    # ---- anisotropic controls: the exact law ------------------------------ #
    prog.phase("anisotropic controls")
    controls = [_survey(L1.Level1Fill(layers=1), "staircase straight"),
                _survey(L1.Level1Fill(layers=1, twist=L1._GAMMA),
                        "staircase gamma")]
    for i, seed in enumerate((11, 23)):
        controls.append(_survey(_cut_variant(seed), f"cut variant {i}"))
    controls.append(_survey(L1.Level1Fill(layers=3, grow_vertices=3,
                                          grow_seed=7), "grown variant"))
    prog.finish("surveyed")
    for s in controls:
        print(f"  {s['label']:22} {s['emergent']:16} {s['gram_dev']:>10.1e} "
              f"{s['worst_dev']:>18.2e} {s['worst_law_residual']:>13.1e} "
              f"{s['n_pairs']:>6}")
    check("the deviation obeys the EXACT law on every anisotropic fill: "
          "Z_T - amp = w~^dag (G1 - I) alpha~",
          all(s["worst_law_residual"] < 1e-9 for s in controls))
    check("the staircase chart is measurably anisotropic (the diagonal "
          "choices break the end symmetry -- the constructive finding)",
          all(s["gram_dev"] > 1e-4 for s in controls[:2]))

    # ---- composition: stacked fills multiply ------------------------------ #
    prog.phase("stacked composition")
    stack = CellsFill(_equivariant_prism_cells(layers=2, twist=L1._GAMMA),
                      layers=2)                        # cumulative: gamma^2
    single = fills_3d[1][1]
    u_single = _u3(single)
    u_stack = _u3(stack)
    vr = ValueReader(stack)
    qs, alphas = _states(3, 5), [vr.beta0] + _states(2, 6)
    comp_dev = 0.0
    for q in qs:
        for a in alphas:
            z = vr.transport_value(q, a)
            amp_product = complex(np.vdot(q, u_single @ (u_single @ a)))
            comp_dev = max(comp_dev, abs(z - amp_product))
    prog.finish("composed")
    print(f"\n  Stacked composition: the 2-layer gamma-twisted prism "
          f"(transport {L1.match_gate(stack.emergent_gate())}) vs the "
          f"PRODUCT of the single layer's measured transport:")
    print(f"      worst |Z_T(stack) - <q| u_gamma u_gamma |alpha>| = "
          f"{comp_dev:.2e}")
    check("stacked fills compose: the 2-layer twisted prism's value equals "
          "the matrix product of the layers' transports", comp_dev < 1e-9)
    check("the stack's emergent transport is the square of the layer's",
          float(np.max(np.abs(_E @ (u_stack - u_single @ u_single) @ _E.T)))
          < 1e-9)

    # ---- composition across levels ----------------------------------------- #
    prog.phase("cross-level reduction")
    reg = BASE.Register()
    h_b = reg.harmonic_form(reg.sign * (vr.beta0))
    s0 = 1.0 / float(np.vdot(h_b, h_b).real)
    vr1 = ValueReader(fills_3d[0][1])
    cross_dev = 0.0
    for q in _states(3, 9):
        h_q = reg.harmonic_form(reg.sign * q)
        a0 = s0 * complex(np.vdot(h_q, h_b))             # level-0 amplitude
        z1 = vr1.transport_value(q, vr1.beta0)           # level-1 trivial value
        flat = complex(np.vdot(q, vr1.beta0))
        cross_dev = max(cross_dev, abs(a0 - flat), abs(z1 - flat))
    prog.finish("reduced")
    print(f"\n  Cross-level reduction: level-0 anchored amplitude on the end "
          f"register vs the straight fill's value of trivial evolution:")
    print(f"      worst deviation from the flat amplitude (both layers) = "
          f"{cross_dev:.2e}")
    check("the level-1 value of trivial evolution REDUCES to the level-0 "
          "amplitude (anchors compose consistently)", cross_dev < 1e-9)

    # ---- 4d fills: the S_3 family, transposition included ------------------ #
    prog.phase("4d fills (hexjoin ends)")
    fills_4d = [("4d straight", LAW.Level1FillS3()),
                ("4d sigma twist", LAW.Level1FillS3(twist=LAW._SIGMA)),
                ("4d rho twist", LAW.Level1FillS3(twist=LAW._RHO))]
    surveys_4d = [_survey(f, lbl, n_states=3) for lbl, f in fills_4d]
    prog.finish("surveyed")
    print(f"\n  {'fill':22} {'transport':16} {'max|G1-I|':>10} "
          f"{'worst |Z_T - amp|':>18} {'law residual':>13} {'pairs':>6}")
    for s in surveys_4d:
        print(f"  {s['label']:22} {s['emergent']:16} {s['gram_dev']:>10.1e} "
              f"{s['worst_dev']:>18.2e} {s['worst_law_residual']:>13.1e} "
              f"{s['n_pairs']:>6}")
    check("the exact law holds across the 4d S_3 family (staircase "
          "4-prisms: anisotropic chart, zero law residual)",
          all(s["worst_law_residual"] < 1e-9 for s in surveys_4d))
    check("a TRANSPOSITION's value obeys the law (CNOT on the rho twist) "
          "-- impossible on icosahedron ends",
          any(s["emergent"] == "CNOT" and s["worst_law_residual"] < 1e-9
              for s in surveys_4d))

    # ---- no value for non-transports --------------------------------------- #
    swap_u = next(u for n, u in L1._v_candidates() if n == "SWAP")
    a = _CP_IN.astype(complex)
    pair = np.concatenate([fills_3d[0][1].sign0 * a,
                           fills_3d[0][1].sign1 * (swap_u @ a)])
    coeffs, *_ = np.linalg.lstsq(fills_3d[0][1].P6.T, pair, rcond=None)
    leak = float(np.linalg.norm(pair - coeffs @ fills_3d[0][1].P6))
    print(f"\n  No-value certificate: SWAP on the straight fill leaves the "
          f"carried graph (pair leak {leak:.2f}) -- a non-transport has no "
          f"value, not a wrong one.")
    check("a non-transport has no carried pair (leak certificate)",
          leak > 1e-6)

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "level1_value_reading.json")
        with open(path, "w") as handle:
            json.dump({"c3_family": surveys, "controls": controls,
                       "composition_dev": comp_dev, "cross_level": cross_dev,
                       "s3_family": surveys_4d}, handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- H3 holds one level up: the anchored cross-end pairing "
        "equals the transport amplitude at machine precision on every "
        "ISOMETRIC fill (the equivariant C_3 family), and every anisotropic "
        "fill -- the staircase prisms, the cut and growth variants, and the "
        "4d S_3 family with its transposition -- deviates by EXACTLY the "
        "level-0 law (zero law residual). Stacked fills MULTIPLY (two "
        "twisted layers value as the product of their transports), and the "
        "level-1 value of trivial evolution reduces to the level-0 "
        "amplitude -- the hierarchy composes."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
