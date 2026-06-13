# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The level-1 stabilizer law: S_3 transport on hexagon-join ends.

#279 charted the level-1 transports on icosahedron-end prism fills: exactly
C_3 -- the setwise stabilizer of the end's hole triple in its automorphism
group (the icosahedron has no hole-transposing automorphism, and the
randomized interior catalog could not enlarge the set). This example tests
the law it suggests --

    the level-1 realizable transports are the END's mapping classes:
    the image of the hole-triple stabilizer in Sym(holes) --

on the second seed the framework already owns: the hexagon-join L_2
register, whose stabilizer is the FULL S_3 (order 12 inside the 288-element
automorphism group, all six hole permutations induced). The fills are
4-dimensional staircase prisms (each tetrahedron x interval -> four
4-simplices) between two copies of the holed S^3; transport is read at
k = 2 from each end's three tet-boundary sphere periods. Predictions, all
falsifiable:

  * the straight fill transports exactly the identity (the level-1 anchor,
    b_2 of the fill = 2 by homotopy with the end);
  * each twisted fill (top end glued through a stabilizer element)
    transports exactly its induced hole permutation -- including the
    TRANSPOSITIONS the icosahedron ends provably cannot transport;
  * the factor swap (a_i <-> b_i), a nontrivial twist inducing the
    identity on holes, transports the identity -- mapping classes, not
    vertex maps, are what transport;
  * everything outside the six S_3 permutation gates floors with a
    certified period leak.

A note on gating: `dualComplexIsValid` is rigorous for n <= 3 only, and no
gated topology moves are used at dimension 4 in this example -- the fills
here are explicit constructions, not search outputs.

Run:
    python examples/cobordism/level1_stabilizer_law.py
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


BASE = _load("spectral_gate_realizability")   # thread caps + the gate battery
L2 = _load("l2_register_realizability")       # hexjoin cells + tet facets
L1 = _load("level1_fill_realizability")       # transport semantics + matching
np = BASE.np
tessera = BASE.tessera
cob = tessera.cobordism

REALIZE = BASE.REALIZE
CERT_FLOOR = BASE.CERT_FLOOR
_CP_IN = BASE._CP_IN

# The level-0 end: the hexagon join minus its three canonical tet holes (the
# holed S^3 whose ker L_2 carries the register).
_W3_CELLS = [c for c in L2._HEXJOIN if c not in set(L2._HEXJOIN_HOLES)]
_HOLES = list(L2._HEXJOIN_HOLES)
_N_END = 12                                    # vertices per layer

# --------------------------------------------------------------------------- #
# The hole-triple stabilizer of the hexagon join (order 12; image = full S_3).
# Vertex labels: a_i = i, b_j = 6 + j. Generators:
#   shift  sigma: (a_i, b_j) -> (a_{i+2}, b_{j+2})   -- a hole 3-cycle
#   reflect  rho: (a_i, b_j) -> (a_{1-i}, b_{1-j})   -- a hole transposition
#   swap     tau: a_i <-> b_i                        -- fixes every hole
# --------------------------------------------------------------------------- #
def _vmap(fa, fb):
    return {**{i: fa(i) % 6 for i in range(6)},
            **{6 + j: 6 + (fb(j) % 6) for j in range(6)}}


_SIGMA = _vmap(lambda i: i + 2, lambda j: j + 2)
_RHO = _vmap(lambda i: 1 - i, lambda j: 1 - j)
_TAU = {**{i: 6 + i for i in range(6)}, **{6 + j: j for j in range(6)}}
_IDENT = {v: v for v in range(_N_END)}

TWISTS = [
    ("straight", _IDENT),
    ("sigma", _SIGMA),
    ("sigma^2", L1._compose(_SIGMA, _SIGMA)),
    ("rho", _RHO),
    ("sigma.rho", L1._compose(_SIGMA, _RHO)),
    ("rho.sigma", L1._compose(_RHO, _SIGMA)),
    ("tau (factor swap)", _TAU),
]


def _is_end_automorphism(perm):
    """*perm* preserves the holed join's tet set and its hole set."""
    cells = {tuple(sorted(perm[v] for v in c)) for c in _W3_CELLS}
    holes = {tuple(sorted(perm[v] for v in h)) for h in _HOLES}
    return cells == set(_W3_CELLS) and holes == set(_HOLES)


def _prism4_cells(cells=_W3_CELLS, twist=None):
    """The staircase triangulation of (holed S^3) x I: for each tetrahedron
    (a<b<c<d), the four 4-simplices (a0,b0,c0,d0,d1), (a0,b0,c0,c1,d1),
    (a0,b0,b1,c1,d1), (a0,a1,b1,c1,d1) with x1 = twist(x) + 12. Adjacent
    prisms split shared walls by the same vertex-order rule. The staircase is
    the dimension-generic prism builder; this is its 4d (single-layer) face."""
    return tessera.Spacetime.prismCells([list(c) for c in cells], 1, twist)


def _bulk4(cells):
    """A pre-geometric 4-complex (top cells = 4-simplices), unit edge pin."""
    return tessera.Spacetime.fromCells(4, [list(c) for c in cells])


_REG_SIGN_CACHE = []


def _reg_sign():
    """The hexjoin L_2 register's induced-orientation sign pattern -- a
    property of the end, applied deterministically at both ends (the level-1
    icosahedron run showed per-fill null-vector normalization is
    sign-unstable)."""
    if not _REG_SIGN_CACHE:
        _REG_SIGN_CACHE.append(L2.RegisterL2().sign.copy())
    return _REG_SIGN_CACHE[0]


class Level1FillS3:
    """The level-1 register over hexagon-join ends: ker L_2 of a 4-dimensional
    fill whose boundary pair is two copies of the holed S^3. Transport is the
    pair of end sphere-period vectors; a fill realizes u' iff graph(u') lies
    in the restriction R -- the same semantics as the icosahedron-end fills
    of #279, one dimension up."""

    def __init__(self, twist=None):
        self.cells4 = _prism4_cells(twist=twist)
        self.holes0 = [tuple(sorted(h)) for h in _HOLES]
        self.holes1 = [tuple(sorted(v + 12 for v in h)) for h in _HOLES]
        self.reg_facets = [f for hole in (self.holes0 + self.holes1)
                           for f, _s in L2._tet_facets(hole)]
        self.fidx = {f: i for i, f in enumerate(self.reg_facets)}

        self.st = _bulk4(self.cells4)
        self.es = cob.EigenstateSynthesis(self.st, 2)
        self.cells = [tuple(int(v) for v in c) for c in self.es.cellSimplices()]
        harmonics = cob.HodgeLaplacian(self.st).harmonics(2)
        self.dim = len(harmonics)
        if self.dim:
            self.H_full = np.array(
                [[complex(h.amplitudeFor(list(c))) for c in self.cells]
                 for h in harmonics])
        else:
            self.H_full = np.zeros((0, len(self.cells)), dtype=complex)
        self._reg_col = [self.cells.index(f) for f in self.reg_facets]
        h_reg = self.H_full[:, self._reg_col] if self.dim else self.H_full
        rows = []
        for r in range(self.dim):
            p0 = [self._period(h_reg[r], hole) for hole in self.holes0]
            p1 = [self._period(h_reg[r], hole) for hole in self.holes1]
            rows.append(p0 + p1)
        self.P6 = np.array(rows).reshape(self.dim, 6)
        self.sign0 = _reg_sign()
        self.sign1 = _reg_sign()

    def _period(self, vec, hole):
        return sum(s * vec[self.fidx[f]] for f, s in L2._tet_facets(hole))

    @property
    def rank(self):
        return int(np.linalg.matrix_rank(self.P6, tol=1e-9)) if self.dim else 0

    def end_charge_leak(self):
        if not self.dim:
            return 0.0
        return float(max(np.max(np.abs(self.P6[:, 0:3] @ self.sign0)),
                         np.max(np.abs(self.P6[:, 3:6] @ self.sign1))))

    def harmonic_form(self, pair6):
        coeffs, *_ = np.linalg.lstsq(self.P6.T, pair6, rcond=None)
        full = (coeffs @ self.H_full).astype(complex)
        leak = pair6 - coeffs @ self.P6
        for k, hole in enumerate(self.holes0 + self.holes1):
            first_facet = L2._tet_facets(hole)[0][0]      # sign +1 by convention
            full[self._reg_col[self.fidx[first_facet]]] += leak[k]
        return full

    def spectral_residual(self, pair6):
        psi = self.harmonic_form(np.asarray(pair6, dtype=complex))
        return float(self.es.residual([complex(z) for z in psi]))

    def emergent_gate(self):
        if self.dim != 2:
            return None
        e_basis = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
                            [1.0, 1.0, -2.0] / np.sqrt(6.0)])
        A = (self.P6[:, 0:3] * self.sign0) @ e_basis.T
        B = (self.P6[:, 3:6] * self.sign1) @ e_basis.T
        if np.linalg.matrix_rank(A, tol=1e-9) < 2:
            return None
        return np.linalg.solve(A, B).T


def battery(fill, on_progress=None):
    """The 13 canonical blocks as candidate transports, hand-calculated
    targets (a, u'a) in each end's signed conventions."""
    a = _CP_IN.astype(complex)
    rows = []
    for name, u in L1._v_candidates():
        b = u @ a
        pair = np.concatenate([fill.sign0 * a, fill.sign1 * b])
        res = fill.spectral_residual(pair)
        coeffs, *_ = np.linalg.lstsq(fill.P6.T, pair, rcond=None)
        leak = float(np.linalg.norm(pair - coeffs @ fill.P6))
        rows.append({"gate": name, "residual": res, "leak": leak,
                     "realizable": bool(res < REALIZE)})
        if on_progress is not None:
            on_progress()
    return rows


_S3_GATES = ("Identity", "SWAP", "CNOT", "reversed-CNOT",
             "3-cycle (0231)", "3-cycle (0312)")


def _v_classes():
    """Group the 13 canonical candidates by their ACTION ON V: transport can
    only see the V-block, so gates equal on the charge-zero plane co-realize
    on any fill. (Discovered by this experiment: H(x)H = SWAP - J/2 with J
    the all-ones matrix, and J annihilates V -- on the carried register,
    double-Hadamard IS the swap of the holonomy generators.) Returns
    {gate name: sorted tuple of class member names}."""
    e_basis = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
                        [1.0, 1.0, -2.0] / np.sqrt(6.0)])
    cands = L1._v_candidates()
    classes = {}
    for name, u in cands:
        members = sorted(n2 for n2, u2 in cands
                         if np.max(np.abs((u2 - u) @ e_basis.T)) < 1e-9)
        classes[name] = tuple(members)
    return classes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("The level-1 stabilizer law: S_3 transport on hexagon-join ends\n"
          "  (4-dimensional prism fills between two holed-S^3 registers; the\n"
          "  end's hole-triple stabilizer is the FULL S_3, so all six hole\n"
          "  permutations -- including transpositions -- should transport)\n")
    prog = BASE._progress()

    for name, perm in TWISTS:
        check(f"{name} is an automorphism of the holed join",
              _is_end_automorphism(perm))

    results = []
    for name, perm in TWISTS:
        prog.phase(f"fill: {name}")
        fill = Level1FillS3(twist=perm)
        rows = battery(fill)
        prog.finish(f"{name} scored")
        realized = [r["gate"] for r in rows if r["realizable"]]
        floored = [r for r in rows if not r["realizable"]]
        emergent = L1.match_gate(fill.emergent_gate())
        results.append({"twist": name, "dim": fill.dim, "rank": fill.rank,
                        "emergent": emergent, "realized": realized,
                        "end_leak": fill.end_charge_leak(),
                        "floor_lo": min((r["residual"] for r in floored),
                                        default=None),
                        "leak_lo": min((r["leak"] for r in floored),
                                       default=None)})
        print(f"  {name:18} dim={fill.dim} rank={fill.rank} "
              f"end-leak={fill.end_charge_leak():.1e} "
              f"emergent={emergent or 'NOT A GRAPH'} realizes={realized}")

    classes = _v_classes()
    nontrivial_classes = {n: m for n, m in classes.items() if len(m) > 1}
    print(f"\n  V-action classes among the 13 candidates (gates equal on the "
          f"charge-zero plane co-transport): "
          f"{sorted(set(nontrivial_classes.values())) or 'all distinct'}")

    by_twist = {r["twist"]: r for r in results}
    straight = by_twist["straight"]
    check("the straight fill carries a 2-dim register (b_2 by homotopy)",
          straight["dim"] == 2)
    check("level-1 anchor: the straight fill transports exactly the "
          "identity's V-class",
          tuple(sorted(straight["realized"])) == classes["Identity"]
          and straight["emergent"] == "Identity")
    check("the factor swap (nontrivial twist, trivial hole action) "
          "transports the identity",
          tuple(sorted(by_twist["tau (factor swap)"]["realized"]))
          == classes["Identity"])
    twist_names = ["sigma", "sigma^2", "rho", "sigma.rho", "rho.sigma"]
    nontrivial = sorted({g for t in twist_names
                         for g in by_twist[t]["realized"]})
    check("each nontrivial twist transports exactly ONE V-class: its own "
          "mapping class",
          all(by_twist[t]["emergent"] is not None
              and tuple(sorted(by_twist[t]["realized"]))
              == classes[by_twist[t]["emergent"]]
              for t in twist_names))
    transported = sorted({g for r in results for g in r["realized"]})
    expected = sorted({m for g in _S3_GATES for m in classes[g]})
    check("the transported set is exactly the S_3 permutation gates' "
          "V-classes (the law: stabilizer image, as actions on V)",
          transported == expected)
    check("transpositions transport on hexjoin ends (impossible on "
          "icosahedron ends, #279)",
          any(g in ("SWAP", "CNOT", "reversed-CNOT") for g in nontrivial))
    check("every floored transport is leak-certified",
          all((r["leak_lo"] or 1.0) > 1e-6 for r in results))
    check("both ends conserve signed charge on every fill",
          all(r["end_leak"] < 1e-9 for r in results))

    print("\n  The law table (level-1 realizable transports = the end's "
          "hole-triple stabilizer image, as V-actions):")
    print("      icosahedron ends (#279):  stabilizer C_3  ->  "
          "{Identity, 3-cycle (0231), 3-cycle (0312)}")
    print(f"      hexagon-join ends (here): stabilizer S_3  ->  "
          f"{transported}")
    print("      (H(x)H rides with SWAP: H(x)H = SWAP - J/2 and J kills V, "
          "so they are the SAME action on the carried register.)")

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "level1_stabilizer_law.json")
        with open(path, "w") as handle:
            json.dump({"results": results, "transported": transported},
                      handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the level-1 conservation law holds on both seeds: the "
        "realizable transports are exactly the end's mapping classes (the "
        "hole-triple stabilizer image), C_3 on icosahedron ends and the "
        "full S_3 -- transpositions included -- on hexagon-join ends."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
