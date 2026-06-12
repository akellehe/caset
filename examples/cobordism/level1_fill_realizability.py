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

"""Hierarchical synthesis, level 1: a bulk between two realized registers.

The recursion that makes several interactions interpretable as time: a
completed level-0 interaction is itself a carrier -- the register bulk
W_AB holds its post-interaction state as a harmonic, so W_AB *is*
geo(Psi_AB) by the framework's own definition. This example treats two
copies of the canonical (icosahedral, three-holed) register as the pinned
boundary pair of a NEW three-dimensional bulk and runs the same staged
procedure one level up: hand-calculate the level-1 post-interaction state,
drive the genuine L_1 residual of the fill to zero, certify floors
otherwise. Time is the nesting level; the level-0 holonomy holes sweep
worldtubes through the level-1 bulk.

Semantics (the level-1 register). The level-1 state space is the pair of
boundary period vectors (p_0, p_1) in V (+) V -- each end's three
circle-periods, individually charge-zero. A level-1 gate u' is a map
V -> V between the ends; a fill realizes u' iff (a, u'a) is carried by
ker L_1 of the fill for carried inputs a -- iff graph(u') lies in R, the
restriction of ker L_1 to boundary periods. The controls and predictions,
all hand-derivable and all falsifiable:

  * NO bulk (the disjoint union of the two registers): ker L_1 = V (+) V,
    R is everything -- every u' "carried" trivially. No interaction
    without a bulk: the saturated case.
  * The TRIVIAL fill (the prism W x I): homotopy-equivalent to W, so
    ker L_1 stays 2-dimensional and R is the DIAGONAL -- the graph of the
    identity. The cylinder carries exactly u' = 1 (the level-1 T1 anchor)
    and floors every other gate.
  * A TWISTED fill (the prism glued through the icosahedron's order-3
    hole symmetry): the pullback harmonics transport periods through the
    twist, so R is the graph of the corresponding hole 3-cycle -- the
    twist realizes exactly that gate. The hole triple's setwise
    stabilizer is the bare C_3 (no transpositions), so prism-class fills
    are predicted to realize exactly the C_3 subgroup of the canonical
    thirteen: holonomy transport by mapping classes.
  * Interior surgery and gated stellar growth (multi-layer fills only:
    every prism tet spans adjacent layers, so vertex-interior tets first
    exist at three layers) deform R; the EMERGENT GATE of a fill is read
    off as the V-block u' with R = graph(u') whenever R is a graph.

Everything battery-shaped is imported from spectral_gate_realizability.py;
the 3-complex builder is imported from l2_register_realizability.py; all
topology moves are gated on the C++ dual-complex validity check (#275).

Run:
    python examples/cobordism/level1_fill_realizability.py
    python examples/cobordism/level1_fill_realizability.py --draws 60 --jobs 10
    python examples/cobordism/level1_fill_realizability.py --layers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter

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


BASE = _load("spectral_gate_realizability")   # sets the 10-CPU caps on import
L2 = _load("l2_register_realizability")
np = BASE.np
tessera = BASE.tessera
cob = tessera.cobordism

REALIZE = BASE.REALIZE
CERT_FLOOR = BASE.CERT_FLOOR
_CP_IN = BASE._CP_IN

# The level-0 register surface W: the icosahedron minus the three canonical
# holonomy holes (the pair of pants whose ker L_1 carries the register).
_W_FACES = [f for f in (tuple(sorted(t)) for t in BASE._ICO)
            if f not in set(BASE._CLASS_HOLES)]
_N_REG = max(v for f in BASE._ICO for v in f) + 1      # 12 vertices per layer

# The icosahedron's order-3 hole symmetry (the full setwise stabilizer of the
# canonical triple; see the register paper's isometric-chart remark). Twisted
# prisms glue the top layer through gamma -- the mapping-class transport.
_GAMMA = {0: 3, 1: 7, 2: 8, 3: 4, 4: 0, 5: 2,
          6: 11, 7: 9, 8: 5, 9: 1, 10: 6, 11: 10}
_IDENT = {v: v for v in range(_N_REG)}

def _end_sign(faces, circles):
    """An end's induced-orientation sign pattern (the level-0 ``reg.sign``
    analog). It is a property of the end SURFACE, not of the fill: each end's
    three circles bound that end's fundamental 2-chain, so every fill's
    harmonics satisfy the same signed charge constraint at each end.
    Deterministic -- ``ChainComplex.endSignCovector`` reads it straight off
    the fundamental chain, no per-fill null-vector normalization."""
    return np.asarray(cob.ChainComplex.endSignCovector(
        [list(f) for f in faces], [list(t) for t in circles]), dtype=float)


def _compose(g, h):
    return {v: g[h[v]] for v in h}


def _prism_cells(faces=_W_FACES, layers=1, twist=None):
    """The staircase triangulation of W x [0, layers]: for each face
    (a<b<c) and layer l, the three tets (a0,b0,c0,c1), (a0,b0,b1,c1),
    (a0,a1,b1,c1) with x0 = phi_l(x) + 12*l and x1 = phi_{l+1}(x) + 12*(l+1).
    Adjacent prisms split their shared quad walls by the same vertex-order
    rule, so the complex is consistent. *twist* (a vertex permutation of the
    register surface, e.g. _GAMMA) is applied cumulatively per layer, gluing
    the top end through the symmetry -- the mapping-torus-style twisted
    product whose harmonics transport periods through the twist."""
    twist = twist or _IDENT
    phi = [_IDENT]
    for _ in range(layers):
        phi.append(_compose(twist, phi[-1]))
    cells = []
    for ell in range(layers):
        lo, hi = phi[ell], phi[ell + 1]
        for (a, b, c) in faces:
            a0, b0, c0 = (lo[a] + 12 * ell, lo[b] + 12 * ell, lo[c] + 12 * ell)
            a1, b1, c1 = (hi[a] + 12 * (ell + 1), hi[b] + 12 * (ell + 1),
                          hi[c] + 12 * (ell + 1))
            cells += [tuple(sorted((a0, b0, c0, c1))),
                      tuple(sorted((a0, b0, b1, c1))),
                      tuple(sorted((a0, a1, b1, c1)))]
    return sorted(set(cells)), phi[layers]


def _hole_circles(shift, perm=None):
    """The three register circles of one end: the canonical hole triangles,
    relabeled through *perm* and shifted into the end's layer."""
    perm = perm or _IDENT
    return [tuple(sorted(perm[v] + shift for v in h))
            for h in BASE._CLASS_HOLES]


class Level1Fill:
    """The level-1 register: ker L_1 of a 3-dimensional fill whose boundary
    pair is two copies of the canonical level-0 register surface. Holds the
    fill, its degree-1 EigenstateSynthesis (the genuine L_1 residual core),
    the harmonic 1-forms, and the six-circle period matrix split into the two
    ends -- the restriction space R whose graphs are the realizable level-1
    gates. Surgery (`extra_holes`, interior tets) and gated stellar growth
    deform R; every move is accepted only while the dual complex stays valid.
    """

    def __init__(self, faces=_W_FACES, layers=1, twist=None, extra_holes=(),
                 grow_vertices=0, grow_seed=0):
        self.layers = int(layers)
        cells, _top_perm = _prism_cells(faces, self.layers, twist)
        self.seed_cells = cells
        # Each end is labeled by its OWN canonical register classes: the top
        # copy's hole set equals the canonical holes shifted into its layer
        # (a twist permutes the holes among themselves), so the twist shows
        # up where it belongs -- in the transported periods -- not in the
        # bookkeeping.
        self.circles0 = _hole_circles(0)
        self.circles1 = _hole_circles(12 * self.layers)
        self.reg_edges = [e for tri in (self.circles0 + self.circles1)
                          for e in BASE._cedges(tri)]
        self.eidx = {e: i for i, e in enumerate(self.reg_edges)}
        # Each end's surface faces, for the sign covector: layer 0 carries the
        # identity labeling, the top layer the shifted one (a twist permutes
        # the face set among itself, so the set is labeling-equal either way).
        self.end_faces0 = [tuple(sorted(f)) for f in faces]
        self.end_faces1 = [tuple(sorted(v + 12 * self.layers for v in f))
                           for f in faces]

        self.st = L2._bulk(cells)
        self.es = cob.EigenstateSynthesis(self.st, 1)
        self.grown = self._stellar_grow(grow_vertices, grow_seed)
        self.extra_opened = []
        for cell in extra_holes:
            cs = tuple(sorted(cell))
            avail = {tuple(sorted(int(v) for v in c))
                     for c in self.es.interiorTopCells()}
            if cs in avail and self.es.removeInteriorCell(list(cs)):
                ok, _why = self.es.dualComplexValid()
                if not ok:
                    self.es.restoreLastRemoval()
                    continue
                self.extra_opened.append(cs)
        self.read_spectral()

    def read_spectral(self):
        """(Re)read the spectral state of the CURRENT complex: dual verdict,
        harmonics, the six-circle period matrix, and each end's induced-
        orientation sign covector (the level-0 ``reg.sign`` analog, computed
        per end so twisted tops stay consistent). Call after any in-place
        surgery beyond the constructor's."""
        ok, why = self.es.dualComplexValid()
        self.dual_valid, self.dual_reason = bool(ok), str(why)
        self.cells = [tuple(int(v) for v in c) for c in self.es.cellSimplices()]
        self.H_full = np.asarray(
            cob.HodgeLaplacian(self.st).harmonicMatrix(1),
            dtype=complex).reshape(-1, len(self.cells))
        self.dim = int(self.H_full.shape[0])
        self._reg_col = [self.cells.index(e) for e in self.reg_edges]
        self.P6 = np.asarray(
            self.es.cyclePeriods([list(t) for t in (self.circles0 + self.circles1)]),
            dtype=complex).reshape(self.dim, 6)
        self.sign0 = _end_sign(self.end_faces0, self.circles0)
        self.sign1 = _end_sign(self.end_faces1, self.circles1)
        return self

    def _stellar_grow(self, n, seed):
        """ADD up to *n* interior vertices by the gated composed stellar move
        (cone a vertex onto an interior tet's four faces, then remove the
        parent), exactly as the L_2 register does. Pure prisms have interior
        tets only at three or more layers (every tet spans adjacent layers),
        so the budget is unused on thin fills -- documented geometry."""
        rng = random.Random(int(seed))
        grown = 0
        for _ in range(int(n)):
            sites = sorted(tuple(sorted(int(v) for v in c))
                           for c in self.es.interiorTopCells())
            if not sites:
                break
            cell = rng.choice(sites)
            fan = [list(f) for f, _s in L2._tet_facets(cell)]
            if not self.es.attachInteriorVertex(fan):
                continue
            if not self.es.removeInteriorCell(list(cell)):
                self.es.detachLastInteriorVertex()
                continue
            ok, _why = self.es.dualComplexValid()
            if not ok:
                self.es.restoreLastRemoval()
                self.es.detachLastInteriorVertex()
                continue
            grown += 1
        if grown:
            for e in self.st.getEdgeList().toVector():
                e.setSquaredLength(1.0)
                e.setPhase(0.0)
        return grown

    # ---- the level-1 register read-outs ---------------------------------- #
    @property
    def rank(self):
        return int(np.linalg.matrix_rank(self.P6, tol=1e-9)) if self.dim else 0

    def end_charge_leak(self):
        """max |sign-weighted charge| over the two ends and all harmonics:
        each end's periods must individually conserve the (induced-
        orientation signed) charge -- Proposition-1 structure end by end."""
        if not self.dim:
            return 0.0
        return float(max(np.max(np.abs(self.P6[:, 0:3] @ self.sign0)),
                         np.max(np.abs(self.P6[:, 3:6] @ self.sign1))))


    def harmonic_form(self, pair6):
        """The carried representative of a six-period target (p_0, p_1), plus
        the minimal leak attached to one edge per circle so the cochain's
        periods are exact -- the level-1 twin of the register construction."""
        coeffs, *_ = np.linalg.lstsq(self.P6.T, pair6, rcond=None)
        full = (coeffs @ self.H_full).astype(complex)
        leak = pair6 - coeffs @ self.P6
        for k, tri in enumerate(self.circles0 + self.circles1):
            full[self._reg_col[self.eidx[BASE._cedges(tri)[0]]]] += leak[k]
        return full

    def spectral_residual(self, pair6):
        """The genuine Hodge residual of the 1-form with the given six
        periods on the fill -> 0 iff the pair (p_0, p_1) is carried by R.
        One C++ call (the lstsq-project-leak-residual verdict primitive)."""
        return float(self.es.residualForPeriods(
            [list(t) for t in (self.circles0 + self.circles1)],
            [complex(z) for z in np.asarray(pair6, dtype=complex)]))

    def emergent_gate(self):
        """The V-block u' with R = graph(u'), in the flat-orthonormal basis
        of each end's symmetrized (sign-corrected) charge-zero plane -- or
        None when R is not a graph (the end-0 block is singular). The
        cylinder must emit the identity; a gamma-twisted fill must emit the
        corresponding hole 3-cycle."""
        if self.dim != 2:
            return None
        e_basis = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
                            [1.0, 1.0, -2.0] / np.sqrt(6.0)])
        A = (self.P6[:, 0:3] * self.sign0) @ e_basis.T   # dim x 2, end 0
        B = (self.P6[:, 3:6] * self.sign1) @ e_basis.T   # dim x 2, end 1
        if np.linalg.matrix_rank(A, tol=1e-9) < 2:
            return None
        return np.linalg.solve(A, B).T                   # u2 with B = A u2^T


# --------------------------------------------------------------------------- #
# The disconnected-union control: two register surfaces, no fill. ker L_1 is
# the direct sum, R is all of V (+) V -- every gate trivially "carried".
# --------------------------------------------------------------------------- #
def union_control():
    faces = list(_W_FACES) + [tuple(v + 12 for v in f) for f in _W_FACES]
    st = BASE._surface(faces)
    dim = len(cob.HodgeLaplacian(st).harmonics(1))
    return {"dim": dim, "saturated": bool(dim == 4)}


# --------------------------------------------------------------------------- #
# The level-1 battery: the thirteen canonical level-0 gates' holonomy blocks
# as candidate transports u' : V -> V, plus leak controls. A candidate is
# scored by the genuine residual of the hand-calculated pair (a, u'a).
# --------------------------------------------------------------------------- #
def _v_candidates():
    """(name, 3x3 block) for the canonical thirteen -- the V-preserving
    blocks, the only well-typed level-1 transports -- in battery order."""
    out = []
    for name, U, _fam in BASE._gates():
        if name in BASE.CANONICAL_SET:
            out.append((name, np.asarray(U, dtype=complex)[1:4, 1:4]))
    return out


def level1_battery(fill, on_progress=None):
    """Score every canonical candidate u' on *fill*: the input a is the
    V-generic level-0 post-interaction periods (the carried state of W_AB),
    the target is the hand-calculated pair (a, u'a) -- converted to raw
    periods through each end's induced-orientation signs, exactly as the
    level-0 scripts apply ``reg.sign``."""
    a = _CP_IN.astype(complex)
    rows = []
    for name, u in _v_candidates():
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


def match_gate(u2, tol=1e-6):
    """Name the canonical candidate whose V-block equals *u2*, if any."""
    if u2 is None:
        return None
    e_basis = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
                        [1.0, 1.0, -2.0] / np.sqrt(6.0)])
    for name, u in _v_candidates():
        cand = e_basis @ u @ e_basis.T
        if np.max(np.abs(cand - u2)) < tol:
            return name
    return None


# --------------------------------------------------------------------------- #
# The surgery catalog: seeded interior cuts + growth on a thick fill; record
# how R deforms and which gates the variants carry.
# --------------------------------------------------------------------------- #
def _catalog_worker(task):
    idx, base_seed, layers, max_cut, max_add = task
    rng = random.Random(base_seed * 1_000_003 + idx)
    grow = rng.randint(0, max_add)
    fill = Level1Fill(layers=layers, grow_vertices=grow,
                      grow_seed=rng.randrange(1, 2**31))
    sites = sorted(tuple(sorted(int(v) for v in c))
                   for c in fill.es.interiorTopCells())
    rng.shuffle(sites)
    cut = []
    for cell in sites[:rng.randint(0, max_cut)]:
        if fill.es.removeInteriorCell(list(cell)):
            ok, _why = fill.es.dualComplexValid()
            if not ok:
                fill.es.restoreLastRemoval()
                continue
            cut.append(cell)
    # re-read the spectral state after the catalog's own cuts
    try:
        fill.read_spectral()
    except Exception:
        return None
    if not fill.dim:
        return None
    realized = [r["gate"] for r in level1_battery(fill) if r["realizable"]]
    u2 = fill.emergent_gate()
    return {"dim": fill.dim, "rank": fill.rank, "n_cut": len(cut),
            "n_grown": fill.grown, "dual_valid": fill.dual_valid,
            "is_graph": bool(u2 is not None),
            "emergent": match_gate(u2),
            "realized": realized, "n_realized": len(realized),
            "end_leak": fill.end_charge_leak()}


def surgery_catalog(draws, jobs, layers=3, base_seed=12345, max_cut=3,
                    max_add=6, on_progress=None):
    tasks = [(i, base_seed, layers, max_cut, max_add) for i in range(draws)]
    results = [r for r in BASE._parallel_map(_catalog_worker, tasks, jobs,
                                             on_progress=on_progress) if r]
    carried = Counter(g for r in results for g in r["realized"])
    return {
        "draws": draws, "scored": len(results),
        "n_dual_invalid": sum(1 for r in results if not r["dual_valid"]),
        "dims": sorted(Counter(r["dim"] for r in results).items()),
        "graphs": sum(1 for r in results if r["is_graph"]),
        "emergent": sorted(Counter(r["emergent"] for r in results
                                   if r["emergent"]).items()),
        "carried_counts": sorted(carried.items()),
        "max_realized": max((r["n_realized"] for r in results), default=0),
        "beyond_c3": sorted({g for r in results for g in r["realized"]
                             if g not in ("Identity", "3-cycle (0231)",
                                          "3-cycle (0312)")}),
    }


# --------------------------------------------------------------------------- #
# Report + main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--layers", type=int, default=1,
                    help="prism layers of the anchor fill (default 1)")
    ap.add_argument("--draws", type=int, default=40,
                    help="surgery-catalog draws on the 3-layer fill (0 = skip)")
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()
    jobs = max(1, min(args.jobs, 10))

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("Hierarchical synthesis, level 1: a 3-bulk between two realized "
          "registers\n  (the level-0 bulks are the boundary carriers; the "
          "level-1 state space is the pair of end periods V (+) V; a fill "
          "realizes u' iff graph(u') lies in its restriction space R)\n")
    prog = BASE._progress()

    # -- the no-bulk control ------------------------------------------------ #
    uc = union_control()
    print(f"  No-bulk control (disjoint union of the two registers): "
          f"ker L_1 dim = {uc['dim']} -- R is ALL of V(+)V, every transport "
          f"trivially 'carried'. No interaction without a bulk.")
    check("the disconnected union is saturated (dim 4)", uc["saturated"])

    # -- the trivial fill: the level-1 anchor ------------------------------- #
    prog.phase("building the trivial fill")
    cyl = Level1Fill(layers=args.layers)
    prog.finish("fill ready")
    print(f"\n  Trivial fill (the {args.layers}-layer prism W x I, "
          f"|V|={int(cyl.st.getVertexList().size())}, "
          f"|C_3|={len(cyl.seed_cells)}): dim ker L_1 = {cyl.dim}, "
          f"rank R = {cyl.rank}, end-charge leak = {cyl.end_charge_leak():.1e}, "
          f"dual valid = {cyl.dual_valid}")
    diag_dev = (float(np.max(np.abs(cyl.P6[:, 0:3] * cyl.sign0
                                    - cyl.P6[:, 3:6] * cyl.sign1)))
                if cyl.dim else float("inf"))
    print(f"      R vs the diagonal (graph of the identity): "
          f"max|p_0 - p_1| = {diag_dev:.2e}")
    u2 = cyl.emergent_gate()
    print(f"      emergent gate: {match_gate(u2) or 'NOT A GRAPH'}")
    check("the fill keeps a valid dual complex", cyl.dual_valid)
    check("ker L_1 of the trivial fill is 2-dimensional (homotopy with W)",
          cyl.dim == 2)
    check("both ends conserve signed charge harmonic-by-harmonic",
          cyl.end_charge_leak() < 1e-9)
    check("R of the trivial fill is the diagonal (graph of the identity)",
          diag_dev < 1e-9)
    check("the emergent gate of the trivial fill is the identity",
          match_gate(u2) == "Identity")

    prog.phase("level-1 battery on the trivial fill", total=13)
    rows = level1_battery(cyl, on_progress=prog.on_tick)
    prog.finish("battery scored")
    realized = [r for r in rows if r["realizable"]]
    floored = [r for r in rows if not r["realizable"]]
    print(f"\n  Level-1 battery on the trivial fill (the 13 canonical blocks "
          f"as transports, hand-calculated targets (a, u'a)):")
    print(f"      realized ({len(realized)}): "
          + ", ".join(r["gate"] for r in realized))
    if floored:
        print(f"      floored ({len(floored)}): residuals in "
              f"{min(r['residual'] for r in floored):.2f}.."
              f"{max(r['residual'] for r in floored):.2f}, leaks in "
              f"{min(r['leak'] for r in floored):.2f}.."
              f"{max(r['leak'] for r in floored):.2f}")
    check("T1 at level 1: the identity transports across the trivial fill",
          any(r["gate"] == "Identity" and r["realizable"] for r in rows))
    check("ONLY the identity transports across the trivial fill",
          [r["gate"] for r in realized] == ["Identity"])
    check("every floored transport is certified by a nonzero leak",
          all(r["leak"] > 1e-6 for r in floored))

    # -- twisted fills: mapping-class transport ----------------------------- #
    print("\n  Twisted fills (top layer glued through the C_3 hole symmetry):")
    twist_ok = True
    for twist, label in ((_GAMMA, "gamma"),
                         (_compose(_GAMMA, _GAMMA), "gamma^2")):
        tw = Level1Fill(layers=1, twist=twist)
        em = match_gate(tw.emergent_gate())
        t_rows = level1_battery(tw)
        t_real = [r["gate"] for r in t_rows if r["realizable"]]
        print(f"      {label}: dim = {tw.dim}, emergent gate = {em}, "
              f"realizes = {t_real}")
        twist_ok &= (tw.dim == 2 and em is not None
                     and em.startswith("3-cycle") and t_real == [em]
                     and tw.dual_valid)
    check("each twisted fill realizes exactly its hole 3-cycle "
          "(mapping-class transport)", twist_ok)

    # -- interior room and the surgery catalog ------------------------------ #
    thin_sites = len(list(cyl.es.interiorTopCells()))
    thick = Level1Fill(layers=3)
    thick_sites = len(list(thick.es.interiorTopCells()))
    print(f"\n  Interior room: {args.layers}-layer fill has {thin_sites} "
          f"interior tets; the 3-layer fill has {thick_sites} (every prism "
          f"tet spans adjacent layers, so surgery first becomes possible at "
          f"three layers).")
    check("the thin fill has no interior tets (documented geometry)",
          thin_sites == 0 if args.layers <= 2 else True)
    check("the 3-layer fill has interior tets (surgery room)",
          thick_sites > 0)

    catalog = None
    if args.draws > 0:
        prog.phase("surgery catalog on the 3-layer fill", total=args.draws)
        catalog = surgery_catalog(args.draws, jobs, base_seed=args.seed,
                                  on_progress=prog.on_tick)
        prog.finish("catalog scored")
        print(f"\n  Surgery catalog ({catalog['scored']}/{catalog['draws']} "
              f"gated draws; cuts + growth on the 3-layer fill):")
        print(f"      dual-invalid final states: {catalog['n_dual_invalid']}")
        print(f"      dim ker L_1 distribution: {dict(catalog['dims'])}")
        print(f"      graph-like R: {catalog['graphs']}; emergent gates: "
              f"{dict(catalog['emergent']) or '(none)'}")
        print(f"      transports carried across variants: "
              f"{dict(catalog['carried_counts']) or '(none)'}")
        if catalog["beyond_c3"]:
            print(f"        => a variant carries a transport beyond the C_3 "
                  f"prediction: {catalog['beyond_c3']} -- the level-1 "
                  f"criterion is richer than mapping-class transport.")
        else:
            print("        => NO variant carries a transport beyond "
                  "{identity, the two 3-cycles}: on prism-class fills the "
                  "level-1 realizable set is the C_3 of mapping-class "
                  "transport, exactly as the twist construction predicts.")
        check("every catalog draw keeps a valid dual complex",
              catalog["n_dual_invalid"] == 0)

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "level1_fill_realizability.json")
        with open(path, "w") as handle:
            json.dump({"union": uc, "trivial_fill": {
                           "dim": cyl.dim, "rank": cyl.rank,
                           "diag_dev": diag_dev,
                           "battery": rows},
                       "catalog": catalog}, handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the hierarchical step closes: completed level-0 "
        "registers serve as boundary carriers, the level-1 anchor holds "
        "(the trivial fill transports exactly the identity, at machine "
        "zero), twisted fills realize exactly their hole 3-cycles "
        "(mapping-class transport), every floored transport is certified "
        "by a period leak, and every topology move keeps a valid dual "
        "complex. Time enters as the nesting level."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
