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

"""Mediated selection over the level-1 fill ensemble.

The mediation objective F_beta = r_U(W) + beta * |S_Regge(W*)| (#248) was
built to choose among the many bulks realizing one operation -- and #279
produced exactly that degenerate ensemble: level-1 fills (straight prisms
of several thicknesses, twisted prisms, gated cut and growth variants) all
realizing the identity transport at machine zero. The spectral layer cannot
tell them apart; this example asks what the gravitational term selects.

Every ensemble member is verified to realize the identity (the residual
side of F_beta ties at machine zero), then scored by the magnitude of the
dual Lorentzian Regge action (ReggeSolver.dualReggeAction, #247) on the
unit-pinned complex, with a Lorentzian-native comparison column: the same
prisms rebuilt with vertex times = layer, letting the tracked metric rule
assign timelike inter-layer edges (the CDT-natural fill; no nulls arise --
intra-layer edges have time difference 0, inter-layer exactly 1).

The selection questions charted (not assumed):
  * thickness: does F_beta prefer the thinnest fill between the two events?
  * twists: gluing through a symmetry is a relabeling -- the action must be
    blind to it (an isometry of the complex).
  * cuts vs growth at fixed thickness: the #281 spike showed cuts RAISE the
    action and stellar growth LOWERS it -- the catalog quantifies both.

Run:
    python examples/cobordism/level1_mediated_selection.py
    python examples/cobordism/level1_mediated_selection.py --variants 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
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
L1 = _load("level1_fill_realizability")
np = BASE.np
tessera = BASE.tessera

REALIZE = BASE.REALIZE
_CP_IN = BASE._CP_IN

BETAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]


def _regge_magnitude(st):
    """|S_Regge(W*)| -- the modulus of the dual Lorentzian Regge action on the
    circumcentric dual (the same reading mediated_gate_battery.py uses for the
    level-0 stages). The Sorkin action is complex in general; the mediation
    objective consumes the magnitude (#258)."""
    s = tessera.ReggeSolver(st, tessera.MatterConfiguration()).dualReggeAction()
    return float(abs(s))


def _identity_residual(fill):
    """The r_U side of F_beta for the identity transport on *fill*."""
    a = _CP_IN.astype(complex)
    pair = np.concatenate([fill.sign0 * a, fill.sign1 * a])
    return fill.spectral_residual(pair)


def _layered_time_bulk(layers, twist=None):
    """The same prism complex with vertex times = layer (the CDT-natural
    assignment): the tracked metric rule then makes intra-layer edges
    spacelike (equal times) and inter-layer edges timelike (time difference
    one). No unit re-pin -- this is the Lorentzian-native fill."""
    cells, _ = L1._prism_cells(layers=layers, twist=twist)
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {}
    for i in sorted({v for c in cells for v in c}):
        vmap[i] = st.createVertex(i, [float(i // 12)])   # time-only coordinate
    for c in cells:
        t = sorted(c)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
    return st


def _cut_variant(seed, n_cut):
    """A gated cut variant of the 3-layer fill (the #279 catalog flow,
    deterministic): remove up to *n_cut* interior tets, dual-validity gated,
    then re-read the spectral state."""
    fill = L1.Level1Fill(layers=3)
    rng = random.Random(seed)
    sites = sorted(tuple(sorted(int(v) for v in c))
                   for c in fill.es.interiorTopCells())
    rng.shuffle(sites)
    cut = 0
    for cell in sites[:n_cut]:
        ok, _why = fill.es.removeInteriorCellChecked(list(cell))
        if ok:
            cut += 1
    fill.read_spectral()
    return fill, cut


def build_ensemble(n_variants=8, base_seed=12345):
    """The level-1 identity-realizing ensemble: straight prisms (1, 2, 3
    layers), the two twisted prisms, and gated cut / growth variants of the
    3-layer fill. Returns rows of (label, fill, meta)."""
    members = []
    for layers in (1, 2, 3):
        members.append((f"straight L={layers}", L1.Level1Fill(layers=layers),
                        {"layers": layers, "n_cut": 0, "n_grown": 0,
                         "twist": None}))
    members.append(("gamma twist L=1", L1.Level1Fill(layers=1, twist=L1._GAMMA),
                    {"layers": 1, "n_cut": 0, "n_grown": 0, "twist": "gamma"}))
    members.append(("gamma^2 twist L=1",
                    L1.Level1Fill(layers=1,
                                  twist=L1._compose(L1._GAMMA, L1._GAMMA)),
                    {"layers": 1, "n_cut": 0, "n_grown": 0,
                     "twist": "gamma^2"}))
    rng = random.Random(base_seed)
    for i in range(n_variants):
        if i % 2 == 0:
            fill, cut = _cut_variant(base_seed * 7 + i, n_cut=1 + (i // 2) % 3)
            members.append((f"cut x{cut} (draw {i})", fill,
                            {"layers": 3, "n_cut": cut, "n_grown": 0,
                             "twist": None}))
        else:
            grow = 1 + (i // 2) % 4
            fill = L1.Level1Fill(layers=3, grow_vertices=grow,
                                 grow_seed=rng.randrange(1, 2**31))
            members.append((f"grown +{fill.grown} (draw {i})", fill,
                            {"layers": 3, "n_cut": 0, "n_grown": fill.grown,
                             "twist": None}))
    return members


def score_ensemble(members, on_progress=None):
    rows = []
    for label, fill, meta in members:
        res = _identity_residual(fill)
        emergent = L1.match_gate(fill.emergent_gate())
        # twisted fills carry their own mapping class; the identity-residual
        # tie holds within the untwisted family -- twists are scored for the
        # action-neutrality claim, with their own transport noted.
        realizes_identity = bool(res < REALIZE)
        s_mag = _regge_magnitude(fill.st)
        n_tets = len([c for c in fill.es.topCells()])
        rows.append({"label": label, **meta,
                     "dim": fill.dim, "dual_valid": fill.dual_valid,
                     "identity_residual": res,
                     "realizes_identity": realizes_identity,
                     "emergent": emergent,
                     "S": s_mag, "n_tets": int(n_tets),
                     "S_per_tet": s_mag / max(int(n_tets), 1)})
        if on_progress is not None:
            on_progress()
    return rows


def f_beta_table(rows, betas=BETAS):
    """F_beta = identity residual + beta * |S| per member; the argmin per
    beta over the identity-realizing members (the transport-degenerate
    family the mediation objective was built to break ties in)."""
    family = [r for r in rows if r["realizes_identity"]]
    table = []
    for beta in betas:
        scored = sorted(family,
                        key=lambda r: r["identity_residual"] + beta * r["S"])
        table.append({"beta": beta, "winner": scored[0]["label"],
                      "F": scored[0]["identity_residual"]
                           + beta * scored[0]["S"]})
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--variants", type=int, default=8,
                    help="cut/growth variants of the 3-layer fill (default 8)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("Mediated selection over the level-1 fill ensemble\n"
          "  (every member realizes the same transport at machine zero --\n"
          "  the spectral layer cannot tell them apart; the dual Lorentzian\n"
          "  Regge action is what selects)\n")
    prog = BASE._progress()
    prog.phase("building the ensemble")
    members = build_ensemble(n_variants=args.variants, base_seed=args.seed)
    prog.finish(f"{len(members)} fills")
    prog.phase("scoring", total=len(members))
    rows = score_ensemble(members, on_progress=prog.on_tick)
    prog.finish("scored")

    print(f"  {'fill':24} {'tets':>5} {'r(identity)':>12} {'|S_Regge|':>10} "
          f"{'|S|/tet':>8}  transport")
    for r in rows:
        print(f"  {r['label']:24} {r['n_tets']:>5} "
              f"{r['identity_residual']:>12.1e} {r['S']:>10.4f} "
              f"{r['S_per_tet']:>8.4f}  {r['emergent']}")

    by = {r["label"]: r for r in rows}
    s1 = by["straight L=1"]
    s3 = by["straight L=3"]
    check("every fill keeps a valid dual complex",
          all(r["dual_valid"] for r in rows))
    check("every untwisted fill realizes the identity at machine zero "
          "(the spectral tie F_beta breaks)",
          all(r["realizes_identity"] for r in rows if not r["twist"]))
    check("twists are action-neutral (gluing through a symmetry is an "
          "isometry of the complex)",
          abs(by["gamma twist L=1"]["S"] - s1["S"]) < 1e-9
          and abs(by["gamma^2 twist L=1"]["S"] - s1["S"]) < 1e-9)
    cuts = [r for r in rows if r["n_cut"] > 0]
    grown = [r for r in rows if r["n_grown"] > 0]
    check("every gated cut RAISES the action above the straight 3-layer fill",
          all(r["S"] > s3["S"] for r in cuts))
    growth_lowers = all(r["S"] < s3["S"] for r in grown)
    print(f"\n  growth vs the straight 3-layer fill: "
          f"{'LOWERS' if growth_lowers else 'mixed'} the action "
          f"({', '.join(f'{r[chr(83)]:.2f}' for r in grown)} vs {s3['S']:.2f})")

    table = f_beta_table(rows)
    print("\n  F_beta selection (argmin over the identity-realizing family):")
    for t in table:
        print(f"      beta = {t['beta']:>5}: {t['winner']}  "
              f"(F = {t['F']:.4f})")
    sel = {t["winner"] for t in table if t["beta"] > 0}
    check("a positive beta selects a single consistent winner",
          len(sel) == 1)
    winner = next(iter(sel)) if len(sel) == 1 else None
    min_s = min(r["S"] for r in rows if r["realizes_identity"])
    check("the winner is the minimal-action fill",
          winner is not None and abs(by[winner]["S"] - min_s) < 1e-12)
    check("the minimal-action fill is the THINNEST straight prism "
          "(mediation selects the smallest interpolating geometry)",
          winner == "straight L=1")

    # ---- the Lorentzian-native column ------------------------------------ #
    print("\n  Lorentzian-native fills (vertex times = layer; tracked metric "
          "rule assigns timelike inter-layer edges):")
    lorentz = []
    for layers in (1, 2, 3):
        st = _layered_time_bulk(layers)
        s = _regge_magnitude(st)
        lorentz.append({"layers": layers, "S": s})
        print(f"      L={layers}: |S_Regge| = {s:.4f}")
    check("the Lorentzian-native actions are finite (no null/degenerate "
          "dual cells -- intra-layer dt=0, inter-layer dt=1)",
          all(np.isfinite(r["S"]) for r in lorentz))
    check("Lorentzian-native selection agrees: thinner is smaller",
          lorentz[0]["S"] < lorentz[1]["S"] < lorentz[2]["S"])

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "level1_mediated_selection.json")
        with open(path, "w") as handle:
            json.dump({"ensemble": rows, "f_beta": table,
                       "lorentzian": lorentz}, handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the mediation objective does real work on the level-1 "
        "ensemble: among fills the spectral layer cannot distinguish, "
        "F_beta is blind to twists (isometries), penalizes cuts, and "
        "selects the thinnest straight prism -- the minimal interpolating "
        "geometry between the two interaction events."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
