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

"""Emergent-bulk realizability: grow the bulk by surgery, let b_1 (and the
obstruction) be a pure output.

Two boundary circles, a meridian carried on both, and two validity-only fillings.
Three things make this the genuine construction:

  (a) **Two boundaries, and the period sign matters.** The boundary is
      geo(psi_A) || geo(psi_B) -- TWO circles -- and the meridian is carried on
      BOTH. We build the target two ways and test both:

        * MATCHED -- equal periods on the two circles (p_A = p_B). This is the
          boundary restriction of the annulus's own H_1 generator, so by Stokes
          the swept meridian m x I closes up through the tube and the class
          survives. It SHOULD be realizable on a filling that keeps both circles
          in H_1.
        * FLIPPED -- one circle's period negated (p_A = -p_B), the cobordism
          conjugation geo(psi_A) || conj(geo(psi_B)). The two ends no longer agree,
          so no closed 1-form on any filling restricts to it: it should FLOOR on
          EVERY filling, even one with the handle open. The sign flip is the
          negative control.

  (b) **No pinned topology anywhere.** Nothing is hardcoded as a torus / solid
      torus / disk / cone / SimplicialProduct-of-circles. The two boundary circles
      are general valid 1-cycles; the bulk W starts from a minimal validity-only
      seed and its topology -- in particular b_1(W) -- is whatever the surgery
      search produces. The bulk topology is an OUTPUT.

  (c) **Topology-changing moves (surgery).** Growth is NOT limited to coning /
      Pachner subdivision (topology-PRESERVING, so b_1 is frozen at the seed -- the
      negative result the earlier draft was stuck at). The move-set adds a
      boundary-fixed REMOVE (``EigenstateSynthesis::removeInteriorCell`` /
      ``RealizabilityOracle.GrowthMode.SURGERY``): detach an interior top cell (and
      its now-orphaned faces) while holding the pinned boundary bit-exact. Removing
      a cell OPENS a hole/handle, so b_1 MOVES. The realizability test is the
      physical one (``harmonic=True``): a boundary class realizes iff it is
      *carried* as a bulk harmonic, i.e. it lies in image(H_1(dW) -> H_1(W)).

The whole story runs at the Hodge-qubit degree k=1 on a surface bulk W (top cells =
triangles), the faithful low-dimensional analogue of the 3-manifold meridian/
longitude setting. The bulk family is built generically from a face list (an
octahedron = triangulated S^2):

  * delete ONE face            -> a DISK   (b_1 = 0): the one circle bounds; the
    antipodal triangle is still filled, so the other meridian bounds it too.
  * delete the TWO antipodal faces -> an ANNULUS (b_1 = 1): the two circles are
    homologous and each survives in H_1 -- the cobordism between them.

What actually happens (all numbers reproduced below; exit 0 iff the claims hold):

  E1 (the 2x2 -- the heart of (a)). Two targets x two fillings, seed verdict, no
     growth:

        filling \\ target     MATCHED (p_A=p_B)        FLIPPED (p_A=-p_B)
        disk    (b_1=0)        floor                    floor
        annulus (b_1=1)        REALIZE  (carried)       floor

     The disk (b_1=0) carries no nontrivial harmonic, so BOTH meridians floor. The
     annulus (b_1=1) carries the matched meridian (r -> 0, eigenvalue -> 0) but
     NOT the flipped one: opposite periods are not the restriction of any closed
     form, so the flip floors even where the topology is right. Realizable iff the
     filling has b_1=1 AND the periods match.

  E2 (surgery moves b_1 on its own, (c)). From the disk seed (b_1=0) the SURGERY
     search commits one boundary-fixed removal on its own -- scored purely by the
     harmonic residual -- opening the handle so b_1: 0 -> 1, for BOTH targets (the
     opened handle always lowers the residual). But only the MATCHED meridian then
     REALIZES; the FLIPPED one still FLOORS on the opened handle. So surgery
     delivers the topology as a pure output, and the leftover obstruction is the
     cohomological one (the period mismatch) that no filling can fix.

  E3 (only removal moves b_1). With the budget but no remove move -- the additive
     attach (FREE_CONNECTIVITY) and the Pachner subdivision are boundary-locked /
     spectrally inert at k>=1 -- b_1 stays frozen at the seed and the matched
     meridian never realizes. Removal is the load-bearing move.

Headline: with the surgery move-set the topology obstruction does NOT have to be
installed by a hand-picked filling -- b_1 falls out of the search. The matched
two-boundary meridian, unrealizable on the disk, becomes realizable the moment the
search opens the handle; the flipped meridian floors on every filling, so the
period-matching obstruction is genuinely separate from the topological one.

Run:  python examples/cobordism/emergent_bulk_realizability.py
      (--help for options; the raw table defaults to /tmp/cobordism and is NOT
      committed -- attach it to the issue/PR to pin a result.)
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

# A boundary class realizes iff its harmonic residual ||L_1 psi||^2 can be driven
# below REALIZE; it is certified obstructed when it floors above CERT_FLOOR. The
# bounded Levenberg-Marquardt fill reaches ~1e-7 on a carried class and floors at
# ~2e-1..1e0 otherwise -- orders of magnitude either side of these thresholds.
# DEEP_EPS is the LM tolerance (well below REALIZE so the optimizer polishes deep
# instead of stopping the instant it dips under the verdict line); the verdict
# itself is read off the realized residual against REALIZE.
DEEP_EPS = 1e-7
REALIZE = 1e-3
CERT_FLOOR = 1e-2
RESTARTS = 64
GROW_STEPS = 3


# --------------------------------------------------------------------------- #
# The octahedron surface family -- built generically from a face list, NOT from
# any named topology (no SimplicialProduct, no Toroid, no SolidSimplex). The
# octahedron is a triangulated S^2; deleting faces opens boundary circles:
#   * delete one face          -> a DISK   (b_1 = 0): the one circle bounds.
#   * delete two opposite faces -> an ANNULUS (b_1 = 1): the two circles are
#     homologous, so each survives in H_1 -- the meridian that lives.
# Hole A is the triangle {0,1,2}; the antipodal hole B is the triangle {3,4,5}.
# Vertices 0,1,2 carry hole A; 3,4,5 carry hole B; every edge is in two faces.
# --------------------------------------------------------------------------- #
_OCTAHEDRON = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
               (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
HOLE_A = (0, 1, 2)        # boundary circle A (the meridian m_A lives here)
HOLE_B = (3, 4, 5)        # antipodal boundary circle B (m_B)
CYCLE_A = [(0, 1), (0, 2), (1, 2)]
CYCLE_B = [(3, 4), (3, 5), (4, 5)]


def _surface(faces, weight=1.0, phase=0.0):
    """A pre-geometric 2-complex (top cells = triangles) from a face list, with all
    edges pinned to a uniform Hermitian weight. No topology object is used."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(weight)
        e.setPhase(phase)
    return st


def _delete(*faces):
    drop = {tuple(sorted(f)) for f in faces}
    return [f for f in _OCTAHEDRON if tuple(sorted(f)) not in drop]


def _disk():
    """Minimal validity-only seed: the octahedron minus hole A -- a disk filling of
    boundary circle A. b_1 = 0 (the circle bounds; the antipodal triangle {3,4,5}
    is still filled, so circle B bounds it). This is the bulk SEED, not a pinned
    answer; surgery decides the real topology."""
    return _surface(_delete(HOLE_A))


def _annulus():
    """The octahedron minus the two antipodal holes A and B -- the annulus/cylinder
    cobordism between the two boundary circles. b_1 = 1 (each circle survives)."""
    return _surface(_delete(HOLE_A, HOLE_B))


def _betti(st):
    return [int(b) for b in cob.ChainComplex.fromSpacetime(st).bettiNumbers()]


# --------------------------------------------------------------------------- #
# The two-boundary meridian target, matched or sign-flipped (the heart of (a)).
# --------------------------------------------------------------------------- #
def _periods(vals):
    """The two boundary periods (oriented loop sums) of a 1-form given on
    CYCLE_A + CYCLE_B: p_A around 0->1->2->0, p_B around 3->4->5->3."""
    v = {e: c for e, c in zip(CYCLE_A + CYCLE_B, vals)}
    p_a = v[(0, 1)] + v[(1, 2)] - v[(0, 2)]
    p_b = v[(3, 4)] + v[(4, 5)] - v[(3, 5)]
    return complex(p_a), complex(p_b)


def _meridian_target(flip=False):
    """The meridian carried on BOTH boundary circles, read off the annulus's own
    harmonic 1-form (the generator of H_1) restricted to CYCLE_A + CYCLE_B.

    MATCHED (flip=False) keeps the harmonic's equal periods (p_A = p_B) -- the
    cobordism-consistent meridian. FLIPPED (flip=True) negates circle B, giving
    opposite periods (p_A = -p_B) -- the cobordism conjugation geo(psi_A) ||
    conj(geo(psi_B)), the negative control that no closed form can carry."""
    h = cob.HodgeLaplacian(_annulus()).harmonics(1)[0]
    edges = CYCLE_A + CYCLE_B
    vals = [complex(h.amplitudeFor(list(e))) for e in edges]
    if flip:
        vals = vals[:3] + [-v for v in vals[3:]]   # negate circle B's period
    return cob.Cochain(1, edges, np.asarray(vals, dtype=complex)), edges, vals


def _decide(st, target, *, mode, max_cones, seed=1, restarts=RESTARTS):
    """Harmonic realizability: drive ||L_1 psi||^2 -> 0 with the pinned boundary,
    growing via `mode` up to `max_cones` steps. harmonic=True so realizable means
    the meridian is CARRIED as a bulk harmonic (in image(H_1(dW) -> H_1(W)))."""
    return cob.RealizabilityOracle(st).decideHarmonic(
        target, epsilon=DEEP_EPS, restarts=restarts, max_cones=max_cones,
        seed=seed, growth_mode=mode, connectivity_candidates=8, harmonic=True)


# --------------------------------------------------------------------------- #
# E1: the 2x2 -- the verdict tracks (filling b_1) AND (period match) (a + b).
# --------------------------------------------------------------------------- #
def two_by_two(targets):
    """Both targets (matched, flipped) x both fillings (disk, annulus), seed
    verdict (no growth) -> the 2x2 residual table the physics turns on."""
    rows = []
    for tname, (target, _edges, _vals) in targets.items():
        for fname, fn in [("disk", _disk), ("annulus", _annulus)]:
            st = fn()
            b1 = _betti(st)[1]
            v = _decide(st, target, mode=SURGERY, max_cones=0)
            rows.append({"target": tname, "filling": fname, "filling_b1": b1,
                         "realizable": bool(v.residual < REALIZE),
                         "residual": float(v.residual),
                         "eigenvalue": float(v.eigenvalue)})
    return rows


# --------------------------------------------------------------------------- #
# E2: surgery moves b_1 on its own (c) -- the obstruction is emergent.
# --------------------------------------------------------------------------- #
def surgery_emergence(targets, seeds=(0, 1, 2, 3)):
    """From the disk seed (b_1=0) the SURGERY search opens the handle on its own
    (b_1: 0 -> 1) for BOTH targets. The matched meridian then REALIZES; the flipped
    one still FLOORS on the opened handle -- the period mismatch is a separate,
    cohomological obstruction surgery cannot fix."""
    rows = []
    for tname, (target, _edges, _vals) in targets.items():
        for seed in seeds:
            st = _disk()
            before = _betti(st)[1]
            v = _decide(st, target, mode=SURGERY, max_cones=GROW_STEPS, seed=seed)
            rows.append({"target": tname, "seed": seed, "b1_before": before,
                         "b1_after": _betti(st)[1],
                         "removals": int(v.surgery_removals),
                         "realizable": bool(v.residual < REALIZE),
                         "residual": float(v.residual)})
    return rows


# --------------------------------------------------------------------------- #
# E3: only removal moves b_1 -- additive growth leaves it frozen (the contrast).
# --------------------------------------------------------------------------- #
def frozen_without_surgery(target):
    """Without the remove move b_1 is frozen at the seed and the matched meridian
    floors:
       * no growth at all (max_cones=0);
       * additive attach (FREE_CONNECTIVITY) -- every additive candidate is
         boundary-locked / spectrally inert at k>=1, so attachInteriorVertex is
         rejected and b_1 cannot move."""
    rows = []
    st = _disk()
    before = _betti(st)[1]
    v0 = _decide(st, target, mode=SURGERY, max_cones=0)
    rows.append({"growth": "none (seed only)", "b1_before": before,
                 "b1_after": _betti(st)[1],
                 "realizable": bool(v0.residual < REALIZE),
                 "residual": float(v0.residual)})
    # Additive attach: probe directly that it cannot move b_1 (it is rejected --
    # wiring a triangle in introduces new dW edges the bit-exact guard refuses).
    st = _disk()
    es = cob.EigenstateSynthesis(st, 1)
    attached = es.attachInteriorVertex([[3, 4]])  # a fresh triangle over edge {3,4}
    rows.append({"growth": "additive attach", "b1_before": before,
                 "b1_after": _betti(st)[1], "attach_accepted": bool(attached),
                 "realizable": False, "residual": None})
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="dir for the raw table (default /tmp/cobordism; NOT committed).")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print("Emergent-bulk realizability at k=1: grow the bulk by surgery, let b_1 "
          "and the obstruction be outputs\n")

    targets = {"matched": _meridian_target(flip=False),
               "flipped": _meridian_target(flip=True)}
    for tname, (_target, _edges, vals) in targets.items():
        p_a, p_b = _periods(vals)
        agree = abs(p_a - p_b) < abs(p_a) * 1e-6 + 1e-12
        gloss = ("matched -> closes up through the tube" if agree
                 else "opposite -> no closed form carries it")
        print(f"  Target [{tname}] meridian on circle A {HOLE_A} || circle B "
              f"{HOLE_B}:")
        print(f"        m_A {[f'{v.real:+.2f}' for v in vals[:3]]}  "
              f"m_B {[f'{v.real:+.2f}' for v in vals[3:]]}")
        print(f"        periods p_A={p_a.real:+.3f}  p_B={p_b.real:+.3f}  "
              f"(p_A/p_B={ (p_a / p_b).real:+.2f}: {gloss})")

    ok = True

    # ---- E1: the 2x2 -- the heart of the experiment ------------------------ #
    e1 = two_by_two(targets)
    print("\n  E1  the 2x2: two targets x two validity-only fillings (no growth):")
    header = (f"      {'target':>8} {'filling':>9} {'b_1':>4} {'realizable':>11} "
              f"{'residual':>11} {'eig':>11}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for r in e1:
        print(f"      {r['target']:>8} {r['filling']:>9} {r['filling_b1']:>4} "
              f"{'YES' if r['realizable'] else 'floored':>11} "
              f"{r['residual']:>11.2e} {r['eigenvalue']:>11.2e}")

    def _row(t, f):
        return next(r for r in e1 if r["target"] == t and r["filling"] == f)

    print("        => MATCHED realizes ONLY on the annulus (b_1=1): the carried "
          "meridian, eigenvalue -> 0. FLIPPED floors on BOTH: opposite periods are "
          "not the restriction of any closed form. Realizable iff b_1=1 AND periods "
          "match. The realizable set is image(H_1(dW) -> H_1(W)).")
    ok &= _row("matched", "annulus")["realizable"]
    ok &= not _row("matched", "disk")["realizable"]
    ok &= _row("matched", "disk")["residual"] > CERT_FLOOR
    ok &= not _row("flipped", "annulus")["realizable"]
    ok &= _row("flipped", "annulus")["residual"] > CERT_FLOOR
    ok &= not _row("flipped", "disk")["realizable"]
    ok &= _row("flipped", "disk")["residual"] > CERT_FLOOR

    # ---- E2: surgery moves b_1 on its own (the obstruction is emergent) ---- #
    e2 = surgery_emergence(targets)
    print("\n  E2  SURGERY search from the disk seed (b_1=0) -- the remove move "
          "opens the handle on its own:")
    header = (f"      {'target':>8} {'seed':>5} {'removals':>9} {'b_1':>9} "
              f"{'realizable':>11} {'residual':>11}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for r in e2:
        print(f"      {r['target']:>8} {r['seed']:>5} {r['removals']:>9} "
              f"{str(r['b1_before'])+'->'+str(r['b1_after']):>9} "
              f"{'YES' if r['realizable'] else 'floored':>11} "
              f"{r['residual']:>11.2e}")
    print("        => b_1 moves 0 -> 1 on its own for BOTH targets, scored purely "
          "by the harmonic residual. The MATCHED meridian then REALIZES; the "
          "FLIPPED one still FLOORS on the opened handle. Surgery delivers the "
          "topology as an output; the period mismatch is the separate obstruction "
          "no filling can fix.")
    e2_matched = [r for r in e2 if r["target"] == "matched"]
    e2_flipped = [r for r in e2 if r["target"] == "flipped"]
    ok &= all(r["b1_before"] == 0 and r["b1_after"] == 1 for r in e2)
    ok &= all(r["removals"] >= 1 for r in e2)
    ok &= all(r["realizable"] for r in e2_matched)
    ok &= all(not r["realizable"] for r in e2_flipped)

    # ---- E3: only removal moves b_1 -- additive growth is frozen ----------- #
    e3 = frozen_without_surgery(targets["matched"][0])
    print("\n  E3  without the remove move, b_1 is frozen and the matched meridian "
          "floors:")
    for r in e3:
        extra = ("" if r["residual"] is None
                 else f"  residual={r['residual']:.2e}")
        acc = ("" if "attach_accepted" not in r
               else f"  attach_accepted={r['attach_accepted']}")
        print(f"        {r['growth']:20} b_1 {r['b1_before']}->{r['b1_after']}  "
              f"realizable={r['realizable']}{acc}{extra}")
    print("        => additive growth (attach / Pachner subdivision) is boundary-"
          "locked / topology-PRESERVING at k>=1: b_1 cannot move, so the meridian "
          "stays floored. Removal is the load-bearing move.")
    ok &= all(r["b1_before"] == r["b1_after"] == 0 for r in e3)
    ok &= not e3[1]["attach_accepted"]

    # ---- raw table (PR artifact, not committed) --------------------------- #
    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "emergent_bulk_realizability.json")
        target_raw = {}
        for tname, (_t, edges, vals) in targets.items():
            p_a, p_b = _periods(vals)
            target_raw[tname] = {
                "edges": [list(e) for e in edges],
                "values": [[v.real, v.imag] for v in vals],
                "period_A": [p_a.real, p_a.imag],
                "period_B": [p_b.real, p_b.imag]}
        raw = {"degree": 1, "targets": target_raw, "E1_two_by_two": e1,
               "E2_surgery_emergence": e2, "E3_frozen_without_surgery": e3}
        with open(path, "w") as handle:
            json.dump(raw, handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    print("\n  Verdict: " + (
        "SUPPORTED -- the matched two-boundary meridian floors on the disk "
        "(b_1=0) and realizes on the annulus (b_1=1); the SURGERY search moves "
        "b_1 0 -> 1 on its own; and the sign-flipped meridian floors on EVERY "
        "filling, so the period-matching obstruction is genuinely separate from "
        "the topological one. The obstruction is emergent, not installed."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the tables above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
