# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pair-loop dual-basis flavor read on a 3-hole register (#561, part of #410/#559).

The `docs/theory/cobordism/proton-spin/pair_loop_quarks.tex` experiment (its §7
"genuinely new, cheap experiment"), implemented literally — the shared reader
core behind the ``PairLoopFlavor`` observable and the
``examples/cobordism/pair_loop_flavor.py`` fixture battery (which imports from
here; the machinery lives in exactly one place). On a structure with three
register holes carrying the color singlet `[1, ω, ω²]`:

1. **Joint read.** One correlated multi-hole read — the SINGLE joint carried
   representative `ψ = EigenstateSynthesis(st, 3).carriedRepresentative(holes,
   target)` — never three independent per-hole extractions. The per-hole
   carried weight is the period `w_h = ∮_[h] ψ` over hole `h`'s boundary
   3-cycle (the five (-1)^j-signed tetrahedral facets of the removed 4-cell).
2. **Pair loops, no new geometry.** The loop `γ_ij` encircling holes `i, j` is
   homologous to `[i] + [j]`, so its period is `w_i + w_j` — formed
   arithmetically from the per-hole weights. Imposing the singlet
   (`Σ_h w_h = 0`) gives the Poincaré duality `[γ_ij] = -[k]`: each pair loop
   is minus the complementary hole (`pair_loops` / `complement_hole`).
3. **Flavor/taste charge.** The per-hole Dirac–Kähler charge read of the
   retired `dk_joint_spin.per_hole_flavor` was `DiracKahler.charge(lift(3, ψ))
   = Σ_c W_c |ψ_c|²` — the metric-weighted norm² (the DK charge density `j⁰_c
   = W_c |ψ_c|²` summed). #509 retired `DiracKahler`, but the same number needs
   only the surviving `HodgeLaplacian.weights(3)` (the per-3-cell metric
   volume weights `W_c`, in the same canonical cell order as
   `EigenstateSynthesis.cellSimplices()`). Localized to a cycle it is the
   charge that cycle carries: per hole `q_h = Σ_{c ∈ ∂h} W_c |ψ_c|²`, and per
   pair loop the weighted norm of ψ on the loop's combined cycle support
   `∂i ⊔ ∂j`. Because distinct holes have disjoint boundary facets this is
   exactly `q_i + q_j` (additivity is structural, not an approximation — see
   the criterion-(b) honesty note below). Equivalently: the symmetric metric
   operator is assembled with `B₄ = W₃^{1/2} ∂₄ W₄^{-1/2}`, so the CLOSED
   cochain behind a harmonic row ψ̃ is `χ = √W₃ ψ̃` (`∂₄ᵀχ = 0`), and the
   weighted norm is the plain norm² of that closed representative on the
   cycle — the charge lives on the topological object.

## Orientation (the RELABEL lesson)

`cyclePeriods`' raw facet signs orient every hole by its sorted vertex tuple —
a labeling convention, not physics: a vertex relabeling can flip one hole's
boundary orientation and silently pin `-t_h`. The label-free orientation is the
induced one: `ChainComplex.endSignCovector(top_cells, holes)` returns the
per-hole signs `σ_h = ±1` from the complex's own coherent (facet-sharing)
orientation, under which every closed form's signed periods obey
`Σ_h σ_h p_h = 0`. Targets are pinned as `σ_h · t_h` and weights read back as
`w_h = σ_h ∮_h ψ`, so `w` carries the singlet in the induced orientation. The
one remaining ambiguity is a single GLOBAL sign (the propagation root), which
is `-1 ∈ U(1)` on the register — gauge, not physics (see the GAUGE gate).

## Pre-registered criteria (pair_loop_quarks.tex §7; falsifiable)

(a) **Multiplicity 2:1** — the three pair-loop charges cluster as {u, u, d}:
    two alike, one apart. Quantified by `rho` = (spread of the closest pair) /
    (its separation from the odd one); `rho < RHO_MAX = 0.5` counts as 2:1.
(b) **The odd-one-out loop is the diquark loop** — the dual of the spectator
    (step-2) quark hole. HONESTY NOTE: with the additive pair charge, the
    charge-odd loop is automatically the dual of the charge-odd hole, so on a
    fixture (no build history) only the odd loop's IDENTITY is reportable.
    The falsifiable content of (b) is that the charge-odd hole coincides with
    the quark the build added last (the two step-1 holes are the diquark) —
    pass `diquark_pair` (the step-1 hole indices) when the specimen's build
    history is known; the criterion then passes iff the odd loop IS that pair.
    Build history travels via the campaign record / geometry-dump metadata —
    never guessed, never re-derived by re-running a build (the engine build is
    not process-deterministic).
(c) **GAUGE and RELABEL invariance** — see the gates below.

Failure of (b) on a genuine specimen falsifies the dual-flavor framing; either
outcome is a finding.

## Gates (post-hoc validation, never a loop condition)

* GAUGE — this readout is built from periods and metric-weighted norms only:
  no embedding, no per-cell frame exists to SO(4)-rotate (that gate is vacuous
  here by construction). The register's surviving gauge freedom is the global
  U(1) phase of the target, `t → e^{iθ} t` — which contains the Z₃ cyclic
  recolor of the singlet (a cyclic shift of `[1, ω, ω²]` is `ω^{±1}` times it)
  and the global orientation flip (`-1`). Charges must be invariant and loop
  periods covariant (`w → e^{iθ} w`).
* RELABEL — random vertex-id permutation, rebuild the complex from the
  permuted cells and edge lengths, re-read with the permuted holes: charges,
  `rho`, and the odd-one-out identity must match, and the oriented loop
  periods must match up to the global orientation sign.

Both gates hold to machine precision: directly-read quantities (charges,
periods, duality residuals) reproduce to `GATE_TOL = 1e-12` (observed ~1e-16
on the fixtures); the derived clustering ratio `rho` divides two small charge
differences, which amplifies eigensolve roundoff (observed ~1e-13), so tests
give `d_rho` the looser `RHO_GATE_TOL = 1e-9` while the odd-one-out identity
must match exactly.
"""
import cmath
import json
import math
import os
import random

import numpy as np

import tessera as T
from tessera.observe.register import (
    OMEGA,
    SINGLET,
    build_complex,
    induced_orientation_signs,
)
from tessera.observe.register import register_holes as _validated_register_holes

cob = T.cobordism

#: The three pair loops as (i, j) hole-index pairs; `γ_ij` encircles holes i, j.
PAIR_LOOPS = ((0, 1), (0, 2), (1, 2))

#: Criterion (a): the closest pair's spread over its separation from the odd
#: one must stay below this for a 2:1 (u:u:d) verdict.
RHO_MAX = 0.5

#: Both gates must reproduce every directly-read quantity to this tolerance.
GATE_TOL = 1e-12

#: Looser gate tolerance for the derived ratio `rho` (a quotient of two small
#: charge differences — it amplifies eigensolve roundoff by ~1/separation).
RHO_GATE_TOL = 1e-9

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "tests", "fixtures", "composite_spin")


def complement_hole(pair):
    """The hole index dual to the pair loop `γ_ij`: `[γ_ij] = -[k]`, the third
    index. The Poincaré-duality bookkeeping of pair_loop_quarks.tex Eq. (dual)."""
    (i, j) = pair
    return 3 - i - j


def load_fixture(name, fixtures_dir=_FIXTURES):
    """A composite_spin fixture as (meta, cells, edges): `cells` the 5-vertex
    top 4-cells, `edges` {(min_id, max_id): complex squared length}. The
    default directory resolves inside the repo tree (test-fixture
    convenience); pass `fixtures_dir` anywhere else."""
    with open(os.path.join(fixtures_dir, name)) as f:
        meta = json.load(f)
    cells = [list(c) for c in meta["cells"]]
    edges = {}
    for key, (re, im) in meta["edges"].items():
        a, b = (int(x) for x in key.split(","))
        edges[(a, b)] = complex(re, im)
    return meta, cells, edges


def build_spacetime(cells, edges, perm=None):
    """The live complex from explicit top cells + edge squared lengths,
    optionally relabeled by `perm` (vertex id -> vertex id). The skeleton is
    materialized in C++ (`ReggeSolver` via `build_complex`), so the degree-3
    cell universe exists."""
    if perm is not None:
        cells = [[perm[v] for v in c] for c in cells]
        edges = {tuple(sorted((perm[a], perm[b]))): z
                 for (a, b), z in edges.items()}
    return build_complex(cells, edges, dimensions=4)


def register_holes(st, count=3):
    """The structure's first `count` register holes (removed top 4-cells), in
    `MultiCobordism.emergent_holes` order — delegating to the single validated
    selection entry point (`tessera.observe.register.register_holes`: a
    deficit raises, a surplus warns naming the dropped holes) and returning
    the selected holes."""
    holes, _dropped = _validated_register_holes(st, count, degree=3)
    return holes


def _facet_indices(cell_index, hole):
    """Hole `h`'s five boundary tetrahedra as (cochain index, (-1)^j sign)
    pairs. Facet j of the sorted hole drops vertex v_j and carries (-1)^j —
    the same boundary-operator convention `cyclePeriods` documents, mirrored
    here so ψ's periods are read in the operator's own indexing. Facets are
    matched by vertex SET (never by an imposed order); the physical
    orientation is supplied separately by `induced_orientation_signs`."""
    hs = sorted(hole)
    out = []
    for j in range(len(hs)):
        facet = frozenset(hs[:j] + hs[j + 1:])
        out.append((cell_index[facet], (-1) ** j))
    return out


def joint_read(st, holes, target=SINGLET, es=None, sigma=None, weights=None,
               cell_index=None):
    """The single correlated multi-hole read. Returns a dict:

    * ``sigma`` — induced-orientation signs of the three holes;
    * ``psi`` — the joint carried representative of `σ·target` over `holes`;
    * ``r_u`` — `residualForPeriods` of that pin (~0 = the register carries it);
    * ``w`` — oriented per-hole carried weights `w_h = σ_h ∮_h ψ` (== target);
    * ``q`` — per-hole DK-style charges `q_h = Σ_{c ∈ ∂h} W_c |ψ_c|²`;
    * ``loop_w`` — pair-loop periods `w_i + w_j` in `PAIR_LOOPS` order;
    * ``loop_q`` — pair-loop charges (weighted norm² of ψ on `∂i ⊔ ∂j`);
    * ``dual_residual`` — `|w_i + w_j + w_k|` per loop (the `[γ_ij] = -[k]`
      bookkeeping; ~0 whenever the pinned target sums to zero).

    ``es`` / ``sigma`` / ``weights`` / ``cell_index`` let a shared read
    context (``tessera.observe.Register``) inject its cached
    `EigenstateSynthesis`, orientation signs, metric weights and canonical
    cell index; by default each is built fresh (identical values — the read
    is deterministic given the complex).
    """
    if len(holes) != 3 or len(target) != 3:
        raise ValueError("the pair-loop read is over exactly 3 holes")
    if es is None:
        es = cob.EigenstateSynthesis(st, 3)
    if cell_index is None:
        cell_index = {frozenset(t): i for i, t in enumerate(es.cellSimplices())}
    if weights is None:
        weights = np.asarray(cob.HodgeLaplacian(st).weights(3), float)
    if sigma is None:
        sigma = induced_orientation_signs(st, holes)
    hole_lists = [list(h) for h in holes]
    raw_target = [s * t for s, t in zip(sigma, target)]
    psi = np.asarray(es.carriedRepresentative(hole_lists, raw_target), complex)
    r_u = es.residualForPeriods(hole_lists, raw_target)

    facets = [_facet_indices(cell_index, h) for h in holes]
    w = [sigma[h] * sum(sgn * psi[c] for c, sgn in facets[h]) for h in range(3)]
    q = [float(sum(weights[c] * abs(psi[c]) ** 2 for c, _ in facets[h]))
         for h in range(3)]

    loop_w, loop_q, dual_residual = [], [], []
    for (i, j) in PAIR_LOOPS:
        loop_w.append(w[i] + w[j])
        support = {c for c, _ in facets[i]} | {c for c, _ in facets[j]}
        loop_q.append(float(sum(weights[c] * abs(psi[c]) ** 2 for c in support)))
        dual_residual.append(abs(w[i] + w[j] + w[complement_hole((i, j))]))
    return dict(sigma=list(sigma), psi=psi, r_u=r_u, w=w, q=q,
                loop_w=loop_w, loop_q=loop_q, dual_residual=dual_residual)


def odd_one_out(loop_q):
    """(odd loop index, rho): the loop whose charge sits farthest from the
    mean of the other two, and `rho` = |spread of the other two| / |that
    separation| — small `rho` is a clean 2:1 (u:u:d) clustering."""
    separations = []
    for k in range(3):
        others = [loop_q[i] for i in range(3) if i != k]
        separations.append(abs(loop_q[k] - (others[0] + others[1]) / 2.0))
    odd = int(np.argmax(separations))
    others = [loop_q[i] for i in range(3) if i != odd]
    rho = (abs(others[0] - others[1]) / separations[odd]
           if separations[odd] > 0 else float("inf"))
    return odd, rho


def evaluate_criteria(read, diquark_pair=None, rho_max=RHO_MAX):
    """The pre-registered criteria on a finished `joint_read`.

    (a) `multiplicity_2_1`: `rho < rho_max`.
    (b) `odd_is_diquark_loop`: only decidable with `diquark_pair` (the step-1
        hole-index pair from the specimen's build history) — True iff the
        charge-odd loop is exactly that pair; None on fixtures (no history).
    Duality bookkeeping and the gates are (2)/(c), reported alongside.
    """
    odd, rho = odd_one_out(read["loop_q"])
    verdict = dict(odd_loop=PAIR_LOOPS[odd],
                   dual_hole=complement_hole(PAIR_LOOPS[odd]),
                   rho=rho,
                   multiplicity_2_1=bool(rho < rho_max),
                   odd_is_diquark_loop=None)
    if diquark_pair is not None:
        verdict["odd_is_diquark_loop"] = (
            tuple(sorted(diquark_pair)) == PAIR_LOOPS[odd])
    return verdict


def read_structure(st, holes=None, target=SINGLET, diquark_pair=None):
    """The specimen entry point: `joint_read` + `evaluate_criteria` on a live
    complex (e.g. a built proton block: `st = Proton.block()`, holes from
    `Proton.quark_holes()`, `diquark_pair` from the build's step-1 history)."""
    holes = register_holes(st) if holes is None else [tuple(h) for h in holes]
    read = joint_read(st, holes, target)
    return read, evaluate_criteria(read, diquark_pair=diquark_pair)


# ---------------------------------------------------------------------------
# The gates — post-hoc validation, never a loop condition.
# ---------------------------------------------------------------------------

def gauge_gate(st, holes, base, theta=2.0 * math.pi * 0.371, target=SINGLET):
    """GAUGE: re-read with the target rotated by the global U(1) phase
    `e^{iθ}`. Returns the residual dict — every entry must be < `GATE_TOL`:
    charges invariant, loop periods covariant, duality/oddity unchanged."""
    phase = cmath.exp(1j * theta)
    gauged = joint_read(st, holes, [phase * t for t in target])
    odd0, rho0 = odd_one_out(base["loop_q"])
    odd1, rho1 = odd_one_out(gauged["loop_q"])
    return {
        "d_charge": max(abs(a - b) for a, b in zip(gauged["q"], base["q"])),
        "d_loop_charge": max(abs(a - b)
                             for a, b in zip(gauged["loop_q"], base["loop_q"])),
        "d_loop_period": max(abs(a - phase * b)
                             for a, b in zip(gauged["loop_w"], base["loop_w"])),
        "d_rho": abs(rho1 - rho0),
        "d_odd": float(odd1 != odd0),
    }


def relabel_gate(cells, edges, holes, base, seed=3, target=SINGLET):
    """RELABEL: random vertex-id permutation, rebuild, re-read with the
    permuted holes. Returns the residual dict — every entry must be
    < `GATE_TOL`. Oriented loop periods are compared up to the one global
    orientation sign (the `endSignCovector` propagation root, `-1 ∈ U(1)`)."""
    all_vertices = sorted({v for c in cells for v in c})
    shuffled = all_vertices[:]
    random.Random(seed).shuffle(shuffled)
    perm = dict(zip(all_vertices, shuffled))
    st2 = build_spacetime(cells, edges, perm=perm)
    holes2 = [tuple(perm[v] for v in h) for h in holes]
    relabeled = joint_read(st2, holes2, target)
    flip = -1.0 if (abs(relabeled["w"][0] + base["w"][0])
                    < abs(relabeled["w"][0] - base["w"][0])) else 1.0
    odd0, rho0 = odd_one_out(base["loop_q"])
    odd1, rho1 = odd_one_out(relabeled["loop_q"])
    return {
        "d_charge": max(abs(a - b) for a, b in zip(relabeled["q"], base["q"])),
        "d_loop_charge": max(abs(a - b) for a, b in
                             zip(relabeled["loop_q"], base["loop_q"])),
        "d_loop_period": max(abs(a - flip * b) for a, b in
                             zip(relabeled["loop_w"], base["loop_w"])),
        "d_rho": abs(rho1 - rho0),
        "d_odd": float(odd1 != odd0),
    }


def read_fixture(name, fixtures_dir=_FIXTURES, relabel_seed=3):
    """One full row: joint read + criteria + both gates on a fixture."""
    meta, cells, edges = load_fixture(name, fixtures_dir)
    st = build_spacetime(cells, edges)
    holes = register_holes(st)
    read = joint_read(st, holes)
    verdict = evaluate_criteria(read)
    gauge = gauge_gate(st, holes, read)
    relabel = relabel_gate(cells, edges, holes, read, seed=relabel_seed)
    return dict(name=name, b3=meta["b3"], kind=meta.get("kind", "?"),
                holes=holes, read=read, verdict=verdict,
                gauge=gauge, relabel=relabel)
