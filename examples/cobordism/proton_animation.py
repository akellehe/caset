# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of the ONE-STEP **Proton** build: three q-q̄ pairs → the proton.

Drives the actual `tessera.cobordism.Proton` class through its experimental single-merge
arm: ONE `MultiCobordism` node (`Proton.direct_node`) whose inputs are the three bare
quarks `{1}`, `{ω}`, `{ω²}` (ω = exp(2πi/3)) plus their three anti-quarks (the
elementwise conjugates — three neutral q-q̄ pairs, as pair production demands) and whose
single output is the colorless proton singlet `{1, ω, ω²}`, read off the WHOLE cobordism
— the anti-baryon partner is left to emerge unpinned. No diquark intermediate, no
recombination/formation split — the canonical two-step reference build is animated by
`multicobordism_animation.py`; this example asks whether the proton can emerge in one go,
growing its whole topology from a single Δ⁴ simplex through stage 1's F-lowering
candidate draw.

The node is driven with the COMBINED `run` drive: an **init pass** (`grow_boundaries=True`)
that grows the color register, then an **evolution pass** (`grow_boundaries=False`) with the
boundary frozen. Every `run` iteration interleaves the stage-1 combinatorial update with the
stage-2 geometric relaxation, so the optimizer makes whichever kind of progress helps at each
point — no separate relaxation pass. The animation advances ONE `run` iteration per frame:
stage 1 keeps no state across iterations (the trap-door burst machinery is gone), so
splitting a pass into per-frame calls is exact — every accepted move and relaxation step
gets its own frame.

The figure is one panel row for the single node: traces, the primal complex, then the dual
split into spatial- and temporal-curvature panels:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U` vs step;
  * **color register** — the color-register (hole = quark) count and, separately, the Betti
    number `b_k` vs step (the proton's three registers appear as the node grows);
  * **complex** — a 2-D classical-MDS projection of the node's relaxing 1-skeleton; each
    emergent color hole (register) is outlined in red as a cell and numbered.
    Each frame is normalized to a fixed scale, Procrustes-aligned (rotation/reflection only)
    to the previous frame, and eased, with the view auto-fit — so the structure stays legible
    instead of collapsing into a dot.
  * **dual — spatial / temporal curvature** — the circumcentric dual graph
    (one node per top cell, edges across shared facets) at the primal cell centroids, colored
    by the local Regge curvature. The Lorentzian deficit ε is COMPLEX, so it splits into two
    panels: **spatial** = `Re ε·|★|` (the rotation angle-defect, from timelike hinges) and
    **temporal** = `Im ε·|★|` (the boost / light-cone content, from spacelike hinges — those
    whose normal plane is timelike). Both use a signed diverging colormap centered at 0.
  * **spare rows (single-node runs only)** — the null-face proximity trace, the descending
    singular-value spectrum of the register operator `L_k`, and TWO **mode-localization**
    panels painting the primal skeleton by the summed weight `|ψᵢ|²` of right singular
    vectors of `L_k`: the **near-kernel** panel uses the m smallest-σ modes (the spectrum
    panel's red tail), showing WHICH portion of the complex carries each almost-register;
    directly below it, the **near-null** panel uses the m largest-σ modes, which localize
    on cells whose content is collapsing (`W₃` = squared tet content ∝ det G, so
    σ_max ∝ 1/det G as a tet approaches tangency with the light cone) — it shows WHERE
    the complex is going null, localizing what the min |det G| trace reports as a scalar.
    Next to those, the **annihilation** pair: a trace of the broken conjugate-pair count
    of the eigenvalue spectrum (under the V² convention `L_k` is real and W-pseudo-
    symmetric, so eigenvalues are real — Krein signature ±1, the particle/antiparticle
    split — or in W-null conjugate pairs; pair formation is an opposite-signature
    collision at an exceptional point, the annihilation vertex, and each event steps the
    count by ±1), and the **annihilation heat**: the skeleton painted by Σ|ψ|² over the
    broken pairs — where the annihilated content lives (see `krein_modes.py`).

The figure title reports the live **convergence verdict**: whether the whole cobordism
carries the singlet `{1, ω, ω²}` (color residual `r_state` ≈ 0) on its ≥ 3 emergent color
holes.

It drives only the **public** `Proton` (`direct_node`),
`MultiCobordism` (the combined `run`, plus `betti`, `emergent_holes`,
`regge_action_gradient`, `r_state`, `r_u`, `objective`, `st`), and the geometry readers
(`Spacetime.getTopSimplices`/`getDualAdjacency`/`getSimplices`,
`Simplex.deficitAngle`/`dualVolume`) APIs — the *same* node setup
and drive `Proton.build_direct()` uses, so the animation shows the real class. The C++
engine is untouched.

**Visualization is off by default** — `run_build(...)` takes the fast batched path with no
per-step plotting overhead. Opt in with `visualize=True` (or `--live`/`--save`) to animate,
which is slower.

    # default: run the one-step build fast, no visualization
    python proton_animation.py
    # live (interactive backend):
    python proton_animation.py --live
    # headless: write a GIF (no display needed):
    python proton_animation.py --save proton.gif
    # pre-grow the single-Δ⁴ seed by 12 gated cone-ins before optimizing:
    python proton_animation.py --precone 12
    # give each frame up to 20 fresh candidate draws before it advances, and let
    # each draw search move-sequences up to 100 deep:
    python proton_animation.py --live --max-lookahead 20 --max-lookahead-depth 100
    # record every frame's dual-curvature panels for later numerical analysis
    # (see dual_perp_check.py):
    python proton_animation.py --live --dump-dir proton_dumps/run-1

Two independent lookahead knobs, easy to confuse:

  * ``--max-lookahead-depth`` — how far AHEAD one draw looks: the longest move
    *sequence* stage 1 will search when single moves stall. (Called
    ``--max-lookahead`` in earlier revisions.)
  * ``--max-lookahead`` — how many TIMES a frame redraws candidates before
    giving up and advancing. Candidates are drawn at random, so a stalled frame
    is usually an unlucky draw; only stalled frames spend the extra tries. Note
    that a stalled frame is not wasted — stage 2 still relaxes the geometry —
    so this buys committed *moves*, not progress in general.
"""
import argparse
import itertools
import json
import math
import os
import sys

# --threads must take effect BEFORE tessera loads: OpenMP reads OMP_NUM_THREADS
# at library initialization, so a post-import setting is silently ignored. The
# standing compute authorization for this box is 16 of its 32 cores; pass
# --threads 32 to use all of them for a run.
if "OMP_NUM_THREADS" not in os.environ or "--threads" in sys.argv:
    _n = "16"
    if "--threads" in sys.argv:
        try:
            _n = sys.argv[sys.argv.index("--threads") + 1]
        except IndexError:
            pass
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _n

import numpy as np
from scipy.sparse.csgraph import shortest_path

import tessera

cob = tessera.cobordism

from krein_modes import KreinModes

# Two combined-`run` passes on the one node — init (grow_boundaries=True) then evolution
# (grow_boundaries=False) — each interleaving the stage-1 surgery update with the stage-2
# geometric relaxation every iteration, so the optimizer makes whichever kind of progress
# helps at each point. The animation runs ONE iteration per frame (`_*_CHUNK = 1`): stage 1
# keeps no state across iterations, so per-frame chunking is exact and every accepted move
# and relaxation step is visible. (The batched no-visualization path still runs each pass
# as one call — same result, no per-frame overhead.)
_INIT_STEPS = 180          # init-pass (grow_boundaries=True) iterations per node
_EVOLVE_STEPS = 60         # evolution-pass (grow_boundaries=False) iterations per node
_INIT_CHUNK = 1            # init iterations per frame (1 = smoothest animation)
_EVOLVE_CHUNK = 1          # evolution iterations per frame
_STAGE1_CANDIDATES = 8
_MAX_LOOKAHEAD_DEPTH = 10  # deepen to sequences of up to this many moves on a stall
# Retries of the whole frame when stage 1 commits nothing. Stage-1 candidates are
# drawn at RANDOM (MultiCobordism.cpp: "the batches are random samples, so one miss
# is not proof no improving sequence exists"), and the engine concludes exhaustion
# after 3 consecutive no-effect iterations — but with one iteration per frame a
# stalled draw wastes the whole frame. Re-calling `run` redraws fresh candidates.
# 1 = the historical behaviour (a single draw per frame).
_MAX_LOOKAHEAD_TRIES = 1
                           # (deepened batches scan up to ~128 candidates each)
_COLOR_TOL = 0.5           # singlet r_state below this ⇒ the proton carries the color
_MIN_QUARK_HOLES = 3       # a proton is three quarks ⇒ three color registers

# Dual-complex curvature heat map. Curvature in Regge calculus is the deficit angle on
# hinges (the (d-2)=2-simplices, i.e. triangles); we localize it to each top cell (dual
# node) as the SIGNED sum over its hinges of Re(lorentzian deficit) · |dual volume| — the
# Regge angle-defect action density, keeping ε's sign so negative (saddle) curvature shows.
# `Simplex.deficitAngle` is expensive, so the heat is recomputed only
# every `_HEAT_REFRESH_EVERY` frames on the active node (the frozen node's geometry doesn't
# change, so its heat is cached) — the cheap dual *graph* still redraws every frame.
_HEAT_CMAP = "coolwarm"    # spatial (Re): diverging, blue = negative, white ≈ 0, red = positive
_HEAT_CMAP_IM = "PuOr"     # temporal (Im): distinct diverging map for the boost/rapidity part
_HEAT_REFRESH_EVERY = 4
# Mode weight is a non-negative share (the |ψ|² sum to 1 over the k-cells), so
# both mode maps are SEQUENTIAL, unlike the signed curvature maps: low = the
# modes don't live here, high = they do. The two panels use visually disjoint
# maps so they can never be confused: near-kernel (tail) runs dark → bright
# yellow, near-null (head) runs pale cyan → magenta.
_MODE_CMAP = "magma"
_MODE_CMAP_HEAD = "cool"
# The annihilation heat gets a third disjoint sequential map (dark purple →
# green → yellow), so the three mode panels never read as one another.
_MODE_CMAP_PAIR = "viridis"


def face_gram_determinants(cells, squared_length):
    """Gram determinant of every distinct triangle (2-face) and tetrahedron
    (3-face) of the given top cells, from the signed edge intervals alone via
    the polarization identity  G_ij = ½(ℓ²(v0,vi) + ℓ²(v0,vj) − ℓ²(vi,vj)).

    det G = 0 ⇔ the face is NULL — its span tangent to the light cone — the
    degenerate configurations of #632 where circumcentric dual volumes and
    deficit angles are singular and gradients blow up. |det G| is therefore a
    direct "distance from degeneracy" for each face; the per-frame minimum is
    the complex's closest approach to the null locus.

    `squared_length(u, v)` returns the edge's Re ℓ²; every vertex pair inside a
    top cell is an edge of the complex, so lookups never miss (the Gram
    diagonal's ℓ²(v,v) = 0 is supplied here, not looked up)."""
    def interval(u, v):
        return 0.0 if u == v else squared_length(u, v)
    dets = {2: {}, 3: {}}
    for cell in cells:
        ordered = sorted(cell)
        for k in (2, 3):
            for face in itertools.combinations(ordered, k + 1):
                if face in dets[k]:
                    continue
                v0, rest = face[0], face[1:]
                gram = np.array(
                    [[0.5 * (interval(v0, a) + interval(v0, b)
                             - interval(a, b))
                      for b in rest] for a in rest])
                dets[k][face] = float(np.linalg.det(gram))
    return {k: np.array(list(v.values())) for k, v in dets.items()}


def _min_abs_gram_dets(st):
    """(min |det G| over triangles, over tets) of the live complex — the
    null-face proximity scalars the animation traces per frame."""
    cells = [tuple(sorted(v.getId() for v in c.getVertices()))
             for c in st.getTopSimplices()]
    lengths = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        lengths[(min(a, b), max(a, b))] = (e.getLength() ** 2).real
    if not cells:
        return float("nan"), float("nan")
    dets = face_gram_determinants(
        cells, lambda u, v: lengths[(min(u, v), max(u, v))])
    return (float(np.abs(dets[2]).min()) if len(dets[2]) else float("nan"),
            float(np.abs(dets[3]).min()) if len(dets[3]) else float("nan"))


def _mds_layout(st):
    """2-D classical-MDS coordinates per vertex id, from graph shortest-path distances
    weighted by the relaxed edge lengths `sqrt(|Re ℓ²|)`, **normalized to unit RMS radius**.

    The normalization is the fix for the old "everything pulls into a dot" bug: the raw MDS
    scale tracks the absolute (conformal) edge lengths, which shrink under relaxation, so a
    grow-only view showed an ever-smaller cloud. Dividing out the RMS radius makes every
    frame the same size, so only the *shape* of the structure moves."""
    edges = st.getEdgeList().toVector()
    vids = sorted({v.getId() for e in edges
                   for v in (e.getSource(), e.getTarget())})
    if len(vids) < 2:
        return {v: np.zeros(2) for v in vids}
    idx = {v: i for i, v in enumerate(vids)}
    n = len(vids)
    W = np.full((n, n), np.inf)
    np.fill_diagonal(W, 0.0)
    for e in edges:
        a, b = idx[e.getSource().getId()], idx[e.getTarget().getId()]
        w = math.sqrt(max(abs((e.getLength() ** 2).real), 1e-6))
        W[a, b] = W[b, a] = min(W[a, b], w)
    D = shortest_path(W, method="D", directed=False)
    finite = np.isfinite(D)
    D[~finite] = D[finite].max() * 1.5 if finite.any() else 1.0   # disconnected
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:2]
    coords = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
    coords = coords - coords.mean(0)                                  # center
    rms = math.sqrt((coords ** 2).sum(1).mean()) or 1.0
    coords = coords / rms                                             # unit RMS radius
    return {vids[i]: coords[i] for i in range(n)}


class _StableLayout:
    """Per-panel jitter-free layout state: the normalized MDS embedding, rigidly aligned to
    the previous frame (rotation/reflection only — the scale is already fixed by
    `_mds_layout`'s RMS normalization) and eased toward it, with the view auto-fit to the
    current cloud (also eased, so it tracks the structure without jumping).

    Classical MDS is only defined up to rotation/reflection/scale and is globally sensitive,
    so a small change can reshuffle the whole cloud. Fixing the scale (normalization),
    removing the orientation ambiguity (Procrustes), easing positions, and auto-fitting the
    view together keep the structure steady and legible."""

    def __init__(self, ease=0.3, view_ease=0.25, pad=0.18):
        self._prev = None       # previous frame's eased positions, vid -> xy
        self._view = None       # eased view bbox [xlo, xhi, ylo, yhi]
        self.ease = ease
        self.view_ease = view_ease
        self.pad = pad

    def coords(self, st):
        coords = _mds_layout(st)
        if len(coords) < 2:
            self._prev = {v: np.asarray(p, float) for v, p in coords.items()}
            return self._prev
        if self._prev is None:                               # first frame defines the frame
            self._prev = {v: np.asarray(p, float) for v, p in coords.items()}
            return self._prev
        shared = [v for v in coords if v in self._prev]
        if len(shared) >= 2:                                 # Procrustes (no scale) onto prev
            cur = np.array([coords[v] for v in shared])
            ref = np.array([self._prev[v] for v in shared])
            cur_c, ref_c = cur.mean(0), ref.mean(0)
            cur0, ref0 = cur - cur_c, ref - ref_c
            u, _s, vt = np.linalg.svd(cur0.T @ ref0)
            rot = u @ vt                                     # rotation/reflection only
            aligned = {v: (np.asarray(p) - cur_c) @ rot + ref_c
                       for v, p in coords.items()}
        else:                                                # nothing shared: take raw
            aligned = {v: np.asarray(p, float) for v, p in coords.items()}
        eased = {}
        for v, target in aligned.items():
            prev = self._prev.get(v)
            eased[v] = target if prev is None else prev + self.ease * (target - prev)
        self._prev = eased
        return eased

    def view(self, coords):
        """Auto-fit (and ease) the view to the current cloud — never grow-only, so the
        structure can never shrink to an unreadable dot."""
        pts = np.array(list(coords.values()))
        lo, hi = pts.min(0), pts.max(0)
        pad = self.pad * max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
        box = [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad]
        if self._view is None:
            self._view = box
        else:
            self._view = [self._view[i] + self.view_ease * (box[i] - self._view[i])
                          for i in range(4)]
        return self._view

    def last_view(self):
        """The most recently computed view bbox, WITHOUT advancing the easing —
        for a second panel that shares the primal panel's frame (calling `view`
        again in the same frame would double-step the ease)."""
        return self._view


class ProtonAnimator:
    """Animates the one-step proton build: the single `direct_node` — three bare quarks
    in, the singlet out — growing and relaxing on screen.

    The node is driven with the combined `run` drive: an init pass
    (`grow_boundaries=True`, grows the color register) and an evolution pass
    (`grow_boundaries=False`), each interleaving the stage-1 surgery update with the
    stage-2 geometric relaxation every iteration, advanced one iteration per frame by
    default. (The panel grid is node-count-generic, inherited from the two-step
    animation this example was copied from.)"""

    _PHASE_NAMES = {"init": "growing register", "evolve": "evolving (∂W frozen)"}
    _TITLE_PREFIX = "Proton build (one-step, 3 quarks)"

    def __init__(self, nodes, degree=3, init_steps=_INIT_STEPS, init_chunk=_INIT_CHUNK,
                 evolve_steps=_EVOLVE_STEPS, evolve_chunk=_EVOLVE_CHUNK,
                 stage1_candidates=_STAGE1_CANDIDATES, stage2_beta=1.0,
                 max_lookahead_depth=_MAX_LOOKAHEAD_DEPTH,
                 max_lookahead_tries=_MAX_LOOKAHEAD_TRIES,
                 stage2_alpha0=0.05, stage2_rel_tol=10e-9):
        self._common_init(nodes, degree)
        self.s1c, self.s2_beta = stage1_candidates, stage2_beta
        self.lookahead_depth = max_lookahead_depth
        self.lookahead_tries = max_lookahead_tries
        self.s2_alpha0, self.s2_rel_tol = stage2_alpha0, stage2_rel_tol
        self._schedule = self._make_schedule(len(nodes), init_steps, init_chunk,
                                             evolve_steps, evolve_chunk)
        self._frames = len(self._schedule)

    def _common_init(self, nodes, degree):
        """Shared drawing/recording state: the node list, history buffers, per-panel
        layouts, and curvature cache — everything `_redraw`/`_draw_*`/`verdict` read."""
        self.nodes = nodes                  # [(MultiCobordism, label), ...] in order
        self.k = degree
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [],
                     "phase": [], "node": [], "lookahead": [], "tries": [],
                     "min_det2": [], "min_det3": [], "sigma": [],
                     "mode_w": [], "mode_w_head": [], "mode_cells": [],
                     "pair_count": [], "pair_w": [], "im_leak": []}
        self._boundaries = []       # step indices where a later node begins (trace markers)
        self._layouts = [_StableLayout() for _ in nodes]   # one per complex panel
        self._active = 0            # index of the node currently being driven
        self._done = False          # so the verdict is announced exactly once
        self._curv_cache = {}       # node_index -> (frame_computed, {cell_tuple: curvature})
        self._dump_dir = None       # set by run_build(dump_dir=...); None = no dumping

    @staticmethod
    def _make_schedule(n_nodes, init_steps, init_chunk, evolve_steps, evolve_chunk):
        """A flat list of (node_index, phase, count) ops, one per frame: each node's init
        pass (in `init_chunk`-sized bites), then its evolution pass (in `evolve_chunk`
        bites). Each op is one combined `run` call, so the geometric relaxation is
        interleaved into every iteration rather than scheduled as its own phase."""
        def chunks(total, size):
            done = 0
            while done < total:
                c = min(size, total - done)
                yield c
                done += c
        sched = []
        for i in range(n_nodes):
            sched += [(i, "init", c) for c in chunks(init_steps, init_chunk)]
            sched += [(i, "evolve", c) for c in chunks(evolve_steps, evolve_chunk)]
        return sched

    # ---- one scheduled chunk on the active node ----
    def _advance(self, frame):
        node_index, phase, count = self._schedule[frame]
        if node_index != self._active:                       # a new node begins
            self._active = node_index
            self._boundaries.append(len(self.hist["F"]))
        node, _label = self.nodes[node_index]
        # Redraw fresh candidates while the frame commits nothing: a stall is a
        # random-draw miss, not proof that no improving move exists, so give the
        # frame `lookahead_tries` draws before advancing. The first try that
        # commits anything (last_stage1_lookahead > 0) ends the loop, so the
        # default of 1 try is exactly the historical single-draw behaviour and
        # the retries cost nothing on frames that succeed immediately.
        tries = 0
        for tries in range(1, max(self.lookahead_tries, 1) + 1):
            node.run(max_iters=count, n_candidate_moves=self.s1c,
                     grow_boundaries=(phase == "init"), beta=self.s2_beta,
                     alpha0=self.s2_alpha0, rel_tol=self.s2_rel_tol,
                     max_lookahead=self.lookahead_depth)
            if int(node.last_stage1_lookahead) > 0:
                break
        self._record(node, node_index, phase, tries)

    def _record(self, node, node_index, phase, tries=1):
        st = node.st
        self.hist["F"].append(float(node.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.regge_action_gradient(st)))
        self.hist["rU"].append(float(node.r_u(st)))
        self.hist["b3"].append(int(cob.MultiCobordism.betti(st)[self.k]))
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["phase"].append(phase)
        self.hist["node"].append(node_index)
        # Lookahead depth of the frame's committed stage-1 sequence: 1 = ordinary
        # single move, >1 = the search had to look several moves ahead, 0 = stage-1
        # stall (nothing committed at any depth this frame).
        self.hist["lookahead"].append(int(node.last_stage1_lookahead))
        # How many candidate draws this frame needed; > 1 means earlier draws
        # stalled and were retried (see `_advance`).
        self.hist["tries"].append(int(tries))
        # The register operator's full singular spectrum, sorted DESCENDING —
        # the same metric L_k the near-kernel residual reads. Singular values
        # rather than eigenvalues: the signed operator is non-normal, and the
        # sigma are the honest dimension count of its near-kernel (eigenvalue
        # magnitudes double-count at defective points). numpy's svd returns
        # them descending already.
        cc = cob.ChainComplex.fromSpacetime(st)
        nk_cells = cc.numSimplices(self.k)
        if nk_cells > 0:
            L = np.array(cob.HodgeLaplacian(st).laplacian(self.k),
                         dtype=complex).reshape(nk_cells, nk_cells)
            _u, sigma, vh = np.linalg.svd(L)
            self.hist["sigma"].append(sigma)
            # Mode LOCALIZATION of that tail: each right singular vector is a
            # k-cochain — one component per k-cell, in `kSimplexVertices(k)`
            # order — so |ψᵢ|², summed over the m smallest-σ modes and
            # normalized to total 1, is the per-cell share of the
            # almost-register. Rows of `vh` pair with `sigma` descending, so
            # the tail is the LAST m rows (the conjugate in vᵢ = conj(vh[i])
            # drops out under |·|²).
            m = 3
            if hasattr(node, "expectedRegisterCount"):
                m = int(node.expectedRegisterCount())
            # ... and of the HEAD: the m largest-σ modes (the FIRST m rows)
            # localize on cells whose content is collapsing (σ_max ∝ 1/det G),
            # i.e. material approaching tangency with the light cone.
            for rows, key in ((vh[max(0, nk_cells - m):], "mode_w"),
                              (vh[:m], "mode_w_head")):
                w = (np.abs(rows) ** 2).sum(axis=0)
                total = float(w.sum())
                self.hist[key].append(w / total if total > 0 else w)
            self.hist["mode_cells"].append(
                [tuple(c) for c in cc.kSimplexVertices(self.k)])
            # Annihilation content: the Krein classification of the SAME L_k's
            # eigenvalue spectrum — broken conjugate pairs (W-null) and their
            # per-cell |ψ|². KreinModes re-derives L (an eig beside the svd);
            # both are O(n³) on a small n, cheap next to the engine frame. The
            # classification exists only ON the real-ℓ² locus; live stage-2
            # exploration leaves it (complex intervals), so off-locus frames
            # record nan/empty plus the interval leak — the trace shows the
            # build leave and re-touch the locus, and pair counts appear
            # exactly when the structure exists.
            krein = KreinModes(st, self.k)
            self.hist["im_leak"].append(float(krein.imag_interval_leak))
            if krein.on_locus:
                self.hist["pair_count"].append(float(krein.pair_count))
                self.hist["pair_w"].append(krein.pair_heat())
            else:
                self.hist["pair_count"].append(float("nan"))
                self.hist["pair_w"].append(np.array([]))
                self._pair_note = f"off the real-ℓ² locus: {krein.reason}"
        else:
            self.hist["sigma"].append(np.array([]))
            self.hist["mode_w"].append(np.array([]))
            self.hist["mode_w_head"].append(np.array([]))
            self.hist["pair_count"].append(0.0)
            self.hist["pair_w"].append(np.array([]))
            self.hist["im_leak"].append(0.0)
            self.hist["mode_cells"].append([])
        # Null-face proximity: the smallest |det G| over the complex's triangles
        # and tets — 0 = a face exactly tangent to the light cone (degenerate).
        min_det2, min_det3 = _min_abs_gram_dets(st)
        self.hist["min_det2"].append(min_det2)
        self.hist["min_det3"].append(min_det3)

    # ---- convergence ----
    def verdict(self):
        """The honest, live convergence verdict read off the node's *current* whole cobordism:
        the singlet `r_state` and the emergent color-hole count, and whether both clear the
        proton thresholds (residual < tol, holes ≥ 3)."""
        st = self.nodes[-1][0].st
        res = float(cob.MultiCobordism.r_state(st, self.k, cob.Proton.singlet()))
        holes = len(cob.MultiCobordism.emergent_holes(st, self.k))
        return res < _COLOR_TOL and holes >= _MIN_QUARK_HOLES, res, holes

    # ---- drawing ----
    def _setup(self, plt):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        # One node per row: [traces | primal complex | spatial-curvature dual | temporal-
        # curvature dual]. The two dual panels split the COMPLEX Lorentzian deficit: the
        # spatial one shows its real part (Re ε, the rotation angle-defect, from timelike
        # hinges), the temporal one its imaginary part (Im ε, the boost/light-cone content,
        # from spacelike hinges — those whose normal plane is timelike). The grid is
        # node-count-generic (the joint single-node drive gets one panel row; the two-step
        # keeps its two) with the metrics/register traces always on rows 0/1 of column 0.
        n_nodes = len(self.nodes)
        # Single-node runs get a THIRD row so the two mode-localization panels
        # stack in one column (near-kernel tail above, near-null head below);
        # multi-node grids use every row for complexes and skip the extras.
        n_rows = 3 if n_nodes == 1 else max(2, n_nodes)
        self.fig, axes = plt.subplots(n_rows, 4, figsize=(21, 4.5 * n_rows),
                                      squeeze=False)
        self.axm = axes[0][0]                                # metrics trace
        self.axr = axes[1][0]                                # register trace
        for row in range(2, n_rows):
            axes[row][0].axis("off")
        self._primal_axes = [axes[i][1] for i in range(n_nodes)]
        self._re_axes = [axes[i][2] for i in range(n_nodes)]
        self._im_axes = [axes[i][3] for i in range(n_nodes)]
        # A single-node run leaves rows 1-2's panels free: claim row 1 for the
        # null-face proximity trace, the rolling singular-value spectrum, and
        # the near-kernel mode localization; row 2 for the broken-pair count
        # trace, the annihilation heat, and — directly below the near-kernel
        # one — the near-null (head-of-spectrum) mode localization; blank the
        # rest. (Multi-node grids keep every panel for complexes, so the
        # extras are skipped there.)
        self.ax_null = axes[n_nodes][1] if n_nodes < n_rows else None
        self.ax_spec = axes[n_nodes][2] if n_nodes < n_rows else None
        self.ax_mode = axes[n_nodes][3] if n_nodes < n_rows else None
        self.ax_pair_trace = axes[2][1] if n_nodes == 1 else None
        self.ax_pair = axes[2][2] if n_nodes == 1 else None
        self.ax_mode_head = axes[2][3] if n_nodes == 1 else None
        # The leak shares the pair-count trace panel on a twin y-axis, created
        # ONCE (a per-frame twinx would pile up axes like per-frame colorbars).
        self.ax_pair_leak = (self.ax_pair_trace.twinx()
                             if self.ax_pair_trace is not None else None)
        extras = (self.ax_null, self.ax_spec, self.ax_mode,
                  self.ax_pair_trace, self.ax_pair, self.ax_mode_head)
        for row in range(n_nodes, n_rows):
            for column in (1, 2, 3):
                if any(ax is not None and axes[row][column] is ax
                       for ax in extras):
                    continue
                axes[row][column].axis("off")
        # Persistent colorbars (created ONCE — recreating per frame piles them up). Each dual
        # panel self-normalizes per frame; we just update the mappable's clim. Re (spatial) and
        # Im (temporal) use distinct diverging colormaps so the two channels read apart.
        self._re_sms, self._im_sms = [], []
        for axset, sms, cmap, label in (
                (self._re_axes, self._re_sms, _HEAT_CMAP, "spatial curvature  Re ε·|★|"),
                (self._im_axes, self._im_sms, _HEAT_CMAP_IM, "temporal curvature  Im ε·|★|")):
            for ax in axset:
                sm = ScalarMappable(cmap=cmap, norm=Normalize(-1.0, 1.0))
                cbar = self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=7)
                cbar.ax.tick_params(labelsize=6)
                sms.append(sm)
        # The mode-localization panels' colorbars (also persistent). Sequential,
        # floor pinned at 0: the weights are non-negative shares, so only the
        # top of the range renormalizes per frame.
        self._mode_sm = self._mode_head_sm = self._pair_sm = None
        for ax, attr, cmap, label in (
                (self.ax_mode, "_mode_sm", _MODE_CMAP,
                 "near-kernel mode weight  Σₘ|ψ|² per edge"),
                (self.ax_mode_head, "_mode_head_sm", _MODE_CMAP_HEAD,
                 "near-null mode weight  Σₘ|ψ|² per edge"),
                (self.ax_pair, "_pair_sm", _MODE_CMAP_PAIR,
                 "annihilation heat  Σ_pairs|ψ|² per edge")):
            if ax is not None:
                sm = ScalarMappable(cmap=cmap, norm=Normalize(0.0, 1.0))
                setattr(self, attr, sm)
                cbar = self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=7)
                cbar.ax.tick_params(labelsize=6)
        return self.fig

    def _draw_complex(self, ax, node_index, coords, title):
        node, _label = self.nodes[node_index]
        st = node.st
        ax.clear()
        # Each emergent color hole (register) is a removed top cell — a k=3 hole is a
        # 4-simplex, 5 vertices — whose boundary edges stay in the complex. OUTLINE each
        # register cell by reddening the edges whose endpoints both lie in that hole's vertex
        # set, and badge it with its index at the cell centroid. So one register reads as one
        # numbered red cell — unlike the old per-vertex reddening, where a single 5-vertex
        # hole showed as 5 disconnected red dots and couldn't be counted.
        holes = cob.MultiCobordism.emergent_holes(st, self.k)
        hole_vsets = [set(h) for h in holes]
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a not in coords or b not in coords:
                continue
            p, q = coords[a], coords[b]
            if any(a in vs and b in vs for vs in hole_vsets):     # a register-cell edge
                ax.plot([p[0], q[0]], [p[1], q[1]], color="C3", lw=1.8, zorder=3)
            else:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=0.5, zorder=1)
        if coords:
            pts = np.array(list(coords.values()))
            ax.scatter(pts[:, 0], pts[:, 1], c="0.4", s=8, zorder=2)
            for i, h in enumerate(holes):                          # number each register
                hp = np.array([coords[v] for v in h if v in coords])
                if len(hp):
                    c = hp.mean(0)
                    ax.text(c[0], c[1], str(i + 1), color="white", fontsize=8,
                            fontweight="bold", ha="center", va="center", zorder=4,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="C3", ec="white",
                                      lw=0.8))
            if len(coords) >= 2:
                view = self._layouts[node_index].view(coords)
                ax.set_xlim(view[0], view[1])
                ax.set_ylim(view[2], view[3])
        n_holes = len(holes)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  —  {n_holes} register{'s' if n_holes != 1 else ''}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- dual complex + curvature heat ----
    @staticmethod
    def _cell_curvature(st):
        """Per-top-cell curvature, BOTH channels of the COMPLEX Lorentzian deficit, from the one
        `deficitAngle` per hinge: `Re(deficit)·|★|` — the spatial angle-defect
        (rotation) curvature, carried by timelike hinges — and `Im(deficit)·|★|` — the temporal
        boost / light-cone content, carried by spacelike hinges (those whose normal plane is
        timelike). Both SIGNED (ε<0 = saddle; Im sign = boost direction). Returns
        {cell-tuple: (re_sum, im_sum)}."""
        hinge_re, hinge_im = {}, {}
        for s in st.getSimplices():
            vs = s.getVertices()
            if len(vs) != 3:                     # hinges = (d-2) = 2-simplices (triangles)
                continue
            key = tuple(sorted(v.getId() for v in vs))
            try:
                deficit = complex(s.deficitAngle())
                # complex-tolerant positive dual-measure weight (dualVolume
                # is real today; abs(complex(...)) survives it going complex)
                weight = abs(complex(s.dualVolume()))
                hinge_re[key] = deficit.real * weight
                hinge_im[key] = deficit.imag * weight
            except RuntimeError:                 # boundary/degenerate hinge → no curvature
                # Only the geometric failure is swallowed; a type/contract
                # failure (TypeError, ValueError) must propagate, never
                # render as zero curvature.
                hinge_re[key] = hinge_im[key] = 0.0
        curv = {}
        for c in st.getTopSimplices():
            cell = tuple(sorted(v.getId() for v in c.getVertices()))
            tris = [tuple(sorted(t)) for t in itertools.combinations(cell, 3)]
            curv[cell] = (sum(hinge_re.get(t, 0.0) for t in tris),
                          sum(hinge_im.get(t, 0.0) for t in tris))
        return curv

    def _cell_curvature_cached(self, node_index, st):
        """`_cell_curvature` is expensive, so recompute it only every `_HEAT_REFRESH_EVERY`
        frames on the active (changing) node, and always on the final frame; the frozen
        node's geometry doesn't change, so its last value is reused."""
        frame = len(self.hist["F"])
        cached = self._curv_cache.get(node_index)
        stale = (node_index == self._active
                 and frame - cached[0] >= _HEAT_REFRESH_EVERY) if cached else True
        if cached is None or stale or frame >= self._frames:
            self._curv_cache[node_index] = (frame, self._cell_curvature(st))
        return self._curv_cache[node_index][1]

    def _dump_frame(self, frame, coords_by_node):
        """Write this frame's dual-curvature panels as data, so a claim about the
        picture (e.g. "Re ε and Im ε run perpendicular around frame 200") can be
        checked numerically instead of by eye — reaching frame ~200 takes hours,
        so the frame must be recoverable without re-running.

        Frames are numbered as the window title numbers them (1-based "frame
        N/total"), so the file for the panel on screen is `frame_<N>.json`.

        Costs nothing extra: the curvature comes from `_cell_curvature_cached`,
        the same value the panels just drew, and `heat_frame` records the frame it
        was actually computed on (the cache refreshes every `_HEAT_REFRESH_EVERY`
        frames, so consecutive dumps can share one heat field).

        The complex is written in the schema `observables.LiveComplex.load`
        consumes — identical to `laplacian_clusters.dump_state` — so the existing
        rehydration path works unchanged:

            d = json.load(open("frame_0213.json"))
            n = d["nodes"][0]
            st = tessera.observables.LiveComplex.load(
                n["cells"],
                {(u, v): complex(re, im) for u, v, re, im in n["squared_lengths"]},
                {int(v): t for v, t in n["vertex_times"].items()}, 4)
        """
        payload = {"frame": frame, "dimensions": 4, "nodes": []}
        for key in ("F", "gradN2", "rU", "b3", "holes", "phase", "node",
                    "lookahead", "tries", "min_det2", "min_det3"):
            series = self.hist.get(key) or []
            payload[key] = series[-1] if series else None
        for ni, (cobordism, label) in enumerate(self.nodes):
            st = cobordism.st
            coords = coords_by_node.get(ni, {})
            heat_frame, curv_map = self._curv_cache.get(ni, (None, {}))
            cells, re_heat, im_heat, dual_pos = [], [], [], []
            for c in st.getTopSimplices():
                cell = sorted(v.getId() for v in c.getVertices())
                cells.append(cell)
                re_c, im_c = curv_map.get(tuple(cell), (0.0, 0.0))
                re_heat.append(float(re_c))
                im_heat.append(float(im_c))
                here = [coords[v] for v in cell if v in coords]
                dual_pos.append([float(np.mean([p[0] for p in here])),
                                 float(np.mean([p[1] for p in here]))]
                                if here else [None, None])
            rows, cols, _n = st.getDualAdjacency()
            payload["nodes"].append({
                "label": label,
                "active": ni == self._active,
                "cells": cells,
                "squared_lengths": [
                    [min(e.getSource().getId(), e.getTarget().getId()),
                     max(e.getSource().getId(), e.getTarget().getId()),
                     (e.getLength() ** 2).real, (e.getLength() ** 2).imag]
                    for e in st.getEdgeList().toVector()],
                "vertex_times": {str(v.getId()): float(v.getTime())
                                 for v in st.getVertexList().toVector()},
                # the two panels, as data: index i of each array is the dual node
                # drawn at dual_positions[i], i.e. getTopSimplices()[i] / cells[i]
                "re_heat": re_heat,       # spatial curvature  Re(ε)·|★|
                "im_heat": im_heat,       # temporal curvature Im(ε)·|★|
                "dual_positions": dual_pos,
                "dual_adjacency": [list(map(int, rows)), list(map(int, cols))],
                "heat_frame": heat_frame,  # frame the heat was computed on
            })
        path = os.path.join(self._dump_dir, f"frame_{frame:04d}.json")
        with open(path, "w") as f:
            json.dump(payload, f)

    def _draw_dual(self, ax, sm, node_index, coords, channel, cmap, title):
        """One dual-complex curvature panel (nodes = top cells at their primal centroids, edges
        = shared-facet adjacency), heat-colored by `channel` of the signed per-cell curvature:
        0 = spatial (Re ε, angle-defect), 1 = temporal (Im ε, boost/rapidity). Symmetric
        diverging range centered at 0."""
        st = self.nodes[node_index][0].st
        ax.clear()
        top = st.getTopSimplices()
        curv_map = self._cell_curvature_cached(node_index, st)
        n = len(top)
        pos = np.full((n, 2), np.nan)
        curv = np.zeros(n)
        for i, c in enumerate(top):                       # dual node i ↔ getTopSimplices()[i]
            cell = sorted(v.getId() for v in c.getVertices())
            here = [coords[v] for v in cell if v in coords]
            if here:
                pos[i] = np.mean(here, axis=0)
            curv[i] = curv_map.get(tuple(cell), (0.0, 0.0))[channel]
        rows, cols, _N = st.getDualAdjacency()
        for a, b in zip(rows, cols):                      # dual edges (shared-facet adjacency)
            if a < n and b < n and np.all(np.isfinite(pos[a])) and np.all(np.isfinite(pos[b])):
                ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                        color="0.8", lw=0.4, zorder=1)
        finite = np.all(np.isfinite(pos), axis=1)
        if finite.any():
            cv = curv[finite]
            mag = np.abs(cv)
            vmax = float(np.percentile(mag, 95)) if finite.sum() >= 5 else float(mag.max())
            if not vmax > 0:
                vmax = 1.0
            sm.set_clim(-vmax, vmax)
            shown = np.clip(cv, -vmax, vmax)
            if finite.sum() >= 4:                         # filled heat field where possible
                try:
                    ax.tricontourf(pos[finite, 0], pos[finite, 1], shown, levels=12,
                                   cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.85, zorder=0)
                except Exception:
                    pass
            ax.scatter(pos[finite, 0], pos[finite, 1], c=shown, cmap=cmap,
                       vmin=-vmax, vmax=vmax, s=14, zorder=2, edgecolors="0.3", linewidths=0.2)
            if mag.max() <= 1e-9:                         # channel is identically zero
                # The temporal (Im) channel is ≡0 whenever the geometry is all-spacelike
                # (no timelike hinges → no boost content), so the panel would read as blank;
                # say why instead of showing an empty box. Only the Im channel gets the
                # all-spacelike explanation — and only because compute failures now
                # PROPAGATE out of _cell_curvature (a type failure can no longer zero the
                # channel and masquerade as all-spacelike, #581).
                label = ("≡ 0\n(all-spacelike: no timelike hinges)"
                         if channel == 1 else "≡ 0")
                ax.text(0.5, 0.5, label,
                        transform=ax.transAxes, ha="center", va="center", fontsize=9,
                        color="0.45")
        ax.set_aspect("equal")
        ax.set_title(f"{title}  ({n} cells)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    def _draw_mode_heat(self, ax, node_index, coords, weights, sm, cmap,
                        which=None, title=None, empty_note="no k-cells yet"):
        """One mode-localization panel: WHICH portion of the complex carries the
        given end of the spectrum. `weights` is the per-k-cell share `Σₘ|ψᵢ|²`
        recorded by `_record` (right singular vectors of the metric `L_k` are
        k-cochains, in `mode_cells` order); it is spread from each k-cell onto
        its vertex-pair edges and painted on the SAME stable layout as the
        primal panel. `which` = (name, end) picks the wording: ("near-kernel",
        "smallest") paints the tail — for a genuine (open) register the weight
        sits exactly on the hole-boundary cells, for a causally-tuned
        near-kernel (no hole) it shows where the almost-register is forming —
        and ("near-null", "largest") paints the head, which localizes on cells
        whose content is collapsing (σ_max ∝ 1/det G), i.e. material
        approaching tangency with the light cone. The register hole outlines
        are overlaid (dashed) so on-hole vs off-hole localization reads
        directly. The title's PR is the participation ratio `1/Σᵢwᵢ²` of the
        normalized per-cell weights — the effective number of k-cells the
        modes live on. A caller whose weights are not a σ-end selection (the
        annihilation heat) passes a free-form `title` instead of `which`, and
        `empty_note` says why an empty panel is empty."""
        from matplotlib.collections import LineCollection
        node, _label = self.nodes[node_index]
        st = node.st
        ax.clear()
        cells = self.hist["mode_cells"][-1]
        if title is None:
            name, end = which
            m = 3
            if hasattr(node, "expectedRegisterCount"):
                m = int(node.expectedRegisterCount())
            title = f"{name} |ψ|² — {m} {end}-σ modes"
        if len(weights) == 0 or not coords:
            ax.text(0.5, 0.5, empty_note, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.45")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            return
        edge_w = {}
        for cell, w in zip(cells, weights):
            for a, b in itertools.combinations(sorted(cell), 2):
                edge_w[(a, b)] = edge_w.get((a, b), 0.0) + float(w)
        segments, values, hole_segments = [], [], []
        holes = cob.MultiCobordism.emergent_holes(st, self.k)
        hole_vsets = [set(h) for h in holes]
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a not in coords or b not in coords:
                continue
            segments.append([coords[a], coords[b]])
            values.append(edge_w.get((min(a, b), max(a, b)), 0.0))
            if any(a in vs and b in vs for vs in hole_vsets):
                hole_segments.append([coords[a], coords[b]])
        values = np.array(values)
        vmax = float(values.max()) if values.size and values.max() > 0 else 1.0
        if sm is not None:
            sm.set_clim(0.0, vmax)
        lines = LineCollection(segments, cmap=cmap, linewidths=1.6, zorder=2)
        lines.set_array(values)
        lines.set_clim(0.0, vmax)
        ax.add_collection(lines)
        if hole_segments:                        # register outlines, dashed overlay
            ax.add_collection(LineCollection(hole_segments, colors="C3",
                                             linewidths=0.9, linestyles="--",
                                             alpha=0.9, zorder=3))
        pts = np.array(list(coords.values()))
        ax.scatter(pts[:, 0], pts[:, 1], c="0.55", s=5, zorder=1)
        view = self._layouts[node_index].last_view()
        if view is not None:
            ax.set_xlim(view[0], view[1])
            ax.set_ylim(view[2], view[3])
        participation = 1.0 / float((weights ** 2).sum())   # weights sum to 1
        ax.set_aspect("equal")
        ax.set_title(f"{title}, PR {participation:.1f}/{len(weights)} cells",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    def _redraw(self):
        xs = range(len(self.hist["F"]))
        self.axm.clear()
        self.axm.plot(xs, self.hist["F"], label="F (objective)", color="C0")
        self.axm.plot(xs, self.hist["gradN2"], label="‖∇S‖²", color="C1")
        self.axm.plot(xs, self.hist["rU"], label="r_U", color="C2")
        for b in self._boundaries:
            self.axm.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
        # LOOKAHEAD INDICATOR: ring the F trace where the committed stage-1 sequence
        # needed more than one move of lookahead (numbered with its depth), and mark
        # a grey x where stage 1 stalled outright (no F-lowering sequence found).
        lookahead = self.hist["lookahead"]
        deep = [i for i, d in enumerate(lookahead) if d > 1]
        if deep:
            self.axm.scatter(deep, [self.hist["F"][i] for i in deep], marker="o",
                             s=48, facecolors="none", edgecolors="C3", linewidths=1.4,
                             zorder=5, label="multi-move sequence (lookahead > 1)")
            for i in deep:
                self.axm.annotate(str(lookahead[i]), (i, self.hist["F"][i]),
                                  textcoords="offset points", xytext=(0, 7),
                                  ha="center", fontsize=7, color="C3")
        stalled = [i for i, d in enumerate(lookahead) if d == 0]
        if stalled:
            self.axm.scatter(stalled, [self.hist["F"][i] for i in stalled],
                             marker="x", s=22, color="0.55", zorder=5,
                             label="stage-1 stalled (no sequence found)")
        self.axm.set_yscale("symlog")
        self.axm.set_title("metrics")
        self.axm.set_xlabel("frame")
        self.axm.legend(loc="upper right", fontsize=8)

        self.axr.clear()
        # The register count is the number of emergent color holes (= quarks) — the SAME
        # number the complex panels outline and the titles report. b_k is a Betti number, a
        # *different* topological invariant that can disagree, so it is drawn separately and
        # labelled as such, never as "the register" (which conflated the two before).
        self.axr.plot(xs, self.hist["holes"], label="color registers (holes)", color="C3",
                      marker=".")
        self.axr.plot(xs, self.hist["b3"], label=f"b{self.k} (Betti number)", color="C4",
                      marker=".", alpha=0.55)
        for b in self._boundaries:
            self.axr.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
        self.axr.axhline(_MIN_QUARK_HOLES, color="0.6", ls=":", lw=0.8,
                         label=f"proton = {_MIN_QUARK_HOLES}")
        self.axr.set_title("color register")
        self.axr.set_xlabel("frame")
        self.axr.legend(loc="upper left", fontsize=8)

        # One node per row: primal complex, then its dual split into spatial-curvature (Re ε)
        # and temporal-curvature (Im ε) panels. The active node animates; any other holds its
        # current complex — every node on screen at one time. Each node's layout is computed
        # once and shared by its primal + both dual panels.
        coords_by_node = {}
        for ni in range(len(self.nodes)):
            coords = self._layouts[ni].coords(self.nodes[ni][0].st)
            coords_by_node[ni] = coords
            self._draw_complex(self._primal_axes[ni], ni, coords, self.nodes[ni][1])
            self._draw_dual(self._re_axes[ni], self._re_sms[ni], ni, coords,
                            0, _HEAT_CMAP, "dual — spatial curvature (Re ε)")
            self._draw_dual(self._im_axes[ni], self._im_sms[ni], ni, coords,
                            1, _HEAT_CMAP_IM, "dual — temporal curvature (Im ε)")
        # Dumped AFTER the panels draw, so the cached curvature is exactly what
        # was rendered. A dump failure must not kill a multi-hour run.
        if self._dump_dir:
            try:
                # numbered like the window title ("frame N/total"), so the file
                # for the panel you are looking at is frame_<N>.json
                self._dump_frame(len(self.hist["F"]), coords_by_node)
            except Exception as exc:
                print(f"\nframe dump failed: {exc!r}", flush=True)

        if getattr(self, "ax_null", None) is not None:
            self.ax_null.clear()
            xs = range(len(self.hist["min_det2"]))
            self.ax_null.semilogy(xs, self.hist["min_det2"], color="C0",
                                  marker=".", label="min |det G| — triangles")
            self.ax_null.semilogy(xs, self.hist["min_det3"], color="C3",
                                  marker=".", label="min |det G| — tets")
            self.ax_null.axhline(1e-6, color="0.6", ls=":", lw=0.8,
                                 label="1e-6 (danger: near-degenerate)")
            self.ax_null.set_title("null-face proximity (det G = 0 ⇔ face "
                                   "tangent to the light cone)", fontsize=9)
            self.ax_null.set_xlabel("frame")
            self.ax_null.set_ylabel("min |det G|")
        if getattr(self, "ax_spec", None) is not None and self.hist["sigma"]:
            ax = self.ax_spec
            ax.clear()
            sigma = self.hist["sigma"][-1]
            m = 3
            node = self.nodes[self.hist["node"][-1]][0]
            if hasattr(node, "expectedRegisterCount"):
                m = int(node.expectedRegisterCount())
            if sigma.size:
                # Log axis cannot show exact zeros (an OPEN register's sigma is
                # exactly 0), so display-floor them and mark the floor; the bar
                # sitting ON the floor line is the "this mode is kernel" read.
                floor = max(1e-12, float(sigma[sigma > 0].min()) * 1e-3
                            if (sigma > 0).any() else 1e-12)
                shown = np.maximum(sigma, floor)
                ranks = np.arange(1, sigma.size + 1)
                # The m smallest are the near-kernel tail the objective watches.
                colors = ["C3" if i >= sigma.size - m else "C0" for i in range(sigma.size)]
                ax.bar(ranks, shown, width=0.9, color=colors, zorder=3)
                # Rolling ghost: the last few frames' spectra as fading steps, so
                # the spectrum's drift toward the kernel is visible in one look.
                for age, past in enumerate(reversed(self.hist["sigma"][-6:-1]), 1):
                    if past.size:
                        ax.step(np.arange(1, past.size + 1),
                                np.maximum(past, floor), where="mid",
                                color="0.4", alpha=max(0.05, 0.35 - 0.06 * age),
                                lw=1.0, zorder=2)
                ax.axhline(floor, color="0.6", ls=":", lw=0.8)
                ax.set_yscale("log")
                ax.set_xlim(0.5, sigma.size + 0.5)
            ax.set_title(f"σ(L{self.k}) descending — red: {m}-smallest "
                         "(near-kernel tail)", fontsize=9)
            ax.set_xlabel("rank (descending)")
            ax.set_ylabel("σ")
            self.ax_null.legend(loc="lower right", fontsize=7)
        if getattr(self, "ax_mode", None) is not None and self.hist["mode_w"]:
            active = self.hist["node"][-1]
            self._draw_mode_heat(self.ax_mode, active,
                                 coords_by_node.get(active, {}),
                                 self.hist["mode_w"][-1], self._mode_sm,
                                 _MODE_CMAP, ("near-kernel", "smallest"))
        if (getattr(self, "ax_mode_head", None) is not None
                and self.hist["mode_w_head"]):
            active = self.hist["node"][-1]
            self._draw_mode_heat(self.ax_mode_head, active,
                                 coords_by_node.get(active, {}),
                                 self.hist["mode_w_head"][-1], self._mode_head_sm,
                                 _MODE_CMAP_HEAD, ("near-null", "largest"))
        if getattr(self, "ax_pair", None) is not None and self.hist["pair_w"]:
            active = self.hist["node"][-1]
            n_pairs = self.hist["pair_count"][-1]
            self._draw_mode_heat(
                self.ax_pair, active, coords_by_node.get(active, {}),
                self.hist["pair_w"][-1], self._pair_sm, _MODE_CMAP_PAIR,
                title=("annihilation heat — off the real-ℓ² locus"
                       if math.isnan(n_pairs) else
                       f"annihilation heat — {int(n_pairs)} broken pairs (W-null)"),
                empty_note=getattr(self, "_pair_note",
                                   "no Krein classification yet"))
        if (getattr(self, "ax_pair_trace", None) is not None
                and self.hist["pair_count"]):
            ax = self.ax_pair_trace
            ax.clear()
            xs = range(len(self.hist["pair_count"]))
            ax.plot(xs, self.hist["pair_count"], color="C4", marker=".")
            for b in self._boundaries:
                ax.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
            ax.set_title("broken pairs (gaps = off-locus) + real-ℓ² locus "
                         "distance", fontsize=9)
            ax.set_xlabel("frame")
            ax.set_ylabel("pairs", color="C4")
            if getattr(self, "ax_pair_leak", None) is not None:
                axl = self.ax_pair_leak
                axl.clear()
                axl.plot(xs, self.hist["im_leak"], color="C2", alpha=0.7,
                         lw=1.0)
                axl.set_ylabel("max|Im ℓ²| (locus distance)", color="C2",
                               fontsize=8)
                axl.tick_params(labelsize=6)

    # ---- per-frame text hooks ----
    def _frame_label(self, frame):
        """The short 'what's running' label for a frame — the node label plus the
        current phase name."""
        node_index, phase, _count = self._schedule[frame]
        return f"{self.nodes[node_index][1]} · {self._PHASE_NAMES[phase]}"

    def _verdict_tag(self, ok, res, holes):
        return (f"CONVERGED ✓ — proton {{1,ω,ω²}} carried (r_state={res:.2g}, "
                f"{holes} registers)" if ok else
                f"did NOT converge (r_state={res:.2g}, {holes} registers)")

    def _draw_extras(self):
        """Hook for per-frame figure annotations drawn after `_redraw`; none by default."""

    # ---- the three parts of a frame: announce (pre-compute) · advance (compute) · paint ----
    def _announce(self, frame):
        """Pre-compute heartbeat: set the title to what's about to run and flush an stdout line.
        In `--live` this runs on the GUI thread the instant the worker *starts* a frame, so the
        window immediately shows e.g. 'growing register' for the whole (responsive) compute."""
        label = self._frame_label(frame)
        self.fig.suptitle(
            f"{self._TITLE_PREFIX} — frame {frame + 1}/{self._frames} · {label}")
        print(f"\rframe {frame + 1}/{self._frames} ({label})", end="", flush=True)

    def _paint(self, frame):
        """Redraw the whole figure from the current geometry (GUI thread). On the last frame,
        also read and announce the live convergence verdict. Never touches the engine — safe to
        call while the compute worker is parked waiting for this paint to finish."""
        self._redraw()
        self._draw_extras()
        # Post-compute lookahead tag: the announce title says what's about to run;
        # this rewrites it with what the frame actually did whenever that is
        # noteworthy — a multi-move sequence or a stage-1 stall.
        if not self._done and self.hist["lookahead"]:
            depth = self.hist["lookahead"][-1]
            used = self.hist["tries"][-1] if self.hist["tries"] else 1
            retried = f", {used} tries" if used > 1 else ""
            tag = (f"  ·  LOOKAHEAD: committed a {depth}-move sequence{retried}"
                   if depth > 1
                   else (f"  ·  stage-1 stalled (searched depths 1–"
                         f"{self.lookahead_depth} on {used} "
                         f"{'try' if used == 1 else 'tries'}; nothing F-lowering)")
                   if depth == 0
                   else (f"  ·  committed after {used} tries" if used > 1 else ""))
            if tag:
                label = self._frame_label(frame)
                self.fig.suptitle(f"{self._TITLE_PREFIX} — frame {frame + 1}/"
                                  f"{self._frames} · {label}{tag}")
                print(f"\rframe {frame + 1}/{self._frames} ({label}){tag}",
                      end="", flush=True)
        if frame >= self._frames - 1 and not self._done:   # last frame: announce the verdict
            self._done = True
            ok, res, holes = self.verdict()
            tag = self._verdict_tag(ok, res, holes)
            self.fig.suptitle(f"{self._TITLE_PREFIX} — {tag}")
            print(f"\rframe {frame + 1}/{self._frames} "
                  f"({self._frame_label(frame)}) — {tag}")

    def update(self, frame):
        """One synchronous frame: announce → compute → paint. Used by the `--save` (off-screen
        Agg) path, where blocking the render loop on the compute is harmless. `--live` instead
        drives compute off the GUI thread via `_run_live` so the window stays responsive."""
        if not self._done:
            self._announce(frame)
        self._advance(frame)
        self._paint(frame)
        return []

    # ---- responsive live driver: compute on a worker thread, paint on the GUI thread ----
    def _run_live(self, plt, interval):
        """Animate live without freezing the window. A build frame — one combined `run`
        iteration: a candidate-move batch plus a Hessian-backed relaxation step — can still
        take seconds on a grown complex, so running it inside the `FuncAnimation` callback on
        the GUI thread blocks the event loop and the OS flags the window 'not responding'.
        Instead a background thread runs the build while the GUI thread only paints finished
        frames on a timer.

        A two-way handshake keeps the engine single-threaded even though two threads are live:
        the worker computes frame *n* only after the GUI has finished painting frame *n-1*
        (`paint_done`), and the GUI reads the geometry only while the worker is parked. So the
        engine's `st` is never mutated and read at once — no data race, no snapshotting, and the
        drawing code is unchanged. Responsiveness comes from the bindings releasing the GIL for
        the duration of each compute call, so the GUI thread keeps servicing the event loop."""
        import itertools
        import queue
        import threading
        from matplotlib.animation import FuncAnimation

        q = queue.Queue()                 # worker -> GUI: ('announce'|'paint'|'error', frame)
        paint_done = threading.Event()    # GUI -> worker: safe to mutate the engine again
        paint_done.set()                  # frame 0 may start immediately
        state = {"error": None}

        def worker():
            frame = -1
            try:
                for frame in range(self._frames):
                    paint_done.wait()          # previous frame fully painted → engine idle
                    paint_done.clear()
                    q.put(("announce", frame))
                    self._advance(frame)        # heavy compute; GIL released inside the engine
                    q.put(("paint", frame))
            except BaseException as exc:         # surface a compute failure, don't hang the GUI
                state["error"] = exc
                q.put(("error", frame))
                paint_done.set()

        def on_timer(_ignored_frame):
            while True:
                try:
                    kind, frame = q.get_nowait()
                except queue.Empty:
                    break
                if kind == "announce":
                    if not self._done:
                        self._announce(frame)
                elif kind == "paint":
                    self._paint(frame)          # worker is parked → safe to read the engine
                    paint_done.set()            # release the worker for the next frame
                    if frame >= self._frames - 1:
                        self._anim.event_source.stop()
                else:                            # error: stop and close so plt.show() returns
                    self._anim.event_source.stop()
                    plt.close(self.fig)
            return []

        worker_thread = threading.Thread(target=worker, name="proton-build", daemon=True)
        # A brisk poll so a finished frame paints promptly; the compute cadence is set by the
        # engine, not this interval. cache_frame_data=False: the frame source is unbounded.
        self._anim = FuncAnimation(self.fig, on_timer, frames=itertools.count(),
                                   interval=max(20, interval // 4), repeat=False,
                                   cache_frame_data=False, blit=False)
        worker_thread.start()
        plt.show()
        if state["error"] is not None:
            raise state["error"]


def build_proton_nodes(seed=3, precone=0, precone_timelike=False):
    """The single one-step `MultiCobordism` node the animation drives, as a 1-element
    list: `Proton.direct_node` — the three bare quarks `{1}`, `{ω}`, `{ω²}` plus their
    three anti-quarks (three q-q̄ pairs) as inputs and the proton singlet as the single
    output, read off the WHOLE cobordism — the *same* node setup `Proton.build_direct()`
    uses, on a single-Δ⁴ seed (attempt-0 seed = `seed`). The animation reports the live
    verdict either way.

    `precone` pre-grows the single-Δ⁴ seed by that many gated cone-in moves before
    optimization (forwarded straight to the C++ `MultiCobordism` constructor via `Proton`);
    `precone=0` (the default) leaves the bare seed untouched."""
    p = cob.Proton(seed=seed, precone=precone, precone_timelike=precone_timelike)
    return [
        (p.direct_node(seed), "Proton — 3 q-q̄ pairs → singlet {1, ω, ω²} (one step)"),
    ]


def animate(nodes, save=None, interval=200, dump_dir=None, **kw):
    """Animate the proton node sequence. `save` → write a GIF/MP4 (headless, Agg) with the
    synchronous per-frame `update`; otherwise show a live interactive window driven by
    `_run_live` (compute on a background thread, GUI thread only paints) so the window stays
    responsive through the multi-second surgery frames. `dump_dir` additionally writes each
    frame's dual-curvature panels as JSON (see `ProtonAnimator._dump_frame`). Returns the
    animator."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim_state = ProtonAnimator(nodes, **kw)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        anim_state._dump_dir = dump_dir
        print(f"per-frame dumps -> {dump_dir}/frame_0000.json, ...")
    anim_state._setup(plt)
    anim_state.fig.suptitle(f"{anim_state._TITLE_PREFIX} — live")
    if save:
        fa = FuncAnimation(anim_state.fig, anim_state.update,
                           frames=anim_state._frames, interval=interval,
                           repeat=False, blit=False)
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        fa.save(save, writer=writer, dpi=90)
        print(f"saved animation -> {save}")
        anim_state._anim = fa  # keep a ref so it isn't GC'd
    else:
        anim_state._run_live(plt, interval)   # responsive live window; keeps its own _anim ref
    return anim_state


def run_build(nodes, visualize=False, save=None, degree=3, init_steps=_INIT_STEPS,
              evolve_steps=_EVOLVE_STEPS,
              stage1_candidates=_STAGE1_CANDIDATES,
              max_lookahead_depth=_MAX_LOOKAHEAD_DEPTH,
              max_lookahead_tries=_MAX_LOOKAHEAD_TRIES,
              stage2_beta=1.0, interval=200,  # interval: ms/frame; GIF/MP4 fps = 1000/interval
              stage2_alpha0=0.05, stage2_rel_tol=10e-9,
              dump_dir=None, **anim_kw):
    """Run the one-step proton build over `nodes` with the combined `run` drive: an init
    pass (`grow_boundaries=True`) then an evolution pass (`grow_boundaries=False`), each
    interleaving the stage-1 surgery update with the stage-2 geometric relaxation every
    iteration.

    Visualization is **off by default**: with ``visualize=False`` (and no ``save``) this
    takes the fast **batched** path — each node's passes run to completion in one call each,
    no per-step layout/redraw overhead — and returns each node's final metrics plus the
    convergence verdict. Opt in with ``visualize=True`` (live window) or ``save=...``
    (GIF/MP4) to animate it step-by-step (slower); that returns the per-step history."""
    if not visualize and not save:
        out = []
        for node, label in nodes:
            # The batched path runs each pass in ONE `run` call, so the engine's
            # own internal retry loop already spans every iteration — the
            # per-frame `max_lookahead_tries` retry has no meaning here and is
            # deliberately not applied.
            node.run(max_iters=init_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=True, beta=stage2_beta,
                     max_lookahead=max_lookahead_depth)
            node.run(max_iters=evolve_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=False, beta=stage2_beta,
                     max_lookahead=max_lookahead_depth)
            st = node.st
            out.append((label, {
                "F": float(node.objective()),
                "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
                "rU": float(node.r_u(st)),
                "b3": int(cob.MultiCobordism.betti(st)[degree]),
                "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}))
        st = nodes[-1][0].st
        res = float(cob.MultiCobordism.r_state(st, degree, cob.Proton.singlet()))
        holes = len(cob.MultiCobordism.emergent_holes(st, degree))
        out.append(("verdict", {"converged": res < _COLOR_TOL and holes >= _MIN_QUARK_HOLES,
                                "color_residual": res, "registers": holes}))
        return out
    return animate(nodes, save=save, degree=degree, init_steps=init_steps,
                   evolve_steps=evolve_steps,
                   stage1_candidates=stage1_candidates,
                   max_lookahead_depth=max_lookahead_depth,
                   max_lookahead_tries=max_lookahead_tries,
                   stage2_beta=stage2_beta, stage2_alpha0=stage2_alpha0,
                   stage2_rel_tol=stage2_rel_tol, interval=interval,
                   dump_dir=dump_dir, **anim_kw).hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Visualization is OFF by default (fast). Opt in with --live or --save.
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--init", type=int, default=_INIT_STEPS,
                    help="init-pass (grow_boundaries=True) combined-run iterations")
    ap.add_argument("--evolve", type=int, default=_EVOLVE_STEPS,
                    help="evolution-pass (grow_boundaries=False) combined-run iterations")
    ap.add_argument("--max-lookahead-depth", type=int,
                    default=_MAX_LOOKAHEAD_DEPTH, dest="max_lookahead_depth",
                    help="on a stall, deepen the stage-1 search to sequences of up "
                         "to this many moves (1 = single moves only). This was "
                         "called --max-lookahead before the retry flag took that "
                         "name")
    ap.add_argument("--max-lookahead", type=int, default=_MAX_LOOKAHEAD_TRIES,
                    dest="max_lookahead_tries",
                    help="how many times to redraw stage-1 candidates within one "
                         "frame before giving up and advancing (1 = one draw, the "
                         "historical behaviour). Candidates are drawn at random, so "
                         "a stalled frame is usually a bad draw rather than a dead "
                         "end; only stalled frames cost extra tries")
    ap.add_argument("--precone", type=int, default=0,
                    help="pre-grow the single-Δ⁴ seed by this many gated "
                         "cone-in moves before optimization (0 = bare seed)")
    ap.add_argument("--precone-timelike", action="store_true", dest="precone_timelike",
                    help="draw every precone cone-in as the TIMELIKE disposition, so "
                         "the pre-grown material carries causal content")
    ap.add_argument("--candidates", type=int, default=_STAGE1_CANDIDATES,
                    help="stage-1 candidate moves per batch (the search BREADTH; "
                         "the depth-1 batch is scored in parallel across "
                         "--threads workers, so breadth is nearly free up to "
                         "the worker count)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="stage-2 weight on ||grad S||^2 (beta != 1 mixes the "
                         "two halves' scales in the shared F trace)")
    ap.add_argument("--alpha0", type=float, default=0.05,
                    help="stage-2 initial line-search step scale")
    ap.add_argument("--rel-tol", type=float, default=10e-9, dest="rel_tol",
                    help="stage-2 in-loop diminishing-returns cut (the exit "
                         "path re-checks at 1e-12 regardless): larger = less "
                         "geometric work per committed move, more moves per "
                         "second")
    ap.add_argument("--init-chunk", type=int, default=_INIT_CHUNK, dest="init_chunk",
                    help="init-pass run iterations per animation frame")
    ap.add_argument("--evolve-chunk", type=int, default=_EVOLVE_CHUNK,
                    dest="evolve_chunk",
                    help="evolution-pass run iterations per animation frame")
    ap.add_argument("--threads", type=int, default=16,
                    help="OpenMP worker count for the engine (candidate batch, "
                         "action gradient/Hessian). The box authorization is "
                         "16 of 32 cores; pass 32 explicitly to use all")
    ap.add_argument("--dump-dir", dest="dump_dir",
                    help="write each frame's dual-curvature panels (Re ε / Im ε per "
                         "dual node, their positions, the dual adjacency, and the "
                         "rehydratable complex) as JSON here — so a claim about a "
                         "late frame can be checked without re-running to it")
    ap.add_argument("--hodge-weights", choices=("content", "squared"),
                    default="squared", dest="hodge_weights",
                    help="which quantity the Hodge inner-product weight W_k is "
                         "built from, for EVERY operator in the run (r_U, the "
                         "near-kernel residual, the register readout, the "
                         "spectrum/mode panels): "
                         "'content' = V, the k-content — an edge weighs its "
                         "length, so a timelike cell's weight is IMAGINARY; "
                         "'squared' = V², the engine default — an edge weighs "
                         "exactly its ℓ², real and signed. Both are fully "
                         "Lorentzian; this example defaults to 'squared' (the "
                         "engine default; multicobordism_animation.py defaults "
                         "to 'content')")
    args = ap.parse_args()
    # One flip, process-wide, BEFORE any node is built (flipping mid-run would
    # mix conventions across cached spectra).
    convention = {"content": cob.HodgeWeightConvention.Content,
                  "squared": cob.HodgeWeightConvention.SquaredContent}[args.hodge_weights]
    cob.HodgeLaplacian.setDefaultWeightConvention(convention)
    ProtonAnimator._TITLE_PREFIX += (
        f"  ·  W = {'V' if args.hodge_weights == 'content' else 'V²'}")
    nodes = build_proton_nodes(seed=args.seed, precone=args.precone,
                               precone_timelike=args.precone_timelike)
    result = run_build(nodes, visualize=args.live, save=args.save, init_steps=args.init,
                       evolve_steps=args.evolve,
                       init_chunk=args.init_chunk, evolve_chunk=args.evolve_chunk,
                       stage1_candidates=args.candidates, stage2_beta=args.beta,
                       stage2_alpha0=args.alpha0, stage2_rel_tol=args.rel_tol,
                       max_lookahead_depth=args.max_lookahead_depth,
                       max_lookahead_tries=args.max_lookahead_tries,
                       dump_dir=args.dump_dir)
    if not args.live and not args.save:
        print("one-step proton build finished (visualization off by default — pass --live "
              "or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
