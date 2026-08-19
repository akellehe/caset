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
import time

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
    os.environ["OMP_NUM_THREADS"] = _n
# BLAS pools serve ONLY numpy's panel-side eigensolves (the engine's Eigen
# kernels thread through OpenMP, not BLAS), and those matrices are small: a
# perf sample of the live drive showed the 32-thread OpenBLAS pool spinning in
# blas_thread_server for 1.3% of all cycles while contending with the engine
# team. One BLAS thread is the right size regardless of --threads; an explicit
# value in the caller's environment wins.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# Workers SLEEP at OpenMP barriers instead of spinning. GNU OpenMP's default
# wait policy spins, and a perf sample of the live drive showed 55% of all
# self-time inside libgomp doing exactly that: the engine's parallel regions
# (candidate batch, action gradient/Hessian) are short, so at high --threads
# the idle workers burn most of the CPU between regions and contend with the
# serial sections. Like the thread count above, this must be set BEFORE
# OpenMP initializes; an explicit value in the caller's environment wins.
os.environ.setdefault("OMP_WAIT_POLICY", "passive")

import numpy as np
from scipy.sparse.csgraph import shortest_path

import tessera

cob = tessera.cobordism

from geometry_state import GeometryState
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
# The O(n³) spectrum/mode/Krein recording refreshes at least this often (in
# frames); commits and node switches always refresh, so only pure-relaxation
# frames ever reuse — bounded metric staleness, exact combinatorics (#671).
_SPECTRA_REFRESH_EVERY = 4
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
    diagonal's ℓ²(v,v) = 0 is supplied here, not looked up).

    Vectorized: faces are deduplicated once, their Gram matrices stacked, and
    one batched ``np.linalg.det`` call factorizes them all — the same LAPACK
    routine the old per-face calls ran, so the values are unchanged; only the
    per-face Python/numpy call overhead is gone (#671)."""
    def interval(u, v):
        return 0.0 if u == v else squared_length(u, v)
    faces = {2: set(), 3: set()}
    for cell in cells:
        ordered = sorted(cell)
        for k in (2, 3):
            faces[k].update(itertools.combinations(ordered, k + 1))
    out = {}
    for k in (2, 3):
        ordered_faces = sorted(faces[k])
        if not ordered_faces:
            out[k] = np.array([])
            continue
        grams = np.empty((len(ordered_faces), k, k))
        for f, face in enumerate(ordered_faces):
            v0, rest = face[0], face[1:]
            for i, a in enumerate(rest):
                for j, b in enumerate(rest):
                    grams[f, i, j] = 0.5 * (interval(v0, a) + interval(v0, b)
                                            - interval(a, b))
        out[k] = np.linalg.det(grams)
    return out


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
                 stage2_alpha0=0.05, stage2_rel_tol=10e-9, relax_budget=10,
                 spectra_every=_SPECTRA_REFRESH_EVERY, no_combinatorial_moves=False,
                 relax_chunk=None, status=True, checkpoint=0, checkpoint_dir=None):
        self._common_init(nodes, degree)
        self.s1c, self.s2_beta = stage1_candidates, stage2_beta
        self.lookahead_depth = max_lookahead_depth
        self.lookahead_tries = max_lookahead_tries
        self.s2_alpha0, self.s2_rel_tol = stage2_alpha0, stage2_rel_tol
        self.relax_budget = relax_budget
        # Relaxation-only drive (#716): no combinatorial moves of any kind —
        # no Pachner moves, no surgical cones, no disposition flips — and no
        # growth of the blocks' scoring regions, so the triangulation is fixed
        # and F descends within one region.
        #
        # The two relaxation knobs are DISTINCT and neither shadows the other.
        # `relax_budget` is the engine's relaxBudgetPerMove: how much
        # relaxation follows a committed move in the interleaved drive.
        # `relax_chunk` belongs to the relaxation-only drive alone, where a
        # frame IS a block of stage-2 iterations and there is no committed move
        # to budget against; it defaults to relax_budget so the two agree when
        # unset.
        self.no_combinatorial_moves = bool(no_combinatorial_moves)
        self.relax_chunk = int(relax_chunk) if relax_chunk else int(relax_budget)
        self.status = bool(status)
        # Every `checkpoint` frames, write the state (#722). This is the
        # orientation-faithful record — cells in intrinsic vertex order — not
        # the panel dump, whose cells are sorted for drawing.
        self.checkpoint = max(0, int(checkpoint))
        self.checkpoint_dir = checkpoint_dir
        self._t0 = time.time()
        self.spectra_every = max(1, int(spectra_every))
        self._schedule = self._make_schedule(len(nodes), init_steps, init_chunk,
                                             evolve_steps, evolve_chunk)
        self._frames = len(self._schedule)

    def _common_init(self, nodes, degree):
        """Shared drawing/recording state: the node list, history buffers, per-panel
        layouts, and curvature cache — everything `_redraw`/`_draw_*`/`verdict` read."""
        self.nodes = nodes                  # [(MultiCobordism, label), ...] in order
        self.k = degree
        self._last_relax_steps = None       # accepted stage-2 steps this frame
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [],
                     "phase": [], "node": [], "lookahead": [], "tries": [],
                     "min_det2": [], "min_det3": [], "sigma": [],
                     "mode_w": [], "mode_w_head": [], "mode_cells": [],
                     "pair_count": [], "pair_w": [], "im_leak": [],
                     "sigma_cancel": [], "sigma_cancel_soft": [],
                     "pair_src": [], "spec_frame": []}
        self._boundaries = []       # step indices where a later node begins (trace markers)
        self._layouts = [_StableLayout() for _ in nodes]   # one per complex panel
        self._active = 0            # index of the node currently being driven
        self._done = False          # so the verdict is announced exactly once
        self._curv_cache = {}       # node_index -> (frame_computed, {cell_tuple: curvature})
        self._dump_dir = None       # set by run_build(dump_dir=...); None = no dumping
        # Persistent-artist state (#670): trace Line2D handles updated via
        # set_data instead of clear-and-replot; the spec_frame the spectra
        # panels last DREW (they skip when it hasn't advanced — the draw-side
        # multiplier of #671's recording gate); per-node cached layouts so a
        # frozen node's panels are not re-laid-out and redrawn every frame.
        self._trace_artists = {}
        self._drawn_spec_frame = None
        self._drawn_spec_node = None
        self._drawn_nodes = set()
        self._last_coords = {}

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
        if self.no_combinatorial_moves:
            # Stage 2 alone. `run_stage2` returns its objective trace, so the
            # number of ACCEPTED relaxation steps this frame is len(trace) - 1.
            trace = node.run_stage2(beta=self.s2_beta, max_iters=self.relax_chunk,
                                    alpha0=self.s2_alpha0, rel_tol=self.s2_rel_tol)
            self._last_relax_steps = max(len(trace) - 1, 0)
            tries = 1
        else:
            for tries in range(1, max(self.lookahead_tries, 1) + 1):
                node.run(max_iters=count, n_candidate_moves=self.s1c,
                         grow_boundaries=(phase == "init"), beta=self.s2_beta,
                         alpha0=self.s2_alpha0, rel_tol=self.s2_rel_tol,
                         max_lookahead=self.lookahead_depth,
                         relax_budget_per_move=self.relax_budget)
                if int(node.last_stage1_lookahead) > 0:
                    break
            self._last_relax_steps = None   # `run` does not report its trace
        self._record(node, node_index, phase, tries)
        if self.checkpoint and (frame + 1) % self.checkpoint == 0:
            self._write_checkpoint(node, frame, phase)
        if self.status:
            self._print_status(node, frame, phase, tries)

    def _write_checkpoint(self, node, frame, phase):
        """Write this frame's state through `GeometryState` — top cells in
        INTRINSIC vertex order, so the orientation is recoverable from the file
        (`inspect_state.py` reads it back). The panel dump written by
        `_dump_frame` cannot serve here: it sorts each cell's vertices for
        drawing, which discards exactly that order."""
        directory = self.checkpoint_dir or self._dump_dir or "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"state_{frame + 1:04d}.json")
        GeometryState.write(node.st, path, meta={
            "frame": frame + 1,
            "phase": phase,
            "F": float(self.hist["F"][-1]) if self.hist["F"] else None,
            "gradN2": float(self.hist["gradN2"][-1]) if self.hist["gradN2"] else None,
            "rU": float(self.hist["rU"][-1]) if self.hist["rU"] else None,
            "degree": self.k,
        })
        if self.status:
            print(f"  checkpoint -> {path}", flush=True)

    def _print_status(self, node, frame, phase, tries):
        """One line per frame on stdout: where the drive is, and what is
        blocking it. Everything here is read from signals the engine already
        publishes — no extra computation beyond what `_record` just stored.

        The two obstruction signals are the point of this line:

        * `stage1` — `last_stage1_lookahead` is the lookahead depth whose
          candidate was committed, or 0 when the whole batch found nothing that
          lowered F. Zero means the combinatorial search is stuck: every drawn
          move either failed the validity gate or did not improve the
          objective, so the topology cannot currently leave this region.
        * `stage2` — `last_stage2_stationary` is true when the line search
          halved its step all the way down without finding a descending trial.
          That is the geometric relaxation reporting it has reached the bottom
          of this region (to `--rel-tol`), not a failure.
        """
        history = self.hist
        objective = history["F"][-1]
        change = (objective - history["F"][-2]) if len(history["F"]) > 1 else 0.0
        gradient_norm_squared = history["gradN2"][-1]
        bare_residual = history["rU"][-1]
        cell_count = len(node.st.getTopSimplices())
        if self.no_combinatorial_moves:
            stage1_note = "stage1 off (--no-combinatorial-moves)"
        elif int(node.last_stage1_lookahead) > 0:
            stage1_note = (f"stage1 committed at depth "
                           f"{int(node.last_stage1_lookahead)}"
                           + (f" after {tries} draws" if tries > 1 else ""))
        else:
            stage1_note = (f"stage1 STUCK: none of {self.s1c} candidates lowered F"
                           + (f" in {tries} draws" if tries > 1 else ""))
        if node.last_stage2_stationary:
            stage2_note = "stage2 STATIONARY (no descending step found)"
        elif self._last_relax_steps is not None:
            stage2_note = f"stage2 {self._last_relax_steps} steps accepted"
        else:
            stage2_note = "stage2 descending"
        print(f"f{frame + 1:04d} {phase:<6} | F {objective:.6e} "
              f"dF {change:+.3e} | grad2 {gradient_norm_squared:.3e} "
              f"G*rU {objective - gradient_norm_squared:.3e} rU {bare_residual:.3e} | "
              f"cells {cell_count} b{self.k} {history['b3'][-1]} "
              f"holes {history['holes'][-1]} | {stage1_note} | {stage2_note} | "
              f"{time.time() - self._t0:.0f}s", flush=True)

    def _record(self, node, node_index, phase, tries=1):
        st = node.st
        self.hist["F"].append(float(node.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.regge_action_gradient(st)))
        self.hist["rU"].append(float(node.r_u(st)))
        # Betti is TOPOLOGY: it can only change when stage 1 commits a move, so
        # relaxation-only frames reuse the last value exactly (#671). A commit,
        # a node switch, or the first frame always recomputes.
        committed = int(node.last_stage1_lookahead) > 0
        same_node = bool(self.hist["node"]) and self.hist["node"][-1] == node_index
        if committed or not same_node or not self.hist["b3"]:
            b3_now = int(cob.MultiCobordism.betti(st)[self.k])
        else:
            b3_now = self.hist["b3"][-1]
        self.hist["b3"].append(b3_now)
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
        # The spectrum/mode/Krein block is O(n^3) DISPLAY work. Combinatorial
        # consistency is exact — a committed move always refreshes, so the
        # recorded mode_cells can never be stale against the live complex —
        # and pure-relaxation frames reuse the last spectra for at most
        # ``spectra_every - 1`` frames (metric staleness only), like the
        # curvature heat's ``_HEAT_REFRESH_EVERY``. ``spec_frame`` records the
        # frame each row was actually computed on (#671). Verdict quantities
        # (r_state, holes, F, r_U) are untouched and stay per-frame exact.
        prior = len(self.hist["sigma"])
        refresh = (committed or not same_node or prior == 0
                   or prior - self.hist["spec_frame"][-1] >= self.spectra_every)
        if refresh:
            self._record_spectra(st, node)
            self.hist["spec_frame"].append(prior)
        else:
            for key in ("sigma", "mode_w", "mode_w_head", "mode_cells",
                        "im_leak", "pair_src", "pair_count", "pair_w",
                        "sigma_cancel", "sigma_cancel_soft", "spec_frame"):
                self.hist[key].append(self.hist[key][-1])
        # Null-face proximity: the smallest |det G| over the complex's triangles
        # and tets — 0 = a face exactly tangent to the light cone (degenerate).
        min_det2, min_det3 = _min_abs_gram_dets(st)
        self.hist["min_det2"].append(min_det2)
        self.hist["min_det3"].append(min_det3)

    def _record_spectra(self, st, node):
        """The O(n^3) spectrum/mode-localization/Krein recording block,
        verbatim from _record; gated there by ``spectra_every`` (#671)."""
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
            source = "live"
            if not krein.on_locus and krein.imag_interval_leak > 0:
                # The LIVE state spends most of a build OFF the real-ℓ² locus
                # (stage 2 explores complex intervals). The classification no
                # longer projects or refuses there: KreinModes now carries the
                # BILINEAR W-form generalization (#694) — quasi-null modes,
                # defined for genuinely complex ℓ², reducing exactly to the
                # broken-pair set on the locus — so the panels read the LIVE
                # operator with the imaginary part included, labeled as such.
                source = "bilinear"
            if krein.on_locus or krein.null_indices is not None:
                self.hist["pair_src"].append(source)
                # On the locus: the exact broken-pair count. Off it: half the
                # quasi-null MODE count — equal to pair_count whenever both
                # are defined, so the trace is continuous across locus
                # crossings and the off-locus value is labeled by pair_src.
                # On the locus: the exact broken-pair count. Off it: the
                # DE-ROTATED pair count (#703) — pairs about the measured
                # dominant ray, the same definition the locus reads at phi=0.
                self.hist["pair_count"].append(
                    float(krein.pair_count) if krein.on_locus
                    else float(len(krein.derotated_pair_partners)))
                # Weights/eigenvectors are indexed by the (possibly projected)
                # operator's own cell order; remap onto the LIVE `mode_cells`
                # order every panel consumer uses. The projection changes
                # lengths only, so the cell SETS coincide and the map is total.
                live_cells = self.hist["mode_cells"][-1]
                position = {cell: i for i, cell in enumerate(krein.cells)}
                permutation = np.array([position[c] for c in live_cells])
                heat = (krein.pair_heat() if krein.on_locus
                        else krein.cell_weight(sorted(
                            {i for pair in krein.derotated_pair_partners
                             for i in pair}
                            | set(krein.null_indices)
                            | set(krein.forming_indices))))
                self.hist["pair_w"].append(
                    heat[permutation] if heat.size else heat)
                # Which SINGULAR directions the broken pairs span. There is no
                # canonical σ↔λ map on a non-normal operator, so the honest
                # marking is subspace overlap: per descending rank r, the
                # weight of its right singular vector inside the span of the
                # pair modes (both conjugate partners — the factor 2; on the
                # locus the operator is real, so the partners' overlaps
                # coincide). One fully-cancelled direction contributes ≈ 1;
                # the total is 2·pairs.
                cancel = np.zeros(sigma.size)
                soft_cancel = np.zeros(sigma.size)
                if krein.on_locus:
                    # one representative per pair, x2 for both partners
                    marked = [(i, 2.0) for i in krein.pair_indices]
                    soft_marked = []
                else:
                    # off-locus (#703): HARD = exact structure — de-rotated
                    # conjugate pairs (both partners, once each) plus exact
                    # W-null modes; SOFT = the forming quasi-null band
                    # (q below a quarter of the median), minus the hard set.
                    hard = ({i for pair in krein.derotated_pair_partners
                             for i in pair} | set(krein.null_indices))
                    marked = [(i, 1.0) for i in sorted(hard)]
                    soft_marked = [(i, 1.0)
                                   for i in krein.forming_indices
                                   if i not in hard]
                for rows, marks in ((cancel, marked),
                                    (soft_cancel, soft_marked)):
                    for i, factor in marks:
                        v = krein.eigenvectors[permutation, i]
                        nrm = np.linalg.norm(v)
                        if nrm > 0:
                            rows += factor * np.abs(vh @ (v / nrm)) ** 2
                self.hist["sigma_cancel"].append(cancel)
                self.hist["sigma_cancel_soft"].append(soft_cancel)
            else:
                self.hist["pair_src"].append("none")
                self.hist["pair_count"].append(float("nan"))
                self.hist["pair_w"].append(np.array([]))
                self.hist["sigma_cancel"].append(np.array([]))
                self.hist["sigma_cancel_soft"].append(np.array([]))
                self._pair_note = f"off the real-ℓ² locus: {krein.reason}"
        else:
            self.hist["sigma"].append(np.array([]))
            self.hist["mode_w"].append(np.array([]))
            self.hist["mode_w_head"].append(np.array([]))
            self.hist["pair_count"].append(0.0)
            self.hist["pair_w"].append(np.array([]))
            self.hist["im_leak"].append(0.0)
            self.hist["sigma_cancel"].append(np.array([]))
            self.hist["sigma_cancel_soft"].append(np.array([]))
            self.hist["pair_src"].append("none")
            self.hist["mode_cells"].append([])

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
        # NULL edges (#730). An interval is null when l^2 vanishes, so the test is
        # on |l^2| against the complex's OWN scale rather than an absolute number.
        # Two bands, because they mean different things: at-tolerance is a
        # genuinely degenerate edge, while merely NEAR null is where a
        # balanced-edge run starts every edge by construction (Re l^2 = 0) and is
        # not yet a degeneracy.
        edges = st.getEdgeList().toVector()
        squared = {}
        for e in edges:
            a, b = e.getSource().getId(), e.getTarget().getId()
            squared[(a, b)] = e.getLength() ** 2
        scale = max((abs(z) for z in squared.values()), default=1.0) or 1.0
        null_edges, near_null_edges = set(), set()
        census = {"spacelike": 0, "timelike": 0, "null": 0, "undecided": 0}
        for key, z in squared.items():
            if abs(z) <= 1e-9 * scale:
                null_edges.add(key)
            elif abs(z) <= 1e-3 * scale:
                near_null_edges.add(key)
            # Causal character. NULL means l^2 itself vanishes; that is a
            # different condition from Re l^2 = 0, which is where a
            # balanced-edge run starts EVERY edge (l^2 = i*m has |l^2| = m, so
            # it is causally UNDECIDED, not lightlike). Counting them apart
            # keeps a run full of undecided edges from reading as degenerate.
            real, imaginary = z.real, z.imag
            if abs(z) <= 1e-9 * scale:
                census["null"] += 1
            elif abs(real) <= 1e-9 * abs(z):
                census["undecided"] += 1
            elif abs(imaginary) > 1e-9 * abs(z):
                census["undecided"] += 1
            elif real > 0:
                census["spacelike"] += 1
            else:
                census["timelike"] += 1
        for e in edges:
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a not in coords or b not in coords:
                continue
            p, q = coords[a], coords[b]
            if any(a in vs and b in vs for vs in hole_vsets):     # a register-cell edge
                ax.plot([p[0], q[0]], [p[1], q[1]], color="C3", lw=1.8, zorder=3)
            elif (a, b) in null_edges:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="C1", lw=1.6, zorder=3)
            elif (a, b) in near_null_edges:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="C1", lw=0.9, alpha=0.55,
                        ls=(0, (2, 2)), zorder=2)
            else:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=0.5, zorder=1)
        if coords:
            pts = np.array(list(coords.values()))
            ax.scatter(pts[:, 0], pts[:, 1], c="0.4", s=8, zorder=2)
            # The boundary STATES (#730). Nothing structurally protects these
            # regions — pinnedBoundaryVertices() is empty by design and the
            # states are held only by their r_U terms — so seeing where they
            # sit is the only way to watch whether the objective is holding
            # them. Rings, not fills, so the register outlines stay readable.
            for block_set, colour, name in (
                    (getattr(node, "inputs", []), "C0", "input state ∂W⁻"),
                    (getattr(node, "outputs", []), "C4", "output state ∂W⁺")):
                marked = {v for block in block_set for v in block.vertices}
                here = np.array([coords[v] for v in marked if v in coords])
                if len(here):
                    ax.scatter(here[:, 0], here[:, 1], s=70, facecolors="none",
                               edgecolors=colour, linewidths=1.3, zorder=5,
                               label=f"{name} ({len(marked)} vertices)")
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
        census_tag = "  ".join(f"{count} {name}"
                               for name, count in census.items() if count)
        if near_null_edges:
            census_tag += f"  ({len(near_null_edges)} near-null)"
        ax.set_title(f"{title}  —  {n_holes} register{'s' if n_holes != 1 else ''}"
                     f"  ·  {len(edges)} edges: {census_tag}", fontsize=9)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.7,
                      handletextpad=0.3, borderpad=0.3)
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
                    "lookahead", "tries", "min_det2", "min_det3", "spec_frame"):
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
        weight_power = float((weights ** 2).sum())
        # weights sum to 1 — except the legitimate all-zero case (e.g. the
        # annihilation heat with zero broken pairs), where PR reads 0.
        participation = 1.0 / weight_power if weight_power > 0 else 0.0
        ax.set_aspect("equal")
        ax.set_title(f"{title}, PR {participation:.1f}/{len(weights)} cells",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    def _redraw(self):
        xs = range(len(self.hist["F"]))
        self.axm.clear()
        self.axm.plot(xs, self.hist["F"], label="F (objective)", color="C0")
        self.axm.plot(xs, self.hist["gradN2"], label="‖∇S‖²", color="C1")
        # The residual AS IT ENTERS F: Γ·r_U = F − ‖∇S‖² by definition, so no
        # engine getter is needed and the identity is exact. At large Γ this
        # trace visually hugs F — which is the point: the bare r_U line below
        # (kept, dashed, pre-prefactor) is orders of magnitude away from the
        # term the objective actually trades against ‖∇S‖², and plotting only
        # the bare residual made every committed move look F-increasing (the
        # two big lines render flat at their scale while ‖∇S‖² visibly rises).
        gamma_ru = [f - g for f, g in zip(self.hist["F"], self.hist["gradN2"])]
        self.axm.plot(xs, gamma_ru, label="Γ·r_U (= F − ‖∇S‖²)", color="C4",
                      lw=1.0, alpha=0.9)
        self.axm.plot(xs, self.hist["rU"], label="r_U (bare, pre-Γ)", color="C2",
                      ls="--", alpha=0.6)
        # Per-frame ΔF, signed, on the same symlog axis: descent shows as a
        # NEGATIVE trace regardless of F's absolute scale, so "is the
        # objective actually decreasing" is answerable at a glance even when
        # F's own line renders flat.
        if len(self.hist["F"]) > 1:
            dF = [b - a for a, b in zip(self.hist["F"], self.hist["F"][1:])]
            self.axm.plot(list(xs)[1:], dF, label="ΔF (per frame)",
                          color="C5", marker=".", ms=3, lw=0.8)
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
            if ni != self._active and ni in self._drawn_nodes:
                # A non-active node's complex is frozen: keep its last layout
                # and panels as drawn (#670). The cached coords keep the frame
                # dump payload identical to a fresh draw.
                coords_by_node[ni] = self._last_coords[ni]
                continue
            coords = self._layouts[ni].coords(self.nodes[ni][0].st)
            coords_by_node[ni] = coords
            self._last_coords[ni] = coords
            self._drawn_nodes.add(ni)
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
            xs = range(len(self.hist["min_det2"]))
            ta = self._trace_artists
            if "null2" not in ta:                      # first frame: build once
                self.ax_null.clear()
                (ta["null2"],) = self.ax_null.semilogy(
                    xs, self.hist["min_det2"], color="C0",
                    marker=".", label="min |det G| — triangles")
                (ta["null3"],) = self.ax_null.semilogy(
                    xs, self.hist["min_det3"], color="C3",
                    marker=".", label="min |det G| — tets")
                self.ax_null.axhline(1e-6, color="0.6", ls=":", lw=0.8,
                                     label="1e-6 (danger: near-degenerate)")
                self.ax_null.set_title("null-face proximity (det G = 0 ⇔ face "
                                       "tangent to the light cone)", fontsize=9)
                self.ax_null.set_xlabel("frame")
                self.ax_null.set_ylabel("min |det G|")
            else:                                      # later frames: data only
                ta["null2"].set_data(list(xs), self.hist["min_det2"])
                ta["null3"].set_data(list(xs), self.hist["min_det3"])
                self.ax_null.relim()
                self.ax_null.autoscale_view()
        # The spectrum/mode/annihilation panels draw hist rows the recorder
        # refreshes on the #671 cadence; between refreshes their inputs are
        # IDENTICAL, so redrawing is pure waste — skip until spec_frame
        # advances (or the active node changes). Zero visual change.
        spec_now = self.hist["spec_frame"][-1] if self.hist["spec_frame"] else None
        active_now = self.hist["node"][-1] if self.hist["node"] else 0
        spec_dirty = (spec_now is None
                      or spec_now != self._drawn_spec_frame
                      or active_now != self._drawn_spec_node)
        if (getattr(self, "ax_spec", None) is not None and self.hist["sigma"]
                and spec_dirty):
            ax = self.ax_spec
            ax.clear()
            sigma = self.hist["sigma"][-1]
            n_cancelled = 0
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
                bars = ax.bar(ranks, shown, width=0.9, color=colors, zorder=3)
                # Green-edge the CANCELLED directions: ranks whose right
                # singular vector lies majority (≥ ½) inside the span of the
                # exact pair structure — broken conjugate pairs on the locus,
                # de-rotated pairs about the dominant ray plus exact W-null
                # modes off it (#703). Dotted green edge: the FORMING band
                # (quasi-null q below a quarter of the median), not yet exact.
                cancel = (self.hist["sigma_cancel"][-1]
                          if self.hist["sigma_cancel"] else np.array([]))
                soft = (self.hist["sigma_cancel_soft"][-1]
                        if self.hist["sigma_cancel_soft"] else np.array([]))
                n_forming = 0
                if cancel.size == sigma.size:
                    for rank_index, (bar, share) in enumerate(zip(bars, cancel)):
                        if share >= 0.5:
                            # Hatched, not just edged: a 1-pixel outline on a
                            # narrow bar is invisible at panel size.
                            bar.set_hatch("//")
                            bar.set_edgecolor("C2")
                            bar.set_linewidth(1.4)
                            n_cancelled += 1
                        elif (soft.size == sigma.size
                              and soft[rank_index] >= 0.5):
                            bar.set_edgecolor("C2")
                            bar.set_linewidth(1.2)
                            bar.set_linestyle((0, (1, 1)))
                            n_forming += 1
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
            cancel_tag = (f"; green edge: {n_cancelled} cancelled (pairs)"
                          if n_cancelled else "")
            if n_forming:
                cancel_tag += f"; dotted: {n_forming} forming"
            ax.set_title(f"σ(L{self.k}) descending — red: {m}-smallest "
                         f"(near-kernel tail){cancel_tag}", fontsize=9)
            ax.set_xlabel("rank (descending)")
            ax.set_ylabel("σ")
            self.ax_null.legend(loc="lower right", fontsize=7)
        if getattr(self, "ax_mode", None) is not None and self.hist["mode_w"] and spec_dirty:
            active = self.hist["node"][-1]
            self._draw_mode_heat(self.ax_mode, active,
                                 coords_by_node.get(active, {}),
                                 self.hist["mode_w"][-1], self._mode_sm,
                                 _MODE_CMAP, ("near-kernel", "smallest"))
        if (getattr(self, "ax_mode_head", None) is not None
                and self.hist["mode_w_head"]) and spec_dirty:
            active = self.hist["node"][-1]
            self._draw_mode_heat(self.ax_mode_head, active,
                                 coords_by_node.get(active, {}),
                                 self.hist["mode_w_head"][-1], self._mode_head_sm,
                                 _MODE_CMAP_HEAD, ("near-null", "largest"))
        if getattr(self, "ax_pair", None) is not None and self.hist["pair_w"] and spec_dirty:
            active = self.hist["node"][-1]
            n_pairs = self.hist["pair_count"][-1]
            src = self.hist["pair_src"][-1] if self.hist["pair_src"] else ""
            src_tag = {"projected": ", Re-projected",
                       "bilinear": ", bilinear W-form (off-locus)"}.get(src, "")
            self._draw_mode_heat(
                self.ax_pair, active, coords_by_node.get(active, {}),
                self.hist["pair_w"][-1], self._pair_sm, _MODE_CMAP_PAIR,
                title=("annihilation heat — unavailable"
                       if math.isnan(n_pairs) else
                       f"annihilation heat — {int(n_pairs)} broken pairs "
                       f"(W-null{src_tag})"),
                empty_note=getattr(self, "_pair_note",
                                   "no Krein classification yet"))
        if spec_dirty:
            self._drawn_spec_frame = spec_now
            self._drawn_spec_node = active_now
        if (getattr(self, "ax_pair_trace", None) is not None
                and self.hist["pair_count"]):
            ax = self.ax_pair_trace
            xs = range(len(self.hist["pair_count"]))
            ta = self._trace_artists
            if "pairs" not in ta:                      # first frame: build once
                ax.clear()
                (ta["pairs"],) = ax.plot(xs, self.hist["pair_count"],
                                         color="C4", marker=".")
                ax.set_title("broken pairs (Re-projected off-locus) + real-ℓ² "
                             "locus distance", fontsize=9)
                ax.set_xlabel("frame")
                ax.set_ylabel("pairs", color="C4")
                ta["pairs_boundaries"] = 0
            else:                                      # later frames: data only
                ta["pairs"].set_data(list(xs), self.hist["pair_count"])
                ax.relim()
                ax.autoscale_view()
            while ta["pairs_boundaries"] < len(self._boundaries):
                b = self._boundaries[ta["pairs_boundaries"]]
                ax.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
                ta["pairs_boundaries"] += 1
            if getattr(self, "ax_pair_leak", None) is not None:
                axl = self.ax_pair_leak
                if "leak" not in ta:
                    axl.clear()
                    (ta["leak"],) = axl.plot(xs, self.hist["im_leak"],
                                             color="C2", alpha=0.7, lw=1.0)
                    axl.set_ylabel("max|Im ℓ²| (locus distance)", color="C2",
                                   fontsize=8)
                    axl.tick_params(labelsize=6)
                else:
                    ta["leak"].set_data(list(xs), self.hist["im_leak"])
                    axl.relim()
                    axl.autoscale_view()

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
    def _run_live(self, plt, interval, save=None):
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
        the duration of each compute call, so the GUI thread keeps servicing the event loop.

        With `save` set, the window is ALSO recorded: a writer is held open around
        `plt.show()` and grabs the canvas right after each paint, on the GUI
        thread. Recording has to happen here rather than in a second pass,
        because a frame commits moves and relaxation into the complex and the
        run cannot be replayed. Closing the window early finalizes whatever was
        recorded, so a partial run still yields a playable file."""
        import itertools
        import queue
        import signal
        import subprocess
        import threading
        import contextlib
        from matplotlib.animation import FuncAnimation

        q = queue.Queue()                 # worker -> GUI: ('announce'|'paint'|'error', frame)
        paint_done = threading.Event()    # GUI -> worker: safe to mutate the engine again
        paint_done.set()                  # frame 0 may start immediately
        state = {"error": None, "interrupted": False}

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
                    if writer is not None:
                        # savefig-based: renders the figure itself, so the grab
                        # reflects the paint that just happened. Before
                        # releasing the worker, so the engine cannot advance
                        # mid-capture.
                        writer.grab_frame()
                    paint_done.set()            # release the worker for the next frame
                    if frame >= self._frames - 1:
                        self._anim.event_source.stop()
                else:                            # error: stop and close so plt.show() returns
                    self._anim.event_source.stop()
                    plt.close(self.fig)
            return []

        writer = None
        if save:
            from matplotlib.animation import FFMpegWriter, PillowWriter
            writer_class = PillowWriter if save.endswith(".gif") else FFMpegWriter
            # The playback rate the single-flag save path uses: `interval` is
            # milliseconds per frame.
            writer = writer_class(fps=max(1, round(1000 / max(interval, 1))))

        worker_thread = threading.Thread(target=worker, name="proton-build", daemon=True)
        # A brisk poll so a finished frame paints promptly; the compute cadence is set by the
        # engine, not this interval. cache_frame_data=False: the frame source is unbounded.
        self._anim = FuncAnimation(self.fig, on_timer, frames=itertools.count(),
                                   interval=max(20, interval // 4), repeat=False,
                                   cache_frame_data=False, blit=False)
        # `saving` must stay open for the whole show() loop, since frames are
        # grabbed as they are painted. Without a save path this is a no-op
        # context, so the single-flag live path runs exactly as before.
        recording = (writer.saving(self.fig, save, 90) if writer is not None
                     else contextlib.nullcontext())

        # An interrupt has to CLOSE THE FIGURE rather than raise. Qt's event
        # loop is C++ and does not return to Python when the signal lands, so a
        # KeyboardInterrupt is never delivered and the run would simply carry on
        # (measured: SIGINT left a live run going, and killing it then left an
        # unplayable 48-byte MP4 with no trailer, or no GIF at all). Closing the
        # figure makes plt.show() return through its ordinary path, so the
        # writer finalizes exactly as it does when the window is closed. The
        # animation timer re-enters Python every few tens of milliseconds, so
        # the handler runs promptly.
        def on_interrupt(_signal_number, _frame):
            state["interrupted"] = True
            try:
                self._anim.event_source.stop()
            finally:
                plt.close(self.fig)

        # A Python signal handler only runs between bytecodes in the main
        # thread, and while Qt's loop is in C++ nothing there executes. The
        # animation's own timer is not enough to rely on: it STOPS on the last
        # frame, after which a finished run would sit in show() with no Python
        # running and ignore the interrupt entirely (measured). This timer does
        # nothing except return to Python a few times a second, for as long as
        # the window is open, so a pending signal is always delivered promptly.
        signal_pump = self.fig.canvas.new_timer(interval=200)
        signal_pump.add_callback(lambda: None)

        previous_handler = signal.signal(signal.SIGINT, on_interrupt)
        interrupted_encoder = False
        try:
            with recording:
                signal_pump.start()
                worker_thread.start()
                plt.show()
        except subprocess.CalledProcessError:
            # An interrupt reaches the whole process group, so ffmpeg takes it
            # too. It finalizes the file on its way out but exits non-zero, and
            # the writer reports that as a failure over a recording that is
            # actually playable (measured: 17 frames written). Only tolerated
            # when WE were interrupted; any other encoder failure still raises.
            if not state.get("interrupted"):
                raise
            interrupted_encoder = True
        finally:
            signal_pump.stop()
            signal.signal(signal.SIGINT, previous_handler)
        if state.get("interrupted"):
            print("interrupted — stopped after "
                  f"{len(self.hist['F'])} frames", flush=True)
        if writer is not None:
            written = os.path.getsize(save) if os.path.exists(save) else 0
            if interrupted_encoder and not written:
                print(f"interrupted before anything could be written to {save}")
            else:
                print(f"saved animation -> {save} ({written} bytes)")
        if state["error"] is not None:
            raise state["error"]


def build_proton_nodes(seed=3, precone=0, precone_timelike=False, gamma=50.0,
                       balanced_edges=False, singular_value_ratio=False,
                       degree=3, einstein_hilbert=True):
    """The single one-step `MultiCobordism` node the animation drives, as a 1-element
    list: `Proton.direct_node` — the three bare quarks `{1}`, `{ω}`, `{ω²}` plus their
    three anti-quarks (three q-q̄ pairs) as inputs and the proton singlet as the single
    output, read off the WHOLE cobordism — the *same* node setup `Proton.build_direct()`
    uses, on a single-Δ⁴ seed (attempt-0 seed = `seed`). The animation reports the live
    verdict either way.

    `precone` pre-grows the single-Δ⁴ seed by that many gated cone-in moves before
    optimization (forwarded straight to the C++ `MultiCobordism` constructor via `Proton`);
    `precone=0` (the default) leaves the bare seed untouched. `gamma` is the r_U
    weight in F (the engine default 50)."""
    p = cob.Proton(seed=seed, precone=precone, precone_timelike=precone_timelike,
                   gamma=gamma, balanced_edges=balanced_edges,
                   singular_value_ratio=singular_value_ratio,
                   register_degree=degree, einstein_hilbert=einstein_hilbert)
    return [
        (p.direct_node(seed), "Proton — 3 q-q̄ pairs → singlet {1, ω, ω²} (one step)"),
    ]


def animate(nodes, save=None, interval=200, dump_dir=None, visualize=False, **kw):
    """Animate the proton node sequence.

    Three combinations, all supported:

    * `save` alone → write a GIF/MP4 headlessly (Agg) with the synchronous
      per-frame `update`.
    * `visualize` alone → a live interactive window driven by `_run_live`
      (compute on a background thread, GUI thread only paints) so the window
      stays responsive through the multi-second surgery frames.
    * BOTH → the live window, recording each frame as it is painted. The run
      cannot be replayed to record it afterwards: every frame commits moves and
      relaxation into the complex, so the recording has to capture the frames
      the window is already drawing.

    `dump_dir` additionally writes each frame's dual-curvature panels as JSON
    (see `ProtonAnimator._dump_frame`). Returns the animator."""
    import matplotlib
    if save and not visualize:
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
    if save and not visualize:
        fa = FuncAnimation(anim_state.fig, anim_state.update,
                           frames=anim_state._frames, interval=interval,
                           repeat=False, blit=False)
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        fa.save(save, writer=writer, dpi=90)
        print(f"saved animation -> {save}")
        anim_state._anim = fa  # keep a ref so it isn't GC'd
    else:
        # `save` here means "record the window as it plays" (see the docstring).
        anim_state._run_live(plt, interval, save=save)
    return anim_state


def run_build(nodes, visualize=False, save=None, degree=3, init_steps=_INIT_STEPS,
              no_combinatorial_moves=False, relax_chunk=None, status=True,
              checkpoint=0, checkpoint_dir=None,
              evolve_steps=_EVOLVE_STEPS,
              stage1_candidates=_STAGE1_CANDIDATES,
              max_lookahead_depth=_MAX_LOOKAHEAD_DEPTH,
              max_lookahead_tries=_MAX_LOOKAHEAD_TRIES,
              stage2_alpha0=0.05, stage2_rel_tol=10e-9,
              stage2_beta=1.0, interval=200,  # interval: ms/frame; GIF/MP4 fps = 1000/interval
              relax_budget=10,
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
        chunked = bool(no_combinatorial_moves)
        for node, label in nodes:
            if chunked:
                # A chunked headless drive, so `--no-combinatorial-moves` and
                # `--relax-chunk` mean the same thing with and without a
                # window, and the status line is available either way. The
                # DEFAULT headless path below is left exactly as it was.
                chunk = int(relax_chunk) if relax_chunk else int(relax_budget)
                started = time.time()
                for phase, steps in (("init", init_steps), ("evolve", evolve_steps)):
                    for step_index in range(steps):
                        if no_combinatorial_moves:
                            trace = node.run_stage2(
                                beta=stage2_beta, max_iters=chunk,
                                alpha0=stage2_alpha0, rel_tol=stage2_rel_tol)
                            accepted = max(len(trace) - 1, 0)
                        else:
                            node.run(max_iters=1, n_candidate_moves=stage1_candidates,
                                     grow_boundaries=(phase == "init"),
                                     beta=stage2_beta, alpha0=stage2_alpha0,
                                     rel_tol=stage2_rel_tol,
                                     max_lookahead=max_lookahead_depth,
                                     relax_budget_per_move=relax_budget)
                            accepted = None
                        if status:
                            st_now = node.st
                            stage1_note = ("stage1 off (--no-combinatorial-moves)" if no_combinatorial_moves
                                           else (f"stage1 committed at depth "
                                                 f"{int(node.last_stage1_lookahead)}"
                                                 if int(node.last_stage1_lookahead) > 0
                                                 else "stage1 STUCK: no candidate lowered F"))
                            stage2_note = ("stage2 STATIONARY (no descending step found)"
                                           if node.last_stage2_stationary
                                           else (f"stage2 {accepted} steps accepted"
                                                 if accepted is not None
                                                 else "stage2 descending"))
                            print(f"{phase:<6} {step_index + 1:04d} | "
                                  f"F {float(node.objective()):.6e} | "
                                  f"cells {len(st_now.getTopSimplices())} | "
                                  f"{stage1_note} | {stage2_note} | "
                                  f"{time.time() - started:.0f}s", flush=True)
                        if checkpoint and (step_index + 1) % checkpoint == 0:
                            directory = checkpoint_dir or dump_dir or "."
                            os.makedirs(directory, exist_ok=True)
                            path = os.path.join(
                                directory, f"state_{phase}_{step_index + 1:04d}.json")
                            GeometryState.write(node.st, path, meta={
                                "frame": step_index + 1, "phase": phase,
                                "F": float(node.objective()), "degree": degree})
                            if status:
                                print(f"  checkpoint -> {path}", flush=True)
                        if no_combinatorial_moves and node.last_stage2_stationary:
                            break   # nothing left to relax and nothing to reopen it
                st = node.st
                out.append((label, {
                    "F": float(node.objective()),
                    "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
                    "rU": float(node.r_u(st)),
                    "b3": int(cob.MultiCobordism.betti(st)[degree]),
                    "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}))
                continue
            # The batched path runs each pass in ONE `run` call, so the engine's
            # own internal retry loop already spans every iteration — the
            # per-frame `max_lookahead_tries` retry has no meaning here and is
            # deliberately not applied.
            node.run(max_iters=init_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=True, beta=stage2_beta,
                     alpha0=stage2_alpha0, rel_tol=stage2_rel_tol,
                     max_lookahead=max_lookahead_depth,
                     relax_budget_per_move=relax_budget)
            node.run(max_iters=evolve_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=False, beta=stage2_beta,
                     alpha0=stage2_alpha0, rel_tol=stage2_rel_tol,
                     max_lookahead=max_lookahead_depth,
                     relax_budget_per_move=relax_budget)
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
    return animate(nodes, save=save, visualize=visualize, degree=degree,
                   init_steps=init_steps,
                   evolve_steps=evolve_steps,
                   stage1_candidates=stage1_candidates,
                   max_lookahead_depth=max_lookahead_depth,
                   max_lookahead_tries=max_lookahead_tries,
                   stage2_beta=stage2_beta, stage2_alpha0=stage2_alpha0,
                   stage2_rel_tol=stage2_rel_tol, relax_budget=relax_budget,
                   no_combinatorial_moves=no_combinatorial_moves, relax_chunk=relax_chunk, status=status,
                   checkpoint=checkpoint, checkpoint_dir=checkpoint_dir,
                   interval=interval,
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
    ap.add_argument("--gamma", type=float, default=50.0,
                    help="weight on r_U in F (engine default 50). The r_U VALUE "
                         "is dominated by flat step residuals; the movable "
                         "register-seeking signal is the near-kernel term, "
                         "measured ~5e-4 on ~90-tet hosts — so at 50 it loses "
                         "candidate auctions to ||grad S||^2 harvests (~0.1-1). "
                         "gamma ~ 5e3 makes the register channel competitive at "
                         "that scale; the term is O(1) at the bare seed, so "
                         "large gamma also makes early growth strongly "
                         "register-driven")
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
    ap.add_argument("--relax-budget", type=int, default=10, dest="relax_budget",
                    help="cap on stage-2 relaxation updates after each "
                         "committed move (and on the tight exit re-check). The "
                         "stationarity test at --rel-tol is the real "
                         "terminator; this only bounds slow descent tails of "
                         "threshold-sized line-search micro-steps")
    ap.add_argument("--init-chunk", type=int, default=_INIT_CHUNK, dest="init_chunk",
                    help="init-pass run iterations per animation frame")
    ap.add_argument("--balanced-edges", action="store_true", dest="balanced_edges",
                    help="wire EVERY new edge (seed, precone, and combinatorial "
                         "moves) with equal real and imaginary length components "
                         "at the same per-class magnitude — l = sqrt(a/2)*(1+i) "
                         "same-time, sqrt(alpha*a/2)*(1+i) cross-slice, so "
                         "Re l^2 = 0 exactly: every edge is born causally "
                         "undecided ON the null locus and stage 2 must choose "
                         "its character")
    ap.add_argument("--singular-value-ratio", action="store_true",
                    dest="singular_value_ratio",
                    help="score r_U's whole-complex term as the scale-invariant "
                         "ratio of the lower-half sum of the singular values of "
                         "the metric L_k to the upper-half sum (range [0, 1], "
                         "approaches 0 as the lower spectrum collapses), in "
                         "place of the singlet period residual + near-kernel "
                         "pair; the input-block residuals still anchor the "
                         "quark inputs and the singlet stays the read-out "
                         "verdict")
    ap.add_argument("--no-combinatorial-moves", action="store_true",
                    dest="no_combinatorial_moves",
                    help="drive ONLY the geometric relaxation: no combinatorial "
                         "moves of any kind — no Pachner moves (add, remove, "
                         "flip, inverse flip), no surgical cone moves, no "
                         "disposition flips — and no growth of the blocks' "
                         "scoring regions. The triangulation is fixed for the "
                         "whole run and F descends within one region of "
                         "configuration space")
    ap.add_argument("--degree", type=int, default=3, dest="degree",
                    help="the register degree k the residuals target — which "
                         "Hodge Laplacian L_k's eigenvalues r_U minimizes. "
                         "Reaches both the objective (Proton's register "
                         "degree) and the readout panels (Betti, holes, "
                         "sigma, modes, Krein). Default 3 (L_3 on a "
                         "4-manifold)")
    ap.add_argument("--relax-chunk", type=int, default=None, dest="relax_chunk",
                    help="stage-2 iterations one frame advances in the "
                         "relaxation-only drive, so a descent can be watched "
                         "instead of finishing inside a single frame. Requires "
                         "--no-combinatorial-moves: in the interleaved drive a "
                         "frame is one move plus its relaxation, and the amount "
                         "of that relaxation is --relax-budget. Default: "
                         "--relax-budget")
    ap.add_argument("--no-einstein-hilbert", action="store_false",
                    dest="einstein_hilbert",
                    help="drop the discrete Einstein-Hilbert term from the "
                         "objective, optimizing F = gamma*r_U alone instead of "
                         "||grad S_Regge||^2 + gamma*r_U. NOTE stage 2 builds "
                         "its descent direction from the Regge gradient and "
                         "Hessian only, so without that term it searches along "
                         "a ray the objective no longer contains: the run stays "
                         "monotone (the line search still accepts only trials "
                         "that lower the true F) but accepts far fewer steps, "
                         "and the combinatorial moves do most of the work")
    ap.add_argument("--checkpoint", type=int, default=0, metavar="STEPS",
                    help="every STEPS frames, write the complex's state to "
                         "state_<frame>.json: top cells in INTRINSIC vertex "
                         "order plus every edge interval and vertex time, which "
                         "is enough to rebuild the exact state and to read its "
                         "orientation back (inspect_state.py). 0 (default) "
                         "writes none. The per-frame --dump-dir record cannot "
                         "serve this: it sorts each cell's vertices for drawing")
    ap.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir",
                    help="where to write state files (default: --dump-dir, "
                         "else the working directory)")
    ap.add_argument("--no-status", action="store_false", dest="status",
                    help="silence the per-frame status line (F, its change, "
                         "the objective's two terms, cell/Betti/hole counts, "
                         "and the stage-1/stage-2 obstruction signals)")
    ap.add_argument("--spectra-every", type=int, default=_SPECTRA_REFRESH_EVERY,
                    dest="spectra_every",
                    help="recompute the O(n^3) spectrum/mode/Krein panels at "
                         "least every N frames (commits and node switches "
                         "always refresh; 1 = the historical every-frame "
                         "behaviour)")
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
    if args.relax_chunk and not args.no_combinatorial_moves:
        ap.error("--relax-chunk applies to the relaxation-only drive; pass "
                 "--no-combinatorial-moves with it. In the interleaved drive a "
                 "frame is one move plus its relaxation, and that relaxation is "
                 "sized by --relax-budget.")
    nodes = build_proton_nodes(seed=args.seed, precone=args.precone,
                               precone_timelike=args.precone_timelike,
                               gamma=args.gamma,
                               balanced_edges=args.balanced_edges,
                               singular_value_ratio=args.singular_value_ratio,
                               degree=args.degree,
                               einstein_hilbert=args.einstein_hilbert)
    result = run_build(nodes, visualize=args.live, save=args.save, init_steps=args.init,
                       evolve_steps=args.evolve,
                       init_chunk=args.init_chunk, evolve_chunk=args.evolve_chunk,
                       stage1_candidates=args.candidates, stage2_beta=args.beta,
                       stage2_alpha0=args.alpha0, stage2_rel_tol=args.rel_tol,
                       relax_budget=args.relax_budget,
                       max_lookahead_depth=args.max_lookahead_depth,
                       max_lookahead_tries=args.max_lookahead_tries,
                       spectra_every=args.spectra_every,
                       degree=args.degree, checkpoint=args.checkpoint,
                       checkpoint_dir=args.checkpoint_dir,
                       no_combinatorial_moves=args.no_combinatorial_moves,
                       relax_chunk=args.relax_chunk, status=args.status,
                       dump_dir=args.dump_dir)
    if not args.live and not args.save:
        print("one-step proton build finished (visualization off by default — pass --live "
              "or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
