# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of the canonical two-step **Proton** build (#522, #526).

Drives the actual `tessera.cobordism.Proton` class and animates the proton assembling in
its two physical steps, each growing its whole topology from a single Δ⁴ simplex through
stage 1's F-lowering candidate draw —

  * **Step A — recombination** (`Proton.recombination_node`): two neutral q-q̄ pairs ⟶
    a colored diquark `{1, ω}` ⊔ anti-diquark `{1, ω²}`;
  * **Step B — formation** (`Proton.formation_node`): the diquark `{1, ω}` + the third
    quark `{ω²}` ⟶ the colorless proton singlet `{1, ω, ω²}` (ω = exp(2πi/3)).

**Both steps are shown at one time.** The figure is a 2×2 grid whose *bottom row holds two
complex panels — Step A on the left, Step B on the right — both visible the whole run.*
Step A animates first (Step B waiting on its seed simplex); then Step B animates while Step
A holds its finished complex, so at the end you see the full diquark next to the full
proton, side by side.

Each node is driven with the COMBINED `run` drive: an **init pass** (`grow_boundaries=True`)
that grows the color register, then an **evolution pass** (`grow_boundaries=False`) with the
boundary frozen. Every `run` iteration interleaves the stage-1 combinatorial update with the
stage-2 geometric relaxation, so the optimizer makes whichever kind of progress helps at each
point — no separate relaxation pass. The animation advances ONE `run` iteration per frame:
stage 1 keeps no state across iterations (the trap-door burst machinery is gone), so
splitting a pass into per-frame calls is exact — every accepted move and relaxation step
gets its own frame.

The figure is a 2×4 grid, one **step per row**: traces, the primal complex, then the dual
split into spatial- and temporal-curvature panels:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U` vs step (a dashed line marks the Step A → Step B boundary);
  * **color register** — the color-register (hole = quark) count and, separately, the Betti
    number `b_k` vs step (the proton's three registers appear as Step B grows);
  * **complex — Step A / Step B** — 2-D classical-MDS projections of each node's relaxing
    1-skeleton; each emergent color hole (register) is outlined in red as a cell and numbered.
    Each frame is normalized to a fixed scale, Procrustes-aligned (rotation/reflection only)
    to the previous frame, and eased, with the view auto-fit — so the structure stays legible
    instead of collapsing into a dot.
  * **dual — spatial / temporal curvature — Step A / Step B** — the circumcentric dual graph
    (one node per top cell, edges across shared facets) at the primal cell centroids, colored
    by the local Regge curvature. The Lorentzian deficit ε is COMPLEX, so it splits into two
    panels: **spatial** = `Re ε·|★|` (the rotation angle-defect, from timelike hinges) and
    **temporal** = `Im ε·|★|` (the boost / light-cone content, from spacelike hinges — those
    whose normal plane is timelike). Both use a signed diverging colormap centered at 0.

The figure title reports the live **convergence verdict**: whether Step B's whole cobordism
carries the singlet `{1, ω, ω²}` (color residual `r_state` ≈ 0) on its ≥ 3 emergent color
holes.

It drives only the **public** `Proton` (`recombination_node`/`formation_node`),
`MultiCobordism` (the combined `run`, plus `betti`, `emergent_holes`,
`regge_action_gradient`, `r_state`, `r_u`, `objective`, `st`), and the geometry readers
(`Spacetime.getTopSimplices`/`getDualAdjacency`/`getSimplices`,
`Simplex.deficitAngle`/`dualVolume`) APIs — the *same* node setups
and drive `Proton.build()` uses, so the animation shows the real class. The C++ engine is
untouched.

**Visualization is off by default** — `run_build(...)` takes the fast batched path with no
per-step plotting overhead. Opt in with `visualize=True` (or `--live`/`--save`) to animate,
which is slower.

    # default: run the two-step build fast, no visualization
    python multicobordism_animation.py
    # live (interactive backend):
    python multicobordism_animation.py --live
    # headless: write a GIF (no display needed):
    python multicobordism_animation.py --save proton.gif
    # pre-grow each node's single-Δ⁴ seed by 12 gated cone-ins before optimizing:
    python multicobordism_animation.py --precone 12

RL-driven variant (`--rl`): the SAME two-step build and the SAME 2×4 charts, but each step is
driven by a trained libtorch **RL policy** instead of the fixed `Proton.build()` schedule — the
policy chooses each `buildStep` macro-move (GROW/EVOLVE/RELAX + intensity). Two policies are
trained, ONE PER STEP, because Step A recombination (success = `r_U → 0`) and Step B formation
(carry the singlet) are different RL targets; within a step, the single policy spans both the
combinatorial (GROW/EVOLVE surgery) and geometric (RELAX) moves. Policies are cached in
`--policy-dir` (default `/tmp`) — trained once (over many complete convergences to learn the
search space, not one problem), then reused. `--train` forces a retrain.

    # RL two-step build; trains + caches both policies on first run, reuses them after:
    python multicobordism_animation.py --rl --live
    # retrain thoroughly (many convergences over the space) then watch it:
    python multicobordism_animation.py --train --train-iters 40 --episodes 8 --live
    # robuster policy: keep the best of 4 independent trainings per step:
    python multicobordism_animation.py --train --best-of 4 --save proton_rl.gif
"""
import argparse
import itertools
import math
import os
import shutil

import numpy as np
from scipy.sparse.csgraph import shortest_path

import tessera

cob = tessera.cobordism

# Two combined-`run` passes per node — init (grow_boundaries=True) then evolution
# (grow_boundaries=False) — each interleaving the stage-1 surgery update with the stage-2
# geometric relaxation every iteration, so the optimizer makes whichever kind of progress
# helps at each point. The animation runs ONE iteration per frame (`_*_CHUNK = 1`): stage 1
# keeps no state across iterations, so per-frame chunking is exact and every accepted move
# and relaxation step is visible. (The batched no-visualization path still runs each pass
# as one call — same result, no per-frame overhead.) These totals are sized so Step B
# reliably grows its three quark holes and carries the singlet.
_INIT_STEPS = 180          # init-pass (grow_boundaries=True) iterations per node
_EVOLVE_STEPS = 60         # evolution-pass (grow_boundaries=False) iterations per node
_INIT_CHUNK = 1            # init iterations per frame (1 = smoothest animation)
_EVOLVE_CHUNK = 1          # evolution iterations per frame
_STAGE1_CANDIDATES = 8
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
        w = math.sqrt(max(abs((e.getLength()**2).real), 1e-6))
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


class ProtonAnimator:
    """Animates the proton's two steps — Step A (recombination) then Step B (formation) —
    with BOTH steps' complexes on screen at one time.

    Each node is driven with the combined `run` drive: an init pass
    (`grow_boundaries=True`, grows the color register) and an evolution pass
    (`grow_boundaries=False`), each interleaving the stage-1 surgery update with the
    stage-2 geometric relaxation every iteration, advanced one iteration per frame by
    default. The two complex panels are bound one each to Step A and Step B for the whole
    run, so Step A's finished diquark stays visible beside the proton as Step B forms."""

    _PHASE_NAMES = {"init": "growing register", "evolve": "evolving (∂W frozen)"}
    _TITLE_PREFIX = "Proton build (two-step)"

    def __init__(self, nodes, degree=3, init_steps=_INIT_STEPS, init_chunk=_INIT_CHUNK,
                 evolve_steps=_EVOLVE_STEPS, evolve_chunk=_EVOLVE_CHUNK,
                 stage1_candidates=_STAGE1_CANDIDATES, stage2_beta=1.0):
        self._common_init(nodes, degree)
        self.s1c, self.s2_beta = stage1_candidates, stage2_beta
        self._schedule = self._make_schedule(len(nodes), init_steps, init_chunk,
                                             evolve_steps, evolve_chunk)
        self._frames = len(self._schedule)

    def _common_init(self, nodes, degree):
        """Shared drawing/recording state used by BOTH the fixed-schedule drive and the
        RL-policy drive (`RLPolicyAnimator`). Sets the node list, history buffers, per-panel
        layouts, and curvature cache — everything `_redraw`/`_draw_*`/`verdict` read — so the
        two drives differ only in how they advance a node (fixed passes vs one policy action),
        never in what is drawn."""
        self.nodes = nodes                  # [(MultiCobordism, "Step A: ..."), ...] in order
        self.k = degree
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [],
                     "phase": [], "node": []}
        self._boundaries = []       # step indices where a later node begins (trace markers)
        self._layouts = [_StableLayout() for _ in nodes]   # one per complex panel
        self._active = 0            # index of the node currently being driven
        self._done = False          # so the verdict is announced exactly once
        self._curv_cache = {}       # node_index -> (frame_computed, {cell_tuple: curvature})

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
        node.run(max_iters=count, n_candidate_moves=self.s1c,
                 grow_boundaries=(phase == "init"), beta=self.s2_beta)
        self._record(node, node_index, phase)

    def _record(self, node, node_index, phase):
        st = node.st
        self.hist["F"].append(float(node.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.regge_action_gradient(st)))
        self.hist["rU"].append(float(node.r_u(st)))
        self.hist["b3"].append(int(cob.MultiCobordism.betti(st)[self.k]))
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["phase"].append(phase)
        self.hist["node"].append(node_index)

    # ---- convergence ----
    def verdict(self):
        """The honest, live convergence verdict read off Step B's *current* whole cobordism:
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
        n_rows = max(2, n_nodes)
        self.fig, axes = plt.subplots(n_rows, 4, figsize=(21, 4.5 * n_rows),
                                      squeeze=False)
        self.axm = axes[0][0]                                # metrics trace
        self.axr = axes[1][0]                                # register trace
        for row in range(2, n_rows):
            axes[row][0].axis("off")
        self._primal_axes = [axes[i][1] for i in range(n_nodes)]
        self._re_axes = [axes[i][2] for i in range(n_nodes)]
        self._im_axes = [axes[i][3] for i in range(n_nodes)]
        for row in range(n_nodes, n_rows):                   # single node: blank row 1 panels
            for column in (1, 2, 3):
                axes[row][column].axis("off")
        # Persistent colorbars (created ONCE — recreating per frame piles them up). Each dual
        # panel self-normalizes per frame; we just update the mappable's clim. Re (spatial) and
        # Im (temporal) use distinct diverging colormaps so the two channels read apart.
        self._primal_sms, self._re_sms, self._im_sms = [], [], []
        for axset, sms, cmap, label in (
                (self._primal_axes, self._primal_sms, _HEAT_CMAP,
                 "hinge curvature  Re ε·|★|"),
                (self._re_axes, self._re_sms, _HEAT_CMAP, "spatial curvature  Re ε·|★|"),
                (self._im_axes, self._im_sms, _HEAT_CMAP_IM, "temporal curvature  Im ε·|★|")):
            for ax in axset:
                sm = ScalarMappable(cmap=cmap, norm=Normalize(-1.0, 1.0))
                cbar = self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=7)
                cbar.ax.tick_params(labelsize=6)
                sms.append(sm)
        return self.fig

    def _draw_complex(self, ax, sm, node_index, coords, title):
        node, _label = self.nodes[node_index]
        st = node.st
        ax.clear()
        # Curvature heat on the PRIMAL complex, mirroring the dual panels: in 4D
        # the hinges are exactly the triangles, so each projected hinge triangle
        # is filled with its own spatial curvature Re ε·|★| (signed, diverging
        # map, symmetric range) under the wireframe. The temporal channel stays
        # on the dual panels.
        hinge_re, _hinge_im = self._hinge_curvature_cached(node_index, st)
        tris, vals = [], []
        for tri, re_val in hinge_re.items():
            if all(v in coords for v in tri):
                tris.append([coords[v] for v in tri])
                vals.append(re_val)
        if tris:
            from matplotlib.collections import PolyCollection
            vmax = max(max(abs(v) for v in vals), 1e-12)
            from matplotlib.colors import Normalize
            norm = Normalize(-vmax, vmax)
            pc = PolyCollection(tris, array=np.array(vals), cmap=_HEAT_CMAP,
                                norm=norm, alpha=0.5, edgecolors="none",
                                zorder=0.5)
            ax.add_collection(pc)
            sm.set_clim(-vmax, vmax)
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
    def _hinge_curvature(st):
        """Per-HINGE curvature, BOTH channels of the COMPLEX Lorentzian deficit:
        `Re(deficit)·|★|` — the spatial angle-defect (rotation) curvature — and
        `Im(deficit)·|★|` — the temporal boost / light-cone content. Both SIGNED
        (ε<0 = saddle; Im sign = boost direction). Returns {tri-tuple: (re, im)};
        in 4D the hinges are exactly the triangles, so this is also what the
        primal panels paint directly."""
        hinge_re, hinge_im = {}, {}
        for s in st.getSimplices():
            vs = s.getVertices()
            if len(vs) != 3:                     # hinges = (d-2) = 2-simplices (triangles)
                continue
            key = tuple(sorted(v.getId() for v in vs))
            try:
                deficit = complex(s.deficitAngle())
                # complex-tolerant positive dual-measure weight (dualVolume
                # is complex; the heat weight is its magnitude)
                weight = abs(complex(s.dualVolume()))
                hinge_re[key] = deficit.real * weight
                hinge_im[key] = deficit.imag * weight
            except RuntimeError:                 # boundary/degenerate hinge → no curvature
                # Only the geometric failure is swallowed; a type/contract
                # failure (TypeError, ValueError) must propagate, never
                # render as zero curvature.
                hinge_re[key] = hinge_im[key] = 0.0
        return hinge_re, hinge_im

    @classmethod
    def _cell_curvature(cls, st):
        """Per-top-cell curvature: the per-hinge channels of `_hinge_curvature`
        summed over each cell's triangles. Returns {cell-tuple: (re_sum, im_sum)}."""
        hinge_re, hinge_im = cls._hinge_curvature(st)
        curv = {}
        for c in st.getTopSimplices():
            cell = tuple(sorted(v.getId() for v in c.getVertices()))
            tris = [tuple(sorted(t)) for t in itertools.combinations(cell, 3)]
            curv[cell] = (sum(hinge_re.get(t, 0.0) for t in tris),
                          sum(hinge_im.get(t, 0.0) for t in tris))
        return curv, (hinge_re, hinge_im)

    def _cell_curvature_cached(self, node_index, st):
        """`_cell_curvature` is expensive, so recompute it only every `_HEAT_REFRESH_EVERY`
        frames on the active (changing) node, and always on the final frame; the frozen
        node's geometry doesn't change, so its last value is reused."""
        frame = len(self.hist["F"])
        cached = self._curv_cache.get(node_index)
        stale = (node_index == self._active
                 and frame - cached[0] >= _HEAT_REFRESH_EVERY) if cached else True
        if cached is None or stale or frame >= self._frames:
            cellcurv, hingecurv = self._cell_curvature(st)
            self._curv_cache[node_index] = (frame, cellcurv, hingecurv)
        return self._curv_cache[node_index][1]

    def _hinge_curvature_cached(self, node_index, st):
        """The per-hinge channels behind `_cell_curvature_cached`, same cache entry."""
        self._cell_curvature_cached(node_index, st)
        return self._curv_cache[node_index][2]

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

    def _redraw(self):
        xs = range(len(self.hist["F"]))
        self.axm.clear()
        self.axm.plot(xs, self.hist["F"], label="F (objective)", color="C0")
        self.axm.plot(xs, self.hist["gradN2"], label="‖∇S‖²", color="C1")
        self.axm.plot(xs, self.hist["rU"], label="r_U", color="C2")
        for b in self._boundaries:
            self.axm.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
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
        for ni in range(len(self.nodes)):
            coords = self._layouts[ni].coords(self.nodes[ni][0].st)
            self._draw_complex(self._primal_axes[ni], self._primal_sms[ni],
                               ni, coords, self.nodes[ni][1])
            self._draw_dual(self._re_axes[ni], self._re_sms[ni], ni, coords,
                            0, _HEAT_CMAP, "dual — spatial curvature (Re ε)")
            self._draw_dual(self._im_axes[ni], self._im_sms[ni], ni, coords,
                            1, _HEAT_CMAP_IM, "dual — temporal curvature (Im ε)")

    # ---- per-frame text hooks (overridden by the RL drive) ----
    def _frame_label(self, frame):
        """The short 'what step / what's running' label for a frame — the node label plus the
        current phase name. The RL drive overrides this to report the policy's last macro-move."""
        node_index, phase, _count = self._schedule[frame]
        return f"{self.nodes[node_index][1]} · {self._PHASE_NAMES[phase]}"

    def _verdict_tag(self, ok, res, holes):
        return (f"CONVERGED ✓ — proton {{1,ω,ω²}} carried (r_state={res:.2g}, "
                f"{holes} registers)" if ok else
                f"did NOT converge (r_state={res:.2g}, {holes} registers)")

    def _draw_extras(self):
        """Hook for per-frame figure annotations drawn after `_redraw` (the RL drive adds its
        training-parameter footnote here); the fixed drive has none."""

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

        def close_after_this_tick():
            """Tear the window down on a LATER event-loop turn, never from inside
            `on_timer`. `plt.close` fires the figure's close_event, which runs
            `Animation._stop()` and sets `event_source = None` — and `on_timer` is called
            by `TimedAnimation._step`, which touches `self.event_source` again the moment
            it returns. Closing in-callback therefore raises `AttributeError` inside the
            GUI toolkit's timer slot, and PyQt aborts the process on an unhandled
            exception in a slot: the whole run dies with a core dump instead of the
            compute error. A one-shot timer runs the teardown after `_step` has finished."""
            self._anim.event_source.stop()
            closer = self.fig.canvas.new_timer(interval=1)
            closer.single_shot = True
            closer.add_callback(lambda: plt.close(self.fig))
            self._closer = closer               # keep a ref so it isn't GC'd before it fires
            closer.start()

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
                else:                            # error: close so plt.show() returns
                    close_after_this_tick()
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


def build_proton_nodes(seed=3, precone=0):
    """The two `MultiCobordism` nodes the `Proton` class drives, in build order, for the
    animation: Step A recombination then Step B formation, each on its own single-Δ⁴ seed.

    Built via `Proton.recombination_node`/`formation_node` — the *same* node setups
    `Proton.build()` uses — with `Proton.build`'s attempt-0 seeds (A = `seed`, B =
    `seed + 1`). The default `seed=3` converges on attempt 0 (Step B grows three quark holes
    and carries the singlet); the animation reports the live verdict either way.

    `precone` pre-grows each node's single-Δ⁴ seed by that many gated cone-in moves before
    optimization (forwarded straight to the C++ `MultiCobordism` constructor via `Proton`);
    `precone=0` (the default) leaves the bare seed untouched."""
    p = cob.Proton(seed=seed, precone=precone)
    return [
        (p.recombination_node(seed), "Step A — recombination (→ diquark {1, ω})"),
        (p.formation_node(seed + 1), "Step B — formation (→ proton {1, ω, ω²})"),
    ]


def animate(nodes, save=None, interval=200, **kw):
    """Animate the proton node sequence. `save` → write a GIF/MP4 (headless, Agg) with the
    synchronous per-frame `update`; otherwise show a live interactive window driven by
    `_run_live` (compute on a background thread, GUI thread only paints) so the window stays
    responsive through the multi-second surgery frames. Returns the animator."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim_state = ProtonAnimator(nodes, **kw)
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
              stage2_beta=1.0, interval=200,  # interval: ms/frame; GIF/MP4 fps = 1000/interval
              **anim_kw):
    """Run the two-step proton build over `nodes` with the combined `run` drive: an init
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
            node.run(max_iters=init_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=True, beta=stage2_beta)
            node.run(max_iters=evolve_steps, n_candidate_moves=stage1_candidates,
                     grow_boundaries=False, beta=stage2_beta)
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
                   stage2_beta=stage2_beta, interval=interval, **anim_kw).hist


# ---- RL-policy drive (#553): watch trained libtorch policies assemble the proton ----
# The SAME rich 2×4 panels as the fixed Proton drive above (metrics, register+Betti, primal
# A/B, dual spatial/temporal curvature) — only the DRIVE changes: each active frame the step's
# trained policy chooses one `buildStep` macro-action and the env drives the CANONICAL
# `MultiCobordism` (nothing about proton construction is reimplemented here).
#
# TWO policies, one per physics step, because the two steps are DIFFERENT RL problems: Step A
# recombination has no whole-cobordism target (success = r_U → 0) while Step B formation must
# carry the proton singlet {1, ω, ω²} — a checkpoint trained for one does NOT transfer to the
# other. WITHIN a step, a SINGLE policy spans both the combinatorial moves (GROW/EVOLVE =
# stage-1 surgery) and the geometric move (RELAX = stage-2 relaxation); it learns to interleave
# them, all scored by the one objective F = ‖∇S‖² + Γ·r_U.
_RL_MOVE_NAMES = {0: "GROW", 1: "EVOLVE", 2: "RELAX"}
_OBS_DIM, _N_MOVES, _PARAM_DIM = 17, 3, 2   # the env's fixed (obs, move, param) dims

# The two physics steps as RL problems: (cache name, formation flag, env-builder, label).
_RL_STEPS = [
    ("recombination", False, "make_recombination_env",
     "Step A — recombination (→ diquark {1, ω})"),
    ("formation", True, "make_formation_env",
     "Step B — formation (→ proton {1, ω, ω²})"),
]


def _policy_cache_path(cache_dir, name):
    return os.path.join(cache_dir, f"tessera_proton_policy_{name}.pt")


def _train_step_policy(name, formation, cache_dir, train_iters, episodes, best_of,
                       hidden=64, verbose=True):
    """Train an RL policy for ONE physics step and cache it to `cache_dir`. Training runs many
    COMPLETE convergences over a rotating set of seeds and is scored on HELD-OUT seeds, so the
    policy learns the search SPACE (generalizes) rather than one convergence instance. With
    `best_of > 1`, train that many independent policies (different init seeds) and keep the one
    with the highest held-out carry rate. Returns (path, best_carry_rate, params-dict)."""
    import tessera.rl as rl
    env_cfg = rl.carry_profile_env()
    train_cfg = rl.carry_profile_train()
    train_cfg.iterations = train_iters
    train_cfg.episodes_per_iter = episodes
    train_cfg.hidden = hidden
    path = _policy_cache_path(cache_dir, name)
    convergences = train_iters * episodes
    if verbose:
        print(f"[train:{name}] {best_of}×({train_iters} iters × {episodes} eps) = "
              f"{best_of * convergences} complete convergences over rotating seeds "
              f"(learning the space; held-out eval) → {path}", flush=True)
    best_rate, best_tmp = -1.0, None
    for n in range(best_of):
        train_cfg.agent_seed = n
        tmp = f"{path}.cand{n}"
        res = rl.benchmark(env_cfg, train_cfg, formation=formation, checkpoint_path=tmp)
        rate = float(res.rl.carry_rate)
        if verbose:
            print(f"[train:{name}]   candidate {n + 1}/{best_of}: carry_rate={rate:.2f}  "
                  f"(random={res.random.carry_rate:.2f}, "
                  f"grow_only={res.grow_only.carry_rate:.2f}, {res.train_time_s:.0f}s)",
                  flush=True)
        if rate > best_rate:
            best_rate, best_tmp = rate, tmp
    shutil.copyfile(best_tmp, path)
    for n in range(best_of):                                    # clean up the candidate files
        try:
            os.remove(f"{path}.cand{n}")
        except OSError:
            pass
    return path, best_rate, {"iters": train_iters, "eps": episodes,
                             "convergences": best_of * convergences, "best_of": best_of,
                             "carry": round(best_rate, 2)}


def ensure_rl_steps(cache_dir="/tmp", retrain=False, train_iters=20, episodes=6, best_of=1,
                    hidden=64, verbose=True):
    """Return `[(env, policy, label), ...]` for BOTH steps in build order, training + caching a
    per-step policy in `cache_dir` when it is missing or `retrain` is set (otherwise the cached
    checkpoint is reused — train once). Also returns a `train_info` dict for on-figure display."""
    import tessera.rl as rl
    steps, info = [], {"iters": train_iters, "eps/iter": episodes, "best_of": best_of}
    for name, formation, builder, label in _RL_STEPS:
        path = _policy_cache_path(cache_dir, name)
        if retrain or not os.path.exists(path):
            _p, rate, _params = _train_step_policy(name, formation, cache_dir, train_iters,
                                                   episodes, best_of, hidden, verbose)
            info[f"{name} carry"] = round(rate, 2)
        else:
            if verbose:
                print(f"[cache] reusing {name} policy: {path}  (pass --train to retrain)",
                      flush=True)
            info[f"{name} carry"] = "cached"
        env = getattr(rl, builder)(rl.carry_profile_env())
        policy = rl.load_policy(path, _OBS_DIM, _N_MOVES, _PARAM_DIM, hidden)
        steps.append((env, policy, label))
    return steps, info


class RLPolicyAnimator(ProtonAnimator):
    """Animate the FULL two-step proton build — Step A recombination then Step B formation —
    driven by trained RL policies (one per step). Reuses `ProtonAnimator`'s rich 2×4 panels
    verbatim (metrics F/‖∇S‖²/r_U, register+Betti, primal A/B, dual spatial/temporal
    curvature); the ONLY difference is the drive — each active frame the step's policy chooses
    one `buildStep` macro-action (GROW/EVOLVE/RELAX + intensity) and the env drives the
    canonical `MultiCobordism`. A short glide of hold-frames follows each macro-action so the
    few large actions still read as a smooth build. The figure footnote shows the training
    parameters and the title shows the current step + the policy's last macro-move.

    `steps` is `[(env, policy, label), ...]` in build order; `train_info` is displayed."""

    _TITLE_PREFIX = "Proton build under RL policy (two-step)"

    def __init__(self, steps, degree=3, seed=3, deterministic=True, max_actions=3,
                 glide_frames=6, train_info=None):
        self.envs = [s[0] for s in steps]
        self.policies = [s[1] for s in steps]
        self.det = bool(deterministic)
        self.glide = max(1, glide_frames)
        self.max_actions = max_actions
        self.train_info = train_info or {}
        self._train_txt = None
        # Reset each env FIRST (that seeds its single-Δ⁴ node), then bind the node list the
        # shared drawing reads. Different seed per step so A and B don't share a trajectory.
        self._obs = [env.reset(seed + i) for i, env in enumerate(self.envs)]
        self._steps_taken = [0 for _ in self.envs]
        self._env_done = [False for _ in self.envs]
        self._last_move = [None for _ in self.envs]
        nodes = [(env.node, steps[i][2]) for i, env in enumerate(self.envs)]
        self._common_init(nodes, degree)
        self._schedule = self._make_rl_schedule(len(self.envs), max_actions, self.glide)
        self._frames = len(self._schedule)
        self._record(self.nodes[0][0], 0, "reset")             # initial metrics point

    @staticmethod
    def _make_rl_schedule(n_steps, max_actions, glide):
        """One `act` frame per macro-action, each followed by `glide-1` hold frames (no engine
        step — just the layout easing toward the new complex), for every step in build order."""
        sched = []
        for i in range(n_steps):
            for _ in range(max_actions):
                sched.append((i, "act"))
                sched += [(i, "hold")] * (glide - 1)
        return sched

    def _advance(self, frame):
        import tessera.rl as rl
        node_index, kind = self._schedule[frame]
        if node_index != self._active:                         # a new step begins
            self._active = node_index
            self._boundaries.append(len(self.hist["F"]))
        if kind != "act":                                      # hold frame: no engine step
            return
        if self._env_done[node_index] or self._steps_taken[node_index] >= self.max_actions:
            return
        env = self.envs[node_index]
        a = rl.select_action(self.policies[node_index], self._obs[node_index], self.det)
        res = env.step(rl.Move(a.move), list(a.params))        # drives the canonical buildStep
        self._obs[node_index] = res.obs
        self._env_done[node_index] = bool(res.done)
        self._last_move[node_index] = int(res.move)
        self._steps_taken[node_index] += 1
        self.nodes[node_index] = (env.node, self.nodes[node_index][1])   # keep the node fresh
        self._record(env.node, node_index, _RL_MOVE_NAMES.get(int(res.move), "?"))

    # The RL drive differs from the fixed drive only in these three text/annotation hooks; it
    # inherits `update` (for --save) and `_run_live` (responsive live window) unchanged, so the
    # RL animation gets the same GUI-freeze fix for free.
    def _frame_label(self, frame):
        node_index, _kind = self._schedule[frame]
        mv = self._last_move[node_index]
        return (f"{self.nodes[node_index][1]}  ·  RL move: "
                f"{_RL_MOVE_NAMES.get(mv, '—') if mv is not None else '—'}")

    def _verdict_tag(self, ok, res, holes):
        return (f"CARRIED ✓ — proton {{1,ω,ω²}} (r_state={res:.2g}, {holes} registers)"
                if ok else f"did NOT carry (r_state={res:.2g}, {holes} registers)")

    def _draw_extras(self):
        """Add the training-parameter footnote ONCE (a per-frame fig.text would pile up)."""
        if not self.train_info or self._train_txt is not None:
            return
        txt = "   ".join(f"{k}={v}" for k, v in self.train_info.items())
        self._train_txt = self.fig.text(
            0.5, 0.006, f"RL policy trained over many complete convergences —  {txt}",
            ha="center", va="bottom", fontsize=7, color="0.4")


def run_rl(cache_dir="/tmp", retrain=False, train_iters=20, episodes=6, best_of=1, seed=3,
           deterministic=True, visualize=False, save=None, degree=3, interval=200):
    """Drive the FULL two-step proton build (Step A recombination + Step B formation) with
    trained RL policies (cached per step in `cache_dir`, trained over many complete convergences
    when missing or `retrain`). Off by default it runs one episode per step and returns
    `(result, train_info)`; `--live`/`--save` animate the same rich 2×4 panels as the fixed
    Proton drive, annotated with the policy's macro-moves and the training parameters."""
    import tessera.rl as rl
    steps, train_info = ensure_rl_steps(cache_dir, retrain, train_iters, episodes, best_of,
                                        verbose=True)
    max_actions = rl.carry_profile_env().max_actions
    if not visualize and not save:
        out = []
        for i, (env, policy, label) in enumerate(steps):
            obs, done, n = env.reset(seed + i), False, 0
            while not done and n < max_actions:
                a = rl.select_action(policy, obs, deterministic)
                res = env.step(rl.Move(a.move), list(a.params))
                obs, done, n = res.obs, bool(res.done), n + 1
            st = env.node.st
            out.append((label, {
                "F": float(env.node.objective()),
                "rU": float(env.node.r_u(st)),
                "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}))
        st = steps[-1][0].node.st
        res_ = float(cob.MultiCobordism.r_state(st, degree, cob.Proton.singlet()))
        holes = len(cob.MultiCobordism.emergent_holes(st, degree))
        out.append(("verdict", {"converged": res_ < _COLOR_TOL and holes >= _MIN_QUARK_HOLES,
                                "color_residual": res_, "registers": holes}))
        return out, train_info
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    anim = RLPolicyAnimator(steps, degree=degree, seed=seed, deterministic=deterministic,
                            max_actions=max_actions, train_info=train_info)
    anim._setup(plt)
    anim.fig.suptitle(f"{anim._TITLE_PREFIX} — live")
    if save:                                  # off-screen Agg: synchronous per-frame render
        fa = FuncAnimation(anim.fig, anim.update, frames=anim._frames, interval=interval,
                           repeat=False, blit=False)
        fa.save(save, writer="pillow" if save.endswith(".gif") else "ffmpeg", dpi=90)
        print(f"saved animation -> {save}")
        anim._anim = fa   # keep a ref so it isn't GC'd
    else:                                     # responsive live window (compute off GUI thread)
        anim._run_live(plt, interval)
    return anim.verdict(), train_info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Visualization is OFF by default (fast). Opt in with --live or --save.
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    # ---- RL-driven two-step build (trained policies, one per step) ----
    ap.add_argument("--rl", action="store_true",
                    help="drive the FULL two-step build with trained RL policies (one per step, "
                         "cached in --policy-dir; trains over many convergences if missing) "
                         "instead of the fixed Proton drive — SAME charts as the fixed drive")
    ap.add_argument("--train", action="store_true",
                    help="(RL) force-retrain both step policies, overwriting the cache "
                         "(implies --rl)")
    ap.add_argument("--train-iters", type=int, default=20, dest="train_iters",
                    help="(RL) PPO iterations per step; × --episodes = complete convergences")
    ap.add_argument("--episodes", type=int, default=6,
                    help="(RL) complete-convergence episodes per iteration — seed diversity, so "
                         "the policy learns the space, not one convergence problem")
    ap.add_argument("--best-of", type=int, default=1, dest="best_of",
                    help="(RL) train N independent policies per step, keep the best carry rate")
    ap.add_argument("--policy-dir", default="/tmp", dest="policy_dir",
                    help="(RL) directory for the cached per-step policy checkpoints")
    # ---- fixed Proton.build() drive knobs ----
    ap.add_argument("--init", type=int, default=_INIT_STEPS,
                    help="init-pass (grow_boundaries=True) combined-run iterations per node")
    ap.add_argument("--evolve", type=int, default=_EVOLVE_STEPS,
                    help="evolution-pass (grow_boundaries=False) combined-run iterations "
                         "per node")
    ap.add_argument("--precone", type=int, default=0,
                    help="pre-grow each node's single-Δ⁴ seed by this many gated "
                         "cone-in moves before optimization (0 = bare seed)")
    ap.add_argument("--hodge-weights", choices=("content", "squared"),
                    default="content", dest="hodge_weights",
                    help="which quantity the Hodge inner-product weight W_k is "
                         "built from, for EVERY operator in the run (r_U, the "
                         "near-kernel residual, the register readout): "
                         "'content' = V, the k-content — an edge weighs its "
                         "length, so a timelike cell's weight is IMAGINARY; "
                         "'squared' = V², the engine default — an edge weighs "
                         "exactly its ℓ², real and signed. Both are fully "
                         "Lorentzian; this example defaults to 'content'.")
    args = ap.parse_args()
    # One flip, process-wide, BEFORE any node is built (flipping mid-run would
    # mix conventions across cached spectra).
    _CONVENTION = {"content": cob.HodgeWeightConvention.Content,
                   "squared": cob.HodgeWeightConvention.SquaredContent}[args.hodge_weights]
    cob.HodgeLaplacian.setDefaultWeightConvention(_CONVENTION)
    ProtonAnimator._TITLE_PREFIX += f"  ·  W = {'V' if args.hodge_weights == 'content' else 'V²'}"
    if args.rl or args.train:   # RL-driven two-step build — SAME charts as the fixed drive
        result, train_info = run_rl(cache_dir=args.policy_dir, retrain=args.train,
                                    train_iters=args.train_iters, episodes=args.episodes,
                                    best_of=args.best_of, seed=args.seed,
                                    visualize=args.live, save=args.save)
        if not args.live and not args.save:
            print("RL-driven two-step proton build finished (pass --live/--save to watch it):")
            print("  training:  " + "  ".join(f"{k}={v}" for k, v in train_info.items()))
            for label, metrics in result:
                print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
        return
    nodes = build_proton_nodes(seed=args.seed, precone=args.precone)
    result = run_build(nodes, visualize=args.live, save=args.save, init_steps=args.init,
                       evolve_steps=args.evolve)
    if not args.live and not args.save:
        print("two-step proton build finished (visualization off by default — pass --live "
              "or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
