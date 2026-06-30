# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of the canonical two-step **Proton** build (#522, #526).

Drives the actual `tessera.cobordism.Proton` class and animates the proton assembling in
its two physical steps, each growing its whole topology from a single Δ⁴ simplex via the
trap door —

  * **Step A — recombination** (`Proton.recombination_node`): two neutral q-q̄ pairs ⟶
    a colored diquark `{1, ω}` ⊔ anti-diquark `{1, ω²}`;
  * **Step B — formation** (`Proton.formation_node`): the diquark `{1, ω}` + the third
    quark `{ω²}` ⟶ the colorless proton singlet `{1, ω, ω²}` (ω = exp(2πi/3)).

**Both steps are shown at one time.** The figure is a 2×2 grid whose *bottom row holds two
complex panels — Step A on the left, Step B on the right — both visible the whole run.*
Step A animates first (Step B waiting on its seed simplex); then Step B animates while Step
A holds its finished complex, so at the end you see the full diquark next to the full
proton, side by side.

Each node is driven the way `Proton.build()` drives it: an **init pass**
(`grow_boundaries=True`) that grows the color register, an **evolution pass**
(`grow_boundaries=False`) with the boundary frozen, then the **geometric relaxation**
(`run_stage2`). Each surgery pass runs as a *single* `run_stage1` call (one frame): its
grow-burst self-recovery and `patience` early-stop both work only *within one call*, so
driving surgery one step at a time (as the old demo did) never grows the register — which is
exactly why the proton never showed its three quark holes. `run_stage2` is then advanced one
relaxation iteration per frame, so the geometric settling animates smoothly.

The four panels:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U` vs step (a dashed line marks the Step A → Step B boundary);
  * **register** — the Betti `b₃` (the emergent color register) and the register-hole count
    vs step (the proton's three registers appear as Step B grows);
  * **complex — Step A** and **complex — Step B** — 2-D classical-MDS projections of each
    node's relaxing 1-skeleton (register-hole vertices highlighted). Each frame is
    normalized to a fixed scale, Procrustes-aligned (rotation/reflection only) to the
    previous frame, and eased toward it, with the view auto-fit to the cloud — so the
    structure stays legible instead of collapsing into a dot.

The figure title reports the live **convergence verdict**: whether Step B's whole cobordism
carries the singlet `{1, ω, ω²}` (color residual `r_state` ≈ 0) on its ≥ 3 emergent color
holes.

It drives only the **public** `Proton` (`recombination_node`/`formation_node`) and
`MultiCobordism` (`run_stage1`/`run_stage2`, plus `betti`, `emergent_holes`,
`regge_action_gradient`, `r_state`, `r_u`, `objective`, `st`) APIs — the *same* node setups
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
"""
import argparse
import math

import numpy as np
from scipy.sparse.csgraph import shortest_path

import tessera

cob = tessera.cobordism

# The drive `Proton.build()` uses per node, mirrored here. The surgery (init/evolution)
# passes each run as ONE `run_stage1` call per node — `run_stage1`'s grow-burst self-recovery
# and its `patience` early-stop both work *within a single call*, so chunking a pass into
# small per-frame calls neither grows the register nor early-stops (it just does far more,
# slower steps). The register growth is therefore shown as one surgery frame per pass; the
# *geometric relaxation* is what animates smoothly, one `run_stage2` iteration per frame.
# `_*_CHUNK` cap a pass's per-frame call (defaulting to the whole pass = one call =
# `Proton.build()`'s exact drive); `_STAGE1_PATIENCE` matches `Proton.build()`. These totals
# are sized so Step B reliably grows its three quark holes and carries the singlet.
_INIT_STEPS = 180          # init-pass (grow_boundaries=True) steps per node
_EVOLVE_STEPS = 60         # evolution-pass (grow_boundaries=False) steps per node
_INIT_CHUNK = _INIT_STEPS  # init steps per frame (whole pass = one call, so it converges)
_EVOLVE_CHUNK = _EVOLVE_STEPS  # evolution steps per frame (whole pass = one call)
_STAGE2_ITERS = 10         # geometric-relaxation iterations per node (one per frame)
_STAGE1_PATIENCE = 15      # matches Proton.build(): early-stop a pass after this many stalls
_STAGE1_CANDIDATES = 8
_COLOR_TOL = 0.5           # singlet r_state below this ⇒ the proton carries the color
_MIN_QUARK_HOLES = 3       # a proton is three quarks ⇒ three color registers


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
        w = math.sqrt(max(abs(e.getSquaredLength().real), 1e-6))
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

    Each node is driven the way `Proton.build()` drives it: an init pass
    (`grow_boundaries=True`, grows the color register) and an evolution pass
    (`grow_boundaries=False`), advanced in chunks of several surgery steps per frame, then
    `run_stage2` one relaxation iteration per frame. The two complex panels are bound one
    each to Step A and Step B for the whole run, so Step A's finished diquark stays visible
    beside the proton as Step B forms."""

    _PHASE_NAMES = {"init": "growing register", "evolve": "evolving (∂W frozen)",
                    "stage2": "relaxing geometry"}

    def __init__(self, nodes, degree=3, init_steps=_INIT_STEPS, init_chunk=_INIT_CHUNK,
                 evolve_steps=_EVOLVE_STEPS, evolve_chunk=_EVOLVE_CHUNK,
                 stage2_iters=_STAGE2_ITERS, stage1_candidates=_STAGE1_CANDIDATES,
                 stage1_patience=_STAGE1_PATIENCE, stage2_beta=1.0):
        self.nodes = nodes                  # [(MultiCobordism, "Step A: ..."), ...] in order
        self.k = degree
        self.s1c, self.s1pat, self.s2_beta = stage1_candidates, stage1_patience, stage2_beta
        self._schedule = self._make_schedule(len(nodes), init_steps, init_chunk,
                                             evolve_steps, evolve_chunk, stage2_iters)
        self._frames = len(self._schedule)
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [],
                     "phase": [], "node": []}
        self._boundaries = []       # step indices where a later node begins (trace markers)
        self._layouts = [_StableLayout() for _ in nodes]   # one per complex panel
        self._active = 0            # index of the node currently being driven
        self._done = False          # so the verdict is announced exactly once

    @staticmethod
    def _make_schedule(n_nodes, init_steps, init_chunk, evolve_steps, evolve_chunk,
                       stage2_iters):
        """A flat list of (node_index, phase, count) ops, one per frame: each node's init
        pass (in `init_chunk`-sized bites), then its evolution pass (in `evolve_chunk`
        bites), then `stage2_iters` single relaxation iterations."""
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
            sched += [(i, "stage2", 1) for _ in range(stage2_iters)]
        return sched

    # ---- one scheduled chunk on the active node ----
    def _advance(self, frame):
        node_index, phase, count = self._schedule[frame]
        if node_index != self._active:                       # a new node begins
            self._active = node_index
            self._boundaries.append(len(self.hist["F"]))
        node, _label = self.nodes[node_index]
        if phase == "init":
            node.run_stage1(max_steps=count, n_candidate_moves=self.s1c,
                            patience=self.s1pat, grow_boundaries=True)
        elif phase == "evolve":
            node.run_stage1(max_steps=count, n_candidate_moves=self.s1c,
                            patience=self.s1pat, grow_boundaries=False)
        else:
            node.run_stage2(beta=self.s2_beta, max_iters=count)
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
        self.fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        (self.axm, self.axr), (self.axA, self.axB) = axes
        return self.fig

    def _draw_complex(self, ax, node_index, title):
        node, _label = self.nodes[node_index]
        st = node.st
        coords = self._layouts[node_index].coords(st)
        ax.clear()
        hole_vs = {v for h in cob.MultiCobordism.emergent_holes(st, self.k) for v in h}
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a in coords and b in coords:
                p, q = coords[a], coords[b]
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.8", lw=0.5, zorder=1)
        if len(coords) >= 1:
            pts = np.array(list(coords.values()))
            cols = ["C3" if v in hole_vs else "0.4" for v in coords]
            sz = [40 if v in hole_vs else 8 for v in coords]
            ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=sz, zorder=2)
            if len(coords) >= 2:
                view = self._layouts[node_index].view(coords)
                ax.set_xlim(view[0], view[1])
                ax.set_ylim(view[2], view[3])
        n_holes = len(cob.MultiCobordism.emergent_holes(st, self.k))
        ax.set_aspect("equal")
        ax.set_title(f"{title}  ({n_holes} register{'s' if n_holes != 1 else ''})",
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
        self.axm.set_yscale("symlog")
        self.axm.set_title("metrics")
        self.axm.set_xlabel("frame")
        self.axm.legend(loc="upper right", fontsize=8)

        self.axr.clear()
        self.axr.plot(xs, self.hist["b3"], label=f"b{self.k} (register)", color="C3",
                      marker=".")
        self.axr.plot(xs, self.hist["holes"], label="register holes", color="C4",
                      marker=".")
        for b in self._boundaries:
            self.axr.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
        self.axr.axhline(_MIN_QUARK_HOLES, color="0.6", ls=":", lw=0.8)
        self.axr.set_title("emergent register")
        self.axr.set_xlabel("frame")
        self.axr.legend(loc="upper left", fontsize=8)

        # Both complex panels, every frame: Step A on the left, Step B on the right. The
        # active node animates; the other holds its current (finished, or not-yet-started)
        # complex — so both steps are on screen at one time.
        self._draw_complex(self.axA, 0, self.nodes[0][1])
        self._draw_complex(self.axB, 1, self.nodes[1][1])

    def update(self, frame):
        node_index, phase, _count = self._schedule[frame]
        label = f"{self.nodes[node_index][1]} · {self._PHASE_NAMES[phase]}"
        # A visible heartbeat *before* the step: a surgery frame is several seconds of real
        # compute during which the GUI window can't repaint, so announce what's running first
        # (title + flushed stdout line) — otherwise the window looks hung mid-frame.
        if not self._done:
            self.fig.suptitle(
                f"Proton build (two-step) — frame {frame + 1}/{self._frames} · {label}")
            print(f"\rframe {frame + 1}/{self._frames} ({label})", end="", flush=True)
        self._advance(frame)
        self._redraw()
        if frame >= self._frames - 1 and not self._done:   # last frame: announce the verdict
            self._done = True
            ok, res, holes = self.verdict()
            tag = (f"CONVERGED ✓ — proton {{1,ω,ω²}} carried (r_state={res:.2g}, "
                   f"{holes} registers)" if ok else
                   f"did NOT converge (r_state={res:.2g}, {holes} registers)")
            self.fig.suptitle(f"Proton build (two-step) — {tag}")
            print(f"\rframe {frame + 1}/{self._frames} ({label}) — {tag}")
        return []


def build_proton_nodes(seed=3):
    """The two `MultiCobordism` nodes the `Proton` class drives, in build order, for the
    animation: Step A recombination then Step B formation, each on its own single-Δ⁴ seed.

    Built via `Proton.recombination_node`/`formation_node` — the *same* node setups
    `Proton.build()` uses — with `Proton.build`'s attempt-0 seeds (A = `seed`, B =
    `seed + 1`). The default `seed=3` converges on attempt 0 (Step B grows three quark holes
    and carries the singlet); the animation reports the live verdict either way."""
    p = cob.Proton(seed=seed)
    return [
        (p.recombination_node(seed), "Step A — recombination (→ diquark {1, ω})"),
        (p.formation_node(seed + 1), "Step B — formation (→ proton {1, ω, ω²})"),
    ]


def animate(nodes, save=None, interval=200, **kw):
    """Animate the proton node sequence. `save` → write a GIF/MP4 (headless, Agg);
    otherwise show a live interactive window. Returns the animator."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim_state = ProtonAnimator(nodes, **kw)
    anim_state._setup(plt)
    anim_state.fig.suptitle("Proton build (two-step) — live")
    fa = FuncAnimation(anim_state.fig, anim_state.update,
                       frames=anim_state._frames, interval=interval,
                       repeat=False, blit=False)
    if save:
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        fa.save(save, writer=writer, dpi=90)
        print(f"saved animation -> {save}")
    else:
        plt.show()
    anim_state._anim = fa  # keep a ref so it isn't GC'd in live mode
    return anim_state


def run_build(nodes, visualize=False, save=None, degree=3, init_steps=_INIT_STEPS,
              evolve_steps=_EVOLVE_STEPS, stage2_iters=_STAGE2_ITERS,
              stage1_candidates=_STAGE1_CANDIDATES, stage1_patience=_STAGE1_PATIENCE,
              stage2_beta=1.0, interval=200,  # interval: ms/frame; GIF/MP4 fps = 1000/interval
              **anim_kw):
    """Run the two-step proton build over `nodes`, driving each node the way
    `Proton.build()` does: init pass (`grow_boundaries=True`) → evolution pass
    (`grow_boundaries=False`) → `run_stage2`.

    Visualization is **off by default**: with ``visualize=False`` (and no ``save``) this
    takes the fast **batched** path — each node's passes run to completion in one call each,
    no per-step layout/redraw overhead — and returns each node's final metrics plus the
    convergence verdict. Opt in with ``visualize=True`` (live window) or ``save=...``
    (GIF/MP4) to animate it step-by-step (slower); that returns the per-step history."""
    if not visualize and not save:
        out = []
        for node, label in nodes:
            node.run_stage1(max_steps=init_steps, n_candidate_moves=stage1_candidates,
                            patience=stage1_patience, grow_boundaries=True)
            node.run_stage1(max_steps=evolve_steps, n_candidate_moves=stage1_candidates,
                            patience=stage1_patience, grow_boundaries=False)
            node.run_stage2(beta=stage2_beta, max_iters=stage2_iters)
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
                   evolve_steps=evolve_steps, stage2_iters=stage2_iters,
                   stage1_candidates=stage1_candidates, stage1_patience=stage1_patience,
                   stage2_beta=stage2_beta, interval=interval, **anim_kw).hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Visualization is OFF by default (fast). Opt in with --live or --save.
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--init", type=int, default=_INIT_STEPS,
                    help="init-pass (grow_boundaries=True) steps per node")
    ap.add_argument("--evolve", type=int, default=_EVOLVE_STEPS,
                    help="evolution-pass (grow_boundaries=False) steps per node")
    ap.add_argument("--stage2", type=int, default=_STAGE2_ITERS,
                    help="geometric-relaxation iterations per node")
    args = ap.parse_args()
    nodes = build_proton_nodes(seed=args.seed)
    result = run_build(nodes, visualize=args.live, save=args.save, init_steps=args.init,
                       evolve_steps=args.evolve, stage2_iters=args.stage2)
    if not args.live and not args.save:
        print("two-step proton build finished (visualization off by default — pass --live "
              "or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
