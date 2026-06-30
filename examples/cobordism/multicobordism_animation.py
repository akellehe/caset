# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of the canonical two-step **Proton** build (#522).

Supersedes the old MultiCobordism recombination demo: this drives the actual
`tessera.cobordism.Proton` class and animates the proton assembling in two steps, each
growing its whole topology from a single Δ⁴ simplex via the trap door —

  * **Step A — recombination** (`Proton.recombination_node`): two neutral q-q̄ pairs ⟶
    a colored diquark `{1, ω}` ⊔ anti-diquark `{1, ω²}`;
  * **Step B — formation** (`Proton.formation_node`): the diquark `{1, ω}` + the third
    quark `{ω²}` ⟶ the colorless proton singlet `{1, ω, ω²}` (ω = exp(2πi/3)).

The two nodes are animated in sequence, one move/iteration at a time, in a live
three-panel matplotlib figure:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U` vs step (a dashed line marks the Step A → Step B
    boundary);
  * **register** — the Betti `b₃` (the emergent color register) and the register-hole
    count vs step;
  * **complex** — a 2-D classical-MDS projection of the current 1-skeleton (using the
    relaxed edge lengths), register-hole vertices highlighted. Each frame is
    Procrustes-aligned to the previous and eased toward it so the layout glides; the
    continuity resets at the Step A → Step B boundary (Step B starts a fresh simplex).

It drives only the **public** `Proton` (the `recombination_node`/`formation_node`
factories) and `MultiCobordism` (`run_stage1`/`run_stage2` with single-step counts, which
continue the optimizer state across calls, plus `betti`, `emergent_holes`,
`regge_action_gradient`, `r_u`, `objective`, `st`) APIs — the *same* node setups
`Proton.build()` uses, so the animation shows the real class. The C++ engine is untouched,
and there is no dependency on the (retiring) `emergent_optimizer`.

**Visualization is off by default** — `run_build(...)` takes the fast batched path with no
per-step plotting overhead. Opt in with `visualize=True` (or `--live`/`--save`) to
animate, which is slower.

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


def _mds_layout(st):
    """2-D classical-MDS coordinates per vertex id, from graph shortest-path
    distances weighted by the relaxed edge lengths `sqrt(|Re ℓ²|)`. Reflects the
    actual relaxing geometry, not just the abstract topology."""
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
    return {vids[i]: coords[i] for i in range(n)}


class ProtonAnimator:
    """Steps a SEQUENCE of `MultiCobordism` nodes — the proton's Step A (recombination)
    then Step B (formation) — one move/iteration at a time, recording and drawing the
    progress. Each node runs `stage1_steps` surgery steps then `stage2_iters` relaxation
    iterations; the complex-layout continuity resets at each node boundary (the next node
    starts from a fresh single simplex)."""

    _STAGE_NAMES = {1: "combinatorial surgery", 2: "geometric relaxation"}

    def __init__(self, nodes, degree=3, stage1_steps=40, stage1_candidates=8,
                 stage2_iters=30, stage2_beta=1.0):
        self.nodes = nodes                  # [(MultiCobordism, "Step A: ..."), ...] in order
        self.k = degree
        self.s1, self.s1c = stage1_steps, stage1_candidates
        self.s2, self.s2_beta = stage2_iters, stage2_beta
        self._per_node = stage1_steps + stage2_iters
        self._frames = len(nodes) * self._per_node
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [],
                     "stage": [], "node": []}
        self._boundaries = []       # step indices where a later node begins (trace markers)
        self._prev = None           # previous frame's drawn positions (for continuity)
        self._ease = 0.3            # how fast vertices glide to new targets (0=frozen, 1=snap)
        self._view = None           # complex-panel bbox; only grows, so the view never jitters
        self._done = False          # so "done" is announced exactly once
        self._cur, self._cur_label = nodes[0]   # node currently being animated

    # ---- one optimizer step on the current node (stage 1 = surgery, stage 2 = relax) ----
    def _advance(self, frame):
        node_index, local = divmod(frame, self._per_node)
        node, label = self.nodes[node_index]
        if local == 0 and node_index > 0:        # a NEW node starts: reset layout continuity
            self._prev = None
            self._view = None
            self._boundaries.append(len(self.hist["F"]))
        self._cur, self._cur_label = node, label
        if local < self.s1:
            # Exactly one greedy surgery step per frame. Keep this at 1: each step grows the
            # complex and re-evaluates the global spectral r_U, so the per-frame cost climbs
            # super-linearly with max_steps. patience is irrelevant at 1 step.
            node.run_stage1(max_steps=1, n_candidate_moves=self.s1c, patience=10 ** 9)
            stage = 1
        else:
            node.run_stage2(beta=self.s2_beta, max_iters=1)
            stage = 2
        self._record(node, stage, node_index)

    def _record(self, node, stage, node_index):
        st = node.st
        self.hist["F"].append(float(node.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.regge_action_gradient(st)))
        self.hist["rU"].append(float(node.r_u(st)))
        self.hist["b3"].append(int(cob.MultiCobordism.betti(st)[self.k]))
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["stage"].append(stage)
        self.hist["node"].append(node_index)

    # ---- stable layout ----
    def _stable_coords(self, st):
        """A jitter-free layout: the raw MDS embedding, rigidly aligned to the *previous*
        frame and then eased toward it, so vertices glide instead of pop.

        Classical MDS is recomputed each step and is only defined up to rotation,
        reflection, and scale — and it's globally sensitive, so a small distance change
        (or two MDS eigenvalues crossing, or a new vertex appearing) can reshuffle the
        whole cloud. Two steps tame that:

        * **align** the new embedding onto the previous frame over *all* shared vertices
          via full Procrustes (scale + rotation + reflection);
        * **ease** each vertex from its old position a fraction `self._ease` of the way to
          the aligned target (exponential smoothing).

        New vertices (from surgery) start directly at their aligned position; removed ones
        simply drop out. `_prev` is reset to None at a node boundary, so the next node's
        fresh simplex defines its own frame instead of being aligned to the old complex."""
        coords = _mds_layout(st)
        if len(coords) < 2:
            return coords
        if self._prev is None:                               # first frame defines the frame
            self._prev = {v: np.asarray(p, float) for v, p in coords.items()}
            return self._prev
        shared = [v for v in coords if v in self._prev]
        if len(shared) >= 2:                                 # full Procrustes onto previous
            cur = np.array([coords[v] for v in shared])
            ref = np.array([self._prev[v] for v in shared])
            cur_c, ref_c = cur.mean(0), ref.mean(0)
            cur0, ref0 = cur - cur_c, ref - ref_c
            u, sng, vt = np.linalg.svd(cur0.T @ ref0)
            rot = u @ vt                                     # rotation/reflection
            denom = float((cur0 ** 2).sum())
            scale = (sng.sum() / denom) if denom > 1e-12 else 1.0
            aligned = {v: scale * (np.asarray(p) - cur_c) @ rot + ref_c
                       for v, p in coords.items()}
        else:                                                # nothing shared: take raw
            aligned = {v: np.asarray(p, float) for v, p in coords.items()}
        eased = {}
        for v, target in aligned.items():
            prev = self._prev.get(v)
            eased[v] = target if prev is None else prev + self._ease * (target - prev)
        self._prev = eased
        return eased

    # ---- drawing ----
    def _setup(self, plt):
        self.fig, (self.axm, self.axr, self.axg) = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle("Proton build (two-step) — live")
        return self.fig

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
        self.axm.set_xlabel("step")
        self.axm.legend(loc="upper right", fontsize=8)

        self.axr.clear()
        self.axr.plot(xs, self.hist["b3"], label=f"b{self.k} (register)", color="C3",
                      marker=".")
        self.axr.plot(xs, self.hist["holes"], label="register holes", color="C4",
                      marker=".")
        for b in self._boundaries:
            self.axr.axvline(b - 0.5, color="0.6", ls="--", lw=0.8)
        self.axr.set_title("emergent register")
        self.axr.set_xlabel("step")
        self.axr.legend(loc="upper left", fontsize=8)

        self.axg.clear()
        st = self._cur.st
        coords = self._stable_coords(st)
        hole_vs = {v for h in cob.MultiCobordism.emergent_holes(st, self.k) for v in h}
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a in coords and b in coords:
                p, q = coords[a], coords[b]
                self.axg.plot([p[0], q[0]], [p[1], q[1]], color="0.8", lw=0.5, zorder=1)
        if coords:
            pts = np.array(list(coords.values()))
            cols = ["C3" if v in hole_vs else "0.4" for v in coords]
            sz = [40 if v in hole_vs else 8 for v in coords]
            self.axg.scatter(pts[:, 0], pts[:, 1], c=cols, s=sz, zorder=2)
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.1 * max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
            box = [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad]
            if self._view is None:
                self._view = box
            else:                                # grow-only limits: the view never pans back
                self._view = [min(self._view[0], box[0]), max(self._view[1], box[1]),
                              min(self._view[2], box[2]), max(self._view[3], box[3])]
            self.axg.set_xlim(self._view[0], self._view[1])
            self.axg.set_ylim(self._view[2], self._view[3])
        self.axg.set_aspect("equal")
        stage = self.hist["stage"][-1] if self.hist["stage"] else 1
        self.axg.set_title(f"{self._cur_label} — {self._STAGE_NAMES[stage]} "
                           f"(red = register holes)")
        self.axg.set_xticks([]); self.axg.set_yticks([])

    def update(self, frame):
        self._advance(frame)
        self._redraw()
        stage = self.hist["stage"][-1]
        label = f"{self._cur_label} · {self._STAGE_NAMES[stage]}"
        # A visible heartbeat: a step counter + label in the title and a flushed stdout line.
        # Stage-2 frames are several seconds of real compute (a dense Regge Hessian over
        # every edge) during which the GUI window can't repaint — the terminal line updates
        # even while the window is frozen mid-frame.
        if frame >= self._frames - 1 and not self._done:   # last step: announce done
            self._done = True
            self.fig.suptitle(f"Proton build (two-step) — {label} — done")
            print(f"\rstep {frame + 1}/{self._frames} ({label}) — done")
        elif not self._done:
            self.fig.suptitle(
                f"Proton build (two-step) — step {frame + 1}/{self._frames} · {label}")
            print(f"\rstep {frame + 1}/{self._frames} ({label})", end="", flush=True)
        return []


def build_proton_nodes(seed=3):
    """The two `MultiCobordism` nodes the `Proton` class drives, in build order, for the
    animation: Step A recombination then Step B formation, each on its own single-Δ⁴ seed.

    Built via `Proton.recombination_node`/`formation_node` — the *same* node setups
    `Proton.build()` uses — with `Proton.build`'s attempt-0 seeds (A = `seed`,
    B = `seed + 1`). The animation then drives each node's two stages (`run_stage1`
    combinatorial surgery, including the trap door that grows the complex from one simplex
    on a stall; `run_stage2` geometric relaxation) one step at a time."""
    p = cob.Proton(seed=seed)
    return [
        (p.recombination_node(seed), "Step A: recombination (→ diquark {1, ω})"),
        (p.formation_node(seed + 1), "Step B: formation (→ proton {1, ω, ω²})"),
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


def run_build(nodes, visualize=False, save=None, degree=3, stage1_steps=70,
              stage1_candidates=10, stage2_iters=100, stage2_beta=1.0,
              interval=200):   # ms/frame; keep > 0 — GIF/MP4 save() uses fps = 1000/interval
    """Run the two-step proton build over `nodes`.

    Visualization is **off by default**: with ``visualize=False`` (and no ``save``) this
    takes the fast **batched** path — each node's `run_stage1`/`run_stage2` run to
    completion in one call each, no per-step layout/redraw overhead — and returns each
    node's final metrics. Opt in with ``visualize=True`` (live window) or ``save=...``
    (GIF/MP4) to animate it step-by-step (slower); that returns the per-step history."""
    if not visualize and not save:
        out = []
        for node, label in nodes:
            node.run_stage1(max_steps=stage1_steps, n_candidate_moves=stage1_candidates)
            node.run_stage2(beta=stage2_beta, max_iters=stage2_iters)
            st = node.st
            out.append((label, {
                "F": float(node.objective()),
                "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
                "rU": float(node.r_u(st)),
                "b3": int(cob.MultiCobordism.betti(st)[degree]),
                "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}))
        return out
    return animate(nodes, save=save, degree=degree, stage1_steps=stage1_steps,
                   stage1_candidates=stage1_candidates, stage2_iters=stage2_iters,
                   stage2_beta=stage2_beta, interval=interval).hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Visualization is OFF by default (fast). Opt in with --live or --save.
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--stage1", type=int, default=40)
    ap.add_argument("--stage2", type=int, default=30)
    args = ap.parse_args()
    nodes = build_proton_nodes(seed=args.seed)
    result = run_build(nodes, visualize=args.live, save=args.save,
                       stage1_steps=args.stage1, stage2_iters=args.stage2)
    if not args.live and not args.save:
        print("two-step proton build finished (visualization off by default — pass --live "
              "or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
