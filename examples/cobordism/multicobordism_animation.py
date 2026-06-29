# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of a `MultiCobordism` optimization (#493).

The demo is a 2→2 recombination: two q-q̄ pairs ⟶ a diquark ⊔ an anti-diquark
(#491), built with the established `construct_inputs`/`construct_outputs` flow. Watch
the emergent register grow and the objective converge **as it runs**: this drives the
engine's two stages one move/iteration at a time and refreshes a live matplotlib figure
each step. Three panels:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U`, traced vs step;
  * **register** — the Betti `b₃` (the emergent color register) and the
    register-hole count, traced vs step;
  * **complex** — a 2-D projection (classical MDS on the dual/edge graph using the
    relaxed edge lengths) of the current 1-skeleton, with the vertices on the
    register holes highlighted. Each frame is Procrustes-aligned to the previous one
    and eased toward it, so the layout glides instead of bouncing and incremental
    changes are easy to read.

It drives only the **public** `MultiCobordism` API (`run_stage1`/`run_stage2`
with single-step counts — which continue the optimizer state across calls —
plus `betti`, `emergent_holes`, `regge_action_gradient`, `r_u`, `objective`, `st`). The C++
engine is untouched.

**Visualization is off by default** — `run_optimization(opt)` takes the fast
batched path with no per-step plotting overhead. Opt in with `visualize=True`
(or `--live`/`--save`) to animate, which is slower.

    # default: run the optimization fast, no visualization
    python multicobordism_animation.py
    # live (interactive backend):
    python multicobordism_animation.py --live
    # headless: write a GIF (no display needed):
    python multicobordism_animation.py --save recombination.gif
"""
import argparse
import cmath
import importlib.util
import math
import os
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * math.pi / 3)
_HERE = os.path.dirname(os.path.abspath(__file__))

# The 2→2 recombination this demo animates: two neutral q-q̄ pairs in, a diquark ⊔
# anti-diquark out (#491). The diquark color is the canonical √3-normalized 3̄ anti-
# triplet (`FixedBipartiteSequenceTopology`); the anti-diquark is the conjugate triplet
# on a distinct color axis.
_PAIRS = [[1, -1, 0], [1, 0, -1]]                  # two neutral q-q̄ color combos (Σ = 0)
_DIQUARK = [math.sqrt(3.0), 0.0, 0.0]              # canonical 3̄ anti-triplet
_ANTIDIQUARK = [0.0, math.sqrt(3.0), 0.0]          # conjugate triplet, distinct axis


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eo = _load("emergent_optimizer")


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


class MultiCobordismAnimator:
    """Steps a `MultiCobordism` and records/draws its progress one step at a time."""

    # What each stage is doing, for the on-figure labels.
    _STAGE_NAMES = {1: "combinatorial surgery", 2: "geometric relaxation"}

    def __init__(self, opt, degree=3, stage1_steps=40, stage1_candidates=8,
                 stage2_iters=30, stage2_beta=1.0):
        self.opt = opt
        self.k = degree
        self.s1, self.s1c = stage1_steps, stage1_candidates
        self.s2, self.s2_beta = stage2_iters, stage2_beta
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [], "stage": []}
        self._frames = stage1_steps + stage2_iters
        self._prev = None           # previous frame's drawn positions (for continuity)
        self._ease = 0.3            # how fast vertices glide to new targets (0=frozen, 1=snap)
        self._view = None           # complex-panel bbox; only grows, so the view never jitters
        self._done = False          # so "done" is announced exactly once

    # ---- one optimizer step (stage 1 = surgery, then stage 2 = relaxation) ----
    def _advance(self, frame):
        if frame < self.s1:
            # Exactly one greedy surgery step per frame — the animation advances "one move
            # at a time" (see the module docstring). Keep this at 1: each step grows the
            # complex and re-evaluates the global spectral r_U, so the per-frame cost climbs
            # super-linearly with max_steps (a 70-step frame measured ~360x a 1-step frame,
            # turning the ~9 s surgery phase into ~52 min). patience is irrelevant at 1 step.
            self.opt.run_stage1(max_steps=1, n_candidates=self.s1c, patience=10 ** 9)
            stage = 1
        else:
            self.opt.run_stage2(beta=self.s2_beta, max_iters=1)
            stage = 2
        self._record(stage)

    def _record(self, stage):
        st = self.opt.st
        self.hist["F"].append(float(self.opt.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.regge_action_gradient(st)))
        self.hist["rU"].append(float(self.opt.r_u(st)))
        self.hist["b3"].append(int(cob.MultiCobordism.betti(st)[self.k]))
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["stage"].append(stage)

    # ---- stable layout ----
    def _stable_coords(self, st):
        """A jitter-free layout: the raw MDS embedding, rigidly aligned to the *previous*
        frame and then eased toward it, so vertices glide instead of pop.

        Classical MDS is recomputed each step and is only defined up to rotation,
        reflection, and scale — and it's globally sensitive, so a small distance change
        (or two MDS eigenvalues crossing, or a new vertex appearing) can reshuffle the
        whole cloud. Two steps tame that:

        * **align** the new embedding onto the previous frame over *all* shared vertices
          via full Procrustes (scale + rotation + reflection) — this removes MDS's
          orientation/scale ambiguity, the dominant source of bounce;
        * **ease** each vertex from its old position a fraction `self._ease` of the way to
          the aligned target (exponential smoothing) — residual hops become smooth glides.

        New vertices (from surgery) start directly at their aligned position; removed ones
        simply drop out."""
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
        self.fig, (self.axm, self.axr, self.axg) = plt.subplots(
            1, 3, figsize=(15, 5))
        self.fig.suptitle("MultiCobordism optimization — live")
        return self.fig

    def _redraw(self):
        xs = range(len(self.hist["F"]))
        self.axm.clear()
        self.axm.plot(xs, self.hist["F"], label="F (objective)", color="C0")
        self.axm.plot(xs, self.hist["gradN2"], label="‖∇S‖²", color="C1")
        self.axm.plot(xs, self.hist["rU"], label="r_U", color="C2")
        self.axm.set_yscale("symlog")
        self.axm.set_title("metrics")
        self.axm.set_xlabel("step")
        self.axm.legend(loc="upper right", fontsize=8)

        self.axr.clear()
        self.axr.plot(xs, self.hist["b3"], label=f"b{self.k} (register)", color="C3",
                      marker=".")
        self.axr.plot(xs, self.hist["holes"], label="register holes", color="C4",
                      marker=".")
        self.axr.set_title("emergent register")
        self.axr.set_xlabel("step")
        self.axr.legend(loc="upper left", fontsize=8)

        self.axg.clear()
        st = self.opt.st
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
        self.axg.set_title(f"complex — stage {stage}: {self._STAGE_NAMES[stage]} "
                           f"(red = register holes)")
        self.axg.set_xticks([]); self.axg.set_yticks([])

    def update(self, frame):
        self._advance(frame)
        self._redraw()
        stage = self.hist["stage"][-1]
        label = f"stage {stage}: {self._STAGE_NAMES[stage]}"
        # A visible heartbeat: a step counter + stage name in the title and a flushed
        # stdout line. Stage-2 frames are several seconds of real compute (a dense Regge
        # Hessian over every edge) during which the GUI window can't repaint — without this
        # it looks hung even though it's advancing. The terminal line updates even while
        # the window is frozen mid-frame.
        if frame >= self._frames - 1 and not self._done:   # last step: announce done
            self._done = True
            self.fig.suptitle(f"MultiCobordism optimization — {label} — done")
            print(f"\rstep {frame + 1}/{self._frames} ({label}) — done")
        elif not self._done:
            self.fig.suptitle(
                f"MultiCobordism optimization — step {frame + 1}/{self._frames} · "
                f"{label}")
            print(f"\rstep {frame + 1}/{self._frames} ({label})",
                  end="", flush=True)
        return []


def build_demo_recombination(seed=3, n_refine=16, rounds=10):
    """A small demo system: recombine two q-q̄ pairs into a diquark ⊔ anti-diquark on a
    bare S⁴ (a 2→2 event, #491), wired exactly like `dk_joint_spin.build_pair_creation`.

    `construct_inputs` builds the two input pairs and `construct_outputs` the two output
    blocks (diquark, anti-diquark); the animation then drives the standard two stages —
    `run_stage1` (combinatorial surgery) and `run_stage2` (geometric relaxation) — one
    step at a time so you watch the registers grow and the objective converge."""
    host = eo.build_closed_s4(n_refine=n_refine, seed=seed)
    opt = cob.MultiCobordism(host, _PAIRS, [_DIQUARK, _ANTIDIQUARK],
                             degrees=[3], gamma=1.0, seed=seed)
    sv = [v.getId() for v in host.getVertexList().toVector()]
    opt.construct_inputs(sv[:2], rounds=rounds)
    opt.construct_outputs(sv[2:4], rounds=rounds)
    return opt


def animate(opt, save=None, interval=200, **kw):
    """Animate `opt`'s optimization. `save` → write a GIF/MP4 (headless, Agg);
    otherwise show a live interactive window. Returns the animator."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim_state = MultiCobordismAnimator(opt, **kw)
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


def run_optimization(opt, visualize=False, save=None, degree=3, stage1_steps=70,
                     stage1_candidates=10, stage2_iters=100, stage2_beta=1.0,
                     interval=200):   # ms/frame; keep > 0 — GIF/MP4 save() uses fps = 1000/interval
    """Run the two-stage optimization.

    Visualization is **off by default**: with ``visualize=False`` (and no
    ``save``) this takes the fast **batched** path — `run_stage1`/`run_stage2`
    run to completion in one call each, with no per-step layout/redraw overhead —
    and returns the final metrics. Opt in with ``visualize=True`` (live window) or
    ``save=...`` (GIF/MP4) to animate it step-by-step (slower); that returns the
    per-step history."""
    if not visualize and not save:
        opt.run_stage1(max_steps=stage1_steps, n_candidates=stage1_candidates)
        opt.run_stage2(beta=stage2_beta, max_iters=stage2_iters)
        st = opt.st
        return {"F": float(opt.objective()),
                "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
                "rU": float(opt.r_u(st)),
                "b3": int(cob.MultiCobordism.betti(st)[degree]),
                "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}
    return animate(opt, save=save, degree=degree, stage1_steps=stage1_steps,
                   stage1_candidates=stage1_candidates, stage2_iters=stage2_iters,
                   stage2_beta=stage2_beta, interval=interval).hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Visualization is OFF by default (fast). Opt in with --live or --save.
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--refine", type=int, default=16)
    ap.add_argument("--stage1", type=int, default=40)
    ap.add_argument("--stage2", type=int, default=30)
    args = ap.parse_args()
    opt = build_demo_recombination(seed=args.seed, n_refine=args.refine)
    result = run_optimization(opt, visualize=args.live, save=args.save,
                              stage1_steps=args.stage1, stage2_iters=args.stage2)
    if not args.live and not args.save:
        print("optimization finished (visualization off by default — pass --live "
              "or --save to watch it):")
        print("  " + "  ".join(f"{k}={v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
