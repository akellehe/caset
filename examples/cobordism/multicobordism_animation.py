# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of a `MultiCobordism` optimization (#493).

Watch the emergent register grow and the objective converge **as it runs**: this
drives the engine's two stages one move/iteration at a time and refreshes a live
matplotlib figure each step. Three panels:

  * **metrics** — the objective `F`, the Regge stationarity term `‖∇S‖²`, and the
    realizability residual `r_U`, traced vs step;
  * **register** — the Betti `b₃` (the emergent color register) and the
    register-hole count, traced vs step;
  * **complex** — a 2-D projection (classical MDS on the dual/edge graph using the
    relaxed edge lengths) of the current 1-skeleton, with the vertices on the
    register holes highlighted.

It drives only the **public** `MultiCobordism` API (`run_stage1`/`relax_stage2`
with single-step counts — which continue the optimizer state across calls —
plus `betti`, `emergent_holes`, `grad_norm2`, `r_u`, `objective`, `st`). The C++
engine is untouched.

    # live (interactive backend):
    python multicobordism_animation.py
    # headless: write a GIF (no display needed):
    python multicobordism_animation.py --save merge.gif
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

    def __init__(self, opt, degree=3, stage1_steps=40, stage1_candidates=8,
                 stage2_iters=30, stage2_beta=1.0):
        self.opt = opt
        self.k = degree
        self.s1, self.s1c = stage1_steps, stage1_candidates
        self.s2, self.s2_beta = stage2_iters, stage2_beta
        self.hist = {"F": [], "gradN2": [], "rU": [], "b3": [], "holes": [], "stage": []}
        self._frames = stage1_steps + stage2_iters

    # ---- one optimizer step (stage 1 = surgery, then stage 2 = relaxation) ----
    def _advance(self, frame):
        if frame < self.s1:
            self.opt.run_stage1(max_steps=1, n_candidates=self.s1c, patience=10 ** 9)
            stage = 1
        else:
            self.opt.relax_stage2(beta=self.s2_beta, max_iters=1)
            stage = 2
        self._record(stage)

    def _record(self, stage):
        st = self.opt.st
        self.hist["F"].append(float(self.opt.objective()))
        self.hist["gradN2"].append(float(cob.MultiCobordism.grad_norm2(st)))
        self.hist["rU"].append(float(self.opt.r_u(st)))
        self.hist["b3"].append(int(cob.MultiCobordism.betti(st)[self.k]))
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["stage"].append(stage)

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
        coords = _mds_layout(st)
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
        stage = self.hist["stage"][-1] if self.hist["stage"] else 1
        self.axg.set_title(f"complex (stage {stage}; red = register holes)")
        self.axg.set_xticks([]); self.axg.set_yticks([])

    def update(self, frame):
        self._advance(frame)
        self._redraw()
        return []


def build_demo_merge(seed=3, n_refine=16):
    """A small demo system: merge two color states into the singlet on a bare S⁴."""
    host = eo.build_closed_s4(n_refine=n_refine, seed=seed)
    opt = cob.MultiCobordism(host, [[1, -1, 0], [1, 0, -1]], [[1, _W, _W * _W]],
                             degrees=[3], gamma=1.0, seed=seed)
    sv = [v.getId() for v in host.getVertexList().toVector()]
    opt.construct_inputs(sv[:2], rounds=10)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", help="write a GIF/MP4 instead of showing live")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--refine", type=int, default=16)
    ap.add_argument("--stage1", type=int, default=40)
    ap.add_argument("--stage2", type=int, default=30)
    args = ap.parse_args()
    opt = build_demo_merge(seed=args.seed, n_refine=args.refine)
    animate(opt, save=args.save, stage1_steps=args.stage1, stage2_iters=args.stage2)


if __name__ == "__main__":
    main()
