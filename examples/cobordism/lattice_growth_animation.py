# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Live view of the lattice growing — or failing to — from a single Δ⁴ simplex (#625).

`multicobordism_animation.py` shows the finished physics: the proton assembling in two
steps. This one shows the *search*, because the question it was written for is why a
complex sometimes climbs from one simplex to a few dozen cells and sometimes sits at the
seed forever.

## What is actually going on in there

`MultiCobordism::runStage1` has exactly two ways to change the complex, and they pull in
opposite directions:

  * **`step()`** — the descent. Prices candidate moves against
    `ΔF`, `F = ‖∇S‖² + Γ·r_U`, and commits the best one that LOWERS `F`.
  * **`trapDoorMove()`** — the escape. Commits a gated move that need NOT lower `F`, and
    fires **only when `step()` reports it cannot descend**.

The trap door is the only thing that grows the complex. So growth happens exactly when the
descent fails, which makes the *reach of the search* a control parameter on the physics:
a search that always finds something to descend into never lets the trap door fire.

That matters because `deltaF` prices a combinatorial move at the **current, un-relaxed**
metric. A freshly grown cell is scored carrying curvature it has not yet been given a
chance to relax, so growth reads as a loss and undoing it reads as a gain. Measured on the
canonical single-Δ⁴ seed: one gated cone-in takes `‖∇S‖²` from 0.2568 to 1.0337 (10 edges
to 14), and the cone-out that undoes it pays `ΔF = −0.777`.

`Γ·r_U` cannot arbitrate early on. It sits pinned at its full-leak floor until the complex
is large enough to hold holes at all, so in the early regime it contributes no gradient in
either direction — the descent's only visible slope points at collapse, and the complex has
to be carried across that stretch blind.

**This is not a claim that the objective prefers small complexes.** `S` is the structure
term; its deficits are curvature, and `‖∇S‖²` measures distance from `δS = 0`. Growth
raising `‖∇S‖²` before relaxation is growth introducing curvature, which is what growth is
for. The asymmetry is in *when the move is priced*, not in what `F` rewards.

## The panels

A 2×3 grid. The top row is the search, the bottom row is the lattice.

  * **size** — top cells and edges vs frame. A climb means the trap door is winning; a flat
    line at the seed means it never fires.
  * **objective** — `F` and its two parts, `‖∇S‖²` and `Γ·r_U`, on a log axis. Watch
    `Γ·r_U` sit flat at the leak floor and then drop once registers appear: that is the
    moment the register term starts arbitrating.
  * **net size change** — signed Δ(top cells) per frame, green for growth and red for
    shrinkage. **This is the treadmill made visible.** Alternating bars of similar height
    mean the trap door and the descent are cancelling.
  * **complex** — 2-D classical-MDS projection of the relaxing 1-skeleton, each emergent
    register (a removed top cell) outlined in red and numbered. Layout, normalization,
    Procrustes alignment and easing are reused verbatim from
    `multicobordism_animation.py`, so the two views are directly comparable.
  * **dual — spatial / temporal curvature** — the circumcentric dual at primal cell
    centroids, colored by local Regge curvature. The Lorentzian deficit is COMPLEX, so it
    splits: spatial = `Re ε·|★|` (angle defect, timelike hinges), temporal = `Im ε·|★|`
    (boost content, spacelike hinges). Signed diverging colormap centered at 0.

Each surgery pass is ONE `run_stage1` call and one frame — its grow-burst recovery and
`patience` early-stop both work only within a single call, so driving surgery step-by-step
never grows the register. `run_stage2` is then advanced one iteration per frame so the
geometric settling animates.

Drives only public API: `Proton`/`ProtonIngredients` node factories, `MultiCobordism`
(`run_stage1`/`run_stage2`, `betti`, `emergent_holes`, `regge_action_gradient`, `r_u`,
`objective`, `st`) and the geometry readers. The C++ engine is untouched.

    # fast, no plotting
    python lattice_growth_animation.py
    # live (interactive backend)
    python lattice_growth_animation.py --live
    # headless GIF
    python lattice_growth_animation.py --save growth.gif
    # the formation node (a pinned singlet output) instead of the joint node
    python lattice_growth_animation.py --node formation --live
"""
import argparse
import math
import os
import sys

import numpy as np

import tessera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multicobordism_animation import (  # noqa: E402  (path set above)
    ProtonAnimator, _StableLayout,
)

cob = tessera.cobordism

_INIT_PASSES = 6        # surgery passes with grow_boundaries=True (the init pass)
_EVOLVE_PASSES = 6      # surgery passes with the boundary frozen
_INIT_STEPS = 30        # run_stage1 max_steps per init pass
_EVOLVE_STEPS = 30      # run_stage1 max_steps per evolve pass
_CANDIDATES = 8         # run_stage1 n_candidate_moves (the per-step trial budget)
_PATIENCE = 15
_RELAX_FRAMES = 40      # run_stage2 iterations, one per frame
_HEAT_REFRESH_EVERY = 4


class LatticeGrowthAnimator:
    """Drives ONE node from its single-Δ⁴ seed and animates the size/objective traces
    beside the lattice, so growth and the descent that undoes it are visible together."""

    def __init__(self, node, label, degree=3):
        self.node = node
        self.label = label
        self.k = degree
        self.layout = _StableLayout()
        self.hist = {"cells": [], "edges": [], "F": [], "grad": [], "gamma_ru": [],
                     "holes": [], "phase": []}
        self._curv_cache = None
        self._schedule = (["grow"] * _INIT_PASSES + ["evolve"] * _EVOLVE_PASSES
                          + ["relax"] * _RELAX_FRAMES)
        self.frames = len(self._schedule)

    # ---- drive ----
    def _advance(self, frame):
        """One frame of the SAME drive `Proton.build()` uses: a surgery pass is one whole
        `run_stage1` call, then stage 2 advances one relaxation iteration per frame."""
        phase = self._schedule[frame]
        if phase == "grow":
            self.node.run_stage1(_INIT_STEPS, _CANDIDATES, _PATIENCE, True)
        elif phase == "evolve":
            self.node.run_stage1(_EVOLVE_STEPS, _CANDIDATES, _PATIENCE, False)
        else:
            self.node.run_stage2(1.0, 1, 0.05, 1e-15)
        return phase

    def _record(self, phase):
        # Re-read `st` every frame: `step()` COMMITS BY REPLACING the complex
        # (`spacetime_ = build(bestSnapshot)`), so a cached handle silently describes a
        # complex the node no longer holds.
        st = self.node.st
        gradient = cob.MultiCobordism.regge_action_gradient(st)
        residual = self.node.r_u(st)
        self.hist["cells"].append(len(st.getTopSimplices()))
        self.hist["edges"].append(len(st.getEdgeList().toVector()))
        self.hist["F"].append(self.node.objective())
        self.hist["grad"].append(gradient)
        self.hist["gamma_ru"].append(self.gamma * residual)
        self.hist["holes"].append(len(cob.MultiCobordism.emergent_holes(st, self.k)))
        self.hist["phase"].append(phase)
        return st

    # ---- panels ----
    def _draw_size(self, ax):
        ax.clear()
        cells, edges = self.hist["cells"], self.hist["edges"]
        ax.plot(cells, color="C0", lw=1.6, label="top cells")
        ax.plot(edges, color="C7", lw=1.0, ls="--", label="edges")
        self._mark_phases(ax)
        net = (cells[-1] - cells[0]) if len(cells) > 1 else 0
        ax.set_title(f"size — top cells and edges  (net {net:+d} cells)", fontsize=8)
        ax.set_xlabel("frame", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=6)

    def _draw_objective(self, ax):
        ax.clear()
        ax.semilogy(np.maximum(self.hist["F"], 1e-30), color="C0", lw=1.6, label="F")
        ax.semilogy(np.maximum(self.hist["grad"], 1e-30), color="C1", lw=1.0,
                    label=r"$\|\nabla S\|^2$")
        ax.semilogy(np.maximum(self.hist["gamma_ru"], 1e-30), color="C2", lw=1.0,
                    label=r"$\Gamma\cdot r_U$")
        self._mark_phases(ax)
        # Descriptive, with the CURRENT split stated: whether the register term is at
        # its leak floor or already carried varies by node (the joint node carries
        # almost immediately; a node with pinned inputs sits at the floor for a long
        # time), so the title must not assert either.
        ru = self.hist["gamma_ru"][-1] if self.hist["gamma_ru"] else float("nan")
        grad = self.hist["grad"][-1] if self.hist["grad"] else float("nan")
        ax.set_title(f"objective — $\\|\\nabla S\\|^2$={grad:.4g}  "
                     f"$\\Gamma\\cdot r_U$={ru:.3g}", fontsize=8)
        ax.set_xlabel("frame", fontsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)

    def _draw_net_change(self, ax):
        """Signed Δ(top cells) per frame. Alternating bars of similar height are the
        trap door and the descent cancelling each other out."""
        ax.clear()
        cells = self.hist["cells"]
        deltas = np.diff(cells) if len(cells) > 1 else np.zeros(0)
        if len(deltas):
            colors = ["C2" if d > 0 else ("C3" if d < 0 else "0.8") for d in deltas]
            ax.bar(range(1, len(deltas) + 1), deltas, color=colors, width=0.9)
        ax.axhline(0.0, color="0.4", lw=0.8)
        grew = int(sum(1 for d in deltas if d > 0))
        shrank = int(sum(1 for d in deltas if d < 0))
        net = int(deltas.sum()) if len(deltas) else 0
        ax.set_title(f"net size change — {grew} grew, {shrank} shrank, net {net:+d}",
                     fontsize=8)
        ax.set_xlabel("frame", fontsize=7)
        ax.tick_params(labelsize=6)

    def _mark_phases(self, ax):
        """Dashed boundaries between the init, evolve and relax phases."""
        for boundary in (_INIT_PASSES, _INIT_PASSES + _EVOLVE_PASSES):
            if len(self.hist["F"]) > boundary:
                ax.axvline(boundary - 0.5, color="0.6", lw=0.8, ls=":")

    def _draw_complex(self, ax, st, coords):
        ax.clear()
        holes = cob.MultiCobordism.emergent_holes(st, self.k)
        hole_vsets = [set(h) for h in holes]
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a not in coords or b not in coords:
                continue
            p, q = coords[a], coords[b]
            if any(a in vs and b in vs for vs in hole_vsets):
                ax.plot([p[0], q[0]], [p[1], q[1]], color="C3", lw=1.8, zorder=3)
            else:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=0.5, zorder=1)
        if coords:
            pts = np.array(list(coords.values()))
            ax.scatter(pts[:, 0], pts[:, 1], c="0.15", s=28, zorder=2)
            for i, h in enumerate(holes):
                hp = np.array([coords[v] for v in h if v in coords])
                if len(hp):
                    c = hp.mean(0)
                    ax.text(c[0], c[1], str(i + 1), color="white", fontsize=8,
                            fontweight="bold", ha="center", va="center", zorder=4,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="C3", ec="white",
                                      lw=0.8))
            if len(coords) >= 2:
                view = self.layout.view(coords)
                ax.set_xlim(view[0], view[1])
                ax.set_ylim(view[2], view[3])
        n = len(holes)
        ax.set_aspect("equal")
        ax.set_title(f"complex — {len(st.getTopSimplices())} cells, "
                     f"{len(coords)} vertices, {n} register{'s' if n != 1 else ''}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    def _curvature(self, st, frame):
        """`_cell_curvature` is expensive; refresh it every `_HEAT_REFRESH_EVERY` frames
        and always on the last one. Reuses the existing implementation verbatim."""
        stale = (self._curv_cache is None
                 or frame - self._curv_cache[0] >= _HEAT_REFRESH_EVERY
                 or frame >= self.frames - 1)
        if stale:
            self._curv_cache = (frame, ProtonAnimator._cell_curvature(st))
        return self._curv_cache[1]

    def _draw_dual(self, ax, st, coords, curv, channel, title):
        """One dual node per top cell at that cell's primal centroid, edges from the COO
        `(rows, cols, N)` adjacency — indices into `getTopSimplices()` order, so the two
        must be read in the same order."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        ax.clear()
        tops = st.getTopSimplices()
        positions = np.full((len(tops), 2), np.nan)
        values = np.zeros(len(tops))
        for i, c in enumerate(tops):
            cell = tuple(sorted(v.getId() for v in c.getVertices()))
            pts = np.array([coords[v] for v in cell if v in coords])
            if len(pts):
                positions[i] = pts.mean(0)
            values[i] = curv.get(cell, (0.0, 0.0))[channel]
        finite = np.all(np.isfinite(positions), axis=1)
        if not finite.any():
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            return
        rows, cols, _n = st.getDualAdjacency()
        for a, b in zip(rows, cols):
            if a < len(tops) and b < len(tops) and finite[a] and finite[b]:
                ax.plot([positions[a, 0], positions[b, 0]],
                        [positions[a, 1], positions[b, 1]],
                        color="0.85", lw=0.5, zorder=1)
        span = float(np.max(np.abs(values[finite])))
        # A dead-flat channel is a FINDING, not a broken panel: Im ε ≡ 0 means no
        # spacelike hinge carries boost content, i.e. the complex holds no timelike
        # edges at all. Say so, rather than rendering a uniform blank that reads as a
        # rendering failure. The scale is printed for the same reason — a diverging
        # colormap normalized to its own max looks identical at 1e-30 and at 1.
        if span <= 1e-15:
            ax.text(0.5, 0.5, "identically zero\n(no timelike hinges)", fontsize=8,
                    color="0.45", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}  —  ≡ 0", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            return
        norm = mcolors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
        ax.scatter(positions[finite, 0], positions[finite, 1], c=values[finite],
                   cmap=plt.get_cmap("coolwarm" if channel == 0 else "PuOr"),
                   norm=norm, s=40, zorder=2, edgecolors="white", lw=0.4)
        view = self.layout.view(coords)
        ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")
        ax.set_title(f"{title}  —  ±{span:.3g}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- run ----
    def run(self, live=False, save=None, interval=200):
        self.gamma = _GAMMA
        if not (live or save):
            for frame in range(self.frames):
                phase = self._advance(frame)
                self._record(phase)
            return self.summary()

        import matplotlib
        if save and not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        fig.suptitle(f"lattice growth — {self.label} — from a single Δ⁴", fontsize=11)

        def update(frame):
            phase = self._advance(frame)
            st = self._record(phase)
            coords = self.layout.coords(st)
            curv = self._curvature(st, frame)
            self._draw_size(axes[0][0])
            self._draw_objective(axes[0][1])
            self._draw_net_change(axes[0][2])
            self._draw_complex(axes[1][0], st, coords)
            self._draw_dual(axes[1][1], st, coords, curv, 0,
                            r"dual — spatial curvature  $\mathrm{Re}\,\epsilon\cdot|\star|$")
            self._draw_dual(axes[1][2], st, coords, curv, 1,
                            r"dual — temporal curvature  $\mathrm{Im}\,\epsilon\cdot|\star|$")
            fig.suptitle(
                f"lattice growth — {self.label} — frame {frame + 1}/{self.frames} "
                f"[{phase}]   cells={self.hist['cells'][-1]}  "
                f"F={self.hist['F'][-1]:.4g}  "
                f"registers={self.hist['holes'][-1]}", fontsize=11)
            return []

        anim = FuncAnimation(fig, update, frames=self.frames, interval=interval,
                             blit=False, repeat=False)
        if save:
            anim.save(save, fps=max(1, int(1000 / interval)))
            print(f"wrote {save}")
        if live:
            plt.show()
        return self.summary()

    def summary(self):
        cells = self.hist["cells"]
        deltas = np.diff(cells) if len(cells) > 1 else np.zeros(0)
        return {
            "final_cells": cells[-1] if cells else 0,
            "max_cells": max(cells) if cells else 0,
            "grew": int(sum(1 for d in deltas if d > 0)),
            "shrank": int(sum(1 for d in deltas if d < 0)),
            "final_F": self.hist["F"][-1] if self.hist["F"] else float("nan"),
            "final_grad": self.hist["grad"][-1] if self.hist["grad"] else float("nan"),
            "final_gamma_ru": (self.hist["gamma_ru"][-1] if self.hist["gamma_ru"]
                               else float("nan")),
            "registers": self.hist["holes"][-1] if self.hist["holes"] else 0,
        }


_GAMMA = 50.0


def build_node(which, seed):
    """The canonical factory nodes — NOT a hand-rolled MultiCobordism. Both seed a single
    Δ⁴ via `Proton::buildMinimalSeed` with `precone=0`, so the topology is entirely
    emergent."""
    if which == "formation":
        return cob.Proton(seed=seed).formation_node(seed), "formation node (singlet pinned)"
    return (cob.ProtonIngredients(seed=seed).joint_node(seed),
            "joint node (three neutral pairs, nothing pinned)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="interactive window")
    ap.add_argument("--save", help="write a GIF/MP4 (no display needed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--node", choices=("joint", "formation"), default="joint")
    ap.add_argument("--interval", type=int, default=200, help="ms per frame")
    args = ap.parse_args()

    node, label = build_node(args.node, args.seed)
    animator = LatticeGrowthAnimator(node, label)
    summary = animator.run(live=args.live, save=args.save, interval=args.interval)
    print(f"final cells      : {summary['final_cells']} (max {summary['max_cells']})")
    print(f"frames grew/shrank: {summary['grew']} / {summary['shrank']}")
    print(f"registers        : {summary['registers']}")
    print(f"F                : {summary['final_F']:.6g}  "
          f"= ||grad S||^2 {summary['final_grad']:.6g} "
          f"+ gamma*r_U {summary['final_gamma_ru']:.6g}")


if __name__ == "__main__":
    main()
