# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Real-simplex render of the hierarchical merge sequence (companion to
``merge_cobordism.py``). Every vertex/edge is from a built MergeCobordism;
time is the vertical axis, the register layout is the horizontal plane."""

from __future__ import annotations

import os

import numpy as np

try:
    import networkx as nx
except Exception:                       # pragma: no cover
    nx = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection   # noqa: E402

# states (the four inputs) + bulks (the three merges)
cA, cB, cC, cD = "#0072B2", "#E69F00", "#009E73", "#D55E00"
cAB, cCD, cABCD = "#CC79A7", "#56B4E9", "#B8860B"


def _register_layout(merge):
    """A 2D layout of the 12-vertex register surface from the merge's
    intra-input-A edges (a real spring layout, deterministic seed)."""
    if nx is None:
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        P = np.c_[np.cos(ang), np.sin(ang)]
        return P
    G = nx.Graph()
    G.add_nodes_from(range(12))
    for (a, b) in merge.edges():
        if a < 12 and b < 12:
            G.add_edge(a, b)
    pos = nx.spring_layout(G, seed=3, k=1.3)
    P = np.array([pos[v] for v in range(12)])
    P -= P.mean(0)
    P /= np.abs(P).max()
    return P


def _draw_merge(ax, merge, P, offA, offB, offR, tb, tt, cIA, cIB, cR):
    """Draw one merge's real 1-skeleton: input A at (P+offA, tb), input B at
    (P+offB, tb), result R at (P+offR, tt); intra-block edges in the block
    colors, the timelike input→result bulk edges thin/faint in the result
    color."""
    blocks = {0: (offA, tb, cIA), 1: (offB, tb, cIB), 2: (offR, tt, cR)}

    def xyz(v):
        off, t, _ = blocks[v // 12]
        x, y = P[v % 12]
        return (x + off, y, t)

    intra, bulk = [], []
    for (a, b) in merge.edges():
        seg = [xyz(a), xyz(b)]
        (intra if a // 12 == b // 12 else bulk).append((seg, a // 12))
    ax.add_collection3d(Line3DCollection(
        [s for s, _ in bulk], colors=cR, lw=0.6, alpha=0.4))
    for blk, c in ((0, cIA), (1, cIB), (2, cR)):
        segs = [s for s, k in intra if k == blk]
        ax.add_collection3d(Line3DCollection(segs, colors=c, lw=2.2))
        pts = np.array([xyz(v) for v in range(12 * blk, 12 * blk + 12)])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=12,
                   depthshade=False)


def _style(ax, title, tmax):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=0)
    ax.set_zlim(-0.25, tmax + 0.25)
    ax.set_zticks(range(tmax + 1))
    ax.set_zticklabels([f"t={i}" for i in range(tmax + 1)], fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=15, azim=-71)
    ax.set_box_aspect((3.0, 1.1, 1.5))


def render(merge, outdir):
    """The merge sequence on the shared register layout (the merge's real
    1-skeleton drawn three times at the tower positions: AB and CD at t:0→1,
    ABCD at t:1→2). Returns the saved path."""
    P = _register_layout(merge)
    # tower x-offsets: AB-inputs, AB-result; CD-inputs, CD-result; ABCD-result
    Aoff, Boff, ABr = -4.4, -1.8, -3.1
    Coff, Doff, CDr = 1.8, 4.4, 3.1
    ABCDr = 0.0

    fig = plt.figure(figsize=(16, 5.8))

    a1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_merge(a1, merge, P, Aoff, Boff, ABr, 0, 1, cA, cB, cAB)
    _draw_merge(a1, merge, P, Coff, Doff, CDr, 0, 1, cC, cD, cCD)
    for off, lb, c in [(Aoff, r"$\psi_A$", cA), (Boff, r"$\psi_B$", cB),
                       (Coff, r"$\psi_C$", cC), (Doff, r"$\psi_D$", cD)]:
        a1.text(off, 0, -0.28, lb, color=c, fontsize=9, ha="center")
    a1.text(ABr, 0, 1.22, r"$\mathrm{geo}(U_{AB})$", color=cAB, fontsize=8.5, ha="center")
    a1.text(CDr, 0, 1.22, r"$\mathrm{geo}(U_{CD})$", color=cCD, fontsize=8.5, ha="center")
    _style(a1, "1 · two merges  (ψ at t=0 → bulk at t=1)", 1)

    a2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_merge(a2, merge, P, ABr, CDr, ABCDr, 1, 2, cAB, cCD, cABCD)
    a2.text(ABr, 0, 0.78, r"$\mathrm{geo}(U_{AB})$", color=cAB, fontsize=8.5, ha="center")
    a2.text(CDr, 0, 0.78, r"$\mathrm{geo}(U_{CD})$", color=cCD, fontsize=8.5, ha="center")
    a2.text(ABCDr, 0, 2.22, r"$\mathrm{geo}(U_{ABCD})$", color=cABCD, fontsize=8.5, ha="center")
    _style(a2, "2 · the results merge  (now at t=1 → bulk at t=2)", 2)

    a3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_merge(a3, merge, P, Aoff, Boff, ABr, 0, 1, cA, cB, cAB)
    _draw_merge(a3, merge, P, Coff, Doff, CDr, 0, 1, cC, cD, cCD)
    _draw_merge(a3, merge, P, ABr, CDr, ABCDr, 1, 2, cAB, cCD, cABCD)
    _style(a3, "3 · full sequence", 2)

    leg = [Line2D([0], [0], color=c, lw=3, label=l) for c, l in
           [(cA, r"$\psi_A$"), (cB, r"$\psi_B$"), (cC, r"$\psi_C$"),
            (cD, r"$\psi_D$"), (cAB, r"geo($U_{AB}$)"),
            (cCD, r"geo($U_{CD}$)"), (cABCD, r"geo($U_{ABCD}$)")]]
    fig.legend(handles=leg, loc="lower center", ncol=7, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Hierarchical MERGE sequence — real synthesized simplices "
                 "(inputs spatial at t, bulk merges to a single object at t+1)",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "merge_cobordism.png")
    fig.savefig(path, dpi=120, facecolor="white")
    return path
