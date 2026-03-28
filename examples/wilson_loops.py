#!/usr/bin/env python3
"""Wilson loop visualization: compute and display loops in all three modes.

Builds a CDT spacetime, generates hinge/geodesic/dual-lattice loops,
evaluates each in COMBINATORIAL, DEFICIT_ANGLE, and CAUSAL modes, then
renders the loop path overlaid on a 3D force-directed layout of the
spatial slice.

Usage:
    python examples/wilson_loops.py
    python examples/wilson_loops.py --n-simplices 200 --save wilson.gif
"""
import argparse
import math
from collections import defaultdict, deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

import caset


# =========================================================================
# Layout helpers (adapted from curvature_slice_gif.py)
# =========================================================================

def _slice_subgraph(st, t):
    verts = [v for v in st.getVertexList().toVector()
             if round(v.getTime()) == t]
    vid_set = {v.getId() for v in verts}
    edges = []
    seen = set()
    for v in verts:
        for e in v.getEdges():
            fp = id(e)
            if fp in seen:
                continue
            seen.add(fp)
            s, t_ = e.getSource().getId(), e.getTarget().getId()
            if s in vid_set and t_ in vid_set and e.getSquaredLength() > 0:
                edges.append(e)
    return verts, edges


def _force_layout_3d(verts, edges, *, iters=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(verts)
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    pos = rng.standard_normal((n, 3)) * 0.5

    edge_idx = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))

    step = 0.5
    for _ in range(iters):
        forces = np.zeros_like(pos)
        for si, ti in edge_idx:
            d = pos[ti] - pos[si]
            dist = max(np.linalg.norm(d), 1e-6)
            f = 0.01 * (dist - 1.0) * d / dist
            forces[si] += f
            forces[ti] -= f
        cap = min(n, 200)
        for a in range(cap):
            for b in range(a + 1, cap):
                d = pos[a] - pos[b]
                d2 = np.dot(d, d) + 1e-6
                f = 0.5 / d2 * d / math.sqrt(d2)
                forces[a] += f
                forces[b] -= f
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        forces = np.where(norms > step, forces / norms * step, forces)
        pos += forces
        step *= 0.995
    return pos, vid_to_idx, edge_idx


# =========================================================================
# Dual-graph layout: one node per top-simplex, edges between adjacent ones
# =========================================================================

def _loop_layout(loop, *, iters=300, rng=None):
    """Force-directed layout of ONLY the loop simplices as a cycle graph.

    Each loop simplex is a node.  Edges connect consecutive simplices
    (including the closing edge).  Returns (positions, loop_edges).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(loop.simplices)
    if n < 2:
        return np.zeros((max(n, 1), 3)), []

    # Cycle edges: 0-1, 1-2, ..., (n-1)-0
    edges = [(i, (i + 1) % n) for i in range(n)]

    # Initialize on a circle in the xy-plane, then perturb in z
    pos = np.zeros((n, 3))
    for i in range(n):
        angle = 2 * math.pi * i / n
        pos[i] = [math.cos(angle), math.sin(angle),
                  rng.uniform(-0.1, 0.1)]

    # Force-directed refinement
    step = 0.3
    for _ in range(iters):
        forces = np.zeros_like(pos)
        for a, b in edges:
            d = pos[b] - pos[a]
            dist = max(np.linalg.norm(d), 1e-6)
            f = 0.1 * (dist - 1.0) * d / dist
            forces[a] += f
            forces[b] -= f
        for a in range(n):
            for b in range(a + 1, n):
                d = pos[a] - pos[b]
                d2 = np.dot(d, d) + 1e-6
                f = 0.5 / d2 * d / math.sqrt(d2)
                forces[a] += f
                forces[b] -= f
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        forces = np.where(norms > step, forces / norms * step, forces)
        pos += forces
        step *= 0.995

    return pos, edges


def _render_loop_frame(pos, edges, title,
                       fig_size=(7, 7), elev=25, azim=45):
    """Render the loop as a clean cycle with numbered nodes."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    n = len(pos)

    # Draw loop edges
    if edges:
        segs = [[pos[a], pos[b]] for a, b in edges]
        lc = Line3DCollection(segs, linewidths=2.5,
                              colors="red", alpha=0.9)
        ax.add_collection(lc)

    # Draw loop nodes with index labels
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c="red", s=80, edgecolors="black", linewidths=0.8,
               zorder=10, depthshade=True)
    for i in range(n):
        ax.text(pos[i, 0], pos[i, 1], pos[i, 2] + 0.15,
                str(i), fontsize=8, ha="center", va="bottom",
                fontweight="bold", zorder=11)

    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    fig.subplots_adjust(right=0.95, left=0.05, top=0.92, bottom=0.05)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# =========================================================================
# Main
# =========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Wilson loop visualization across three evaluation modes")
    p.add_argument("--n-simplices", type=int, default=50)
    p.add_argument("--save", type=str, default="wilson_loops.gif")
    args = p.parse_args()

    # Build spacetime
    print(f"Building spacetime with {args.n_simplices} simplices...")
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)
    target = st.getN41()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
    cdt.tune()
    cdt.sweep(10)
    print(f"  Vertices: {st.getVertexCount()}, "
          f"Top simplices: {st.getSimplexCount()}")

    # Create WilsonLoop
    wl = caset.WilsonLoop(st)

    # Find the hinge with the largest fan (most top-simplices around it)
    hinge = None
    best_fan = 0
    for s in st.getSimplices():
        if len(s.getVertices()) != 3:
            continue
        loop_candidate = wl.hingeLoop(s)
        fan = len(loop_candidate)
        if fan > best_fan:
            best_fan = fan
            hinge = s

    # Find a top-simplex for geodesic/dual-lattice
    start_simplex = None
    for s in st.getSimplices():
        if len(s.getVertices()) == 5:
            start_simplex = s
            break

    # Generate loops
    loops = {}
    if hinge is not None:
        loop = wl.hingeLoop(hinge)
        if len(loop) >= 2:
            loops["Hinge"] = loop
    if start_simplex is not None:
        loop = wl.geodesicLoop(start_simplex)
        if len(loop) >= 2:
            loops["Geodesic"] = loop
        loop = wl.dualLatticeLoop(start_simplex, 8)
        if len(loop) >= 2:
            loops["Dual-lattice"] = loop

    if not loops:
        print("No loops found. Try a larger spacetime.")
        return

    # Evaluate and print results
    modes = [
        ("COMBINATORIAL", caset.WilsonMode.COMBINATORIAL),
        ("DEFICIT_ANGLE", caset.WilsonMode.DEFICIT_ANGLE),
        ("CAUSAL",        caset.WilsonMode.CAUSAL),
    ]

    print(f"\n{'Loop':<15} {'Mode':<16} {'Value':>8} {'Size':>5} "
          f"{'Hinges':>7} {'Winding':>8}")
    print("-" * 65)
    for loop_name, loop in loops.items():
        for mode_name, mode in modes:
            r = wl.evaluate(loop, mode)
            print(f"{loop_name:<15} {mode_name:<16} {r.value:>8.4f} "
                  f"{r.loopSize:>5} {r.enclosedHinges:>7} "
                  f"{r.causalWindingNumber:>8}")

    # Measure all hinge loops for area-law plot
    wl.measureAllHinges(caset.WilsonMode.DEFICIT_ANGLE)
    measurements = wl.getMeasurements()
    avg = wl.getAverageBySize()
    if avg:
        print(f"\nHinge loop averages by size:")
        for size, val in sorted(avg.items()):
            print(f"  size={size}: W={val:.4f}")

    # Render GIF: for each loop type, show rotating views
    frames = []
    azimuths = list(range(0, 360, 30))  # 12 rotation angles

    for loop_name, loop in loops.items():
        r = wl.evaluate(loop, caset.WilsonMode.DEFICIT_ANGLE)
        title = (f"{loop_name} loop (size={r.loopSize}, "
                 f"W={r.value:.3f})")

        loop_pos, loop_edges = _loop_layout(loop)

        for az in azimuths:
            img = _render_loop_frame(loop_pos, loop_edges, title,
                                     elev=25, azim=az)
            frames.append(img)

    # Assemble GIF
    pil_frames = [Image.fromarray(f[:, :, :3]) for f in frames]
    pil_frames[0].save(args.save, save_all=True,
                       append_images=pil_frames[1:],
                       duration=200, loop=0)
    print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
