#!/usr/bin/env python3
"""Render a GIF showing each spatial time slice as a 2D radial embedding
centered on the point-mass worldline vertex, with curvature encoded as a heat map.

Each frame = one time slice.  Vertices are placed at radius = BFS distance
from the worldline vertex and spread angularly by force-directed layout.
Color = area-weighted deficit angle.  The point mass is always at the
visual center.

Usage (standalone):
    python examples/curvature_slice_gif.py --n-simplices 50 --mass 1.0

Usage (as a library):
    from curvature_slice_gif import render_curvature_gif
    render_curvature_gif(st, solver, worldline, "curvature.gif")
"""
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection
from PIL import Image

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.plot import (build_spacetime, time_slices, spatial_subgraph,
                              bfs_distances, save_gif, radial_layout_2d)
from tessera.utils.progress import SingleTaskProgress


# =========================================================================
# Per-vertex curvature
# =========================================================================

def _vertex_curvatures(verts, solver, t):
    """Compute area-weighted average deficit angle for each vertex at time *t*."""
    curvatures = np.zeros(len(verts))
    for idx, v in enumerate(verts):
        total = 0.0
        count = 0
        for s in v.getSimplices():
            sv = s.getVertices()
            if len(sv) != 3:
                continue
            if not all(round(hv.getTime()) == t for hv in sv):
                continue
            eps = solver.lorentzianDeficitAngle(s)
            area = tessera.ReggeSolver.hingeArea(s)
            total += eps * area
            count += 1
        if count > 0:
            curvatures[idx] = total / count
    return curvatures


# =========================================================================
# 2D radial layout: center at origin, radius = BFS distance
# =========================================================================

def _radial_layout_2d(verts, edges, center_vid, bfs_dist, *,
                      iters=200, prev_angles=None, seed=42):
    """Place vertices in 2D: radius = BFS distance, angles from force layout.

    The radius-constrained angular solve runs in C++
    (``tessera.ForceLayout.layout2D`` via :func:`radial_layout_2d`); this
    wrapper keeps the per-frame angular-continuity bookkeeping in Python:
    seeding initial angles from the previous frame and recording the solved
    angles for the next one.

    Returns (pos_2d, vid_to_idx, edge_idx, angles_by_dist) where
    angles_by_dist maps BFS distance to list of (angle, vid) for seeding
    the next frame.
    """
    rng = np.random.default_rng(seed)
    n = len(verts)
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    center_idx = vid_to_idx[center_vid]

    # Group vertices by BFS distance
    max_d = max(bfs_dist.values()) if bfs_dist else 1

    # Initialize: place on circles by BFS distance with angular spread.
    # Radius is pinned to the BFS distance; the center stays at the origin.
    init_pos = np.zeros((n, 2))
    target_radii = np.zeros(n)
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        r = float(d)
        target_radii[idx] = r
        # Try to use previous frame's angular structure for stability
        if prev_angles and d in prev_angles and prev_angles[d]:
            # Pick the angle closest to a uniform distribution
            used = len([1 for vi in verts[:idx]
                       if bfs_dist.get(vi.getId(), -1) == d])
            n_at_d = sum(1 for vi in verts
                        if bfs_dist.get(vi.getId(), -1) == d)
            base_angle = prev_angles[d][0] if prev_angles[d] else 0
            angle = base_angle + 2 * math.pi * used / max(n_at_d, 1)
        else:
            angle = rng.uniform(0, 2 * math.pi)
        init_pos[idx] = [r * math.cos(angle), r * math.sin(angle)]

    # Build edge index
    edge_idx = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))

    # Radius-constrained, tangential-only angular solve (C++).
    pos = radial_layout_2d(n, edge_idx, target_radii, center_idx=center_idx,
                           init_pos=init_pos, iters=iters, seed=seed)

    # Record angles for next frame
    angles_by_dist = {}
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        angle = math.atan2(pos[idx, 1], pos[idx, 0])
        angles_by_dist.setdefault(d, []).append(angle)
    for d in angles_by_dist:
        angles_by_dist[d].sort()

    return pos, vid_to_idx, edge_idx, angles_by_dist


# =========================================================================
# Render a 2D frame
# =========================================================================

def _render_2d_frame(pos, edge_idx, curvatures, center_idx, t,
                     vmin, vmax, fig_size, cmap_name, axis_limit):
    """Render a 2D radial frame with curvature heatmap."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect("equal")

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Draw edges
    if edge_idx:
        segs = [[pos[s], pos[t]] for s, t in edge_idx]
        lc = LineCollection(segs, linewidths=0.5, colors=(0.7, 0.7, 0.7, 0.5))
        ax.add_collection(lc)

    # Draw vertices colored by curvature
    sizes = np.full(len(pos), 30.0)
    sizes[center_idx] = 120.0
    ax.scatter(pos[:, 0], pos[:, 1],
               c=curvatures, cmap=cmap, norm=norm,
               s=sizes, edgecolors="k", linewidths=0.3, zorder=5)

    # Center vertex star
    ax.scatter([pos[center_idx, 0]], [pos[center_idx, 1]],
               c="black", s=150, marker="*", zorder=10)

    # Fixed axis limits centered on origin
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_axis_off()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Curvature (A * deficit)", fontsize=9)

    ax.set_title(f"t = {t}", fontsize=12, pad=10)

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# =========================================================================
# Public entry point
# =========================================================================

def render_curvature_gif(st, solver, worldline, output_path="curvature.gif",
                         *, fig_size=(6, 6), cmap_name="RdYlBu_r",
                         layoutIters=200, frame_duration_ms=500,
                         **_kwargs):
    """Render a GIF with one frame per time slice showing the spatial
    subgraph as a 2D radial embedding with curvature heat map.

    The point mass (worldline vertex) is always at the visual center.
    Radius = BFS graph distance.  Color = area-weighted deficit angle.
    """
    wl_by_time = {}
    for v in worldline:
        wl_by_time[round(v.getTime())] = v

    times = time_slices(st)

    # --- Pass 1: compute layouts and curvatures ---
    slice_data = []
    prev_angles = None
    for t in times:
        if t not in wl_by_time:
            continue
        center = wl_by_time[t]
        verts, edges = spatial_subgraph(st, t)
        if len(verts) < 3:
            continue

        bfs_dist = bfs_distances(center, st)
        reachable_ids = set(bfs_dist.keys())
        verts = [v for v in verts if v.getId() in reachable_ids]
        edges = [e for e in edges
                 if e.getSource().getId() in reachable_ids
                 and e.getTarget().getId() in reachable_ids]
        if len(verts) < 3:
            continue

        pos, vid_to_idx, edge_idx, prev_angles = _radial_layout_2d(
            verts, edges, center.getId(), bfs_dist,
            iters=layoutIters, prev_angles=prev_angles)

        curvatures = _vertex_curvatures(verts, solver, t)
        center_idx = vid_to_idx[center.getId()]
        slice_data.append((t, pos, edge_idx, curvatures, center_idx))

    if not slice_data:
        print("  No renderable time slices found.")
        return output_path

    # Global curvature scale
    all_curv = np.concatenate([d[3] for d in slice_data])
    vmin, vmax = float(all_curv.min()), float(all_curv.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 0.5
        vmax += 0.5

    # Global axis limit: max radius across all frames + padding
    max_radius = max(np.linalg.norm(d[1], axis=1).max() for d in slice_data)
    axis_limit = max_radius * 1.3

    # --- Pass 2: render frames ---
    frames = []
    for t, pos, edge_idx, curvatures, center_idx in slice_data:
        img = _render_2d_frame(pos, edge_idx, curvatures, center_idx, t,
                               vmin, vmax, fig_size, cmap_name, axis_limit)
        frames.append(img)

    save_gif(frames, output_path, duration_ms=frame_duration_ms)
    return output_path


# =========================================================================
# Standalone CLI
# =========================================================================

def main():
    monitor = MemoryMonitor()
    p = argparse.ArgumentParser(
        description="Curvature-slice GIF: spatial slices with heat map")
    p.add_argument("--n-simplices", type=int, default=50)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--save", type=str, default="curvature_slices.gif")
    args = p.parse_args()

    prog = SingleTaskProgress(memory_monitor=monitor)
    prog.phase("building", extra=f"{args.n_simplices} simplices")

    st, _ = build_spacetime(args.n_simplices)

    verts = st.getVertexList().toVector()
    center = max(verts, key=lambda v: v.degree())

    matter = tessera.MatterConfiguration()
    worldline = tessera.MatterConfiguration.buildWorldline(center, st)
    matter.setWorldlineMass(center, args.mass, st)

    solver = tessera.ReggeSolver(st, matter)
    prog.phase("solving", total=args.max_iters)
    converged, F, iters = solver.solve(
        tol=args.tol, maxIters=args.max_iters,
        learningRate=args.learning_rate)
    print(f"{'Converged' if converged else 'Did not converge'} "
          f"after {iters} iters, F={F:.6f}")

    prog.phase("rendering", extra=args.save)
    render_curvature_gif(st, solver, worldline, args.save)
    prog.finish(f"saved {args.save}")


if __name__ == "__main__":
    main()
