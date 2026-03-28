#!/usr/bin/env python3
"""Render a GIF showing each spatial time slice as a 3D force-directed embedding
centered on the point-mass worldline vertex, with curvature encoded as a heat map.

Each frame = one time slice.  Vertices are colored by the area-weighted deficit
angle summed over incident spatial hinges.  The worldline vertex is pinned at the
origin and marked with a star.

Usage (standalone):
    python examples/curvature_slice_gif.py --n-simplices 50 --mass 1.0

Usage (from regge_point_mass.py):
    from curvature_slice_gif import render_curvature_gif
    render_curvature_gif(st, solver, worldline, "curvature.gif")
"""
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import caset
from plot_utils import (build_spacetime, time_slices, spatial_subgraph,
                        bfs_distances, force_layout_3d, pca_align,
                        draw_edges, render_frame, save_gif)


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
            eps = solver.deficitAngle(s)
            area = caset.ReggeSolver.hingeArea(s)
            total += eps * area
            count += 1
        if count > 0:
            curvatures[idx] = total / count
    return curvatures


# =========================================================================
# Layout with BFS-based initialization
# =========================================================================

def _layout_centered(verts, edges, center_vid, bfs_dist, *,
                     iters=300, seed=42):
    """Force-directed layout with center pinned at origin, initialized
    on concentric spheres by BFS distance."""
    rng = np.random.default_rng(seed)
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    center_idx = vid_to_idx[center_vid]
    n = len(verts)

    # Initialize on concentric spheres
    max_d = max(bfs_dist.values()) if bfs_dist else 1
    init_pos = np.zeros((n, 3))
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        r = float(d)
        theta = rng.uniform(0, math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        init_pos[idx] = [r * math.sin(theta) * math.cos(phi),
                         r * math.sin(theta) * math.sin(phi),
                         r * math.cos(theta)]

    # Build edge index and rest lengths
    edge_idx = []
    rest_lens = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))
            rest_lens.append(math.sqrt(abs(e.getSquaredLength())))

    pos = force_layout_3d(n, edge_idx, center_idx=center_idx,
                          init_pos=init_pos, rest_lengths=rest_lens,
                          iters=iters, seed=seed)
    return pos, vid_to_idx, edge_idx


# =========================================================================
# Public entry point
# =========================================================================

def render_curvature_gif(st, solver, worldline, output_path="curvature.gif",
                         *, fig_size=(6, 6), cmap_name="RdYlBu_r",
                         layout_iters=300, frame_duration_ms=500,
                         elev=25.0, azim=45.0):
    """Render a GIF with one frame per time slice showing the spatial
    subgraph as a force-directed 3D embedding with curvature heat map."""
    wl_by_time = {}
    for v in worldline:
        wl_by_time[round(v.getTime())] = v

    times = time_slices(st)

    # --- Pass 1: compute layouts and curvatures ---
    slice_data = []
    prev_axes = None
    for t in times:
        if t not in wl_by_time:
            continue
        center = wl_by_time[t]
        verts, edges = spatial_subgraph(st, t)
        if len(verts) < 3:
            continue

        bfs_dist = bfs_distances(center, verts, edges)
        reachable_ids = set(bfs_dist.keys())
        verts = [v for v in verts if v.getId() in reachable_ids]
        edges = [e for e in edges
                 if e.getSource().getId() in reachable_ids
                 and e.getTarget().getId() in reachable_ids]
        if len(verts) < 3:
            continue

        pos, vid_to_idx, edge_idx = _layout_centered(
            verts, edges, center.getId(), bfs_dist, iters=layout_iters)
        pos, prev_axes = pca_align(pos, prev_axes)

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

    # --- Pass 2: render frames ---
    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    frames = []
    for t, pos, edge_idx, curvatures, center_idx in slice_data:

        def draw(ax, _pos=pos, _edge_idx=edge_idx, _curv=curvatures,
                 _cidx=center_idx, _t=t):
            draw_edges(ax, _pos, _edge_idx)
            sizes = np.full(len(_pos), 20.0)
            sizes[_cidx] = 80.0
            ax.scatter(_pos[:, 0], _pos[:, 1], _pos[:, 2],
                       c=_curv, cmap=cmap, norm=norm,
                       s=sizes, edgecolors="k", linewidths=0.2,
                       depthshade=True)
            ax.scatter([_pos[_cidx, 0]], [_pos[_cidx, 1]], [_pos[_cidx, 2]],
                       c="black", s=100, marker="*", zorder=10)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                         label="Curvature (A * deficit)")

        img = render_frame(draw, figsize=fig_size, title=f"t = {t}",
                           elev=elev, azim=azim)
        frames.append(img)

    save_gif(frames, output_path, duration_ms=frame_duration_ms)
    return output_path


# =========================================================================
# Standalone CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Curvature-slice GIF: spatial slices with heat map")
    p.add_argument("--n-simplices", type=int, default=50)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--save", type=str, default="curvature_slices.gif")
    args = p.parse_args()

    st, _ = build_spacetime(args.n_simplices)

    verts = st.getVertexList().toVector()
    center = max(verts, key=lambda v: v.degree())

    matter = caset.MatterConfiguration()
    worldline = caset.MatterConfiguration.buildWorldline(center, st)
    matter.setWorldlineMass(center, args.mass, st)

    solver = caset.ReggeSolver(st, matter)
    print(f"Solving (max {args.max_iters} iters)...")
    converged, F, iters = solver.solve(
        tol=args.tol, max_iters=args.max_iters,
        learning_rate=args.learning_rate)
    print(f"{'Converged' if converged else 'Did not converge'} "
          f"after {iters} iters, F={F:.6f}")

    print(f"Rendering {args.save}...")
    render_curvature_gif(st, solver, worldline, args.save)
    print(f"Done. Output: {args.save}")


if __name__ == "__main__":
    main()
