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
from collections import deque
from itertools import permutations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

import caset


# =========================================================================
# Spatial subgraph extraction
# =========================================================================

def _time_slices(st):
    """Return sorted list of integer time values present in the spacetime."""
    times = set()
    for v in st.getVertexList().toVector():
        times.add(round(v.getTime()))
    return sorted(times)


def _slice_subgraph(st, t):
    """Return (vertices, spacelike_edges) for time slice *t*."""
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
            src_id = e.getSource().getId()
            tgt_id = e.getTarget().getId()
            if src_id in vid_set and tgt_id in vid_set and e.getSquaredLength() > 0:
                edges.append(e)
    return verts, edges


# =========================================================================
# BFS graph distance
# =========================================================================

def _bfs_distances(center, verts, edges):
    """BFS from *center* through *edges*.  Returns {vertex_id: distance}."""
    adj = {}
    for v in verts:
        adj[v.getId()] = []
    for e in edges:
        s, t = e.getSource().getId(), e.getTarget().getId()
        adj[s].append(t)
        adj[t].append(s)

    dist = {center.getId(): 0}
    queue = deque([center.getId()])
    while queue:
        vid = queue.popleft()
        for nbr in adj.get(vid, []):
            if nbr not in dist:
                dist[nbr] = dist[vid] + 1
                queue.append(nbr)
    return dist


# =========================================================================
# Force-directed 3D layout
# =========================================================================

def _force_layout_3d(verts, edges, center_vid, bfs_dist, *,
                     iters=300, spring_k=0.01, repulsion_k=0.5,
                     cooling=0.995, rng=None):
    """Spring-electrical layout in 3D.  Center vertex pinned at origin."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(verts)
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    center_idx = vid_to_idx[center_vid]

    # Initialise on concentric spheres by BFS distance
    max_d = max(bfs_dist.values()) if bfs_dist else 1
    pos = np.zeros((n, 3))
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        r = float(d)
        theta = rng.uniform(0, math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        pos[idx] = [r * math.sin(theta) * math.cos(phi),
                     r * math.sin(theta) * math.sin(phi),
                     r * math.cos(theta)]

    # Edge list with rest lengths
    edge_idx = []
    rest_lens = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))
            rest_lens.append(math.sqrt(abs(e.getSquaredLength())))

    step = 0.5
    eps = 1e-6
    for _ in range(iters):
        forces = np.zeros_like(pos)

        # Spring forces along edges
        for (si, ti), rl in zip(edge_idx, rest_lens):
            delta = pos[ti] - pos[si]
            d = max(np.linalg.norm(delta), eps)
            f = spring_k * (d - rl) * delta / d
            forces[si] += f
            forces[ti] -= f

        # Repulsion (all pairs, capped for large slices)
        cap = min(n, 200)
        for a in range(cap):
            for b in range(a + 1, cap):
                delta = pos[a] - pos[b]
                d2 = np.dot(delta, delta) + eps
                d = math.sqrt(d2)
                f = repulsion_k / d2 * delta / d
                forces[a] += f
                forces[b] -= f

        forces[center_idx] = 0.0
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        forces = np.where(norms > step, forces / norms * step, forces)
        pos += forces
        step *= cooling

    return pos, vid_to_idx, edge_idx


# =========================================================================
# PCA alignment
# =========================================================================

def _pca_align(pos, prev_axes=None):
    """Align *pos* to PCA axes.  Resolve sign/permutation ambiguity against
    *prev_axes* if provided.  Returns (aligned_pos, axes_3x3)."""
    centroid = pos.mean(axis=0)
    centered = pos - centroid
    if centered.shape[0] < 2:
        return centered, np.eye(3)

    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt[:3]

    if prev_axes is not None:
        # Find best permutation of axes that aligns with prev_axes
        best_perm = None
        best_score = -1.0
        for perm in permutations(range(min(3, len(axes)))):
            score = sum(abs(np.dot(axes[perm[i]], prev_axes[i]))
                        for i in range(3))
            if score > best_score:
                best_score = score
                best_perm = perm
        axes = axes[list(best_perm)]
        # Resolve sign flips
        for i in range(3):
            if np.dot(axes[i], prev_axes[i]) < 0:
                axes[i] = -axes[i]

    aligned = centered @ axes.T
    return aligned, axes


# =========================================================================
# Per-vertex curvature
# =========================================================================

def _vertex_curvatures(verts, solver, t):
    """Compute area-weighted average deficit angle for each vertex at time *t*.
    Returns array of length len(verts)."""
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
# Rendering
# =========================================================================

def _render_frame(pos, edge_idx, curvatures, center_idx, t,
                  vmin, vmax, fig_size, cmap_name, elev, azim):
    """Render a single matplotlib 3D frame and return an RGBA numpy array."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Draw edges
    if edge_idx:
        segs = [[pos[s], pos[t]] for s, t in edge_idx]
        lc = Line3DCollection(segs, linewidths=0.4, colors=(0.6, 0.6, 0.6, 0.5))
        ax.add_collection(lc)

    # Draw vertices
    sizes = np.full(len(pos), 20.0)
    sizes[center_idx] = 80.0
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=curvatures, cmap=cmap, norm=norm,
               s=sizes, edgecolors="k", linewidths=0.2, depthshade=True)

    # Highlight center
    ax.scatter([pos[center_idx, 0]], [pos[center_idx, 1]], [pos[center_idx, 2]],
               c="black", s=100, marker="*", zorder=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label="Curvature (A * deficit)")

    ax.set_title(f"t = {t}")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Consistent axis limits across frames would require a global pass;
    # per-frame auto-scaling is acceptable for now.
    fig.subplots_adjust(right=0.85)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    plt.close(fig)
    return img


# =========================================================================
# Public entry point
# =========================================================================

def render_curvature_gif(st, solver, worldline, output_path="curvature.gif",
                         *, fig_size=(6, 6), cmap_name="RdYlBu_r",
                         layout_iters=300, frame_duration_ms=500,
                         elev=25.0, azim=45.0):
    """Render a GIF with one frame per time slice showing the spatial
    subgraph as a force-directed 3D embedding with curvature heat map.

    Args:
        st: caset.Spacetime (after solving)
        solver: caset.ReggeSolver
        worldline: list of Vertex from buildWorldline
        output_path: output .gif path
        fig_size: matplotlib figure size in inches
        cmap_name: matplotlib colormap name
        layout_iters: force-directed iteration count
        frame_duration_ms: GIF inter-frame delay in ms
        elev: 3D view elevation angle
        azim: 3D view azimuth angle

    Returns:
        output_path
    """
    wl_by_time = {}
    for v in worldline:
        wl_by_time[round(v.getTime())] = v

    times = _time_slices(st)
    rng = np.random.default_rng(42)

    # --- Pass 1: compute layouts and curvatures ---
    slice_data = []  # (t, pos, edge_idx, curvatures, center_idx)
    prev_axes = None
    for t in times:
        if t not in wl_by_time:
            continue
        center = wl_by_time[t]
        verts, edges = _slice_subgraph(st, t)
        if len(verts) < 3:
            continue

        bfs_dist = _bfs_distances(center, verts, edges)
        # Keep only the connected component containing center
        reachable_ids = set(bfs_dist.keys())
        verts = [v for v in verts if v.getId() in reachable_ids]
        edges = [e for e in edges
                 if e.getSource().getId() in reachable_ids
                 and e.getTarget().getId() in reachable_ids]
        if len(verts) < 3:
            continue

        pos, vid_to_idx, edge_idx = _force_layout_3d(
            verts, edges, center.getId(), bfs_dist,
            iters=layout_iters, rng=rng)
        pos, prev_axes = _pca_align(pos, prev_axes)

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
    frames = []
    for t, pos, edge_idx, curvatures, center_idx in slice_data:
        img = _render_frame(pos, edge_idx, curvatures, center_idx, t,
                            vmin, vmax, fig_size, cmap_name, elev, azim)
        frames.append(img)

    # Assemble GIF
    pil_frames = [Image.fromarray(f[:, :, :3]) for f in frames]
    pil_frames[0].save(output_path, save_all=True,
                       append_images=pil_frames[1:],
                       duration=frame_duration_ms, loop=0)
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

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)

    target = st.getN41()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
    cdt.tune()
    cdt.sweep(10)

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
