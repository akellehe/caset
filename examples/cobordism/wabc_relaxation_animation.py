# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Animate the W_ABC singlet relaxation (#361).

Renders the relaxation of the trivalent proton junction over its iterations, with two
views (``--view``):

  * ``full``  -- primal force-layout (edges colored by l^2, width ~ curvature, null
    edges magenta), the dual graph, and the live residual + photon-count traces.
  * ``pants`` -- a CLEARER view of the pair of pants: the base surface on a clean
    spectral sphere embedding, with the four windows (A, B, C inputs -> R result)
    color-coded and labeled, the bulk faded, and the cross-layer worldlines drawn as
    radial stubs (magenta = null = photon); plus the residual + photon-count trace.

Capture is stepped + emergent-first: the geometry at iteration k is read off a fresh
deterministic relax to k steps (max_iters=k). The junction is Lorentzian (cross-layer
worldlines start timelike), so a worldline whose l^2 relaxes through 0 is the emergent
photon. The output GIF is an issue/PR artifact (not committed).

Run:  python examples/cobordism/wabc_relaxation_animation.py [--view pants] [--out x.gif]
"""

import argparse

import numpy as np

import tessera
from tessera.utils import plot

cob = tessera.cobordism
_NEUTRAL = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
_NULL_BAND = 0.05          # |l^2| < band  -> treated as null (photon) for the viz
_WORLDLINE_LSQ = -0.3      # cross-layer worldlines start timelike (so null can emerge)
_WCOLOR = [(0.85, 0.12, 0.12), (0.12, 0.62, 0.20),
           (0.13, 0.35, 0.85), (0.85, 0.60, 0.0)]   # A red, B green, C blue, R gold
_WLABEL = ["A", "B", "C", "R"]


def _build(k):
    """The Lorentzian W_ABC junction relaxed to exactly k iterations (deterministic)."""
    t = cob.TripartiteRegisterTopology()
    t.set_lorentzian_worldlines(_WORLDLINE_LSQ)
    return cob.TransportCobordism(_NEUTRAL, max_iters=k, seed=0, topology=t)


def _windows(m):
    ih = [tuple(sorted(h)) for h in m.input_holes]
    return [ih[0:3], ih[3:6], ih[6:9], [tuple(sorted(h)) for h in m.result_holes]]


def _edges(m):
    return [(e.getSource().getId(), e.getTarget().getId(), e.getSquaredLength().real)
            for e in m.cobordism.getEdgeList().toVector()]


def _deficits(m):
    m.cobordism.materializeFacets()
    d = {}
    for s in m.cobordism.getSimplices():
        vs = [v.getId() for v in s.getVertices()]
        if len(vs) == 2:
            d[(min(vs), max(vs))] = abs(s.lorentzianDeficitAngle().real)
    return d


def _dual(m, pos, vid_to_idx):
    tets = []
    for s in m.cobordism.getSimplices():
        vs = [v.getId() for v in s.getVertices()]
        if len(vs) == 4 and all(v in vid_to_idx for v in vs):
            tets.append(tuple(sorted(vs)))
    tets = list(dict.fromkeys(tets))
    centroids = np.array([np.mean([pos[vid_to_idx[v]] for v in t], axis=0)
                          for t in tets]) if tets else np.zeros((0, 3))
    face_to_tets = {}
    for ti, t in enumerate(tets):
        for drop in range(4):
            face_to_tets.setdefault(t[:drop] + t[drop + 1:], []).append(ti)
    dedges = [tuple(ts) for ts in face_to_tets.values() if len(ts) == 2]
    return centroids, dedges


def _box(ax, p):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    if len(p):
        c = p.mean(0); r = np.abs(p - c).max() + 1e-9
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def _residual_panel(ax, trace, k, kmax, resid, nnull):
    ks = [t[0] for t in trace]
    ax.plot(ks, [t[1] for t in trace], "-o", ms=3, color="navy",
            label="stationarity residual")
    ax.set_xlabel("relaxation iteration"); ax.set_ylabel("stat. residual", color="navy")
    ax.set_xlim(0, kmax); ax.set_yscale("log"); ax.axvline(k, color="0.7", lw=1, ls="--")
    ax2 = ax.twinx()
    ax2.plot(ks, [t[2] for t in trace], "-s", ms=3, color="magenta")
    ax2.set_ylabel("null (photon) edges", color="magenta")
    ax.set_title(f"residual {resid:.2e} · null edges {nnull}", fontsize=9)


# =========================================================================
# View: full (force layout primal + dual + residuals)
# =========================================================================
def _frame_full(geom, pos, vid_to_idx, dual, trace, kmax, l2max, defmax):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    edges, defmap, k, resid, nnull = geom
    fig = plt.figure(figsize=(15, 5))
    axA = fig.add_subplot(1, 3, 1, projection="3d")
    segs, colors, widths = [], [], []
    for (a, b, l2) in edges:
        ia, ib = vid_to_idx.get(a), vid_to_idx.get(b)
        if ia is None or ib is None:
            continue
        segs.append([pos[ia], pos[ib]])
        defv = defmap.get((min(a, b), max(a, b)), 0.0)
        if abs(l2) < _NULL_BAND:
            colors.append((1.0, 0.0, 1.0, 0.95)); widths.append(2.6)
        else:
            x = max(-1.0, min(1.0, l2 / (l2max + 1e-9)))
            colors.append((max(0.0, -x) * 0.9 + 0.1, 0.1, max(0.0, x) * 0.9 + 0.1, 0.6))
            widths.append(0.5 + 2.0 * defv / (defmax + 1e-9))
    axA.add_collection(Line3DCollection(segs, colors=colors, linewidths=widths))
    axA.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=3, c="0.3", depthshade=True)
    _box(axA, pos)
    axA.set_title(f"primal  (iter {k})\nblue spacelike · red timelike · "
                  f"magenta null(photon) · width~curvature", fontsize=9)
    axB = fig.add_subplot(1, 3, 2, projection="3d")
    dpos, dedges = dual
    if len(dpos):
        axB.add_collection(Line3DCollection([[dpos[a], dpos[b]] for a, b in dedges],
                                            colors=(0.0, 0.5, 0.2, 0.35), linewidths=0.6))
        axB.scatter(dpos[:, 0], dpos[:, 1], dpos[:, 2], s=4, c="green", depthshade=True)
        _box(axB, dpos)
    axB.set_title("dual  (top-simplex centroids + face adjacency)", fontsize=9)
    _residual_panel(fig.add_subplot(1, 3, 3), trace, k, kmax, resid, nnull)
    return _rasterize(fig)


# =========================================================================
# View: pants (clean spectral sphere of the base surface, windows highlighted)
# =========================================================================
def _sphere_coords(edges, stride):
    """Spectral (graph-Laplacian) embedding of the base-layer (id < stride) surface
    onto a clean sphere -- a far clearer view of the 4-window junction than the
    force-layout of the full prism."""
    base = sorted({v for (a, b, _l) in edges for v in (a, b) if v < stride})
    sidx = {v: i for i, v in enumerate(base)}
    n = len(base)
    adj = np.zeros((n, n))
    for (a, b, _l) in edges:
        if a < stride and b < stride:
            adj[sidx[a], sidx[b]] = adj[sidx[b], sidx[a]] = 1.0
    lap = np.diag(adj.sum(1)) - adj
    _w, vecs = np.linalg.eigh(lap)
    coords = vecs[:, 1:4].copy()                       # 3 lowest non-trivial modes
    coords /= (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-9)  # onto unit sphere
    return coords, sidx


def _frame_pants(geom, scoords, sidx, windows, trace, kmax, l2scale):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    edges, _defmap, k, resid, nnull = geom
    wv = [set(v for h in w for v in h) for w in windows]
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    segs, cols, wid = [], [], []
    stubs, scol, swid = [], [], []
    for (a, b, l2) in edges:
        if a in sidx and b in sidx:                    # base-surface edge
            segs.append([scoords[sidx[a]], scoords[sidx[b]]])
            wc = next((_WCOLOR[i] for i, s in enumerate(wv) if a in s and b in s), None)
            if wc:
                cols.append(wc + (0.95,)); wid.append(2.6)   # a window cuff
            else:
                cols.append((0.55, 0.55, 0.62, 0.16)); wid.append(0.5)  # faded bulk
        else:                                          # cross-layer worldline (radial stub)
            anchor = a if a in sidx else (b if b in sidx else None)
            if anchor is None:
                continue
            p = scoords[sidx[anchor]]
            ln = 0.05 + 0.40 * (abs(l2) ** 0.5) / (l2scale + 1e-9)
            stubs.append([p, p * (1.0 + ln)])
            if abs(l2) < _NULL_BAND:
                scol.append((1.0, 0.0, 1.0, 0.95)); swid.append(2.8)   # photon
            else:
                scol.append((0.15, 0.15, 0.70, 0.45)); swid.append(0.9)
    ax.add_collection(Line3DCollection(segs, colors=cols, linewidths=wid))
    ax.add_collection(Line3DCollection(stubs, colors=scol, linewidths=swid))
    for i, s in enumerate(wv):
        pts = [scoords[sidx[v]] for v in s if v in sidx]
        if pts:
            c = np.mean(pts, 0); c = c / (np.linalg.norm(c) + 1e-9) * 1.30
            ax.text(c[0], c[1], c[2], _WLABEL[i], color=_WCOLOR[i],
                    fontsize=15, fontweight="bold", ha="center")
    _box(ax, np.vstack([scoords, scoords * 1.45]))
    ax.set_title(f"pair of pants: windows A,B,C → R   (iter {k})\n"
                 f"colored cuffs · faded bulk · radial worldlines (magenta = null photon)",
                 fontsize=9)
    _residual_panel(fig.add_subplot(1, 2, 2), trace, k, kmax, resid, nnull)
    return _rasterize(fig)


def _rasterize(fig):
    import matplotlib.pyplot as plt
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.10, top=0.86, wspace=0.30)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# =========================================================================
# Capture + render
# =========================================================================
def capture(ks):
    """Stepped capture: for each k, the relaxed geometry + residual + null count.
    Returns (geoms, trace, layout, windows, stride)."""
    geoms, trace = [], []
    layout = windows = None
    stride = None
    for k in ks:
        m = _build(k)
        edges = _edges(m)
        defmap = _deficits(m)
        nnull = sum(1 for (_a, _b, l2) in edges if abs(l2) < _NULL_BAND)
        resid = float(m.stats.stat_action_residual)
        if layout is None:
            windows = _windows(m)
            stride = m.cobordism.getVertexCount() // 3   # base/mid/top prism layers
            mf = _build(ks[-1])
            pos, vid_to_idx, _ = plot.layout_from_spacetime(
                mf.cobordism.getVertexList().toVector(),
                mf.cobordism.getEdgeList().toVector(), seed=0)
            layout = (pos, vid_to_idx, _dual(mf, pos, vid_to_idx))
        trace.append((k, resid, nnull))
        geoms.append((edges, defmap, k, resid, nnull))
    return geoms, trace, layout, windows, stride


def render(geoms, trace, layout, windows, stride, out, view="full"):
    l2max = max(abs(l2) for g in geoms for (_a, _b, l2) in g[0]) or 1.0
    defmax = max((max(g[1].values()) if g[1] else 0.0) for g in geoms) or 1.0
    kmax = max(t[0] for t in trace) or 1
    if view == "pants":
        scoords, sidx = _sphere_coords(geoms[-1][0], stride)
        l2scale = l2max ** 0.5
        frames = [_frame_pants(g, scoords, sidx, windows, trace[:i + 1], kmax, l2scale)
                  for i, g in enumerate(geoms)]
    else:
        pos, vid_to_idx, dual = layout
        frames = [_frame_full(g, pos, vid_to_idx, dual, trace[:i + 1], kmax, l2max, defmax)
                  for i, g in enumerate(geoms)]
    plot.save_gif(frames, out, duration_ms=350)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", choices=["full", "pants"], default="full",
                    help="full = force-layout primal+dual+residuals; "
                         "pants = clean sphere of the base surface with the 4 windows")
    ap.add_argument("--out", default="wabc_relaxation.gif")
    ap.add_argument("--iters", type=int, nargs="*",
                    default=[0, 2, 5, 10, 15, 20, 30, 40, 50, 60])
    args = ap.parse_args()
    print(f"capturing {len(args.iters)} frames (stepped relax): {args.iters}")
    geoms, trace, layout, windows, stride = capture(args.iters)
    render(geoms, trace, layout, windows, stride, args.out, view=args.view)
    print(f"wrote {args.out}  ({len(geoms)} frames, view={args.view})")
    print("null-edge (photon) count by iteration:", {t[0]: t[2] for t in trace})


if __name__ == "__main__":
    main()
