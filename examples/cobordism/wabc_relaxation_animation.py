# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Animate the W_ABC singlet relaxation (#361).

Renders the relaxation of the trivalent proton junction over its iterations:
  * PRIMAL geometry  -- edges colored by squared length l^2 (spacelike > 0 blue,
    timelike < 0 red), width ~ |Re deficit| (the curvature gradient); null/photon
    edges (l^2 ~ 0) highlighted magenta+fat;
  * DUAL geometry     -- top-simplex centroids joined by face-adjacency;
  * LIVE RESIDUALS    -- the stationarity residual and the null-edge (photon) count
    traced over the relaxation.

The junction is Lorentzian (cross-layer worldlines start timelike), so a worldline
whose l^2 relaxes through 0 is a null edge -- the emergent photon. The complex is
invariant under relaxation (only l^2 changes), so the 3D layout is computed once and
held fixed; the relaxation signal is the per-edge color/width and the residual trace.

Capture is stepped + emergent-first: the geometry at iteration k is read off a fresh
deterministic relax to k steps (max_iters=k), nothing fabricated. The output GIF is an
issue/PR artifact (not committed).

Run:  python examples/cobordism/wabc_relaxation_animation.py [--out wabc.gif]
"""

import argparse

import numpy as np

import tessera
from tessera.utils import plot

cob = tessera.cobordism
_NEUTRAL = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
_NULL_BAND = 0.05          # |l^2| < band  -> treated as null (photon) for the viz
_WORLDLINE_LSQ = -0.3      # cross-layer worldlines start timelike (so null can emerge)


def _build(k):
    """The Lorentzian W_ABC junction relaxed to exactly k iterations (deterministic)."""
    t = cob.TripartiteRegisterTopology()
    # The emergent photon is a worldline relaxing through l^2 ~ 0. On the symmetric apex
    # interior (#413, now the default) the worldlines stay timelike and no null edge
    # emerges; the prism (x I) triangulation is the one whose worldlines relax through
    # null, so this animation -- whose subject IS the emergent photon -- exercises the
    # prism path explicitly.
    t.set_symmetric_interior(False)
    t.set_lorentzian_worldlines(_WORLDLINE_LSQ)
    return cob.TransportCobordism(_NEUTRAL, max_iters=k, seed=0, topology=t)


def _edges(m):
    """List of (src_id, tgt_id, l2) for every edge of the relaxed complex."""
    out = []
    for e in m.cobordism.getEdgeList().toVector():
        out.append((e.getSource().getId(), e.getTarget().getId(),
                    e.getSquaredLength().real))
    return out


def _deficits(m):
    """edge (min,max) -> |Re(lorentzian deficit angle)| (the curvature per hinge)."""
    m.cobordism.materializeFacets()
    d = {}
    for s in m.cobordism.getSimplices():
        vs = [v.getId() for v in s.getVertices()]
        if len(vs) == 2:
            d[(min(vs), max(vs))] = abs(s.lorentzianDeficitAngle().real)
    return d


def _dual(m, pos, vid_to_idx):
    """Dual nodes (top-simplex centroids) + dual edges (tets sharing a triangle)."""
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
            face = t[:drop] + t[drop + 1:]
            face_to_tets.setdefault(face, []).append(ti)
    dedges = [(a, b) for ts in face_to_tets.values() if len(ts) == 2
              for a, b in [ts]]
    return centroids, dedges


def _frame(geom, pos, vid_to_idx, dual, trace, kmax, l2max, defmax):
    """Render one multi-panel RGBA frame from captured geometry `geom`."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    edges, defmap, k, resid, nnull = geom
    fig = plt.figure(figsize=(15, 5))

    # --- Panel A: primal geometry ---
    axA = fig.add_subplot(1, 3, 1, projection="3d")
    segs, colors, widths = [], [], []
    for (a, b, l2) in edges:
        ia, ib = vid_to_idx.get(a), vid_to_idx.get(b)
        if ia is None or ib is None:
            continue
        segs.append([pos[ia], pos[ib]])
        defv = defmap.get((min(a, b), max(a, b)), 0.0)
        if abs(l2) < _NULL_BAND:                      # null = photon
            colors.append((1.0, 0.0, 1.0, 0.95)); widths.append(2.6)
        else:
            x = max(-1.0, min(1.0, l2 / (l2max + 1e-9)))
            # red (timelike) <- 0 -> blue (spacelike)
            colors.append((max(0.0, -x) * 0.9 + 0.1, 0.1,
                           max(0.0, x) * 0.9 + 0.1, 0.6))
            widths.append(0.5 + 2.0 * defv / (defmax + 1e-9))
    axA.add_collection(Line3DCollection(segs, colors=colors, linewidths=widths))
    axA.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=3, c="0.3", depthshade=True)
    _box(axA, pos)
    axA.set_title(f"primal  (iter {k})\nblue spacelike · red timelike · "
                  f"magenta null(photon)·width~curvature", fontsize=9)

    # --- Panel B: dual geometry ---
    axB = fig.add_subplot(1, 3, 2, projection="3d")
    dpos, dedges = dual
    if len(dpos):
        axB.add_collection(Line3DCollection(
            [[dpos[a], dpos[b]] for a, b in dedges],
            colors=(0.0, 0.5, 0.2, 0.35), linewidths=0.6))
        axB.scatter(dpos[:, 0], dpos[:, 1], dpos[:, 2], s=4, c="green",
                    depthshade=True)
        _box(axB, dpos)
    axB.set_title("dual  (top-simplex centroids + face adjacency)", fontsize=9)

    # --- Panel C: live residual + photon trace ---
    axC = fig.add_subplot(1, 3, 3)
    ks = [t[0] for t in trace]
    axC.plot(ks, [t[1] for t in trace], "-o", ms=3, color="navy",
             label="stationarity residual")
    axC.set_xlabel("relaxation iteration"); axC.set_ylabel("stat. residual", color="navy")
    axC.set_xlim(0, kmax); axC.set_yscale("log")
    axC.axvline(k, color="0.7", lw=1, ls="--")
    axc2 = axC.twinx()
    axc2.plot(ks, [t[2] for t in trace], "-s", ms=3, color="magenta",
              label="null (photon) edges")
    axc2.set_ylabel("null-edge count", color="magenta")
    axC.set_title(f"residual {resid:.2e} · null edges {nnull}", fontsize=9)

    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.10, top=0.86, wspace=0.30)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def _box(ax, p):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    if len(p):
        c = p.mean(0); r = np.abs(p - c).max() + 1e-9
        for setlim, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(c[ci] - r, c[ci] + r)


def capture(ks):
    """Stepped capture: for each k, the relaxed geometry + residual + null count."""
    geoms, trace = [], []
    layout = None
    for k in ks:
        m = _build(k)
        edges = _edges(m)
        defmap = _deficits(m)
        nnull = sum(1 for (_a, _b, l2) in edges if abs(l2) < _NULL_BAND)
        resid = float(m.stats.stat_action_residual)
        if layout is None:                      # one fixed layout (final-ish geometry)
            mf = _build(ks[-1])
            pos, vid_to_idx, _ = plot.layout_from_spacetime(
                mf.cobordism.getVertexList().toVector(),
                mf.cobordism.getEdgeList().toVector(), seed=0)
            dual = _dual(mf, pos, vid_to_idx)
            layout = (pos, vid_to_idx, dual)
        trace.append((k, resid, nnull))
        geoms.append((edges, defmap, k, resid, nnull))
    return geoms, trace, layout


def render(geoms, trace, layout, out):
    pos, vid_to_idx, dual = layout
    l2max = max(abs(l2) for g in geoms for (_a, _b, l2) in g[0]) or 1.0
    defmax = max((max(g[1].values()) if g[1] else 0.0) for g in geoms) or 1.0
    kmax = max(t[0] for t in trace) or 1
    frames = [_frame(g, pos, vid_to_idx, dual, trace[:i + 1], kmax, l2max, defmax)
              for i, g in enumerate(geoms)]
    plot.save_gif(frames, out, duration_ms=350)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="wabc_relaxation.gif")
    ap.add_argument("--iters", type=int, nargs="*",
                    default=[0, 2, 5, 10, 15, 20, 30, 40, 50, 60])
    args = ap.parse_args()
    print(f"capturing {len(args.iters)} frames (stepped relax): {args.iters}")
    geoms, trace, layout = capture(args.iters)
    render(geoms, trace, layout, args.out)
    print(f"wrote {args.out}  ({len(geoms)} frames)")
    print("null-edge (photon) count by iteration:",
          {t[0]: t[2] for t in trace})


if __name__ == "__main__":
    main()
