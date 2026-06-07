# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Visualize the §5.0 realizability pipeline (#166).

Make the realizability story legible: *see* systems A and B as boundary
manifolds and *see* them glued into a bulk cobordism ``W_AB`` with a filled
interior.

Given a realizable operation ``U : H_B -> H_A`` the pipeline produces three
concrete simplicial objects, all rendered here:

  * ``geo(psi_A)`` -- the geometric image of A's principal boundary state, the
    dominant left Schmidt vector of ``vec(U)`` (== left singular vector of U),
    synthesized by ``cobordism.GeometrySynthesizer`` (#134): the simplest complex
    whose k=0 Hodge-Laplacian has ``psi_A`` as an eigenvector.
  * ``geo(psi_B)`` -- the same for B's principal boundary state (the right
    Schmidt vector).
  * ``W_AB`` -- the bulk cobordism synthesized by ``cobordism.RealizabilityOracle``
    (#138): the output surface (a pinned boundary) carries ``vec(U)`` and the
    interior is filled so the output-boundary Laplacian eigenvector matches the
    bent target, driving the residual ``r`` to zero. ``W_AB`` is the verdict's
    ``witness``.

The output is ONE composite static PNG with four panels (the locked render
style, issue #166):

  1. System A      -- geo(psi_A), vertices colored by |psi| (phase -> hue).
  2. System B      -- geo(psi_B), same encoding.
  3. Before gluing -- geo(psi_A), geo(psi_B) and the bare output surface as
                      disjoint, per-component color-coded pieces.
  4. After gluing  -- the assembled W_AB: boundary cells solid in the output-
                      surface color with the eigenvector overlaid, interior
                      cells lighter/dashed, the residual ``r`` annotated.

Panels 3 and 4 share a single ``pca_align`` layout for the output surface, so the
eye tracks how the same boundary connects through the filled bulk.

Images are written to ``/tmp/cobordism`` and are NOT committed to the repo tree
(the script is the committed artifact; attach the PNG to the issue/PR). The
layout is seeded for reproducibility and the compute is capped at the box's
shared-workload budget.

Run:  python examples/cobordism/visualize_realizability.py
      (use --help for options; the composite PNG defaults to
       /tmp/cobordism/visualize_realizability.png)
"""

from __future__ import annotations

import argparse
import os

# Cap BLAS / OpenMP threads at launch (shared box; #feedback_cpu_cap). Set before
# numpy / tessera import so the native pools pick them up.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tessera
from tessera.utils.plot import (layout_from_spacetime, render_frame, draw_edges,
                                pca_align)

cob = tessera.cobordism

# A realizable, generic complex operation U : H_B -> H_A. Generic (not the
# constant zero-mode) so vec(U) is a non-trivial boundary state and the filled
# interior genuinely participates in the eigenvector.
U_DEFAULT = [[0.8 + 0.0j, 0.2 + 0.3j],
             [0.3 - 0.2j, 0.9 + 0.0j]]

# Per-component colors for the "before gluing" panel (disjoint pieces).
COLOR_A = (0.20, 0.42, 0.86)        # system A   -- blue
COLOR_B = (0.15, 0.64, 0.36)        # system B   -- green
COLOR_SURF = (0.92, 0.55, 0.10)     # output sfc -- orange
COLOR_INT = (0.62, 0.62, 0.66)      # interior   -- grey (lighter/dashed)


# =========================================================================
# Simplicial fixtures (the #138 / #134 bulk-synthesis idiom)
# =========================================================================

def _spacetime(dim, topology, spacetime_type=tessera.CDT):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, spacetime_type, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _wheel_bulk(weight=1.0, phase=0.0):
    """The bulk W_AB seed: a 2D wheel -- rim 0-1-2-3 (an S^1, the pinned output
    surface, the four smallest ids = the dA*dB output-boundary support) coned
    over a single interior hub vertex 4 by the triangles 014,124,234,304.

    getBoundary() is the rim 4-cycle; the spokes 0-4,1-4,2-4,3-4 and the hub are
    the interior the oracle fills. All edges are pinned Hermitian (weight, phase)
    -- the oracle rewrites only the interior spokes.
    """
    st = _spacetime(2, tessera.Toroid())
    v = {i: st.createVertex(i) for i in range(5)}
    for tri in [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]:
        st.createSimplex([v[i] for i in tri])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(weight)
        e.setPhase(phase)
    return st


def _from_simplices(num_vertices, simplices):
    """A Hermitian-weighted seed complex (the #134 idiom: two smallest ids are
    the logical qubit pair, the rest are zero-amplitude auxiliaries)."""
    st = _spacetime(4, tessera.Toroid(), tessera.HERMITIAN_WEIGHTED)
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


# =========================================================================
# Choi / Schmidt boundary states of U
# =========================================================================

def _schmidt_qubits(U):
    """Return (psi_A, psi_B): the dominant Schmidt vectors of vec(U), i.e. the
    principal left / right singular vectors of U (the operation's principal
    boundary states on H_A and H_B). Global phase gauge-fixed so c0 is real >= 0.
    """
    Umat = np.asarray(U, dtype=complex)
    ua, _s, vh = np.linalg.svd(Umat)
    psi_a = ua[:, 0]
    psi_b = vh.conj()[:, 0]

    def _gauge(z):
        if abs(z[0]) > 1e-12:
            z = z * np.exp(-1j * np.angle(z[0]))
        return z

    return _gauge(psi_a), _gauge(psi_b)


# =========================================================================
# Graph extraction + scalar invariants
# =========================================================================

def _graph(st):
    """Sorted vertices, edge index-pairs, and boundary masks for *st*.

    Returns a dict with: ``ids`` (sorted vertex ids), ``vid_to_idx``,
    ``verts`` (Vertex objects in id order), ``edges`` (Edge objects),
    ``edge_idx`` (list of (i,j) index pairs), ``edge_timelike`` (bool list),
    ``bnd_edge`` (bool list, True if the edge is on getBoundary()),
    ``bnd_vid`` (set of boundary vertex ids).
    """
    verts = sorted(st.getVertexList().toVector(), key=lambda v: v.getId())
    ids = [v.getId() for v in verts]
    vid_to_idx = {vid: i for i, vid in enumerate(ids)}

    bnd_faces = st.getBoundary()
    bnd_edge_keys = set()
    bnd_vid = set()
    for face in bnd_faces:
        for a in face:
            bnd_vid.add(a)
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                bnd_edge_keys.add((min(face[i], face[j]), max(face[i], face[j])))

    edges = list(st.getEdgeList().toVector())
    edge_idx, edge_timelike, bnd_edge = [], [], []
    for e in edges:
        s, t = e.getSource().getId(), e.getTarget().getId()
        if s == t or s not in vid_to_idx or t not in vid_to_idx:
            continue
        edge_idx.append((vid_to_idx[s], vid_to_idx[t]))
        edge_timelike.append(e.getSquaredLength() < 0.0)
        bnd_edge.append((min(s, t), max(s, t)) in bnd_edge_keys)
    return dict(ids=ids, vid_to_idx=vid_to_idx, verts=verts, edges=edges,
                edge_idx=edge_idx, edge_timelike=edge_timelike,
                bnd_edge=bnd_edge, bnd_vid=bnd_vid)


def _dim(st):
    """Intrinsic simplicial dimension: largest simplex (any rank) minus one."""
    best = 0
    for v in st.getVertexList().toVector():
        for s in v.getSimplices():
            best = max(best, len(s.getVertices()) - 1)
    return best


def _betti1(st):
    b = cob.ChainComplex.fromSpacetime(st).bettiNumbers()
    return int(b[1]) if len(b) > 1 else 0


def _caption(name, st, extra=""):
    return (f"{name}\n"
            f"dim {_dim(st)}  |V| {st.getVertexCount()}  "
            f"|E| {st.getEdgeList().size()}  b1 {_betti1(st)}{extra}")


# =========================================================================
# Color / layout helpers
# =========================================================================

def _amp_rgba(z, max_mag, *, alpha=0.97, floor=0.42):
    """Map a complex amplitude to RGBA: phase -> hue, |amp| -> brightness.

    Zero-amplitude (auxiliary / interior-decoupled) vertices desaturate to a
    dim grey so the harmonic support reads clearly.
    """
    mag = abs(z)
    frac = (mag / max_mag) if max_mag > 0 else 0.0
    if mag <= 1e-9:
        return (0.55, 0.55, 0.58, alpha)
    hue = (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)
    val = floor + (1.0 - floor) * frac
    return (*mcolors.hsv_to_rgb([hue, 0.85, val]), alpha)


def _amp_size(z, max_mag, *, lo=45.0, hi=320.0):
    frac = (abs(z) / max_mag) if max_mag > 0 else 0.0
    return lo + (hi - lo) * frac


def _normalize(pos):
    """Center to the origin and scale the longest half-extent to 1."""
    pos = np.asarray(pos, dtype=float)
    if len(pos) == 0:
        return pos
    centered = pos - pos.mean(axis=0)
    scale = np.abs(centered).max()
    return centered / scale if scale > 1e-12 else centered


def _autocrop(img, *, bg=248, pad=10):
    """Trim near-white borders so the geometry fills its montage cell."""
    rgb = img[:, :, :3]
    mask = (rgb < bg).any(axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y0:y1, x0:x1]


def _square(img):
    """Center an image on a square white canvas (uniform montage cells)."""
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.full((s, s, img.shape[2]), 255, dtype=img.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


def _finish(img):
    return _square(_autocrop(img))


def _draw_vertices(ax, pos, colors, sizes, *, edgecolor="k", lw=0.4, zorder=5):
    pos = np.asarray(pos)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=sizes,
               edgecolors=edgecolor, linewidths=lw, depthshade=False,
               zorder=zorder)


# =========================================================================
# Panel builders -- each returns an RGBA frame via render_frame
# =========================================================================

def _amp_vector(st, amp_by_id):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    return np.array([amp_by_id.get(vid, 0.0 + 0.0j) for vid in ids], dtype=complex)


def _panel_geo(st, amp_by_id, *, azim=35, layout_seed=42):
    """A single synthesized boundary manifold geo(psi), vertices by |psi|."""
    g = _graph(st)
    pos, _vmap, edge_idx = layout_from_spacetime(g["verts"], g["edges"],
                                                 seed=layout_seed, iters=400)
    pos, _ = pca_align(pos)
    pos = _normalize(pos)

    amp = _amp_vector(st, amp_by_id)
    max_mag = float(np.abs(amp).max()) or 1.0
    colors = [_amp_rgba(z, max_mag) for z in amp]
    sizes = [_amp_size(z, max_mag) for z in amp]

    def draw(ax):
        draw_edges(ax, pos, edge_idx, edge_types=g["edge_timelike"],
                   linewidth=1.1)
        _draw_vertices(ax, pos, colors, sizes)
        _set_equal_3d(ax, pos)

    return _finish(render_frame(draw, figsize=(6.4, 6.4), azim=azim))


def _panel_before(stA, stB, gW, posW_n, *, azim=35, layout_seed=42):
    """geo(psi_A) | output surface | geo(psi_B) -- three disjoint components,
    color-coded. The output-surface piece reuses the shared W_AB layout."""
    gA, gB = _graph(stA), _graph(stB)
    posA = _normalize(pca_align(layout_from_spacetime(
        gA["verts"], gA["edges"], seed=layout_seed, iters=400)[0])[0])
    posB = _normalize(pca_align(layout_from_spacetime(
        gB["verts"], gB["edges"], seed=layout_seed, iters=400)[0])[0])

    # Output surface = the rim (boundary vertices of W_AB), in the SHARED layout.
    rim_idx = [i for i, vid in enumerate(gW["ids"]) if vid in gW["bnd_vid"]]
    rim_old_to_new = {old: k for k, old in enumerate(rim_idx)}
    pos_surf = posW_n[rim_idx]
    surf_edges = [(rim_old_to_new[a], rim_old_to_new[b])
                  for (a, b), is_bnd in zip(gW["edge_idx"], gW["bnd_edge"])
                  if is_bnd and a in rim_old_to_new and b in rim_old_to_new]

    gap = 2.7
    posA = posA + np.array([-gap, 0.0, 0.0])
    posB = posB + np.array([+gap, 0.0, 0.0])

    def draw(ax):
        # A (blue), surface (orange), B (green) -- disjoint.
        draw_edges(ax, posA, gA["edge_idx"], default_color=(*COLOR_A, 0.55),
                   linewidth=1.0)
        _draw_vertices(ax, posA, [(*COLOR_A, 0.97)] * len(posA),
                       [90.0] * len(posA))
        draw_edges(ax, pos_surf, surf_edges, default_color=(*COLOR_SURF, 0.7),
                   linewidth=1.6)
        _draw_vertices(ax, pos_surf, [(*COLOR_SURF, 0.97)] * len(pos_surf),
                       [120.0] * len(pos_surf))
        draw_edges(ax, posB, gB["edge_idx"], default_color=(*COLOR_B, 0.55),
                   linewidth=1.0)
        _draw_vertices(ax, posB, [(*COLOR_B, 0.97)] * len(posB),
                       [90.0] * len(posB))
        allpos = np.vstack([posA, pos_surf, posB])
        _set_equal_3d(ax, allpos, cube=False)
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A,
                   markersize=10, label="A: geo(psi_A)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURF,
                   markersize=10, label="output surface"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_B,
                   markersize=10, label="B: geo(psi_B)"),
        ]
        ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))

    return _finish(render_frame(draw, figsize=(6.4, 6.4), azim=azim))


def _panel_after(gW, posW_n, state, *, azim=35):
    """The assembled W_AB: boundary solid in the surface color with the output-
    boundary eigenvector overlaid; interior lighter/dashed."""
    amp = np.asarray(state, dtype=complex)
    max_mag = float(np.abs(amp).max()) or 1.0

    bnd_set = gW["bnd_vid"]
    colors, sizes, edgecolors = [], [], []
    for i, vid in enumerate(gW["ids"]):
        z = amp[i] if i < len(amp) else 0.0 + 0.0j
        colors.append(_amp_rgba(z, max_mag))
        sizes.append(_amp_size(z, max_mag))
        edgecolors.append(COLOR_SURF if vid in bnd_set else COLOR_INT)

    bnd_edges = [e for e, b in zip(gW["edge_idx"], gW["bnd_edge"]) if b]
    int_edges = [e for e, b in zip(gW["edge_idx"], gW["bnd_edge"]) if not b]

    def draw(ax):
        draw_edges(ax, posW_n, int_edges, default_color=(*COLOR_INT, 0.7),
                   linewidth=1.0, linestyle="dashed")
        draw_edges(ax, posW_n, bnd_edges, default_color=(*COLOR_SURF, 0.85),
                   linewidth=2.0, linestyle="solid")
        _draw_vertices(ax, posW_n, colors, sizes, edgecolor=edgecolors, lw=1.3)
        _set_equal_3d(ax, posW_n)
        ax.text2D(0.02, 0.02,
                  "boundary: solid + eigenvector glow\n"
                  "interior: dashed (filled)",
                  transform=ax.transAxes, fontsize=8.5, color=(0.2, 0.2, 0.2))

    return _finish(render_frame(draw, figsize=(6.4, 6.4), azim=azim))


def _set_equal_3d(ax, pos, *, cube=True, pad=1.18):
    """Frame the data tightly and strip the 3D grid/panes (clean abstract look)."""
    pos = np.asarray(pos)
    if len(pos) == 0:
        ax.set_axis_off()
        return
    mn, mx = pos.min(axis=0), pos.max(axis=0)
    c = (mn + mx) / 2.0
    ranges = mx - mn
    if cube:
        r = float(ranges.max()) or 1.0
        half = np.array([r, r, r]) / 2.0 * pad
    else:
        half = np.maximum(ranges, 1e-6) / 2.0 * pad
    ax.set_xlim(c[0] - half[0], c[0] + half[0])
    ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_zlim(c[2] - half[2], c[2] + half[2])
    try:
        ax.set_box_aspect(tuple(np.maximum(half, 1e-6)))
    except Exception:
        pass
    ax.set_axis_off()


# =========================================================================
# Pipeline + composite
# =========================================================================

def run_pipeline(U=None, *, seed_w=0, seed_a=3, seed_b=7, restarts=96,
                 epsilon_w=1e-10, epsilon_geo=1e-9, max_cones=4):
    """Run a realizable U through the §5.0 pipeline.

    Returns a dict with the three spacetimes (``stA``, ``stB``, ``witness``),
    the embedded boundary amplitudes (``ampA``, ``ampB``), the witness state /
    target / residual, and the Schmidt qubits (``psiA``, ``psiB``).
    """
    U = U_DEFAULT if U is None else U
    psiA, psiB = _schmidt_qubits(U)

    # geo(psi_A): edge seed -> grows to its minimal complex.
    gsA = cob.GeometrySynthesizer(_from_simplices(2, [(0, 1)]))
    geoA = gsA.synthesize(complex(psiA[0]), complex(psiA[1]),
                          epsilon=epsilon_geo, restarts=restarts,
                          max_cones=max_cones, seed=seed_a)
    stA = gsA.spacetime()

    # geo(psi_B): Delta^4 seed (K_5) -- already enough freedom.
    gsB = cob.GeometrySynthesizer(_from_simplices(5, [(0, 1, 2, 3, 4)]))
    geoB = gsB.synthesize(complex(psiB[0]), complex(psiB[1]),
                          epsilon=epsilon_geo, restarts=restarts,
                          max_cones=max_cones, seed=seed_b)
    stB = gsB.spacetime()

    # W_AB: the realizability oracle fills the pinned-boundary wheel interior.
    bulk = _wheel_bulk()
    flat = [complex(z) for row in U for z in row]
    verdict = cob.RealizabilityOracle(bulk).decide(
        flat, 2, 2, epsilon=epsilon_w, restarts=restarts, max_cones=0,
        seed=seed_w)

    ids_a = sorted(v.getId() for v in stA.getVertexList().toVector())
    ids_b = sorted(v.getId() for v in stB.getVertexList().toVector())
    ampA = {ids_a[0]: complex(psiA[0]), ids_a[1]: complex(psiA[1])}
    ampB = {ids_b[0]: complex(psiB[0]), ids_b[1]: complex(psiB[1])}

    return dict(U=np.asarray(U, dtype=complex), psiA=psiA, psiB=psiB,
                stA=stA, stB=stB, geoA=geoA, geoB=geoB,
                ampA=ampA, ampB=ampB,
                witness=verdict.witness, state=np.asarray(verdict.state),
                target=np.asarray(verdict.target),
                residual=verdict.residual, realizable=verdict.realizable,
                eigenvalue=verdict.eigenvalue,
                interior_vertex_count=verdict.interior_vertex_count)


def render(out_dir="/tmp/cobordism", *, U=None, layout_seed=42, azim=35,
           dpi=130, **pipeline_kwargs):
    """Run the pipeline and write the four-panel composite PNG.

    Returns the dict from :func:`run_pipeline` augmented with ``composite`` (the
    PNG path).
    """
    os.makedirs(out_dir, exist_ok=True)
    res = run_pipeline(U=U, **pipeline_kwargs)

    # Shared W_AB layout (panels 3 and 4): one seeded force layout, pca-aligned.
    gW = _graph(res["witness"])
    posW, _vmap, _eidx = layout_from_spacetime(gW["verts"], gW["edges"],
                                               seed=layout_seed, iters=500)
    posW_n = _normalize(pca_align(posW)[0])

    capA = _caption("System A -- geo(psi_A)", res["stA"])
    capB = _caption("System B -- geo(psi_B)", res["stB"])
    capBefore = "Before gluing -- disjoint A | output surface | B"
    flag = "realizable" if res["realizable"] else "obstructed"
    capAfter = _caption(
        "After gluing -- W_AB (interior filled)", res["witness"],
        extra=f"\nr = {res['residual']:.1e}  ({flag}),  lambda = "
              f"{res['eigenvalue']:.3f}")

    img_a = _panel_geo(res["stA"], res["ampA"], azim=azim,
                       layout_seed=layout_seed)
    img_b = _panel_geo(res["stB"], res["ampB"], azim=azim,
                       layout_seed=layout_seed)
    img_before = _panel_before(res["stA"], res["stB"], gW, posW_n, azim=azim,
                               layout_seed=layout_seed)
    img_after = _panel_after(gW, posW_n, res["state"], azim=azim)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.2))
    panels = [(img_a, capA), (img_b, capB),
              (img_before, capBefore), (img_after, capAfter)]
    for ax, (img, cap) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(cap, fontsize=10.5, linespacing=1.3)
    Umat = res["U"]
    fig.suptitle(
        "Realizability of U : H_B -> H_A  --  systems A, B, and the "
        "W_AB cobordism\n"
        f"U = [[{Umat[0,0]:.2g}, {Umat[0,1]:.2g}], "
        f"[{Umat[1,0]:.2g}, {Umat[1,1]:.2g}]]   "
        f"(realizable, r = {res['residual']:.1e})",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    composite = os.path.join(out_dir, "visualize_realizability.png")
    fig.savefig(composite, dpi=dpi)
    plt.close(fig)
    res["composite"] = composite
    return res


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="output directory for the composite PNG "
                         "(default /tmp/cobordism, not committed).")
    ap.add_argument("--layout-seed", type=int, default=42,
                    help="seed for the force-directed layout (default 42).")
    ap.add_argument("--restarts", type=int, default=96,
                    help="multi-restart count for the synthesizers (default 96).")
    ap.add_argument("--azim", type=float, default=35.0,
                    help="azimuth of the 3D view in degrees (default 35).")
    ap.add_argument("--dpi", type=int, default=130, help="PNG dpi (default 130).")
    args = ap.parse_args()

    res = render(out_dir=args.out, layout_seed=args.layout_seed,
                 azim=args.azim, dpi=args.dpi, restarts=args.restarts)

    print("Realizability visualization -- four-panel composite\n")
    print(f"  U realizable : {res['realizable']}  "
          f"(residual r = {res['residual']:.3e})")
    print(f"  geo(psi_A)   : |V| {res['stA'].getVertexCount()}  "
          f"|E| {res['stA'].getEdgeList().size()}  "
          f"(cones {res['geoA'].cones_applied}, r {res['geoA'].residual:.1e})")
    print(f"  geo(psi_B)   : |V| {res['stB'].getVertexCount()}  "
          f"|E| {res['stB'].getEdgeList().size()}  "
          f"(cones {res['geoB'].cones_applied}, r {res['geoB'].residual:.1e})")
    print(f"  W_AB witness : |V| {res['witness'].getVertexCount()}  "
          f"|E| {res['witness'].getEdgeList().size()}  "
          f"(interior vertices {res['interior_vertex_count']})")
    print(f"\n  composite PNG: {res['composite']}")

    raise SystemExit(0 if res["realizable"] else 1)


if __name__ == "__main__":
    main()
