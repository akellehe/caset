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

"""Visualize the k=1 State-Operation-Cobordism re-test (#202).

Makes the k=1 re-test (``correspondence_retest_k1.py``) legible: for every
operator/target the re-test exercises -- the **longitude**, the **meridian**, the
two **ker L_1 basis** harmonics, and the **H3** transition ``(psi_A, psi_B, U)``
-- render the four diagrams the State-Operation-Cobordism story calls for, all
through the project's existing layout/render infrastructure (``tessera.utils.plot``
``layout_from_spacetime`` / ``render_frame`` / ``draw_edges`` / ``pca_align``, plus
the panel helpers reused from ``visualize_realizability.py``; no hand-rolled
plotter):

  1. ``geo(psi_A)`` -- the geometric image of the boundary state the operator
     asks the bulk to carry (the target boundary harmonic, in its qubit
     coordinates on ker L_1(T^2)), synthesized by ``GeometrySynthesizer`` (#134):
     the minimal complex whose k=0 Hodge-Laplacian has ``psi_A`` as an
     eigenvector. Vertices colored by |amplitude| (phase -> hue).
  2. ``geo(psi_B)`` -- the same for ``psi_B``: for the four boundary-harmonic
     targets this is the **homology class the solid torus actually carries** (the
     longitude, the generator of H_1(W)), the invariant the re-test compares
     against; for the H3 case it is the genuine right boundary state.
  3. **The candidate operator cobordism W** -- the grown solid-torus complex the
     free-connectivity engine builds for that target (``decideHarmonic``'s
     ``witness``). Boundary dW = T^2 drawn thick/solid, the filled interior thin/
     dashed, every edge colored by the realized **k=1 harmonic 1-form** (the
     ``Verdict.state``): for the longitude the 1-form lights up the carried cycle
     (r -> 0); for a floored target it cannot match (r floors), and the engine
     has coned in one Pachner vertex (free == cone at k=1).
  4. **Everything connected** -- ``geo(psi_A) || W || geo(psi_B)`` assembled in one
     shared layout: the boundary state on the left, the operator cobordism in the
     middle, the carried/target state on the right.

The realizability story reads straight off the panels: a target is realizable iff
``geo(psi_A)`` matches ``geo(psi_B)`` -- iff the requested boundary harmonic IS the
carried homology class. The longitude matches (realized, r ~ 1e-29); the meridian
and the raw basis do not (floored).

Images are written to ``/tmp/cobordism`` and are NOT committed to the repo tree
(this script is the committed artifact; the PNGs are uploaded to the
``issue-attachments`` release and referenced from the docs report by URL). The
layout is seeded for reproducibility and the BLAS/OpenMP pools are capped at the
box's shared-workload budget.

Run:  python examples/cobordism/visualize_correspondence_retest_k1.py
      (--help for options; writes 4 PNGs per operator into /tmp/cobordism)
"""

from __future__ import annotations

# Cap BLAS / OpenMP threads at launch (shared box) before numpy / tessera import.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

import tessera  # noqa: E402
from tessera.utils.plot import (layout_from_spacetime, render_frame,  # noqa: E402
                                draw_edges, pca_align)

# Sibling example imports (the re-test fixtures + the locked render helpers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correspondence_retest_k1 as R  # noqa: E402
from visualize_realizability import (  # noqa: E402
    _from_simplices, _graph, _amp_rgba, _amp_size, _normalize, _finish,
    _draw_vertices, _set_equal_3d, _amp_vector, _caption,
    COLOR_A, COLOR_B, COLOR_INT)

cob = tessera.cobordism
FREE = R.FREE

ZERO_EDGE = (0.74, 0.74, 0.77, 0.45)   # decoupled (|amp| ~ 0) edges desaturate


# =========================================================================
# Boundary-qubit coordinates + geo(psi) synthesis
# =========================================================================

def _gauge(c0, c1):
    """Normalize a qubit and fix the global phase so c0 is real >= 0."""
    z = np.array([complex(c0), complex(c1)], dtype=complex)
    n = np.linalg.norm(z)
    if n > 1e-15:
        z = z / n
    if abs(z[0]) > 1e-12:
        z = z * np.exp(-1j * np.angle(z[0]))
    return complex(z[0]), complex(z[1])


def _qubit_coords(target, harmonics):
    """The (c0, c1) coordinates of a boundary harmonic in the ker L_1(T^2)
    generator frame -- least-squares projection onto the two harmonics."""
    H = np.column_stack([np.asarray(h.coeffs()) for h in harmonics])
    ab, *_ = np.linalg.lstsq(H, np.asarray(target.coeffs()), rcond=None)
    return _gauge(ab[0], ab[1])


def _geo(c0, c1, *, restarts=80, max_cones=4, seed=0, epsilon=1e-9):
    """geo(psi): the minimal complex whose k=0 Hodge Laplacian carries (c0, c1)
    (GeometrySynthesizer, the #134 idiom). Returns (spacetime, amp_by_id, geo)."""
    gs = cob.GeometrySynthesizer(_from_simplices(2, [(0, 1)]))
    geo = gs.synthesize(complex(c0), complex(c1), epsilon=epsilon,
                        restarts=restarts, max_cones=max_cones, seed=seed)
    st = gs.spacetime()
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    amp = {ids[0]: complex(c0), ids[1]: complex(c1)}
    return st, amp, geo


# =========================================================================
# The candidate operator cobordism W (the grown solid torus)
# =========================================================================

def _witness_for(cochain, *, max_cones, seed=1, restarts=8):
    """The grown bulk W the engine builds for a target boundary harmonic: free-
    connectivity decideHarmonic on a fresh pinned solid torus, boundary dW fixed."""
    W = R._pinned_solid_torus()
    return cob.RealizabilityOracle(W).decideHarmonic(
        cochain, epsilon=R.EPSILON, restarts=restarts, max_cones=max_cones,
        seed=seed, growth_mode=FREE, connectivity_candidates=8)


def _edge_amp(witness, state):
    """Map the realized k=1 harmonic 1-form (Verdict.state, one component per
    k=1 cell) onto edges keyed by sorted vertex-id pair."""
    cells = [tuple(int(i) for i in c)
             for c in cob.EigenstateSynthesis(witness, 1).cellSimplices()]
    amp = np.asarray(state, dtype=complex)
    out = {}
    for i, c in enumerate(cells):
        if len(c) == 2 and i < len(amp):
            out[(min(c), max(c))] = amp[i]
    return out


# =========================================================================
# Edge-amplitude drawing (the k=1 1-form), via the same Line3DCollection
# primitive draw_edges() is built on
# =========================================================================

def _draw_amp_edges(ax, pos, edge_idx, amps, max_mag, *, lw=1.6, ls="solid"):
    if not edge_idx:
        return
    segs = [[pos[a], pos[b]] for a, b in edge_idx]
    colors = [ZERO_EDGE if abs(z) <= 1e-9 else _amp_rgba(z, max_mag)
              for z in amps]
    ax.add_collection(Line3DCollection(segs, linewidths=lw, colors=colors,
                                       linestyles=ls))


def _w_edge_amps(graph, edge_amp):
    return [edge_amp.get((min(graph["ids"][a], graph["ids"][b]),
                          max(graph["ids"][a], graph["ids"][b])), 0.0 + 0.0j)
            for a, b in graph["edge_idx"]]


# =========================================================================
# Panel builders
# =========================================================================

def _panel_geo(st, amp_by_id, *, azim=35, layout_seed=42):
    """A single synthesized boundary manifold geo(psi); vertices by |psi|."""
    g = _graph(st)
    pos, _vmap, edge_idx = layout_from_spacetime(g["verts"], g["edges"],
                                                 seed=layout_seed, iters=400)
    pos = _normalize(pca_align(pos)[0])
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


def _panel_cobordism(witness, edge_amp, *, azim=35, layout_seed=42):
    """The grown solid-torus cobordism W: boundary dW (T^2) thick/solid, interior
    thin/dashed, every edge colored by the realized k=1 harmonic 1-form."""
    g = _graph(witness)
    pos, _vmap, _eidx = layout_from_spacetime(g["verts"], g["edges"],
                                              seed=layout_seed, iters=500)
    pos = _normalize(pca_align(pos)[0])
    amps = _w_edge_amps(g, edge_amp)
    max_mag = float(np.max(np.abs(amps))) if amps else 1.0
    max_mag = max_mag or 1.0
    bnd = g["bnd_edge"]

    def draw(ax):
        int_e = [e for e, b in zip(g["edge_idx"], bnd) if not b]
        int_a = [z for z, b in zip(amps, bnd) if not b]
        bnd_e = [e for e, b in zip(g["edge_idx"], bnd) if b]
        bnd_a = [z for z, b in zip(amps, bnd) if b]
        _draw_amp_edges(ax, pos, int_e, int_a, max_mag, lw=1.0, ls="dashed")
        _draw_amp_edges(ax, pos, bnd_e, bnd_a, max_mag, lw=2.6, ls="solid")
        _draw_vertices(ax, pos, [(0.55, 0.55, 0.58, 0.95)] * len(pos),
                       [55.0] * len(pos), edgecolor=COLOR_INT, lw=0.6)
        _set_equal_3d(ax, pos)
        ax.text2D(0.02, 0.02,
                  "boundary dW = T^2: thick solid\n"
                  "interior fill: thin dashed\n"
                  "edge color = realized k=1 1-form",
                  transform=ax.transAxes, fontsize=8.5, color=(0.2, 0.2, 0.2))

    return _finish(render_frame(draw, figsize=(6.4, 6.4), azim=azim))


def _panel_assembled(stA, ampA, witness, edge_amp, stB, ampB, *, azim=35,
                     layout_seed=42):
    """geo(psi_A) || W || geo(psi_B): three components in one shared layout."""
    gA, gB, gW = _graph(stA), _graph(stB), _graph(witness)
    posA = _normalize(pca_align(layout_from_spacetime(
        gA["verts"], gA["edges"], seed=layout_seed, iters=400)[0])[0])
    posB = _normalize(pca_align(layout_from_spacetime(
        gB["verts"], gB["edges"], seed=layout_seed, iters=400)[0])[0])
    posW = _normalize(pca_align(layout_from_spacetime(
        gW["verts"], gW["edges"], seed=layout_seed, iters=500)[0])[0])

    ampAv, ampBv = _amp_vector(stA, ampA), _amp_vector(stB, ampB)
    mA = float(np.abs(ampAv).max()) or 1.0
    mB = float(np.abs(ampBv).max()) or 1.0
    wamps = _w_edge_amps(gW, edge_amp)
    mW = (float(np.max(np.abs(wamps))) if wamps else 1.0) or 1.0
    bndW = gW["bnd_edge"]

    gap = 3.0
    posA = posA + np.array([-gap, 0.0, 0.0])
    posB = posB + np.array([+gap, 0.0, 0.0])

    def draw(ax):
        draw_edges(ax, posA, gA["edge_idx"], default_color=(*COLOR_A, 0.5),
                   linewidth=1.0)
        _draw_vertices(ax, posA, [_amp_rgba(z, mA) for z in ampAv],
                       [_amp_size(z, mA) for z in ampAv], edgecolor=COLOR_A,
                       lw=0.6)
        int_e = [e for e, b in zip(gW["edge_idx"], bndW) if not b]
        int_a = [z for z, b in zip(wamps, bndW) if not b]
        bnd_e = [e for e, b in zip(gW["edge_idx"], bndW) if b]
        bnd_a = [z for z, b in zip(wamps, bndW) if b]
        _draw_amp_edges(ax, posW, int_e, int_a, mW, lw=0.9, ls="dashed")
        _draw_amp_edges(ax, posW, bnd_e, bnd_a, mW, lw=2.2, ls="solid")
        _draw_vertices(ax, posW, [(0.55, 0.55, 0.58, 0.9)] * len(posW),
                       [45.0] * len(posW), edgecolor=COLOR_INT, lw=0.5)
        draw_edges(ax, posB, gB["edge_idx"], default_color=(*COLOR_B, 0.5),
                   linewidth=1.0)
        _draw_vertices(ax, posB, [_amp_rgba(z, mB) for z in ampBv],
                       [_amp_size(z, mB) for z in ampBv], edgecolor=COLOR_B,
                       lw=0.6)
        _set_equal_3d(ax, np.vstack([posA, posW, posB]), cube=False)
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A,
                   markersize=10, label="geo(psi_A)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_INT,
                   markersize=10, label="W (solid torus, k=1 1-form)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_B,
                   markersize=10, label="geo(psi_B)"),
        ]
        ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.0,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))

    return _finish(render_frame(draw, figsize=(7.8, 6.4), azim=azim))


# =========================================================================
# Per-operator orchestration
# =========================================================================

def _save_panel(frame, caption, path, *, dpi=120, figsize=(6.6, 7.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame)
    ax.set_axis_off()
    ax.set_title(caption, fontsize=11, linespacing=1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def build_case(case, *, out_dir, azim, layout_seed, dpi, restarts):
    """Render and save the four diagrams for one operator/target."""
    qA = _gauge(*case["qA"])
    qB = _gauge(*case["qB"])

    stA, ampA, geoA = _geo(*qA, restarts=restarts, seed=case["seed_a"])
    stB, ampB, geoB = _geo(*qB, restarts=restarts, seed=case["seed_b"])
    v = _witness_for(case["cochain"], max_cones=case["max_cones"])
    edge_amp = _edge_amp(v.witness, v.state)

    flag = "realizable" if v.realizable else "floored"
    key = case["key"]
    paths = {}

    cap_a = _caption(f"1. geo(psi_A) -- {case['label']}", stA,
                     extra=f"\npsi_A = ({qA[0]:.2g}, {qA[1]:.2g})  "
                           f"(r={geoA.residual:.1e})")
    cap_b = _caption(f"2. geo(psi_B) -- {case['psiB_label']}", stB,
                     extra=f"\npsi_B = ({qB[0]:.2g}, {qB[1]:.2g})  "
                           f"(r={geoB.residual:.1e})")
    cap_w = _caption(f"3. operator cobordism W ({flag})", v.witness,
                     extra=f"\nr = {v.residual:.2e}  cones = {v.cones_applied}  "
                           f"tri_cand = {v.triangle_candidates}")
    cap_x = (f"4. geo(psi_A) || W || geo(psi_B) assembled -- {case['label']}\n"
             f"{flag}: r = {v.residual:.2e}  "
             f"(realizable iff geo(psi_A) == geo(psi_B))")

    img_a = _panel_geo(stA, ampA, azim=azim, layout_seed=layout_seed)
    img_b = _panel_geo(stB, ampB, azim=azim, layout_seed=layout_seed)
    img_w = _panel_cobordism(v.witness, edge_amp, azim=azim,
                             layout_seed=layout_seed)
    img_x = _panel_assembled(stA, ampA, v.witness, edge_amp, stB, ampB,
                             azim=azim, layout_seed=layout_seed)

    for tag, img, cap, fsize in [
            ("geoA", img_a, cap_a, (6.6, 7.0)),
            ("geoB", img_b, cap_b, (6.6, 7.0)),
            ("W", img_w, cap_w, (6.6, 7.0)),
            ("assembled", img_x, cap_x, (8.0, 7.0))]:
        path = os.path.join(out_dir, f"retest_k1_{key}_{tag}.png")
        _save_panel(img, cap, path, dpi=dpi, figsize=fsize)
        paths[tag] = path

    return dict(key=key, label=case["label"], realizable=bool(v.realizable),
                residual=float(v.residual), cones=int(v.cones_applied),
                triangle_candidates=int(v.triangle_candidates),
                witness_V=v.witness.getVertexCount(),
                witness_E=v.witness.getEdgeList().size(), paths=paths)


def _cases():
    """The five operators/targets the re-test exercises, with their qubit ends."""
    space = cob.BoundaryStateSpace(R._torus())
    H = space.harmonics()
    longitude, meridian = R._longitude_and_meridian(R._pinned_solid_torus(), space)
    q_long = _qubit_coords(longitude, H)        # the carried homology class
    q_meri = _qubit_coords(meridian, H)
    q_b0 = _qubit_coords(H[0], H)
    q_b1 = _qubit_coords(H[1], H)
    carried = "carried class H_1(W) (longitude)"
    return [
        dict(key="longitude", label="longitude (carried)", cochain=longitude,
             qA=q_long, qB=q_long, psiB_label=carried, max_cones=0,
             seed_a=3, seed_b=3),
        dict(key="meridian", label="meridian (bounds a disk)", cochain=meridian,
             qA=q_meri, qB=q_long, psiB_label=carried, max_cones=1,
             seed_a=5, seed_b=3),
        dict(key="basis0", label="ker L_1 basis #0", cochain=H[0],
             qA=q_b0, qB=q_long, psiB_label=carried, max_cones=1,
             seed_a=7, seed_b=3),
        dict(key="basis1", label="ker L_1 basis #1", cochain=H[1],
             qA=q_b1, qB=q_long, psiB_label=carried, max_cones=1,
             seed_a=11, seed_b=3),
        # H3: the genuine transition (psi_A, psi_B, U=I) -- the Choi anchor; W is
        # the realized longitude cobordism (the witness the re-test reads Z(W) off).
        dict(key="h3", label="H3 transition (psi_A, psi_B, U=I)", cochain=longitude,
             qA=(1.0 + 0.0j, 0.5j), qB=(0.6 + 0.0j, 0.8 + 0.0j),
             psiB_label="H3 right state psi_B", max_cones=0,
             seed_a=13, seed_b=17),
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="output directory for the PNGs (default /tmp/cobordism, "
                         "not committed).")
    ap.add_argument("--layout-seed", type=int, default=42,
                    help="seed for the force-directed layout (default 42).")
    ap.add_argument("--restarts", type=int, default=80,
                    help="multi-restart count for the geo synthesizers (default 80).")
    ap.add_argument("--azim", type=float, default=35.0,
                    help="azimuth of the 3D view in degrees (default 35).")
    ap.add_argument("--dpi", type=int, default=120, help="PNG dpi (default 120).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("k=1 State-Operation-Cobordism re-test -- four diagrams per operator\n")

    results = []
    for case in _cases():
        res = build_case(case, out_dir=args.out, azim=args.azim,
                         layout_seed=args.layout_seed, dpi=args.dpi,
                         restarts=args.restarts)
        results.append(res)
        verdict = "realizable" if res["realizable"] else "floored"
        print(f"  {res['label']:32} {verdict:11} r={res['residual']:.2e}  "
              f"W |V|={res['witness_V']} |E|={res['witness_E']} "
              f"cones={res['cones']} tri_cand={res['triangle_candidates']}")
        for tag in ("geoA", "geoB", "W", "assembled"):
            print(f"        {tag:10} -> {res['paths'][tag]}")

    n = sum(len(r["paths"]) for r in results)
    print(f"\n  wrote {n} diagrams ({len(results)} operators x 4) into {args.out}")
    print("  (PNGs are PR artifacts -- upload to the issue-attachments release; "
          "not committed.)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
