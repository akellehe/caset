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

"""Visualize emergent-bulk realizability: surgery makes b_1 a pure output (#207).

Make the emergent-bulk story legible. The boundary is two meridian circles
geo(psi_A) || geo(psi_B); the bulk is grown from a validity-only seed; surgery
opens the handle so b_1 moves on its own, and the matched meridian realizes
exactly where b_1=1.

The octahedron (a triangulated S^2) is drawn as a Schlegel diagram on ONE fixed
layout, so the topology of each filling reads at a glance:

  * hole A = {0,1,2} is the OUTER triangle, hole B = {3,4,5} the INNER triangle,
    with the six "band" 2-cells between them;
  * a FILLED 2-cell is shaded; a removed face is an OPEN (white, dashed) hole --
    a boundary circle dW.

So the panels are immediate:

  1. geo(psi_A) || geo(psi_B) -- the two bare boundary circles (outer A, inner B),
     edges colored by the matched meridian 1-form (equal periods p_A = p_B).
  2. disk filling (b_1=0)     -- only hole A is open; the inner triangle {3,4,5} is
     FILLED (a disk), so circle B bounds it. The meridian FLOORS.
  3. annulus filling (b_1=1)  -- BOTH triangles open: the six band cells form a
     ring (an annulus). Edges colored by the realized carried harmonic 1-form. The
     meridian REALIZES, eigenvalue -> 0.
  4. surgery: disk -> annulus -- the boundary-fixed REMOVE of the interior top cell
     {3,4,5} punches the inner triangle out: b_1 0 -> 1, dW held bit-exact.

A fifth composite tiles 1-4. Every panel carries a large title and an explicit
legend. Images go to /tmp/cobordism and are NOT committed (this script is the
committed artifact; the PNGs are uploaded to the issue-attachments release and
referenced from the docs report by URL). BLAS/OpenMP pools are capped at the box's
shared budget.

Run:  python examples/cobordism/visualize_emergent_bulk.py
      (--help for options; writes the panels + composite into /tmp/cobordism)
"""

from __future__ import annotations

# Cap BLAS / OpenMP threads at launch (shared box) before numpy / tessera import.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "10")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

import tessera  # noqa: E402

# Sibling example import: the fixtures + the decide() helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emergent_bulk_realizability as E  # noqa: E402
from visualize_realizability import _amp_rgba  # noqa: E402

cob = tessera.cobordism

HOLE_A = tuple(sorted(E.HOLE_A))    # {0,1,2} -- outer triangle
HOLE_B = tuple(sorted(E.HOLE_B))    # {3,4,5} -- inner triangle

# Fixed Schlegel layout: hole A is the big outer triangle, hole B the small inner
# triangle, each inner vertex placed between the two outer vertices it bonds to
# (3<->{0,2}, 4<->{0,1}, 5<->{1,2}), so the six band faces never cross.
_R, _r = 1.0, 0.44
POS = {
    0: (_R * np.cos(np.radians(90)),  _R * np.sin(np.radians(90))),
    1: (_R * np.cos(np.radians(210)), _R * np.sin(np.radians(210))),
    2: (_R * np.cos(np.radians(330)), _R * np.sin(np.radians(330))),
    3: (_r * np.cos(np.radians(30)),  _r * np.sin(np.radians(30))),
    4: (_r * np.cos(np.radians(150)), _r * np.sin(np.radians(150))),
    5: (_r * np.cos(np.radians(270)), _r * np.sin(np.radians(270))),
}

FACE_FILL = (0.78, 0.84, 0.92, 0.85)     # a filled 2-cell (light blue)
SURGERY_FILL = (0.96, 0.78, 0.78, 0.95)  # the cell surgery removes (red wash)
HOLE_EDGE = (0.45, 0.45, 0.50)           # open-hole dashed outline
ZERO_EDGE = (0.72, 0.72, 0.76)           # decoupled (|amp| ~ 0) edge


# =========================================================================
# Topology + the 1-form on edges
# =========================================================================

def _faces(st):
    out = set()
    for v in st.getVertexList().toVector():
        for s in v.getSimplices():
            ids = tuple(sorted(x.getId() for x in s.getVertices()))
            if len(ids) == 3:
                out.add(ids)
    return sorted(out)


def _edges(st):
    out = set()
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        if a != b:
            out.add((min(a, b), max(a, b)))
    return sorted(out)


def _boundary_keys(st):
    keys = set()
    for face in st.getBoundary():
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                keys.add((min(face[i], face[j]), max(face[i], face[j])))
    return keys


def _target_edge_amp(target):
    out = {}
    for cell, c in zip(target.simplices(), np.asarray(target.coeffs())):
        ids = tuple(int(i) for i in cell)
        if len(ids) == 2:
            out[(min(ids), max(ids))] = complex(c)
    return out


def _witness_edge_amp(witness, state):
    cells = [tuple(int(i) for i in c)
             for c in cob.EigenstateSynthesis(witness, 1).cellSimplices()]
    amp = np.asarray(state, dtype=complex)
    out = {}
    for i, c in enumerate(cells):
        if len(c) == 2 and i < len(amp):
            out[(min(c), max(c))] = amp[i]
    return out


# =========================================================================
# 2D drawing on the fixed Schlegel layout
# =========================================================================

def _draw_complex(ax, st, edge_amp, *, max_mag, draw_faces=True, punch_holes=(),
                  surgery_cell=None, only_edges=None, label_holes=True):
    """Draw a filling on the Schlegel layout: shaded 2-cells, the inner hole(s)
    punched white, the 1-form on edges (boundary thick, interior dashed), vertices.

    hole A = {0,1,2} is the OUTER triangle (the unbounded Schlegel face) -- it is
    never punched; its thick outer edges ARE the boundary circle A. Only inner
    removed faces (hole B) are white-punched, so a disk reads as the fully filled
    outer triangle and an annulus as that triangle with B punched out (a ring)."""
    bkeys = _boundary_keys(st)
    surg = tuple(sorted(surgery_cell)) if surgery_cell else None

    # Filled 2-cells (shade); the surgery cell gets the red wash.
    if draw_faces:
        for f in _faces(st):
            col = SURGERY_FILL if f == surg else FACE_FILL
            ax.add_patch(Polygon([POS[v] for v in f], closed=True, facecolor=col,
                                 edgecolor="none", zorder=1))

    # Inner holes (removed bounded faces): white interior + dashed outline + label.
    for h in punch_holes:
        h = tuple(sorted(h))
        ax.add_patch(Polygon([POS[v] for v in h], closed=True, facecolor="white",
                             edgecolor=HOLE_EDGE, lw=1.6, ls=(0, (4, 3)),
                             zorder=2))
        if label_holes:
            cx = np.mean([POS[v][0] for v in h])
            cy = np.mean([POS[v][1] for v in h])
            tag = "B" if h == HOLE_B else ("A" if h == HOLE_A else "?")
            ax.text(cx, cy, f"hole {tag}\n(dW)", ha="center", va="center",
                    fontsize=10.5, color=HOLE_EDGE, zorder=3, style="italic")

    # Edges colored by the 1-form; boundary thick solid, interior thin dashed.
    edges = only_edges if only_edges is not None else _edges(st)
    for (u, v) in edges:
        key = (min(u, v), max(u, v))
        z = edge_amp.get(key, 0.0 + 0.0j)
        col = ZERO_EDGE if abs(z) <= 1e-9 else _amp_rgba(z, max_mag)[:3]
        is_bnd = key in bkeys
        (x0, y0), (x1, y1) = POS[u], POS[v]
        ax.plot([x0, x1], [y0, y1], color=col,
                lw=5.0 if is_bnd else 1.8,
                ls="solid" if is_bnd else (0, (5, 3)),
                solid_capstyle="round", zorder=4)

    # Vertices, labeled.
    vids = sorted({v for e in _edges(st) for v in e})
    for vid in vids:
        x, y = POS[vid]
        ax.plot([x], [y], "o", color=(0.18, 0.18, 0.22), ms=11, zorder=5)
        ax.annotate(str(vid), (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=13, fontweight="bold",
                    color=(0.1, 0.1, 0.1), zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.5)
    ax.set_ylim(-1.3, 1.35)
    ax.set_axis_off()


# =========================================================================
# Verdicts (seed) + the surgery-grown witness
# =========================================================================

def _verdict(fill_fn, target):
    st = fill_fn()
    return st, E._decide(st, target, mode=E.SURGERY, max_cones=0)


def _surgery_grow(target, seed=0):
    st = E._disk()
    return st, E._decide(st, target, mode=E.SURGERY, max_cones=E.GROW_STEPS,
                         seed=seed)


# =========================================================================
# Legends
# =========================================================================

def _form_legend(max_mag):
    bright = _amp_rgba(complex(max_mag), max_mag)[:3]
    return [
        Line2D([0], [0], color=bright, lw=5, label="boundary circle dW (thick)"),
        Line2D([0], [0], color=ZERO_EDGE, lw=1.8, ls=(0, (5, 3)),
               label="interior edge (thin dashed)"),
        Patch(facecolor=FACE_FILL, edgecolor="none", label="filled 2-cell"),
        Patch(facecolor="white", edgecolor=HOLE_EDGE, ls="--",
              label="open hole = boundary circle"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=bright,
               markersize=11, label="edge hue = 1-form phase, brightness = |amp|"),
    ]


def _save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Orchestration
# =========================================================================

def build(out_dir, *, dpi=130, layout_seed=42):
    matched, _edges_t, mvals = E._meridian_target(flip=False)
    p_a, p_b = E._periods(mvals)
    tgt_amp = _target_edge_amp(matched)
    tgt_max = max((abs(z) for z in tgt_amp.values()), default=1.0)

    disk_st, disk_v = _verdict(E._disk, matched)
    ann_st, ann_v = _verdict(E._annulus, matched)
    grow_st, grow_v = _surgery_grow(matched, seed=layout_seed)
    b1 = (E._betti(disk_st)[1], E._betti(ann_st)[1], E._betti(grow_st)[1])

    ann_amp = _witness_edge_amp(ann_v.witness, ann_v.state)
    ann_max = max((abs(z) for z in ann_amp.values()), default=1.0)
    grow_amp = _witness_edge_amp(grow_v.witness, grow_v.state)
    grow_max = max((abs(z) for z in grow_amp.values()), default=1.0)

    paths = {}

    # --- 1. geo(psi_A) || geo(psi_B): the two bare boundary circles only. ---
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    _draw_complex(ax, ann_st, tgt_amp, max_mag=tgt_max, draw_faces=False,
                  only_edges=E.CYCLE_A + E.CYCLE_B, label_holes=False)
    ax.text(*POS[0], "  circle A (m_A)", fontsize=12, fontweight="bold",
            color=(0.78, 0.12, 0.12), va="bottom")
    ax.text(0.0, -0.02, "circle B (m_B)", fontsize=11, fontweight="bold",
            color=(0.12, 0.3, 0.78), ha="center")
    ax.set_title("geo(psi_A)  ||  geo(psi_B):  the two boundary meridians\n"
                 f"matched, equal periods  p_A = {p_a.real:+.2f},  "
                 f"p_B = {p_b.real:+.2f}", fontsize=15, fontweight="bold",
                 linespacing=1.3)
    ax.legend(handles=[
        Line2D([0], [0], color=_amp_rgba(complex(tgt_max), tgt_max)[:3], lw=5,
               label="meridian 1-form on a boundary circle"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=_amp_rgba(complex(tgt_max), tgt_max)[:3],
               markersize=11, label="hue = phase, brightness = |amplitude|")],
        loc="lower center", ncol=1, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, -0.13))
    paths["boundary"] = os.path.join(out_dir, "emergent_bulk_boundary.png")
    _save(fig, paths["boundary"], dpi)

    # --- 2. disk filling (b_1=0): meridian floors. ---
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    _draw_complex(ax, disk_st, tgt_amp, max_mag=tgt_max)
    ax.set_title("Disk filling   (b_1 = 0)\n"
                 f"meridian FLOORS:  r = {disk_v.residual:.2e}   "
                 f"(eigenvalue {disk_v.eigenvalue:.2e})",
                 fontsize=15, fontweight="bold", linespacing=1.3)
    ax.text(0.0, -1.22, "the inner triangle {3,4,5} is FILLED, so circle B bounds "
            "it: the disk carries no nontrivial harmonic", ha="center",
            fontsize=10.5, style="italic", color=(0.25, 0.25, 0.25))
    ax.legend(handles=_form_legend(tgt_max), loc="lower center", ncol=2,
              fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.20))
    paths["disk"] = os.path.join(out_dir, "emergent_bulk_disk.png")
    _save(fig, paths["disk"], dpi)

    # --- 3. annulus filling (b_1=1): the carried harmonic; meridian realizes. ---
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    _draw_complex(ax, ann_st, ann_amp, max_mag=ann_max, punch_holes=(HOLE_B,))
    ax.set_title("Annulus filling   (b_1 = 1)   --   the harmonic on the annulus\n"
                 f"meridian REALIZES:  r = {ann_v.residual:.2e}   "
                 f"(eigenvalue {ann_v.eigenvalue:.1e} ~ 0)",
                 fontsize=15, fontweight="bold", linespacing=1.3)
    ax.text(0.0, -1.22, "both triangles removed: the six band cells form a ring; "
            "the cobordism carries the meridian as a bulk harmonic", ha="center",
            fontsize=10.5, style="italic", color=(0.25, 0.25, 0.25))
    ax.legend(handles=_form_legend(ann_max), loc="lower center", ncol=2,
              fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.20))
    paths["annulus"] = os.path.join(out_dir, "emergent_bulk_annulus.png")
    _save(fig, paths["annulus"], dpi)

    # --- 4. surgery: disk -> annulus (before / after). ---
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 8.0))
    _draw_complex(axes[0], E._disk(), tgt_amp, max_mag=tgt_max,
                  surgery_cell=HOLE_B)
    axes[0].set_title("before: disk seed  (b_1 = 0)\n"
                      "surgery will REMOVE interior cell {3,4,5} (red)",
                      fontsize=13.5, fontweight="bold", linespacing=1.3)
    _draw_complex(axes[1], grow_st, grow_amp, max_mag=grow_max,
                  punch_holes=(HOLE_B,))
    axes[1].set_title(f"after: handle open  (b_1 = {b1[2]})\n"
                      f"matched meridian REALIZES:  r = {grow_v.residual:.2e}",
                      fontsize=13.5, fontweight="bold", linespacing=1.3)
    fig.suptitle("Surgery move:  b_1 moves 0 -> 1 on its own  "
                 f"(removals = {grow_v.surgery_removals}, dW held bit-exact)",
                 fontsize=16.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths["surgery"] = os.path.join(out_dir, "emergent_bulk_surgery.png")
    _save(fig, paths["surgery"], dpi)

    # --- 4b. the period sign matters: matched vs flipped on the SAME annulus. ---
    # Same topology (b_1=1); only circle B's period sign differs. The conjugation
    # geo(psi_A) || conj(geo(psi_B)) (p_A = -p_B) floors even here -- the
    # cohomological obstruction is separate from the topological one.
    flipped, _fe, fvals = E._meridian_target(flip=True)
    fp_a, fp_b = E._periods(fvals)
    ftgt_amp = _target_edge_amp(flipped)
    ftgt_max = max((abs(z) for z in ftgt_amp.values()), default=tgt_max)
    _, fann_v = _verdict(E._annulus, flipped)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 8.2))
    _draw_complex(axes[0], ann_st, tgt_amp, max_mag=tgt_max, punch_holes=(HOLE_B,))
    axes[0].set_title("matched   p_A = p_B = "
                      f"{p_a.real:+.2f}\nmeridian REALIZES:  r = {ann_v.residual:.2e}",
                      fontsize=13.5, fontweight="bold", linespacing=1.3,
                      color=(0.0, 0.45, 0.0))
    _draw_complex(axes[1], ann_st, ftgt_amp, max_mag=ftgt_max, punch_holes=(HOLE_B,))
    axes[1].set_title("flipped  (conjugation)   p_A = "
                      f"{fp_a.real:+.2f},  p_B = {fp_b.real:+.2f}\n"
                      f"meridian FLOORS:  r = {fann_v.residual:.2e}",
                      fontsize=13.5, fontweight="bold", linespacing=1.3,
                      color=(0.7, 0.0, 0.0))
    fig.suptitle("The period sign matters: SAME annulus (b_1 = 1), only circle B's "
                 "period flipped\nmatched realizes; the conjugation floors -- a "
                 "cohomological obstruction no filling can fix",
                 fontsize=15.5, fontweight="bold")
    fig.legend(handles=_form_legend(tgt_max), loc="lower center", ncol=5,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    paths["flipped"] = os.path.join(out_dir, "emergent_bulk_flipped.png")
    _save(fig, paths["flipped"], dpi)

    # --- 5. composite: tile the four panels. ---
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 16.0))
    _draw_complex(axes[0, 0], ann_st, tgt_amp, max_mag=tgt_max, draw_faces=False,
                  only_edges=E.CYCLE_A + E.CYCLE_B, label_holes=False)
    axes[0, 0].set_title("1. geo(psi_A) || geo(psi_B): two boundary meridians "
                         f"(p_A=p_B={p_a.real:+.2f})", fontsize=13,
                         fontweight="bold")
    _draw_complex(axes[0, 1], disk_st, tgt_amp, max_mag=tgt_max)
    axes[0, 1].set_title(f"2. disk (b_1=0): meridian FLOORS  r={disk_v.residual:.1e}",
                         fontsize=13, fontweight="bold")
    _draw_complex(axes[1, 0], ann_st, ann_amp, max_mag=ann_max,
                  punch_holes=(HOLE_B,))
    axes[1, 0].set_title("3. annulus (b_1=1): meridian REALIZES  "
                         f"r={ann_v.residual:.1e}", fontsize=13, fontweight="bold")
    _draw_complex(axes[1, 1], grow_st, grow_amp, max_mag=grow_max,
                  punch_holes=(HOLE_B,))
    axes[1, 1].set_title(f"4. surgery-grown bulk (b_1: 0->{b1[2]})  "
                         f"r={grow_v.residual:.1e}", fontsize=13,
                         fontweight="bold")
    fig.legend(handles=_form_legend(ann_max), loc="lower center", ncol=5,
               fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Emergent-bulk realizability at k=1: surgery makes b_1 an OUTPUT\n"
        "the matched two-boundary meridian realizes iff b_1=1 "
        "(disk floors, annulus realizes; surgery opens the handle on its own)",
        fontsize=16.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    paths["composite"] = os.path.join(out_dir, "emergent_bulk_composite.png")
    _save(fig, paths["composite"], dpi)

    return paths, dict(disk=disk_v, annulus=ann_v, grow=grow_v, b1=b1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="output directory for the PNGs (default /tmp/cobordism, "
                         "not committed).")
    ap.add_argument("--layout-seed", type=int, default=42,
                    help="seed for the surgery search (default 42).")
    ap.add_argument("--dpi", type=int, default=130, help="PNG dpi (default 130).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Emergent-bulk visualization -- the disk/annulus/surgery panels\n")
    paths, verds = build(args.out, dpi=args.dpi, layout_seed=args.layout_seed)
    b1d, b1a, b1g = verds["b1"]
    print(f"  disk    b_1={b1d}  meridian r={verds['disk'].residual:.2e} (floored)")
    print(f"  annulus b_1={b1a}  meridian r={verds['annulus'].residual:.2e} "
          f"(realized, eig={verds['annulus'].eigenvalue:.1e})")
    print(f"  surgery b_1: 0->{b1g}  removals={verds['grow'].surgery_removals}  "
          f"r={verds['grow'].residual:.2e} (realized)")
    print()
    for tag in ("boundary", "disk", "annulus", "surgery", "flipped", "composite"):
        print(f"  {tag:10} -> {paths[tag]}")
    print("\n  (PNGs are PR artifacts -- upload to the issue-attachments release; "
          "not committed.)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
