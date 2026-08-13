# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Spacetime-lattice view of a dump run: the complex itself, in its own metric.

EXPERIMENT (untracked). What the spectral panels abstract away — the lattice,
laid out by its OWN geometry, with the register forming in place.

Layout: classical MDS on shortest-path distances built from the edge proper
lengths |l| = sqrt(|l^2|), Procrustes-aligned frame to frame so the animation
does not tumble. (Vertex times are identically 0 in this pipeline — causality
lives in the edge DISPOSITIONS, Re l^2 < 0 = timelike, drawn dotted — so a
"time axis" would be a degenerate line; the metric layout is the honest
picture.)

Panels
  A  the lattice: dispositions, the certified register's boundary 3-sphere
     (red), and open 5-clique cores (orange) — nascent registers.
  B  the degree-3 mode: the soft mode of lambda_min^+(L_3) before birth
     (watch it concentrate on the would-be hole), the harmonic zero mode
     after — the register itself, drawn where it lives.
  C  the lightest 1-form (vector) mode — "the photon-shaped excitation".
     b_1 = 0 on every frame and every U(1) phase is 0, so there is NO
     massless photon in this data; its mass gap lambda_min(L_1) is printed.
  D  matter/antimatter seeding: per-vertex (quark-block count - antiquark-
     block count) from the six input blocks, RECONSTRUCTED by rebuilding the
     Proton with this run's parameters and verified cell-for-cell against
     pre.json. The blocks overlap almost completely — which is itself the
     answer to "can I see a proton and an antiproton": they are not disjoint
     lumps, they are six overlapping target-tagged regions of one complex.
  E  traces: lambda_min^+(L_3), lambda_min(L_1), b_3, open cores.
  F  churn: how much actually moves — topology changes (cell-set edits) vs
     geometry (mean |delta l^2| per frame).

Usage (repo root):
    python examples/cobordism/lattice_view.py laplacian_dumps/run-20260811-121826
    python examples/cobordism/lattice_view.py laplacian_dumps/run-20260811-152429 \
        --precone 12 --seed 3
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laplacian_clusters as lc

import tessera

cob = tessera.cobordism


def load_dump(path):
    with open(path) as f:
        d = json.load(f)
    st = tessera.observables.LiveComplex.load(
        d["cells"],
        {(u, v): complex(re, im) for u, v, re, im in d["squared_lengths"]},
        {int(v): t for v, t in d["vertex_times"].items()},
        d["dimensions"])
    return st, d


def metric_layout(vids, edge_len, prev):
    """Classical MDS on shortest-path |l| distances, Procrustes-aligned to the
    previous frame's positions (rotation/reflection only — MDS axes are
    arbitrary, so without this the picture tumbles frame to frame)."""
    n = len(vids)
    idx = {v: i for i, v in enumerate(vids)}
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for (a, b), w in edge_len.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            D[i, j] = D[j, i] = min(D[i, j], w)
    for k in range(n):  # Floyd-Warshall (n <= ~25 here)
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = finite.max() * 1.5 if finite.size else 1.0
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:2]
    X = V[:, order] * np.sqrt(np.maximum(w[order], 0))

    if prev:
        shared = [(i, prev[v]) for i, v in enumerate(vids) if v in prev]
        if len(shared) >= 3:
            A = np.array([X[i] for i, _ in shared])
            Bp = np.array([p for _, p in shared])
            Ac, Bc = A - A.mean(0), Bp - Bp.mean(0)
            U, _s, Vt = np.linalg.svd(Ac.T @ Bc)
            R = U @ Vt
            X = (X - A.mean(0)) @ R + Bp.mean(0)
    return {v: X[i] for i, v in enumerate(vids)}


def reconstruct_blocks(pre_path, seed, precone, timelike, alternate):
    """Rebuild the Proton's six input blocks and VERIFY cell-for-cell against
    pre.json. Returns (quark_sets, antiquark_sets) or None if it doesn't match
    (wrong parameters for this run -> the panel is skipped rather than lying)."""
    try:
        with open(pre_path) as f:
            want = {tuple(sorted(c)) for c in json.load(f)["cells"]}
        p = cob.Proton(seed=seed, precone=precone, precone_timelike=timelike,
                       precone_alternate=alternate)
        node = p.direct_node(seed)
        got = {tuple(sorted(v.getId() for v in c.getVertices()))
               for c in node.st.getTopSimplices()}
        if got != want:
            return None
        blocks = [set(b.vertices) for b in node.inputs]
        return blocks[:3], blocks[3:]
    except Exception as exc:
        print(f"  (block reconstruction skipped: {exc!r})")
        return None


def analyze(path, prev_pos):
    st, d = load_dump(path)
    vids = sorted({v for c in d["cells"] for v in c})
    edge_len, l2 = {}, {}
    timelike = {}
    for u, v, re_, im_ in d["squared_lengths"]:
        key = (min(u, v), max(u, v))
        edge_len[key] = float(np.sqrt(abs(re_))) or 1e-6
        l2[key] = re_
        timelike[key] = re_ < 0
    pos = metric_layout(vids, edge_len, prev_pos)

    cc = cob.ChainComplex.fromSpacetime(st)
    H = cob.HodgeLaplacian(st)
    b = list(cc.bettiNumbers())
    edges1 = [tuple(e) for e in cc.kSimplexVertices(1)]
    tets = [tuple(t) for t in cc.kSimplexVertices(3)]
    eidx = {e: i for i, e in enumerate(edges1)}

    ev3 = np.asarray(H.eigenvalues(3), dtype=float)
    V3 = np.asarray(H.eigenvectors(3)).reshape(len(tets), len(tets))
    col3 = 0 if b[3] > 0 else b[3]
    lam3 = float(ev3[b[3]]) if b[3] < len(ev3) else float("nan")
    psi3 = np.abs(V3[:, col3])
    amp3 = np.zeros(len(edges1))
    for ti, tet in enumerate(tets):
        for pair in itertools.combinations(tet, 2):
            amp3[eidx[pair]] += psi3[ti]
    amp3 /= max(amp3.max(), 1e-12)

    ev1 = np.asarray(H.eigenvalues(1), dtype=float)
    V1 = np.asarray(H.eigenvectors(1)).reshape(len(edges1), len(edges1))
    lam1 = float(ev1[0]) if ev1.size else float("nan")
    amp1 = np.abs(V1[:, 0])
    amp1 /= max(amp1.max(), 1e-12)

    cells = {tuple(sorted(c)) for c in d["cells"]}
    _v, _i, A, _W = lc.primal_graph(st)
    opens = {tuple(sorted(vids[i] for i in cl)) for cl in lc.five_cliques(A)}
    opens -= cells
    holes = [tuple(sorted(h)) for h in cob.MultiCobordism.emergent_holes(st, 3)]
    return {
        "pos": pos, "edges": edges1, "timelike": timelike, "l2": l2,
        "amp3": amp3, "lam3": lam3, "born": b[3] > 0,
        "amp1": amp1, "lam1": lam1, "b3": b[3],
        "holes": holes, "opens": opens, "cells": cells,
        "F": d.get("F", float("nan")), "vids": vids,
    }


def clique_edges(tuples5):
    return {tuple(sorted(p)) for t in tuples5
            for p in itertools.combinations(t, 2)}


def event_closeup(frames, tags, i, out_png, run_name):
    """Three-panel before/at/after close-up of one event frame: what the move
    removed (edges/cells) and which nascent cores died with it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    lo, hi = max(i - 1, 0), min(i + 1, len(frames) - 1)
    show = [lo, i, hi]
    gone_e = set(frames[lo]["edges"]) - set(frames[i]["edges"])
    gone_cells = frames[lo]["cells"] - frames[i]["cells"]
    new_cells = frames[i]["cells"] - frames[lo]["cells"]
    dying = frames[lo]["opens"] - frames[i]["opens"]
    born = frames[i]["opens"] - frames[lo]["opens"]

    allx = np.array([p for j in show for p in frames[j]["pos"].values()])
    pad = 0.12 * (allx.max(0) - allx.min(0) + 1e-9)

    span = allx.max(0) - allx.min(0) + 2 * pad
    fig, axes = plt.subplots(1, 3,
                             figsize=(17, 2.2 + 5.0 * span[1] / max(span[0], 1e-9)))
    fig.subplots_adjust(top=0.82, bottom=0.03, wspace=0.06)
    for ax, j in zip(axes, show):
        fr = frames[j]
        dying_e = clique_edges(dying) if j == lo else set()
        base = [e for e in fr["edges"] if e not in dying_e and e not in gone_e]
        sl = [[fr["pos"][a], fr["pos"][b]] for a, b in base
              if not fr["timelike"].get((a, b))]
        tl = [[fr["pos"][a], fr["pos"][b]] for a, b in base
              if fr["timelike"].get((a, b))]
        ax.add_collection(LineCollection(sl, colors="#c2c8d2", linewidths=0.9))
        ax.add_collection(LineCollection(tl, colors="#c2c8d2", linewidths=0.9,
                                         linestyles=":"))
        if j == lo:
            ax.add_collection(LineCollection(
                [[fr["pos"][a], fr["pos"][b]] for a, b in sorted(dying_e)
                 if a in fr["pos"] and b in fr["pos"]],
                colors="#dd8452", linewidths=2.2, alpha=0.95))
        killer = [[fr["pos"][a], fr["pos"][b]] for a, b in sorted(gone_e)
                  if a in fr["pos"] and b in fr["pos"]]
        if killer and j == lo:
            ax.add_collection(LineCollection(killer, colors="#c44e52",
                                             linewidths=3.4, zorder=4))
        P = np.array(list(fr["pos"].values()))
        ax.scatter(P[:, 0], P[:, 1], s=34, color="#1c2129", zorder=5)
        ax.set_xlim(allx[:, 0].min() - pad[0], allx[:, 0].max() + pad[0])
        ax.set_ylim(allx[:, 1].min() - pad[1], allx[:, 1].max() + pad[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        role = ("BEFORE" if j == lo else "AFTER" if j == i else "settled")
        ax.set_title(f"{role} — {tags[j]}   ({len(fr['cells'])} cells, "
                     f"{len(fr['opens'])} open cores, F={fr['F']:.3f})",
                     fontsize=10)
    axes[0].legend(handles=[
        Line2D([], [], color="#c44e52", lw=3.4, label="edge the move deletes"),
        Line2D([], [], color="#dd8452", lw=2.2, label="cores that die with it")],
        fontsize=8, loc="upper left")
    fig.suptitle(
        f"annihilation close-up — {run_name}, {tags[i]}\n"
        f"one edge deletion {sorted(gone_e)} kills {len(dying)} nascent "
        f"register core(s) at once   ·   cells −{sorted(gone_cells)} "
        f"+{sorted(new_cells)}"
        + (f"   ·   {len(born)} core(s) born" if born else ""),
        fontsize=12)
    fig.savefig(out_png, dpi=130)
    print(f"event close-up -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--precone", type=int, default=12)
    ap.add_argument("--precone-timelike", dest="tl", action="store_true",
                    default=True)
    ap.add_argument("--precone-alternate", dest="alt", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--events-only", action="store_true",
                    help="render only the event close-ups, no animation")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "iter_*.json")))
    pre = os.path.join(args.run_dir, "pre.json")
    if os.path.exists(pre):
        paths.insert(0, pre)
    if not paths:
        raise SystemExit(f"no dumps in {args.run_dir}")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    out_gif = os.path.join(args.run_dir, "lattice.gif")
    out_png = os.path.join(args.run_dir, "lattice_final.png")

    blocks = reconstruct_blocks(pre, args.seed, args.precone, args.tl, args.alt)
    print(f"input blocks: "
          + ("reconstructed and verified against pre.json" if blocks
             else "NOT reconstructible with these parameters (panel D skipped)"))

    frames, tags, prev_pos = [], [], {}
    t0 = time.time()
    for p in paths:
        fr = analyze(p, prev_pos)
        prev_pos = fr["pos"]
        frames.append(fr)
        tags.append(os.path.splitext(os.path.basename(p))[0])
        print(f"\r{len(frames)}/{len(paths)} ({time.time()-t0:.0f}s)",
              end="", flush=True)
    print()

    # event ledger
    births = [i for i in range(1, len(frames))
              if frames[i]["b3"] > frames[i - 1]["b3"]]
    deaths = [i for i in range(1, len(frames))
              if frames[i]["b3"] < frames[i - 1]["b3"]]
    core_born = [i for i in range(1, len(frames))
                 if frames[i]["opens"] - frames[i - 1]["opens"]]
    core_died = [i for i in range(1, len(frames))
                 if frames[i - 1]["opens"] - frames[i]["opens"]]
    topo = [i for i in range(1, len(frames))
            if frames[i]["cells"] != frames[i - 1]["cells"]]
    geo = [0.0] + [
        float(np.mean([abs(frames[i]["l2"][e] - frames[i - 1]["l2"].get(e, 0.0))
                       for e in frames[i]["l2"]]))
        for i in range(1, len(frames))]
    print(f"events: register births {[tags[i] for i in births] or 'none'}; "
          f"register DEATHS (annihilation) {[tags[i] for i in deaths] or 'none'}")
    print(f"        core births {[tags[i] for i in core_born] or 'none'}; "
          f"core deaths {[tags[i] for i in core_died] or 'none'}")
    print(f"        topology-changing frames: {len(topo)}/{len(frames)-1}; "
          f"max mean|Δℓ²| {max(geo):.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    allx = np.array([p for fr in frames for p in fr["pos"].values()])
    pad = 0.12 * (allx.max(0) - allx.min(0) + 1e-9)
    xlim = (allx[:, 0].min() - pad[0], allx[:, 0].max() + pad[0])
    ylim = (allx[:, 1].min() - pad[1], allx[:, 1].max() + pad[1])

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.5))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes
    fig.subplots_adjust(hspace=0.28, wspace=0.16, top=0.88)

    def frame_axes(ax, title):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)

    def segs_for(fr, keys):
        return [[fr["pos"][a], fr["pos"][b]] for a, b in keys
                if a in fr["pos"] and b in fr["pos"]]

    def paint(i):
        fr = frames[i]
        for ax in axes.ravel():
            ax.clear()
        P = np.array([fr["pos"][v] for v in fr["vids"]])

        # --- A: the lattice ------------------------------------------------
        hole_e = clique_edges(fr["holes"])
        open_e = clique_edges(fr["opens"]) - hole_e
        sl, tl = [], []
        for e in fr["edges"]:
            if e in hole_e or e in open_e:
                continue
            (tl if fr["timelike"].get(e) else sl).append(
                [fr["pos"][e[0]], fr["pos"][e[1]]])
        ax_a.add_collection(LineCollection(sl, colors="#7d8697", linewidths=0.9,
                                           alpha=0.75))
        ax_a.add_collection(LineCollection(tl, colors="#7d8697", linewidths=1.0,
                                           alpha=0.9, linestyles=":"))
        ax_a.add_collection(LineCollection(segs_for(fr, sorted(open_e)),
                                           colors="#dd8452", linewidths=1.8,
                                           alpha=0.9))
        ax_a.add_collection(LineCollection(segs_for(fr, sorted(hole_e)),
                                           colors="#c44e52", linewidths=3.0,
                                           alpha=0.95))
        ax_a.scatter(P[:, 0], P[:, 1], s=30, color="#1c2129", zorder=3)
        frame_axes(ax_a, f"A · the lattice in its own metric — {len(fr['cells'])} "
                         f"cells, {len(fr['vids'])} vertices, b₃={fr['b3']}\n"
                         f"solid = spacelike, dotted = timelike (causality is "
                         f"in the dispositions, not a time axis)")
        ax_a.legend(handles=[
            Line2D([], [], color="#c44e52", lw=3, label="certified register ∂≅S³"),
            Line2D([], [], color="#dd8452", lw=1.8, label="open 5-clique core"),
            Line2D([], [], color="#7d8697", lw=1, ls=":", label="timelike edge")],
            fontsize=7, loc="upper left")

        # --- B: degree-3 mode ----------------------------------------------
        keys = [e for e in fr["edges"]]
        a3 = np.array([fr["amp3"][j] for j, _e in enumerate(keys)])
        ax_b.add_collection(LineCollection(segs_for(fr, keys), cmap="viridis",
                                           array=a3, linewidths=0.6 + 3.4 * a3,
                                           alpha=0.95))
        ax_b.scatter(P[:, 0], P[:, 1], s=14, color="#1c2129", zorder=3)
        frame_axes(ax_b, "B · degree-3 mode — "
                         + ("HARMONIC zero mode: the register itself"
                            if fr["born"] else
                            f"soft mode, λ⁺min(L₃) = {fr['lam3']:.3f}")
                         + "\n(bright/thick = where the mode lives)")

        # --- C: photon-shaped mode -----------------------------------------
        a1 = np.array([fr["amp1"][j] for j, _e in enumerate(keys)])
        ax_c.add_collection(LineCollection(segs_for(fr, keys), cmap="magma",
                                           array=a1, linewidths=0.6 + 3.4 * a1,
                                           alpha=0.95))
        ax_c.scatter(P[:, 0], P[:, 1], s=14, color="#1c2129", zorder=3)
        frame_axes(ax_c, f"C · lightest 1-form (vector) mode — mass "
                         f"λmin(L₁) = {fr['lam1']:.3f}\n"
                         f"b₁ = 0 and all U(1) phases = 0 ⇒ NO massless photon "
                         f"in this data")

        # --- D: matter/antimatter seeding ----------------------------------
        if blocks:
            q, qbar = blocks
            score = np.array([sum(v in s for s in q) - sum(v in s for s in qbar)
                              for v in fr["vids"]], dtype=float)
            ax_d.add_collection(LineCollection(
                segs_for(fr, keys), colors="#c9ced8", linewidths=0.7, alpha=0.8))
            sc = ax_d.scatter(P[:, 0], P[:, 1], c=score, cmap="coolwarm",
                              vmin=-3, vmax=3, s=95, edgecolors="#1c2129",
                              linewidths=0.5, zorder=3)
            cb = fig.colorbar(sc, ax=ax_d, fraction=0.04, pad=0.02)
            cb.set_label("quark blocks − antiquark blocks", fontsize=7.5)
            cb.ax.tick_params(labelsize=7)
            overlap = len(set.intersection(*(q + qbar)))
            frame_axes(ax_d, f"D · matter/antimatter seeding (as SEEDED; blocks "
                             f"are not re-dumped)\nall six blocks share "
                             f"{overlap} vertices — overlapping tagged regions, "
                             f"not separate lumps")
        else:
            frame_axes(ax_d, "D · matter/antimatter seeding — unavailable\n"
                             "(rebuild parameters don't match this run)")
            ax_d.text(0.5, 0.5, "pass --seed/--precone matching the run",
                      transform=ax_d.transAxes, ha="center", fontsize=9,
                      color="#7d8697")

        # --- E: traces -------------------------------------------------------
        cur = list(range(i + 1))
        ax_e.plot(cur, [frames[j]["lam3"] for j in cur], color="#55a868",
                  lw=1.7, label="λ⁺min(L₃) register early-warning")
        ax_e.plot(cur, [frames[j]["lam1"] for j in cur], color="#4c72b0",
                  lw=1.4, label="λmin(L₁) photon-mass gap")
        ax_e.plot(cur, [frames[j]["b3"] for j in cur], color="#c44e52", lw=2.0,
                  drawstyle="steps-post", label="b₃ certified registers")
        ax_e.plot(cur, [len(frames[j]["opens"]) for j in cur], color="#dd8452",
                  ls="--", lw=1.2, label="open 5-clique cores")
        for e in births:
            ax_e.axvline(e, color="#c44e52", ls="--", lw=1.0, alpha=0.7)
        for e in core_died:
            ax_e.axvline(e, color="#8172b3", ls=":", lw=1.2, alpha=0.8)
        ax_e.axvline(i, color="0.8", lw=0.8)
        ax_e.set_xlim(-1, len(frames))
        ax_e.set_xlabel("frame")
        ax_e.set_title(f"E · events — register births {len(births)}, "
                       f"register deaths {len(deaths)}, core deaths "
                       f"{len(core_died)}", fontsize=9.5)
        ax_e.legend(fontsize=7, loc="center right")

        # --- F: churn ---------------------------------------------------------
        ax_f.plot(range(len(frames)), geo, color="#937860", lw=1.0,
                  label="geometry: mean |Δℓ²| per frame")
        for t in topo:
            ax_f.axvline(t, color="#c44e52", lw=1.4, alpha=0.75)
        ax_f.axvline(i, color="0.8", lw=0.8)
        ax_f.set_yscale("symlog", linthresh=1e-6)
        ax_f.set_xlim(-1, len(frames))
        ax_f.set_xlabel("frame")
        ax_f.set_title(f"F · how much actually moves — {len(topo)} "
                       f"topology-changing frames of {len(frames)-1}\n"
                       f"(red lines); geometry relaxes continuously between them",
                       fontsize=9.5)
        ax_f.legend(fontsize=7, loc="upper right")

        fig.suptitle(
            f"lattice view — {run_name} — {tags[i]} — F={fr['F']:.2f}\n"
            f"no charge label on the register: C-symmetric background "
            f"(Im ℓ² ≡ 0) + C-blind r_state ⇒ register and anti-register are "
            f"degenerate here", fontsize=12)
        return []

    for i in sorted(set(core_died + deaths)):
        event_closeup(frames, tags, i,
                      os.path.join(args.run_dir, f"event_{tags[i]}.png"),
                      run_name)
    if args.events_only:
        return

    paint(len(frames) - 1)
    fig.savefig(out_png, dpi=125)
    print(f"final -> {out_png}")
    ids = list(range(0, len(frames), args.stride))
    if ids[-1] != len(frames) - 1:
        ids.append(len(frames) - 1)
    anim = FuncAnimation(fig, paint, frames=ids, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=args.fps))
    print(f"animation ({len(ids)} frames) -> {out_gif}")


if __name__ == "__main__":
    main()
