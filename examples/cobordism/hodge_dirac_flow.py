# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Hodge-Dirac spectral flow over a finished laplacian_clusters dump run.

EXPERIMENT (untracked): matter/antimatter structure in the Hodge spectrum.

Assembles the supercharge Q = d + delta on the full cochain complex from the
engine's own boundary matrices (ChainComplex.boundaryMatrix) and metric weights
(HodgeLaplacian.weights), in the symmetrized basis B_k = W_{k-1}^{1/2} d_k
W_k^{-1/2}, so that Q^2 = (+)_k L_k reproduces the engine's metric Hodge
Laplacians exactly. Per frame of a dump run it then measures:

  * spec(Q): every nonzero Laplacian eigenvalue resolved into a +/-sqrt(lambda)
    pair -- the SUSY doublet (alpha, d alpha) recombined into the positive/
    negative-frequency ("particle/antiparticle") branches of the first-order
    operator. Zero modes = harmonics = self-conjugate (their own antiparticles).
  * the McKean-Singer supertrace  sum_k (-1)^k Tr e^{-t L_k} = chi  at every t
    (pair-annihilation of the whole nonzero spectrum, leaving the index), and
    eta(t) = sum sign(lambda) e^{-t lambda^2} of Q, which the parity grading
    forces to vanish IDENTICALLY -- exact matter/antimatter symmetry of the
    +/- spectrum, twist or no twist. Both verified numerically per frame.
  * spectral-flow events: frames where b_k jumps (a +/- pair of Q eigenvalues
    lands on / lifts off zero) = creation/annihilation of a topological zero
    mode (a register). lambda_min^+(L_3) is the continuous early-warning
    signal that descends toward zero as a register's boundary sphere
    assembles (compare: open 5-cliques precede certified holes).
  * the Lorentzian d'Alembertian spectra (engine lorentzianEigenvalues):
    complex-conjugate eigenvalue pairs = mode/anti-mode (positive/negative
    frequency) pairs; negative-real modes = timelike content.
  * the engine's own charge conjugation: per-cluster r_state of the proton
    singlet {1,w,w^2} vs its conjugate on the k=3 spectral clusters --
    alpha_c = r_conj(c) - r_singlet(c) is the matter(-3..+3)antimatter
    polarization of each cluster (baryon vs anti-baryon sector).

Usage (from the repo root):
    python examples/cobordism/hodge_dirac_flow.py laplacian_dumps/run-20260811-121826
    python examples/cobordism/hodge_dirac_flow.py laplacian_dumps/run-20260811-202435 \
        --gif-stride 2 --fps 12
Outputs (inside the run directory by default): qflow.gif, qflow_final.png,
and a per-frame table + verification summary on stdout.
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")  # two engine runs share this box

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laplacian_clusters as lc  # primal graph, clustering, clique census

import tessera

cob = tessera.cobordism

TGRID = np.logspace(-3, 2, 160)  # heat-kernel time grid for the supertrace
ZTOL_FLOOR = 1e-10               # absolute floor for "zero" eigenvalues


def load_dump(path):
    """Rehydrate one state dump (dump_state schema) into (Spacetime, meta)."""
    with open(path) as f:
        d = json.load(f)
    st = tessera.observables.LiveComplex.load(
        d["cells"],
        {(u, v): complex(re, im) for u, v, re, im in d["squared_lengths"]},
        {int(v): t for v, t in d["vertex_times"].items()},
        d["dimensions"])
    return st, d


def q_blocks(st):
    """Symmetrized Hodge-Dirac blocks B_k = W_{k-1}^{1/2} d_k W_k^{-1/2} from
    the engine's boundary matrices and Euclidean |volume| weights, plus the
    f-vector. Q^2 = (+)_k (B_k^T B_k + B_{k+1} B_{k+1}^T) = the engine's
    symmetric metric Hodge Laplacians (verified to ~1e-14 offline)."""
    cc = cob.ChainComplex.fromSpacetime(st)
    dim = cc.dimension()
    n = list(cc.fVector())
    H = cob.HodgeLaplacian(st)
    W = {k: np.asarray(H.weights(k), dtype=float) for k in range(dim + 1)}
    Bs = {}
    for k in range(1, dim + 1):
        bm = np.asarray(cc.boundaryMatrix(k), dtype=float).reshape(n[k - 1], n[k])
        Bs[k] = (np.sqrt(W[k - 1])[:, None] * bm) / np.sqrt(W[k])[None, :]
    return cc, H, dim, n, Bs


def up_spectra(Bs, n, dim):
    """Nonzero 'up' Gram spectra g[k] (the (k,k+1) doublet family; each entry
    lambda gives the Q pair +/-sqrt(lambda)), ranks, and Betti numbers."""
    g, rank = {}, {0: 0, dim + 1: 0}
    for k in range(dim):
        M = Bs[k + 1]
        G = M @ M.T if M.shape[0] <= M.shape[1] else M.T @ M
        ev = np.linalg.eigvalsh(G) if G.size else np.zeros(0)
        tol = max(ZTOL_FLOOR, (max(G.shape or [1]) * np.abs(ev).max() * 1e-12
                               if ev.size else 0.0))
        g[k] = np.sort(ev[ev > tol])
        rank[k + 1] = int(g[k].size)
    b = [n[k] - rank[k] - rank[k + 1] for k in range(dim + 1)]
    return g, b


def dense_q_check(Bs, n, dim, g, b):
    """Assemble Q = d + delta ONCE as a dense block matrix and diagonalize it
    raw: verifies that spec(Q) really is the +/-sqrt(g) pairing plus sum(b)
    zeros (the +/- symmetry is a property of the operator, not an artifact of
    the paired construction), and measures eta from the raw spectrum."""
    N = sum(n)
    off = np.cumsum([0] + list(n))
    Q = np.zeros((N, N))
    for k, M in Bs.items():  # M: C_k -> C_{k-1}, shape n[k-1] x n[k]
        Q[off[k - 1]:off[k], off[k]:off[k + 1]] = M
        Q[off[k]:off[k + 1], off[k - 1]:off[k]] = M.T
    ev = np.sort(np.linalg.eigvalsh(Q))
    model = np.sort(np.concatenate(
        [np.concatenate([np.sqrt(gk), -np.sqrt(gk)]) for gk in g.values()]
        + [np.zeros(sum(b))]))
    dev = np.abs(ev - model).max()
    sym = np.abs(ev + ev[::-1]).max()  # spec(Q) == -spec(Q)
    tol = Q.shape[0] * np.abs(ev).max() * 1e-12  # eta is over the NONZERO spectrum
    nz = ev[np.abs(ev) > tol]
    eta_raw = np.abs(np.sign(nz) @ np.exp(-np.outer(TGRID, nz ** 2).T)).max()
    return dev, sym, eta_raw


def laplacian_eigs(g, b, k, dim):
    """spec(L_k) reconstructed from the doublet families:
    g[k-1] u g[k] u 0^{b_k} (down + up + kernel)."""
    parts = [np.zeros(b[k])]
    if k - 1 in g:
        parts.append(g[k - 1])
    if k in g and k < dim:
        parts.append(g[k])
    return np.concatenate(parts)


def analyze_frame(path, k_clusters, seed):
    """All measurements for one dump frame."""
    st, meta = load_dump(path)
    cc, H, dim, n, Bs = q_blocks(st)
    g, b = up_spectra(Bs, n, dim)
    betti_engine = list(cc.bettiNumbers())
    chi = int(cc.eulerCharacteristic())

    # --- verification: supertrace == chi for all t; eta(Q) == 0 -----------
    supertrace = np.zeros_like(TGRID)
    for k in range(dim + 1):
        lk = laplacian_eigs(g, b, k, dim)
        supertrace += (-1) ** k * np.exp(-np.outer(TGRID, lk)).sum(axis=1)
    qs_nz = np.concatenate([np.sqrt(gk) for gk in g.values()] or [np.zeros(0)])
    qs_signed = np.concatenate([qs_nz, -qs_nz])
    eta = np.sign(qs_signed) @ np.exp(-np.outer(TGRID, qs_signed ** 2).T)

    # --- Lorentzian d'Alembertian spectra (signed |volume| weights) -------
    lor = {k: np.asarray(H.lorentzianEigenvalues(k, True), dtype=complex)
           for k in range(1, dim + 1)}
    lor_complex = sum(int((np.abs(v.imag) > 1e-9).sum()) for v in lor.values())
    lor_negreal = sum(int(((np.abs(v.imag) <= 1e-9) & (v.real < -1e-9)).sum())
                      for v in lor.values())

    # --- primal-graph clustering + charge-conjugation polarization --------
    vids, idx, A, _W = lc.primal_graph(st)
    evals, evecs = lc.laplacian_spectrum(A)
    labels, q_mod = lc.cluster_fixed_k(A, evecs, k_clusters, seed)
    rstates = lc.cluster_rstates(st, vids, labels, k_clusters, degree=3)
    alpha = sorted((d["conjugate"] - d["singlet"] for d in rstates),
                   reverse=True)  # +3 = pure matter (baryon) sector cluster
    singlet = cob.Proton.singlet()
    conj = [complex(z).conjugate() for z in singlet]
    r_s = float(cob.MultiCobordism.r_state(st, 3, singlet))
    r_c = float(cob.MultiCobordism.r_state(st, 3, conj))

    filled, open_cliques, holes = lc.clique_census(st, vids, idx, A)

    lmin3 = min([gk.min() for kk, gk in g.items() if kk in (2, 3) and gk.size]
                or [np.nan])
    im_max = max((abs(im) for _u, _v, _re, im in meta["squared_lengths"]),
                 default=0.0)  # 0 => C-symmetric (self-conjugate) background
    return {
        "meta": meta, "n": n, "chi": chi, "b": b, "betti_engine": betti_engine,
        "g": g, "dim": dim,
        "supertrace_dev": float(np.abs(supertrace - chi).max()),
        "eta_max": float(np.abs(eta).max()),
        "lor": lor, "lor_complex": lor_complex, "lor_negreal": lor_negreal,
        "alpha": alpha, "q_mod": q_mod, "r_singlet": r_s, "r_conjugate": r_c,
        "rs_min": min(d["singlet"] for d in rstates),
        "rc_min": min(d["conjugate"] for d in rstates),
        "cliques_filled": filled, "cliques_open": open_cliques, "holes": holes,
        "lmin3": float(lmin3), "F": meta.get("F", float("nan")),
        "im_max": im_max,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

FAMILY_COLORS = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868", 3: "#c44e52"}
FAMILY_NAMES = {0: "(0,1) vertex-edge", 1: "(1,2) edge-triangle",
                2: "(2,3) triangle-tet", 3: "(3,4) tet-pentachoron"}


def build_animation(frames, tags, out_gif, out_png, fps, gif_stride, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    nF = len(frames)
    xs = np.arange(nF)

    # flow-panel y window: show at least the ~12 lowest modes everywhere
    y_star = 0.0
    for fr in frames:
        allq = np.sort(np.concatenate([np.sqrt(gk) for gk in fr["g"].values()]))
        if allq.size:
            y_star = max(y_star, allq[min(11, allq.size - 1)])
    y_star *= 1.15

    # batched flow-panel arrays (one scatter per family per paint, not O(nF^2))
    flow = {}
    for fam in range(4):
        fx, fy = [], []
        for j, fr in enumerate(frames):
            gk = fr["g"].get(fam, np.zeros(0))
            q = np.sqrt(gk[gk <= y_star ** 2])
            fx.extend([j] * (2 * q.size))
            fy.extend(np.concatenate([q, -q]).tolist())
        flow[fam] = (np.array(fx), np.array(fy))
    zero_sizes = np.array([16 + 30 * sum(fr["b"]) for fr in frames])

    # Lorentzian panel extent (global, so the plane doesn't jump)
    all_lor = np.concatenate([v for fr in frames for v in fr["lor"].values()])
    re_lo, re_hi = np.percentile(all_lor.real, [2, 98])
    im_hi = np.percentile(np.abs(all_lor.imag), 99) * 1.2 + 1e-6
    pad = 0.08 * (re_hi - re_lo)

    hole_events = [i for i in range(1, nF)
                   if frames[i]["b"][3] > frames[i - 1]["b"][3]]
    kill_events = [i for i in range(1, nF)
                   if frames[i]["b"][3] < frames[i - 1]["b"][3]]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10))
    (ax_flow, ax_lor), (ax_can, ax_tr) = axes
    fig.subplots_adjust(hspace=0.34, wspace=0.24, top=0.90)

    def paint(i):
        fr = frames[i]
        for ax in (ax_flow, ax_lor, ax_can, ax_tr):
            ax.clear()

        # --- P1: spectral flow of Q ------------------------------------
        for fam, (fx, fy) in flow.items():
            past = fx <= i
            ax_flow.scatter(fx[past], fy[past], s=6, color=FAMILY_COLORS[fam],
                            alpha=0.5, linewidths=0)
            curm = fx == i
            ax_flow.scatter(fx[curm], fy[curm], s=18, color=FAMILY_COLORS[fam],
                            alpha=1.0, linewidths=0)
        ax_flow.scatter(np.arange(i + 1), np.zeros(i + 1),
                        s=zero_sizes[:i + 1], color="black", zorder=3,
                        linewidths=0)
        for e in hole_events:
            ax_flow.axvline(e, color="#c44e52", ls="--", lw=1.1, alpha=0.8)
        for e in kill_events:
            ax_flow.axvline(e, color="#4c72b0", ls=":", lw=1.1, alpha=0.8)
        ax_flow.axhline(0, color="0.5", lw=0.6)
        ax_flow.set_xlim(-1, nF)
        ax_flow.set_ylim(-y_star, y_star)
        ax_flow.set_xlabel("frame")
        ax_flow.set_ylabel(r"spec(Q),  Q = d + $\delta$")
        b = fr["b"]
        ax_flow.set_title(
            f"Q spectral flow — every mode a ±pair (particle/antiparticle "
            f"branches);\nzero modes (black) self-conjugate: "
            f"b = {b}, χ = {fr['chi']}   [red dash = hole born, "
            f"blue dot = hole filled]", fontsize=10)
        ax_flow.legend(handles=[
            Line2D([], [], marker="o", ls="", color=FAMILY_COLORS[k],
                   label=FAMILY_NAMES[k], markersize=5) for k in range(4)]
            + [Line2D([], [], marker="o", ls="", color="black",
                      label="zero modes (harmonics)", markersize=6)],
            fontsize=7, loc="upper right", ncol=2)

        # --- P2: Lorentzian d'Alembertian spectra in the complex plane --
        for k, v in fr["lor"].items():
            ax_lor.scatter(v.real, v.imag, s=22, alpha=0.75,
                           color=FAMILY_COLORS[k - 1], label=f"$L_{k}$ (n={v.size})")
        ax_lor.axhline(0, color="0.6", lw=0.7)
        ax_lor.axvline(0, color="0.6", lw=0.7)
        ax_lor.set_xlim(re_lo - pad, re_hi + pad)
        ax_lor.set_ylim(-im_hi, im_hi)
        ax_lor.set_xlabel(r"Re $\lambda$")
        ax_lor.set_ylabel(r"Im $\lambda$")
        ax_lor.set_title(
            f"Lorentzian d'Alembertian spectrum (signed volumes) — conjugate "
            f"pairs = mode/anti-mode;\n{fr['lor_complex']} complex modes, "
            f"{fr['lor_negreal']} negative-real (timelike) modes", fontsize=10)
        ax_lor.legend(fontsize=7, loc="upper right")

        # --- P3: pair-annihilation of the supertrace + eta == 0 ---------
        total = np.zeros_like(TGRID)
        for k in range(fr["dim"] + 1):
            lk = laplacian_eigs(fr["g"], fr["b"], k, fr["dim"])
            tr = (-1) ** k * np.exp(-np.outer(TGRID, lk)).sum(axis=1)
            total += tr
            ax_can.plot(TGRID, tr, lw=1.0, alpha=0.75,
                        label=rf"$(-1)^{k}\,\mathrm{{Tr}}\,e^{{-tL_{k}}}$")
        ax_can.plot(TGRID, total, color="black", lw=2.4,
                    label=r"supertrace $=\chi$")
        ax_can.axhline(fr["chi"], color="0.4", lw=0.6, ls=":")
        ax_can.axhline(0, color="#8172b3", lw=1.6, ls="--",
                       label=r"$\eta(Q)\equiv 0$ (C-symmetry)")
        ax_can.set_xscale("log")
        ax_can.set_xlabel("heat-kernel time  t")
        ax_can.set_title(
            f"McKean–Singer pair-annihilation: towers cancel at every t, "
            f"residue = χ = {fr['chi']}\n"
            f"max|supertrace−χ| = {fr['supertrace_dev']:.1e},   "
            f"max|η(Q)| = {fr['eta_max']:.1e}", fontsize=10)
        ax_can.legend(fontsize=7, loc="upper right")

        # --- P4: traces — early warning, census, C-polarization ---------
        cur = slice(0, i + 1)
        ax_tr.plot(xs[cur], [frames[j]["lmin3"] for j in range(i + 1)],
                   color="#55a868", lw=1.6,
                   label=r"$\lambda^+_{\min}(L_3)$ (register early-warning)")
        ax_tr.plot(xs[cur], [frames[j]["b"][3] for j in range(i + 1)],
                   color="#c44e52", lw=1.8, drawstyle="steps-post",
                   label=r"$b_3$ (certified holes = zero modes)")
        oc = [frames[j]["cliques_open"] for j in range(i + 1)]
        ax_tr.plot(xs[cur], oc, color="#dd8452", lw=1.1, ls="--",
                   label="open 5-cliques (nascent registers)")
        ax_tr.set_xlim(-1, nF)
        ax_tr.set_xlabel("frame")
        ax_tr.set_ylabel("eigenvalue / count")
        ax_tr2 = ax_tr.twinx()
        alph = np.array([frames[j]["alpha"] for j in range(i + 1)])
        ax_tr2.plot(xs[cur], [frames[j]["rs_min"] for j in range(i + 1)],
                    color="#8172b3", ls="-", lw=1.3, alpha=0.9,
                    label=r"min$_c$ r$_{singlet}$ (0 = a cluster carries matter)")
        ax_tr2.plot(xs[cur], [frames[j]["rc_min"] for j in range(i + 1)],
                    color="#8172b3", ls="--", lw=1.3, alpha=0.9,
                    label=r"min$_c$ r$_{conj}$ (antimatter carry)")
        ax_tr2.plot(xs[cur], alph[:, 0], color="#937860", ls=":", lw=1.4,
                    label=r"max$_c$ α (C-polarization)")
        ax_tr2.set_ylim(-3.3, 3.3)
        ax_tr2.set_ylabel("carry (0 = full, 3 = none) / polarization α",
                          color="#8172b3")
        ax_tr2.tick_params(axis="y", labelcolor="#8172b3")
        if all(frames[j]["im_max"] == 0.0 for j in range(len(frames))):
            ax_tr2.annotate("Im ℓ² ≡ 0: C-symmetric background ⇒ α ≡ 0 exactly\n"
                            "(self-conjugate point — matter/antimatter degenerate)",
                            xy=(0.98, 0.05), xycoords="axes fraction",
                            ha="right", fontsize=7.5, color="#8172b3",
                            style="italic")
        for e in hole_events:
            ax_tr.axvline(e, color="#c44e52", ls="--", lw=1.0, alpha=0.7)
        for e in kill_events:
            ax_tr.axvline(e, color="#4c72b0", ls=":", lw=1.0, alpha=0.7)
        ax_tr.axvline(i, color="0.75", lw=0.8)
        h1, l1 = ax_tr.get_legend_handles_labels()
        h2, l2 = ax_tr2.get_legend_handles_labels()
        ax_tr.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax_tr.set_title("creation events vs early-warning vs charge polarization",
                        fontsize=10)

        fig.suptitle(
            f"Hodge–Dirac flow — {run_name} — {tags[i]}  "
            f"(cells {fr['n']}, χ={fr['chi']}, F={fr['F']:.2f})\n"
            f"Q = d + δ on ⊕ₖCᵏ from engine ∂ₖ and |vol| weights; "
            f"spec(Q) = ±√spec(⊕Lₖ);  zero modes ≅ Hₖ",
            fontsize=12)
        return []

    # final-frame PNG first (also serves as a render test)
    paint(nF - 1)
    fig.savefig(out_png, dpi=130)
    print(f"final summary -> {out_png}")

    frame_ids = list(range(0, nF, gif_stride))
    if frame_ids[-1] != nF - 1:
        frame_ids.append(nF - 1)
    t0 = time.time()
    anim = FuncAnimation(fig, paint, frames=frame_ids, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    print(f"animation ({len(frame_ids)} frames) -> {out_gif} "
          f"({time.time()-t0:.0f}s)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a laplacian_dumps/run-* directory")
    ap.add_argument("--k", type=int, default=3,
                    help="fixed spectral-cluster count for the C-polarization")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--gif-stride", type=int, default=1, dest="gif_stride")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "iter_*.json")))
    pre = os.path.join(args.run_dir, "pre.json")
    if os.path.exists(pre):
        paths.insert(0, pre)
    if not paths:
        raise SystemExit(f"no dumps in {args.run_dir}")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    out_gif = args.gif or os.path.join(args.run_dir, "qflow.gif")
    out_png = args.png or os.path.join(args.run_dir, "qflow_final.png")

    # honest one-frame check: diagonalize the dense Q itself
    st0, _ = load_dump(paths[0])
    _cc, _H, dim0, n0, Bs0 = q_blocks(st0)
    g0, b0 = up_spectra(Bs0, n0, dim0)
    dev, sym, eta_raw = dense_q_check(Bs0, n0, dim0, g0, b0)
    print(f"dense-Q check on {os.path.basename(paths[0])}: "
          f"max|spec(Q) - (+/-sqrt g, 0^b)| = {dev:.2e}, "
          f"max|spec + spec_reversed| = {sym:.2e} (+/- symmetry), "
          f"max|eta_raw| = {eta_raw:.2e}")

    frames, tags = [], []
    mismatches = 0
    t0 = time.time()
    for p in paths:
        fr = analyze_frame(p, args.k, args.seed)
        if fr["b"] != fr["betti_engine"]:
            mismatches += 1
            print(f"\n  WARNING b_k mismatch at {p}: mine={fr['b']} "
                  f"engine={fr['betti_engine']}")
        frames.append(fr)
        tags.append(os.path.splitext(os.path.basename(p))[0])
        print(f"\ranalyzed {len(frames)}/{len(paths)} "
              f"({time.time()-t0:.0f}s)", end="", flush=True)
    print()

    # ---- per-frame table -------------------------------------------------
    print(f"\n{'frame':>10} {'n_k':>24} {'b_k':>13} {'open5':>5} {'b3':>3} "
          f"{'lmin3+':>9} {'lorC':>5} {'lorNeg':>6} {'alpha(sorted)':>20}")
    for tag, fr in zip(tags, frames):
        print(f"{tag:>10} {str(fr['n']):>24} {str(fr['b']):>13} "
              f"{fr['cliques_open']:>5} {fr['b'][3]:>3} {fr['lmin3']:>9.4f} "
              f"{fr['lor_complex']:>5} {fr['lor_negreal']:>6} "
              f"{'[' + ', '.join(f'{a:+.2f}' for a in fr['alpha']) + ']':>20}")

    st_dev = max(fr["supertrace_dev"] for fr in frames)
    eta_max = max(fr["eta_max"] for fr in frames)
    chi_ok = all(sum((-1) ** k * x for k, x in enumerate(fr["b"])) == fr["chi"]
                 for fr in frames)
    print(f"\nVERIFICATION over {len(frames)} frames:")
    print(f"  b_k (rank bookkeeping) vs engine bettiNumbers: "
          f"{len(frames)-mismatches}/{len(frames)} exact")
    print(f"  sum (-1)^k b_k == chi(cells) on every frame: {chi_ok}")
    print(f"  max |supertrace(t) - chi| over all frames, all t: {st_dev:.2e}")
    print(f"  max |eta(Q)(t)|  (structural C-symmetry of spec Q): {eta_max:.2e}")
    im_run = max(fr["im_max"] for fr in frames)
    print(f"  max |Im l^2| over run: {im_run:.3g}"
          + ("  -> C-SYMMETRIC background (alpha == 0 is exact)"
             if im_run == 0 else "  -> C-breaking phases present"))
    hole_events = [i for i in range(1, len(frames))
                   if frames[i]["b"][3] > frames[i - 1]["b"][3]]
    print(f"  hole-birth (pair-landing) frames: "
          f"{[tags[i] for i in hole_events] or 'none'}")

    build_animation(frames, tags, out_gif, out_png, args.fps, args.gif_stride,
                    run_name)


if __name__ == "__main__":
    main()
