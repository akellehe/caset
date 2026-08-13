# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Primal-graph Laplacian cluster analysis of the one-step proton build.

Treats the simplicial complex ITSELF as the primal graph — one graph node per
complex vertex, with an adjacency link wherever two vertices share an edge — and
asks whether the wave function's states are visible as CLUSTERS of that graph:
pre-interaction, the six prepared q-q̄ input blocks; post-interaction, whatever
the relaxed whole carries (its emergent holes / registers).

For each snapshot (PRE = the seeded, preconed direct node before any drive;
POST = after a combined-`run` drive):

  * build the adjacency A, degree D, Laplacian L = D − A, and the symmetric
    normalized L_sym = I − D^{-1/2} A D^{-1/2};
  * eigendecompose L_sym: the zero modes count connected components, the Fiedler
    value measures how cuttable the complex is, and the leading eigengap profile
    suggests a natural cluster count;
  * spectral k-means over k = 2..K_MAX on the first nontrivial eigenvectors,
    keeping the k with the best Newman–Girvan modularity Q — scored by the
    ENGINE's own `SparseGraph.modularity`, so the number is the same Q the
    modularity observables report;
  * cross-tabulate the winning clusters against the six input blocks' vertex
    sets and against the emergent holes' vertex tuples.

`--live` watches the clusters evolve: one combined-`run` iteration (one move +
full relaxation — minutes per frame on grown complexes) per frame on a worker
thread, painting the k = 3 spectral clustering (the three-quark hypothesis:
the states should appear as THREE clusters, registers not required) beside the
modularity trace. The window stays responsive through the long computes because
the engine bindings release the GIL.

Usage:
    python laplacian_clusters.py                       # batch PRE/POST report
    python laplacian_clusters.py --precone 25 --iters 8
    python laplacian_clusters.py --save clusters.png   # embedding figure
    python laplacian_clusters.py --live --iters 100    # watch it evolve
"""
import argparse
import faulthandler
import os
import time

faulthandler.enable()  # a native crash prints the Python stack (which binding died)

import numpy as np
from scipy.cluster.vq import kmeans2

import tessera

cob = tessera.cobordism
obs = tessera.observables

_BLOCK_NAMES = ["q{1}", "q{ω}", "q{ω²}", "q̄{1}", "q̄{ω̄}", "q̄{ω̄²}"]


def primal_graph(st):
    """The complex's 1-skeleton as (sorted vertex ids, index map, dense
    adjacency, Re ℓ² weights) — the weight matrix carries each link's
    disposition (Re ℓ² < 0 = timelike)."""
    edges = st.getEdgeList().toVector()
    vids = sorted({v.getId() for e in edges
                   for v in (e.getSource(), e.getTarget())})
    idx = {v: i for i, v in enumerate(vids)}
    n = len(vids)
    A = np.zeros((n, n))
    W = np.zeros((n, n))
    for e in edges:
        a, b = idx[e.getSource().getId()], idx[e.getTarget().getId()]
        A[a, b] = A[b, a] = 1.0
        W[a, b] = W[b, a] = e.getSquaredLength().real
    return vids, idx, A, W


def affinity_matrix(A, W):
    """Geometry-weighted affinity: w_uv = exp(−|Re ℓ²_uv| / s) on existing links,
    s = the median |Re ℓ²| — geometrically CLOSE vertices couple strongly, so the
    spectral embedding and clusters finally see the optimizer's relaxed metric
    instead of bare connectivity. The causal SIGN is deliberately not injected
    (negative weights make the Laplacian indefinite — balance theory, a different
    instrument); disposition stays visible as the dotted/solid edge styling."""
    magnitudes = np.abs(W[A > 0])
    scale = float(np.median(magnitudes)) if magnitudes.size else 1.0
    scale = scale if scale > 1e-12 else 1.0
    return A * np.exp(-np.abs(W) / scale)


def laplacian_spectrum(A):
    """(eigenvalues, eigenvectors) of the symmetric normalized Laplacian L_sym
    of the given (possibly affinity-weighted) adjacency."""
    d = A.sum(1)
    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    L_sym = np.eye(len(A)) - (dinv_sqrt[:, None] * A * dinv_sqrt[None, :])
    return np.linalg.eigh(L_sym)


def spectral_clusters(A, evals, evecs, k_max, seed):
    """Spectral k-means for k = 2..k_max; return (labels, k, Q) with the best
    Newman–Girvan modularity Q as scored by the engine's SparseGraph."""
    rows, cols = np.nonzero(np.triu(A, 1))
    graph = obs.SparseGraph.fromCOO(rows.astype(np.uint32).tolist(),
                                    cols.astype(np.uint32).tolist(), len(A))
    best = (np.zeros(len(A), dtype=int), 1, 0.0)
    rng = np.random.default_rng(seed)
    for k in range(2, min(k_max, len(A) - 1) + 1):
        # Embed in the first k nontrivial eigenvectors, row-normalized.
        emb = evecs[:, 1:k + 1]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        # kmeans2 can collapse clusters from a bad draw; take the best of a few.
        for attempt in range(4):
            _c, labels = kmeans2(emb, k, minit="++",
                                 seed=int(rng.integers(2**31 - 1)))
            q = graph.modularity(labels.astype(int).tolist())
            if q > best[2]:
                best = (labels.astype(int), k, q)
    return best


def cluster_fixed_k(A, evecs, k, seed):
    """k-means at a FIXED k in the first k nontrivial eigenvectors; returns
    (labels, Q) with Q from the engine's SparseGraph (best of a few attempts)."""
    rows, cols = np.nonzero(np.triu(A, 1))
    graph = obs.SparseGraph.fromCOO(rows.astype(np.uint32).tolist(),
                                    cols.astype(np.uint32).tolist(), len(A))
    emb = evecs[:, 1:k + 1]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    best_labels, best_q = np.zeros(len(A), dtype=int), -1.0
    for attempt in range(4):
        _c, labels = kmeans2(emb, k, minit="++",
                             seed=int(rng.integers(2**31 - 1)))
        q = graph.modularity(labels.astype(int).tolist())
        if q > best_q:
            best_labels, best_q = labels.astype(int), q
    return best_labels, best_q


def cluster_rstates(st, vids, labels, k, degree=3):
    """Per-cluster carry diagnostics: `r_state` of the proton singlet AND its
    conjugate against each cluster's CLOSED STAR — the sub-complex of every
    ambient top cell TOUCHING the cluster (sharing at least one vertex),
    rebuilt through the same cells→Spacetime factory the engine's block
    residuals use. The star, unlike the inside-cells-only sub-complex, wraps
    around the holes a cluster borders: a hole is a MISSING cell whose boundary
    vertices typically straddle cluster lines, and a tight clique's own
    interior is contractible (full leak by construction — the failure mode of
    the stricter readout). Full leak (= ‖target‖² = 3) means the star carries
    nothing of that state; ≈ 0 means it carries it. A cluster with singlet ≈ 0
    and conjugate ≈ 3 IS a proton-sector community; the mirrored pattern is
    the anti-baryon partner."""
    singlet = cob.Proton.singlet()
    conjugate = [complex(z).conjugate() for z in singlet]
    cells = [tuple(sorted(v.getId() for v in c.getVertices()))
             for c in st.getTopSimplices()]
    diagnostics = []
    for cluster in range(k):
        cluster_vertices = {vids[i] for i in range(len(vids))
                            if labels[i] == cluster}
        star = [list(cell) for cell in cells
                if cluster_vertices & set(cell)]
        if not star:  # empty star: full leak by definition
            diagnostics.append({"cells": 0, "singlet": 3.0, "conjugate": 3.0})
            continue
        sub = tessera.spacetime.Spacetime.fromCells(4, star)
        diagnostics.append({
            "cells": len(star),
            "singlet": float(cob.MultiCobordism.r_state(sub, degree, singlet)),
            "conjugate": float(cob.MultiCobordism.r_state(sub, degree,
                                                          conjugate)),
        })
    return diagnostics


def five_cliques(A):
    """All 5-cliques of the adjacency matrix, as sorted index tuples (ordered
    recursive extension — fine at these graph sizes)."""
    n = len(A)
    neighbors = [set(np.nonzero(A[i])[0].tolist()) for i in range(n)]
    found = []

    def extend(clique, candidates):
        if len(clique) == 5:
            found.append(tuple(clique))
            return
        for v in sorted(candidates):
            extend(clique + [v],
                   {u for u in candidates if u > v} & neighbors[v])

    for v in range(n):
        extend([v], {u for u in neighbors[v] if u > v})
    return found


def clique_partition(st, vids, A):
    """(filled_count, open 5-clique tuples, certified hole tuples) — the raw
    objects behind `clique_census` and the soft-mode carry panel. A certified
    hole is itself an open clique, so it appears in both of the last two."""
    cells = {tuple(sorted(v.getId() for v in c.getVertices()))
             for c in st.getTopSimplices()}
    filled = 0
    open_tuples = []
    for clique in five_cliques(A):
        vertex_ids = tuple(sorted(vids[i] for i in clique))
        if vertex_ids in cells:
            filled += 1
        else:
            open_tuples.append(vertex_ids)
    holes = [tuple(sorted(h))
             for h in cob.MultiCobordism.emergent_holes(st, 3)]
    return filled, open_tuples, holes


def clique_census(st, vids, idx, A):
    """The register census of one frame: (#filled 5-cliques, #open 5-cliques,
    #certified holes). A FILLED 5-clique has a top cell on it; an OPEN one has
    all ten edges but no filling cell — a nascent register, the persistent
    graph structure that precedes (and outnumbers) the certified homology: the
    register measured in run-20260811-121826 assembled its clique core 18
    frames before the hole opened and the objective paid out."""
    cells = {tuple(sorted(v.getId() for v in c.getVertices()))
             for c in st.getTopSimplices()}
    filled = open_count = 0
    for clique in five_cliques(A):
        vertex_ids = tuple(sorted(vids[i] for i in clique))
        if vertex_ids in cells:
            filled += 1
        else:
            open_count += 1
    holes = len(cob.MultiCobordism.emergent_holes(st, 3))
    return filled, open_count, holes


def soft_mode_carry(st, candidates, degree=3, temperature=None):
    """The SOFT-MODE (nascent) carry of each candidate register cycle.

    A certified hole carries exact harmonics (eigenvalue 0 of the metric Hodge
    Laplacian); a still-filled or uncertified 5-clique cannot — its boundary
    cycle has zero period against every exact harmonic (dh = 0 on present
    cells). But it CAN couple to the LOW-LYING, almost-harmonic modes, and that
    coupling is the cluster-first picture's pre-certification carry: the
    register state condensing in a soft mode whose eigenvalue certification
    drives to zero.

    For each candidate 5-tuple (its facets must exist as k-cells), computes
      κ_q = Σ_i exp(−λ_i/T) · |period_i(q)|²
    over ALL modes i of L_k (metric weights), where period_i(q) is the signed
    drop-vⱼ facet sum — the engine's own period convention — and T defaults to
    a tenth of the median eigenvalue, so exact harmonics count fully and the
    bulk spectrum is exponentially discounted. Returns per-candidate dicts
    {clique, kappa, softest_lambda} where softest_lambda is the lowest
    eigenvalue contributing ≥10% of the candidate's peak |period|."""
    hodge = cob.HodgeLaplacian(st)
    eigenvalues = np.array(hodge.eigenvalues(degree, True))
    n_modes = len(eigenvalues)
    if n_modes == 0:
        return []
    # Flat layout is (cell, mode): column i is mode i's cochain — verified by
    # the coboundary test (harmonic's period over a PRESENT cell ≈ 0, over the
    # certified hole large).
    modes = np.array(hodge.eigenvectors(degree, True)).reshape(n_modes, n_modes).T
    cell_index = {tuple(sorted(c)): i
                  for i, c in enumerate(
                      cob.EigenstateSynthesis(st, degree).cellSimplices())}
    if temperature is None:
        positive = eigenvalues[eigenvalues > 1e-9]
        temperature = (float(np.median(positive)) / 10.0) if len(positive) else 1.0
    weights = np.exp(-np.clip(eigenvalues, 0.0, None) / temperature)
    out = []
    for candidate in candidates:
        five = sorted(candidate)
        facet_rows = []
        signs = []
        ok = True
        for j in range(5):
            facet = tuple(sorted(five[:j] + five[j + 1:]))
            if facet not in cell_index:
                ok = False
                break
            facet_rows.append(cell_index[facet])
            signs.append((-1) ** j)
        if not ok:
            continue
        periods = np.abs(sum(s * modes[r, :]
                             for s, r in zip(signs, facet_rows)))
        kappa = float(np.sum(weights * periods ** 2))
        peak = periods.max() if periods.size else 0.0
        coupled = np.where(periods >= 0.1 * peak)[0] if peak > 0 else []
        softest = float(eigenvalues[coupled].min()) if len(coupled) else float("nan")
        out.append({"clique": tuple(five), "kappa": kappa,
                    "softest_lambda": softest})
    out.sort(key=lambda d: -d["kappa"])
    return out


def eigengap_suggestion(evals, k_max):
    """The k in 2..k_max with the largest gap λ_{k+1} − λ_k (a standard
    spectral-clustering cluster-count heuristic)."""
    upper = min(k_max, len(evals) - 1)
    gaps = {k: evals[k] - evals[k - 1] for k in range(2, upper + 1)}
    return max(gaps, key=gaps.get) if gaps else 1


def cross_tab(name, member_sets, vids, idx, labels, k):
    """Print how each named vertex set distributes over the clusters."""
    for label_name, vset in zip(name, member_sets):
        counts = np.zeros(k, dtype=int)
        missing = 0
        for v in vset:
            if v in idx:
                counts[labels[idx[v]]] += 1
            else:
                missing += 1
        total = max(counts.sum(), 1)
        top = int(counts.argmax())
        purity = counts[top] / total
        dist = " ".join(f"c{c}:{n}" for c, n in enumerate(counts) if n)
        tail = f"  (+{missing} vertices no longer live)" if missing else ""
        print(f"    {label_name:>8}: {dist:<40} -> mostly c{top} "
              f"(purity {purity:.2f}){tail}")


def analyze(tag, node, k_max, seed, degree=3):
    st = node.st
    vids, idx, A, W = primal_graph(st)
    evals, evecs = laplacian_spectrum(affinity_matrix(A, W))
    n_components = int((evals < 1e-10).sum())
    labels, k, q = spectral_clusters(A, evals, evecs, k_max, seed)
    print(f"\n== {tag} ==")
    print(f"  primal graph: {len(vids)} vertices, {int(A.sum() // 2)} links, "
          f"{n_components} connected component{'s' if n_components != 1 else ''}")
    print(f"  Fiedler value λ₂ = {evals[1]:.4f}   "
          f"leading spectrum: {np.round(evals[:min(8, len(evals))], 3)}")
    print(f"  eigengap suggests k = {eigengap_suggestion(evals, k_max)}; "
          f"best modularity: k = {k}, Q = {q:.3f}")
    sizes = np.bincount(labels, minlength=k)
    print(f"  cluster sizes: {sizes.tolist()}")
    print("  input blocks over clusters:")
    cross_tab(_BLOCK_NAMES, [set(b.vertices) for b in node.inputs],
              vids, idx, labels, k)
    holes = cob.MultiCobordism.emergent_holes(st, degree)
    if holes:
        print("  emergent holes over clusters:")
        cross_tab([f"hole{i}" for i in range(len(holes))],
                  [set(h) for h in holes], vids, idx, labels, k)
    else:
        print("  emergent holes: none yet")
    print("  per-cluster carry (r_state; full leak = 3, ≈0 = carried):")
    for c, d in enumerate(cluster_rstates(st, vids, labels, k, degree)):
        print(f"    c{c}: {d['cells']:3d} cells   "
              f"singlet {{1,ω,ω²}} = {d['singlet']:.3f}   "
              f"conjugate = {d['conjugate']:.3f}")
    return vids, idx, A, evecs, labels, k, q


def plot_embeddings(path, snapshots):
    """Scatter each snapshot in its (v₂, v₃) spectral embedding, colored by
    cluster, with each vertex's block memberships ringed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snapshots), figsize=(7 * len(snapshots), 6),
                             squeeze=False)
    for ax, (tag, node, (vids, idx, _A, evecs, labels, k, q)) in zip(
            axes[0], snapshots):
        xy = evecs[:, 1:3]
        ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=42, zorder=2)
        for block_index, block in enumerate(node.inputs):
            for v in block.vertices:
                if v in idx:
                    i = idx[v]
                    ax.scatter([xy[i, 0]], [xy[i, 1]], s=120 + 40 * block_index,
                               facecolors="none",
                               edgecolors=plt.cm.tab10(block_index % 10),
                               linewidths=0.7, zorder=1)
        ax.set_title(f"{tag} — spectral embedding (k={k}, Q={q:.2f})\n"
                     f"fill = cluster, rings = input-block membership",
                     fontsize=10)
        ax.set_xlabel("v₂ (Fiedler)"); ax.set_ylabel("v₃")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nsaved embedding figure -> {path}")


def dump_state(st, path, meta=None):
    """Write a rehydratable state dump: top cells, per-edge complex ℓ², vertex
    times (the schema `observables.LiveComplex.load` consumes), plus any `meta`.
    Rehydrate any frame later without replaying the run:

        d = json.load(open(path))
        st = tessera.observables.LiveComplex.load(
            d["cells"],
            {(u, v): complex(re, im) for u, v, re, im in d["squared_lengths"]},
            {int(v): t for v, t in d["vertex_times"].items()},
            d["dimensions"])
    """
    import json
    cells = [sorted(v.getId() for v in c.getVertices())
             for c in st.getTopSimplices()]
    squared_lengths = []
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        z = e.getSquaredLength()
        squared_lengths.append([min(a, b), max(a, b), z.real, z.imag])
    vertex_times = {str(v.getId()): float(v.getTime())
                    for v in st.getVertexList().toVector()}
    payload = {"dimensions": 4, "cells": cells,
               "squared_lengths": squared_lengths,
               "vertex_times": vertex_times}
    payload.update(meta or {})
    with open(path, "w") as f:
        json.dump(payload, f)


def _live_snapshot(node, k, k_max, seed):
    """All engine reads for one live frame, done on the WORKER thread while the
    engine is idle: returns plain numpy/python data for the GUI to paint."""
    st = node.st
    vids, idx, A, W = primal_graph(st)
    S = affinity_matrix(A, W)   # metric-weighted: the geometry shapes the spectrum
    evals, evecs = laplacian_spectrum(S)
    labels, q_fixed = cluster_fixed_k(A, evecs, k, seed)
    _lab, k_best, q_best = spectral_clusters(A, evals, evecs, k_max, seed)
    edge_rows, edge_cols = np.nonzero(np.triu(A, 1))
    filled, open_tuples, hole_tuples = clique_partition(st, vids, A)
    certified = set(hole_tuples)
    carried = soft_mode_carry(st, open_tuples)
    kappa_uncert = max((d["kappa"] for d in carried
                        if d["clique"] not in certified), default=float("nan"))
    kappa_cert = max((d["kappa"] for d in carried
                      if d["clique"] in certified), default=float("nan"))
    soft_lambda = next((d["softest_lambda"] for d in carried
                        if d["clique"] not in certified), float("nan"))
    return {
        "kappa_uncert": kappa_uncert,
        "kappa_cert": kappa_cert,
        "soft_lambda": soft_lambda,
        "xy": evecs[:, 1:3].copy(),
        "labels": labels,
        "edges": np.column_stack([edge_rows, edge_cols]),
        "timelike": np.array([W[a, b] < 0
                              for a, b in zip(edge_rows, edge_cols)]),
        "cliques_filled": filled,
        "cliques_open": len(open_tuples),
        "holes": len(hole_tuples),
        "rstates": cluster_rstates(st, vids, labels, k),
        "sizes": np.bincount(labels, minlength=k).tolist(),
        "q_fixed": q_fixed,
        "k_best": k_best, "q_best": q_best,
        "fiedler": float(evals[1]) if len(evals) > 1 else 0.0,
        "cells": len(st.getTopSimplices()),
        "F": float(node.objective()),
        "depth": int(node.last_stage1_lookahead),
    }


def run_live(node, iters, k, k_max, seed, dump_dir=None):
    """Watch the clusters evolve: the worker thread drives one combined-run
    iteration per frame and precomputes the frame's arrays (the engine bindings
    release the GIL, so the GUI stays responsive); the GUI thread only paints.
    The same two-way handshake as the animation examples keeps the engine
    single-threaded: the worker starts frame n only after frame n-1 painted."""
    import queue
    import threading
    import itertools as _it
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frame_queue = queue.Queue()
    paint_done = threading.Event()
    paint_done.set()

    def worker():
        try:
            def dump(tag):
                if dump_dir:
                    dump_state(node.st, os.path.join(dump_dir, f"{tag}.json"),
                               meta={"F": float(node.objective())})
            dump("pre")
            frame_queue.put((-1, _live_snapshot(node, k, k_max, seed)))  # PRE frame
            for i in range(iters):
                paint_done.wait()
                paint_done.clear()
                node.run(max_iters=1, n_candidate_moves=8,
                         grow_boundaries=True, max_lookahead=10)
                dump(f"iter_{i:04d}")
                frame_queue.put((i, _live_snapshot(node, k, k_max, seed)))
            frame_queue.put(None)
        except BaseException as exc:  # surface, don't hang the GUI
            print(f"\nworker failed: {exc!r}", flush=True)
            frame_queue.put(None)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    (ax_emb, ax_q), (ax_census, ax_soft) = axes
    history = {"q_fixed": [], "q_best": [], "k_best": [], "cells": [],
               "cliques_filled": [], "cliques_open": [], "holes": [],
               "kappa_uncert": [], "kappa_cert": [], "soft_lambda": []}

    def paint(frame_index, snap):
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm
        ax_emb.clear()
        xy, labels = snap["xy"], snap["labels"]
        # The primal graph's links, drawn in the embedding: intra-cluster links
        # tinted with their cluster's color, inter-cluster links faint grey — so
        # "how clustered" reads directly as colored knots with few grey threads
        # between them. Disposition shows as linestyle: SOLID = spacelike
        # (Re ℓ² > 0), DOTTED = timelike (Re ℓ² < 0).
        groups = {  # (intra?, timelike?) -> (segments, colors)
            (True, False): ([], []), (True, True): ([], []),
            (False, False): ([], []), (False, True): ([], []),
        }
        for (a, b), is_timelike in zip(snap["edges"], snap["timelike"]):
            intra = labels[a] == labels[b]
            segments, colors = groups[(intra, bool(is_timelike))]
            segments.append([xy[a], xy[b]])
            colors.append(cm.tab10(labels[a] % 10) if intra else "0.82")
        for (intra, is_timelike), (segments, colors) in groups.items():
            if not segments:
                continue
            ax_emb.add_collection(LineCollection(
                segments, colors=colors,
                linewidths=1.1 if intra else 0.6,
                alpha=0.55 if intra else 1.0,
                linestyles=":" if is_timelike else "-",
                zorder=2 if intra else 1))
        ax_emb.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10",
                       vmin=0, vmax=9, s=60, edgecolors="0.3", linewidths=0.4,
                       zorder=3)
        carry = "   ".join(
            f"c{c}: s={d['singlet']:.2f}|c̄={d['conjugate']:.2f}"
            for c, d in enumerate(snap["rstates"]))
        ax_emb.set_title(
            f"spectral embedding, k={k} clusters (sizes {snap['sizes']}) · "
            f"links: solid=spacelike, dotted=timelike\n"
            f"iter {frame_index + 1}/{iters} · {snap['cells']} cells · "
            f"F={snap['F']:.2f} · λ₂={snap['fiedler']:.3f} · "
            f"lookahead={snap['depth']}\n"
            f"r_state per cluster (full leak 3, ≈0 carried):  {carry}",
            fontsize=9)
        ax_emb.set_xlabel("v₂ (Fiedler)")
        ax_emb.set_ylabel("v₃")

        history["q_fixed"].append(snap["q_fixed"])
        history["q_best"].append(snap["q_best"])
        history["k_best"].append(snap["k_best"])
        history["cells"].append(snap["cells"])
        xs = range(len(history["q_fixed"]))
        ax_q.clear()
        ax_q.plot(xs, history["q_fixed"], color="C0", marker=".",
                  label=f"Q at k={k} (three-cluster hypothesis)")
        ax_q.plot(xs, history["q_best"], color="C1", marker=".", alpha=0.6,
                  label="best Q over k=2..%d" % k_max)
        for x, kb in zip(xs, history["k_best"]):
            ax_q.annotate(str(kb), (x, history["q_best"][x]),
                          textcoords="offset points", xytext=(0, 6),
                          ha="center", fontsize=6, color="C1")
        ax_q.axhline(0.3, color="0.6", ls=":", lw=0.8,
                     label="Q=0.3 (conventional 'real structure')")
        ax_q.set_xlabel("iteration")
        ax_q.set_ylabel("modularity Q")
        ax_q.set_title("cluster quality (best-k annotated)", fontsize=10)
        ax_q.legend(loc="lower right", fontsize=8)

        history["cliques_filled"].append(snap["cliques_filled"])
        history["cliques_open"].append(snap["cliques_open"])
        history["holes"].append(snap["holes"])
        ax_census.clear()
        ax_census.plot(xs, history["cliques_filled"], color="0.6", lw=0.9,
                       label="filled 5-cliques (cells)")
        ax_census.plot(xs, history["cliques_open"], color="C3", marker=".",
                       label="OPEN 5-cliques (nascent registers)")
        ax_census.plot(xs, history["holes"], color="C2", marker=".",
                       label="certified holes (emergent_holes)")
        ax_census.set_title("clique census — registers as graph structure\n"
                            "(open cliques precede certified holes)",
                            fontsize=10)
        ax_census.set_xlabel("iteration")
        ax_census.set_ylabel("count")
        ax_census.legend(loc="upper left", fontsize=8)

        history["kappa_uncert"].append(snap["kappa_uncert"])
        history["kappa_cert"].append(snap["kappa_cert"])
        history["soft_lambda"].append(snap["soft_lambda"])
        ax_soft.clear()
        ax_soft.semilogy(xs, history["kappa_uncert"], color="C3", marker=".",
                         label="best UNCERTIFIED κ (nascent carry)")
        ax_soft.semilogy(xs, history["kappa_cert"], color="C2", marker=".",
                         label="best certified κ")
        lam = snap["soft_lambda"]
        ax_soft.set_title("soft-mode carry κ = Σ e^(−λ/T)·|period|²\n"
                          f"condensation depth (softest coupled λ, uncert.): "
                          f"{lam:.3f}" if lam == lam else
                          "soft-mode carry κ (no uncertified candidates)",
                          fontsize=9)
        ax_soft.set_xlabel("iteration")
        ax_soft.set_ylabel("κ")
        ax_soft.legend(loc="lower right", fontsize=8)
        rs = " ".join(f"[{d['singlet']:.2f}|{d['conjugate']:.2f}]"
                      for d in snap["rstates"])
        print(f"\riter {frame_index + 1}/{iters}: cells={snap['cells']} "
              f"F={snap['F']:.2f} Q{k}={snap['q_fixed']:.3f} "
              f"best k={snap['k_best']} Q={snap['q_best']:.3f} "
              f"r_state s|c̄ {rs}",
              end="", flush=True)

    # After the run finishes, HOLD the final frame on screen for a day (so an
    # overnight run is still there in the morning), then close so the process
    # exits on its own. Closing the window by hand ends it any time.
    hold_seconds = 86400
    finished_at = [None]

    def on_timer(_frame):
        import time as _time
        if finished_at[0] is not None:
            if _time.time() - finished_at[0] >= hold_seconds:
                plt.close(fig)
            return []
        while True:
            try:
                item = frame_queue.get_nowait()
            except queue.Empty:
                return []
            if item is None:
                finished_at[0] = _time.time()
                print(f"\nrun complete — window stays up for "
                      f"{hold_seconds // 3600}h (close it to exit sooner)")
                return []
            frame_index, snap = item
            paint(frame_index, snap)
            paint_done.set()

    animation = FuncAnimation(fig, on_timer, frames=_it.count(), interval=250,
                              repeat=False, cache_frame_data=False, blit=False)
    threading.Thread(target=worker, name="laplacian-live", daemon=True).start()
    plt.show()


def load_dump(path):
    """Rehydrate one state dump (see `dump_state`) into a Spacetime + metadata."""
    import json
    with open(path) as f:
        d = json.load(f)
    st = tessera.observables.LiveComplex.load(
        d["cells"],
        {(u, v): complex(re, im) for u, v, re, im in d["squared_lengths"]},
        {int(v): t for v, t in d["vertex_times"].items()},
        d["dimensions"])
    return st, d


def view_dumps(dump_dir, k, k_max, seed, hold_seconds=86400):
    """Rebuild the live view from a finished run's state dumps — the cheap
    analysis replays in minutes; the expensive engine work is already in the
    dumps — then hold the final frame on screen (24h, or close it by hand)."""
    import glob
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    paths = sorted(glob.glob(os.path.join(dump_dir, "iter_*.json")))
    pre = os.path.join(dump_dir, "pre.json")
    if os.path.exists(pre):
        paths.insert(0, pre)
    if not paths:
        raise SystemExit(f"no dumps found in {dump_dir}")
    history = {"q_fixed": [], "q_best": [], "k_best": [],
               "cliques_filled": [], "cliques_open": [], "holes": []}
    final = None
    for i, path in enumerate(paths):
        st, meta = load_dump(path)
        vids, idx, A, W = primal_graph(st)
        S = affinity_matrix(A, W)
        evals, evecs = laplacian_spectrum(S)
        labels, q_fixed = cluster_fixed_k(A, evecs, k, seed)
        _lab, k_best, q_best = spectral_clusters(A, evals, evecs, k_max, seed)
        history["q_fixed"].append(q_fixed)
        history["q_best"].append(q_best)
        history["k_best"].append(k_best)
        filled, open_cliques, holes = clique_census(st, vids, idx, A)
        history["cliques_filled"].append(filled)
        history["cliques_open"].append(open_cliques)
        history["holes"].append(holes)
        if path == paths[-1]:
            rows, cols = np.nonzero(np.triu(A, 1))
            final = {
                "xy": evecs[:, 1:3], "labels": labels,
                "edges": np.column_stack([rows, cols]),
                "timelike": np.array([W[a, b] < 0
                                      for a, b in zip(rows, cols)]),
                "sizes": np.bincount(labels, minlength=k).tolist(),
                "rstates": cluster_rstates(st, vids, labels, k),
                "cells": len(st.getTopSimplices()),
                "F": meta.get("F", float("nan")),
                "fiedler": float(evals[1]) if len(evals) > 1 else 0.0,
            }
        print(f"\ranalyzed {i + 1}/{len(paths)} dumps", end="", flush=True)
    print()

    fig, (ax_emb, ax_q, ax_census) = plt.subplots(1, 3, figsize=(19.5, 6.5))
    xy, labels = final["xy"], final["labels"]
    groups = {(intra, tl): ([], []) for intra in (True, False)
              for tl in (True, False)}
    for (a, b), is_timelike in zip(final["edges"], final["timelike"]):
        intra = labels[a] == labels[b]
        segments, colors = groups[(intra, bool(is_timelike))]
        segments.append([xy[a], xy[b]])
        colors.append(cm.tab10(labels[a] % 10) if intra else "0.82")
    for (intra, is_timelike), (segments, colors) in groups.items():
        if segments:
            ax_emb.add_collection(LineCollection(
                segments, colors=colors, linewidths=1.1 if intra else 0.6,
                alpha=0.55 if intra else 1.0,
                linestyles=":" if is_timelike else "-",
                zorder=2 if intra else 1))
    ax_emb.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9,
                   s=60, edgecolors="0.3", linewidths=0.4, zorder=3)
    carry = "   ".join(f"c{c}: s={d['singlet']:.2f}|c̄={d['conjugate']:.2f}"
                       for c, d in enumerate(final["rstates"]))
    ax_emb.set_title(
        f"FINAL frame of {os.path.basename(os.path.normpath(dump_dir))} · "
        f"k={k} (sizes {final['sizes']}) · solid=spacelike, dotted=timelike\n"
        f"{final['cells']} cells · F={final['F']:.2f} · "
        f"λ₂={final['fiedler']:.3f}\n"
        f"r_state per cluster (full leak 3, ≈0 carried):  {carry}", fontsize=9)
    ax_emb.set_xlabel("v₂ (Fiedler)"); ax_emb.set_ylabel("v₃")

    xs = range(len(history["q_fixed"]))
    ax_q.plot(xs, history["q_fixed"], color="C0",
              label=f"Q at k={k} (three-cluster hypothesis)")
    ax_q.plot(xs, history["q_best"], color="C1", alpha=0.6,
              label=f"best Q over k=2..{k_max}")
    ax_q.axhline(0.3, color="0.6", ls=":", lw=0.8,
                 label="Q=0.3 (conventional 'real structure')")
    ax_q.set_xlabel("frame"); ax_q.set_ylabel("modularity Q")
    from collections import Counter
    ax_q.set_title("cluster quality over the whole run · best-k counts: "
                   + " ".join(f"k={kk}:{n}" for kk, n
                              in sorted(Counter(history['k_best']).items())),
                   fontsize=10)
    ax_q.legend(loc="lower right", fontsize=8)

    ax_census.plot(xs, history["cliques_filled"], color="0.6", lw=0.9,
                   label="filled 5-cliques (cells)")
    ax_census.plot(xs, history["cliques_open"], color="C3", marker=".",
                   label="OPEN 5-cliques (nascent registers)")
    ax_census.plot(xs, history["holes"], color="C2", marker=".",
                   label="certified holes (emergent_holes)")
    ax_census.set_title("clique census — registers as graph structure",
                        fontsize=10)
    ax_census.set_xlabel("frame"); ax_census.set_ylabel("count")
    ax_census.legend(loc="upper left", fontsize=8)

    timer = fig.canvas.new_timer(interval=hold_seconds * 1000)
    timer.single_shot = True
    timer.add_callback(lambda: plt.close(fig))
    timer.start()
    print(f"viewing final state — window stays up for {hold_seconds // 3600}h "
          f"(close it to exit sooner)")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--precone", type=int, default=12,
                    help="pre-grow the direct node's seed by this many cone-ins")
    ap.add_argument("--precone-timelike", action="store_true", default=True,
                    dest="precone_timelike",
                    help="ALL-timelike precone material (default on; the "
                         "spacelike landscape stalls)")
    ap.add_argument("--precone-alternate", action="store_true",
                    dest="precone_alternate",
                    help="instead alternate the precone cone-ins timelike/"
                         "spacelike (balanced causal content; wins over "
                         "--precone-timelike)")
    ap.add_argument("--iters", type=int, default=3,
                    help="combined-run iterations between PRE and POST (each = "
                         "one move + full relaxation, so a few can be minutes)")
    ap.add_argument("--k-max", type=int, default=8, dest="k_max",
                    help="largest cluster count tried")
    ap.add_argument("--k", type=int, default=3,
                    help="(live) the FIXED cluster count painted each frame — "
                         "3 = the three-quark hypothesis")
    ap.add_argument("--live", action="store_true",
                    help="watch the clusters evolve: one combined-run iteration "
                         "per frame (minutes per frame on grown complexes)")
    ap.add_argument("--save", help="write the spectral-embedding figure here")
    ap.add_argument("--dump-dir", dest="dump_dir",
                    help="(live) directory for per-frame state dumps "
                         "(default: laplacian_dumps/run-<timestamp>); "
                         "rehydrate any frame with observables.LiveComplex.load "
                         "— see dump_state's docstring")
    ap.add_argument("--view", metavar="DUMP_DIR",
                    help="rebuild the view from a finished run's dumps (cheap "
                         "analysis only, no engine work) and hold the final "
                         "frame on screen")
    args = ap.parse_args()

    if args.view:
        view_dumps(args.view, args.k, args.k_max, args.seed)
        return

    p = cob.Proton(seed=args.seed, precone=args.precone,
                   precone_timelike=args.precone_timelike,
                   precone_alternate=args.precone_alternate)
    node = p.direct_node(args.seed)

    if args.live:
        iters = args.iters if args.iters > 3 else 500  # live default: much longer
        dump_dir = args.dump_dir or os.path.join(
            "laplacian_dumps", time.strftime("run-%Y%m%d-%H%M%S"))
        os.makedirs(dump_dir, exist_ok=True)
        print(f"state dumps -> {dump_dir}/ (pre.json, iter_0000.json, ...)")
        run_live(node, iters, args.k, args.k_max, args.seed, dump_dir=dump_dir)
        return

    pre = analyze("PRE-interaction (seeded, undriven)", node, args.k_max, args.seed)
    snapshots = [("PRE", node, pre)]

    if args.iters > 0:
        print(f"\ndriving: {args.iters} combined-run iterations "
              f"(one move + full relaxation each) ...", flush=True)
        node.run(max_iters=args.iters, n_candidate_moves=8,
                 grow_boundaries=True, max_lookahead=10)
        post = analyze("POST-interaction (driven)", node, args.k_max, args.seed)
        snapshots.append(("POST", node, post))

    if args.save:
        plot_embeddings(args.save, snapshots)


if __name__ == "__main__":
    main()
