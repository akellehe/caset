"""Greedy hill-climb on Newman's leading-eigenvector modularity of the
dual graph of a 5-simplex complex.  At each step we propose a small
batch of add/remove moves on the simplex set, evaluate Q via Newman's
spectral bisection on the resulting dual graph, and accept the best
improving move.  Whenever Q crosses the next increment, we stop
optimizing and measure the spectral dimension via discrete diffusion.
Plot D_S vs Q at the end.

The dual graph is built by face-sharing: simplices share an edge iff
they share 4 of their 5 vertices.  A "leaf" add (replace one vertex of
an existing simplex with a fresh vertex) creates a pendant in the dual
graph -- the prototypical Q-raising move.  Remove moves can also raise
Q by cutting bridges between dense regions.

Modularity follows Newman (2006), "Modularity and community structure
in networks", PNAS 103(23): recursive spectral bisection on the
modularity matrix B = A - k k^T / 2m.  For a subgroup S:
  B^(S)_ij = B_ij - delta_ij * sum_{k in S} B_ik    (i, j in S)
Split by sign of the leading eigenvector when the leading eigenvalue is
positive and Delta Q = s^T B^(S) s / (4m) > 0; otherwise S is indivisible.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def build_initial_chain(n_vertices):
    """Initial chain of 5-simplices: simp_i = (i, i+1, ..., i+4).
    Dual graph is a path on n_vertices - 4 nodes."""
    return [tuple(range(i, i + 5)) for i in range(n_vertices - 4)]


def _realize_tree_as_simplices(adj, n_nodes, rng):
    """Realize a tree (given as an adjacency dict) as 5-simplices.

    Each child shares 4 vertices with its parent by replacing one of
    the parent's 5 vertices with a fresh one.  To avoid spurious face-
    sharings between non-adjacent tree nodes, each node's children must
    replace pairwise-distinct positions, none of which equals the
    position the node itself was created at.  This caps total node
    degree at 5 in the realization.

    Returns list of 5-tuples or None if max degree > 5.
    """
    deg = np.array([len(adj[i]) for i in range(n_nodes)])
    if deg.max() > 5:
        return None

    root = int(deg.argmax())
    parent = [-1] * n_nodes
    own_pos = [-1] * n_nodes
    used_by_children = [set() for _ in range(n_nodes)]
    visited = np.zeros(n_nodes, dtype=bool)
    visited[root] = True
    order = [root]

    queue = [root]
    while queue:
        next_q = []
        for u in queue:
            forbidden = set(used_by_children[u])
            if u != root:
                forbidden.add(own_pos[u])
            available = [p for p in range(5) if p not in forbidden]

            children = [v for v in adj[u] if not visited[v]]
            if len(children) > len(available):
                return None

            rng.shuffle(available)
            for child, pos in zip(children, available):
                visited[child] = True
                parent[child] = u
                own_pos[child] = pos
                used_by_children[u].add(pos)
                order.append(child)
                next_q.append(child)
        queue = next_q

    next_v = [0]

    def fresh():
        v = next_v[0]
        next_v[0] += 1
        return v

    simps = [None] * n_nodes
    simps[root] = tuple(fresh() for _ in range(5))
    for v in order[1:]:
        p = parent[v]
        new_simp = list(simps[p])
        new_simp[own_pos[v]] = fresh()
        simps[v] = tuple(new_simp)

    return simps


def build_initial_ba(n_simplices, rng):
    """Initialize n_simplices simplices via Barabasi-Albert preferential
    attachment with m=1 (tree topology).  Each new simplex attaches to
    an existing one with probability proportional to its current dual-
    graph degree (using `degree + 1` so the bare root is selectable).
    The 5-simplex realization caps any node's degree at 5, so attachment
    skips simplices already at that cap -- a soft-capped BA tree.
    """
    if n_simplices <= 0:
        return []

    next_v = [0]

    def fresh():
        v = next_v[0]
        next_v[0] += 1
        return v

    simps = [tuple(fresh() for _ in range(5))]
    own_pos = [None]
    used_by_children = [set()]

    for _ in range(1, n_simplices):
        eligible = []
        weights = []
        for j in range(len(simps)):
            cap = 5 if j == 0 else 4
            if len(used_by_children[j]) < cap:
                deg = len(used_by_children[j]) + (0 if j == 0 else 1)
                eligible.append(j)
                weights.append(deg + 1)
        if not eligible:
            break

        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        p = eligible[int(rng.choice(len(eligible), p=weights))]

        forbidden = set(used_by_children[p])
        if p != 0:
            forbidden.add(own_pos[p])
        available = [pos for pos in range(5) if pos not in forbidden]
        if not available:
            break
        pos = int(rng.choice(available))

        new_simp = list(simps[p])
        new_simp[pos] = fresh()
        simps.append(tuple(new_simp))
        own_pos.append(pos)
        used_by_children.append(set())
        used_by_children[p].add(pos)

    return simps


def build_initial_mst(n_simplices, rng, status_every=200):
    """Initialize `n_simplices` simplices arranged as the MST of a
    complete graph with iid uniform edge weights, realized as a tree of
    5-simplices.  Re-samples weights until the MST has max degree <= 5
    (the cap of this realization scheme).  Logs a progress message every
    `status_every` attempts.
    """
    if n_simplices <= 0:
        return []
    if n_simplices == 1:
        return [(0, 1, 2, 3, 4)]

    attempt = 0
    while True:
        attempt += 1
        W = rng.random((n_simplices, n_simplices))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        mst = sparse.csgraph.minimum_spanning_tree(sparse.csr_matrix(W))
        mst = mst.maximum(mst.T).tocsr()

        adj = {i: mst.getrow(i).indices.tolist() for i in range(n_simplices)}
        deg = np.array([len(adj[i]) for i in range(n_simplices)])
        if deg.max() <= 5:
            simps = _realize_tree_as_simplices(adj, n_simplices, rng)
            if simps is not None:
                if attempt > 1:
                    print(f"  MST init: realized after {attempt} attempts")
                return simps

        if attempt % status_every == 0:
            print(f"  MST init: {attempt} attempts so far "
                  f"(max degree this try = {int(deg.max())})")


def build_dual_adjacency(simplices):
    """Build the symmetric binary dual-graph adjacency.  Two simplices
    are adjacent iff they share a 4-face (i.e., share 4 of 5 vertices).
    """
    n = len(simplices)
    if n == 0:
        return sparse.csc_matrix((0, 0))

    face_index = {}
    for i, simp in enumerate(simplices):
        sorted_simp = sorted(simp)
        for j in range(5):
            face = tuple(sorted_simp[:j] + sorted_simp[j + 1:])
            face_index.setdefault(face, []).append(i)

    rows, cols = [], []
    for simps in face_index.values():
        for a in range(len(simps)):
            for b in range(a + 1, len(simps)):
                rows.append(simps[a])
                cols.append(simps[b])
                rows.append(simps[b])
                cols.append(simps[a])

    if not rows:
        return sparse.csc_matrix((n, n))
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A


def newman_modularity(A):
    """Newman's leading-eigenvector modularity (PNAS 2006).

    Recursive spectral bisection on the modularity matrix
    B = A - k k^T / (2m).  At each subgroup S we form the generalized
    modularity matrix B^(S) (B restricted to S with a diagonal
    correction), take its leading eigenpair, and split by the sign of
    the leading eigenvector when the leading eigenvalue is positive and
    Delta Q = s^T B^(S) s / (4m) > 0.

    Returns (Q, communities) where communities is a list of frozensets
    of node indices.  Dense path; fine up to a few thousand nodes.
    """
    A_csr = A.tocsr()
    n = A_csr.shape[0]
    if n == 0:
        return 0.0, []

    deg = np.asarray(A_csr.sum(axis=0)).ravel()
    m2 = float(deg.sum())  # 2m
    if m2 == 0:
        return 0.0, [frozenset(range(n))]

    A_dense = A_csr.toarray().astype(float)
    B = A_dense - np.outer(deg, deg) / m2

    communities = []
    stack = [np.arange(n)]
    while stack:
        S = stack.pop()
        if len(S) == 0:
            continue
        if len(S) == 1:
            communities.append(frozenset(S.tolist()))
            continue

        B_S = B[np.ix_(S, S)].copy()
        np.fill_diagonal(B_S, B_S.diagonal() - B_S.sum(axis=1))

        eigvals, eigvecs = np.linalg.eigh(B_S)
        beta = eigvals[-1]
        if beta <= 1e-10:
            communities.append(frozenset(S.tolist()))
            continue

        u = eigvecs[:, -1]
        signs = np.where(u >= 0.0, 1, -1)
        if abs(signs.sum()) == len(signs):
            communities.append(frozenset(S.tolist()))
            continue

        dQ = float(signs @ B_S @ signs) / (2.0 * m2)  # = s^T B^(S) s / 4m
        if dQ <= 1e-10:
            communities.append(frozenset(S.tolist()))
            continue

        stack.append(S[signs == 1])
        stack.append(S[signs == -1])

    Q = 0.0
    for c in communities:
        c_arr = np.fromiter(c, dtype=int)
        if c_arr.size == 0:
            continue
        intra = float(A_dense[np.ix_(c_arr, c_arr)].sum())
        d_c = float(deg[c_arr].sum())
        Q += intra / m2 - (d_c / m2) ** 2

    return Q, communities


def newman_modularity_q(A):
    return newman_modularity(A)[0]


def build_transition_matrix(A):
    if A.shape[0] == 0:
        return None
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    return (A @ sparse.diags(1.0 / deg)).tocsc()


def diffuse_sparse(T, starts, max_sigma):
    N = T.shape[0]
    n_walks = len(starts)
    return_probs = np.zeros((n_walks, max_sigma + 1))
    return_probs[:, 0] = 1.0
    prob = np.zeros((N, n_walks))
    for w, s in enumerate(starts):
        prob[s, w] = 1.0
    walk_idx = np.arange(n_walks)
    for sigma in range(1, max_sigma + 1):
        prob = T @ prob
        return_probs[:, sigma] = prob[starts, walk_idx]
    return return_probs


def compute_spectral_dimension(return_prob):
    sigma = np.arange(len(return_prob))
    valid = (sigma > 1) & (return_prob > 0)
    s = sigma[valid].astype(float)
    p = return_prob[valid]
    if len(s) < 2:
        return s, np.zeros(len(s))
    log_s, log_p = np.log(s), np.log(p)
    ds = np.zeros(len(log_s))
    ds[1:-1] = (log_p[2:] - log_p[:-2]) / (log_s[2:] - log_s[:-2])
    ds[0] = (log_p[1] - log_p[0]) / (log_s[1] - log_s[0])
    ds[-1] = (log_p[-1] - log_p[-2]) / (log_s[-1] - log_s[-2])
    return s, -2.0 * ds


def measure_spectral_dim(A, n_walks, max_sigma, rng):
    """Returns (D_S(small sigma), D_S(large sigma))."""
    T = build_transition_matrix(A)
    if T is None or T.shape[0] == 0:
        return float('nan'), float('nan')
    N = T.shape[0]
    n = min(n_walks, N)
    starts = rng.choice(N, size=n, replace=False)
    rp = diffuse_sparse(T, starts, max_sigma)
    rp_avg = rp.mean(axis=0)
    _, ds_vals = compute_spectral_dimension(rp_avg)
    if len(ds_vals) == 0:
        return float('nan'), float('nan')
    n_tail = max(1, len(ds_vals) // 5)
    return float(np.mean(ds_vals[:n_tail])), float(np.mean(ds_vals[-n_tail:]))


def propose_moves(simplices, n_candidates, fresh_vertex, rng,
                  p_leaf=0.5, p_remove=0.35):
    """Generate candidate (kind, payload) moves.

    - 'add_leaf': take an existing simplex, replace one vertex with a
      brand-new vertex.  Creates a pendant in the dual graph.
    - 'add_internal': pick 5 random existing vertices.  May densify or
      bridge; greedy filter weeds out the bad ones.
    - 'remove': drop an existing simplex.
    """
    n = len(simplices)
    moves = []
    seen = set()
    simp_set = set(simplices)
    all_vertices = list({v for s in simplices for v in s})

    for _ in range(n_candidates):
        u = rng.random()
        if u < p_leaf and n > 0:
            base = list(simplices[rng.integers(n)])
            base[rng.integers(5)] = fresh_vertex()
            new_simp = tuple(sorted(base))
            if new_simp not in simp_set and new_simp not in seen:
                moves.append(('add', new_simp))
                seen.add(new_simp)
        elif u < p_leaf + p_remove and n > 5:
            moves.append(('remove', int(rng.integers(n))))
        else:
            if len(all_vertices) >= 5:
                verts = tuple(sorted(rng.choice(all_vertices, size=5,
                                                 replace=False).tolist()))
                if verts not in simp_set and verts not in seen:
                    moves.append(('add', verts))
                    seen.add(verts)
    return moves


def apply_move(simplices, action):
    kind, payload = action
    if kind == 'add':
        return simplices + [payload]
    return simplices[:payload] + simplices[payload + 1:]


def hill_climb(initial_simplices, target_dq, n_candidates,
               max_iterations, n_walks, max_sigma, rng):
    """Greedy hill climb on Newman Q.  Records (Q, D_S) crossings of
    each Q-increment of size `target_dq`."""
    simplices = list(initial_simplices)
    next_v = max((v for s in simplices for v in s), default=-1) + 1

    def fresh():
        nonlocal next_v
        v = next_v
        next_v += 1
        return v

    A = build_dual_adjacency(simplices)
    Q = newman_modularity_q(A)
    measurements = []
    ds_s, ds_l = measure_spectral_dim(A, n_walks, max_sigma, rng)
    measurements.append({'Q': Q, 'ds_small': ds_s, 'ds_large': ds_l,
                          'n': len(simplices), 'iter': 0})
    print(f"  iter=   0  N={len(simplices):4d}  Q={Q:.4f}  "
          f"D_S(s)={ds_s:.3f}  D_S(l)={ds_l:.3f}")
    next_threshold = Q + target_dq

    for it in range(1, max_iterations + 1):
        moves = propose_moves(simplices, n_candidates, fresh, rng)
        if not moves:
            print("  no candidate moves; stopping")
            break

        best_dQ = 0.0
        best = None
        for move in moves:
            trial = apply_move(simplices, move)
            trial_A = build_dual_adjacency(trial)
            trial_Q = newman_modularity_q(trial_A)
            if trial_Q - Q > best_dQ:
                best_dQ = trial_Q - Q
                best = (trial, trial_A, trial_Q)

        if best is None:
            print(f"  iter={it:4d}  local max at Q={Q:.4f}; stopping")
            break

        simplices, A, Q = best

        if Q >= next_threshold:
            ds_s, ds_l = measure_spectral_dim(A, n_walks, max_sigma, rng)
            measurements.append({'Q': Q, 'ds_small': ds_s, 'ds_large': ds_l,
                                  'n': len(simplices), 'iter': it})
            print(f"  iter={it:4d}  N={len(simplices):4d}  Q={Q:.4f}  "
                  f"D_S(s)={ds_s:.3f}  D_S(l)={ds_l:.3f}")
            next_threshold = Q + target_dq

    return measurements


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', choices=['chain', 'mst', 'ba'],
                        default='chain',
                        help='Initial topology: linear chain, MST of a '
                             'random complete graph, or Barabasi-Albert '
                             '(m=1, soft-capped at degree 5)')
    parser.add_argument('--n-vertices', type=int, default=80,
                        help='Initial chain length (chain mode); for mst '
                             'mode the initial simplex count is n_vertices - 4')
    parser.add_argument('--target-dq', type=float, default=0.05,
                        help='Q increment between measurements')
    parser.add_argument('--n-candidates', type=int, default=15,
                        help='Candidate moves evaluated per iteration')
    parser.add_argument('--max-iterations', type=int, default=400)
    parser.add_argument('--n-walks', type=int, default=80)
    parser.add_argument('--max-sigma', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=str,
                        default='./spectral_vs_modularity.png')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_init = max(1, args.n_vertices - 4)
    if args.init == 'mst':
        print(f"Building MST initial graph on {n_init} simplices...")
        initial = build_initial_mst(n_init, rng)
        print(f"Initial MST: {len(initial)} simplices")
    elif args.init == 'ba':
        print(f"Building Barabasi-Albert initial graph on {n_init} "
              f"simplices (m=1, soft cap at degree 5)...")
        initial = build_initial_ba(n_init, rng)
        print(f"Initial BA: {len(initial)} simplices")
    else:
        initial = build_initial_chain(args.n_vertices)
        print(f"Initial chain: {len(initial)} simplices on "
              f"{args.n_vertices} vertices")
    print(f"Greedy hill climb: dq={args.target_dq}, "
          f"max_iter={args.max_iterations}, candidates={args.n_candidates}")
    print()

    t0 = time.time()
    measurements = hill_climb(initial, args.target_dq,
                              args.n_candidates, args.max_iterations,
                              args.n_walks, args.max_sigma, rng)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s, {len(measurements)} measurements")

    Qs = np.array([m['Q'] for m in measurements])
    Dl = np.array([m['ds_large'] for m in measurements])
    Ds = np.array([m['ds_small'] for m in measurements])
    Ns = np.array([m['n'] for m in measurements])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(Qs, Dl, 'o-', linewidth=1.5, markersize=6,
             label=r'$D_S$ (large $\sigma$)')
    ax1.plot(Qs, Ds, 's--', linewidth=1.0, markersize=5, alpha=0.7,
             label=r'$D_S$ (small $\sigma$)')
    ax1.set_xlabel('Modularity $Q$ (Newman leading eigenvector)', fontsize=13)
    ax1.set_ylabel(r'Spectral dimension $D_S$', fontsize=13)
    ax1.set_title('Spectral dimension along greedy modularity ascent',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2.plot(Qs, Ns, 'k.-', linewidth=1.0, markersize=5)
    ax2.set_xlabel('Modularity $Q$ (Newman leading eigenvector)', fontsize=13)
    ax2.set_ylabel('Number of simplices', fontsize=13)
    ax2.set_title('Simplex count along trajectory', fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"Saved {args.save}")
