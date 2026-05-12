"""How fast does D_S(sigma) converge with thermalization?

Smaller, faster setup: target ~ 4000 N41 simplices.  Each checkpoint
prints volume, slice-width stats, and spectral-dimension samples at
specific sigmas (well below sqrt(N4) saturation).
"""
import argparse
import time

import numpy as np
from scipy import sparse

import tessera


def build_T(st):
    rows, cols, N = st.getDualAdjacency()
    if N == 0 or len(rows) == 0:
        return None, N
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A.data[:] = 1.0
    deg = np.array(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    T = A @ sparse.diags(1.0 / deg)
    return T.tocsc(), N


def diffuse(T, starts, max_sigma):
    N = T.shape[0]
    n_walks = len(starts)
    rp = np.zeros((n_walks, max_sigma + 1))
    rp[:, 0] = 1.0
    prob = np.zeros((N, n_walks))
    for w, s in enumerate(starts):
        prob[s, w] = 1.0
    walk_idx = np.arange(n_walks)
    for sigma in range(1, max_sigma + 1):
        prob = T @ prob
        rp[:, sigma] = prob[starts, walk_idx]
    return rp


def spectral_dim_at(rp_avg, sigma_targets):
    """Return D_S evaluated at specific sigma values (centered diff)."""
    sigmas = np.arange(len(rp_avg))
    valid = (sigmas > 1) & (rp_avg > 0)
    s = sigmas[valid].astype(float)
    p = rp_avg[valid]
    if len(s) < 3:
        return [None] * len(sigma_targets)
    log_s = np.log(s)
    log_p = np.log(p)
    ds = np.zeros(len(log_s))
    ds[1:-1] = (log_p[2:] - log_p[:-2]) / (log_s[2:] - log_s[:-2])
    ds[0] = (log_p[1] - log_p[0]) / (log_s[1] - log_s[0])
    ds[-1] = (log_p[-1] - log_p[-2]) / (log_s[-1] - log_s[-2])
    D_S = -2.0 * ds
    out = []
    for sig_t in sigma_targets:
        idx = np.searchsorted(s, sig_t)
        if 0 < idx < len(s):
            out.append(float(D_S[idx]))
        else:
            out.append(None)
    return out


def slice_widths(st):
    times = list(st.getTimeSlices())
    return [len(st.getVerticesAtTime(t)) for t in times]


def measure(st, n_walks, max_sigma, sigma_targets):
    T, N = build_T(st)
    if T is None or N < 2:
        return None
    starts = np.random.choice(N, size=min(n_walks, N), replace=False)
    rp = diffuse(T, starts, max_sigma).mean(axis=0)
    ds_at = spectral_dim_at(rp, sigma_targets)
    return {
        "N4": st.getSimplexCount(),
        "N41": st.getN41(),
        "ds_at": ds_at,
        "widths": slice_widths(st),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-simplices", type=int, default=8000)
    ap.add_argument("--max-build", type=int, default=1600)
    ap.add_argument("--total-therm", type=int, default=4000)
    ap.add_argument("--n-checkpoints", type=int, default=12)
    ap.add_argument("--max-sigma", type=int, default=80)
    ap.add_argument("--n-walks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    sigma_targets = [4, 12, 30, 60]

    np.random.seed(args.seed)
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    initial_size = min(args.n_simplices, args.max_build)
    st.build(initial_size)
    target = (st.getN41() if args.n_simplices <= args.max_build
              else args.n_simplices // 2)
    print(f"build: N4={st.getSimplexCount()} target={target}", flush=True)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    cdt.tune()
    print(f"tune:  N4={st.getSimplexCount()} N41={st.getN41()}", flush=True)

    header = (f"{'sweeps':>7} {'N4':>5} {'N41':>5} {'ws_min':>6} "
              f"{'ws_mean':>7} {'ws_max':>6} "
              + " ".join(f"DS@{s:>3}" for s in sigma_targets)
              + f" {'sweep_s':>7}")
    print(header, flush=True)

    # Logarithmic checkpoints: 0, 25, 50, 100, 200, 400, 800, ...
    checkpoints = [0]
    s = 25
    while s <= args.total_therm:
        checkpoints.append(s)
        s *= 2
    if checkpoints[-1] != args.total_therm:
        checkpoints.append(args.total_therm)
    sweeps_done = 0

    def log_row(sweeps, m, dt):
        widths = m["widths"]
        ds_str = " ".join(f"{d:>5.2f}" if d is not None else "  N/A"
                          for d in m["ds_at"])
        print(f"{sweeps:>7} {m['N4']:>5} {m['N41']:>5} {min(widths):>6} "
              f"{sum(widths)/len(widths):>7.1f} {max(widths):>6} {ds_str} "
              f"{dt:>7.2f}", flush=True)

    m = measure(st, args.n_walks, args.max_sigma, sigma_targets)
    if m:
        log_row(0, m, 0.0)

    for next_chk in checkpoints[1:]:
        batch = next_chk - sweeps_done
        t0 = time.time()
        cdt.sweep(batch)
        dt = time.time() - t0
        sweeps_done = next_chk
        m = measure(st, args.n_walks, args.max_sigma, sigma_targets)
        if m is None:
            print(f"{sweeps_done:>7} (empty)", flush=True)
        else:
            log_row(sweeps_done, m, dt)


if __name__ == "__main__":
    main()
