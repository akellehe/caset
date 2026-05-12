"""Head-to-head comparison: does raising max_build help D_S?
And does raising target N4 help?

Each scenario runs a single config with the same compute budget.
We measure D_S at fixed sigmas.

Scenarios:
  A: max_build=1600, target=8000   (current default ratio)
  B: max_build=8000, target=8000   (init == target — no growth needed)
  C: max_build=1600, target=20000  (5x bigger N4 budget)

A and C share the "narrow start" but differ in target.  B has
"wide start" but small target.  C has "narrow start" and big target.
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


def run(label, max_build, target, n_therm, max_sigma, n_walks, seed):
    np.random.seed(seed)
    sig_obj = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig_obj)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    initial = min(target, max_build)  # build at most max_build simplices
    st.build(initial)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    t0 = time.time()
    cdt.tune()
    t_tune = time.time() - t0

    t0 = time.time()
    cdt.sweep(n_therm)
    t_therm = time.time() - t0

    sigma_targets = [4, 12, 30, 60]
    T, N = build_T(st)
    if T is None or N < 2:
        print(f"[{label}] empty triangulation")
        return
    starts = np.random.choice(N, size=min(n_walks, N), replace=False)
    rp = diffuse(T, starts, max_sigma).mean(axis=0)
    ds = spectral_dim_at(rp, sigma_targets)
    widths = slice_widths(st)
    n_layers = len(widths)
    n4 = st.getSimplexCount()
    n41 = st.getN41()

    print(f"\n=== {label} ===")
    print(f"  setup: max_build={max_build} target={target} n_therm={n_therm}")
    print(f"  build: initial={initial} simplices, {n_layers} time layers (T)")
    print(f"  final: N4={n4}  N41={n41}  layers={n_layers}")
    print(f"  slice widths: min={min(widths)} mean={sum(widths)/len(widths):.1f} "
          f"max={max(widths)}")
    ds_str = "  ".join(f"DS@{s}={d:.2f}" if d is not None else f"DS@{s}=N/A"
                       for s, d in zip(sigma_targets, ds))
    print(f"  {ds_str}")
    print(f"  timing: tune={t_tune:.1f}s therm={t_therm:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--n-therm", type=int, default=400)
    ap.add_argument("--max-sigma", type=int, default=80)
    ap.add_argument("--n-walks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    scenarios = {
        "A": dict(max_build=1600,  target=8000),   # current default ratio
        "B": dict(max_build=8000,  target=8000),   # init == target
        "C": dict(max_build=1600,  target=20000),  # bigger N4
    }
    if args.scenario == "all":
        keys = ["A", "B", "C"]
    else:
        keys = [args.scenario]
    for k in keys:
        s = scenarios[k]
        run(k, s["max_build"], s["target"], args.n_therm,
            args.max_sigma, args.n_walks, args.seed)


if __name__ == "__main__":
    main()
