"""Probe the geometry that spectral_dimension.py actually feeds into the
diffusion process.  Goal: validate or refute the narrow-tube hypothesis.

We measure:
  * Volume profile N3(tau) — how thick is each spatial slice?
  * Spatial-vertex count per slice — a thin tube has ~5 verts/slice.
  * Dual-graph degree distribution — a quasi-1D tube has mean degree ~2.
  * Spectral dimension at sigma -> 0 and large sigma at three checkpoints:
      (0) post-build, (1) post-tune, (2) post-therm.

Run it small first to sanity-check, then larger.  Resource-friendly:
single thread, single configuration.
"""
import argparse
import os
import time

import numpy as np
from scipy import sparse

import tessera


def build_transition_matrix(st):
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


def spectral_dim(rp_avg):
    sigma = np.arange(len(rp_avg))
    valid = (sigma > 1) & (rp_avg > 0)
    s = sigma[valid].astype(float)
    p = rp_avg[valid]
    log_s = np.log(s)
    log_p = np.log(p)
    if len(log_s) < 2:
        return s, np.zeros(len(s))
    ds = np.zeros(len(log_s))
    ds[1:-1] = (log_p[2:] - log_p[:-2]) / (log_s[2:] - log_s[:-2])
    ds[0] = (log_p[1] - log_p[0]) / (log_s[1] - log_s[0])
    ds[-1] = (log_p[-1] - log_p[-2]) / (log_s[-1] - log_s[-2])
    return s, -2.0 * ds


def slice_widths(st):
    """Return the number of vertices at each time slice."""
    times = list(st.getTimeSlices())
    return {t: len(st.getVerticesAtTime(t)) for t in times}


def volume_profile(st, d=4):
    """N4(t) profile: count of (d+1)-simplices whose minimum-time vertex is at t."""
    profile = {}
    dPlus1 = d + 1
    for s in st.getSimplices():
        verts = s.getVertices()
        if len(verts) != dPlus1:
            continue
        ts = sorted(int(v.getTime()) for v in verts)
        if len(set(ts)) != 2:
            continue
        profile[ts[0]] = profile.get(ts[0], 0) + 1
    return profile


def report(label, st, max_sigma=200, n_walks=20):
    N = st.getSimplexCount()
    n41 = st.getN41()
    n32 = st.getN32()
    nverts = st.getVertexCount()
    profile = volume_profile(st)
    widths = slice_widths(st)

    T, N_dual = build_transition_matrix(st)
    if T is None:
        print(f"[{label}] empty triangulation")
        return
    deg = np.array(T.astype(bool).sum(axis=0)).ravel()
    starts = np.random.choice(N_dual, size=min(n_walks, N_dual), replace=False)
    rp = diffuse(T, starts, max_sigma).mean(axis=0)
    sig, ds = spectral_dim(rp)
    n_head = max(1, len(ds) // 5)
    n_tail = max(1, len(ds) // 5)
    ds_small = float(np.mean(ds[:n_head]))
    ds_large = float(np.mean(ds[-n_tail:]))

    if profile:
        n4_min = min(profile.values())
        n4_max = max(profile.values())
        n4_mean = sum(profile.values()) / len(profile)
        n_slabs = len(profile)
    else:
        n4_min = n4_max = n4_mean = n_slabs = 0
    if widths:
        w_min = min(widths.values())
        w_max = max(widths.values())
        w_mean = sum(widths.values()) / len(widths)
        n_layers = len(widths)
    else:
        w_min = w_max = w_mean = n_layers = 0

    print(f"\n=== {label} ===")
    print(f"  N4_total={N}  N41={n41}  N32={n32}  Nverts={nverts}")
    print(f"  Time layers={n_layers}  spatial-verts/slice min/mean/max="
          f"{w_min}/{w_mean:.1f}/{w_max}")
    print(f"  Slabs={n_slabs}  N4/slab min/mean/max="
          f"{n4_min}/{n4_mean:.1f}/{n4_max}")
    print(f"  Dual-graph degree min/mean/max="
          f"{deg.min()}/{deg.mean():.2f}/{deg.max()}")
    print(f"  D_S(small ~sigma=2..{n_head+1})={ds_small:.2f}  "
          f"D_S(large ~sigma={max_sigma-n_tail+1}..{max_sigma})={ds_large:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-simplices", type=int, default=8000)
    ap.add_argument("--max-build", type=int, default=1600,
                    help="Cap initial build at this many simplices "
                         "(80*20=1600 is current default).")
    ap.add_argument("--n-therm", type=int, default=200)
    ap.add_argument("--max-sigma", type=int, default=200)
    ap.add_argument("--n-walks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    print(f"Probe: n_simplices={args.n_simplices} max_build={args.max_build} "
          f"n_therm={args.n_therm}")

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    st.build(min(args.n_simplices, args.max_build))
    target = st.getN41() if args.n_simplices <= args.max_build else args.n_simplices // 2
    print(f"Initial build done.  N4={st.getSimplexCount()} "
          f"N41={st.getN41()}  target N41={target}")
    report("post-build", st, args.max_sigma, args.n_walks)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    t0 = time.time()
    cdt.tune()
    print(f"\nTune done in {time.time()-t0:.1f}s")
    report("post-tune", st, args.max_sigma, args.n_walks)

    t0 = time.time()
    chunk = max(1, args.n_therm // 10)
    for start in range(0, args.n_therm, chunk):
        batch = min(chunk, args.n_therm - start)
        cdt.sweep(batch)
        if (start // chunk) % 2 == 0:
            print(f"  therm progress: {start+batch}/{args.n_therm} sweeps; "
                  f"N4={st.getSimplexCount()} N41={st.getN41()}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"Thermalization done in {time.time()-t0:.1f}s")
    report("post-therm", st, args.max_sigma, args.n_walks)


if __name__ == "__main__":
    main()
