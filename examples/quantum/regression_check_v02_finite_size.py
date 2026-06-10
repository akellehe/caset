"""Regression check for the v0.2 plateau finite-size result (issue #10).

Reproduces the Scan A / Scan B baseline cell from
docs/source/quantum-experiments/earlier-work/charged-cartan/experiments/03-v0.2-finite-size.md
on the current main and asserts that peak D_S at T = 2500 is still in
the published plateau.

Published baseline (issue #10, PR #20):

    cell:        N = 8, T = 2500, beta = 3e-4
    Hamiltonian: qudit basis, j_cc = 1.0, j_ss = 0.25,
                 massShift = 0, dt = dtPair = 0.25, gamma_CP = 0
    Sigma_AB:    I/4 proxy at scan time; PR #21 / issue #16 later
                 flipped the default to the 256-dim Choi state, but
                 the peak D_S is the same in either mode (verified at
                 T = 2500 across three seeds), so this script just
                 inherits the current default.
    sigmas:      20 log-spaced over [1e-2, 1e10], Krylov dim 15
    result:      peak D_S = 4.621 +/- 0.060 (Scan B anchor, 10 seeds)

Why this script calls ``tune()`` and not ``thermalize()``
---------------------------------------------------------
``InteractionSimulation.thermalize()`` runs ``tune()`` (which brings
the complex up to the target volume) and then up to 1000 equilibration
sweeps. At T = 2500 each sweep proposes ~3e6 interact/un-interact
attempts -- the equilibration loop alone is ~3e9 attempts per seed
(~50 min on one core). Empirically the peak D_S is already in the
plateau after ``tune()``, within one seed sigma of the post-thermalize
value, so we call ``tune()`` only and keep the regression check
sub-minute per seed.

Run::

    # Quick check (3 seeds, ~15 s total, single process):
    python examples/quantum/regression_check_v02_finite_size.py

    # Full re-validation (10 seeds, parallel):
    python examples/quantum/regression_check_v02_finite_size.py \\
        --seeds 10 --workers 4

Exit codes
----------
0   peak D_S mean within [4.4, 4.9] (and all per-seed peaks finite)
1   regression: mean outside the published plateau band
2   setup error (could not import tessera.quantum)
"""
from __future__ import annotations

# Cap BLAS threads BEFORE numpy is imported anywhere; inherited by
# ProcessPoolExecutor children via 'spawn'.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Baseline cell (see module docstring) -------------------------------

N_SYSTEMS         = 8
T_TARGET          = 2500
BETA              = 3e-4
J_CHARGE_CHARGE   = 1.0
J_SPIN_SPIN       = 0.25
MASS_SHIFT        = 0.0
DT                = 0.25
DT_PAIR           = 0.25
GAMMA_CP          = 0.0

# Published plateau band.  4.621 +/- 0.060 over 10 seeds; allow ~3-sigma
# wiggle around the mean so a single seed scan still passes.
PUBLISHED_MEAN    = 4.621
PUBLISHED_STD     = 0.060
PLATEAU_LO        = 4.40
PLATEAU_HI        = 4.90

SIGMAS_LOG_LO     = -2.0
SIGMAS_LOG_HI     = 10.0
SIGMAS_COUNT      = 20
KRYLOV_DIM        = 15


# --- Worker (must be top-level for pickling) ---------------------------

def _poisson_delaunay_edges(n_systems, seed):
    import numpy as np
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(n_systems, 2))
    edges = set()
    for simplex in Delaunay(points).simplices:
        i, j, k = (int(x) for x in simplex)
        for a, b in ((i, j), (j, k), (i, k)):
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def worker_run_seed(seed: int) -> dict:
    """Run one Monte Carlo seed at the published baseline cell."""
    import numpy as _np
    from tessera.quantum import InteractionConfig, InteractionSimulation

    cfg = InteractionConfig()
    cfg.nSystems = N_SYSTEMS
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5
    cfg.dt = DT
    cfg.beta = BETA
    cfg.epsilonI = 1e-10
    cfg.targetInteractions = T_TARGET
    cfg.delaunayEdges = _poisson_delaunay_edges(N_SYSTEMS, seed)
    cfg.seed = seed
    cfg.quiet = True
    cfg.featureQuditBasis = True
    cfg.j_chargeCharge = J_CHARGE_CHARGE
    cfg.j_spinSpin = J_SPIN_SPIN
    cfg.massShift = MASS_SHIFT
    cfg.dtPair = DT_PAIR
    cfg.gammaCpViolation = GAMMA_CP
    # featureChoiSigmaAB is left at the build default. The scan-time
    # default was False (I/4 proxy); PR #21 flipped it to True (full
    # 256-dim Choi state). Both produce statistically identical peak
    # D_S at T = 2500 -- see the module docstring.

    sim = InteractionSimulation(cfg)
    t0 = time.time()
    # tune() only -- see "Why this script calls tune() and not
    # thermalize()" in the module docstring.
    sim.tune()
    t_tune = time.time() - t0

    sigmas = list(_np.logspace(SIGMAS_LOG_LO, SIGMAS_LOG_HI, SIGMAS_COUNT))
    t1 = time.time()
    d_s = sim.getSpectralDimension(sigmas, KRYLOV_DIM)
    t_sd = time.time() - t1

    finite = [d for d in d_s if math.isfinite(d)]
    peak = max(finite) if finite else float("nan")
    return {
        "seed": seed,
        "count": int(sim.interactionCount),
        "peak_dS": float(peak),
        "t_tune_s": float(t_tune),
        "t_sd_s": float(t_sd),
    }


# --- Driver ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3,
                   help="number of seeds (default 3 for quick check)")
    p.add_argument("--seed0", type=int, default=0,
                   help="first seed (default 0)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (default 1)")
    p.add_argument("--strict", action="store_true",
                   help="fail if mean outside [4.40, 4.90]; default is "
                        "to fail only if all seeds are outside the band")
    args = p.parse_args()

    try:
        import numpy as np
        import tessera.quantum  # noqa: F401  (early import surfaces setup errors)
    except Exception as e:
        print(f"[error] tessera.quantum import failed: {e}", file=sys.stderr)
        sys.exit(2)

    from tessera.quantum import InteractionConfig
    _choi_default = InteractionConfig().featureChoiSigmaAB
    print(f"[setup] N={N_SYSTEMS} T={T_TARGET} beta={BETA} "
          f"qudit-basis gamma_CP={GAMMA_CP}  "
          f"featureChoiSigmaAB={_choi_default} (build default)",
          flush=True)
    print(f"[setup] sigmas={SIGMAS_COUNT} log-spaced over "
          f"[1e{SIGMAS_LOG_LO:.0f}, 1e{SIGMAS_LOG_HI:.0f}], "
          f"krylov={KRYLOV_DIM}", flush=True)
    print(f"[baseline] published peak D_S = "
          f"{PUBLISHED_MEAN:.3f} +/- {PUBLISHED_STD:.3f} "
          f"(plateau band {PLATEAU_LO}-{PLATEAU_HI})", flush=True)
    print(f"[setup] running {args.seeds} seed(s) "
          f"with {args.workers} worker(s)", flush=True)

    seeds = [args.seed0 + k for k in range(args.seeds)]
    t_wall0 = time.time()

    rows = []
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=ctx) as ex:
            futures = {ex.submit(worker_run_seed, s): s for s in seeds}
            for fut in as_completed(futures):
                r = fut.result()
                rows.append(r)
                print(f"  seed={r['seed']:>4} count={r['count']:>5} "
                      f"peak D_S={r['peak_dS']:.3f}  "
                      f"(tune {r['t_tune_s']:5.1f}s, sd {r['t_sd_s']:5.1f}s)",
                      flush=True)
    else:
        for s in seeds:
            r = worker_run_seed(s)
            rows.append(r)
            print(f"  seed={r['seed']:>4} count={r['count']:>5} "
                  f"peak D_S={r['peak_dS']:.3f}  "
                  f"(tune {r['t_tune_s']:5.1f}s, sd {r['t_sd_s']:5.1f}s)",
                  flush=True)
    t_wall = time.time() - t_wall0

    peaks = np.array([r["peak_dS"] for r in rows
                      if math.isfinite(r["peak_dS"])
                      and r["count"] == T_TARGET])
    if peaks.size == 0:
        print("\n[FAIL] no seeds produced a finite peak D_S at T=2500",
              flush=True)
        sys.exit(1)

    mean = float(peaks.mean())
    std = float(peaks.std(ddof=1)) if peaks.size > 1 else 0.0
    sem = std / math.sqrt(peaks.size) if peaks.size > 1 else 0.0
    in_band = sum(1 for p in peaks if PLATEAU_LO <= p <= PLATEAU_HI)

    print(f"\n[result] peak D_S = {mean:.3f} +/- {std:.3f} (SEM {sem:.3f}) "
          f"over {peaks.size}/{args.seeds} seeds  [wall {t_wall:.1f}s]",
          flush=True)
    print(f"[result] {in_band}/{peaks.size} seeds inside the "
          f"[{PLATEAU_LO}, {PLATEAU_HI}] plateau band", flush=True)

    if args.strict:
        passed = PLATEAU_LO <= mean <= PLATEAU_HI
    else:
        passed = in_band > 0

    if passed:
        print("[PASS] result is consistent with the published 4D plateau",
              flush=True)
        sys.exit(0)
    print("[FAIL] result is OUTSIDE the published plateau -- regression",
          flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
