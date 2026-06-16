"""Build a proton from quarks via cobordisms -- emergent charge/radius/mass.

Reference is PDG/experiment (lattice-QCD-flavored, not QE). Read out emergent-first:
dimensionless ratios first, then calibrate one scale (tessera is coordinate-free).

Representation (settled; see issue #352)
----------------------------------------
- The three holonomy holes {[a],[b],[a+b]} are the three COLORS (R,G,B). The S3
  permuting them is the Weyl group of color SU(3) -- a discrete color GAUGE
  redundancy, so "which hole is which color" is unphysical; observables must be
  S3-invariant.
- The carried register is the color-neutral subspace. A bare colored quark FLOORS
  (confinement) on EVERY substrate -- single register, merge, AND transport
  (measured). Only color-neutral states are carried anywhere; there are no free
  quarks to merge.
- The carried color singlet is the plain-sum-0 state, e.g. the Z3 cube-root state
  [1, w, w^2], scored in the oriented period frame (the induced-orientation signs
  applied: reg.sign * periods / sign0,sign1 -- see reference_register_sign_convention).
- Fractions are NOT imposed; total electric charge / flavor is DEFERRED pending data.

Approach (reading 1): color is topological (the three holes of ONE neutral
register carrying the singlet); the cobordism's job is the DYNAMICS. So we relax a
merge bulk carrying the color singlet to a stationary action (delta S = 0) and read
the emergent mass (curvature) and radius (geometry extent) out of the relaxed
geometry -- never imposing them.

STAGE 1: color-configuration realizability map (confinement + gauge invariance).
STAGE 3: relax the singlet-carrying bulk to convergence; read mass/radius.
  Objective Phi = ||grad S||^2 + Gamma*r_U (StationaryActionRelaxer): EXTREMIZE the
  full complex Lorentzian action (delta S = 0), keep Im S, never minimize S; r_U is
  the realizability (carried-register) penalty.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

OMEGA = np.exp(2j * np.pi / 3.0)   # primitive cube root of unity (the Z3 color phase)
SIGN_BLOCK = (1, 1, -1)            # induced-orientation signs per register block


def _register_path():
    """Import the working gate-realizability path (needs the tessera C++ build);
    honor the 16-CPU cap before the BLAS is pulled in."""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS"):
        os.environ.setdefault(_v, "16")
    import spectral_gate_realizability as sgr
    return sgr


# --------------------------------------------------------------------------- #
# STAGE 1 -- the color-configuration realizability map (confinement structure)
# --------------------------------------------------------------------------- #
def color_configs():
    """A battery of color configurations (period vectors on the three color holes)
    spanning lone quarks, two-body, and three-body combinations -- no flavor
    assumed. Returns a list of (label, np.array of 3 complex periods)."""
    return [
        ("quark R", np.array([1, 0, 0], dtype=complex)),
        ("quark G", np.array([0, 1, 0], dtype=complex)),
        ("quark B", np.array([0, 0, 1], dtype=complex)),
        ("diquark RG (same)", np.array([1, 1, 0], dtype=complex)),
        ("meson R-Gbar", np.array([1, -1, 0], dtype=complex)),
        ("meson G-Bbar", np.array([0, 1, -1], dtype=complex)),
        ("symmetric RGB", np.array([1, 1, 1], dtype=complex)),
        ("baryon real [1,1,-2]", np.array([1, 1, -2], dtype=complex)),
        ("baryon real [2,-1,-1]", np.array([2, -1, -1], dtype=complex)),
        ("baryon Z3 [1,w,w^2]", np.array([1, OMEGA, OMEGA**2], dtype=complex)),
        ("ref _CP_IN", np.array([1.0, 0.3, -1.3], dtype=complex)),
    ]


def score(reg, periods):
    """Genuine stage-1 spectral residual of the color configuration on the
    surgery-grown register, in the oriented period frame (reg.sign applied -- the
    established convention), plus the leak |plain sum|. r -> 0 iff carried."""
    residual = float(reg.spectral_residual(reg.sign * periods))
    leak = float(abs(complex(periods.sum())))
    return residual, leak


def realizability_map(sgr=None, reg=None):
    if sgr is None:
        sgr = _register_path()
    if reg is None:
        reg = sgr.Register()
    rows = []
    for label, periods in color_configs():
        residual, leak = score(reg, periods)
        rows.append({"config": label, "leak": leak, "residual": residual,
                     "realizable": bool(residual < sgr.REALIZE)})
    return rows


def gauge_invariance(reg, periods):
    """S3 (color gauge) invariance test: every permutation of the three hole
    periods must give the same residual. Returns (residuals, spread)."""
    from itertools import permutations
    res = [float(reg.spectral_residual(reg.sign * periods[list(p)]))
           for p in permutations(range(3))]
    return res, float(np.max(res) - np.min(res))


# --------------------------------------------------------------------------- #
# STAGE 3 -- relax the singlet-carrying merge bulk; read mass/radius emergent.
# --------------------------------------------------------------------------- #
def color_singlet_periods(perm=None):
    """The carried color singlet [1, w, w^2] (plain-sum-0) in the oriented period
    frame (SIGN_BLOCK applied), the per-block known harmonic. *perm* permutes the
    three colors -- an S3 gauge transformation (observables must be invariant)."""
    singlet = np.array([1, OMEGA, OMEGA**2], dtype=complex)
    if perm is not None:
        singlet = singlet[list(perm)]
    return np.array(SIGN_BLOCK, dtype=complex) * singlet


def relax_singlet(gamma=1e3, maxiter=1000, perm=None):
    """Reading (1): relax the merge bulk carrying the color singlet (the diagonal
    known harmonic on the three blocks), extremizing the full complex action
    (delta S = 0) subject to the carried register. Reads mass (curvature / action)
    and radius (relaxed spatial geometry extent) out of the converged geometry --
    never imposed. *perm* permutes the singlet's colors (an S3 gauge check)."""
    import stationary_action_relaxation as sar
    rx = sar.StationaryActionRelaxer(gamma=gamma)
    target = np.tile(color_singlet_periods(perm), 3)   # singlet on A, B, R (diagonal)
    rx.target = target
    rx.target_c = [complex(z) for z in target]

    _P0, _g0, sr0, rU0, S0 = rx.objective(rx.x0)
    curv_ref = rx.curvature()
    res = rx.relax(maxiter=maxiter)
    _Pf, _gf, srf, rUf, Sf = rx.objective(res.x)
    curv_relax = rx.curvature()

    spatial = res.x[res.x > 0]                          # spacelike edges (l^2 > 0)
    timelike = res.x[res.x < 0]                         # timelike edges (l^2 < 0)
    radius_rms = float(np.sqrt(np.mean(spatial))) if spatial.size else 0.0
    total_def_ref = float(sum(curv_ref[d][0] for d in curv_ref))
    total_def_relax = float(sum(curv_relax[d][0] for d in curv_relax))
    return {"n_var": len(rx.VAR), "dim": rx.dim, "iters": int(res.nit),
            "success": bool(res.success), "sr0": sr0, "srf": srf,
            "rU0": rU0, "rUf": rUf, "S0": S0, "Sf": Sf,
            "radius_rms": radius_rms, "l2_min": float(res.x.min()),
            "l2_max": float(res.x.max()), "n_spatial": int(spatial.size),
            "n_timelike": int(timelike.size), "total_def_ref": total_def_ref,
            "total_def_relax": total_def_relax,
            "curv_ref": curv_ref, "curv_relax": curv_relax}


def _print_stage1(sgr, reg):
    print(f"carried register: dim ker L1 = {reg.dim}, orientation sign = "
          f"{tuple(int(s) for s in reg.sign)}, REALIZE < {sgr.REALIZE:g}\n")
    print("STAGE 1 -- color-configuration realizability map")
    print(f"{'config':>24} {'leak |Sigma|':>13} {'residual r_U':>15} {'realizable':>11}")
    for r in realizability_map(sgr, reg):
        print(f"{r['config']:>24} {r['leak']:13.6f} {r['residual']:15.3e} "
              f"{str(r['realizable']):>11}")
    print("\nGauge (S3 color-relabel) invariance:")
    for label, periods in [("baryon Z3 [1,w,w^2]",
                            np.array([1, OMEGA, OMEGA**2], dtype=complex)),
                           ("ref _CP_IN",
                            np.array([1.0, 0.3, -1.3], dtype=complex))]:
        _res, spread = gauge_invariance(reg, periods)
        print(f"  {label:>22}: spread over 6 perms = {spread:.3e} "
              f"({'INVARIANT' if spread < 1e-6 else 'GAUGE ARTIFACT'})")


def _print_stage3(out):
    print(f"\nSTAGE 3 -- relax singlet-carrying merge bulk (Phi=||grad S||^2 + "
          f"Gamma*r_U, extremize S)")
    print(f"  variable bulk edges = {out['n_var']}, register dim = {out['dim']}, "
          f"iters = {out['iters']}, success = {out['success']}")
    print(f"  ||grad S||^2: {out['sr0']:.4f} -> {out['srf']:.4f}   (delta S -> 0)")
    print(f"  r_U: {out['rU0']:.3e} -> {out['rUf']:.3e}   (register stays carried)")
    print(f"  S (action): {out['S0']:.3f} -> {out['Sf']:.3f}")
    print(f"  geometry: {out['n_spatial']} spacelike + {out['n_timelike']} timelike "
          f"edges, l^2 in [{out['l2_min']:.3f}, {out['l2_max']:.3f}]")
    print(f"  RADIUS proxy  (rms spacelike length): {out['radius_rms']:.4f}")
    print(f"  MASS proxy    (total Re-deficit): {out['total_def_ref']:.3f} (ref) "
          f"-> {out['total_def_relax']:.3f} (relaxed)")
    print("  curvature Re-deficit by shell from color holes:")
    print("    ref  :", " ".join(f"{out['curv_ref'][d][0]:+.3f}"
                                  for d in sorted(out['curv_ref'])))
    print("    relax:", " ".join(f"{out['curv_relax'][d][0]:+.3f}"
                                  for d in sorted(out['curv_relax'])))


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--relax", action="store_true",
                    help="stage 3: relax the singlet-carrying bulk to convergence")
    ap.add_argument("--gamma", type=float, default=1e3)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--perm", type=str, default=None,
                    help="S3 gauge check: permute the singlet colors, e.g. 1,2,0")
    args = ap.parse_args()

    sgr = _register_path()
    reg = sgr.Register()
    _print_stage1(sgr, reg)

    if args.relax:
        perm = tuple(int(i) for i in args.perm.split(",")) if args.perm else None
        out = relax_singlet(gamma=args.gamma, maxiter=args.maxiter, perm=perm)
        _print_stage3(out)
    else:
        print("\n(stage 3 relaxation: rerun with --relax)")


if __name__ == "__main__":
    _main()
