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

Approach: build the proton from first principles. A free colored quark does NOT
carry (confinement); a pair-created q-qbar pair is color-neutral (Sigma = 0) and
DOES carry. So we pin the carriable (neutral-pair) INPUT states as the boundary of
a merge cobordism, relax the bulk to a stationary action (delta S = 0) while
PRESERVING each input's residual, and read the EMERGENT result (its color charge,
radius = geometry extent, mass = curvature) out of the relaxed geometry
after-the-fact. Nothing is imposed on the result -- the singlet is never inserted.

STAGE 1: color-configuration realizability map (confinement + gauge invariance).
STAGE 3: first-principles merge build (build_merge_from_inputs).
  Objective Phi = ||grad S||^2 + Gamma*(r_U_A + r_U_B): the full complex Lorentzian
  dual Regge action is the MEDIATOR (||grad S||^2 -> 0 is delta S = 0, keep Im S,
  never minimize S); the per-state r_U terms hold each input carried (the rule
  "inputs evolve only while their residual is preserved"). See merge_cobordism's
  "ONLY VALID WAY" note; the merge substrate asserts manifold-validity.
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
# STAGE 3 -- FIRST-PRINCIPLES merge build: pin the carriable INPUT states as the
# fixed boundary, relax the bulk to delta S = 0 while preserving each input's
# residual, and read the EMERGENT result block after-the-fact. Nothing is imposed
# on the result -- the singlet is never inserted. (The removed relax_singlet
# imposed [1,w,w^2] on every block by fiat; that was not a build.)
# --------------------------------------------------------------------------- #
def neutral_pairs():
    """Pair-created q-qbar pairs -- the carriable inputs the proton is built from.
    A pair is color-neutral (plain sum Sigma = 0) and therefore CARRIES, unlike a
    free colored quark (Sigma != 0, confined). Three plain period vectors on the
    color holes (R-Gbar, R-Bbar, G-Bbar); SIGN_BLOCK is applied downstream."""
    return {"R-Gbar": np.array([1, -1, 0], dtype=complex),
            "R-Bbar": np.array([1, 0, -1], dtype=complex),
            "G-Bbar": np.array([0, 1, -1], dtype=complex)}


def build_merge_from_inputs(psi_a, psi_b, gamma=1e5, maxiter=200):
    """Build a merge cobordism from FIRST PRINCIPLES. Pin the two INPUT states
    psi_a, psi_b as the boundary data (their periods, held by a SEPARATE per-state
    residual r_U_A, r_U_B), relax the merge bulk to a stationary action
    (delta S = 0) while preserving those residuals, and read the EMERGENT result
    block after-the-fact. NOTHING is imposed on the result -- the singlet is never
    inserted. Inputs must be carriable (neutral, Sigma = 0); free colored quarks
    do not carry (confinement) and the relaxation strains (r_U floors, geometry
    blows up).

    Objective  Phi = ||grad S||^2 + gamma*(r_U_A + r_U_B): the full complex
    Lorentzian dual Regge action is the MEDIATOR (||grad S||^2 -> 0 is delta S = 0,
    Im S kept, never minimize S); the per-state r_U terms ARE the rule "inputs
    evolve only while their residual is preserved". The gravity gradient uses the
    exact analytic Hessian (Gauss-Newton): grad(||grad S||^2) = 2 Re(H conj(grad S)).

    psi_a, psi_b : length-3 complex PLAIN period vectors on the color holes
                   (SIGN_BLOCK is applied here). The merge substrate asserts
                   manifold-validity in its constructor (the gate the weld skipped)."""
    import stationary_action_relaxation as sar
    from scipy.optimize import minimize
    rx = sar.StationaryActionRelaxer(gamma=gamma)
    rx.VAR = sorted(rx.emap.keys())                   # all edges evolve...
    rx.x0 = np.array([rx.emap[k].getSquaredLength().real for k in rx.VAR])
    sgn = np.array(SIGN_BLOCK, dtype=complex)
    pA = [complex(z) for z in sgn * np.asarray(psi_a, dtype=complex)]
    pB = [complex(z) for z in sgn * np.asarray(psi_b, dtype=complex)]
    Ah = [list(h) for h in rx.holes[0:3]]             # color holes on input A
    Bh = [list(h) for h in rx.holes[3:6]]             # color holes on input B
    ix = [rx.EIDX[k] for k in rx.VAR]                 # VAR edges in gradient order
    G = float(gamma)

    def phi(x):
        g, _S = rx._dS_VAR(x)
        statres = float(np.vdot(g, g).real)           # ||grad S||^2 (full complex)
        # PURE GRAVITY: dualReggeAction / actionGradientExact / actionHessianExact
        # never read the matter config (no matter_ reference). The empty default
        # MatterConfiguration() is ONLY ReggeSolver's required ctor arg -- NO matter
        # is imposed or added. Matter is observed solely as curvature (the deficit
        # in S); the lone matter term in Phi is the r_U penalty (no Dirichlet source).
        H = np.array(sar.tessera.ReggeSolver(rx.st, sar.tessera.MatterConfiguration())
                     .actionHessianExact(), dtype=complex)[np.ix_(ix, ix)]
        grad_stat = 2.0 * np.real(H @ np.conj(g))     # exact Gauss-Newton step
        rUA = float(rx.es.residualForPeriods(Ah, pA))
        rUB = float(rx.es.residualForPeriods(Bh, pB))
        gA = np.asarray(rx.es.residualForPeriodsGradient(Ah, pA), float)
        gB = np.asarray(rx.es.residualForPeriodsGradient(Bh, pB), float)
        drU = np.array([gA[rx.cidx1[k]] + gB[rx.cidx1[k]] for k in rx.VAR])
        return statres + G * (rUA + rUB), grad_stat + G * drU

    curv_ref = rx.curvature()
    res = minimize(phi, rx.x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-12})
    rx.set_var(res.x)
    g, S = rx._dS_VAR(res.x)
    curv_relax = rx.curvature()

    # EMERGENT result block (holes 6,7,8 = the R block), read AFTER relaxation --
    # not pinned, not imposed: whatever the two inputs merged into.
    P = np.array(rx.es.cyclePeriods([list(h) for h in rx.holes]),
                 dtype=complex).reshape(rx.dim, 9)
    result = P[0, 6:9]
    spatial = res.x[res.x > 0]
    return {"n_var": len(rx.VAR), "dim": rx.dim, "iters": int(res.nit),
            "success": bool(res.success), "statres": float(np.vdot(g, g).real),
            "S": complex(S), "result": result, "sigma_R": complex(result.sum()),
            "rU_A": float(rx.es.residualForPeriods(Ah, pA)),
            "rU_B": float(rx.es.residualForPeriods(Bh, pB)),
            "radius_rms": float(np.sqrt(np.mean(spatial))) if spatial.size else 0.0,
            "l2_min": float(res.x.min()), "l2_max": float(res.x.max()),
            "n_spatial": int(spatial.size), "n_timelike": int((res.x < 0).sum()),
            "total_def_relax": float(sum(curv_relax[d][0] for d in curv_relax)),
            "total_def_ref": float(sum(curv_ref[d][0] for d in curv_ref))}


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


def _print_stage3(out, label_a, label_b):
    print(f"\nSTAGE 3 -- FIRST-PRINCIPLES merge: inputs '{label_a}' (x) '{label_b}' "
          f"pinned as boundary; bulk relaxed to delta S = 0; result EMERGENT")
    print(f"  variable edges = {out['n_var']}, register dim = {out['dim']}, "
          f"iters = {out['iters']}, success = {out['success']}")
    print(f"  mediator ||grad S||^2 = {out['statres']:.4f}  (delta S -> 0; "
          f"S = {out['S'].real:.3f}{out['S'].imag:+.3f}i, Im kept)")
    print(f"  inputs carried? r_U_A = {out['rU_A']:.2e}, r_U_B = {out['rU_B']:.2e} "
          f"(residual preserved => carriable)")
    print("  EMERGENT result periods (read after-the-fact, NOT imposed):")
    print("      [" + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}i" for z in out['result'])
          + "]")
    print(f"      color leak |Sigma_R| = {abs(out['sigma_R']):.3e} "
          f"({'singlet (neutral)' if abs(out['sigma_R']) < 1e-3 else 'NOT a singlet'})")
    print(f"  geometry: {out['n_spatial']} spacelike + {out['n_timelike']} timelike, "
          f"l^2 in [{out['l2_min']:.3f}, {out['l2_max']:.3f}]")
    print(f"  RADIUS proxy (rms spacelike length): {out['radius_rms']:.4f}")
    print(f"  MASS proxy   (total Re-deficit): {out['total_def_relax']:.3f} "
          f"(ref {out['total_def_ref']:.3f})")


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--relax", action="store_true",
                    help="stage 3: first-principles merge build from neutral-pair inputs")
    ap.add_argument("--gamma", type=float, default=1e5)
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    sgr = _register_path()
    reg = sgr.Register()
    _print_stage1(sgr, reg)

    if args.relax:
        (la, pa), (lb, pb), *_ = neutral_pairs().items()
        out = build_merge_from_inputs(pa, pb, gamma=args.gamma, maxiter=args.maxiter)
        _print_stage3(out, la, lb)
    else:
        print("\n(stage 3 first-principles merge: rerun with --relax)")


if __name__ == "__main__":
    _main()
