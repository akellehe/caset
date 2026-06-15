"""Build a proton from quarks via merge cobordisms -- emergent charge/radius/mass.

Reference is PDG/experiment (lattice-QCD-flavored, not QE). Read out emergent-first:
dimensionless ratios first, then calibrate one scale (tessera is coordinate-free).

Representation (settled; see issue #352)
----------------------------------------
- The three holonomy holes {[a],[b],[a+b]} are the three COLORS (R,G,B). The S3
  permuting them is the Weyl group of color SU(3) -- a discrete color GAUGE
  redundancy, so "which hole is which color" is unphysical. Observables must be
  S3-invariant (a hard gauge-invariance test).
- The carried register is the Sigma=0 subspace of the three hole-periods
  (color-neutral). Hypothesis: a lone quark (one hole, Sigma != 0) FLOORS
  (confinement); color-singlet (Sigma=0) configurations realize (r_U -> 0).
- Fractions are NOT imposed on the classes (Z2 gives halves, not thirds; thirds
  live in the Z3 < S3 cyclic structure, the cube-roots of unity). Total electric
  charge is read out, target +1.
- What flavor is / where it lives is DEFERRED -- we gather data first.

STAGE 1 (this module): the color-configuration realizability map
----------------------------------------------------------------
Score a battery of color configurations (period vectors on the three holes) with
the register's genuine stage-1 spectral residual (the same `spectral_residual`
primitive `synthesize_state` uses), and let the data show the confinement/singlet
structure -- which configurations the register carries vs floors -- with NO flavor
assumption. Includes the S3 (gauge) invariance check.

Later stages: the q (x) q -> (diquark/meson) -> proton merge-cobordism sequence,
the StationaryActionRelaxer (Phi = ||grad S||^2 + Gamma*r_U; extremize S, keep
Im S), and the emergent charge/radius/mass readout.
"""

from __future__ import annotations

import os

import numpy as np

OMEGA = np.exp(2j * np.pi / 3.0)   # primitive cube root of unity (the Z3 color phase)


def _register_path():
    """Import the working gate-realizability path (needs the tessera C++ build);
    honor the 16-CPU cap before the BLAS is pulled in."""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS"):
        os.environ.setdefault(_v, "16")
    import spectral_gate_realizability as sgr
    return sgr


def color_configs():
    """A battery of color configurations -- period vectors on the three color
    holes -- spanning lone quarks, two-body, and three-body combinations. No
    flavor assumed; we are mapping which the register carries.

    Returns a list of (label, np.array of 3 complex periods).
    """
    return [
        # lone quark on one color hole (Sigma != 0): the confinement probe
        ("quark R", np.array([1, 0, 0], dtype=complex)),
        ("quark G", np.array([0, 1, 0], dtype=complex)),
        ("quark B", np.array([0, 0, 1], dtype=complex)),
        # two same-sign (diquark, Sigma != 0)
        ("diquark RG (same)", np.array([1, 1, 0], dtype=complex)),
        # two opposite (meson-like q qbar, Sigma = 0)
        ("meson R-Gbar", np.array([1, -1, 0], dtype=complex)),
        ("meson G-Bbar", np.array([0, 1, -1], dtype=complex)),
        # three equal (the symmetric / trivial-rep direction, Sigma != 0)
        ("symmetric RGB", np.array([1, 1, 1], dtype=complex)),
        # three-body Sigma = 0, real
        ("baryon real [1,1,-2]", np.array([1, 1, -2], dtype=complex)),
        ("baryon real [2,-1,-1]", np.array([2, -1, -1], dtype=complex)),
        # three-body Z3 singlet (the natural color singlet: cube-root phases)
        ("baryon Z3 [1,w,w^2]", np.array([1, OMEGA, OMEGA**2], dtype=complex)),
        # the generic carried input from the gate battery (Sigma = 0 reference)
        ("ref _CP_IN", np.array([1.0, 0.3, -1.3], dtype=complex)),
    ]


def score(sgr, reg, periods):
    """Genuine stage-1 spectral residual of the color configuration on the
    surgery-grown register (same primitive as synthesize_state), plus the leak
    |Sigma| (the un-carried period sum). r -> 0 iff carried (color singlet)."""
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
        residual, leak = score(sgr, reg, periods)
        rows.append({"config": label, "leak": leak, "residual": residual,
                     "realizable": bool(residual < sgr.REALIZE)})
    return rows


def gauge_invariance(sgr, reg, periods):
    """S3 (color gauge) invariance test: every permutation of the three hole
    periods must give the same residual. Returns (residuals over the 6
    permutations, spread). A nonzero spread is a gauge artifact."""
    from itertools import permutations
    res = [float(reg.spectral_residual(reg.sign * periods[list(p)]))
           for p in permutations(range(3))]
    return res, float(np.max(res) - np.min(res))


def _main():
    sgr = _register_path()
    reg = sgr.Register()
    print(f"carried register: dim ker L1 = {reg.dim}, orientation sign = "
          f"{tuple(int(s) for s in reg.sign)}, REALIZE < {sgr.REALIZE:g}\n")

    print("STAGE 1 -- color-configuration realizability map")
    print(f"{'config':>24} {'leak |Sigma|':>13} {'residual r_U':>15} {'realizable':>11}")
    for r in realizability_map(sgr, reg):
        print(f"{r['config']:>24} {r['leak']:13.6f} {r['residual']:15.3e} "
              f"{str(r['realizable']):>11}")

    print("\nGauge (S3 color-relabel) invariance -- residual must be permutation-"
          "invariant:")
    for label, periods in [("baryon Z3 [1,w,w^2]",
                            np.array([1, OMEGA, OMEGA**2], dtype=complex)),
                           ("ref _CP_IN",
                            np.array([1.0, 0.3, -1.3], dtype=complex))]:
        res, spread = gauge_invariance(sgr, reg, periods)
        print(f"  {label:>22}: spread over 6 perms = {spread:.3e} "
              f"({'INVARIANT' if spread < 1e-6 else 'GAUGE ARTIFACT'})")


if __name__ == "__main__":
    _main()
