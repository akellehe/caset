# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Stage-1 correctness oracle for the State-Operation-Cobordism correspondence.

Runs the algebraic-layer checks C1-C5 from
``docs/source/quantum-experiments/state-operation-cobordism/cobordism.md`` (the n=1 layer that needs no mesh
dynamics) and prints a pass/fail table with residuals, then sweeps the U(1) flux
and writes the spectrum-vs-flux and gap-vs-flux curves.

The pieces under test, all on a Hermitian-weighted complex (an ``Edge`` carries a
real ``squaredLength`` magnitude and a U(1) ``phase``):

* C1  value = amplitude       -- quantum.ChoiJamiolkowski (map-state duality)
* C2  rank = Schmidt = conn.  -- quantum.ChoiJamiolkowski (separable vs entangled)
* C3  gauge invariance        -- cobordism.HodgeLaplacian (rephase edges)
* C4  flux lives in spectrum  -- observables.SpectralGap / HarmonicDimension
* C5  tree vs cycle           -- SpectralGap on a tree (b1=0) vs cycles (b1>=1)

Supported by C1-C5: predictions P4 (rank=Schmidt=connectivity) and P5 (flux is
gauge-invariant and visible in the spectrum); and the amplitude half of P1.

Run:  python examples/cobordism/algebraic_correspondence.py
      (use --help for options; figures default to /tmp/cobordism and are not
      committed -- attach them to the issue/PR if you want to pin a result.)
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import tessera

cob = tessera.cobordism
obs = tessera.observables
cj = tessera.quantum.ChoiJamiolkowski


# --------------------------------------------------------------------------- #
# Fixtures (Hermitian-weighted complexes)
# --------------------------------------------------------------------------- #
def _build_topology(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for s in simplices:
        st.createSimplex([verts[i] for i in s])
    return st


def _set_uniform(st, sq=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(sq)
        e.setPhase(phase)


def _triangle(phi=0.0):
    st = _build_topology(tessera.SimplexBoundarySphere(1))
    _set_uniform(st, 1.0, 0.0)
    if phi:
        st.getEdgeList().toVector()[0].setPhase(phi)  # total flux on one edge
    return st


def _path():
    st = _from_simplices(3, [(0, 1), (1, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _testbed():
    # square 00-01-11-10 + entangling diagonal 00-11 (b1 = 2).
    st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _rand_state(rng, dim=2):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return [complex(x) for x in v / np.linalg.norm(v)]


# --------------------------------------------------------------------------- #
# Checks: each returns (passed: bool, residual: float, detail: str)
# --------------------------------------------------------------------------- #
def check_c1(rng):
    """value = amplitude: <psiA|U|psiB> == <vec(U_T)|vec(U)> == Tr(U_T^H U)."""
    worst = 0.0
    for _ in range(64):
        psiA, psiB = _rand_state(rng), _rand_state(rng)
        U = [complex(x) for x in rng.normal(size=4) + 1j * rng.normal(size=4)]
        amp = cj.transitionAmplitude(psiA, U, psiB, 2, 2)
        UT = cj.transitionOperator(psiA, psiB, 2, 2)
        hs = np.vdot(np.array(cj.vectorize(UT, 2, 2)), np.array(cj.vectorize(U, 2, 2)))
        tr = np.trace(np.array(UT).reshape(2, 2).conj().T @ np.array(U).reshape(2, 2))
        worst = max(worst, abs(amp - hs), abs(amp - tr))
    return worst < 1e-12, worst, "amplitude == HS contraction == Tr(U_T^H U)"


def check_c2(rng):
    """rank = Schmidt = connectivity: U_T separable (1), cup / sigma_x entangled (2)."""
    psiA, psiB = _rand_state(rng), _rand_state(rng)
    r_ut = cj.schmidtRank(cj.transitionOperator(psiA, psiB, 2, 2), 2, 2)
    r_id = cj.schmidtRank([1, 0, 0, 1], 2, 2)
    r_sx = cj.schmidtRank([0, 1, 1, 0], 2, 2)
    cup = np.array(cj.vectorize([1, 0, 0, 1], 2, 2))      # vec(I2) = |00> + |11>
    cup_ok = np.allclose(cup, [1, 0, 0, 1], atol=1e-12)
    ok = (r_ut == 1) and (r_id == 2) and (r_sx == 2) and cup_ok
    detail = f"rank(U_T)={r_ut} (sep/disconnected), rank(I2)={r_id}, rank(sx)={r_sx} (ent/connected)"
    return ok, 0.0 if ok else 1.0, detail


def check_c3(rng):
    """gauge invariance: rephasing edges leaves spec(L) and every cycle flux fixed."""
    st = _testbed()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(float(rng.uniform(0.5, 2.0)))
        e.setPhase(float(rng.uniform(-math.pi, math.pi)))
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    cycles = ([0, 1, 2], [0, 2, 3])
    wl = tessera.WilsonLoop(st)
    by_id = {v.getId(): v for v in st.getVertexList().toVector()}
    e0 = np.array(sorted(cob.HodgeLaplacian(st).eigenvalues()))
    # Cycle flux = WilsonLoop U(1)-connection holonomy (oriented Edge.phase sum
    # around the vertex cycle, reduced mod 2pi).  Reduction cancels in the
    # dphi residual below, so the gauge-invariance check is unchanged.
    f0 = np.array([wl.evaluateU1Connection([by_id[i] for i in c]).value for c in cycles])
    alpha = {vid: float(rng.uniform(-math.pi, math.pi)) for vid in ids}
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        e.setPhase(e.getPhase() + alpha[s] - alpha[t])
    e1 = np.array(sorted(cob.HodgeLaplacian(st).eigenvalues()))
    f1 = np.array([wl.evaluateU1Connection([by_id[i] for i in c]).value for c in cycles])
    dphi = np.angle(np.exp(1j * (f1 - f0)))  # flux residual mod 2pi
    res = max(float(np.max(np.abs(e1 - e0))), float(np.max(np.abs(dphi))))
    return res < 1e-10, res, "spec(L) and both cycle fluxes invariant under vertex rephasing"


def check_c4(flux_pts):
    """flux lives in the spectrum: gap follows the Aharonov-Bohm ring formula, and
    the harmonic zero-mode is present only at zero flux."""
    worst = 0.0
    for phi in np.linspace(0.0, 2 * math.pi, flux_pts, endpoint=False):
        ring = sorted(2.0 - 2.0 * math.cos((phi + 2 * math.pi * k) / 3.0) for k in range(3))
        worst = max(worst, abs(obs.SpectralGap().compute(_triangle(phi)) - (ring[1] - ring[0])))
    hd0 = obs.HarmonicDimension().compute(_triangle(0.0))
    hdpi = obs.HarmonicDimension().compute(_triangle(math.pi))
    ok = worst < 1e-9 and hd0 == 1.0 and hdpi == 0.0
    return ok, worst, f"gap == 2-2cos((Phi+2pi k)/3); harmonic dim {int(hd0)}@0 -> {int(hdpi)}@pi"


def check_c5(rng):
    """tree vs cycle: a tree's spectrum is phase-independent (pure gauge); a cycle's
    depends on its flux."""
    gaps_tree = []
    for _ in range(8):
        st = _path()
        for e in st.getEdgeList().toVector():
            e.setPhase(float(rng.uniform(-math.pi, math.pi)))
        gaps_tree.append(obs.SpectralGap().compute(st))
    tree_var = float(np.max(gaps_tree) - np.min(gaps_tree))
    cycle_swing = abs(obs.SpectralGap().compute(_triangle(0.0))
                      - obs.SpectralGap().compute(_triangle(math.pi)))
    ok = tree_var < 1e-9 and cycle_swing > 1.0
    return ok, tree_var, f"tree gap variation {tree_var:.2e} (~0); cycle gap swing {cycle_swing:.3f} (flux-driven)"


# --------------------------------------------------------------------------- #
# Flux-sweep plots (spectrum and gap vs flux)
# --------------------------------------------------------------------------- #
def write_plots(out_dir, flux_pts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed; skipping plots -- `pip install -e ".[examples]"`)')
        return []
    phis = np.linspace(0.0, 2 * math.pi, max(flux_pts, 50))
    spectrum = np.array([sorted(cob.HodgeLaplacian(_triangle(p)).eigenvalues()) for p in phis])
    gap = np.array([obs.SpectralGap().compute(_triangle(p)) for p in phis])
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(spectrum.shape[1]):
        ax.plot(phis, spectrum[:, k], label=f"$\\lambda_{k}$")
    ax.set(xlabel="flux $\\Phi$", ylabel="eigenvalue",
           title="Triangle Hodge spectrum vs U(1) flux")
    ax.legend()
    p1 = os.path.join(out_dir, "spectrum_vs_flux.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phis, gap, color="crimson")
    ax.axvline(math.pi, ls="--", color="gray", lw=0.8)
    ax.set(xlabel="flux $\\Phi$", ylabel="spectral gap $\\lambda_1-\\lambda_0$",
           title="Interference signature: gap collapses at $\\Phi=\\pi$")
    p2 = os.path.join(out_dir, "gap_vs_flux.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)
    return paths


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--flux-points", type=int, default=25,
                    help="samples on Phi in [0,2pi) for C4 / plots (default 25).")
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="directory for the flux-sweep PNGs (default /tmp/cobordism).")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures.")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    checks = [
        ("C1  value = amplitude", lambda: check_c1(rng)),
        ("C2  rank = Schmidt = connectivity", lambda: check_c2(rng)),
        ("C3  gauge invariance", lambda: check_c3(rng)),
        ("C4  flux lives in the spectrum", lambda: check_c4(args.flux_points)),
        ("C5  tree vs cycle", lambda: check_c5(rng)),
    ]

    print("State-Operation-Cobordism correspondence -- Stage 1 (algebraic oracle)\n")
    print(f"  {'check':36} {'result':6} {'residual':>11}  detail")
    print("  " + "-" * 92)
    all_ok = True
    for name, fn in checks:
        ok, residual, detail = fn()
        all_ok &= ok
        print(f"  {name:36} {'PASS' if ok else 'FAIL':6} {residual:11.2e}  {detail}")
    print("  " + "-" * 92)
    print(f"\n  P4, P5 supported and the amplitude half of P1: "
          f"{'YES' if all_ok else 'NO -- a check FAILED'}")

    if not args.no_plot:
        print("\n  flux-sweep figures:")
        for p in write_plots(args.out, args.flux_points):
            print(f"    {p}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
