#!/usr/bin/env python3
"""Lightcone-vs-majorization scan across MajorizationPredicate variants.

Re-runs the regimes from ``lightcone_vs_majorization_cone_overflow.py``
through each of the three concrete majorization-predicate variants
(``StandardMajorization``, ``LogConcaveMajorization``,
``PeakRadialMajorization`` — see ``include/quantum/majorization.hpp``
for the contract and the bibliographic references). The objective is
to test whether the ~50% strong-falsification fraction observed in the
classical-majorization scan is sensitive to *which* notion of
majorization is used to build ≼_maj — in particular whether
restricting to log-concave spectra or to relative-to-peak entrywise
dominance qualitatively changes the agreement with the Lieb-Robinson
cone.

The output of this script is a table per regime with columns:

    variant              majKind from the report (i.e. predicate name)
    v_LR                 Lieb-Robinson velocity used to build ≼_LR
    τ(maj,LR)            Kendall-τ on the both-comparable subset
    n_maj∉LR             # ≼_maj pairs outside the LR cone
    frac                 n_maj∉LR / |≼_maj|
    n_LR∉maj             # ≼_LR pairs not related by ≼_maj
    |≼_maj|              total strict ≼_maj-pair count
    runtime              wall-clock seconds for this row

References for the variants:

* ``StandardMajorization`` — Nielsen 1999, Phys. Rev. Lett. 83, 436,
  eq. (1).  arXiv: quant-ph/9811053.
* ``LogConcaveMajorization`` — Brändén 2015, "Unimodality, log-concavity,
  real-rootedness and beyond", arXiv: 1410.6601, §1 (definition).
* ``PeakRadialMajorization`` — novel; structurally analogous to the
  L^p-norm dominance characterisations of catalytic majorization in
  Aubrun-Nechita 2008, Comm. Math. Phys. 278, 133, arXiv: 0707.0211,
  Theorem 1.1 / Proposition 2.5 (in the opposite direction).
"""
from __future__ import annotations

import time

from tessera.quantum import (
    TDVPConfig,
    SchwingerQuench,
    StandardMajorization,
    LogConcaveMajorization,
    PeakRadialMajorization,
)


def make_cfg(N: int, m_over_g: float, T: float,
             maxBondDim: int = 80,
             dt: float = 0.1, snapshotEvery: int = 5) -> TDVPConfig:
    """Identical regime configuration to lightcone_vs_majorization_cone_overflow.py.

    Keeping the parameters in lock-step with the classical scan is
    important: the only thing that changes between scripts is the
    ``MajorizationPredicate`` instance handed to
    ``SchwingerQuench.compareCausalOrders``. Comparing rows across the two scans
    therefore measures the *predicate sensitivity* of the
    causal-order agreement in isolation.
    """
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = maxBondDim
    cfg.dmrgNSweeps    = 12
    cfg.dmrgKrylovDim  = 4
    cfg.dmrgCutoff     = 1e-12
    cfg.i0 = 3; cfg.d = 3
    cfg.dt = dt; cfg.T = T
    cfg.maxBondDim = maxBondDim
    cfg.cutoff     = 1e-10
    cfg.krylovDim  = 12
    cfg.snapshotEvery = snapshotEvery
    cfg.quiet = True
    cfg.conserveQns = True
    return cfg


# Predicate instances are reused across regimes; each is stateless
# beyond its tolerance, so a single instance per variant is fine.
PREDICATES = (
    StandardMajorization(),
    LogConcaveMajorization(),
    PeakRadialMajorization(),
)


def scan_regime(label: str, cfg: TDVPConfig, vlr_values: list[float]) -> None:
    """Print the variant × v_LR agreement table for one regime."""
    print(f"\n{label}")
    print(f"  N={cfg.N}  m/g={cfg.m/cfg.g}  d={cfg.d}  "
          f"T={cfg.T}  max_bond={cfg.maxBondDim}  "
          f"snapshotEvery={cfg.snapshotEvery}")
    print(f"  {'variant':>12}  {'vLr':>5}  {'τ(maj,LR)':>10}  "
          f"{'n_maj∉LR':>10}  {'frac':>7}  "
          f"{'n_LR∉maj':>10}  {'|≼_maj|':>10}  {'runtime':>8}")
    for v in vlr_values:
        for pred in PREDICATES:
            t0 = time.time()
            r = SchwingerQuench(cfg).compareCausalOrders(vLr=v, predicate=pred)
            dt_run = time.time() - t0
            a = r.majVsLr
            n_maj_total = a.nConcordant + a.nDiscordant + a.nOnlyA
            frac = (a.nOnlyA / n_maj_total) if n_maj_total > 0 else 0.0
            print(f"  {r.majKind:>12}  {v:>5.2f}  "
                  f"{a.kendallTau:>10.4f}  "
                  f"{a.nOnlyA:>10}  {frac:>7.4f}  "
                  f"{a.nOnlyB:>10}  {n_maj_total:>10}  "
                  f"{dt_run:>7.1f}s")


if __name__ == "__main__":
    print("Lightcone vs. majorization — variant scan")
    print("==========================================")
    print("Compares ≼_LR cone-overflow statistics across three majorization")
    print("variants:")
    print("  • standard    — Nielsen 1999 PRL 83, 436 eq. (1)")
    print("  • log-concave — standard restricted to spectra with")
    print("                  λᵢ² ≥ λᵢ₋₁·λᵢ₊₁ (Brändén 2015 §1)")
    print("  • peak-radial — μᵢ/μ₁ ≤ λᵢ/λ₁ entrywise (relative-to-peak")
    print("                  dominance, strictly stronger than standard)")
    print()
    print("If the ~50% out-of-cone fraction observed under standard")
    print("majorization is an artefact of cross-shape comparisons, it")
    print("should drop sharply under log-concave (which forbids them)")
    print("or peak-radial (which strengthens the relation).")
    print()

    # We probe the two physically-bracketing values of v_LR rather than
    # the full sweep: 1.0 is the free-fermion bound (the
    # methodology-page criterion-1 reference), 16.0 is a 16× loose cone
    # used in lightcone_vs_majorization_cone_overflow.py. Anything
    # between is monotone interpolation in the standard variant; we
    # don't expect that to change for the others.
    vlrs = [1.0, 16.0]

    scan_regime("Regime A — light quark (m/g=0.5), N=10, T=1.0",
                make_cfg(N=10, m_over_g=0.5, T=1.0), vlrs)
    scan_regime("Regime B — heavy quark (m/g=5.0), N=10, T=1.0",
                make_cfg(N=10, m_over_g=5.0, T=1.0), vlrs)
    scan_regime("Regime D — light quark, N=14, T=1.0",
                make_cfg(N=14, m_over_g=0.5, T=1.0), vlrs)
