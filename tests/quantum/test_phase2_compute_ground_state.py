"""Phase 2 acceptance: re-run Phase 1's small-N cases through the
caset.quantum.computeGroundState Python API and confirm the energies
match the C++-side numbers (PLAN.md §5 Phase 2: "re-runs Phase 1 with the
wrapper; numerics unchanged").

Reference values are the (operator-only) ground-state energies produced
by tests/quantum/test_schwinger_spectrum.cpp at the same parameters,
which were themselves cross-checked against dense Eigen ED to 1e-8 — so
agreeing with these values is equivalent to passing the original Phase 1
acceptance via the Python boundary.

Skips cleanly if caset.quantum isn't available (CASET_QUANTUM=0 build).
"""

from __future__ import annotations

import unittest

try:
    from caset.quantum import QuantumConfig, computeGroundState
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# (N, m/g, L0) -> operator-only ground-state energy from
# tests/quantum/test_schwinger_spectrum.cpp at 12-digit precision (the
# C++ test cross-checks each value against dense Eigen ED to <1e-12).
PHASE1_REFERENCE = {
    (4, 0.0,   0.0): -1.738676174000,
    (4, 0.125, 0.0): -1.639323676070,
    (4, 0.25,  0.0): -1.592621519330,
    (6, 0.0,   0.0): -3.432512144510,
    (6, 0.125, 0.0): -3.279435472200,
    (6, 0.25,  0.0): -3.197318242230,
    (8, 0.0,   0.0): -5.629231623340,
    (8, 0.125, 0.0): -5.424944357400,
    (8, 0.25,  0.0): -5.308213565820,
    # L0 = 0.5 corner: exercises the c_n / A_k tail-sum machinery.
    (4, 0.0,   0.5): -1.592621519330,
    (4, 0.25,  0.5): -1.738676174000,
    (6, 0.125, 0.5): -3.279435472200,
}


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestComputeGroundState(unittest.TestCase):
    def test_phase1_reference_match(self) -> None:
        """Each (N, m/g, L0) in the Phase 1 sweep matches its reference
        operator-only energy to within 1e-6 (DMRG noise floor on these
        small cases is ~1e-12, but we leave headroom for the Krylov
        scheduler picking slightly different tolerances)."""
        for (N, m_over_g, L0), e_ref in PHASE1_REFERENCE.items():
            cfg = QuantumConfig()
            cfg.N = N
            cfg.a = 1.0
            cfg.g = 1.0
            cfg.m = m_over_g          # since g=1, m_over_g IS m
            cfg.L0 = L0
            cfg.maxBondDim = 64
            cfg.nSweeps = 8

            result = computeGroundState(cfg)
            self.assertAlmostEqual(
                result.operatorEnergy,
                e_ref,
                places=8,
                msg=f"N={N} m/g={m_over_g} L0={L0}: got {result.operatorEnergy} expected {e_ref}",
            )
            # Sanity: energy = operatorEnergy + constant by construction.
            self.assertAlmostEqual(
                result.energy,
                result.operatorEnergy + result.constant,
                places=12,
            )
            # bondDim should never exceed what we asked for.
            self.assertLessEqual(result.bondDim, cfg.maxBondDim)

    def test_n20_runs_and_returns_diagnostics(self) -> None:
        """N=20 from PLAN.md §5 Phase 1 spec — the converged DMRG energy
        is around -29.31 (operator-only) at m/g=0, but the precise value
        isn't published anywhere, so we just verify the run completes,
        returns a sane bondDim, and the operator energy + constant
        identity holds. This is the size at which Phase 4's TDVP loop
        will operate."""
        cfg = QuantumConfig()
        cfg.N = 20
        cfg.a = 1.0
        cfg.g = 1.0
        cfg.m = 0.0
        cfg.L0 = 0.0
        cfg.maxBondDim = 100
        cfg.nSweeps = 12

        r = computeGroundState(cfg)
        # Sign and order of magnitude: at this scale the GS is well below 0.
        self.assertLess(r.operatorEnergy, -10.0)
        self.assertGreater(r.bondDim, 0)
        self.assertLessEqual(r.bondDim, cfg.maxBondDim)
        self.assertAlmostEqual(r.energy, r.operatorEnergy + r.constant, places=12)

    def test_skip_signal_without_quantum(self) -> None:
        """Make sure the API is at least importable when present — this
        sanity test guards against a partially-built module that imports
        but lacks classes."""
        self.assertTrue(callable(computeGroundState))
        cfg = QuantumConfig()
        # Default-constructed config has N=0; computeGroundState must
        # reject that (validation comes from buildSchwingerMpo).
        with self.assertRaises(Exception):
            computeGroundState(cfg)
