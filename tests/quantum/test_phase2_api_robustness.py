"""Phase 2 robustness and API surface tests for caset.quantum.

These tests exercise the Python API beyond the Phase 1 reference-value
matching covered in test_phase2_compute_ground_state.py — they verify
validation errors, variational descent, reproducibility, the
conserve_qns flag, L0 dependence, repr formatting, and analytic limits
through the Python boundary.

Skip cleanly when caset was built without CASET_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

try:
    from caset.quantum import (
        QuantumConfig,
        GroundStateResult,
        compute_ground_state,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _basic_config(N: int = 4, m: float = 0.0, g: float = 1.0,
                  L0: float = 0.0, max_bond_dim: int = 32,
                  n_sweeps: int = 8) -> "QuantumConfig":
    cfg = QuantumConfig()
    cfg.N = N
    cfg.a = 1.0
    cfg.g = g
    cfg.m = m
    cfg.L0 = L0
    cfg.max_bond_dim = max_bond_dim
    cfg.n_sweeps = n_sweeps
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestQuantumConfigDefaults(unittest.TestCase):
    """Default-constructed QuantumConfig should have documented defaults."""

    def test_defaults(self) -> None:
        cfg = QuantumConfig()
        self.assertEqual(cfg.N, 0)         # must be set by user; 0 is invalid
        self.assertEqual(cfg.a, 1.0)
        self.assertEqual(cfg.m, 0.0)
        self.assertEqual(cfg.g, 1.0)
        self.assertEqual(cfg.L0, 0.0)
        self.assertEqual(cfg.max_bond_dim, 100)
        self.assertEqual(cfg.n_sweeps, 12)
        self.assertEqual(cfg.cutoff, 1e-12)
        self.assertEqual(cfg.krylov_dim, 4)
        self.assertTrue(cfg.quiet)
        self.assertTrue(cfg.conserve_qns)

    def test_repr_contains_key_fields(self) -> None:
        cfg = _basic_config(N=8, m=0.25, g=1.0, L0=0.5)
        text = repr(cfg)
        self.assertIn("QuantumConfig", text)
        self.assertIn("N=8", text)
        # Floats are formatted with std::to_string (default 6 digits) — just
        # check the parameter names appear; value formatting is incidental.
        self.assertIn("m=", text)
        self.assertIn("g=", text)
        self.assertIn("L0=", text)
        self.assertIn("max_bond_dim=", text)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestValidation(unittest.TestCase):
    """compute_ground_state should reject configs the C++ side can't build."""

    def test_default_construction_rejected(self) -> None:
        # N=0 from default-constructed config.
        cfg = QuantumConfig()
        with self.assertRaises(Exception):
            compute_ground_state(cfg)

    def test_N_too_small(self) -> None:
        cfg = _basic_config(N=1)
        with self.assertRaises(Exception):
            compute_ground_state(cfg)

    def test_negative_lattice_spacing(self) -> None:
        cfg = _basic_config(N=4)
        cfg.a = -1.0
        with self.assertRaises(Exception):
            compute_ground_state(cfg)

    def test_zero_lattice_spacing(self) -> None:
        cfg = _basic_config(N=4)
        cfg.a = 0.0
        with self.assertRaises(Exception):
            compute_ground_state(cfg)

    def test_zero_g_is_allowed(self) -> None:
        # g=0 is the free-Dirac limit; documented as allowed.
        cfg = _basic_config(N=4, m=0.0, g=0.0)
        result = compute_ground_state(cfg)
        # Free-fermion half-filled GS at N=4, a=1: Σ_{j=3,4} cos(πj/5)
        e_analytic = math.cos(3 * math.pi / 5) + math.cos(4 * math.pi / 5)
        self.assertAlmostEqual(result.energy, e_analytic, places=8)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestEnergyIdentities(unittest.TestCase):
    """The result fields should satisfy basic algebraic identities."""

    def test_energy_equals_operator_plus_constant(self) -> None:
        for (N, m, L0) in [(4, 0.0, 0.0), (6, 0.125, 0.0), (8, 0.25, 0.5),
                           (10, 0.5, 1.0)]:
            cfg = _basic_config(N=N, m=m, L0=L0)
            r = compute_ground_state(cfg)
            self.assertAlmostEqual(
                r.energy, r.operator_energy + r.constant, places=12,
                msg=f"Identity violated at N={N} m={m} L0={L0}",
            )

    def test_constant_only_depends_on_g_a_L0_N(self) -> None:
        """The c-number constant doesn't depend on m — at fixed (a, g, L0, N)
        all m values should report the same `constant`."""
        cfg0 = _basic_config(N=8, m=0.0)
        cfg1 = _basic_config(N=8, m=0.5)
        cfg2 = _basic_config(N=8, m=2.0)
        c0 = compute_ground_state(cfg0).constant
        c1 = compute_ground_state(cfg1).constant
        c2 = compute_ground_state(cfg2).constant
        self.assertAlmostEqual(c0, c1, places=12)
        self.assertAlmostEqual(c1, c2, places=12)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestVariationalDescent(unittest.TestCase):
    """DMRG is a variational method: enlarging the search space (more bond
    dimension, more sweeps) can only lower (or hold) the energy."""

    def test_increasing_bond_dim_lowers_energy(self) -> None:
        cfg = _basic_config(N=10, m=0.25, max_bond_dim=4, n_sweeps=8)
        energies = []
        for D in (4, 8, 16, 32):
            cfg.max_bond_dim = D
            energies.append(compute_ground_state(cfg).operator_energy)
        # Allow tiny numerical wobble (1e-10) from sweep-to-sweep noise on
        # tightly-converged runs.
        for i in range(len(energies) - 1):
            self.assertLessEqual(
                energies[i + 1], energies[i] + 1e-10,
                msg=f"variational descent violated: {energies}",
            )

    def test_increasing_sweep_count_converges(self) -> None:
        cfg = _basic_config(N=8, m=0.0, max_bond_dim=32)
        cfg.n_sweeps = 4
        e4 = compute_ground_state(cfg).operator_energy
        cfg.n_sweeps = 16
        e16 = compute_ground_state(cfg).operator_energy
        # 16 sweeps must be at least as good as 4; agreement at small N
        # is far better than 1e-8 in practice.
        self.assertLessEqual(e16, e4 + 1e-10)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestReproducibility(unittest.TestCase):
    """Same config in → same result out (DMRG with our schedule is
    deterministic; both runs use the same Néel initial state and the same
    sweep parameters)."""

    def test_two_runs_agree(self) -> None:
        cfg = _basic_config(N=8, m=0.125, L0=0.2, max_bond_dim=32, n_sweeps=8)
        a = compute_ground_state(cfg)
        b = compute_ground_state(cfg)
        self.assertAlmostEqual(a.energy, b.energy, places=12)
        self.assertAlmostEqual(a.operator_energy, b.operator_energy, places=12)
        self.assertEqual(a.bond_dim, b.bond_dim)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestConserveQNsFlag(unittest.TestCase):
    """Toggling conserve_qns must not change the GS energy: total Sz is
    conserved by the Hamiltonian itself, so the U(1) restriction is just
    a computational convenience that pins DMRG to the right sector."""

    def test_qn_vs_no_qn_match(self) -> None:
        cfg_qn = _basic_config(N=8, m=0.125, max_bond_dim=32)
        cfg_qn.conserve_qns = True
        cfg_noqn = _basic_config(N=8, m=0.125, max_bond_dim=32)
        cfg_noqn.conserve_qns = False
        e_qn = compute_ground_state(cfg_qn).energy
        e_noqn = compute_ground_state(cfg_noqn).energy
        # With and without QN the global GS is the same physical state;
        # 1e-6 is loose enough to absorb DMRG noise on the no-QN path
        # (which has more freedom to wander before the Néel initial pins
        # it down).
        self.assertAlmostEqual(e_qn, e_noqn, places=6)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestL0Dependence(unittest.TestCase):
    """The c-number constant depends on L0 quadratically; the operator
    part depends on L0 linearly. Verify both directions."""

    def test_constant_quadratic_in_L0(self) -> None:
        # c_n = L0 + ((-1)^n - 1)/4. The constant is (g²a/2) Σ (c_n² + n/4),
        # so it's quadratic in L0. The minimum should be near L0 = 1/4
        # (compensating for the average ((-1)^n - 1)/4).
        constants = {}
        for L0 in (-0.5, 0.0, 0.25, 0.5, 1.0):
            cfg = _basic_config(N=8, L0=L0)
            constants[L0] = compute_ground_state(cfg).constant
        # The constant at L0 = 0.25 should be strictly lower than at 0 or
        # 0.5 (by symmetry around the minimum).
        self.assertLess(constants[0.25], constants[0.0])
        self.assertLess(constants[0.25], constants[0.5])
        # Going to large |L0| the constant should grow without bound.
        self.assertGreater(constants[1.0], constants[0.0])
        self.assertGreater(constants[-0.5], constants[0.0])

    def test_L0_zero_matches_no_background(self) -> None:
        # L0 = 0 explicitly should give the same result as the default,
        # which is also L0 = 0.
        cfg_default = _basic_config(N=6)
        cfg_explicit = _basic_config(N=6, L0=0.0)
        self.assertAlmostEqual(
            compute_ground_state(cfg_default).energy,
            compute_ground_state(cfg_explicit).energy,
            places=12,
        )


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestAnalyticLimits(unittest.TestCase):
    """Cross-check the Python wrapper against the same analytic limits the
    C++ test_schwinger_limits.cpp covers, but reached through the Python
    API. If a binding-layer bug shifts a coefficient, this catches it."""

    def test_free_fermion_limit(self) -> None:
        """g = 0 (free Dirac), m = 0: GS energy is the half-filled OBC
        chain free-fermion sum Σ_{j=N/2+1..N} (1/a) cos(πj/(N+1))."""
        for N in (4, 6, 8, 10):
            cfg = _basic_config(N=N, m=0.0, g=0.0, max_bond_dim=32, n_sweeps=10)
            e_dmrg = compute_ground_state(cfg).energy
            e_analytic = sum(
                math.cos(math.pi * j / (N + 1))
                for j in range(N // 2 + 1, N + 1)
            )
            self.assertAlmostEqual(
                e_dmrg, e_analytic, places=8,
                msg=f"free-fermion mismatch at N={N}: "
                    f"got {e_dmrg}, expected {e_analytic}",
            )

    def test_strong_coupling_vacuum(self) -> None:
        """m → ∞ limit: GS approaches |↑↓↑↓…⟩, with energy
        E → -mN/2 + g²aN/4 (for L0 = 0, even N)."""
        N, m, g = 6, 50.0, 1.0
        cfg = _basic_config(N=N, m=m, g=g, max_bond_dim=64, n_sweeps=14)
        e_dmrg = compute_ground_state(cfg).energy
        e_asymptotic = -m * N / 2.0 + g * g * 1.0 * N / 4.0
        # Hopping correction is O(t²/m) per bond ~ (N-1)/(m a²)
        allowed = (N - 1) / (m * 1.0) * 2.0
        self.assertLess(
            abs(e_dmrg - e_asymptotic), allowed,
            msg=f"strong-coupling limit off: dmrg={e_dmrg} asympt={e_asymptotic}",
        )


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestResultRepr(unittest.TestCase):
    """The repr of a GroundStateResult should be informative for pytest
    failure messages and notebook output."""

    def test_repr_contains_key_fields(self) -> None:
        cfg = _basic_config(N=4)
        r = compute_ground_state(cfg)
        text = repr(r)
        self.assertIn("GroundStateResult", text)
        self.assertIn("energy=", text)
        self.assertIn("bond_dim=", text)
        self.assertIn("truncation_err=", text)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestDocstrings(unittest.TestCase):
    """Make sure each public symbol carries a non-empty docstring so help()
    in a Python REPL gives the user something to read."""

    def test_module_docstring(self) -> None:
        import caset.quantum
        self.assertIsNotNone(caset.quantum.__doc__)
        self.assertGreater(len(caset.quantum.__doc__), 200)
        # The header should mention Bañuls (primary reference).
        self.assertIn("Bañuls", caset.quantum.__doc__)

    def test_quantum_config_docstring(self) -> None:
        self.assertIsNotNone(QuantumConfig.__doc__)
        self.assertIn("staggered", QuantumConfig.__doc__.lower())

    def test_ground_state_result_docstring(self) -> None:
        self.assertIsNotNone(GroundStateResult.__doc__)
        self.assertIn("energy", GroundStateResult.__doc__.lower())

    def test_compute_ground_state_docstring(self) -> None:
        self.assertIsNotNone(compute_ground_state.__doc__)
        self.assertIn("DMRG", compute_ground_state.__doc__)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestSchwingerExampleScript(unittest.TestCase):
    """Smoke-test the example script — guards against import / API drift."""

    def test_script_runs_to_completion(self) -> None:
        import subprocess
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        script = repo_root / "examples" / "quantum" / "run_schwinger.py"
        self.assertTrue(script.exists(), f"missing script: {script}")
        result = subprocess.run(
            [sys.executable, str(script),
             "--N", "6", "--max-bond-dim", "16", "--n-sweeps", "6"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        # Output should contain the table header and one data row.
        self.assertIn("E_total", result.stdout)
        self.assertIn("bond_dim", result.stdout)
