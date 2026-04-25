"""Pytest wrappers for the C++ quantum acceptance executables.

Each test invokes a standalone C++ binary built when ``CASET_QUANTUM=ON`` is
passed to CMake. The binaries cross-check ITensor's DMRG output against
independent dense diagonalizations (Phase 0 Heisenberg, Phase 1 Schwinger).

The C++ side is the source of truth for what passes — these wrappers only
gate the executables behind pytest discovery and check the return code.
Tests skip cleanly when the quantum subsystem is not built.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_executable(name: str) -> Path | None:
    # Search the build dirs that scikit-build-core (cmake-build-debug/<wheel_tag>/),
    # the conftest.py rebuild path, and a manual cmake invocation (build-quantum/)
    # might write to.
    patterns = [
        f"build-quantum/{name}",
        f"cmake-build-debug/{name}",
        f"cmake-build-debug/*/{name}",
        f"cmake-build-*/*/{name}",
        f"cmake-build-*/{name}",
        f"build/*/{name}",
    ]
    for p in patterns:
        for hit in REPO_ROOT.glob(p):
            if hit.is_file():
                return hit
    return None


class TestQuantumExecutables(unittest.TestCase):
    def _run(self, exe_name: str, must_contain: str, timeout: int) -> None:
        exe = _find_executable(exe_name)
        if exe is None:
            self.skipTest(
                f"{exe_name} not found — build with `cmake -DCASET_QUANTUM=ON` "
                "or set the env var CASET_QUANTUM=1 before running pytest"
            )
        result = subprocess.run(
            [str(exe)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"{exe_name} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(must_contain, result.stdout)

    def test_phase0_itensor_hello(self) -> None:
        """Phase 0: Heisenberg N=8 DMRG vs dense ED, agree to <1e-6."""
        self._run("test_itensor_hello", "PASS", timeout=60)

    def test_phase1_schwinger_spectrum(self) -> None:
        """Phase 1: SchwingerMPO vs dense ED on Sz=0 sector for N=4,6,8 ×
        m/g ∈ {0, 0.125, 0.25}, plus N=20 smoke test."""
        self._run("test_schwinger_spectrum", "ALL PASS", timeout=180)

    def test_phase1_schwinger_limits(self) -> None:
        """Phase 1: analytic-limit checks on SchwingerMPO — free-fermion
        half-filling at g=m=0, and strong-coupling vacuum at m≫g."""
        self._run("test_schwinger_limits", "ALL PASS", timeout=120)

    @pytest.mark.slow
    def test_phase1_schwinger_paper(self) -> None:
        """Phase 1 paper alignment: continuum trend ω_0 → -1/π (Bañuls
        fig. 6), vector mass gap (Bañuls table), chiral condensate
        benchmark observable. Slow — geometric scan over x with finite-N
        DMRG runs scaling as N ~ 20·√x."""
        self._run("test_schwinger_paper", "ALL PASS", timeout=600)

    def test_phase3_majorization(self) -> None:
        """Phase 3 unit tests on the majorization predicate / poset:
        reflexivity, transitivity, antisymmetry, the canonical (1,0)≻(½,½)
        strict relation, and Hasse-diagram transitive reduction on a small
        synthetic chain."""
        self._run("test_majorization", "ALL PASS", timeout=30)

    def test_phase3_schmidt_spectra(self) -> None:
        """Phase 3 Schmidt-spectrum extraction on hand-checkable MPSes:
        product |↑↑↑↑⟩ gives spectrum (1), N-qubit GHZ gives (½,½) at
        every contiguous cut, Bell |Φ⁺⟩ and singlet give (½,½) at the
        center cut, all spectra sum to 1."""
        self._run("test_schmidt_spectra", "ALL PASS", timeout=60)

    def test_phase3_majorization_poset(self) -> None:
        """Phase 3 acceptance (PLAN.md §5): full pipeline
        all_contiguous_spectra() → majorization_poset() on product, GHZ,
        and Bell-vs-product inputs, matching the three acceptance criteria."""
        self._run("test_majorization_poset", "ALL PASS", timeout=60)

    def test_phase3_schwinger_schmidt_cross_check(self) -> None:
        """Phase 3 cross-check: MPS-side Schmidt spectra against dense ED
        of the Schwinger Hamiltonian for small N — verifies the spectra
        extraction works on a non-trivial physical state, not just
        product/GHZ/Bell."""
        self._run("test_schwinger_schmidt_cross_check", "ALL PASS", timeout=120)

    @pytest.mark.slow
    def test_phase4_tdvp_string(self) -> None:
        """Phase 4 acceptance (PLAN.md §5): heavy-quark q-qbar flux-tube
        preservation under 2-site TDVP. d=5, T=5.0 (chosen to match the
        plan's "T = d·a" prescription with d odd for parity). Marked slow
        because it runs ~100 TDVP sweeps at N=14."""
        self._run("test_tdvp_string", "ALL PASS", timeout=300)

    def test_phase1_n4_hamiltonian(self) -> None:
        """PLAN.md §7 trap: 'Verify by independent sum on N=4 before
        trusting the MPO.' Independent symbolic evaluation of the
        plan's H_m + H_E formula on every N=4 basis state, compared to
        build_schwinger_dense to machine precision. Catches L_n²
        expansion errors that MPO-vs-dense would miss."""
        self._run("test_schwinger_n4_hamiltonian", "ALL PASS", timeout=30)

    def test_phase4_tdvp_vs_dense(self) -> None:
        """Cross-check TDVP integrator against full e^{-iHt} dense
        unitary evolution on the 2^N Hilbert space, for N ≤ 8 across
        heavy / light / massless regimes. Guarantees the TDVP runner
        reproduces exact dynamics to better than 1e-4 — far tighter
        than the Phase 4 flux-tube acceptance, and over genuinely
        time-dependent profiles (not just preserved plateaus)."""
        self._run("test_tdvp_vs_dense", "ALL PASS", timeout=120)
