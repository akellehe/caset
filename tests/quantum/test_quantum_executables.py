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
