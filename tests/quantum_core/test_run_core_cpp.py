"""Pytest wrappers for the tests/quantum_core/ C++ binaries.

The C++ tests under tests/quantum_core/ are the source of truth for what
passes; this file only gates them behind pytest discovery so a single
``pytest tests`` invocation exercises both layers.

Unlike tests/quantum/test_quantum_executables.py (which gates the
ITensor-dependent acceptance suite behind ``TESSERA_QUANTUM=1`` and
skips when the binary is missing), these tests link only tessera_core
and so build regardless of the quantum subsystem. If the binary is not
already in the build directory we drive ``cmake --build`` to produce it
ourselves rather than skipping — the user invoked ``pytest`` to run
tests, not to discover that some weren't compiled.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _candidate_build_dirs() -> list[Path]:
    """All build directories that might contain a built test executable.

    Mirrors the search paths in tests/quantum/test_quantum_executables.py
    so this wrapper works under any of the build flows the project
    supports: direct ``cmake -B build``, scikit-build-core's per-wheel-tag
    dir, or an ASAN dev build.
    """
    candidates: list[Path] = []
    for pattern in (
        "build",
        "build-quantum",
        "cmake-build-debug",
        "cmake-build-debug/*",
        "cmake-build-*",
        "cmake-build-*/*",
        "build/*",
    ):
        candidates.extend(REPO_ROOT.glob(pattern))
    # De-dup while preserving order so the most specific paths win.
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c.is_dir() and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _find_executable(name: str) -> Path | None:
    for build_dir in _candidate_build_dirs():
        exe = build_dir / name
        if exe.is_file():
            return exe
    return None


def _try_build(name: str) -> Path | None:
    """Attempt to build the named target, returning the executable path
    on success or ``None`` if no build dir is usable.

    Picks the first existing build directory with a ``CMakeCache.txt``
    (which indicates ``cmake -B`` has been run) and runs
    ``cmake --build <dir> --target <name>``. If no such directory is
    found, returns None so the caller can skip the test cleanly.
    """
    for build_dir in _candidate_build_dirs():
        if not (build_dir / "CMakeCache.txt").is_file():
            continue
        try:
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", name,
                 "--parallel"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        exe = build_dir / name
        if exe.is_file():
            return exe
    return None


class TestQuantumCoreExecutables(unittest.TestCase):
    """Run each tests/quantum_core/*.cpp binary and check the exit code.

    The C++ side prints ``ALL PASSED`` on success and ``SOME FAILED``
    on any assertion failure; we check both the exit code (0) and that
    the success marker is present so a silent crash on stdout doesn't
    masquerade as a pass.
    """

    def _run_cpp(self, exe_name: str, timeout: int = 60) -> None:
        exe = _find_executable(exe_name)
        if exe is None:
            exe = _try_build(exe_name)
        if exe is None:
            self.skipTest(
                f"{exe_name} not found and no buildable cmake directory "
                f"is available. Run `cmake -B build -S .` from {REPO_ROOT} "
                "first, or set up scikit-build-core via "
                "`pip install -e .` so conftest.py provisions the build dir."
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
            msg=(
                f"{exe_name} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            ),
        )
        self.assertIn(
            "ALL PASSED",
            result.stdout,
            msg=(
                f"{exe_name} did not emit the 'ALL PASSED' success marker.\n"
                f"stdout:\n{result.stdout}"
            ),
        )

    def test_quantum_state_core(self) -> None:
        """QuantumState API + invariants: constructors, factories
        (maximallyMixed / computationalBasis / randomMixed), entropy
        and purity reference values, isLocallyPure boundary, Hermitian
        / positive / unit-trace validators.
        """
        self._run_cpp("test_quantum_state_core", timeout=60)

    def test_koashi_imoto_core(self) -> None:
        """Koashi–Imoto decomposition against hand-calculated values:
        partial traces on Bell / product, mutual information of
        product (0), Bell (2 log 2), classical (log 2), I/4 (0);
        conditional B-states; KI on pure product / mixed product /
        Bell / classical / I/4 / Bell ⊗ I_X (mixed L/R); bitwise
        reproducibility; descending block-weight ordering.
        """
        self._run_cpp("test_koashi_imoto_core", timeout=60)
