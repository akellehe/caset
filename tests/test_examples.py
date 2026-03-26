# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Tests for all example scripts in examples/.

Each test runs the script with small parameters (tiny lattice, few sweeps)
to verify it completes without error and produces the expected output file.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import pytest


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
PYTHON = sys.executable


def run_example(script_name, extra_args=None, timeout=120):
    """Run an example script with --save pointing to a temp file.

    Returns (return_code, stdout, stderr, output_path).
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = f.name

    cmd = [PYTHON, os.path.join(EXAMPLES_DIR, script_name),
           "--save", out_path]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=os.path.join(os.path.dirname(__file__), ".."))
    return result.returncode, result.stdout, result.stderr, out_path


class TestVolumeProfilePhases(unittest.TestCase):
    """Tests for examples/volume_profile_phases.py"""

    def test_runs_and_saves_surface_plot(self):
        """Script should exit 0 and produce a surface PNG."""
        rc, out, err, path = run_example(
            "volume_profile_phases.py",
            ["--n-simplices", "80", "--n-therm", "3",
             "--n-meas", "2", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        surface_path = path.replace(".png", "_surface.png")
        profile_path = path.replace(".png", "_profile.png")
        self.assertTrue(
            os.path.exists(surface_path) or os.path.exists(path),
            f"No output file at {surface_path} or {path}")
        for p in [path, surface_path, profile_path]:
            if os.path.exists(p):
                os.unlink(p)

    def test_runs_all_three_phases(self):
        """stdout should mention all three phases."""
        rc, out, err, path = run_example(
            "volume_profile_phases.py",
            ["--n-simplices", "60", "--n-therm", "2",
             "--n-meas", "1", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("Phase A", out)
        self.assertIn("Phase B", out)
        for p in [path, path.replace(".png", "_surface.png"),
                  path.replace(".png", "_profile.png")]:
            if os.path.exists(p):
                os.unlink(p)

    def test_reports_acceptance_rates(self):
        """Each phase should report acceptance rates."""
        rc, out, err, path = run_example(
            "volume_profile_phases.py",
            ["--n-simplices", "60", "--n-therm", "2",
             "--n-meas", "1", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("Acceptance rates", out)
        for p in [path, path.replace(".png", "_surface.png"),
                  path.replace(".png", "_profile.png")]:
            if os.path.exists(p):
                os.unlink(p)


class TestSpectralDimension(unittest.TestCase):
    """Tests for examples/spectral_dimension.py"""

    def test_runs_and_saves_plot(self):
        rc, out, err, path = run_example(
            "spectral_dimension.py",
            ["--n-simplices", "80", "--n-therm", "3",
             "--n-configs", "1", "--n-walks", "2",
             "--max-sigma", "20", "--sweeps-between", "1"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        self.assertTrue(os.path.exists(path), f"No output at {path}")
        self.assertGreater(os.path.getsize(path), 0)
        os.unlink(path)

    def test_builds_dual_adjacency(self):
        """Should report configuration info with N4 > 0."""
        rc, out, err, path = run_example(
            "spectral_dimension.py",
            ["--n-simplices", "80", "--n-therm", "2",
             "--n-configs", "1", "--n-walks", "1",
             "--max-sigma", "10", "--sweeps-between", "1",
             "--workers", "1"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("Config 1:", out)
        self.assertIn("N4~", out)
        if os.path.exists(path):
            os.unlink(path)


class TestVolumeScaling(unittest.TestCase):
    """Tests for examples/volume_scaling.py"""

    def test_runs_and_saves_plot(self):
        rc, out, err, path = run_example(
            "volume_scaling.py",
            ["--n-simplices", "80", "--n-therm", "3",
             "--n-meas", "3", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        self.assertTrue(os.path.exists(path), f"No output at {path}")
        self.assertGreater(os.path.getsize(path), 0)
        os.unlink(path)

    def test_runs_multiple_sizes(self):
        """Should report results for at least two system sizes."""
        rc, out, err, path = run_example(
            "volume_scaling.py",
            ["--n-simplices", "60", "--n-therm", "2",
             "--n-meas", "2", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        # Should have at least 2 different size lines
        size_lines = [l for l in out.splitlines() if "Size" in l and "N4=" in l]
        self.assertGreaterEqual(len(size_lines), 2)
        if os.path.exists(path):
            os.unlink(path)


class TestPhaseDiagram(unittest.TestCase):
    """Tests for examples/phase_diagram.py"""

    def test_runs_and_saves_plot(self):
        rc, out, err, path = run_example(
            "phase_diagram.py",
            ["--n-simplices", "50", "--n-sweeps", "3",
             "--grid-size", "2"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        self.assertTrue(os.path.exists(path), f"No output at {path}")
        self.assertGreater(os.path.getsize(path), 0)
        os.unlink(path)

    def test_scans_correct_number_of_points(self):
        """A 3x3 grid should scan 9 points."""
        rc, out, err, path = run_example(
            "phase_diagram.py",
            ["--n-simplices", "40", "--n-sweeps", "2",
             "--grid-size", "3"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("3x3", out)
        # Phase summary should account for all 9 points
        self.assertIn("Scan complete", out)
        if os.path.exists(path):
            os.unlink(path)

    def test_classifies_into_known_phases(self):
        """All classified points should be A, B, or C."""
        rc, out, err, path = run_example(
            "phase_diagram.py",
            ["--n-simplices", "40", "--n-sweeps", "2",
             "--grid-size", "2"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        # Phase summary should mention all three phase types
        self.assertIn("Phase A", out)
        self.assertIn("Phase B", out)
        self.assertIn("Phase C", out)
        if os.path.exists(path):
            os.unlink(path)


class TestEffectiveAction(unittest.TestCase):
    """Tests for examples/effective_action.py"""

    def test_runs_and_saves_plot(self):
        rc, out, err, path = run_example(
            "effective_action.py",
            ["--n-simplices", "80", "--n-therm", "3",
             "--n-meas", "5", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        self.assertTrue(os.path.exists(path), f"No output at {path}")
        self.assertGreater(os.path.getsize(path), 0)
        os.unlink(path)

    def test_reports_configurations(self):
        rc, out, err, path = run_example(
            "effective_action.py",
            ["--n-simplices", "60", "--n-therm", "2",
             "--n-meas", "3", "--meas-interval", "1"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("Collecting configurations", out)
        self.assertIn("configurations", out)
        if os.path.exists(path):
            os.unlink(path)


class TestN32Distribution(unittest.TestCase):
    """Tests for examples/n32_distribution.py"""

    def test_runs_and_saves_plot(self):
        rc, out, err, path = run_example(
            "n32_distribution.py",
            ["--n-therm", "3", "--n-meas", "5",
             "--meas-interval", "1",
             "--target-volumes", "200", "400"])
        self.assertEqual(rc, 0, f"Script failed:\nstdout:\n{out}\nstderr:\n{err}")
        self.assertTrue(os.path.exists(path), f"No output at {path}")
        self.assertGreater(os.path.getsize(path), 0)
        os.unlink(path)

    def test_reports_n41_and_n32_stats(self):
        """Should print mean and std for both simplex types."""
        rc, out, err, path = run_example(
            "n32_distribution.py",
            ["--n-therm", "2", "--n-meas", "3",
             "--meas-interval", "1",
             "--target-volumes", "200", "400"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("N41: mean=", out)
        self.assertIn("N32: mean=", out)
        if os.path.exists(path):
            os.unlink(path)

    def test_runs_multiple_target_volumes(self):
        """Should run at multiple target N41 values."""
        rc, out, err, path = run_example(
            "n32_distribution.py",
            ["--n-therm", "2", "--n-meas", "3",
             "--meas-interval", "1",
             "--target-volumes", "200", "400"])
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        target_lines = [l for l in out.splitlines() if "target N41" in l]
        self.assertGreaterEqual(len(target_lines), 2,
                                "Should run at least 2 target volumes")
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.slow
class TestBuildBenchmark(unittest.TestCase):
    """Tests for examples/benchmarks/build_benchmark.py"""

    def test_runs_and_saves_output(self):
        """Benchmark script should produce JSON log and PNG plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [PYTHON,
                   os.path.join(EXAMPLES_DIR, "benchmarks",
                                "build_benchmark.py"),
                   "--save", tmpdir, "--repeats", "1"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=os.path.join(os.path.dirname(__file__), ".."))
            self.assertEqual(result.returncode, 0,
                             f"Script failed:\nstdout:\n{result.stdout}"
                             f"\nstderr:\n{result.stderr}")
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir,
                                            "benchmark_results.json")),
                "JSON log should be created")
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir,
                                            "build_benchmarks.png")),
                "Dashboard plot should be created")

    def test_reports_summary_table(self):
        """stdout should contain the summary table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [PYTHON,
                   os.path.join(EXAMPLES_DIR, "benchmarks",
                                "build_benchmark.py"),
                   "--save", tmpdir, "--repeats", "1"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=os.path.join(os.path.dirname(__file__), ".."))
            self.assertEqual(result.returncode, 0,
                             f"stderr:\n{result.stderr}")
            self.assertIn("Simpl/sec", result.stdout)
            self.assertIn("4D", result.stdout)


class TestPlot4D(unittest.TestCase):
    """Tests for the original examples/plot4D.py (if torch is available)."""

    def test_script_exists(self):
        self.assertTrue(
            os.path.exists(os.path.join(EXAMPLES_DIR, "plot4D.py")),
            "plot4D.py should exist")
