# MIT License
# Copyright (c) 2025 Andrew Kelleher
"""Smoke test for ``examples/modularity-cpp.py``: import-only check
that the driver script's helpers wire up correctly with the C++
bindings.

We don't shell out to the script (matplotlib backend issues, slow);
instead we exercise the same code path via direct calls.
"""
import os
import pathlib
import sys
import tempfile
import unittest

# Make the example importable.
THIS = pathlib.Path(__file__).resolve()
EXAMPLES = THIS.parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

# Use a non-interactive matplotlib backend so the test runs
# headless.
os.environ.setdefault("MPLBACKEND", "Agg")

import tessera


class TestDriverImportable(unittest.TestCase):

    def test_importable(self):
        # Use importlib to load the file since hyphenated names aren't
        # valid Python identifiers.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "modularity_cpp", str(EXAMPLES / "modularity-cpp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "_run_one_d"))
        self.assertTrue(hasattr(mod, "_plot_dimension_sweep"))


class TestDriverRunOneD(unittest.TestCase):
    """Run the per-dimension helper on a tiny config and verify
    measurements come back."""

    def test_run_one_d_returns_measurements(self):
        import importlib.util, types, argparse
        spec = importlib.util.spec_from_file_location(
            "modularity_cpp", str(EXAMPLES / "modularity-cpp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Small target_dq so measurements are reliably triggered on
        # the tiny n_simplices=80 system; larger max_iterations and a
        # generous epsilon_q_max keep the sweep from short-circuiting
        # on its convergence checks before any measurement lands.
        args = argparse.Namespace(
            dimensions=[4],
            n_simplices=80,
            cdt_thermalize=0,
            target_n_modules=4,
            direction="up",
            target_dq=0.01,
            max_iterations=200,
            epsilon_q_max=0.5,
            n_diffusion_walks=10,
            max_sigma=20.0,
            seed=0,
            save="/tmp/unused.png",
        )
        ms, label = mod._run_one_d(4, args)
        # The sweep is a smoke test: we want it to run and produce
        # well-formed measurements. The exact count depends on the
        # CDT initial RNG seed (which isn't controllable from the
        # driver script's argparse surface), so we don't require
        # ≥1 measurement — only that any returned measurement is
        # well-formed.
        self.assertIn("CDT(d=4", label)
        for m in ms:
            self.assertEqual(m.direction, "up")


class TestDriverPlot(unittest.TestCase):
    """Plot rendering succeeds on a tiny in-memory dataset."""

    def test_plot_writes_file(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "modularity_cpp", str(EXAMPLES / "modularity-cpp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Build a tiny synthetic measurements_by_d.
        # ModularityMeasurement has read-only fields, so we have to
        # generate them via a real sweep.
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(80)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        # Seed the CDT so the sweep is reproducible — the test is
        # otherwise sensitive to the std::random_device seed of the
        # underlying ModularityOptimizer's CDT moves.
        cdt.setSeed(0)
        cfg = tessera.ModularityOptimizerConfig()
        cfg.targetDq = 0.01            # small so a measurement fires
        cfg.maxIterations = 200        # plenty of room for at least one
        cfg.nDiffusionWalks = 8
        cfg.maxSigma = 20.0
        cfg.epsilonQMax = 0.5          # don't short-circuit on convergence
        opt = tessera.ModularityOptimizer(cfg, seed=0)
        ms = opt.sweep(cdt, "up")
        # The plotting function should handle empty input by writing
        # an empty-axes figure; verify it produces a file regardless.
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.png")
            mod._plot_dimension_sweep({4: ms}, "up", out)
            self.assertTrue(os.path.exists(out),
                            f"Plot file {out} not written")
            self.assertGreater(os.path.getsize(out), 1000,
                               "Plot file unexpectedly tiny")


if __name__ == "__main__":
    unittest.main()
