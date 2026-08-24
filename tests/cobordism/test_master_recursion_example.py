# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#803 — the master recursion driver, exercised end to end.

Per repository convention the DRIVER lives in ``examples/`` and this file
covers the correctness of the instrument: that the recursion actually
ITERATES on a real complex rather than only on hand-written matrices, in both
response modes, and that each level carries the provenance the specification
requires.

The recursion's own identities are certified in
``test_recursive_quotient_python.py``; nothing here re-derives them.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import tessera as T

cob = T.cobordism

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import master_recursion as mr  # noqa: E402


class TestMasterRecursionDriver(unittest.TestCase):
    """Two levels on a real complex, in both response modes."""

    @classmethod
    def setUpClass(cls):
        cls.st = mr.build_host(4)
        cls.supports = mr.level_zero_supports(cls.st, 1.0)

    def test_host_is_a_real_complex_not_a_matrix(self):
        self.assertGreater(len(self.st.getVertexList().toVector()), 4)
        self.assertGreater(len(self.st.getEdgeList().toVector()), 4)

    def test_level_zero_partition_is_nontrivial(self):
        # A single community leaves no interface cell and so nothing for the
        # recursion to reduce; the driver's declared resolution must split.
        self.assertGreaterEqual(len(self.supports), 2)

    def test_static_mode_iterates_two_levels_on_geometry(self):
        level = cob.RecursiveQuotient.overVertexSupports(self.st, 1,
                                                         self.supports)
        levels = [level]
        for _ in range(2):
            partition = level.childPersistentPartition()
            self.assertGreaterEqual(len(partition), 2)
            level = level.nextLevel(partition)
            levels.append(level)

        self.assertEqual([lvl.level for lvl in levels], [0, 1, 2])
        for lvl in levels[1:]:
            with self.subTest(level=lvl.level):
                self.assertEqual(lvl.levelProvenance.origin,
                                 cob.LevelOrigin.StaticResponse)
                self.assertTrue(lvl.staticReduction().certificate.holds())
        self.assertTrue(
            all(p.startswith("L1:L0:")
                for p in levels[2].coordinateProvenance))

    def test_pencil_mode_iterates_two_levels_on_geometry(self):
        level = cob.RecursiveQuotient.overVertexSupports(self.st, 1,
                                                         self.supports)
        lam = complex(0.37, 0.0)
        for _ in range(2):
            response = level.feshbach(lam, 0.0, 1.0)
            partition = cob.RecursiveQuotient.persistentPartition(
                response.response, len(response.coordinates))
            self.assertGreaterEqual(len(partition), 2)
            level = level.nextLevelAtLambda(partition, lam, 0.0, 1.0)
            self.assertEqual(level.levelProvenance.origin,
                             cob.LevelOrigin.BandPencil)
            self.assertEqual(level.levelProvenance.windowLower, 0.0)
            self.assertEqual(level.levelProvenance.windowUpper, 1.0)
        self.assertEqual(level.level, 2)

    def test_certified_bands_come_from_the_fiber_layer(self):
        # E_v is assembled from the fiber layer's certified bands, placed on
        # the level's coordinates by vertex SET.
        level = cob.RecursiveQuotient.overVertexSupports(self.st, 1,
                                                         self.supports)
        bands = mr.certified_bands(self.st, level, self.supports, 1)
        self.assertGreater(len(bands), 0)
        summary = level.certifiedFiberSum(bands)
        self.assertTrue(summary.fromCertifiedBands)
        self.assertEqual(len(summary.summandCertificates), len(bands))
        stage = level.fockStage(summary, 1 << 16)
        self.assertEqual(stage.modes, summary.effectiveRank)

    def test_driver_runs_end_to_end_in_both_modes(self):
        for mode in ("static", "pencil"):
            with self.subTest(mode=mode):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    mr.run(2, mode, complex(0.37, 0.0), (0.0, 1.0), 4, 1, 1.0,
                           1 << 16)
                output = buffer.getvalue()
                self.assertIn("level 0:", output)
                self.assertIn("level 2:", output)
                self.assertIn("Fock stage:", output)

    def test_unknown_is_reported_as_unknown_never_zero(self):
        self.assertEqual(mr.show(float("nan")), "unknown")
        self.assertEqual(mr.show(None), "unknown")
        self.assertEqual(mr.show(float("inf")), "inf")
        self.assertEqual(mr.show(0.0), "0")


if __name__ == "__main__":
    unittest.main()
