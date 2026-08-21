# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Orientation as a local system and accepted-step square-root continuation.

The orientation covector is a global trivialization and therefore exists only
on orientable complexes.  The local system is the underlying Z2 connection: it
also exists on RP2, where its -1 loop holonomy removes the parallel zero mode.

ContentBranchTracker lifts V^2 -> V along accepted steps.  Crossing the
principal cut changes a local sheet and gauge-transforms the incident links, so
the connection spectrum and loop holonomy remain unchanged.
"""

import cmath
import math
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism


def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _path():
    return tessera.Spacetime.fromCells(1, [[0, 1], [1, 2]], 1.0, 0.0)


def _edge(spacetime, a, b):
    key = {a, b}
    for edge in spacetime.getEdgeList().toVector():
        if {edge.getSource().getId(), edge.getTarget().getId()} == key:
            return edge
    raise KeyError((a, b))


def _matrix(flat, size):
    return np.asarray(flat, dtype=complex).reshape(size, size)


class OrientationLocalSystemTest(unittest.TestCase):

    def test_orientable_system_trivializes_to_orientation_covector(self):
        spacetime = _build(tessera.SimplexBoundarySphere(2))
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        cells = chain.orientedTopSimplices()
        local = chain.orientationLocalSystem(cells)

        self.assertTrue(local.orientable)
        self.assertEqual(list(local.trivialization),
                         list(chain.orientationCovector(cells)))
        self.assertTrue(all(value == 1 for value in local.holonomies()))
        self.assertEqual(local.components, 1)

        laplacian = _matrix(local.connectionLaplacian(), len(cells))
        eigenvalues = np.linalg.eigvalsh(laplacian)
        self.assertEqual(int(np.sum(np.abs(eigenvalues) < 1e-9)), 1)

    def test_nonorientable_system_retains_obstruction_instead_of_raising(self):
        spacetime = _build(tessera.RealProjectivePlane())
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        cells = chain.orientedTopSimplices()
        local = chain.orientationLocalSystem(cells)

        self.assertFalse(local.orientable)
        self.assertIn(-1, local.holonomies())
        with self.assertRaises(RuntimeError):
            chain.orientationCovector(cells)

        # A parallel section would be a global orientation. RP2 has none, so
        # the covariant connection Laplacian has no zero mode.
        laplacian = _matrix(local.connectionLaplacian(), len(cells))
        eigenvalues = np.linalg.eigvalsh(laplacian)
        self.assertGreater(np.min(np.abs(eigenvalues)), 1e-9)

        hodge = cobordism.HodgeLaplacian(spacetime)
        np.testing.assert_allclose(
            _matrix(hodge.orientationConnectionLaplacian(), len(cells)),
            laplacian, atol=1e-12)


class ContentBranchTrackerTest(unittest.TestCase):

    def test_one_winding_returns_on_other_sheet_without_changing_holonomy(self):
        spacetime = _path()
        winding = _edge(spacetime, 0, 1)
        fixed = _edge(spacetime, 1, 2)
        fixed.setLength(complex(1.0, 0.0))

        tracker = cobordism.ContentBranchTracker()
        winding.setLength(complex(1.0, 0.0))
        first = tracker.update(spacetime)
        np.testing.assert_allclose(first.contents, [1.0, 1.0], atol=1e-12)
        initial_laplacian = _matrix(first.orientation.connectionLaplacian(), 2)
        initial_eigenvalues = np.linalg.eigvalsh(initial_laplacian)

        branch_flips = 0
        for angle in np.linspace(0.0, 2.0 * math.pi, 33)[1:]:
            squared_length = cmath.exp(1j * angle)
            # Edge stores l; Simplex.volume() takes the principal sqrt of l^2.
            winding.setLength(cmath.sqrt(squared_length))
            snapshot = tracker.update(spacetime)
            branch_flips += snapshot.principal_branch_flips

        self.assertGreaterEqual(branch_flips, 1)
        np.testing.assert_allclose(snapshot.contents, [-1.0, 1.0], atol=1e-9)

        # The sheet flip is a local gauge transformation. The incident link
        # changes sign, but orientability, Wilson-loop class, and spectrum do not.
        self.assertTrue(snapshot.orientation.orientable)
        self.assertEqual(snapshot.orientation.transitions[0].transport, -1)
        self.assertEqual(snapshot.orientation.transitions[0].holonomy, 1)
        final_laplacian = _matrix(snapshot.orientation.connectionLaplacian(), 2)
        np.testing.assert_allclose(np.linalg.eigvalsh(final_laplacian),
                                   initial_eigenvalues, atol=1e-12)
        np.testing.assert_allclose(final_laplacian,
                                   np.diag([-1.0, 1.0]) @ initial_laplacian
                                   @ np.diag([-1.0, 1.0]), atol=1e-12)

    def test_hodge_exposes_canonical_oriented_content_section(self):
        spacetime = _path()
        hodge = cobordism.HodgeLaplacian(spacetime)
        tracker = cobordism.ContentBranchTracker()
        snapshot = tracker.update(spacetime)
        np.testing.assert_allclose(hodge.orientationContentSection(),
                                   snapshot.contents, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
