# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The state record and the orientation read off it (#722).

`GeometryState` writes top cells in their INTRINSIC vertex order, which is what
makes orientation recoverable; the animation's drawing dump sorts them and
cannot serve. `Orientation` derives a coherent orientation by propagation
across interior facets, which is the only route available here: the engine's
`ChainComplex.fundamentalClass` requires a CLOSED oriented d-manifold
(dim ker ∂_d = b_d = 1) and raises on the d-balls-with-boundary every proton
build produces.

The propagation is cross-checked against `fundamentalClass` on closed
manifolds, where both are defined and must agree up to the global sign that an
orientation is only ever determined up to.
"""
import importlib.util
import json
import os
import tempfile
import unittest

import tessera

cobordism = tessera.cobordism

_MODULE = os.path.join(os.path.dirname(__file__), "..", "..",
                       "examples", "cobordism", "geometry_state.py")
_spec = importlib.util.spec_from_file_location("geometry_state", _MODULE)
_geometry_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry_state)
GeometryState = _geometry_state.GeometryState
Orientation = _geometry_state.Orientation


def _build(topology):
    """A built Spacetime carrying `topology` (as test_oriented_tops does)."""
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _sphere():
    return _build(tessera.SimplexBoundarySphere(2))            # S^2, closed


def _torus():
    return _build(tessera.SimplicialProduct(
        tessera.SimplexBoundarySphere(1), tessera.SimplexBoundarySphere(1)))  # T^2


def _ball():
    return tessera.Spacetime.fromCells(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


class OrientationAgreesWithFundamentalClassTest(unittest.TestCase):
    """On a closed oriented manifold both notions exist and must agree."""

    def _assertAgrees(self, spacetime):
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        engine_cells = chain.orientedTopSimplices()
        engine_signs = list(chain.fundamentalClass())
        self.assertEqual(len(engine_signs), len(engine_cells))

        orientation = Orientation.fromSpacetime(spacetime)
        self.assertTrue(orientation.orientable)
        # Both index cells by vertex SET; the engine's own ordering is its
        # column order, so match on the set rather than on position.
        derived = {frozenset(cell): sign
                   for cell, sign in orientation.signs.items()}
        self.assertEqual(len(derived), len(engine_cells))

        ratios = set()
        for cell, engine_sign in zip(engine_cells, engine_signs):
            key = frozenset(int(v) for v in cell)
            self.assertIn(key, derived)
            self.assertIn(engine_sign, (1, -1))
            ratios.add(derived[key] * int(engine_sign))
        # Agreement up to ONE global sign: every cell relates the same way.
        self.assertEqual(len(ratios), 1, f"orientations disagree per-cell: {ratios}")

    def test_two_sphere(self):
        self._assertAgrees(_sphere())

    def test_two_torus(self):
        self._assertAgrees(_torus())


class OrientationWithBoundaryTest(unittest.TestCase):
    """The case the engine's reader cannot answer."""

    def test_fundamental_class_refuses_a_manifold_with_boundary(self):
        chain = cobordism.ChainComplex.fromSpacetime(_ball())
        with self.assertRaises(RuntimeError):
            chain.fundamentalClass()

    def test_propagation_orients_the_ball_and_finds_its_boundary(self):
        orientation = Orientation.fromSpacetime(_ball())
        self.assertTrue(orientation.orientable)
        self.assertEqual(len(orientation.signs), 1)      # one pentatope
        # Every facet of a lone top cell is boundary: a 4-simplex has 5.
        self.assertEqual(len(orientation.boundary_facets), 5)
        self.assertEqual(len(orientation.boundaryComponents()), 1)
        self.assertTrue(all(s in (1, -1)
                            for s in orientation.boundary_facets.values()))

    def test_interior_facets_are_not_boundary(self):
        # Two pentatopes glued on a shared tetrahedron: that facet is interior,
        # so 10 facets total minus the 2 copies of the shared one leaves 8.
        spacetime = tessera.Spacetime.fromCells(
            4, [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], 1.0, 0.0)
        orientation = Orientation.fromSpacetime(spacetime)
        self.assertEqual(len(orientation.signs), 2)
        self.assertEqual(len(orientation.boundary_facets), 8)
        self.assertNotIn(tuple(sorted((1, 2, 3, 4))), orientation.boundary_facets)


class GeometryStateRoundTripTest(unittest.TestCase):

    def test_write_load_rehydrate_preserves_state_and_orientation(self):
        spacetime = tessera.Spacetime.fromCells(
            4, [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], 1.0, 0.0)
        before = Orientation.fromSpacetime(spacetime)
        with tempfile.TemporaryDirectory() as directory:
            path = GeometryState.write(spacetime, os.path.join(directory, "s.json"),
                                       meta={"frame": 7})
            record = GeometryState.load(path)
            self.assertEqual(record["schema"], GeometryState.SCHEMA)
            self.assertEqual(record["frame"], 7)          # meta survives
            restored = GeometryState.rehydrate(record)
            after = Orientation.fromSpacetime(restored)
        self.assertEqual(before.signs, after.signs)
        self.assertEqual(before.boundary_facets, after.boundary_facets)

    def test_cells_keep_intrinsic_order(self):
        # A cell whose stored order is NOT ascending must round-trip unsorted,
        # since that order is what carries the orientation.
        spacetime = tessera.Spacetime.fromCells(4, [[4, 0, 3, 1, 2]], 1.0, 0.0)
        cells = GeometryState.cells(spacetime)
        self.assertEqual(len(cells), 1)
        self.assertEqual(sorted(cells[0]), [0, 1, 2, 3, 4])
        engine_order = [int(v.getId())
                        for v in spacetime.getTopSimplices()[0].getVertices()]
        self.assertEqual(cells[0], engine_order)

    def test_load_rejects_a_foreign_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "other.json")
            with open(path, "w") as handle:
                json.dump({"schema": 99, "cells": []}, handle)
            with self.assertRaises(ValueError):
                GeometryState.load(path)


if __name__ == "__main__":
    unittest.main()
