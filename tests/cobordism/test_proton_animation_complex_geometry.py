# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Complex geometry and objective controls in proton_animation.py."""

import cmath
import importlib.util
import os
import sys
import unittest

import numpy as np

import tessera as T


_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                            "cobordism")
_EXAMPLE = os.path.join(_EXAMPLE_DIR, "proton_animation.py")


def _load_example():
    sys.path.insert(0, _EXAMPLE_DIR)
    spec = importlib.util.spec_from_file_location("proton_animation_tested",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Vertex:
    def __init__(self, identifier):
        self._identifier = identifier

    def getId(self):
        return self._identifier


class _Edge:
    def __init__(self, source, target, squared_length):
        self._source = source
        self._target = target
        self._length = cmath.sqrt(squared_length)

    def getSource(self):
        return self._source

    def getTarget(self):
        return self._target

    def getLength(self):
        return self._length


class _EdgeList:
    def __init__(self, edges):
        self._edges = edges

    def toVector(self):
        return self._edges


class _Spacetime:
    def __init__(self, squared_lengths):
        vertices = [_Vertex(index) for index in range(3)]
        self._edge_list = _EdgeList([
            _Edge(vertices[0], vertices[1], squared_lengths[0]),
            _Edge(vertices[1], vertices[2], squared_lengths[1]),
        ])

    def getEdgeList(self):
        return self._edge_list


class ProtonAnimationComplexGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = _load_example()

    def test_gram_determinant_retains_complex_squared_lengths(self):
        values = {
            (0, 1): 1.0 + 0.8j,
            (0, 2): 1.7 - 0.3j,
            (1, 2): 0.9 + 0.4j,
        }
        measured = self.example.face_gram_determinants(
            [(0, 1, 2)], lambda u, v: values[tuple(sorted((u, v)))])[2][0]
        gram = np.array([
            [values[(0, 1)],
             0.5 * (values[(0, 1)] + values[(0, 2)] - values[(1, 2)])],
            [0.5 * (values[(0, 1)] + values[(0, 2)] - values[(1, 2)]),
             values[(0, 2)]],
        ], dtype=complex)
        self.assertAlmostEqual(measured, np.linalg.det(gram), places=12)
        self.assertNotEqual(measured.imag, 0.0)

    def test_layout_uses_modulus_of_full_complex_interval(self):
        equal = self.example._mds_layout(_Spacetime([1.0 + 0.0j,
                                                      1.0 + 0.0j]))
        phased = self.example._mds_layout(_Spacetime([1.0 + 0.0j,
                                                       1.0 + 3.0j]))

        def adjacent_ratio(coords):
            d01 = np.linalg.norm(coords[0] - coords[1])
            d12 = np.linalg.norm(coords[1] - coords[2])
            return d01 / d12

        self.assertAlmostEqual(adjacent_ratio(equal), 1.0, places=10)
        expected = np.sqrt(abs(1.0 + 0.0j)) / np.sqrt(abs(1.0 + 3.0j))
        self.assertAlmostEqual(adjacent_ratio(phased), expected, places=10)

    def test_builder_applies_objective_and_phase_controls(self):
        node = self.example.build_proton_nodes(
            objective_mode="joint-stationarity", entropy_weight=2.25,
            ignore_complex_phase=True)[0][0]
        self.assertEqual(node.objective_mode,
                         T.cobordism.CobordismObjectiveMode.JointStationarity)
        self.assertEqual(node.hodge_entropy_phase_mode,
                         T.cobordism.HodgeEntropyPhaseMode.IgnoreComplexPhase)
        self.assertAlmostEqual(node.hodge_entropy_weight, 2.25)


if __name__ == "__main__":
    unittest.main()
