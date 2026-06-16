# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The carried register core in C++ (#286): ``HodgeLaplacian.harmonicMatrix``,
``EigenstateSynthesis.cyclePeriods`` / ``residualForPeriods``, and
``ChainComplex.endSignCovector``, each checked against an independent numpy
recomputation through the real pipeline — the same construction the register
layers (``Register``, ``RegisterL2``, ``Level1Fill``, ``Level1FillS3``) now
delegate to:

  1. **The harmonic matrix is the amplitude loop.** Row r, column c of
     ``harmonicMatrix(k)`` equals ``harmonics(k)[r].amplitudeFor(cells[c])``
     entry for entry, at k = 1 on the holed icosahedron and k = 2 on the
     holed hexagon join.
  2. **Periods carry the boundary operator's signs.** ``cyclePeriods`` equals
     the hand-rolled signed facet sums — the (a,b)+(b,c)-(a,c) circle walk at
     k = 1, the (-1)^j tetrahedron facets at k = 2 — bit for bit, and raises
     on a malformed hole or a facet that is not a current k-cell.
  3. **The verdict primitive is the lstsq-project-leak-residual pipeline.**
     ``residualForPeriods`` matches the numpy construction (minimum-norm
     least squares onto the period rows, the leak on each hole's first
     walk-order facet, the genuine residual) for carried, floored, and
     complex targets; zero iff the target lies in the carried period space.
  4. **The end sign covector is the fundamental chain's charge pattern.**
     ``endSignCovector`` annihilates every harmonic's signed periods on the
     canonical 2d and 3d registers, is independent of hole order, equivariant
     under a layer shift, and raises on non-orientable (RP^2) or pinched
     (three-coface) input.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_example(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, "..", "..", "examples", "cobordism",
                           name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_example("spectral_gate_realizability")
L2 = _load_example("l2_register_realizability")


def _holed_icosahedron():
    """The canonical 2d register bulk: the icosahedron with the three
    holonomy holes opened by surgery (k = 1, dim ker L_1 = 2)."""
    st = GATE._surface(GATE._ICO)
    es = cob.EigenstateSynthesis(st, 1)
    for hole in GATE._CLASS_HOLES:
        es.removeInteriorCell(list(hole))
    return st, es, [tuple(h) for h in GATE._CLASS_HOLES]


def _holed_hexjoin():
    """The canonical 3d register bulk: the hexagon-join S^3 with the three
    canonical tetrahedra removed (k = 2, dim ker L_2 = 2)."""
    st = L2._bulk(L2._HEXJOIN)
    es = cob.EigenstateSynthesis(st, 2)
    for hole in L2._HEXJOIN_HOLES:
        es.removeInteriorCell(list(hole))
    return st, es, [tuple(h) for h in L2._HEXJOIN_HOLES]


def _harmonic_matrix_by_loop(st, es, k):
    """The independent reading: the nested amplitudeFor loop the register
    layers used before #286."""
    cells = [tuple(int(v) for v in c) for c in es.cellSimplices()]
    harmonics = cob.HodgeLaplacian(st).harmonics(k)
    return cells, np.array(
        [[complex(h.amplitudeFor(list(c))) for c in cells] for h in harmonics],
        dtype=complex).reshape(len(harmonics), len(cells))


class HarmonicMatrixIsTheAmplitudeLoopTest(unittest.TestCase):
    def test_k1_holed_icosahedron(self):
        st, es, _holes = _holed_icosahedron()
        cells, by_loop = _harmonic_matrix_by_loop(st, es, 1)
        flat = cob.HodgeLaplacian(st).harmonicMatrix(1)
        matrix = np.asarray(flat, dtype=complex).reshape(-1, len(cells))
        self.assertEqual(matrix.shape, by_loop.shape)
        self.assertTrue(np.array_equal(matrix, by_loop))

    def test_k2_holed_hexjoin(self):
        st, es, _holes = _holed_hexjoin()
        cells, by_loop = _harmonic_matrix_by_loop(st, es, 2)
        matrix = np.asarray(cob.HodgeLaplacian(st).harmonicMatrix(2),
                            dtype=complex).reshape(-1, len(cells))
        self.assertEqual(matrix.shape, by_loop.shape)
        self.assertTrue(np.array_equal(matrix, by_loop))

    def test_closed_seed_has_an_empty_matrix(self):
        st = GATE._surface(GATE._ICO)          # closed S^2: b_1 = 0
        self.assertEqual(len(cob.HodgeLaplacian(st).harmonicMatrix(1)), 0)


class CyclePeriodsCarryBoundarySignsTest(unittest.TestCase):
    def test_k1_circle_walk(self):
        st, es, holes = _holed_icosahedron()
        cells, H = _harmonic_matrix_by_loop(st, es, 1)
        idx = {c: i for i, c in enumerate(cells)}
        expected = np.array(
            [[H[r, idx[(a, b)]] + H[r, idx[(b, c)]] - H[r, idx[(a, c)]]
              for (a, b, c) in (tuple(sorted(h)) for h in holes)]
             for r in range(H.shape[0])], dtype=complex)
        periods = np.asarray(es.cyclePeriods([list(h) for h in holes]),
                             dtype=complex).reshape(-1, len(holes))
        self.assertTrue(np.array_equal(periods, expected))

    def test_k2_signed_tet_facets(self):
        st, es, holes = _holed_hexjoin()
        cells, H = _harmonic_matrix_by_loop(st, es, 2)
        idx = {c: i for i, c in enumerate(cells)}
        expected = np.array(
            [[sum(s * H[r, idx[f]] for f, s in L2._tet_facets(h)) for h in holes]
             for r in range(H.shape[0])], dtype=complex)
        periods = np.asarray(es.cyclePeriods([list(h) for h in holes]),
                             dtype=complex).reshape(-1, len(holes))
        self.assertTrue(np.array_equal(periods, expected))

    def test_malformed_hole_raises(self):
        _st, es, _holes = _holed_icosahedron()
        with self.assertRaises(RuntimeError):
            es.cyclePeriods([[0, 1]])          # an edge is not a (k+1)-cell

    def test_unopened_cell_raises(self):
        # A face still present in the complex: its (a,b,c) facets are edges of
        # the complex, but the hole itself was never removed -- the period is
        # still defined by the cycle, so what must raise is a facet that is
        # not a k-cell at all (made-up vertices).
        _st, es, _holes = _holed_icosahedron()
        with self.assertRaises(RuntimeError):
            es.cyclePeriods([[100, 101, 102]])


class ResidualForPeriodsIsTheVerdictPipelineTest(unittest.TestCase):
    def _numpy_pipeline(self, es, holes, cells, H, P, target, leak_edges):
        coeffs, *_ = np.linalg.lstsq(P.T, target, rcond=None)
        psi = (coeffs @ H).astype(complex)
        leak = target - coeffs @ P
        idx = {c: i for i, c in enumerate(cells)}
        for q, edge in enumerate(leak_edges):
            psi[idx[edge]] += leak[q]
        return float(es.residual([complex(z) for z in psi]))

    def test_k1_carried_floored_and_complex_targets(self):
        st, es, holes = _holed_icosahedron()
        cells, H = _harmonic_matrix_by_loop(st, es, 1)
        P = np.asarray(es.cyclePeriods([list(h) for h in holes]),
                       dtype=complex).reshape(-1, len(holes))
        # the k = 1 leak edge is the hole's (a,b) -- the first of the walk
        leak_edges = [tuple(sorted(h))[:2] for h in holes]
        carried = P[0] + 0.25 * P[1]
        rng = np.random.default_rng(2862861)
        for target in (carried, np.array([1.0, 1.0, 1.0], dtype=complex),
                       rng.normal(size=3) + 1j * rng.normal(size=3)):
            with self.subTest(target=target):
                want = self._numpy_pipeline(es, holes, cells, H, P,
                                            np.asarray(target, dtype=complex),
                                            leak_edges)
                got = es.residualForPeriods([list(h) for h in holes],
                                            [complex(z) for z in target])
                self.assertLessEqual(abs(got - want),
                                     1e-9 * max(1.0, abs(want)))
        self.assertLess(
            es.residualForPeriods([list(h) for h in holes],
                                  [complex(z) for z in carried]), 1e-18)

    def test_k2_matches_and_floors(self):
        st, es, holes = _holed_hexjoin()
        cells, H = _harmonic_matrix_by_loop(st, es, 2)
        P = np.asarray(es.cyclePeriods([list(h) for h in holes]),
                       dtype=complex).reshape(-1, len(holes))
        # the k = 2 leak facet is the hole's (b,c,d) -- drop-v0, walk-first
        leak_facets = [tuple(sorted(h))[1:] for h in holes]
        carried = 0.5 * P[0] - P[1]
        floored = np.array([1.0, 1.0, 1.0], dtype=complex)
        for target in (carried, floored):
            with self.subTest(target=target):
                want = self._numpy_pipeline(es, holes, cells, H, P, target,
                                            leak_facets)
                got = es.residualForPeriods([list(h) for h in holes],
                                            [complex(z) for z in target])
                self.assertLessEqual(abs(got - want),
                                     1e-9 * max(1.0, abs(want)))
        self.assertGreater(
            es.residualForPeriods([list(h) for h in holes],
                                  [complex(z) for z in floored]), 1e-2)

    def test_length_mismatch_raises(self):
        _st, es, holes = _holed_icosahedron()
        with self.assertRaises(RuntimeError):
            es.residualForPeriods([list(h) for h in holes], [1.0 + 0j])


class EndSignCovectorTest(unittest.TestCase):
    def test_annihilates_the_carried_periods_2d_and_3d(self):
        for build in (_holed_icosahedron, _holed_hexjoin):
            with self.subTest(build=build.__name__):
                _st, es, holes = build()
                sigma = cob.ChainComplex.endSignCovector(
                    [[int(v) for v in c] for c in es.topCells()],
                    [list(h) for h in holes])
                self.assertEqual(sorted(set(abs(s) for s in sigma)), [1])
                P = np.asarray(es.cyclePeriods([list(h) for h in holes]),
                               dtype=complex).reshape(-1, len(holes))
                self.assertTrue(
                    np.allclose(P @ np.asarray(sigma, dtype=float), 0.0,
                                atol=1e-9))

    def test_hole_order_independence_and_shift_equivariance(self):
        _st, es, holes = _holed_icosahedron()
        tops = [[int(v) for v in c] for c in es.topCells()]
        sigma = cob.ChainComplex.endSignCovector(tops, [list(h) for h in holes])
        reordered = cob.ChainComplex.endSignCovector(
            tops, [list(h) for h in holes[::-1]])
        self.assertEqual(reordered, sigma[::-1])
        shifted = cob.ChainComplex.endSignCovector(
            [[v + 12 for v in c] for c in tops],
            [[v + 12 for v in h] for h in holes])
        self.assertEqual(shifted, sigma)

    def test_non_orientable_raises(self):
        # the minimal 6-vertex RP^2 (test_characteristic_python.py)
        rp2 = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
               (1, 2, 4), (2, 3, 5), (1, 3, 4), (1, 3, 5), (2, 4, 5)]
        with self.assertRaises(RuntimeError):
            cob.ChainComplex.endSignCovector(
                [list(c) for c in rp2[1:]], [list(rp2[0])])

    def test_pinched_facet_raises(self):
        with self.assertRaises(RuntimeError):
            cob.ChainComplex.endSignCovector(
                [[0, 1, 2], [0, 1, 3], [0, 1, 4]], [[0, 1, 2]])

    def test_mixed_dimension_raises(self):
        with self.assertRaises(RuntimeError):
            cob.ChainComplex.endSignCovector([[0, 1, 2, 3]], [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
