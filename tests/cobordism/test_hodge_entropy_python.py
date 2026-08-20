# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Positive-operator Hodge entropy and its complex-z gradient."""

import cmath
import math
import unittest

import numpy as np

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    metric = T.Metric(True, T.Signature(4, T.Lorentzian))
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.getEdgeList().toVector()):
        z = complex(1.0 + 0.021 * (index % 5),
                    0.017 * (1 + (index % 3)))
        edge.setLength(cmath.sqrt(z))
    return st


def _entropy_oracle(st, degree, ignore_phase):
    flat = np.asarray(cob.HodgeLaplacian(st).laplacian(degree), complex)
    n = int(round(math.sqrt(flat.size)))
    L = flat.reshape(n, n)
    M = np.abs(L) if ignore_phase else L
    eigenvalues = np.linalg.eigvalsh(M.conj().T @ M)
    trace = float(eigenvalues.sum())
    if trace <= 0.0:
        return 0.0
    cutoff = np.finfo(float).eps * max(n, 1) * max(trace, 1.0) * 64.0
    p = eigenvalues[eigenvalues > cutoff] / trace
    return float(-(p * np.log(p)).sum())


class HodgeEntropyTest(unittest.TestCase):
    def test_positive_operator_entropy_matches_numpy(self):
        st = _complex_sphere4()
        hl = cob.HodgeLaplacian(st)
        modes = (
            (cob.HodgeEntropyPhaseMode.IncludeComplexPhase, False),
            (cob.HodgeEntropyPhaseMode.IgnoreComplexPhase, True),
        )
        for mode, ignore in modes:
            with self.subTest(mode=mode):
                measured = float(hl.spectralEntropy(3, mode))
                expected = _entropy_oracle(st, 3, ignore)
                self.assertAlmostEqual(measured, expected, places=11)
                self.assertGreaterEqual(measured, 0.0)
                self.assertLessEqual(measured, math.log(hl.laplacian(3).__len__() ** 0.5)
                                     + 1e-12)

    def test_complex_z_gradient_matches_two_axis_finite_difference(self):
        st = _complex_sphere4()
        modes = (
            cob.HodgeEntropyPhaseMode.IncludeComplexPhase,
            cob.HodgeEntropyPhaseMode.IgnoreComplexPhase,
        )
        edges = st.getEdgeList().toVector()
        step = 2e-6
        for mode in modes:
            gradient = np.asarray(
                cob.HodgeLaplacian(st).spectralEntropyGradient(3, mode),
                dtype=complex)
            self.assertEqual(gradient.shape, (len(edges),))
            for edge_index in (0, 2):
                edge = edges[edge_index]
                original_length = complex(edge.getLength())
                z0 = original_length * original_length

                def value(z):
                    edge.setLength(cmath.sqrt(z))
                    return float(cob.HodgeLaplacian(st).spectralEntropy(3, mode))

                f_re_plus = value(z0 + step)
                f_re_minus = value(z0 - step)
                f_im_plus = value(z0 + 1j * step)
                f_im_minus = value(z0 - 1j * step)
                edge.setLength(original_length)

                fd_re = (f_re_plus - f_re_minus) / (2.0 * step)
                fd_im = (f_im_plus - f_im_minus) / (2.0 * step)
                scale = max(abs(fd_re), abs(fd_im), 1.0)
                self.assertLess(abs(gradient[edge_index].real - fd_re) / scale,
                                2e-5)
                self.assertLess(abs(-gradient[edge_index].imag - fd_im) / scale,
                                2e-5)

            expected_norm = float(np.vdot(gradient, gradient).real)
            measured_norm = float(
                cob.HodgeLaplacian(st).spectralEntropyGradientNorm(3, mode))
            self.assertAlmostEqual(measured_norm, expected_norm, places=10)

    def test_degree_zero_entropy_is_observable_but_gradient_is_explicitly_unsupported(self):
        st = _complex_sphere4()
        hl = cob.HodgeLaplacian(st)
        self.assertTrue(math.isfinite(hl.spectralEntropy(0)))
        with self.assertRaises(RuntimeError):
            hl.spectralEntropyGradient(0)


if __name__ == "__main__":
    unittest.main()
