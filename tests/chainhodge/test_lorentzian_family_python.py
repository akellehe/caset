# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Lorentzian protocol (#908, specification §10): the epsilon-rotated family on
declared causal types, epsilon on the certificate, reads at epsilon = 0 only
inside a family with their gap, the labeled extrapolation, and the §10
evidence rows on the rotated curved torus."""
import math

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch
from tests.chainhodge._fixtures import (conformal_torus_rotated, torus33,
                                        torus33_causal_types)

LF = ch.LorentzianFamily
KS = ch.Branch.KontsevichSegal


def _hausdorff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a[:, None] - b[None, :])
    return max(d.min(axis=1).max(), d.min(axis=0).max())


class TestRotate:
    def test_timelike_entries_rotate_and_others_do_not(self):
        K, s = torus33()
        types = torus33_causal_types(K)
        eps = 0.2
        r = LF.rotate(s, types, eps)
        for v, w, t in zip(s, r, types):
            if t == ch.CausalType.Timelike:
                assert w == pytest.approx(v * np.exp(-2j * eps), abs=1e-15)
            else:
                assert w == v
        assert sum(t == ch.CausalType.Timelike for t in types) == 9

    def test_type_count_must_match(self):
        K, s = torus33()
        with pytest.raises(ValueError):
            LF.rotate(s, [ch.CausalType.Spacelike] * 3, 0.1)

    def test_null_edges_are_not_rotated(self):
        s = [1.0, 0.0, -1.0]
        types = [ch.CausalType.Spacelike, ch.CausalType.Null, ch.CausalType.Timelike]
        r = LF.rotate(s, types, 0.3)
        assert r[0] == 1.0 and r[1] == 0.0 and r[2] == pytest.approx(-np.exp(-0.6j), abs=1e-15)


class TestInstanceAndSweep:
    def test_epsilon_on_certificate_and_allowability(self):
        K, s = torus33()
        types = torus33_causal_types(K)
        for eps in (0.05, 0.1, 0.3):
            hodge = LF.instance(K, s, types, eps, ch.Preset.L2, KS)
            cert = hodge.certificate()
            assert cert.epsilon == eps
            assert cert.allowable and cert.margin > 0.0
            assert not cert.continuationAmbiguous
        zero = LF.instance(K, s, types, 0.0, ch.Preset.L2, KS)
        assert zero.certificate().epsilon == 0.0
        assert not zero.certificate().allowable
        assert math.isnan(ch.ChainHodge(K, s).certificate().epsilon)

    def test_sweep_reads_carry_gap_and_are_continuous(self):
        K, s = torus33()
        types = torus33_causal_types(K)
        epsilons = [0.0, 0.01, 0.02, 0.04]
        reads = LF.sweep(K, s, types, epsilons, 1, ch.Preset.L2, KS, 10.0, True)
        assert [r.epsilon for r in reads] == epsilons
        for r in reads:
            assert r.harmonic.nullity == 2
            assert math.isfinite(r.harmonic.gap) and r.harmonic.gap > 1e3
            assert len(r.eigenvalues) == 27
        assert not reads[0].allowable and all(r.allowable for r in reads[1:])
        ev0 = np.array(reads[0].eigenvalues)
        assert np.max(np.abs(ev0.imag)) < 3e-14 * np.max(np.abs(ev0))
        scale = np.max(np.abs(ev0))
        d = [_hausdorff(r.eigenvalues, ev0) / scale for r in reads[1:]]
        assert d[0] < d[1] < d[2]          # the spectrum moves continuously away from the boundary
        assert d[0] < 0.1                  # measured 0.083 at epsilon = 0.01

    def test_zero_epsilon_read_is_a_family_member_with_gap(self):
        K, s = torus33()
        types = torus33_causal_types(K)
        reads = LF.sweep(K, s, types, [0.0], 1, ch.Preset.L2, KS)
        assert len(reads) == 1 and reads[0].epsilon == 0.0
        assert math.isfinite(reads[0].harmonic.gap)


class TestExtrapolation:
    def test_recovers_polynomial_value_at_zero(self):
        eps = [0.1, 0.2, 0.3, 0.4]
        vals = [1.0 + 2.0 * e - 3.0 * e * e + 0.5j * e for e in eps]
        ex = LF.extrapolateToZero(eps, vals, 2)
        assert ex.extrapolated == pytest.approx(1.0, abs=1e-12)
        assert ex.residual < 1e-12 and ex.order == 2
        assert "extrapolation" in ex.label

    def test_rejects_zero_epsilon_and_short_input(self):
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.0, 0.1], [1.0, 1.0])
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.1], [1.0])
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.1, 0.2], [1.0])

    def test_extrapolated_eigenvalue_approaches_the_boundary_value(self):
        K, s = torus33()
        types = torus33_causal_types(K)
        epsilons = [0.005, 0.01, 0.015, 0.02]
        reads = LF.sweep(K, s, types, epsilons, 1, ch.Preset.L2, KS, 10.0, True)
        # The trace of the pencil operator is analytic in epsilon (individual
        # eigenvalues cross and split, so a single one is not an extrapolation
        # target); a quadratic through reads at epsilon <= 0.02 reaches the
        # boundary value to well under a percent.
        traces = [complex(np.sum(r.eigenvalues)) for r in reads]
        ex = LF.extrapolateToZero(epsilons, traces, 2)
        zero = LF.sweep(K, s, types, [0.0], 1, ch.Preset.L2, KS, 10.0, True)[0]
        target = complex(np.sum(zero.eigenvalues))
        assert abs(ex.extrapolated - target) < 1e-3 * abs(target)
        assert abs(target.imag) < 1e-9 * abs(target)


class TestEvidenceRotatedCurvedTorus:
    """§10 finding, reproduced as labeled evidence: with the timelike direction
    rotated by e^{-2 i epsilon} the curved Lorentzian torus is allowable and
    its harmonic images converge at second order; at epsilon = 0 the angles
    are reported with the gap and carry no pass criterion."""

    @staticmethod
    def _angle(N, eps):
        K, s, W = conformal_torus_rotated(N, 0.3, 0.15, eps, seed=1)
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS, 2048, eps)
        read = hodge.harmonicChains(1)
        assert read.nullity == 2
        return float(np.max(np.degrees(subspace_angles(read.images, W)))), read.gap, hodge.certificate()

    @pytest.mark.parametrize("eps", [0.1, 0.3, 0.6])
    def test_rotated_family_converges(self, eps):
        a8, _, c8 = self._angle(8, eps)
        a12, _, c12 = self._angle(12, eps)
        assert c8.allowable and c12.allowable and c8.epsilon == eps
        assert a8 < 5.0
        assert a8 / a12 >= 1.5 ** 1.5

    def test_zero_epsilon_is_reported_with_gap(self):
        a8, g8, c8 = self._angle(8, 0.0)
        a12, g12, c12 = self._angle(12, 0.0)
        assert not c8.allowable and c8.margin == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(a8) and math.isfinite(a12)
        assert math.isfinite(g8) and math.isfinite(g12)
