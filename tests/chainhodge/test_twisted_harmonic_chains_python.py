# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The twisted zero mode of the covariant pencil (#955):
``CovariantChainHodge.harmonicChains`` is the exact kernel with the link
phases on. On the 3x3 torus of the specification: a pure-gauge connection keeps
the twisted Betti number b_1 = 2 and its images are the gauge-dressed untwisted
ones; a flux or a flat holonomy collapses the twisted harmonic space."""
import cmath

import numpy as np
import pytest

from tessera import chainhodge as ch

from ._fixtures import torus33


def _instances(links):
    K, s = torus33(h=1.0, v=0.8, dgl=1.1)
    base = ch.ChainHodge(K, s)
    return K, base, ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)


def _edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def test_trivial_connection_reproduces_the_untwisted_read():
    K, base, cov = _instances([1.0 + 0j] * 27)
    untwisted = base.harmonicChains(1)
    twisted = cov.harmonicChains(1)
    assert untwisted.nullity == twisted.nullity == 2
    assert twisted.degree == 1 and twisted.dense
    # Same kernel, up to the basis: the images span the same subspace.
    Zu, Zt = np.asarray(untwisted.images), np.asarray(twisted.images)
    coeff, *_ = np.linalg.lstsq(Zu, Zt, rcond=None)
    assert np.linalg.norm(Zu @ coeff - Zt) < 1e-12
    # Chains are the dressed metric applied to the images (h = M_1^U z).
    M = np.asarray(cov.Minv(1).todense())
    assert np.linalg.norm(np.asarray(twisted.chains) - M @ Zt) < 1e-12


def test_pure_gauge_keeps_the_rank_and_dresses_the_images():
    K, base, _ = _instances([1.0 + 0j] * 27)
    rng = np.random.default_rng(1)
    g = {v: cmath.exp(1j * rng.uniform(-2, 2)) for v in range(9)}
    # U_xy = g_x / g_y on the canonical (x < y) edge: a pure gauge.
    links = [g[x] / g[y] for (x, y) in _edges(K)]
    cov = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
    twisted = cov.harmonicChains(1)
    assert twisted.nullity == 2
    assert twisted.rank == base.harmonicChains(1).rank
    for k in (0, 2):
        assert cov.harmonicChains(k).nullity == base.harmonicChains(k).nullity


@pytest.mark.parametrize("flux_edge", [0, 13])
def test_flux_collapses_the_twisted_harmonic_space(flux_edge):
    links = [1.0 + 0j] * 27
    links[flux_edge] = cmath.exp(0.7j)
    _, base, cov = _instances(links)
    assert base.harmonicChains(1).nullity == 2
    assert cov.harmonicChains(1).nullity < 2


def test_degree_out_of_range_raises():
    _, _, cov = _instances([1.0 + 0j] * 27)
    with pytest.raises(ValueError):
        cov.harmonicChains(3)
