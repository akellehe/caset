# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Dirac-Kahler operator (d+delta) and the conserved current j^0 (#415).

The Dirac-Kahler operator D = d + delta is the label-independent square root of
the HodgeLaplacian on the inhomogeneous cochain complex (+)_k Omega^k: with the
boundary part d_k and the metric codifferential d_k* = W_k^-1 d_k^T W_{k-1}, and
d^2 = (d*)^2 = 0, the square is block-diagonal per degree and reproduces the
Hodge Laplacian, (d+delta)^2 = L. The gammas are the Clifford action of unit
1-cochains on the Kahler-Atiyah form fiber, {gamma^a, gamma^b} = 2 eta^ab I, with
the 4-fold (16 = 4x4) doubling as the candidate flavor/taste index; j^0 is the
charge density of the conserved U(1) current.

These tests are deterministic by construction (explicit small complexes, the
uniform l^2 = 1 metric so every weight W_k is fixed; the nominal seed=415 fixes
no RNG because none is used) and pin the machinery to concrete numbers so later
flavor/charge subtasks cannot silently drift it.
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders (uniform l^2 = 1 metric; deterministic, no RNG)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    """Build a Spacetime from explicit simplex vertex tuples. createSimplex
    auto-creates every sub-face/edge."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _set_lengths(st, l2map=None, default_l2=1.0):
    for e in st.getEdgeList().toVector():
        key = frozenset((e.getSource().getId(), e.getTarget().getId()))
        e.setSquaredLength((l2map or {}).get(key, default_l2))
        e.setPhase(0.0)
    return st


def _tetra(l2map=None):
    """A single tetrahedron (dimension 3); cells in every degree 0..3.
    Uniform unit edges unless `l2map` overrides specific edges."""
    return _set_lengths(_from_simplices(4, [(0, 1, 2, 3)]), l2map, 1.0)


def _triangle_cycle():
    """S^1 as the 3-cycle 0-1-2-0 (b_1 = 1); all-spacelike unit edges."""
    return _set_lengths(_from_simplices(3, [(0, 1), (1, 2), (2, 0)]), None, 1.0)


def _stacked_laplacian(hl, dk, metric, lorentzian, kmin=0):
    """Block-diagonal stack of HodgeLaplacian.laplacian(k) over k=kmin..n in the
    DiracKahler total-space layout."""
    off = dk.blockOffsets()
    total = dk.totalDimension()
    n = dk.meshDimension()
    L = np.zeros((total, total), dtype=complex)
    for k in range(kmin, n + 1):
        m = off[k + 1] - off[k]
        if m == 0:
            continue
        lk = np.array(hl.laplacian(k, metric, lorentzian)).reshape(m, m)
        L[off[k]:off[k + 1], off[k]:off[k + 1]] = lk
    return L


def _square(dk, metric=True, lorentzian=False):
    total = dk.totalDimension()
    return np.array(dk.square(metric, lorentzian)).reshape(total, total)


def _matrix(dk, metric=True, lorentzian=False):
    total = dk.totalDimension()
    return np.array(dk.matrix(metric, lorentzian)).reshape(total, total)


# --------------------------------------------------------------------------- #
# F1 — (d+delta)^2 == L (the Hodge Laplacian reproduced)
# --------------------------------------------------------------------------- #
class TestSquareIsLaplacian(unittest.TestCase):
    def test_euclidean_square_is_laplacian_all_blocks(self):
        """F1 (Euclidean): D^2 reproduces laplacian(k, metric=True) block-diagonal
        over ALL k (k=0 included: with unit real edges the 0-form Hodge Laplacian
        equals the Hermitian graph Laplacian)."""
        st = _tetra()
        dk = cob.DiracKahler(st)
        hl = cob.HodgeLaplacian(st)
        D2 = _square(dk, metric=True, lorentzian=False)
        L = _stacked_laplacian(hl, dk, metric=True, lorentzian=False, kmin=0)

        # Overall.
        self.assertLess(np.linalg.norm(D2 - L), 1e-10)
        # Per block.
        off = dk.blockOffsets()
        for k in range(dk.meshDimension() + 1):
            blk = D2[off[k]:off[k + 1], off[k]:off[k + 1]]
            lk = L[off[k]:off[k + 1], off[k]:off[k + 1]]
            self.assertLess(np.linalg.norm(blk - lk), 1e-10, f"block k={k}")
        # The convenience residual agrees.
        self.assertLess(dk.laplacianResidual(metric=True, lorentzian=False), 1e-10)

    def test_square_is_block_diagonal(self):
        """The cross-degree blocks of D^2 vanish (d^2 = delta^2 = 0)."""
        dk = cob.DiracKahler(_tetra())
        D2 = _square(dk, metric=True, lorentzian=False)
        off = dk.blockOffsets()
        offdiag = D2.copy()
        for k in range(dk.meshDimension() + 1):
            offdiag[off[k]:off[k + 1], off[k]:off[k + 1]] = 0.0
        self.assertLess(np.linalg.norm(offdiag), 1e-10)

    def test_lorentzian_square_is_dalembertian_blocks(self):
        """F1 (Lorentzian): with signed weights the square reproduces the signed
        d'Alembertian blocks (k>=1) of laplacian(k, lorentzian=True). The k=0
        HodgeLaplacian is the Hermitian graph Laplacian (a separate operator), so
        the d'Alembertian reproduction is over k>=1."""
        # Two timelike edges; deterministic assignment.
        st = _tetra({frozenset((0, 1)): -1.0, frozenset((2, 3)): -1.0})
        dk = cob.DiracKahler(st)
        hl = cob.HodgeLaplacian(st)
        D2 = _square(dk, metric=True, lorentzian=True)
        off = dk.blockOffsets()
        for k in range(1, dk.meshDimension() + 1):
            m = off[k + 1] - off[k]
            lk = np.array(hl.laplacian(k, True, True)).reshape(m, m)
            blk = D2[off[k]:off[k + 1], off[k]:off[k + 1]]
            self.assertLess(np.linalg.norm(blk - lk), 1e-10, f"d'Alembertian k={k}")
        self.assertLess(dk.laplacianResidual(metric=True, lorentzian=True), 1e-10)

    def test_matrix_is_real(self):
        """D = d+delta is a real operator (returned complex with zero imag)."""
        D = _matrix(cob.DiracKahler(_tetra()))
        self.assertLess(np.max(np.abs(D.imag)), 1e-14)

    def test_layout_offsets_consistent(self):
        dk = cob.DiracKahler(_tetra())
        off = dk.blockOffsets()
        self.assertEqual(off[0], 0)
        self.assertEqual(off[-1], dk.totalDimension())
        # Single tetra: |C_0|=4, |C_1|=6, |C_2|=4, |C_3|=1 -> total 15.
        self.assertEqual(list(off), [0, 4, 10, 14, 15])
        self.assertEqual(dk.totalDimension(), 15)


# --------------------------------------------------------------------------- #
# F2 — Clifford relations {gamma_a, gamma_b} = 2 eta_ab
# --------------------------------------------------------------------------- #
class TestCliffordRelations(unittest.TestCase):
    def _check(self, lorentzian):
        dk = cob.DiracKahler(_tetra())
        gdim = dk.gammaDimension()
        gammas = [np.array(g).reshape(gdim, gdim) for g in dk.gammas(lorentzian)]
        d = dk.frameworkDimension()
        eta = np.array(dk.signature(lorentzian)).reshape(d, d)
        I = np.eye(gdim)
        worst = 0.0
        for a in range(d):
            for b in range(d):
                anti = gammas[a] @ gammas[b] + gammas[b] @ gammas[a]
                worst = max(worst, np.linalg.norm(anti - 2 * eta[a, b] * I))
        self.assertLess(worst, 1e-12)
        # The class-reported residual agrees.
        self.assertLess(dk.cliffordResidual(lorentzian), 1e-12)
        return eta

    def test_euclidean_clifford(self):
        eta = self._check(lorentzian=False)
        self.assertTrue(np.allclose(eta, np.eye(4)))

    def test_lorentzian_clifford(self):
        eta = self._check(lorentzian=True)
        self.assertTrue(np.allclose(np.diag(eta), [-1, 1, 1, 1]))

    def test_gamma_shape(self):
        dk = cob.DiracKahler(_tetra())
        self.assertEqual(dk.gammaDimension(), 16)  # 2^4
        self.assertEqual(len(dk.gammas(False)), 4)  # d=4 generators


# --------------------------------------------------------------------------- #
# F3 — j^0 (charge density) vs the carried charge
# --------------------------------------------------------------------------- #
class TestChargeDensity(unittest.TestCase):
    def test_j0_sums_to_carried_charge(self):
        """On a closed 1-harmonic Phi (a harmonicMatrix row lifted to the total
        space), the summed j^0 equals the harmonic's carried charge <Phi,Phi>_W.

        Internal-consistency check (the deferred Gauss-law charge-density ticket,
        #411, will tighten this to a cross-check against the period-derived
        charge); here charge() is cross-checked against an independent weighted
        sum using the public HodgeLaplacian.weights."""
        st = _triangle_cycle()  # S^1, b_1 = 1
        dk = cob.DiracKahler(st)
        hl = cob.HodgeLaplacian(st)
        m1 = dk.blockOffsets()[2] - dk.blockOffsets()[1]
        hm = np.array(hl.harmonicMatrix(1)).reshape(-1, m1)
        self.assertEqual(hm.shape[0], 1)  # b_1 = 1
        h = hm[0]

        field = dk.lift(1, list(h))
        q = dk.charge(field)
        # Independent weighted sum from the public weights API.
        w1 = np.array(hl.weights(1))
        expected = float(np.sum(w1 * np.abs(h) ** 2))
        self.assertAlmostEqual(q, expected, delta=1e-6)

        # charge() == sum(chargeDensity()).
        dens = np.array(dk.chargeDensity(field))
        self.assertAlmostEqual(q, float(dens.sum()), delta=1e-12)
        # Density is nonnegative (a proper charge density) and supported on edges.
        self.assertTrue(np.all(dens >= -1e-15))
        off = dk.blockOffsets()
        self.assertAlmostEqual(float(dens[:off[1]].sum()), 0.0, delta=1e-12)  # no 0-cell charge

    def test_k0_harmonic_charge(self):
        """The k=0 harmonic (constant function) carries unit charge (W_0 = I)."""
        st = _triangle_cycle()
        dk = cob.DiracKahler(st)
        hl = cob.HodgeLaplacian(st)
        n0 = dk.blockOffsets()[1]
        hm = np.array(hl.harmonicMatrix(0)).reshape(-1, n0)
        h = hm[0]
        field = dk.lift(0, list(h))
        self.assertAlmostEqual(dk.charge(field), float(np.sum(np.abs(h) ** 2)),
                               delta=1e-12)


# --------------------------------------------------------------------------- #
# F4 — the 4-fold Dirac-Kahler multiplicity (anti-drift anchor)
# --------------------------------------------------------------------------- #
class TestMultiplicity(unittest.TestCase):
    def test_multiplicity_is_four(self):
        """The Dirac-Kahler doubling is exactly 4 (16 = 4x4 in 4D), the candidate
        flavor/taste index. Pinned so a later change cannot silently alter it.

        Reduced-dimension caveat: the current S^2 x I cobordism is 2+1 D, but the
        multiplicity the operator reports is the fixed 4D framework value."""
        dk = cob.DiracKahler(_tetra())  # mesh is 3D here; framework is 4D
        self.assertEqual(dk.multiplicity(), 4)
        self.assertEqual(dk.frameworkDimension(), 4)
        self.assertEqual(dk.gammaDimension(), 16)
        # The mesh dimension is independent of the framework multiplicity.
        self.assertEqual(dk.meshDimension(), 3)

    def test_multiplicity_independent_of_mesh(self):
        dk1 = cob.DiracKahler(_triangle_cycle())  # 1D mesh
        dk3 = cob.DiracKahler(_tetra())            # 3D mesh
        self.assertEqual(dk1.multiplicity(), 4)
        self.assertEqual(dk3.multiplicity(), 4)


# --------------------------------------------------------------------------- #
# G7 — determinism (the fixed complex gives reproducible objects)
# --------------------------------------------------------------------------- #
class TestDeterminism(unittest.TestCase):
    def test_bit_for_bit_reproducible(self):
        def build():
            st = _tetra({frozenset((0, 1)): -1.0})
            dk = cob.DiracKahler(st)
            return (np.array(dk.matrix(True, False)),
                    np.array(dk.square(True, False)),
                    np.array(dk.matrix(True, True)),
                    np.array(dk.gammas(False)[2]),
                    np.array(dk.gammas(True)[0]))

        a = build()
        b = build()
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))  # bit-for-bit


if __name__ == "__main__":
    unittest.main()
