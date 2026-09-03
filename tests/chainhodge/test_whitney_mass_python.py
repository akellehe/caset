# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Whitney mass matrices (#906): the specification's §14 values (T5a, T5b, T6),
symmetry and sparsity, allowability, the scaling identity of the derivative,
branch agreement on the allowable domain, and relabeling covariance."""
import cmath
import itertools
import math

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

WM = ch.WhitneyMass


def _edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def _lengths(K, table):
    return [complex(table[e]) for e in _edges(K)]


def _torus33(h=1.0, v=-0.5, dgl=0.5):
    """The oracle's 3x3 CDT-like torus: 9 vertices, 27 edges, 18 triangles."""
    n = 3

    def vid(i, j):
        return (i % n) * n + (j % n)

    cells = []
    for i in range(n):
        for j in range(n):
            cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            cells.append([vid(i, j), vid(i, j + 1), vid(i + 1, j + 1)])
    K = cob.ChainComplex.fromTopCells(cells)

    def stype(e):
        a, b = e
        ia, ja = divmod(a, n)
        ib, jb = divmod(b, n)
        di, dj = (ib - ia) % n, (jb - ja) % n
        if di in (1, 2) and dj == 0:
            return "v"
        if di == 0:
            return "h"
        return "d"

    table = {e: {"h": h, "v": v, "d": dgl}[stype(e)] for e in _edges(K)}
    return K, _lengths(K, table)


def _random_allowable(K, rng, scale=0.05):
    """Unit Euclidean lengths with a small complex perturbation (allowable)."""
    n = K.numSimplices(1)
    return [complex(1.0 + scale * rng.normal(), scale * rng.normal()) for _ in range(n)]


def _blade_dot(table, e, f):
    """<u_e, u_f> for edge vectors u_(a,b) = x_b - x_a, by polarization."""
    def S(a, b):
        return 0.0 if a == b else table[(min(a, b), max(a, b))]
    a, b = e
    c, d = f
    return 0.5 * (S(b, c) + S(a, d) - S(b, d) - S(a, c))


def _grassmann_reference(K, s, k):
    """Dense port of the specification oracle's metric(): multiplicity o blade pairing."""
    table = dict(zip(_edges(K), s))
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    Gam = np.zeros((n, n), dtype=complex)
    mult = np.zeros((n, n))
    seen = set()

    def blade(sig, tau):
        if k == 0:
            return 1.0 + 0j
        A = np.array([[_blade_dot(table, (sig[0], sig[i]), (tau[0], tau[j]))
                       for j in range(1, k + 1)] for i in range(1, k + 1)], dtype=complex)
        return np.linalg.det(A) / (math.factorial(k) ** 2)

    for kk in range(k, K.dimension() + 1):
        for rho in K.kSimplexVertices(kk):
            rho = tuple(int(v) for v in rho)
            faces = list(itertools.combinations(rho, k + 1))
            for a in faces:
                for b in faces:
                    i, j = idx[a], idx[b]
                    mult[i, j] += 1.0
                    if (i, j) not in seen:
                        Gam[i, j] = blade(a, b)
                        seen.add((i, j))
    return mult * Gam


def _whitney_reference_d2(K, s, k):
    """Dense port of the specification oracle's whitney_mass() for d = 2, principal branch."""
    table = dict(zip(_edges(K), s))
    d = K.dimension()
    assert d == 2
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    M = np.zeros((n, n), dtype=complex)
    for T in K.orientedTopSimplices():
        T = tuple(int(v) for v in T)
        g = np.array([[_blade_dot(table, (T[0], T[i]), (T[0], T[j])) for j in range(1, 3)]
                      for i in range(1, 3)], dtype=complex)
        vol = np.sqrt(np.linalg.det(g) + 0j) / 2.0
        if k == 0:
            for a in T:
                for b in T:
                    M[idx[(a,)], idx[(b,)]] += vol * (1 + (a == b)) / 12.0
        elif k == 1:
            ginv = np.linalg.inv(g)
            Gam = np.zeros((3, 3), dtype=complex)
            Gam[1:, 1:] = ginv
            Gam[0, 1:] = -ginv.sum(axis=0)
            Gam[1:, 0] = -ginv.sum(axis=1)
            Gam[0, 0] = ginv.sum()
            for (i, j) in itertools.combinations(range(3), 2):
                for (kk, l) in itertools.combinations(range(3), 2):
                    val = vol / 12.0 * ((1 + (i == kk)) * Gam[j, l] - (1 + (i == l)) * Gam[j, kk]
                                        - (1 + (j == kk)) * Gam[i, l] + (1 + (j == l)) * Gam[i, kk])
                    M[idx[(T[i], T[j])], idx[(T[kk], T[l])]] += val
        else:
            M[idx[T], idx[T]] = 1.0 / vol
    return M


FIXTURES = {
    "2-complex": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]],
    "3-complex": [[0, 1, 2, 3], [1, 2, 3, 4]],
    "4-complex": [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
}


class TestT5aEuclideanThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 4.0, 9.0]  # l = (1, 2, 3) on edges (01), (02), (12)

    def test_m1_is_inverse_lengths(self):
        M1 = WM.assemble(self.K, self.s, 1).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 0.5, 1.0 / 3.0]), atol=1e-15)

    def test_m0_exact(self):
        M0 = WM.assemble(self.K, self.s, 0).toarray()
        expected = np.array([[1.0, 1 / 6, 1 / 3], [1 / 6, 4 / 3, 1 / 2], [1 / 3, 1 / 2, 5 / 3]])
        np.testing.assert_allclose(M0, expected, atol=1e-15)

    def test_certificate_euclidean(self):
        cert = WM.certificate(self.K, self.s)
        assert cert.allowable
        assert cert.margin == pytest.approx(math.pi)
        np.testing.assert_allclose(cert.volumes, [1.0, 2.0, 3.0], atol=1e-15)
        assert not cert.continuationAmbiguous


class TestT5bMixedSignatureThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 1.0, -1.0]

    def test_m1_principal_branch(self):
        M1 = WM.assemble(self.K, self.s, 1, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 1.0, -1j]), atol=1e-15)

    def test_continuation_falls_back_and_reports(self):
        M1 = WM.assemble(self.K, self.s, 1, ch.Branch.Continuation).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 1.0, -1j]), atol=1e-15)
        cert = WM.certificate(self.K, self.s, ch.Branch.Continuation)
        assert cert.continuationAmbiguous
        assert list(cert.ambiguousTopSimplices) == [2]
        assert not cert.allowable  # the timelike edge sits on the cut (margin 0)
        assert cert.margin == pytest.approx(0.0, abs=1e-15)


class TestT6LorentzianTorus:
    def test_m2_is_minus_two_root_two_i(self):
        K, s = _torus33()
        M2 = WM.assemble(K, s, 2, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(M2, -2 * math.sqrt(2) * 1j * np.eye(18), atol=1e-14)
        cert = WM.certificate(K, s, ch.Branch.KontsevichSegal)
        np.testing.assert_allclose(cert.gramDeterminants, [-0.5] * 18, atol=1e-15)
        assert cert.margin == pytest.approx(0.0, abs=1e-14)
        assert not cert.allowable

    def test_continuation_agrees_with_kontsevich_segal_up_to_reporting(self):
        K, s = _torus33()
        a = WM.assemble(K, s, 1, ch.Branch.Continuation).toarray()
        b = WM.assemble(K, s, 1, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(a, b, atol=1e-14)
        assert WM.certificate(K, s, ch.Branch.Continuation).continuationAmbiguous

    def test_global_factor_i(self):
        """Real Lorentzian data: every M_k is i times a real matrix (§4.2)."""
        K, s = _torus33()
        for k in range(3):
            M = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            assert np.max(np.abs(M.real)) < 1e-14
            assert np.max(np.abs(M.imag)) > 0

    def test_rotated_family_is_allowable(self):
        """Rotating the timelike part by e^{-2i eps} lands on the allowable side."""
        K, s = _torus33()
        eps = 0.1
        s_eps = [v * cmath.exp(-2j * eps) if v.real < 0 else v for v in s]
        cert = WM.certificate(K, s_eps, ch.Branch.Continuation)
        assert cert.allowable and cert.margin > 0
        assert not cert.continuationAmbiguous


class TestStructure:
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_symmetric_and_sparsity_rule(self, name):
        rng = np.random.default_rng(3)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        tops = [tuple(t) for t in K.orientedTopSimplices()]
        for k in range(K.dimension() + 1):
            M = WM.assemble(K, s, k).toarray()
            np.testing.assert_allclose(M, M.T, atol=1e-14)
            cells = [tuple(c) for c in K.kSimplexVertices(k)]
            for i, j in zip(*np.nonzero(np.abs(M) > 1e-15)):
                union = set(cells[i]) | set(cells[j])
                assert any(union <= set(t) for t in tops)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_scaling_identity(self, name):
        """sum_e s_e dM_k/ds_e = (d/2 - k) M_k at every degree."""
        rng = np.random.default_rng(5)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        d = K.dimension()
        for k in range(d + 1):
            M = WM.assemble(K, s, k).toarray()
            total = np.zeros_like(M)
            for e, se in enumerate(s):
                total += se * WM.assembleDerivative(K, s, k, e).toarray()
            np.testing.assert_allclose(total, (d / 2 - k) * M, atol=1e-12 * max(1.0, np.abs(M).max()))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_contraction_matches_full_derivative(self, name):
        rng = np.random.default_rng(7)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        for k in range(K.dimension() + 1):
            n = K.numSimplices(k)
            X = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
            Y = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
            c = WM.derivativeContraction(K, s, k, X, Y)
            for e in range(K.numSimplices(1)):
                D = WM.assembleDerivative(K, s, k, e).toarray()
                assert c[e] == pytest.approx(np.trace(X.T @ D @ Y), abs=1e-12)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_branches_agree_on_allowable_domain(self, name):
        rng = np.random.default_rng(11)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng, scale=0.2)
        cert = WM.certificate(K, s)
        assert cert.allowable and not cert.continuationAmbiguous
        for k in range(K.dimension() + 1):
            a = WM.assemble(K, s, k, ch.Branch.Continuation).toarray()
            b = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            np.testing.assert_allclose(a, b, atol=1e-13)

    def test_top_degree_is_inverse_volume(self):
        rng = np.random.default_rng(13)
        for name, cells in FIXTURES.items():
            K = cob.ChainComplex.fromTopCells(cells)
            s = _random_allowable(K, rng)
            d = K.dimension()
            Md = WM.assemble(K, s, d).toarray()
            cert = WM.certificate(K, s)
            np.testing.assert_allclose(np.diag(Md), 1.0 / np.array(cert.volumes), rtol=1e-12)
            assert np.max(np.abs(Md - np.diag(np.diag(Md)))) < 1e-15

    def test_local_blocks_sum_to_assembly(self):
        rng = np.random.default_rng(17)
        K = cob.ChainComplex.fromTopCells(FIXTURES["3-complex"])
        s = _random_allowable(K, rng)
        for k in range(4):
            M = WM.assemble(K, s, k).toarray()
            total = np.zeros_like(M)
            for b in WM.topSimplexBlocks(K, s, k):
                idx = list(b.cellIndices)
                total[np.ix_(idx, idx)] += b.block
            np.testing.assert_allclose(total, M, atol=1e-14)


class TestAllowability:
    def test_non_allowable_complex_conformal_on_lorentzian_base(self):
        K, s = _torus33()
        s_bad = [v * (0.3 + 0.2j) for v in s]
        assert WM.allowabilityMargin(K, s_bad) <= 0.0
        assert not WM.certificate(K, s_bad).allowable

    def test_margin_of_unit_gram(self):
        g = 0.5 * (np.ones((3, 3)) + np.eye(3)).astype(complex)
        assert WM.marginOf(g) == pytest.approx(math.pi)
        vol, ambiguous = WM.volumeOnBranch(g)
        assert vol == pytest.approx(math.sqrt(0.5) / 6.0)  # unit regular tetrahedron: sqrt(2)/12
        assert not ambiguous

    def test_grassmann_preset_matches_dense_reference(self):
        """assembleGrassmann equals multiplicity o blade pairing (the CH §6 oracle),
        is real on real data, and each face's own blade block has rank two."""
        K, s = _torus33()
        for k in range(3):
            G = WM.assembleGrassmann(K, s, k).toarray()
            np.testing.assert_allclose(G, _grassmann_reference(K, s, k), atol=1e-14)
            assert np.max(np.abs(G.imag)) == 0.0
            np.testing.assert_allclose(G, G.T, atol=1e-15)
        G1p = WM.assemblePreset(K, s, 1, ch.Preset.GRASSMANN_ALL).toarray()
        np.testing.assert_allclose(G1p, WM.assembleGrassmann(K, s, 1).toarray(), atol=1e-15)
        # The per-face blade block (three edge vectors of one triangle) has rank two (§4.4).
        table = dict(zip(_edges(K), s))
        for t in K.orientedTopSimplices():
            es = list(itertools.combinations(sorted(int(v) for v in t), 2))
            blk = np.array([[_blade_dot(table, a, b) for b in es] for a in es])
            assert np.linalg.matrix_rank(blk, tol=1e-12) == 2

    def test_whitney_matches_dense_reference_d2(self):
        """The sparse assembly equals the dense Whitney mass matrix of the
        specification's oracle at d = 2, on the Lorentzian torus and on random
        complex allowable data."""
        rng = np.random.default_rng(23)
        K, s = _torus33()
        for k in range(3):
            M = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            np.testing.assert_allclose(M, _whitney_reference_d2(K, s, k), atol=1e-13)
        K2 = cob.ChainComplex.fromTopCells(FIXTURES["2-complex"])
        s2 = _random_allowable(K2, rng, scale=0.3)
        for k in range(3):
            M = WM.assemble(K2, s2, k).toarray()
            np.testing.assert_allclose(M, _whitney_reference_d2(K2, s2, k), atol=1e-13)

    def test_errors(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2]])
        with pytest.raises(ValueError):
            WM.assemble(K, [1.0, 1.0], 1)
        with pytest.raises(ValueError):
            WM.assemble(K, [1.0, 1.0, 1.0], 3)
        with pytest.raises(ValueError):
            cob.ChainComplex.fromTopCells([[0, 1, 2], [0, 1]])


class TestRelabeling:
    def test_signed_permutation_similarity(self):
        rng = np.random.default_rng(19)
        K, s = _torus33()
        s = [v + 0.05j for v in s]  # complex, off the cut
        perm = {i: 8 - i for i in range(9)}
        cells = [[perm[v] for v in t] for t in K.orientedTopSimplices()]
        K2 = cob.ChainComplex.fromTopCells(cells)
        edges, edges2 = _edges(K), _edges(K2)
        table = dict(zip(edges, s))
        inv = {v: k for k, v in perm.items()}
        s2 = [table[tuple(sorted((inv[a], inv[b])))] for (a, b) in edges2]
        for k in range(3):
            M = WM.assemble(K, s, k).toarray()
            M2 = WM.assemble(K2, s2, k).toarray()
            cells_k = [tuple(c) for c in K.kSimplexVertices(k)]
            cells2 = [tuple(c) for c in K2.kSimplexVertices(k)]
            P = np.zeros((len(cells_k), len(cells_k)))
            for j, c in enumerate(cells_k):
                image = [perm[v] for v in c]
                i2 = cells2.index(tuple(sorted(image)))
                # sign = parity of the permutation sorting the relabeled tuple
                order = sorted(range(len(image)), key=lambda p: image[p])
                sign = 1.0
                seen = [False] * len(order)
                for start in range(len(order)):
                    if seen[start]:
                        continue
                    length, p = 0, start
                    while not seen[p]:
                        seen[p] = True
                        p = order[p]
                        length += 1
                    if length % 2 == 0:
                        sign = -sign
                P[i2, j] = sign
            np.testing.assert_allclose(P @ M @ P.T, M2, atol=1e-13)
