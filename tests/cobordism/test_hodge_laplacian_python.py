# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Hermitian-weighted Hodge Laplacian, k=0 operator (#90).

Validates the U(1)-weighted graph Laplacian L = D - A assembled from edge
weights (squaredLength * exp(i*phase)) against an independent numpy D-A oracle,
plus the spec checks that live at degree 0: Hermiticity / unitary evolution,
the Aharonov-Bohm flux spectrum on the triangle (C4/C5 anchors), gauge
invariance (C3, computed test-side), and a b1 cross-check against ChainComplex.

Fixtures (per the cobordism plan):
  triangle = SimplexBoundarySphere(1)  (S^1 = boundary of a 2-simplex; b1=1)
  path 0-1-2                            (a tree; b1=0)
  testbed  square 00-01-11-10 + diag 00-11  (b1=2; the representative cyclic
                                             fixture, ids 0=00,1=01,2=11,3=10)
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _build_topology(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _from_simplices(num_vertices, simplices):
    """Build a Spacetime directly from explicit simplex vertex tuples (vertices
    0..num_vertices-1), the hand-crafted-complex idiom shared with the cobordism
    verification tests."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _set_uniform(st, squared_length=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(squared_length)))
        e.setPhase(phase)


def _triangle():
    """S^1 = boundary of a 2-simplex, with unit real edge weights (Φ = 0)."""
    st = _build_topology(tessera.SimplexBoundarySphere(1))
    _set_uniform(st, 1.0, 0.0)
    return st


def _path():
    st = _from_simplices(3, [(0, 1), (1, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _testbed():
    # square 00-01-11-10-00 plus the entangling diagonal 00-11.
    st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _torus():
    """T^2 = S^1 x S^1 (the qubit, spec sec 5.2): b = [1, 2, 1]. Built via the
    proven SimplicialProduct + CDT path the homology tests use, then given unit
    spacelike edge weights so every cell has a well-defined positive volume."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topology = tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                         tessera.SimplexBoundarySphere(1))
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    _set_uniform(st, 1.0, 0.0)
    return st


# --------------------------------------------------------------------------- #
# numpy oracle and small helpers
# --------------------------------------------------------------------------- #
def _ordering(st):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    return ids, {vid: i for i, vid in enumerate(ids)}


def _np_laplacian(st):
    """Independent D - A reference, built from the same edges with the same
    stable (sorted-id) vertex order the operator uses."""
    ids, idx = _ordering(st)
    n = len(ids)
    A = np.zeros((n, n), dtype=complex)
    D = np.zeros(n)
    for e in st.getEdgeList().toVector():
        s = e.getSource().getId()
        t = e.getTarget().getId()
        if s == t:
            continue
        i, j = idx[s], idx[t]
        w = (e.getLength() * e.getLength()).real
        z = w * np.exp(1j * e.getPhase())
        A[i, j] += z
        A[j, i] += np.conj(z)
        D[i] += abs(w)
        D[j] += abs(w)
    L = np.diag(D).astype(complex) - A
    return n, ids, idx, A, D, L


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _cycle_flux(st, cycle):
    """Directed holonomy Σ phase around a closed vertex cycle, honoring each
    edge's stored source->target orientation."""
    total = 0.0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.getSource().getId() == a and e.getTarget().getId() == b:
            total += e.getPhase()
        else:
            total -= e.getPhase()
    return total


def _matrix(flat, n):
    return np.array(flat, dtype=complex).reshape(n, n)


def _cluster_ranges(evals, tol=1e-6):
    """Contiguous index ranges of (near-)equal ascending eigenvalues."""
    ranges = []
    start = 0
    for i in range(1, len(evals) + 1):
        if i == len(evals) or abs(evals[i] - evals[start]) > tol:
            ranges.append((start, i))
            start = i
    return ranges


# --------------------------------------------------------------------------- #
# Metric Hodge Laplacian (k >= 1): independent numpy oracle and kernel helpers
# --------------------------------------------------------------------------- #
def _boundary(cc, k):
    """Boundary matrix d_k as a dense (|C_{k-1}| x |C_k|) numpy array."""
    rows, cols = cc.numSimplices(k - 1), cc.numSimplices(k)
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
    return np.array(cc.boundaryMatrix(k), dtype=float).reshape(rows, cols)


def _np_metric_laplacian(st, k, metric=True):
    """Independent reconstruction of the signed-weight Hodge Laplacian

        L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k

    from the boundary maps (ChainComplex) and the COMPLEX signed weights W_k
    (HodgeLaplacian.weights, or unit weights when metric is False).

    There is no sqrt(W) anywhere. The symmetric W^{-1/2} form the old oracle
    built needs positive weights, and a Lorentzian cell's d-content is
    imaginary, so that form no longer exists (#641)."""
    cc = cob.ChainComplex.fromSpacetime(st)
    n = cc.dimension()
    hl = cob.HodgeLaplacian(st)

    def weight(kk):
        if not metric or kk == 0:
            return np.ones(cc.numSimplices(kk), dtype=complex)
        return np.array(hl.weights(kk), dtype=complex)

    nk = cc.numSimplices(k)
    L = np.zeros((nk, nk), dtype=complex)
    if nk == 0:
        return L
    inv_wk = 1.0 / weight(k)
    if k >= 1 and cc.numSimplices(k - 1) > 0:   # W_k^-1 d_k^T W_{k-1} d_k
        dk = _boundary(cc, k).astype(complex)
        L += np.diag(inv_wk) @ dk.T @ np.diag(weight(k - 1)) @ dk
    if k + 1 <= n and cc.numSimplices(k + 1) > 0:  # d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k
        dkp1 = _boundary(cc, k + 1).astype(complex)
        L += dkp1 @ np.diag(1.0 / weight(k + 1)) @ dkp1.T @ np.diag(weight(k))
    return L


def _betti(st, k):
    return cob.ChainComplex.fromSpacetime(st).bettiNumbers()[k]


def _real_spectrum(evals):
    """Degree 0 is the Hermitian graph Laplacian D - A, so its spectrum is REAL.

    `eigenvalues()` is complex-typed for parity with the k >= 1 d'Alembertian.
    This asserts the imaginary part vanishes rather than discarding it, so a
    degree-0 spectrum that stopped being real fails here instead of being
    silently projected."""
    a = np.asarray(evals)
    np.testing.assert_allclose(a.imag, 0.0, atol=1e-12,
                               err_msg="degree-0 spectrum must be real")
    return a.real


def _kernel_dim_from_eigenvalues(st, k, metric=True, tol=1e-7):
    evals = np.array(cob.HodgeLaplacian(st).eigenvalues(k, metric))
    return int(np.sum(np.abs(evals) < tol))


def _harmonic_dim(st, k, metric=True, tol=1e-9):
    # harmonics() is one Cochain per basis vector of ker L_k, so its length is
    # the harmonic dimension (= b_k) directly.
    return len(cob.HodgeLaplacian(st).harmonics(k, tol, metric))


# --------------------------------------------------------------------------- #
# Assembly + spectrum vs numpy, and known-answer anchors
# --------------------------------------------------------------------------- #
class TestAssemblyAndSpectrum(unittest.TestCase):

    def _check_against_numpy(self, st):
        n, _ids, _idx, A, D, L = _np_laplacian(st)
        hl = cob.HodgeLaplacian(st)
        np.testing.assert_allclose(_matrix(hl.adjacency(), n), A, atol=1e-12)
        np.testing.assert_allclose(np.array(hl.degree()), D, atol=1e-12)
        np.testing.assert_allclose(_matrix(hl.laplacian(), n), L, atol=1e-12)
        np.testing.assert_allclose(np.array(hl.laplacian(0)).reshape(n, n), L,
                                   atol=1e-12)
        np.testing.assert_allclose(_real_spectrum(hl.eigenvalues()),
                                   np.linalg.eigvalsh(L), atol=1e-12)
        return n

    def test_triangle_is_three_vertices_three_edges(self):
        st = _triangle()
        self.assertEqual(st.getVertexCount(), 3)
        self.assertEqual(len(st.getEdgeList().toVector()), 3)

    def test_triangle_spectrum_matches_numpy(self):
        self.assertEqual(self._check_against_numpy(_triangle()), 3)

    def test_path_spectrum_matches_numpy(self):
        self._check_against_numpy(_path())

    def test_testbed_spectrum_matches_numpy(self):
        self.assertEqual(self._check_against_numpy(_testbed()), 4)

    def test_random_weighted_spectra_match_numpy(self):
        # Generic complex Hermitian weights on each fixture: the operator must
        # still equal the independent numpy D - A bit for bit (to tolerance).
        rng = np.random.default_rng(20240601)
        for name, st in (("triangle", _triangle()),
                         ("path", _path()),
                         ("testbed", _testbed())):
            with self.subTest(fixture=name):
                for e in st.getEdgeList().toVector():
                    e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
                    e.setPhase(float(rng.uniform(-math.pi, math.pi)))
                self._check_against_numpy(st)

    def test_triangle_zero_phase_known_eigenvalues(self):
        # Equal-weight S^1 with no flux: L = 2I - A(K3) -> {0, 3, 3}.
        hl = cob.HodgeLaplacian(_triangle())
        np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())), [0.0, 3.0, 3.0],
                                   atol=1e-12)

    def test_path_known_eigenvalues(self):
        # Open path 0-1-2 Laplacian -> {0, 1, 3}.
        hl = cob.HodgeLaplacian(_path())
        np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())), [0.0, 1.0, 3.0],
                                   atol=1e-12)

    def test_complex_weight_round_trips_through_pybind(self):
        # phase = pi/2 on a unit edge -> that adjacency entry is purely +i (and
        # its mirror -i), confirming complex values cross the binding intact.
        st = _path()
        _ids, idx = _ordering(st)
        e = _edge(st, 0, 1)
        e.setLength(cmath.sqrt(complex(1.0)))
        e.setPhase(math.pi / 2.0)
        s, t = e.getSource().getId(), e.getTarget().getId()
        A = _matrix(cob.HodgeLaplacian(st).adjacency(), st.getVertexCount())
        self.assertAlmostEqual(A[idx[s], idx[t]], 1j, places=12)
        self.assertAlmostEqual(A[idx[t], idx[s]], -1j, places=12)


# --------------------------------------------------------------------------- #
# Hermiticity and unitary time evolution
# --------------------------------------------------------------------------- #
class TestHermiticityUnitarity(unittest.TestCase):

    def _fixtures_with_random_weights(self):
        rng = np.random.default_rng(7)
        out = []
        for name, st in (("triangle", _triangle()), ("testbed", _testbed())):
            for e in st.getEdgeList().toVector():
                e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
                e.setPhase(float(rng.uniform(-math.pi, math.pi)))
            out.append((name, st))
        return out

    def test_laplacian_is_hermitian(self):
        for name, st in self._fixtures_with_random_weights():
            with self.subTest(fixture=name):
                n = st.getVertexCount()
                L = _matrix(cob.HodgeLaplacian(st).laplacian(), n)
                self.assertLess(np.linalg.norm(L - L.conj().T), 1e-12)
                self.assertTrue(cob.HodgeLaplacian(st).isHermitian(1e-12))

    def test_time_evolution_is_unitary(self):
        for name, st in self._fixtures_with_random_weights():
            with self.subTest(fixture=name):
                hl = cob.HodgeLaplacian(st)
                self.assertLess(hl.unitarityResidual(), 1e-12)
                self.assertLess(hl.unitarityResidual(2.5), 1e-12)


# --------------------------------------------------------------------------- #
# Flux in the spectrum (Aharonov-Bohm ring) — C4/C5 anchors
# --------------------------------------------------------------------------- #
class TestFluxSpectrum(unittest.TestCase):

    @staticmethod
    def _ring_eigs(phi):
        return sorted(2.0 - 2.0 * math.cos((phi + 2.0 * math.pi * k) / 3.0)
                      for k in range(3))

    def _triangle_with_flux(self, phi):
        # The spectrum depends only on the gauge-invariant total flux, so place
        # all of Φ on a single edge; degree is phase-independent (= 2 each).
        st = _triangle()
        st.getEdgeList().toVector()[0].setPhase(phi)
        return st

    def test_flux_spectrum_matches_ring_formula(self):
        for phi in (0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi, 1.234):
            with self.subTest(phi=phi):
                hl = cob.HodgeLaplacian(self._triangle_with_flux(phi))
                np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())),
                                           self._ring_eigs(phi), atol=1e-12)

    def test_half_flux_quantum_gives_degenerate_pair(self):
        # Φ = π -> {1, 1, 4}; the spectral gap λ1 - λ0 collapses to 0.
        hl = cob.HodgeLaplacian(self._triangle_with_flux(math.pi))
        eigs = sorted(_real_spectrum(hl.eigenvalues()))
        np.testing.assert_allclose(eigs, [1.0, 1.0, 4.0], atol=1e-12)
        self.assertAlmostEqual(eigs[1] - eigs[0], 0.0, places=12)

    def test_flux_lifts_the_zero_mode(self):
        # No flux: one harmonic (the constant 0-cochain, b0 = 1). Any flux lifts
        # it, so the harmonic dimension of L0 drops to 0.
        hl0 = cob.HodgeLaplacian(_triangle())
        self.assertEqual(len(hl0.harmonics()), 1)
        hlpi = cob.HodgeLaplacian(self._triangle_with_flux(math.pi))
        self.assertEqual(len(hlpi.harmonics()), 0)

    def test_zero_mode_is_uniform(self):
        # The Φ=0 harmonic is the uniform vector (equal magnitudes on every vertex).
        n = 3
        harmonics = cob.HodgeLaplacian(_triangle()).harmonics()
        self.assertEqual(len(harmonics), 1)  # one harmonic, a degree-0 Cochain
        h = harmonics[0]
        self.assertEqual(h.degree(), 0)
        vec = np.asarray(h.coeffs())
        self.assertEqual(vec.size, n)  # length N over the vertex ordering
        np.testing.assert_allclose(np.abs(vec), np.full(n, abs(vec[0])), atol=1e-9)


# --------------------------------------------------------------------------- #
# C3 — gauge invariance (computed test-side; the operator has no gauge() method)
# --------------------------------------------------------------------------- #
class TestGaugeInvariance(unittest.TestCase):

    @staticmethod
    def _apply_gauge(st, alpha):
        # Rephase every edge: θ -> θ + α_src - α_tgt  (A -> G A G^†, L -> G L G^†).
        for e in st.getEdgeList().toVector():
            s, t = e.getSource().getId(), e.getTarget().getId()
            e.setPhase(e.getPhase() + alpha[s] - alpha[t])

    def test_testbed_spectrum_eigenvectors_and_flux_are_gauge_invariant(self):
        rng = np.random.default_rng(2718)
        st = _testbed()
        # Random Hermitian weights + base phases (a generic point in the b1=2
        # connection space).
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
            e.setPhase(float(rng.uniform(-math.pi, math.pi)))

        ids, idx = _ordering(st)
        n = len(ids)
        hl_old = cob.HodgeLaplacian(st)
        evals_old = _real_spectrum(hl_old.eigenvalues())
        V_old = _matrix(hl_old.eigenvectors(), n)

        # Two independent cycles of the testbed (b1 = 2).
        cycles = ([0, 1, 2], [0, 2, 3])
        flux_old = [_cycle_flux(st, c) for c in cycles]

        # Random vertex-phase gauge.
        alpha = {vid: float(rng.uniform(-math.pi, math.pi)) for vid in ids}
        self._apply_gauge(st, alpha)

        hl_new = cob.HodgeLaplacian(st)
        evals_new = _real_spectrum(hl_new.eigenvalues())
        V_new = _matrix(hl_new.eigenvectors(), n)

        # (i) spectrum unchanged
        np.testing.assert_allclose(evals_new, evals_old, atol=1e-12)

        # (ii) eigenvectors rephased v -> G v: the spectral projector of every
        # eigenvalue cluster transforms as P -> G P G^† (robust to the
        # eigenvector phase/degeneracy ambiguity).
        g = np.array([np.exp(1j * alpha[vid]) for vid in ids])
        G = np.diag(g)
        for (a, b) in _cluster_ranges(evals_old):
            p_old = V_old[:, a:b] @ V_old[:, a:b].conj().T
            p_new = V_new[:, a:b] @ V_new[:, a:b].conj().T
            np.testing.assert_allclose(p_new, G @ p_old @ G.conj().T, atol=1e-9)

        # (iii) every cycle flux unchanged
        flux_new = [_cycle_flux(st, c) for c in cycles]
        np.testing.assert_allclose(flux_new, flux_old, atol=1e-10)

    def test_tree_eigenvectors_rephase_vectorwise(self):
        # On the tree (non-degenerate {0,1,3}) the per-eigenvector statement
        # v_k -> G v_k holds directly (up to a global phase).
        rng = np.random.default_rng(99)
        st = _path()
        ids, idx = _ordering(st)
        n = len(ids)
        hl_old = cob.HodgeLaplacian(st)
        V_old = _matrix(hl_old.eigenvectors(), n)

        alpha = {vid: float(rng.uniform(-math.pi, math.pi)) for vid in ids}
        TestGaugeInvariance._apply_gauge(st, alpha)
        hl_new = cob.HodgeLaplacian(st)
        np.testing.assert_allclose(_real_spectrum(hl_new.eigenvalues()),
                                   _real_spectrum(hl_old.eigenvalues()), atol=1e-12)
        V_new = _matrix(hl_new.eigenvectors(), n)

        g = np.array([np.exp(1j * alpha[vid]) for vid in ids])
        for k in range(n):
            gv = g * V_old[:, k]
            overlap = abs(np.vdot(gv, V_new[:, k]))  # parallel <=> |overlap| = 1
            self.assertAlmostEqual(overlap, 1.0, places=8)


# --------------------------------------------------------------------------- #
# b1 cross-check against the topological oracle, and degree parameterization
# --------------------------------------------------------------------------- #
class TestBettiCrossCheck(unittest.TestCase):

    def test_first_betti_numbers(self):
        for name, st, expected in (("triangle", _triangle(), 1),
                                   ("path", _path(), 0),
                                   ("testbed", _testbed(), 2)):
            with self.subTest(fixture=name):
                cc = cob.ChainComplex.fromSpacetime(st)
                self.assertEqual(cc.bettiNumbers()[1], expected)


class TestDegreeParameterization(unittest.TestCase):

    def test_negative_degree_raises(self):
        hl = cob.HodgeLaplacian(_triangle())
        for call in (lambda: hl.laplacian(-1),
                     lambda: hl.eigenvalues(-1),
                     lambda: hl.eigenvectors(-2),
                     lambda: hl.harmonics(-1)):
            with self.subTest(call=call):
                with self.assertRaises(RuntimeError):
                    call()

    def test_degree_above_top_dimension_is_empty(self):
        # The triangle is S^1 (top dimension 1): there are no 2- or 3-cells, so
        # L_k is the empty operator (no raise) and ker L_k is trivial.
        hl = cob.HodgeLaplacian(_triangle())
        for k in (2, 3):
            with self.subTest(k=k):
                self.assertEqual(hl.laplacian(k), [])
                self.assertEqual(hl.eigenvalues(k), [])
                self.assertEqual(hl.eigenvectors(k), [])
                self.assertEqual(hl.harmonics(k), [])

    def test_k_zero_is_the_default(self):
        hl = cob.HodgeLaplacian(_triangle())
        np.testing.assert_allclose(np.array(hl.eigenvalues()),
                                   np.array(hl.eigenvalues(0)), atol=1e-12)

    def test_k_zero_ignores_metric_flag(self):
        # The k=0 path is the edge-weighted graph Laplacian regardless of metric.
        hl = cob.HodgeLaplacian(_testbed())
        np.testing.assert_allclose(np.array(hl.eigenvalues(0, True)),
                                   np.array(hl.eigenvalues(0, False)), atol=1e-12)


# --------------------------------------------------------------------------- #
# Metric Hodge Laplacian at k >= 1 (#104)
# --------------------------------------------------------------------------- #
class TestMetricHodgeAssembly(unittest.TestCase):
    """The C++ assembly equals the independent numpy oracle, for both the metric
    (volume) and combinatorial (unit) weightings, and its eigenvalues match."""

    CASES = (("triangle", _triangle, 1), ("path", _path, 1),
             ("testbed", _testbed, 1), ("torus k=1", _torus, 1),
             ("torus k=2", _torus, 2))

    def test_assembly_matches_numpy_oracle(self):
        for name, build, k in self.CASES:
            st = build()
            for metric in (True, False):
                with self.subTest(case=name, metric=metric):
                    nk = cob.ChainComplex.fromSpacetime(st).numSimplices(k)
                    L_cpp = np.array(cob.HodgeLaplacian(st).laplacian(k, metric),
                                     dtype=complex).reshape(nk, nk)
                    L_ref = _np_metric_laplacian(st, k, metric)
                    np.testing.assert_allclose(L_cpp, L_ref, atol=1e-10)

    def test_eigenvalues_match_numpy(self):
        for name, build, k in self.CASES:
            st = build()
            for metric in (True, False):
                with self.subTest(case=name, metric=metric):
                    L_ref = _np_metric_laplacian(st, k, metric)
                    evals = np.array(cob.HodgeLaplacian(st).eigenvalues(k, metric))
                    # The signed-weight operator is generally NOT self-adjoint, so
                    # eigvals (not eigvalsh) and no PSD claim. Both sides are
                    # sorted by (Re, Im) to compare set-wise.
                    def _key(z):
                        return (round(z.real, 9), round(z.imag, 9))
                    np.testing.assert_allclose(
                        sorted(evals, key=_key),
                        sorted(np.linalg.eigvals(L_ref), key=_key), atol=1e-9)

    def test_assembly_matches_oracle_with_random_weights(self):
        # Non-uniform positive edge weights make every W_k a genuine non-scalar
        # diagonal, exercising the full W^{1/2} formula and the column ordering.
        rng = np.random.default_rng(104)
        for name, build, k in (("testbed", _testbed, 1), ("torus k=1", _torus, 1),
                               ("torus k=2", _torus, 2)):
            st = build()
            for e in st.getEdgeList().toVector():
                e.setLength(cmath.sqrt(complex(float(rng.uniform(0.3, 3.0)))))
                e.setPhase(0.0)
            with self.subTest(case=name):
                nk = cob.ChainComplex.fromSpacetime(st).numSimplices(k)
                L_cpp = np.array(cob.HodgeLaplacian(st).laplacian(k, True),
                                 dtype=complex).reshape(nk, nk)
                np.testing.assert_allclose(L_cpp, _np_metric_laplacian(st, k, True),
                                           atol=1e-10)


class TestMetricHodgeKernel(unittest.TestCase):
    """The discrete Hodge theorem: dim ker L_k = b_k, for metric and
    combinatorial weights alike, cross-checked against ChainComplex."""

    def test_first_betti_is_first_harmonic_dimension(self):
        for name, build in (("triangle", _triangle), ("path", _path),
                            ("testbed", _testbed), ("torus", _torus)):
            st = build()
            b1 = _betti(st, 1)
            with self.subTest(fixture=name):
                for metric in (True, False):
                    self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, metric), b1)
                    self.assertEqual(_harmonic_dim(st, 1, metric), b1)

    def test_torus_first_homology_is_the_qubit(self):
        # T^2: dim ker L_1 == 2 (spec sec 5.2), == b_1 from ChainComplex.
        st = _torus()
        self.assertEqual(_betti(st, 1), 2)
        for metric in (True, False):
            with self.subTest(metric=metric):
                self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, metric), 2)
                self.assertEqual(_harmonic_dim(st, 1, metric), 2)

    def test_higher_degree_betti_on_the_torus(self):
        # b_2 = 1 (the fundamental class): dim ker L_2 == 1.
        st = _torus()
        self.assertEqual(_betti(st, 2), 1)
        for metric in (True, False):
            with self.subTest(metric=metric):
                self.assertEqual(_kernel_dim_from_eigenvalues(st, 2, metric), 1)
                self.assertEqual(_harmonic_dim(st, 2, metric), 1)

    def test_random_metric_weights_preserve_the_kernel_dimension(self):
        # ker L_k = b_k for ANY positive weights (the metric only moves the
        # representatives and eigenvalues, not the kernel dimension).
        rng = np.random.default_rng(2)
        st = _torus()
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.3, 3.0)))))
            e.setPhase(0.0)
        self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, True), 2)
        self.assertEqual(_harmonic_dim(st, 1, True), 2)


class TestMetricWeights(unittest.TestCase):
    """weights(k) is the per-k-simplex volume diagonal, in ChainComplex order."""

    def test_vertex_weights_are_unit(self):
        st = _testbed()
        np.testing.assert_allclose(np.array(cob.HodgeLaplacian(st).weights(0)),
                                   np.ones(4), atol=1e-12)

    def test_edge_weights_are_squared_length_in_column_order(self):
        # Distinct edge lengths pin down both the values (the V^2 weight of an
        # edge is exactly l^2) and the canonical sorted-vertex-id column order.
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        lengths = {(0, 1): 1.0, (0, 2): 4.0, (0, 3): 9.0, (1, 2): 16.0, (2, 3): 25.0}
        for e in st.getEdgeList().toVector():
            key = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
            e.setLength(cmath.sqrt(complex(lengths[key])))
            e.setPhase(0.0)
        order = sorted(lengths)  # (0,1),(0,2),(0,3),(1,2),(2,3)
        expected = [lengths[t] for t in order]
        np.testing.assert_allclose(np.array(cob.HodgeLaplacian(st).weights(1)),
                                   expected, atol=1e-12)

    def test_weights_out_of_range_are_empty(self):
        st = _triangle()  # S^1, top dimension 1
        self.assertEqual(cob.HodgeLaplacian(st).weights(-1), [])
        self.assertEqual(cob.HodgeLaplacian(st).weights(5), [])


if __name__ == "__main__":
    unittest.main()
