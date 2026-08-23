# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Recursive static and shifted response reduction (#768).

Exact identities under test, with their domains:

  static      L_eff = L_BB - L_BI L_II^+ L_IB (supported; interior kernels
              RETAINED, never regularized) -- minimization certificate in the
              positive self-adjoint regime, stationarity in the Hermitian-
              indefinite regime, certified block elimination with the
              left-kernel compatibility check in the non-normal regime;
  band        F_B(lam) = L_BB - lam I - L_BI (L_II - lam I)^{-1} L_IB with
              det(L - lam) = det(L_II - lam) det F_B(lam) exactly; algebraic
              multiplicity = det-winding (response + interior reported
              separately), geometric = dim ker F_B(lam);
  surrogate   Craig-Bampton retained-mode basis with declared window,
              discarded-mode gap, and fine-space eigenresiduals;
  next level  abstract labeled sum of retained fibers with embedding J and
              Gram G = J^dag W J under ONE declared policy; operator-valued
              response network; sheaf realization emitted only when certified.

Hand-solved path/star/triangle fixtures carry literal expected matrices;
independent NumPy pinv-Schur / Feshbach references are computed separately
from the C++ path; the block-pencil fixture demonstrates that plain static
Schur does NOT preserve nonzero eigenvalues; a defective pencil shows
algebraic != geometric multiplicity; an overlapping-interface fixture shows a
naive internal direct sum miscounts while the declared policy stays exact.
"""

import cmath
import os
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _causal_specimen import load_dump, rebuild_spacetime  # noqa: E402

MACHINE = 1e-12  # fp-equality across different (exact) arithmetic orders


# --------------------------------------------------------------------------
# helpers: independent NumPy references (separate from the C++ path)
# --------------------------------------------------------------------------
def _flat(matrix):
    return [complex(z) for z in np.asarray(matrix, dtype=complex).reshape(-1)]


def _mat(flat, rows, cols=None):
    flat = np.asarray(flat, dtype=complex)
    cols = rows if cols is None else cols
    return flat.reshape(rows, cols)


def numpy_schur(L, kept, interior):
    """Independent dense reference: L_BB - L_BI pinv(L_II) L_IB."""
    L = np.asarray(L, dtype=complex)
    kept = list(kept)
    interior = list(interior)
    if not interior:
        return L[np.ix_(kept, kept)]
    L_BB = L[np.ix_(kept, kept)]
    L_BI = L[np.ix_(kept, interior)]
    L_IB = L[np.ix_(interior, kept)]
    L_II = L[np.ix_(interior, interior)]
    return L_BB - L_BI @ np.linalg.pinv(L_II) @ L_IB

def numpy_feshbach(L, kept, interior, lam):
    """Independent dense reference: F_B(lam)."""
    L = np.asarray(L, dtype=complex)
    kept = list(kept)
    interior = list(interior)
    F = L[np.ix_(kept, kept)] - lam * np.eye(len(kept))
    if interior:
        L_BI = L[np.ix_(kept, interior)]
        L_IB = L[np.ix_(interior, kept)]
        L_II = L[np.ix_(interior, interior)]
        F = F - L_BI @ np.linalg.solve(
            L_II - lam * np.eye(len(interior)), L_IB)
    return F


def build_graph(edges):
    """Spacetime holding an explicit weighted graph (the
    benchmark_analytic_kernel.py convention); edges are
    (src, tgt, squared_length, phase). Vertices are exactly the edge
    endpoints."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = {}
    for src, tgt, _, _ in edges:
        for v in (src, tgt):
            if v not in verts:
                verts[v] = st.createVertex(v)
    for src, tgt, _, _ in edges:
        st.createSimplex([verts[src], verts[tgt]])
    by_pair = {(e.getSource().getId(), e.getTarget().getId()): e
               for e in st.getEdgeList().toVector()}
    for src, tgt, squared_length, phase in edges:
        edge = by_pair.get((src, tgt))
        sign = 1.0
        if edge is None:
            edge = by_pair[(tgt, src)]
            sign = -1.0
        edge.setLength(cmath.sqrt(complex(squared_length)))
        edge.setPhase(sign * phase)
    return st


def reduction_matrix(quotient):
    read = quotient.staticReduction()
    n = len(read.coordinates)
    return _mat(read.effectiveOperator, n), read


# --------------------------------------------------------------------------
# hand-solved static fixtures (literal expected matrices)
# --------------------------------------------------------------------------
class TestHandSolvedPath(unittest.TestCase):
    # P3 path 0-1-2 with edge weights w1 = 2, w2 = 3:
    # L = [[ 2, -2,  0],
    #      [-2,  5, -3],
    #      [ 0, -3,  3]].
    L = [[2, -2, 0], [-2, 5, -3], [0, -3, 3]]

    def _quotient(self):
        # Middle vertex interior: one covering component plus endpoint
        # claims making 0 and 2 interface.
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 3, [], [[0, 1, 2], [0], [2]])

    def test_kron_reduction_matches_hand_matrix(self):
        # Series conductance w_eff = w1 w2 / (w1 + w2) = 6/5 EXACTLY:
        # L_eff = [[ 6/5, -6/5],
        #          [-6/5,  6/5]].
        q = self._quotient()
        self.assertEqual(list(q.interfaceIndices), [0, 2])
        E, read = reduction_matrix(q)
        expected = np.array([[1.2, -1.2], [-1.2, 1.2]], dtype=complex)
        np.testing.assert_allclose(E, expected, rtol=0, atol=MACHINE)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.StructureExact)
        self.assertEqual(read.certificate.domain, cob.CertificateDomain.Static)
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.PositiveSemidefinite)

    def test_matches_independent_numpy_pinv_schur(self):
        E, _ = reduction_matrix(self._quotient())
        np.testing.assert_allclose(E, numpy_schur(self.L, [0, 2], [1]),
                                   rtol=0, atol=MACHINE)

    def test_minimization_certificate_on_probes(self):
        q = self._quotient()
        cert = q.verifyStatic()
        self.assertTrue(cert.holds())
        self.assertLess(cert.residual, 1e-12)
        # And on an explicit probe: min_x [b;x]^dag L [b;x] = b^dag L_eff b.
        cert = q.staticProbeCertificate([1.0 + 0j, -2.0 + 0j])
        self.assertTrue(cert.holds())

    def test_brute_force_interior_minimum_equals_coarse_energy(self):
        # Independent check of the MINIMIZATION claim itself: scan the
        # interior coordinate on a grid around the analytic minimizer.
        E, _ = reduction_matrix(self._quotient())
        b = np.array([1.0, -2.0])
        coarse = float(np.real(b @ E.real @ b))
        L = np.array(self.L, dtype=float)
        energies = []
        for x in np.linspace(-3.0, 3.0, 2001):
            v = np.array([b[0], x, b[1]])
            energies.append(v @ L @ v)
        # The grid minimum upper-bounds the true minimum to O(step^2).
        self.assertAlmostEqual(min(energies), coarse, places=4)
        self.assertLessEqual(coarse, min(energies) + 1e-12)


class TestHandSolvedStar(unittest.TestCase):
    # Star: center 0, leaves 1,2,3 with weights 1,2,3 (sum W = 6):
    # L = [[ 6, -1, -2, -3],
    #      [-1,  1,  0,  0],
    #      [-2,  0,  2,  0],
    #      [-3,  0,  0,  3]].
    # Eliminating the center: (L_eff)_ij = delta_ij w_i - w_i w_j / W:
    # L_eff = [[ 5/6, -1/3, -1/2],
    #          [-1/3,  4/3, -1  ],
    #          [-1/2, -1,    3/2]].
    L = [[6, -1, -2, -3], [-1, 1, 0, 0], [-2, 0, 2, 0], [-3, 0, 0, 3]]
    EXPECTED = np.array([[5.0 / 6.0, -1.0 / 3.0, -0.5],
                         [-1.0 / 3.0, 4.0 / 3.0, -1.0],
                         [-0.5, -1.0, 1.5]], dtype=complex)

    def _quotient(self):
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1, 2, 3], [1], [2], [3]])

    def test_star_reduction_matches_hand_matrix(self):
        q = self._quotient()
        self.assertEqual(list(q.interfaceIndices), [1, 2, 3])
        self.assertEqual(list(q.interiorIndices(0)), [0])
        E, read = reduction_matrix(q)
        np.testing.assert_allclose(E, self.EXPECTED, rtol=0, atol=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_matches_independent_numpy_pinv_schur(self):
        E, _ = reduction_matrix(self._quotient())
        np.testing.assert_allclose(E, numpy_schur(self.L, [1, 2, 3], [0]),
                                   rtol=0, atol=MACHINE)

    def test_row_sums_vanish(self):
        # Kron reduction of a Laplacian is a Laplacian: zero row sums.
        E, _ = reduction_matrix(self._quotient())
        np.testing.assert_allclose(E.sum(axis=1), 0, rtol=0, atol=MACHINE)


class TestHandSolvedTriangle(unittest.TestCase):
    # Unit triangle 0-1-2; eliminate vertex 2: the 1-1 series path through 2
    # (conductance 1/2) in parallel with the direct edge (1):
    # L_eff = [[ 3/2, -3/2],
    #          [-3/2,  3/2]].
    L = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]

    def _quotient(self):
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 3, [], [[0, 1, 2], [0], [1]])

    def test_triangle_reduction_matches_hand_matrix(self):
        E, read = reduction_matrix(self._quotient())
        expected = np.array([[1.5, -1.5], [-1.5, 1.5]], dtype=complex)
        np.testing.assert_allclose(E, expected, rtol=0, atol=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_matches_independent_numpy_pinv_schur(self):
        E, _ = reduction_matrix(self._quotient())
        np.testing.assert_allclose(E, numpy_schur(self.L, [0, 1], [2]),
                                   rtol=0, atol=MACHINE)

    def test_dense_reference_cross_check(self):
        # DenseReference (the #764 dense oracle) solving L_II X = L_IB gives
        # the same effective operator.
        dense = cob.DenseReference()
        x = dense.solve([2.0 + 0j], 1, [-1.0 + 0j])  # L_II = [2], col of L_IB
        self.assertTrue(x.certificate.holds())
        solved = x.values[0]
        L = np.array(self.L, dtype=complex)
        E_ref = L[:2, :2] - np.outer(L[:2, 2], [solved, solved])
        E, _ = reduction_matrix(self._quotient())
        np.testing.assert_allclose(E, E_ref, rtol=0, atol=MACHINE)


class TestSpacetimeHandFixtures(unittest.TestCase):
    def test_triangle_spacetime_k0_matches_matrix_path(self):
        # The same unit triangle built as a Spacetime (k = 0 Hermitian graph
        # Laplacian), reduced over explicit cell sets.
        st = build_graph([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0),
                             (0, 2, 1.0, 0.0)])
        q = cob.RecursiveQuotient.overCells(
            st, 0, [[[0], [1], [2]], [[0]], [[1]]])
        self.assertEqual(q.regime, cob.CertificateRegime.PositiveSemidefinite)
        E, read = reduction_matrix(q)
        expected = np.array([[1.5, -1.5], [-1.5, 1.5]], dtype=complex)
        np.testing.assert_allclose(E, expected, rtol=0, atol=MACHINE)
        self.assertEqual([c.provenance for c in read.coordinates],
                         ["cell(0)", "cell(1)"])

    # Two disjoint filled triangles bridged by the edge (2,3): the k = 1
    # Hodge operator couples edges through shared vertices (down term) and
    # shared triangle cofaces (up term). Triangulated strip: triangles
    # (i, i+1, i+2) for i = 0..5 over vertices 0..7 (13 edges); supports
    # {0..3} and {4..7}: only edge (0,1) is interior to A and only (6,7) to
    # B; the straddling triangles' edges join the residual component and
    # everything vertex-adjacent to them is interface.
    STRIP = [[i, i + 1, i + 2] for i in range(6)]
    SUPPORT_A = [0, 1, 2, 3]
    SUPPORT_B = [4, 5, 6, 7]

    def _strip(self):
        st = tessera.Spacetime.fromCells(2, self.STRIP, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(1.0)))
        st.materializeFacets()
        return st

    def test_degree_one_classification_and_certificate(self):
        st = self._strip()
        q = cob.RecursiveQuotient.overVertexSupports(
            st, 1, [self.SUPPORT_A, self.SUPPORT_B])
        self.assertEqual(q.dimension, 13)
        self.assertEqual(q.componentCount, 3)  # two supports + residual
        self.assertEqual(len(q.interfaceIndices), 11)
        # Interior edges by PROVENANCE (vertex sets, never index order).
        interior_a = [q.coordinateProvenance[i] for i in q.interiorIndices(0)]
        interior_b = [q.coordinateProvenance[i] for i in q.interiorIndices(1)]
        self.assertEqual(interior_a, ["cell(0,1)"])
        self.assertEqual(interior_b, ["cell(6,7)"])
        self.assertEqual(list(q.interiorIndices(2)), [])  # residual cells
        cert = q.verifyStatic()
        self.assertTrue(cert.holds(), cert.describe())

    def test_degree_one_matches_numpy_schur_on_the_hodge_operator(self):
        st = self._strip()
        hodge = cob.HodgeLaplacian(st)
        L = _mat(hodge.laplacian(1), 13)
        q = cob.RecursiveQuotient.overVertexSupports(
            st, 1, [self.SUPPORT_A, self.SUPPORT_B])
        kept = list(q.interfaceIndices)
        interior = sorted(set(range(13)) - set(kept))
        self.assertEqual(len(interior), 2)
        E, _ = reduction_matrix(q)
        np.testing.assert_allclose(E[:len(kept), :len(kept)],
                                   numpy_schur(L, kept, interior),
                                   rtol=0, atol=1e-10)


# --------------------------------------------------------------------------
# integer interior nullspaces + retention of harmonic modes
# --------------------------------------------------------------------------
class TestIntegerNullspace(unittest.TestCase):
    def test_literal_kernel_basis(self):
        # M = [[1, 1, 0]]: kernel = span{(-1, 1, 0), (0, 0, 1)}.
        basis = cob.integer_nullspace([1, 1, 0], 1, 3)
        self.assertEqual(basis, [[-1, 1, 0], [0, 0, 1]])

    def test_rational_elimination_clears_to_coprime_integers(self):
        # M = [[2, 4], [1, 2]]: rank 1, kernel = span{(-2, 1)}.
        basis = cob.integer_nullspace([2, 4, 1, 2], 2, 2)
        self.assertEqual(basis, [[-2, 1]])

    def test_full_rank_matrix_has_empty_kernel(self):
        self.assertEqual(cob.integer_nullspace([1, 0, 0, 1], 2, 2), [])

    def test_kernel_vectors_annihilate_exactly(self):
        rng = np.random.default_rng(7)
        M = rng.integers(-3, 4, size=(4, 6))
        basis = cob.integer_nullspace([int(x) for x in M.reshape(-1)], 4, 6)
        self.assertEqual(len(basis), 6 - np.linalg.matrix_rank(M))
        for vec in basis:
            self.assertTrue(np.all(M @ np.array(vec) == 0))


class TestHarmonicRetention(unittest.TestCase):
    def _two_triangles(self):
        # Triangle A on {0,1,2} coupled to nothing; triangle B on {10,11,12}
        # likewise: component B is fully interior and its constant mode is a
        # genuine interior harmonic (b_0) that MUST be retained in the fiber.
        return build_graph([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0),
                               (0, 2, 1.0, 0.0), (10, 11, 1.0, 0.0),
                               (11, 12, 1.0, 0.0), (10, 12, 1.0, 0.0)])

    def test_interior_harmonic_mode_is_retained_in_the_fiber(self):
        st = self._two_triangles()
        q = cob.RecursiveQuotient.overCells(
            st, 0,
            [[[0], [1], [2]], [[10], [11], [12]]])
        E, read = reduction_matrix(q)
        # No interface at all: the reduced space is exactly the two retained
        # harmonic (constant) modes, one per component.
        self.assertEqual(list(q.interfaceIndices), [])
        kinds = [c.kind for c in read.coordinates]
        self.assertEqual(kinds, [cob.RetainedCoordinateKind.Harmonic] * 2)
        self.assertEqual([c.component for c in read.coordinates], [0, 1])
        self.assertEqual([c.provenance for c in read.coordinates],
                         ["harmonic[c0#0]", "harmonic[c1#0]"])
        # PSD decoupling: the retained harmonic rows/cols vanish identically.
        np.testing.assert_allclose(E, 0, rtol=0, atol=MACHINE)
        # The retained embedding IS the constant mode on its component.
        emb = np.array(read.coordinates[1].embedding)
        support = np.nonzero(np.abs(emb) > 1e-14)[0]
        self.assertEqual(len(support), 3)
        np.testing.assert_allclose(np.abs(emb[support]), 1 / np.sqrt(3),
                                   rtol=0, atol=1e-14)

    def test_integer_and_numerical_nullities_agree(self):
        st = self._two_triangles()
        q = cob.RecursiveQuotient.overCells(
            st, 0, [[[0], [1], [2]], [[10], [11], [12]]])
        for component in range(2):
            read = q.interiorNullspace(component)
            self.assertEqual(read.nullity, 1)
            self.assertEqual(read.integerNullity, 1)
            # The exact integer topological basis is the constant vector.
            self.assertEqual([abs(x) for x in read.integerBasis[0]], [1, 1, 1])
            self.assertTrue(read.certificate.holds())

    def test_labeled_sum_retains_the_harmonic_fiber_coordinate(self):
        st = self._two_triangles()
        q = cob.RecursiveQuotient.overCells(
            st, 0, [[[0], [1], [2]], [[10], [11], [12]]])
        sum_read = q.labeledFiberSum()
        self.assertEqual(list(sum_read.summandComponents), [0, 1])
        self.assertEqual(list(sum_read.summandRanks), [1, 1])
        # Disjoint supports: the Gram IS the identity here.
        G = _mat(sum_read.gram, 2)
        np.testing.assert_allclose(G, np.eye(2), rtol=0, atol=1e-14)

    def test_matrix_path_singular_interior_matches_numpy_pinv(self):
        # PSD block with an interior kernel: two kept cells coupled through
        # a singular interior pair (the pinv reference must agree).
        L = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1],
                      [-1, 0, 1, 0],
                      [0, -1, 0, 1]], dtype=float)
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1, 2, 3], [0], [1]])
        E, read = reduction_matrix(q)
        kept_block = E[:2, :2]
        np.testing.assert_allclose(kept_block, numpy_schur(L, [0, 1], [2, 3]),
                                   rtol=0, atol=MACHINE)
        # Interior block diag(1,1) is invertible -> no kernel here; make one:
        L2 = np.array([[1, -1, 0, 0],
                       [-1, 2, -1, 0],
                       [0, -1, 1, 0],
                       [0, 0, 0, 0]], dtype=float)
        q2 = cob.RecursiveQuotient.overMatrix(
            _flat(L2), 4, [], [[0, 1, 2, 3], [0], [2]])
        E2, read2 = reduction_matrix(q2)
        self.assertEqual(len(read2.coordinates), 3)  # 2 kept + 1 harmonic
        self.assertEqual(read2.coordinates[2].kind,
                         cob.RetainedCoordinateKind.Harmonic)
        np.testing.assert_allclose(E2[:2, :2],
                                   numpy_schur(L2, [0, 2], [1, 3]),
                                   rtol=0, atol=MACHINE)


# --------------------------------------------------------------------------
# block pencil: static Schur does NOT preserve nonzero eigenvalues;
# the Feshbach pencil does (det factorization)
# --------------------------------------------------------------------------
class TestBlockPencilNegativeControl(unittest.TestCase):
    # Hermitian 4x4 with kept {0,1} and interior {2,3}.
    L = np.array([[2.0, 0.5, 1.0, 0.0],
                  [0.5, 3.0, 0.0, 1.0],
                  [1.0, 0.0, 4.0, 0.5],
                  [0.0, 1.0, 0.5, 5.0]])

    def _quotient(self):
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1, 2, 3], [0], [1]])

    def test_static_schur_fails_to_preserve_nonzero_eigenvalues(self):
        # The EXPLICIT negative control: eigenvalues of the static L_eff are
        # NOT eigenvalues of L (no nonzero-spectrum claim attaches to a
        # static reduction; Certificate.domain distinguishes this).
        E, read = reduction_matrix(self._quotient())
        self.assertEqual(read.certificate.domain, cob.CertificateDomain.Static)
        eff_eigs = np.sort(np.linalg.eigvalsh(E.real))
        full_eigs = np.sort(np.linalg.eigvalsh(self.L))
        for mu in eff_eigs:
            self.assertGreater(np.min(np.abs(full_eigs - mu)), 0.05,
                               "static Schur eigenvalue coincided with the "
                               "fine spectrum -- the negative control lost "
                               "its teeth")

    def test_feshbach_matches_numpy_reference_across_the_window(self):
        q = self._quotient()
        for lam in np.linspace(1.0, 2.6, 9):
            read = q.feshbach(complex(lam), 1.0, 2.6)
            F = _mat(read.response, 2)
            np.testing.assert_allclose(
                F, numpy_feshbach(self.L, [0, 1], [2, 3], lam),
                rtol=0, atol=1e-11)
            self.assertTrue(read.certificate.holds(), read.certificate.describe())
            self.assertEqual(read.certificate.domain,
                             cob.CertificateDomain.BandWindow)
            self.assertEqual(read.windowLower, 1.0)
            self.assertEqual(read.windowUpper, 2.6)

    def test_determinant_factorization_is_exact(self):
        q = self._quotient()
        for lam in (0.7, 1.9 + 0.3j, 2.5):
            read = q.feshbach(complex(lam), 0.0, 3.0)
            self.assertLess(read.determinantResidual, 1e-12)

    def test_true_eigenvalues_are_zeros_of_the_pencil(self):
        # lam in spec(L) <=> 0 in spec(F_B(lam)), away from spec(L_II).
        q = self._quotient()
        interior_eigs = np.linalg.eigvalsh(self.L[2:, 2:])
        for lam in np.linalg.eigvalsh(self.L):
            if np.min(np.abs(interior_eigs - lam)) < 1e-6:
                continue
            F = _mat(q.feshbach(complex(lam), 0.0, 6.0).response, 2)
            smallest = np.min(np.abs(np.linalg.eigvals(F)))
            self.assertLess(smallest, 1e-9)


# --------------------------------------------------------------------------
# multiplicity: algebraic (det winding) vs geometric (dim ker F_B)
# --------------------------------------------------------------------------
class TestMultiplicity(unittest.TestCase):
    def test_semisimple_double_eigenvalue(self):
        # diag(1, 1, 3) coupled weakly? No -- keep it EXACT: block diag with
        # a double eigenvalue 1 on the kept block, interior at 3.
        L = np.diag([1.0, 1.0, 3.0])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [1]])
        read = q.multiplicity(1.0 + 0j, 0.5)
        self.assertEqual(read.algebraic, 2)
        self.assertEqual(read.geometric, 2)
        self.assertTrue(read.semisimple)
        self.assertEqual(read.interiorWinding, 0)
        self.assertTrue(read.certificate.holds())

    def test_defective_pencil_reports_distinct_multiplicities(self):
        # Jordan block on the kept coordinates: algebraic 2, geometric 1.
        L = np.array([[1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 2.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [1]])
        self.assertEqual(q.regime, cob.CertificateRegime.NonNormal)
        read = q.multiplicity(1.0 + 0j, 0.5)
        self.assertEqual(read.responseWinding, 2)
        self.assertEqual(read.interiorWinding, 0)
        self.assertEqual(read.algebraic, 2)
        self.assertEqual(read.geometric, 1)
        self.assertFalse(read.semisimple)
        self.assertTrue(read.certificate.holds())

    def test_interior_winding_is_reported_separately(self):
        # At lam = 2 (an INTERIOR eigenvalue): det F_B has a pole there, so
        # the response winding alone would MISCOUNT; the interior winding
        # restores the exact algebraic multiplicity 1.
        L = np.array([[1.0, 1.0, 0.5],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 2.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [1]])
        read = q.multiplicity(2.0 + 0j, 0.3)
        self.assertEqual(read.interiorWinding, 1)
        self.assertEqual(read.responseWinding, 0)
        self.assertEqual(read.algebraic, 1)

    def test_contour_through_spectrum_is_refused(self):
        L = np.diag([1.0, 1.0, 3.0])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [1]])
        # radius 2.0 puts the interior eigenvalue 3 exactly on the contour
        # of center 1: det(L_II - z) hits 0 -> loud refusal.
        with self.assertRaises(Exception):
            q.multiplicity(1.0 + 0j, 2.0)


# --------------------------------------------------------------------------
# resonance and compatibility
# --------------------------------------------------------------------------
class TestResonanceAndCompatibility(unittest.TestCase):
    def test_resonant_shift_retains_the_resonant_mode(self):
        # P3 with weights 2,3: L_II = [5]; at lam = 5 the shifted interior
        # block is singular -> the resonant mode is retained explicitly.
        L = [[2, -2, 0], [-2, 5, -3], [0, -3, 3]]
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [2]])
        read = q.feshbach(5.0 + 0j, 4.0, 6.0)
        self.assertTrue(read.resonant)
        kinds = [c.kind for c in read.coordinates]
        self.assertEqual(kinds, [cob.RetainedCoordinateKind.Interface,
                                 cob.RetainedCoordinateKind.Interface,
                                 cob.RetainedCoordinateKind.Resonant])
        self.assertEqual(len(read.response), 9)
        # Compatibility: the load (-2, -3) is NOT orthogonal to the
        # resonant kernel (the whole 1-dim space) -> the violation is
        # reported honestly and the certificate does not hold.
        self.assertGreater(read.compatibilityResidual, 0.5)
        self.assertFalse(read.certificate.holds())

    def test_nonnormal_left_kernel_compatibility(self):
        # L_II = [[0, 1], [0, 2]]: right kernel e1, LEFT kernel (2, -1)/sqrt5.
        # Solvability of L_II x = load needs load perp (2, -1):
        # load = (p, q) compatible iff q = 2p.
        def build(p, q):
            L = np.array([[1.0, 0.3, -0.2],
                          [p, 0.0, 1.0],
                          [q, 0.0, 2.0]])
            return cob.RecursiveQuotient.overMatrix(
                _flat(L), 3, [], [[0, 1, 2], [0]])

        compatible = build(1.0, 2.0)
        self.assertEqual(compatible.regime, cob.CertificateRegime.NonNormal)
        read = compatible.staticReduction()
        self.assertLess(read.compatibilityResidual, 1e-12)
        self.assertTrue(read.certificate.holds(), read.certificate.describe())

        incompatible = build(1.0, 0.0)
        read = incompatible.staticReduction()
        self.assertGreater(read.compatibilityResidual, 0.1)
        self.assertFalse(read.certificate.holds())

    def test_defective_interior_is_refused_not_regularized(self):
        # Nilpotent interior block: the kernel-complement block is singular
        # (Jordan structure at the shift). No diagonal regularizer: the
        # certificate refuses.
        L = np.array([[1.0, 0.5, 0.5],
                      [0.4, 0.0, 1.0],
                      [0.6, 0.0, 0.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0]])
        read = q.staticReduction()
        self.assertFalse(read.certificate.holds())

    def test_incompatible_probe_certificate_refuses(self):
        # Probe-level negative control: the SAME non-normal fixture, but the
        # verification probe itself carries the incompatible load.
        L = np.array([[1.0, 0.3, -0.2],
                      [1.0, 0.0, 1.0],
                      [0.0, 0.0, 2.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0]])
        cert = q.staticProbeCertificate([1.0 + 0j])
        self.assertFalse(cert.holds())
        self.assertGreater(cert.residual, 0.1)

    def test_hermitian_indefinite_regime_uses_stationarity(self):
        # W = diag(1, -1, 1) with L = W H (H Hermitian): WL = H is Hermitian
        # while L itself is not -> the HermitianIndefinite regime, whose
        # static claim is stationarity (verified on probes), not a minimum.
        H = np.array([[2.0, -1.0, 0.0],
                      [-1.0, 3.0, -1.0],
                      [0.0, -1.0, 2.0]])
        W = np.diag([1.0, -1.0, 1.0])
        L = W @ H
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [1.0 + 0j, -1.0 + 0j, 1.0 + 0j],
            [[0, 1, 2], [0], [2]])
        self.assertEqual(q.regime, cob.CertificateRegime.HermitianIndefinite)
        cert = q.verifyStatic()
        self.assertTrue(cert.holds(), cert.describe())
        E, _ = reduction_matrix(q)
        np.testing.assert_allclose(E, numpy_schur(L, [0, 2], [1]),
                                   rtol=0, atol=MACHINE)


# --------------------------------------------------------------------------
# overlapping interfaces and the labeled sum
# --------------------------------------------------------------------------
class TestOverlappingInterfaceLabeledSum(unittest.TestCase):
    # P5 path graph; components {0,1,2} and {2,3,4} SHARE interface cell 2.
    L = np.array([[1.0, -1.0, 0.0, 0.0, 0.0],
                  [-1.0, 2.0, -1.0, 0.0, 0.0],
                  [0.0, -1.0, 2.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0, 2.0, -1.0],
                  [0.0, 0.0, 0.0, -1.0, 1.0]])

    def _quotient(self, policy):
        options = cob.RecursiveQuotient.Options()
        options.embeddingPolicy = policy
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 5, [], [[0, 1, 2], [2, 3, 4]], options)

    def test_shared_cell_is_interface_and_both_fibers_carry_it(self):
        q = self._quotient(cob.FiberEmbeddingPolicy.CarryGramExactly)
        self.assertEqual(list(q.interfaceIndices), [2])
        self.assertEqual(list(q.interiorIndices(0)), [0, 1])
        self.assertEqual(list(q.interiorIndices(1)), [3, 4])
        read = q.labeledFiberSum()
        self.assertEqual(list(read.summandComponents), [0, 1])
        self.assertEqual(list(read.summandRanks), [1, 1])
        # Both summand columns embed to the SAME chain-space cell.
        J = _mat(read.embedding, 5, 2)
        np.testing.assert_allclose(J[:, 0], J[:, 1], rtol=0, atol=1e-15)

    def test_naive_direct_sum_miscounts_while_the_gram_is_exact(self):
        q = self._quotient(cob.FiberEmbeddingPolicy.CarryGramExactly)
        read = q.labeledFiberSum()
        G = _mat(read.gram, 2)
        # G = [[1, 1], [1, 1]] EXACTLY: the images coincide.
        np.testing.assert_allclose(G, np.ones((2, 2)), rtol=0, atol=1e-15)
        # Naive internal direct sum would claim rank 2; the Gram says 1.
        self.assertEqual(read.nominalRank, 2)
        self.assertEqual(read.quotientNullity, 1)
        # Independent recomputation of G = J^dag W J (W = I).
        J = _mat(read.embedding, 5, 2)
        np.testing.assert_allclose(G, J.conj().T @ J, rtol=0, atol=1e-15)
        # Declared option CarryGramExactly is exact by construction.
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.effectiveRank, 2)

    def test_near_isometry_policy_refuses_the_overlap(self):
        q = self._quotient(cob.FiberEmbeddingPolicy.CertifiedNearIsometry)
        read = q.labeledFiberSum()
        self.assertAlmostEqual(read.gramDefect, 1.0, places=12)
        self.assertFalse(read.certificate.holds())

    def test_quotient_kernel_policy_restates_the_rank(self):
        q = self._quotient(cob.FiberEmbeddingPolicy.QuotientKernel)
        read = q.labeledFiberSum()
        self.assertEqual(read.policy, cob.FiberEmbeddingPolicy.QuotientKernel)
        self.assertEqual(read.effectiveRank, 1)
        self.assertEqual(read.quotientNullity, 1)
        basis = _mat(read.quotientBasis, 2, 1)
        # The kept direction is the symmetric combination (the kernel of G
        # is the antisymmetric one).
        np.testing.assert_allclose(np.abs(basis), np.full((2, 1), 1 / np.sqrt(2)),
                                   rtol=0, atol=1e-12)

    def test_near_isometry_epsilon_bounds_the_amplitude_error(self):
        # The whitepaper inequality |a^dag G b - a^dag b| <= eps ||a|| ||b||
        # with eps = the reported gramDefect (a 2-norm statement), checked
        # on random vectors against the read's own numbers.
        q = self._quotient(cob.FiberEmbeddingPolicy.CertifiedNearIsometry)
        read = q.labeledFiberSum()
        G = _mat(read.gram, int(read.nominalRank))
        rng = np.random.default_rng(5)
        for _ in range(25):
            a = rng.normal(size=2) + 1j * rng.normal(size=2)
            b = rng.normal(size=2) + 1j * rng.normal(size=2)
            error = abs(np.conj(a) @ G @ b - np.conj(a) @ b)
            bound = read.gramDefect * np.linalg.norm(a) * np.linalg.norm(b)
            self.assertLessEqual(error, bound + 1e-12)

    def test_amplitude_budget_composes(self):
        # eps_AB <= eps_A + eps_B + eps_A eps_B, and the tensor Gram of two
        # near-isometries actually obeys the composed budget.
        compose = cob.RecursiveQuotient.composeNearIsometryBudget
        self.assertAlmostEqual(compose(0.1, 0.2), 0.32, places=15)
        self.assertEqual(compose(0.0, 0.0), 0.0)
        rng = np.random.default_rng(9)
        for _ in range(10):
            def near_isometry(n, scale):
                defect = scale * rng.normal(size=(n, n))
                defect = 0.5 * (defect + defect.T)
                return np.eye(n) + defect
            G_a = near_isometry(2, 0.05)
            G_b = near_isometry(3, 0.03)
            eps_a = np.linalg.norm(G_a - np.eye(2), 2)
            eps_b = np.linalg.norm(G_b - np.eye(3), 2)
            tensor = np.kron(G_a, G_b)
            eps_ab = np.linalg.norm(tensor - np.eye(6), 2)
            self.assertLessEqual(eps_ab, compose(eps_a, eps_b) + 1e-12)

    def test_disjoint_fibers_give_identity_gram(self):
        # Negative control for the overlap: partition WITHOUT overlap keeps
        # the labeled sum an honest (external) direct sum, G = I.
        options = cob.RecursiveQuotient.Options()
        options.embeddingPolicy = cob.FiberEmbeddingPolicy.CertifiedNearIsometry
        q = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 5, [], [[0, 1], [2], [3, 4]], options)
        read = q.labeledFiberSum()
        G = _mat(read.gram, int(read.nominalRank))
        np.testing.assert_allclose(G, np.eye(int(read.nominalRank)),
                                   rtol=0, atol=1e-14)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.quotientNullity, 0)


# --------------------------------------------------------------------------
# response network and sheaf realization
# --------------------------------------------------------------------------
class TestResponseNetworkAndSheaf(unittest.TestCase):
    # P4 path; components {0,1} and {2,3}: interiors {0} and {3}, kept
    # cells {1, 2}; L_eff = [[1, -1], [-1, 1]] (a graph Laplacian).
    L = np.array([[1.0, -1.0, 0.0, 0.0],
                  [-1.0, 2.0, -1.0, 0.0],
                  [0.0, -1.0, 2.0, -1.0],
                  [0.0, 0.0, -1.0, 1.0]])

    def _quotient(self):
        return cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1], [2, 3]])

    def test_network_blocks_tile_the_reduced_operator(self):
        q = self._quotient()
        E, _ = reduction_matrix(q)
        np.testing.assert_allclose(E, [[1, -1], [-1, 1]], rtol=0, atol=MACHINE)
        network = q.responseNetwork()
        self.assertEqual(list(network.stalkDimensions), [1, 1])
        self.assertEqual(len(network.edges), 2)  # both directions
        block = np.array(network.edges[0].block)
        np.testing.assert_allclose(block, [-1.0], rtol=0, atol=MACHINE)
        self.assertEqual(network.coverageResidual, 0.0)
        self.assertTrue(network.certificate.holds())

    def test_sheaf_realization_is_emitted_and_reproduces_blocks(self):
        q = self._quotient()
        sheaf = q.sheafRealization()
        self.assertTrue(sheaf.emitted)
        self.assertTrue(sheaf.simplicial)
        self.assertEqual(list(sheaf.edgeStalkDimensions), [1])
        self.assertLess(sheaf.reconstructionResidual, 1e-12)
        # rho_u^dag rho_v == -L_uv == 1 for the unit effective edge.
        rho_u = np.array(sheaf.restrictionMaps[0])
        rho_v = np.array(sheaf.restrictionMaps[1])
        np.testing.assert_allclose(rho_u.conj() * rho_v, [1.0],
                                   rtol=0, atol=1e-12)

    def test_sheaf_is_refused_when_blocks_are_not_reproduced(self):
        # Ground the ends (diagonal excess): the reduced operator is no
        # longer a sheaf Laplacian; the realization is NOT invented.
        L = self.L + np.diag([2.0, 0.0, 0.0, 2.0])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1], [2, 3]])
        sheaf = q.sheafRealization()
        self.assertFalse(sheaf.emitted)
        self.assertFalse(sheaf.certificate.holds())
        self.assertGreater(sheaf.reconstructionResidual, 0.1)
        self.assertEqual(len(sheaf.restrictionMaps), 0)

    def test_sheaf_is_refused_in_the_non_normal_regime(self):
        L = np.array([[1.0, 0.7, 0.0, 0.0],
                      [-0.2, 2.0, -1.0, 0.0],
                      [0.0, -1.0, 2.0, -1.0],
                      [0.0, 0.0, -1.0, 1.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1], [2, 3]])
        self.assertEqual(q.regime, cob.CertificateRegime.NonNormal)
        sheaf = q.sheafRealization()
        self.assertFalse(sheaf.emitted)
        self.assertFalse(sheaf.certificate.holds())


# --------------------------------------------------------------------------
# nested quotients, lineage, elimination-order independence
# --------------------------------------------------------------------------
class TestNestedQuotients(unittest.TestCase):
    L = TestResponseNetworkAndSheaf.L  # P4

    def test_nested_reduction_equals_one_shot(self):
        one_shot = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1], [2, 3]])
        E_one, _ = reduction_matrix(one_shot)

        # Stage 1: eliminate only {0} (cells 1,2,3 kept as singletons).
        stage1 = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1], [2], [3]])
        E1, _ = reduction_matrix(stage1)
        self.assertEqual(list(stage1.interfaceIndices), [1, 2, 3])
        # Stage 2 on the REDUCED operator: eliminate local cell 2 (= fine 3).
        stage2 = stage1.nextLevel([[0], [1, 2]])
        E2, read2 = reduction_matrix(stage2)
        np.testing.assert_allclose(E2, E_one, rtol=0, atol=MACHINE)
        self.assertEqual(stage2.level, 1)
        self.assertTrue(read2.certificate.holds())

    def test_elimination_order_independence(self):
        # {0} then {3} vs {3} then {0}: identical final response.
        first = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1], [2], [3]])
        a = first.nextLevel([[0], [1, 2]])
        E_a, _ = reduction_matrix(a)

        second = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0], [1], [2, 3]])
        b = second.nextLevel([[0, 1], [2]])
        E_b, _ = reduction_matrix(b)
        np.testing.assert_allclose(E_a, E_b, rtol=0, atol=MACHINE)

    def test_lineage_provenance_is_carried(self):
        stage1 = cob.RecursiveQuotient.overMatrix(
            _flat(self.L), 4, [], [[0, 1], [2], [3]])
        stage2 = stage1.nextLevel([[0], [1, 2]])
        self.assertEqual(list(stage2.coordinateProvenance),
                         ["L0:coord(1)", "L0:coord(2)", "L0:coord(3)"])
        _, read = reduction_matrix(stage2)
        self.assertEqual([c.provenance for c in read.coordinates],
                         ["L0:coord(1)", "L0:coord(2)"])


# --------------------------------------------------------------------------
# selected interior retention
# --------------------------------------------------------------------------
class TestSelectedRetention(unittest.TestCase):
    def test_selected_interior_cell_is_never_eliminated(self):
        L = [[2, -2, 0], [-2, 5, -3], [0, -3, 3]]
        options = cob.RecursiveQuotient.Options()
        options.selectedInteriorIndices = [1]
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [2]], options)
        # Nothing left to eliminate: the reduced operator IS the fine one.
        self.assertEqual(list(q.interfaceIndices), [0, 1, 2])
        E, read = reduction_matrix(q)
        np.testing.assert_allclose(E, np.array(L, dtype=complex),
                                   rtol=0, atol=1e-15)
        kinds = [c.kind for c in read.coordinates]
        self.assertEqual(kinds, [cob.RetainedCoordinateKind.Interface,
                                 cob.RetainedCoordinateKind.Selected,
                                 cob.RetainedCoordinateKind.Interface])

    def test_selected_cell_by_vertex_tuple_on_the_spacetime_path(self):
        st = build_graph([(0, 1, 2.0, 0.0), (1, 2, 3.0, 0.0)])
        options = cob.RecursiveQuotient.Options()
        options.selectedInteriorCells = [[1]]
        q = cob.RecursiveQuotient.overCells(
            st, 0, [[[0], [1], [2]], [[0]], [[2]]], options)
        self.assertEqual(list(q.interfaceIndices), [0, 1, 2])
        _, read = reduction_matrix(q)
        self.assertEqual(read.coordinates[1].kind,
                         cob.RetainedCoordinateKind.Selected)


# --------------------------------------------------------------------------
# Craig-Bampton / AMLS
# --------------------------------------------------------------------------
class TestCraigBampton(unittest.TestCase):
    # P6 path, interior {1,2,3,4} (tridiagonal [2,-1] of size 4):
    # fixed-interface eigenvalues 2 - 2 cos(k pi / 5), k = 1..4:
    #   0.38196601, 1.38196601, 2.61803399, 3.61803399.
    N = 6

    def _fixture(self):
        L = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            L[i, i] += 1.0
            L[i + 1, i + 1] += 1.0
            L[i, i + 1] -= 1.0
            L[i + 1, i] -= 1.0
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), self.N, [], [[0, 1, 2, 3, 4, 5], [0], [5]])
        return L, q

    def test_retained_modes_window_and_gap(self):
        _, q = self._fixture()
        read = q.craigBampton(-1e-6, 0.5, 1.0)
        self.assertEqual(read.windowLower, -1e-6)
        self.assertEqual(read.windowUpper, 0.5)
        self.assertEqual(list(read.retainedModes), [1])
        # Discarded-mode gap: 1.38196601 - 0.5 (literal).
        self.assertAlmostEqual(read.discardedModeGap,
                               2 - 2 * np.cos(2 * np.pi / 5) - 0.5, places=10)

    def test_reduced_pencil_structure(self):
        L, q = self._fixture()
        read = q.craigBampton(-1e-6, 0.5, 1.0)
        K = _mat(read.reducedStiffness, 3)
        M = _mat(read.reducedMass, 3)
        # K_BB block == the exact static Schur complement (the constraint
        # modes ARE the static condensation shapes).
        E, _ = reduction_matrix(q)
        np.testing.assert_allclose(K[:2, :2], E, rtol=0, atol=1e-12)
        # Constraint modes are L-orthogonal to fixed-interface modes.
        np.testing.assert_allclose(K[:2, 2:], 0, rtol=0, atol=1e-12)
        # K_mm = the retained fixed-interface eigenvalue.
        self.assertAlmostEqual(K[2, 2].real, 2 - 2 * np.cos(np.pi / 5),
                               places=10)
        # Mass is Hermitian positive definite.
        np.testing.assert_allclose(M, M.conj().T, rtol=0, atol=1e-13)
        self.assertGreater(np.min(np.linalg.eigvalsh(M.real)), 0.0)

    WINDOW = (-1e-6, 0.5)  # lower edge below the exact 0 mode (fp-safe)

    def test_window_eigenvalues_match_the_fine_operator(self):
        L, q = self._fixture()
        read = q.craigBampton(self.WINDOW[0], self.WINDOW[1], 1.0,
                              residual_tolerance=5e-2)
        fine = np.linalg.eigvalsh(L)
        fine_in_window = fine[(fine >= self.WINDOW[0]) & (fine <= self.WINDOW[1])]
        self.assertEqual(len(read.windowEigenvalues), len(fine_in_window))
        # CERTIFIED APPROXIMATION, not an exact identity: the error is
        # bounded by the reported residual (self-adjoint Bauer-Fike with the
        # residual's own norm scale), and Rayleigh-Ritz bounds each reduced
        # eigenvalue from above by construction.
        scale = np.linalg.norm(L)  # Frobenius, the residual's scale
        for value, residual in zip(read.windowEigenvalues,
                                   read.eigenResiduals):
            self.assertLess(np.min(np.abs(fine - value)),
                            1.05 * residual * scale + 1e-12)
        for value, exact in zip(read.windowEigenvalues, fine_in_window):
            self.assertGreaterEqual(value, exact - 1e-12)
        self.assertEqual(read.certificate.domain,
                         cob.CertificateDomain.BandWindow)
        # Holds against the CALLER-DECLARED surrogate tolerance...
        self.assertTrue(read.certificate.holds())
        # ...and honestly refuses under the strict default tolerance.
        strict = q.craigBampton(self.WINDOW[0], self.WINDOW[1], 1.0)
        self.assertFalse(strict.certificate.holds())
        self.assertEqual(strict.discardedModeGap, read.discardedModeGap)

    def test_larger_cutoff_tightens_the_surrogate(self):
        # More retained fixed-interface modes -> smaller residuals and a
        # closer window eigenvalue (monotone improvement of the surrogate).
        L, q = self._fixture()
        fine = np.linalg.eigvalsh(L)
        coarse = q.craigBampton(self.WINDOW[0], self.WINDOW[1], 1.0)
        finer = q.craigBampton(self.WINDOW[0], self.WINDOW[1], 3.0)
        self.assertEqual(list(finer.retainedModes), [3])
        self.assertEqual(len(coarse.windowEigenvalues), 2)
        self.assertEqual(len(finer.windowEigenvalues), 2)
        err_coarse = abs(coarse.windowEigenvalues[1] - fine[1])
        err_finer = abs(finer.windowEigenvalues[1] - fine[1])
        self.assertLess(err_finer, err_coarse)
        self.assertLess(max(finer.eigenResiduals), max(coarse.eigenResiduals))

    def test_non_normal_regime_is_refused(self):
        L = np.array([[1.0, 0.5, 0.0],
                      [-0.2, 2.0, -1.0],
                      [0.0, -1.0, 1.0]])
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [2]])
        with self.assertRaises(ValueError):
            q.craigBampton(0.0, 0.5, 1.0)

    def test_indefinite_chain_metric_is_refused(self):
        H = np.array([[2.0, -1.0, 0.0],
                      [-1.0, 3.0, -1.0],
                      [0.0, -1.0, 2.0]])
        L = np.diag([1.0, -1.0, 1.0]) @ H
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [1.0 + 0j, -1.0 + 0j, 1.0 + 0j],
            [[0, 1, 2], [0], [2]])
        with self.assertRaises(ValueError):
            q.craigBampton(0.0, 0.5, 1.0)

    def test_bad_window_and_cutoff_are_refused(self):
        _, q = self._fixture()
        with self.assertRaises(ValueError):
            q.craigBampton(1.0, 0.5, 2.0)
        with self.assertRaises(ValueError):
            q.craigBampton(0.0, 1.5, 1.0)  # cutoff below the window edge


# --------------------------------------------------------------------------
# relabeling isomorphism
# --------------------------------------------------------------------------
class TestRelabeling(unittest.TestCase):
    def test_matrix_relabeling_gives_the_permuted_reduction(self):
        rng = np.random.default_rng(11)
        L = np.array(TestHandSolvedStar.L, dtype=float)
        n = 4
        perm = rng.permutation(n)
        P = np.eye(n)[perm]
        L_perm = P @ L @ P.T
        base = cob.RecursiveQuotient.overMatrix(
            _flat(L), n, [], [[0, 1, 2, 3], [1], [2], [3]])
        mapped_components = [[int(np.where(perm == i)[0][0]) for i in comp]
                             for comp in [[0, 1, 2, 3], [1], [2], [3]]]
        relabeled = cob.RecursiveQuotient.overMatrix(
            _flat(L_perm), n, [], mapped_components)
        E_base, _ = reduction_matrix(base)
        E_rel, _ = reduction_matrix(relabeled)
        kept_base = list(base.interfaceIndices)
        kept_rel = list(relabeled.interfaceIndices)
        # Match kept coordinates through the permutation, order-agnostically.
        positions = {int(np.where(perm == i)[0][0]): k
                     for k, i in enumerate(kept_base)}
        reorder = [positions[i] for i in kept_rel]
        np.testing.assert_allclose(E_rel, E_base[np.ix_(reorder, reorder)],
                                   rtol=0, atol=MACHINE)

    def test_spacetime_relabeling_gives_an_isomorphic_network(self):
        # The same weighted star with permuted vertex identifiers: the
        # response network is the isomorphic image (same stalk dimensions,
        # same effective couplings under the id map).
        edges = [(0, 1, 1.0, 0.0), (0, 2, 2.0, 0.0), (0, 3, 3.0, 0.0)]
        st_a = build_graph(edges)
        id_map = {0: 30, 1: 11, 2: 22, 3: 3}
        st_b = build_graph(
            [(id_map[s], id_map[t], w, p) for s, t, w, p in edges])
        cells_a = [[[0], [1], [2], [3]], [[1]], [[2]], [[3]]]
        cells_b = [[[id_map[v[0]]] for v in comp] for comp in cells_a]
        q_a = cob.RecursiveQuotient.overCells(st_a, 0, cells_a)
        q_b = cob.RecursiveQuotient.overCells(st_b, 0, cells_b)
        E_a, read_a = reduction_matrix(q_a)
        E_b, read_b = reduction_matrix(q_b)
        # Map kept cells through provenance (cell ids), then compare.
        prov_a = [c.provenance for c in read_a.coordinates]
        prov_b = [c.provenance for c in read_b.coordinates]
        mapped = [f"cell({id_map[int(p[5:-1])]})" for p in prov_a]
        reorder = [prov_b.index(p) for p in mapped]
        np.testing.assert_allclose(E_b[np.ix_(reorder, reorder)], E_a,
                                   rtol=0, atol=MACHINE)
        net_a = q_a.responseNetwork()
        net_b = q_b.responseNetwork()
        self.assertEqual(sorted(net_a.stalkDimensions),
                         sorted(net_b.stalkDimensions))
        self.assertEqual(len(net_a.edges), len(net_b.edges))

    def test_within_cell_vertex_order_is_never_a_convention(self):
        # Cells are matched by vertex SET: reversing every tuple (and
        # shuffling the support lists) yields the identical reduction.
        st = tessera.Spacetime.fromCells(2, [[0, 1, 2], [1, 2, 3]], 1.0, 0.0)
        st.materializeFacets()
        forward = cob.RecursiveQuotient.overCells(
            st, 1, [[[0, 1], [0, 2], [1, 2]], [[1, 3], [2, 3]]])
        reversed_cells = cob.RecursiveQuotient.overCells(
            st, 1, [[[2, 1], [2, 0], [1, 0]], [[3, 2], [3, 1]]])
        E_f, _ = reduction_matrix(forward)
        E_r, _ = reduction_matrix(reversed_cells)
        np.testing.assert_allclose(E_f, E_r, rtol=0, atol=0)
        self.assertEqual(list(forward.interfaceIndices),
                         list(reversed_cells.interfaceIndices))

    def test_relabeled_sheaf_fixture_reproduces_its_realization(self):
        L = TestResponseNetworkAndSheaf.L
        perm = [2, 0, 3, 1]
        P = np.eye(4)[perm]
        L_perm = P @ L @ P.T
        mapped = [[int(np.where(np.array(perm) == i)[0][0]) for i in comp]
                  for comp in [[0, 1], [2, 3]]]
        q = cob.RecursiveQuotient.overMatrix(_flat(L_perm), 4, [], mapped)
        sheaf = q.sheafRealization()
        self.assertTrue(sheaf.emitted)
        self.assertTrue(sheaf.simplicial)
        self.assertEqual(list(sheaf.edgeStalkDimensions), [1])


class TestComponentOrderIndependence(unittest.TestCase):
    def test_component_input_order_never_changes_the_kept_block(self):
        # The canonical reduced order derives from fine indices and
        # component position only for MODE coordinates; permuting the
        # component list leaves the kept block bit-identical and permutes
        # the mode blocks with the components.
        L = np.array([[1.0, -1.0, 0.0, 0.0],
                      [-1.0, 2.0, -1.0, 0.0],
                      [0.0, -1.0, 2.0, -1.0],
                      [0.0, 0.0, -1.0, 1.0]])
        a = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1], [2, 3]])
        b = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[2, 3], [0, 1]])
        E_a, read_a = reduction_matrix(a)
        E_b, read_b = reduction_matrix(b)
        np.testing.assert_allclose(E_a, E_b, rtol=0, atol=0)
        self.assertEqual(list(a.interfaceIndices), list(b.interfaceIndices))
        # Ownership swaps with the component positions, honestly.
        self.assertEqual([c.component for c in read_a.coordinates],
                         [1 - c.component for c in read_b.coordinates])


# --------------------------------------------------------------------------
# cache: recompute only the affected ancestry; cached == cold
# --------------------------------------------------------------------------
class TestCacheIncremental(unittest.TestCase):
    A_CELLS = [[[0], [1], [2]]]
    B_CELLS = [[[10], [11], [12]]]

    def _spacetime(self):
        return build_graph([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0),
                               (0, 2, 1.0, 0.0), (10, 11, 1.0, 0.0),
                               (11, 12, 1.0, 0.0), (10, 12, 1.0, 0.0)])

    def test_local_change_recomputes_only_the_touched_component(self):
        st = self._spacetime()
        cache = cob.AnalyticCache(st)
        cells = self.A_CELLS[0] + self.B_CELLS[0]
        q = cob.RecursiveQuotient.overCells(
            st, 0, [self.A_CELLS[0], self.B_CELLS[0]],
            cob.RecursiveQuotient.Options(), cache)
        q.staticReduction()
        self.assertEqual(cache.size, 2)  # both components stored
        misses_before = cache.misses

        # Accepted metric move INSIDE component A.
        for e in st.getEdgeList().toVector():
            if e.getSource().getId() in (0, 1) and \
               e.getTarget().getId() in (0, 1):
                e.setLength(cmath.sqrt(complex(1.7)))
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)
        cache.publish(star)
        self.assertEqual(cache.size, 1)  # A dropped, sibling B SURVIVES

        q.invalidate()
        hits_before = cache.hits
        E_incremental, _ = reduction_matrix(q)
        # Exactly one hit (B served) and one recompute (A stored again).
        self.assertEqual(cache.hits - hits_before, 1)
        self.assertEqual(cache.size, 2)

        # Cached == cold: a fresh cacheless quotient on the mutated complex.
        cold = cob.RecursiveQuotient.overCells(
            st, 0, [self.A_CELLS[0], self.B_CELLS[0]])
        E_cold, _ = reduction_matrix(cold)
        np.testing.assert_allclose(E_incremental, E_cold, rtol=0, atol=0)
        self.assertGreaterEqual(cache.misses, misses_before)

    def test_ancestry_recompute_flows_to_the_next_level(self):
        # Weighted P4 chain 0-1-2-3; components {0,1} and {2,3} share the
        # middle coupling: the child level reduces the NONZERO effective
        # operator over kept cells {1, 2}, so ancestry equality is a
        # nontrivial matrix statement.
        def chain(w01):
            return build_graph([(0, 1, w01, 0.0), (1, 2, 1.0, 0.0),
                                (2, 3, 1.0, 0.0)])

        st = chain(1.0)
        cache = cob.AnalyticCache(st)
        cells = [[[0], [1]], [[2], [3]]]
        q = cob.RecursiveQuotient.overCells(
            st, 0, cells, cob.RecursiveQuotient.Options(), cache)
        child = q.nextLevel([[0, 1]])
        E_child_before, _ = reduction_matrix(child)
        self.assertEqual(E_child_before.shape, (1, 1))  # one harmonic fiber

        # Accepted metric move on edge (0, 1), inside component A only.
        for e in st.getEdgeList().toVector():
            if {e.getSource().getId(), e.getTarget().getId()} == {0, 1}:
                e.setLength(cmath.sqrt(complex(2.5)))
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)
        cache.publish(star)
        self.assertEqual(cache.size, 1)  # sibling B's entry survives

        q.invalidate()
        hits_before = cache.hits
        E_parent_after, _ = reduction_matrix(q)
        self.assertEqual(cache.hits - hits_before, 1)  # B served from cache
        child_after = q.nextLevel([[0, 1]])
        E_child_after, _ = reduction_matrix(child_after)

        cold = cob.RecursiveQuotient.overCells(st, 0, cells)
        E_parent_cold, _ = reduction_matrix(cold)
        cold_child = cold.nextLevel([[0, 1]])
        E_cold, _ = reduction_matrix(cold_child)
        # Cached == cold at BOTH levels, bit-for-bit.
        np.testing.assert_allclose(E_parent_after, E_parent_cold,
                                   rtol=0, atol=0)
        self.assertEqual(E_child_after.shape, E_cold.shape)
        np.testing.assert_allclose(E_child_after, E_cold, rtol=0, atol=0)


class TestCachePartitionSafety(unittest.TestCase):
    def test_same_vertex_set_different_partition_never_cross_serves(self):
        # Two quotients over the SAME spacetime and cache whose components
        # share a vertex-id union but classify its cells differently (one
        # retains a selected cell): the partition fingerprint keeps their
        # payloads apart, so each equals its cacheless clone.
        st = build_graph([(0, 1, 2.0, 0.0), (1, 2, 3.0, 0.0)])
        cache = cob.AnalyticCache(st)
        cells = [[[0], [1], [2]], [[0]], [[2]]]

        plain = cob.RecursiveQuotient.overCells(
            st, 0, cells, cob.RecursiveQuotient.Options(), cache)
        E_plain, _ = reduction_matrix(plain)

        options = cob.RecursiveQuotient.Options()
        options.selectedInteriorCells = [[1]]
        selected = cob.RecursiveQuotient.overCells(st, 0, cells, options,
                                                   cache)
        E_selected, _ = reduction_matrix(selected)

        cold = cob.RecursiveQuotient.overCells(st, 0, cells, options)
        E_cold, _ = reduction_matrix(cold)
        np.testing.assert_allclose(E_selected, E_cold, rtol=0, atol=0)
        # And the plain quotient still matches ITS cacheless clone.
        cold_plain = cob.RecursiveQuotient.overCells(st, 0, cells)
        E_cold_plain, _ = reduction_matrix(cold_plain)
        np.testing.assert_allclose(E_plain, E_cold_plain, rtol=0, atol=0)
        self.assertEqual(E_plain.shape, (2, 2))
        self.assertEqual(E_selected.shape, (3, 3))


# --------------------------------------------------------------------------
# dense/sparse crossover agreement
# --------------------------------------------------------------------------
class TestDenseSparseCrossover(unittest.TestCase):
    def test_dense_and_sparse_interior_paths_agree(self):
        # Same fixture through the dense (below-crossover) and sparse
        # (at/above-crossover) interior factor solves.
        rng = np.random.default_rng(3)
        n = 24
        L = np.zeros((n, n))
        for i in range(n - 1):
            w = 0.5 + rng.random()
            L[i, i] += w
            L[i + 1, i + 1] += w
            L[i, i + 1] -= w
            L[i + 1, i] -= w
        components = [list(range(n)), [0], [n - 1]]

        dense_options = cob.RecursiveQuotient.Options()
        dense_options.denseCrossover = 512
        sparse_options = cob.RecursiveQuotient.Options()
        sparse_options.denseCrossover = 1  # force the SparseLU path

        E_dense, read_dense = reduction_matrix(
            cob.RecursiveQuotient.overMatrix(_flat(L), n, [], components,
                                             dense_options))
        E_sparse, read_sparse = reduction_matrix(
            cob.RecursiveQuotient.overMatrix(_flat(L), n, [], components,
                                             sparse_options))
        np.testing.assert_allclose(E_dense, E_sparse, rtol=0, atol=1e-11)
        np.testing.assert_allclose(E_dense, numpy_schur(L, [0, n - 1],
                                                        list(range(1, n - 1))),
                                   rtol=0, atol=1e-10)
        self.assertTrue(read_dense.certificate.holds())
        self.assertTrue(read_sparse.certificate.holds())

    def test_feshbach_agrees_across_the_crossover(self):
        L = TestBlockPencilNegativeControl.L
        sparse_options = cob.RecursiveQuotient.Options()
        sparse_options.denseCrossover = 1
        q_dense = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1, 2, 3], [0], [1]])
        q_sparse = cob.RecursiveQuotient.overMatrix(
            _flat(L), 4, [], [[0, 1, 2, 3], [0], [1]], sparse_options)
        for lam in (0.5, 1.5 + 0.2j):
            F_dense = _mat(q_dense.feshbach(complex(lam), 0.0, 2.0).response, 2)
            F_sparse = _mat(q_sparse.feshbach(complex(lam), 0.0, 2.0).response, 2)
            np.testing.assert_allclose(F_dense, F_sparse, rtol=0, atol=1e-11)


# --------------------------------------------------------------------------
# discovered partitions (PersistentModularity) and vertex supports
# --------------------------------------------------------------------------
class TestDiscoveredPartitions(unittest.TestCase):
    def test_persistent_modularity_supports_drive_the_reduction(self):
        # Two 4-cliques joined by one bridge: modularity discovers the two
        # cliques; the bridge endpoints become the interface.
        edges = []
        for base in (0, 10):
            group = [base + i for i in range(4)]
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append((group[i], group[j], 1.0, 0.0))
        edges.append((3, 10, 1.0, 0.0))  # the bridge
        st = build_graph(edges)

        PM = tessera.PersistentModularity
        graph = PM.fromSpacetime(st, PM.WeightMap.Unit)
        cfg = tessera.PersistentModularityConfig()
        cfg.resolutions = [1.0]
        slice_read = graph.discover(1.0, cfg)
        supports = sorted(tuple(sorted(c.support)) for c in slice_read.components)
        self.assertEqual(supports, [(0, 1, 2, 3), (10, 11, 12, 13)])

        q = cob.RecursiveQuotient.overVertexSupports(
            st, 0, [list(s) for s in supports])
        # Interface = the two bridge endpoints; interiors = the clique bulks.
        self.assertEqual(len(q.interfaceIndices), 2)
        self.assertEqual(len(q.interiorIndices(0)), 3)
        self.assertEqual(len(q.interiorIndices(1)), 3)
        cert = q.verifyStatic()
        self.assertTrue(cert.holds(), cert.describe())
        # Independent NumPy reference on the whole-graph Laplacian.
        hodge = cob.HodgeLaplacian(st)
        ids = sorted(v.getId() for v in st.getVertexList().toVector())
        L = _mat(hodge.laplacian(0), len(ids))
        kept = list(q.interfaceIndices)
        interior = sorted(set(range(len(ids))) - set(kept))
        E, _ = reduction_matrix(q)
        np.testing.assert_allclose(E[:2, :2], numpy_schur(L, kept, interior),
                                   rtol=0, atol=1e-11)

    def test_unclaimed_cells_form_a_residual_component(self):
        st = build_graph([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0)])
        q = cob.RecursiveQuotient.overVertexSupports(st, 0, [[0, 1]])
        self.assertEqual(q.componentCount, 2)  # the given one + residual
        cert = q.verifyStatic()
        self.assertTrue(cert.holds())


# --------------------------------------------------------------------------
# real Lorentzian geometry (stored causal specimen)
# --------------------------------------------------------------------------
class TestCausalSpecimen(unittest.TestCase):
    """The #562 campaign causal specimen (seed 14001000, re_min < 0): a real
    4D complex with one timelike k=1 cell, so the signed SquaredContent
    weights make the d'Alembertian W-self-adjoint against an INDEFINITE
    metric -- the stationarity regime on genuine geometry."""

    @classmethod
    def setUpClass(cls):
        cls.st = rebuild_spacetime(load_dump(14001000))
        ids = sorted(v.getId() for v in cls.st.getVertexList().toVector())
        half = len(ids) // 2
        cls.supports = [ids[:half + 1], ids[half:]]  # overlapping halves

    def _quotient(self):
        return cob.RecursiveQuotient.overVertexSupports(
            self.st, 1, self.supports)

    def test_regime_is_hermitian_indefinite(self):
        q = self._quotient()
        self.assertEqual(q.regime, cob.CertificateRegime.HermitianIndefinite)

    def test_stationarity_certificate_holds_on_geometry(self):
        q = self._quotient()
        read = q.staticReduction()
        self.assertTrue(read.certificate.holds(), read.certificate.describe())
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        # One deterministic probe through the stationarity check.
        kept = len(q.interfaceIndices)
        probe = [0j] * kept
        probe[0] = 1.0 + 0j
        cert = q.staticProbeCertificate(probe)
        self.assertTrue(cert.holds(), cert.describe())

    def test_feshbach_determinant_identity_on_geometry(self):
        q = self._quotient()
        read = q.feshbach(0.37 + 0.11j, 0.0, 1.0)
        self.assertFalse(read.resonant)
        self.assertLess(read.determinantResidual, 1e-9)
        self.assertTrue(read.certificate.holds(), read.certificate.describe())

    def test_complex_length_flips_the_regime_to_non_normal(self):
        # A complex squared length (a relaxed-state artifact) makes the
        # signed weights complex: WL loses Hermiticity and the certificate
        # regime honestly degrades to certified block elimination.
        st = rebuild_spacetime(load_dump(14001000))
        edge = st.getEdgeList().toVector()[0]
        edge.setLength(cmath.sqrt(complex(1.3, 0.4)))
        q = cob.RecursiveQuotient.overVertexSupports(st, 1, self.supports)
        self.assertEqual(q.regime, cob.CertificateRegime.NonNormal)
        read = q.staticReduction()
        self.assertTrue(read.certificate.holds(), read.certificate.describe())


# --------------------------------------------------------------------------
# validation and refusals
# --------------------------------------------------------------------------
class TestValidation(unittest.TestCase):
    def test_uncovered_cells_are_refused(self):
        with self.assertRaises(ValueError):
            cob.RecursiveQuotient.overMatrix(
                _flat(np.eye(3)), 3, [], [[0, 1]])

    def test_out_of_range_component_index_is_refused(self):
        with self.assertRaises(ValueError):
            cob.RecursiveQuotient.overMatrix(
                _flat(np.eye(2)), 2, [], [[0, 5]])

    def test_unknown_cell_is_refused(self):
        st = build_graph([(0, 1, 1.0, 0.0)])
        with self.assertRaises(ValueError):
            cob.RecursiveQuotient.overCells(st, 0, [[[7]]])

    def test_weight_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            cob.RecursiveQuotient.overMatrix(
                _flat(np.eye(3)), 3, [1.0 + 0j], [[0, 1, 2]])

    def test_bad_feshbach_window_is_refused(self):
        q = cob.RecursiveQuotient.overMatrix(
            _flat(np.eye(2)), 2, [], [[0, 1]])
        with self.assertRaises(ValueError):
            q.feshbach(0.5 + 0j, 2.0, 1.0)

    def test_probe_size_mismatch_is_refused(self):
        L = [[2, -2, 0], [-2, 5, -3], [0, -3, 3]]
        q = cob.RecursiveQuotient.overMatrix(
            _flat(L), 3, [], [[0, 1, 2], [0], [2]])
        with self.assertRaises(ValueError):
            q.staticProbeCertificate([1.0 + 0j])


if __name__ == "__main__":
    unittest.main()
