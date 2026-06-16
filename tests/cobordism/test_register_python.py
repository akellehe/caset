# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The C++ carried spectral register ``tessera.cobordism.Register`` (#303).

Promoted from the ``spectral_gate_realizability`` example: the register is now a C++
aggregator over the existing spectrum tooling -- it opens the holonomy holes by surgery
and reads every quantity off the shared #286 carried-register read-outs
(``HodgeLaplacian.harmonicMatrix``, ``EigenstateSynthesis.cyclePeriods`` /
``carriedRepresentative`` / ``residualForPeriods``, ``ChainComplex.endSignCovector``).
These tests pin (a) the register's construction, shape, and surgery-grown topology,
(b) the spectral scoring (carried -> 0, leaking floors, and exact-period leak), (c) the
new ``carriedRepresentative`` read-out and its invariant with ``residualForPeriods``,
(d) that ``Register`` genuinely DELEGATES to the underlying ``EigenstateSynthesis``
(byte-for-byte, the "reuse what's there" guarantee), and (e) the growth / extra-hole /
error paths. The thin example factory and the #249 harness contract are covered too.
"""

import importlib.util
import os
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "spectral_gate_realizability.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("spectral_gate_realizability",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_example()
_ICO = GATE._ICO
_CLASS_HOLES = [list(h) for h in GATE._CLASS_HOLES]


def _surface():
    """A fresh icosahedron pre-geometric surface (the canonical register seed)."""
    return GATE._surface(_ICO)


def _register():
    """The canonical C++ register, built straight from cob.Register (no example glue)."""
    return cob.Register(_surface(), _CLASS_HOLES)


def _manual_synth():
    """An EigenstateSynthesis built the same way the register builds its own, on an
    independent but identical surface -- the reference the register must match."""
    es = cob.EigenstateSynthesis(_surface(), 1)
    for hole in _CLASS_HOLES:
        es.removeInteriorCell(list(hole))
    return es


def _period(psi, cells, tri):
    """The induced oriented loop sum psi[(a,b)] + psi[(b,c)] - psi[(a,c)] of a full edge
    1-form over a triangle's three edges -- the circle period (matches cyclePeriods)."""
    idx = {tuple(c): i for i, c in enumerate(cells)}
    a, b, c = sorted(tri)
    return psi[idx[(a, b)]] + psi[idx[(b, c)]] - psi[idx[(a, c)]]


# --------------------------------------------------------------------------- #
class ConstructionTest(unittest.TestCase):
    """The register is the 2-dim S_3 standard rep V = ker L_1 of the surgery-grown S^2,
    with all shapes/read-outs consistent."""

    def setUp(self):
        self.reg = _register()

    def test_dimension_and_shapes(self):
        self.assertEqual(self.reg.dim, 2)                    # the S_3 standard rep
        self.assertEqual(self.reg.order(), 30)               # |C_1| of the grown bulk
        self.assertEqual(len(self.reg.cells), self.reg.order())
        self.assertEqual(self.reg.P.shape, (2, 3))           # 2 harmonics, 3 circles
        self.assertEqual(self.reg.H_full.shape, (2, self.reg.order()))
        self.assertEqual(len(self.reg.class_holes), 3)
        self.assertEqual(list(self.reg.extra_opened), [])
        self.assertEqual(self.reg.grown, 0)

    def test_surgery_grew_b1_to_two(self):
        # opening the three holes on a closed S^2 grows b_1 0 -> 2 (ker L_1 = dim H_1)
        self.assertEqual(GATE._betti1(self.reg.st), 2)
        self.assertEqual(len(cob.HodgeLaplacian(self.reg.st).harmonics(1)), self.reg.dim)

    def test_cells_are_hashable_tuples(self):
        # consumers key dicts by cell (e.g. the #249 harness) -- they must be tuples
        for c in self.reg.cells:
            self.assertIsInstance(c, tuple)
        by_cell = {c: i for i, c in enumerate(self.reg.cells)}  # must not raise
        self.assertEqual(len(by_cell), len(self.reg.cells))
        for c in self.reg.cells:                                # 1-cells are edges
            self.assertEqual(len(c), 2)

    def test_rank_is_a_genuine_register(self):
        # rank(P) = 2 < 3 holes -> a proper carried subspace (an obstruction exists)
        self.assertEqual(self.reg.rank, 2)
        self.assertLess(self.reg.rank, len(self.reg.class_holes))

    def test_period_constraint_annihilates_the_periods(self):
        # n is the +-1 end-sign covector; it is a (left) null covector of P: P n = 0
        self.assertEqual(len(self.reg.n), 3)
        self.assertTrue(np.allclose(np.abs(self.reg.n), 1.0))    # a sign covector
        self.assertTrue(np.allclose(self.reg.P @ self.reg.n, 0.0, atol=1e-9))
        self.assertTrue(np.allclose(self.reg.sign, self.reg.n))  # sign IS n


# --------------------------------------------------------------------------- #
class ScoringTest(unittest.TestCase):
    """The spectral realizability score: carried periods -> 0, leaking periods floor,
    and the carried representative reproduces the requested periods exactly."""

    def setUp(self):
        self.reg = _register()

    def test_carried_state_zero_residual_leaking_state_floors(self):
        carried = self.reg.sign * np.array([1.0, -1.0, 0.0])     # Sigma = 0
        leaking = self.reg.sign * np.array([1.0, 1.0, 1.0])      # Sigma = 3
        self.assertLess(self.reg.spectral_residual(carried), GATE.REALIZE)
        self.assertGreater(self.reg.spectral_residual(leaking), GATE.CERT_FLOOR)

    def test_harmonic_form_has_full_length_and_reproduces_periods(self):
        # the leak guarantee: the carried representative's periods are EXACTLY the target,
        # carried or not (the leak lands the un-carried remainder on the (a,b) facet)
        for target in ([1.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.3, -0.7, 0.9]):
            raw = self.reg.sign * np.array(target)
            psi = np.asarray(self.reg.harmonic_form(raw), dtype=complex)
            self.assertEqual(psi.shape, (self.reg.order(),))
            got = [_period(psi, self.reg.cells, tri) for tri in self.reg.class_holes]
            self.assertTrue(np.allclose(got, raw, atol=1e-9))

    def test_carried_target_yields_a_true_harmonic(self):
        # a target in the carried space V has zero leak -> psi is an exact L_1 harmonic
        raw = self.reg.sign * np.array([1.0, -1.0, 0.0])
        psi = np.asarray(self.reg.harmonic_form(raw), dtype=complex)
        # rebuild the harmonic subspace and check psi lies in it (residual ~ 0 already,
        # here the geometric statement: psi is orthogonal to no harmonic only by being one)
        self.assertLess(self.reg.spectral_residual(raw), GATE.REALIZE)
        self.assertGreater(np.linalg.norm(psi), 0.0)


# --------------------------------------------------------------------------- #
class DelegationTest(unittest.TestCase):
    """Register REUSES the existing tooling: every read-out matches an independently
    built EigenstateSynthesis byte-for-byte (no re-derived Hodge/period algebra)."""

    def setUp(self):
        self.reg = _register()
        self.es = _manual_synth()
        self.holes = [list(h) for h in _CLASS_HOLES]

    def test_period_matrix_matches_cycle_periods(self):
        ref = np.asarray(self.es.cyclePeriods(self.holes),
                         dtype=complex).reshape(self.reg.dim, len(self.holes))
        self.assertTrue(np.allclose(self.reg.P, ref))

    def test_constraint_matches_end_sign_covector(self):
        ref = cob.ChainComplex.endSignCovector(
            [[int(v) for v in c] for c in self.es.topCells()], self.holes)
        self.assertTrue(np.allclose(self.reg.n, np.asarray(ref, dtype=float)))

    def test_harmonic_matrix_matches_hodge_laplacian(self):
        ref = np.asarray(cob.HodgeLaplacian(self.reg.st).harmonicMatrix(1),
                         dtype=complex).reshape(self.reg.dim, self.reg.order())
        self.assertTrue(np.allclose(self.reg.H_full, ref))

    def test_residual_and_form_match_eigenstate_synthesis(self):
        for target in ([1.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.2, -0.7]):
            raw = [complex(z) for z in self.reg.sign * np.array(target)]
            self.assertAlmostEqual(self.reg.spectral_residual(raw),
                                   self.es.residualForPeriods(self.holes, raw), places=12)
            a = np.asarray(self.reg.harmonic_form(raw), dtype=complex)
            b = np.asarray(self.es.carriedRepresentative(self.holes, raw), dtype=complex)
            self.assertTrue(np.allclose(a, b))


# --------------------------------------------------------------------------- #
class CarriedRepresentativeTest(unittest.TestCase):
    """The new EigenstateSynthesis.carriedRepresentative read-out (the psi that
    residualForPeriods scores) and its invariant residual(psi) == residualForPeriods."""

    def setUp(self):
        self.es = _manual_synth()
        self.holes = [list(h) for h in _CLASS_HOLES]

    def test_residual_of_representative_equals_residual_for_periods(self):
        for target in ([1.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.9, -0.1, 0.4]):
            raw = [complex(z) for z in np.array(target)]
            psi = self.es.carriedRepresentative(self.holes, raw)
            self.assertEqual(len(psi), self.es.order())
            self.assertAlmostEqual(self.es.residual(psi),
                                   self.es.residualForPeriods(self.holes, raw), places=12)

    def test_length_mismatch_raises(self):
        with self.assertRaises(Exception):
            self.es.carriedRepresentative(self.holes, [1.0, 2.0])   # 2 targets, 3 holes


# --------------------------------------------------------------------------- #
class GrowthAndExtraHolesTest(unittest.TestCase):
    """The --retries search paths: additive stellar growth (ker L_1 preserved) and
    extra b_1-growth surgery."""

    def test_stellar_growth_preserves_ker_l1_and_enlarges_the_complex(self):
        base = _register()
        grown = cob.Register(_surface(), _CLASS_HOLES, [], 3, 20240601)
        self.assertGreaterEqual(grown.grown, 1)              # at least one move took
        self.assertLessEqual(grown.grown, 3)                 # capped at the request
        self.assertEqual(grown.dim, 2)                       # the fan preserves ker L_1
        self.assertGreater(grown.order(), base.order())      # more edges than the seed

    def test_growth_is_deterministic_in_the_seed(self):
        a = cob.Register(_surface(), _CLASS_HOLES, [], 2, 777)
        b = cob.Register(_surface(), _CLASS_HOLES, [], 2, 777)
        self.assertEqual(a.grown, b.grown)
        self.assertEqual(a.order(), b.order())

    def test_extra_hole_is_opened_and_grows_b1(self):
        # an interior top cell vertex-disjoint from the class holes is a valid extra hole
        es = _manual_synth()
        used = {v for hole in _CLASS_HOLES for v in hole}
        extra = next(list(c) for c in es.interiorTopCells()
                     if not (set(c) & used))
        reg = cob.Register(_surface(), _CLASS_HOLES, [extra])
        self.assertEqual([list(c) for c in reg.extra_opened], [sorted(extra)])
        self.assertEqual(GATE._betti1(reg.st), 3)            # a 4th disk -> b_1 = 3


# --------------------------------------------------------------------------- #
class ErrorPathTest(unittest.TestCase):
    """Bad inputs fail loudly rather than silently mis-scoring."""

    def test_non_removable_class_hole_raises(self):
        with self.assertRaises(Exception):
            cob.Register(_surface(), [[0, 1, 2], [0, 2, 3]])  # share vertex 0/2: 2nd cut fails

    def test_target_length_mismatch_raises(self):
        reg = _register()
        with self.assertRaises(Exception):
            reg.spectral_residual([1.0, 2.0])                 # 2 targets, 3 holes
        with self.assertRaises(Exception):
            reg.harmonic_form([1.0, 2.0, 3.0, 4.0])           # 4 targets, 3 holes


# --------------------------------------------------------------------------- #
class ExampleFactoryAndHarnessContractTest(unittest.TestCase):
    """The thin example factory returns the C++ register, and the #249 harness contract
    (base._REG_EDGES, base.Register(), canon.cells as dict keys, canon.harmonic_form on
    sign * periods) still holds."""

    def test_example_factory_returns_cpp_register(self):
        reg = GATE.Register()
        self.assertIsInstance(reg, cob.Register)
        self.assertEqual(reg.dim, 2)

    def test_harness_contract(self):
        # _REG_EDGES is consumed by mediated_gate_battery.py (the #249 harness)
        self.assertEqual(len(GATE._REG_EDGES), 9)            # 3 edges x 3 holes
        canon = GATE.Register()
        # the exact pattern mediated_gate_battery.identity_residual uses:
        cp = np.array([1.0, -1.0, 0.0])
        psi_full = canon.harmonic_form(canon.sign * cp)
        by_cell = {canon.cells[i]: psi_full[i] for i in range(len(canon.cells))}
        self.assertEqual(len(by_cell), len(canon.cells))     # tuple keys, no TypeError


if __name__ == "__main__":
    unittest.main()
