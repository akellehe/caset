# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pair-loop dual-basis flavor read (#561) — duality bookkeeping, gates, determinism.

Covers `examples/cobordism/pair_loop_flavor.py` on the controlled
`tests/fixtures/composite_spin` fixtures:

  * the Poincaré-duality bookkeeping — pair-loop index ↔ complementary hole,
    `w_i + w_j = -w_k` under the pinned singlet, and the label-free
    closed-form relation `Σ_h σ_h p_h = 0` behind the induced-orientation
    signs (`ChainComplex.endSignCovector`);
  * the DK-style charge — positive, and structurally additive over a pair
    loop's disjoint cycle support;
  * the GAUGE gate (global U(1) target phase: charges invariant, loop periods
    covariant) and the RELABEL gate (random vertex-id permutation, rebuild,
    re-read: everything must match, loop periods up to the one global
    orientation sign) on a fixture;
  * determinism — two independent rebuild+read passes agree exactly.

Everything here runs on the smallest fixture (74 top cells; a read is ~0.3 s),
so no `slow` marks are needed; the full fixture battery lives in the example's
`__main__`.
"""
import cmath
import importlib.util
import os
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class PairLoopFlavorBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plf = _load("pair_loop_flavor")
        cls.meta, cls.cells, cls.edges = cls.plf.load_fixture("synthetic_b3_3.json")
        cls.st = cls.plf.build_spacetime(cls.cells, cls.edges)
        cls.holes = cls.plf.register_holes(cls.st)
        cls.read = cls.plf.joint_read(cls.st, cls.holes)


class TestDualityBookkeeping(PairLoopFlavorBase):
    def test_pair_loop_index_maps_to_complementary_hole(self):
        # [γ_ij] = -[k]: the loop over (i, j) is dual to the third hole.
        plf = self.plf
        self.assertEqual(plf.PAIR_LOOPS, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual([plf.complement_hole(p) for p in plf.PAIR_LOOPS],
                         [2, 1, 0])
        for pair in plf.PAIR_LOOPS:
            k = plf.complement_hole(pair)
            self.assertEqual(set(pair) | {k}, {0, 1, 2})
            self.assertNotIn(k, pair)

    def test_oriented_weights_pin_the_singlet(self):
        # The joint carried representative's oriented per-hole periods are the
        # pinned singlet exactly (the leak construction), and the pin carries.
        for w, t in zip(self.read["w"], self.plf.SINGLET):
            self.assertLess(abs(w - t), 1e-12)
        self.assertLess(self.read["r_u"], 1e-20)

    def test_pair_loop_period_is_minus_the_complementary_weight(self):
        # w_i + w_j = -w_k, loop by loop, and the recorded duality residual.
        read, plf = self.read, self.plf
        for loop_idx, pair in enumerate(plf.PAIR_LOOPS):
            k = plf.complement_hole(pair)
            self.assertLess(abs(read["loop_w"][loop_idx] + read["w"][k]), 1e-12)
            self.assertLess(read["dual_residual"][loop_idx], 1e-12)

    def test_induced_signs_give_the_closed_form_relation(self):
        # The label-free content of endSignCovector: a CLOSED 3-cochain's
        # signed periods over all of the structure's hole cycles sum to zero
        # (Σ_h σ_h p_h = 0 — the holes' induced boundaries cancel against the
        # present top cells' coherent boundary). The symmetric metric operator
        # is assembled with B₄ = W₃^{1/2} ∂₄ W₄^{-1/2}, so a kernel row ψ̃ of
        # harmonicMatrix(3) satisfies ∂₄ᵀ(√W₃ ψ̃) = 0: the closed cochain is
        # χ = √W₃ ψ̃ (ψ̃ itself is not closed, and its raw signed periods do
        # NOT cancel — the register's period pin is a convention, not Stokes).
        all_holes = [list(h) for h in cob.MultiCobordism.emergent_holes(self.st, 3)]
        sigma = self.plf.induced_orientation_signs(self.st, all_holes)
        es = cob.EigenstateSynthesis(self.st, 3)
        cell_index = {frozenset(t): i for i, t in enumerate(es.cellSimplices())}
        hl = cob.HodgeLaplacian(self.st)
        sqrt_w = np.sqrt(np.asarray(hl.weights(3), float))
        harmonics = np.asarray(hl.harmonicMatrix(3), complex)
        harmonics = harmonics.reshape(-1, len(sqrt_w))
        self.assertGreater(harmonics.shape[0], 0)

        def period(vec, hole):
            pairs = self.plf._facet_indices(cell_index, hole)
            return sum(sign * vec[c] for c, sign in pairs)

        for row in harmonics:
            chi = sqrt_w * row
            signed_sum = sum(s * period(chi, h)
                             for s, h in zip(sigma, all_holes))
            self.assertLess(abs(signed_sum), 1e-12)

    def test_charges_positive_and_additive_over_the_pair_support(self):
        # q_h > 0 (a weighted norm²), and the pair-loop charge over the
        # disjoint union of the two boundary cycles is exactly q_i + q_j.
        read, plf = self.read, self.plf
        for q in read["q"]:
            self.assertGreater(q, 0.0)
        for loop_idx, (i, j) in enumerate(plf.PAIR_LOOPS):
            self.assertAlmostEqual(read["loop_q"][loop_idx],
                                   read["q"][i] + read["q"][j], places=15)


class TestCriteria(PairLoopFlavorBase):
    def test_odd_one_out_on_a_clean_triple(self):
        odd, rho = self.plf.odd_one_out([1.0, 1.01, 2.0])
        self.assertEqual(odd, 2)
        self.assertLess(rho, 0.02)

    def test_evaluate_criteria_verdict_shape(self):
        verdict = self.plf.evaluate_criteria(self.read)
        self.assertIn(verdict["odd_loop"], self.plf.PAIR_LOOPS)
        self.assertEqual(verdict["dual_hole"],
                         self.plf.complement_hole(verdict["odd_loop"]))
        self.assertIsInstance(verdict["multiplicity_2_1"], bool)
        # fixtures carry no build history: criterion (b) must stay undecided
        self.assertIsNone(verdict["odd_is_diquark_loop"])

    def test_criterion_b_decides_with_build_history(self):
        # With a supplied step-1 diquark pair the criterion becomes decidable:
        # True iff the charge-odd loop is exactly that pair.
        odd_pair = self.plf.evaluate_criteria(self.read)["odd_loop"]
        other = next(p for p in self.plf.PAIR_LOOPS if p != odd_pair)
        hit = self.plf.evaluate_criteria(self.read, diquark_pair=odd_pair)
        miss = self.plf.evaluate_criteria(self.read, diquark_pair=other)
        self.assertTrue(hit["odd_is_diquark_loop"])
        self.assertFalse(miss["odd_is_diquark_loop"])


class TestGates(PairLoopFlavorBase):
    def _assert_gate(self, residuals):
        for key, value in residuals.items():
            tol = (self.plf.RHO_GATE_TOL if key == "d_rho"
                   else self.plf.GATE_TOL)
            self.assertLess(value, tol, f"{key} = {value:.3e} exceeds {tol:.0e}")

    def test_gauge_gate(self):
        self._assert_gate(self.plf.gauge_gate(self.st, self.holes, self.read))

    def test_gauge_gate_covers_the_cyclic_recolor(self):
        # A cyclic recolor of the singlet assignment IS a global U(1) phase:
        # shifting [1, ω, ω²] by one slot equals ω² · [1, ω, ω²], so the
        # charges must be blind to which hole carries which color.
        t = list(self.plf.SINGLET)
        shifted = [t[2], t[0], t[1]]
        expected = [t[i] * (self.plf.OMEGA ** 2) for i in range(3)]
        for a, b in zip(shifted, expected):
            self.assertLess(abs(a - b), 1e-15)
        recolored = self.plf.joint_read(self.st, self.holes, shifted)
        for a, b in zip(recolored["q"], self.read["q"]):
            self.assertLess(abs(a - b), self.plf.GATE_TOL)

    def test_relabel_gate_two_permutations(self):
        for seed in (3, 11):
            with self.subTest(seed=seed):
                self._assert_gate(self.plf.relabel_gate(
                    self.cells, self.edges, self.holes, self.read, seed=seed))

    def test_determinism(self):
        st2 = self.plf.build_spacetime(self.cells, self.edges)
        again = self.plf.joint_read(st2, self.plf.register_holes(st2))
        self.assertEqual(again["sigma"], self.read["sigma"])
        for key in ("w", "q", "loop_w", "loop_q", "dual_residual"):
            for a, b in zip(again[key], self.read[key]):
                self.assertLess(abs(a - b), 1e-15)


if __name__ == "__main__":
    unittest.main()
