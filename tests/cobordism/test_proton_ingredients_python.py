# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""ProtonIngredients — the emergent arm of the proton build (#555).

`Proton` is the canonical line in the sand; `ProtonIngredients` runs the same two-step
drive except step B's output-target list is EMPTY — nothing is pinned downstream, the
objective is `F = ‖∇S‖² + Γ·Σᵢ r_U(inputᵢ)`, and the final state is read after the fact.

These tests lock in (1) the engine regression that an empty `output_targets` list is a
supported `MultiCobordism` shape whose `r_u` contains ONLY the input terms, (2) that the
ingredients' step-B node differs from the canonical one in exactly that way (the input
terms scale linearly with the input weight; the canonical node carries a
weight-independent whole-read singlet term on top), and (3) that the slow build reports a
coherent, answer-agnostic summary — deliberately asserting nothing about the singlet or
the hole count, which are observables of the experiment, not requirements.

The JOINT arm (#560) is locked in the same way: `joint_node` collapses the two-step
event graph into one co-optimized node — the Z₃-orbit neutral triple in, a baryon ⊔
antibaryon block pair out (the multi-output `r_u` branch, so `r_u` is NOT linear in the
input weight) — and the bounded slow smoke checks `build_joint` reports the same
answer-agnostic summary plus the per-output-block residuals.
"""
import math
import unittest

import pytest

import tessera

cob = tessera.cobordism


class MultiCobordismEmptyOutputsTest(unittest.TestCase):
    """Engine regression: `output_targets=[]` is a supported shape (nothing pinned
    downstream) — the objective's matter term is the weighted input residuals alone."""

    def test_ctor_accepts_empty_outputs_and_bare_r_u_is_zero(self):
        # A fresh single-Δ⁴ host via the ingredients' own factory (precone=0 leaves it
        # untouched); the raw ctor with NO output targets and no seeded blocks has no
        # residual terms at all.
        host = cob.ProtonIngredients().formation_node(1).st
        node = cob.MultiCobordism(host, [[1.0, -1.0, 0.0]], [], degrees=[3],
                                  gamma=1.0, seed=0)
        self.assertEqual(node.r_u(node.st), 0.0)
        self.assertEqual(len(node.outputs), 0)
        self.assertTrue(math.isfinite(node.objective()))

    def test_r_u_with_empty_outputs_is_exactly_the_weighted_input_terms(self):
        # Seed the input block, then scale the input weight: with no output term the
        # whole of r_u must scale linearly — an output term would break the linearity.
        host = cob.ProtonIngredients().formation_node(2).st
        node = cob.MultiCobordism(host, [[1.0, -1.0, 0.0]], [], degrees=[3],
                                  gamma=1.0, seed=0)
        vertex_id = host.getVertexList().toVector()[0].getId()
        node.seed_inputs([vertex_id])
        node.set_input_residual_weight(1.0)
        r_at_weight_1 = node.r_u(node.st)
        node.set_input_residual_weight(2.0)
        r_at_weight_2 = node.r_u(node.st)
        self.assertAlmostEqual(r_at_weight_2, 2.0 * r_at_weight_1, places=9)


class ProtonIngredientsNodesTest(unittest.TestCase):
    """The two node factories: step A is the canonical node verbatim; step B is the
    canonical formation node minus the singlet output target — exactly one delta."""

    def test_recombination_node_is_the_canonical_shape(self):
        # Delegated to the composed Proton: 2 input blocks and 2 localized output
        # blocks (diquark ⊔ antidiquark), exactly as Proton.recombination_node seeds it.
        node = cob.ProtonIngredients(seed=5).recombination_node(5)
        self.assertEqual(len(node.inputs), 2)
        self.assertEqual(len(node.outputs), 2)

    def test_formation_node_pins_nothing_downstream(self):
        # The ingredients' step-B r_u is PURELY the weighted input terms (scales
        # linearly with the input weight)...
        ingredients_node = cob.ProtonIngredients(seed=5).formation_node(6)
        self.assertEqual(len(ingredients_node.inputs), 2)
        self.assertEqual(len(ingredients_node.outputs), 0)
        ingredients_node.set_input_residual_weight(1.0)
        r_at_weight_1 = ingredients_node.r_u(ingredients_node.st)
        ingredients_node.set_input_residual_weight(2.0)
        r_at_weight_2 = ingredients_node.r_u(ingredients_node.st)
        self.assertAlmostEqual(r_at_weight_2, 2.0 * r_at_weight_1, places=9)

    def test_canonical_formation_node_still_carries_the_singlet_term(self):
        # ...while the CANONICAL formation node has the weight-independent whole-read
        # singlet term on top, so its r_u is strictly sublinear in the input weight.
        # This is the one variable the A/B experiment changes, asserted from both sides.
        canonical_node = cob.Proton(seed=5).formation_node(6)
        canonical_node.set_input_residual_weight(1.0)
        r_at_weight_1 = canonical_node.r_u(canonical_node.st)
        canonical_node.set_input_residual_weight(2.0)
        r_at_weight_2 = canonical_node.r_u(canonical_node.st)
        self.assertGreater(r_at_weight_1, 0.0)
        self.assertLess(r_at_weight_2, 2.0 * r_at_weight_1 - 0.1)


class ProtonIngredientsJointNodeTest(unittest.TestCase):
    """The joint node factory (#560): the two-step event graph collapsed into ONE
    co-optimized node — Z₃-orbit neutral triple in, baryon ⊔ antibaryon blocks out."""

    def test_joint_node_has_three_inputs_and_two_localized_outputs(self):
        # Inputs at v0,v1,v2 and outputs at v3,v4 of the single Δ⁴ seed: 3 input blocks
        # and 2 localized output blocks (the multi-output r_u branch, exactly the shape
        # the 2->2 recombination exercises).
        node = cob.ProtonIngredients(seed=5).joint_node(5)
        self.assertEqual(len(node.inputs), 3)
        self.assertEqual(len(node.outputs), 2)

    def test_joint_node_inputs_are_the_z3_orbit(self):
        # The #398 symmetric-input lesson: the three pairs are ONE Z₃ orbit — {1,-1,0}
        # and its two cyclic rotations — not an ad-hoc triple. Each is neutral (Σ = 0).
        node = cob.ProtonIngredients(seed=5).joint_node(5)
        orbit = [[1, -1, 0], [0, 1, -1], [-1, 0, 1]]
        for block, expected in zip(node.inputs, orbit):
            self.assertEqual([complex(c) for c in block.target],
                             [complex(c) for c in expected])
            self.assertAlmostEqual(abs(sum(block.target)), 0.0, places=12)

    def test_joint_node_output_targets_are_conjugates(self):
        # Baryon [1,w,w^2] at v3, antibaryon = its component-wise conjugate at v4.
        node = cob.ProtonIngredients(seed=5).joint_node(5)
        baryon, antibaryon = node.outputs[0].target, node.outputs[1].target
        self.assertEqual(len(baryon), 3)
        singlet = cob.Proton.singlet()
        for got, expected in zip(baryon, singlet):
            self.assertAlmostEqual(abs(got - expected), 0.0, places=12)
        for got, expected in zip(antibaryon, baryon):
            self.assertAlmostEqual(abs(got - expected.conjugate()), 0.0, places=12)

    def test_conjugate_targets_score_identically_as_multisets(self):
        # The documented subtlety: [1,conj(w),conj(w)^2] is a component-PERMUTATION of
        # [1,w,w^2] and the block residual is relabeling-invariant, so the two targets
        # score identically against any one complex — the conjugation is carried by the
        # block's location in the emergent complex, never by the residual's value.
        host = cob.ProtonIngredients().formation_node(1).st
        baryon = cob.Proton.singlet()
        antibaryon = [c.conjugate() for c in baryon]
        self.assertAlmostEqual(cob.MultiCobordism.r_state(host, 3, baryon),
                               cob.MultiCobordism.r_state(host, 3, antibaryon),
                               places=12)

    def test_joint_r_u_includes_the_output_block_terms(self):
        # The joint node pins outputs as LOCALIZED BLOCKS, so r_u = weight * (input
        # terms) + (output-block terms): strictly sublinear in the input weight —
        # unlike the two-step's step B, whose r_u is purely the weighted input terms.
        node = cob.ProtonIngredients(seed=5).joint_node(6)
        node.set_input_residual_weight(1.0)
        r_at_weight_1 = node.r_u(node.st)
        node.set_input_residual_weight(2.0)
        r_at_weight_2 = node.r_u(node.st)
        self.assertGreater(r_at_weight_1, 0.0)
        self.assertLess(r_at_weight_2, 2.0 * r_at_weight_1 - 0.1)


@pytest.mark.slow
class ProtonIngredientsBuildTest(unittest.TestCase):
    """Slow: one real emergent-arm attempt. The assertions are deliberately
    answer-agnostic — the singlet residual and hole count are REPORTED observables of
    the experiment (any value is a finding), never requirements."""

    @classmethod
    def setUpClass(cls):
        cls.ingredients = cob.ProtonIngredients(seed=1)
        cls.ingredients.build(max_restarts=1)

    def test_summary_is_coherent(self):
        # converged ⟺ stationary AND persistent — never a statement about the singlet.
        converged = self.ingredients.converged()
        stationary = self.ingredients.stationary()
        persistent = self.ingredients.persistent()
        self.assertIsInstance(converged, bool)
        self.assertEqual(converged, stationary and persistent)

    def test_observables_are_read_and_finite(self):
        self.assertTrue(math.isfinite(self.ingredients.singlet_residual()))
        self.assertTrue(math.isfinite(self.ingredients.input_residual()))
        self.assertTrue(math.isfinite(self.ingredients.final_objective()))
        self.assertTrue(math.isfinite(self.ingredients.diquark_residual()))
        self.assertIsInstance(self.ingredients.emergent_holes(), list)

    def test_whole_complex_exists_with_relaxed_metric(self):
        whole = self.ingredients.spacetime()
        self.assertIsNotNone(whole)
        squared = [e.getSquaredLength() for e in whole.getEdgeList().toVector()]
        self.assertTrue(squared, "emergent complex has no edges")
        self.assertTrue(any(abs(l - complex(1.0, 0.0)) > 1e-9 for l in squared),
                        "metric is unit — the relaxed geometry was lost")
        # block() IS the whole (API parity with Proton.block()): same complex, not a carve.
        block = self.ingredients.block()
        self.assertIsNotNone(block)
        self.assertEqual(len(block.getTopSimplices()), len(whole.getTopSimplices()))
        self.assertEqual(len(block.getEdgeList().toVector()),
                         len(whole.getEdgeList().toVector()))


@pytest.mark.slow
class ProtonIngredientsJointBuildTest(unittest.TestCase):
    """Slow: ONE bounded joint-arm attempt (#560) at smoke budgets. As with the
    two-step, the assertions are answer-agnostic — hole count, singlet diagnostic, and
    the per-output-block residuals are REPORTED observables, never requirements."""

    @classmethod
    def setUpClass(cls):
        cls.ingredients = cob.ProtonIngredients(seed=1)
        cls.ingredients.build_joint(max_restarts=1, init_steps=40, evolve_steps=20,
                                    stage2_max_iters=10)

    def test_summary_is_coherent(self):
        # converged ⟺ stationary AND persistent — never a statement about the singlet.
        converged = self.ingredients.converged()
        stationary = self.ingredients.stationary()
        persistent = self.ingredients.persistent()
        self.assertIsInstance(converged, bool)
        self.assertEqual(converged, stationary and persistent)

    def test_observables_are_read_and_finite(self):
        self.assertTrue(math.isfinite(self.ingredients.singlet_residual()))
        self.assertTrue(math.isfinite(self.ingredients.input_residual()))
        self.assertTrue(math.isfinite(self.ingredients.final_objective()))
        self.assertIsInstance(self.ingredients.emergent_holes(), list)

    def test_per_output_block_residuals_are_read(self):
        # The joint arm's extra observables: one residual per output block, finite and
        # non-negative (each is a least-squares residual against the block's own
        # sub-complex — or the full leak while nothing carries the block).
        baryon_r = self.ingredients.baryon_residual()
        antibaryon_r = self.ingredients.antibaryon_residual()
        self.assertTrue(math.isfinite(baryon_r))
        self.assertTrue(math.isfinite(antibaryon_r))
        self.assertGreaterEqual(baryon_r, 0.0)
        self.assertGreaterEqual(antibaryon_r, 0.0)

    def test_diquark_residual_is_nan_without_a_step_a(self):
        # The joint arm collapses the diquark intermediate away — its observable is
        # explicitly not-a-number, never a stale zero pretending step A converged.
        self.assertTrue(math.isnan(self.ingredients.diquark_residual()))

    def test_whole_complex_exists_and_block_is_the_whole(self):
        whole = self.ingredients.spacetime()
        self.assertIsNotNone(whole)
        self.assertTrue(whole.getEdgeList().toVector(),
                        "emergent complex has no edges")
        block = self.ingredients.block()
        self.assertEqual(len(block.getTopSimplices()), len(whole.getTopSimplices()))


if __name__ == "__main__":
    unittest.main()
