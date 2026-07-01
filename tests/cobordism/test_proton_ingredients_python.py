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


if __name__ == "__main__":
    unittest.main()
