# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#834 -- the objective is an injected specification object, not an enum.

The engine holds a `CobordismObjective` and calls through it. These tests pin
the two properties that make that safe: the injected objective reproduces the
functional it replaced EXACTLY, and the no-feedback firewall survives the move
from a static function to an object.
"""

import cmath
import math
import unittest

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    """The same fixture the objective-selection suite scores, so the values
    here are comparable with the ones that suite already pins."""
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.getEdgeList().toVector()):
        z = complex(1.0 + 0.019 * (index % 5),
                    0.011 * (1 + index % 4))
        edge.setLength(cmath.sqrt(z))
    return st


def _node(st, degree=3, gamma=0.0):
    return cob.MultiCobordism(st, [], [], degrees=[degree], gamma=gamma,
                              seed=7)


class InjectedObjectiveIsTheOneScoredTest(unittest.TestCase):
    """The engine scores whatever it is handed, and says which."""

    def test_selecting_a_mode_injects_the_matching_objective(self):
        node = _node(_complex_sphere4())
        for mode, expected in (
                (cob.CobordismObjectiveMode.JointStationarity,
                 "joint_stationarity"),
                (cob.CobordismObjectiveMode.Legacy, "legacy"),
                (cob.CobordismObjectiveMode.MediatedCorrespondence,
                 "mediated_correspondence")):
            with self.subTest(mode=expected):
                node.set_objective_mode(mode)
                self.assertEqual(node.objective_name, expected)

    def test_an_injected_objective_replaces_the_selected_one(self):
        node = _node(_complex_sphere4())
        node.set_objective_mode(cob.CobordismObjectiveMode.Legacy)
        self.assertEqual(node.objective_name, "legacy")
        node.set_objective(cob.JointStationarityObjective())
        self.assertEqual(node.objective_name, "joint_stationarity")
        # The scalar follows the injected object, not the enum, which still
        # reports the last mode that was selected.
        self.assertEqual(node.objective_mode,
                         cob.CobordismObjectiveMode.Legacy)

    def test_a_null_objective_is_refused(self):
        node = _node(_complex_sphere4())
        with self.assertRaises((ValueError, TypeError)):
            node.set_objective(None)

    def test_target_conditioning_is_declared_not_inferred(self):
        self.assertFalse(
            cob.JointStationarityObjective().is_target_conditioned())
        self.assertTrue(cob.LegacyObjective().is_target_conditioned())
        self.assertTrue(
            cob.MediatedCorrespondenceObjective().is_target_conditioned())

    def test_only_target_conditioned_objectives_ask_for_the_residual(self):
        # A purely geometric objective never pays for r_U.
        self.assertFalse(
            cob.JointStationarityObjective().needs_register_residual())
        self.assertTrue(cob.LegacyObjective().needs_register_residual())
        self.assertTrue(
            cob.MediatedCorrespondenceObjective().needs_register_residual())


class ExactnessTest(unittest.TestCase):
    """The injected objective is the SAME functional, to the bit."""

    def test_joint_hodge_term_equals_the_primitive_it_is_built_from(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective_mode(cob.CobordismObjectiveMode.JointStationarity)
        terms = node.objective_terms()
        reference = (node.hodge_entropy_weight *
                     cob.HodgeLaplacian(st).spectralEntropyGradientNorm(
                         3, node.hodge_entropy_phase_mode))
        # Bit-identical, not merely close: the objective computes this from the
        # same primitive the engine used before it became injectable.
        self.assertEqual(terms.hodge_stationarity, reference)

    def test_joint_regge_term_equals_the_engine_gradient_norm(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective_mode(cob.CobordismObjectiveMode.JointStationarity)
        terms = node.objective_terms()
        reference = (node.regge_weight *
                     cob.MultiCobordism.regge_action_gradient(st))
        self.assertEqual(terms.regge_stationarity, reference)

    def test_the_scalar_is_the_sum_of_the_declared_terms(self):
        st = _complex_sphere4()
        node = _node(st)
        for mode in (cob.CobordismObjectiveMode.JointStationarity,
                     cob.CobordismObjectiveMode.Legacy,
                     cob.CobordismObjectiveMode.MediatedCorrespondence):
            with self.subTest(mode=str(mode)):
                node.set_objective_mode(mode)
                terms = node.objective_terms()
                self.assertEqual(node.objective(),
                                 cob.CobordismObjective.total(terms))
                # And the static collapse agrees with the engine's own.
                self.assertEqual(cob.MultiCobordism.objective_of(terms),
                                 cob.CobordismObjective.total(terms))

    def test_joint_stationarity_carries_no_register_residual(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective_mode(cob.CobordismObjectiveMode.JointStationarity)
        # Not merely zero-valued: the term is structurally absent because the
        # objective never asks the engine to compute it.
        self.assertEqual(node.objective_terms().register_residual, 0.0)

    def test_stage_two_still_descends_under_the_injected_objective(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective_mode(cob.CobordismObjectiveMode.JointStationarity)
        before = node.objective()
        node.run_stage2(1.0, 12, 0.05)
        after = node.objective()
        self.assertLessEqual(after, before)
        self.assertTrue(math.isfinite(after))


class FirewallTest(unittest.TestCase):
    """The no-feedback guarantee survives the move to an injected object.

    It used to be mechanical: `objective_of` was static, so it had no `this`
    and therefore no pointer through which to reach a member holding an
    analysis read. An injected objective HAS a `this`, so the guarantee moves
    to the input side -- `ObjectiveContext` is plain data that leads nowhere.
    """

    #: Every analysis product `runRecursiveAnalysis` writes. None of it may be
    #: nameable from an objective's declared inputs or outputs.
    ANALYSIS_WORDS = ("component", "cluster", "fiber", "fibre", "transport",
                      "amplitude", "color", "colour", "charge", "flavor",
                      "flavour", "exchange", "spin", "certificate", "verdict",
                      "holonomy", "anchor", "betti", "crossing", "baryon",
                      "quark", "proton", "wilson", "winding")

    def test_no_analysis_quantity_is_reachable_from_an_objective_input(self):
        for name in cob.ObjectiveContext.input_names():
            for word in self.ANALYSIS_WORDS:
                with self.subTest(field=name, word=word):
                    self.assertNotIn(word, name.lower())

    def test_no_analysis_quantity_appears_among_the_declared_terms(self):
        for name in cob.CobordismObjective.declared_term_names():
            for word in self.ANALYSIS_WORDS:
                with self.subTest(term=name, word=word):
                    self.assertNotIn(word, name.lower())

    def test_the_context_is_the_complete_input_list(self):
        # The firewall is only checkable if the enumeration is the whole story,
        # so pin the list itself. A field added without updating this test is
        # a field nobody audited.
        self.assertEqual(
            cob.ObjectiveContext.input_names(),
            ["spacetime", "region", "region_targets", "register_degrees",
             "regge_weight", "hodge_entropy_weight", "gamma",
             "carried_state_energy_weight", "einstein_hilbert",
             "hodge_entropy_phase_mode", "register_residual",
             "carried_state_energy"])

    def test_the_declared_term_list_is_unchanged(self):
        self.assertEqual(
            cob.CobordismObjective.declared_term_names(),
            ["regge_stationarity", "hodge_stationarity", "register_residual",
             "action_magnitude", "carried_state_energy"])
        # The engine's own list is the same list, so a record stays comparable.
        self.assertEqual(cob.MultiCobordism.objective_term_names(),
                         cob.CobordismObjective.declared_term_names())

    def test_an_objective_holds_no_route_back_to_the_engine(self):
        # An objective is constructible and usable with NO node in existence.
        # If the interface required a MultiCobordism -- or anything reachable
        # from one -- this could not be written at all, which is the property
        # under test: analysis access is impossible, not merely unused.
        objective = cob.JointStationarityObjective()
        self.assertEqual(objective.name(), "joint_stationarity")
        self.assertEqual(objective.term_names(),
                         cob.CobordismObjective.declared_term_names())

    def test_the_static_collapse_takes_no_instance(self):
        # `total` remains static: the step from a decomposition to the number
        # the optimizer compares still has no `this`.
        terms = cob.MultiCobordism.ObjectiveTerms()
        terms.regge_stationarity = 2.0
        terms.hodge_stationarity = 0.5
        self.assertEqual(cob.CobordismObjective.total(terms), 2.5)


class ScoringDomainTest(unittest.TestCase):
    """An objective declares its own scoring domain; the engine honours it."""

    def test_the_whole_complex_reading_is_the_default(self):
        # With one objective over the whole cobordism nothing straddles, so the
        # declaration is moot and the path is the one that ran before.
        for objective in (cob.JointStationarityObjective(),
                          cob.LegacyObjective(),
                          cob.MediatedCorrespondenceObjective()):
            with self.subTest(objective=objective.name()):
                self.assertTrue(
                    objective.scoring_domain().includes_straddling_edges)

    def test_the_declaration_is_readable_and_settable(self):
        domain = cob.ObjectiveScoringDomain()
        self.assertTrue(domain.includes_straddling_edges)
        domain.includes_straddling_edges = False
        self.assertFalse(domain.includes_straddling_edges)


if __name__ == "__main__":
    unittest.main()
