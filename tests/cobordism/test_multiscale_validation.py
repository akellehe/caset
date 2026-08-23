# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#777 — the multiscale-validation harness, exercised end to end.

Per repository convention the STUDY lives in ``examples/`` and its findings in
``docs/design/``; this file covers only the correctness of the INSTRUMENT:

* the analytic small fixtures the study reports are exact at machine
  precision, and the mandated NEGATIVE spin fixture is exactly nonzero;
* the declared one-particle spin convention has the two structural
  properties the study leans on — a FULLY PAIRED rank-1 covariance is
  trivially a J^2 = 3/4 eigenstate, and the same state with one mode left
  unpaired picks up a spin-0 admixture and a nonzero variance — so neither
  read can be mistaken for evidence about the geometry;
* every mandated negative control FIRES on a small host (a control that
  silently passes is a bug in the instrument);
* the ``--quick`` path runs end to end and emits a complete, JSON-round-
  trippable document with the schema the findings report is written from;
* no threshold is per-size: the emitted config carries exactly one declared
  threshold set;
* the small-fit helpers refuse to invent an uncertainty they do not have.

The full study is NOT run here.
"""

import json
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import multiscale_validation as mv  # noqa: E402

QU = T.quantum

#: The smallest host the harness is meaningful on — one committed move, a
#: two-component modularity read, and a full band enumeration.
SMALL = 6
SMALL_SEED = 7


class AnalyticInvariantTest(unittest.TestCase):
    """The exact fixtures must stay exact — that is the study's floor."""

    @classmethod
    def setUpClass(cls):
        cls.invariants = {entry["name"]: entry
                          for entry in mv.analytic_invariants()}

    def test_every_declared_invariant_is_exact(self):
        for name, entry in self.invariants.items():
            with self.subTest(invariant=name):
                self.assertIsNotNone(entry["residual"],
                                     f"{name} reported no residual")
                self.assertTrue(entry["exact"],
                                f"{name} residual {entry['residual']} "
                                f"exceeded {entry['tolerance']}")

    def test_the_expected_invariants_are_all_present(self):
        for name in ("fourier_frame_unitary", "su3_singlet_gram_invariance",
                     "spin_double_cover_d3", "spin_double_cover_d4",
                     "sharp_spin_half_expectation",
                     "sharp_spin_half_variance",
                     "generic_slater_expectation_three_quarters",
                     "generic_slater_variance_is_fifteen_sixteenths",
                     "near_isometry_budget",
                     "berry_cancelled_single_exchange"):
            self.assertIn(name, self.invariants)

    def test_the_negative_spin_fixture_is_genuinely_nonzero(self):
        """<J^2> = 3/4 with Var = 15/16 — the whole point of §5.12."""
        root = 1.0 / math.sqrt(2.0)
        sx = np.array([[0, root, 0], [root, 0, root], [0, root, 0]],
                      dtype=complex)
        sy = np.array([[0, -1j * root, 0], [1j * root, 0, -1j * root],
                       [0, 1j * root, 0]], dtype=complex)
        sz = np.diag([1.0, 0.0, -1.0]).astype(complex)

        def pad(block):
            out = np.zeros((4, 4), dtype=complex)
            out[1:, 1:] = block
            return out

        orbital = np.zeros((4, 1), dtype=complex)
        orbital[0, 0] = math.sqrt(5.0 / 8.0)
        orbital[1, 0] = math.sqrt(3.0 / 8.0)
        state = QU.CovarianceState.fromSlaterFrame(orbital)
        variance = complex(state.wickSpinSquaredVariance(
            pad(sx), pad(sy), pad(sz)).value).real
        self.assertGreater(variance, 0.5)


class DeclaredSpinConventionTest(unittest.TestCase):
    """The convention's structural property, stated so it cannot be misread."""

    def test_a_fully_paired_rank_one_state_is_a_trivial_spin_half_eigenstate(
            self):
        """A single fermion in a fully paired doublet operator is j = 1/2.

        This is why the study labels such reads `trivial_rank1`: `<J^2> = 3/4`
        and `Var(J^2) = 0` are identities of the READOUT and carry no
        information about the geometry at all.
        """
        rng = np.random.default_rng(4)
        for modes in (2, 4, 6, 8):
            orbital = (rng.normal(size=(modes, 1))
                       + 1j * rng.normal(size=(modes, 1)))
            orbital /= np.linalg.norm(orbital)
            state = QU.CovarianceState.fromSlaterFrame(orbital)
            jx, jy, jz = mv.declared_spin_matrices(modes)
            expectation = complex(
                state.wickSpinSquaredExpectation(jx, jy, jz).value)
            variance = complex(
                state.wickSpinSquaredVariance(jx, jy, jz).value)
            with self.subTest(modes=modes):
                self.assertLess(abs(expectation - 0.75), 1e-12)
                self.assertLess(abs(variance), 1e-12)

    def test_an_unpaired_mode_makes_the_same_state_a_nonsharp_read(self):
        """The other half of the caveat: the leftover mode is spin-0.

        With an odd mode count the declared pairing leaves one spinless mode,
        so a rank-1 state with weight on it is a j = 1/2 + j = 0 superposition
        with genuinely nonzero Var(J^2) — an artifact of the convention, not
        of the geometry, and exactly the design spec §5.12 negative shape.
        """
        modes = 5
        orbital = np.zeros((modes, 1), dtype=complex)
        orbital[0, 0] = math.sqrt(0.5)     # inside a doublet
        orbital[4, 0] = math.sqrt(0.5)     # the unpaired, spin-0 mode
        state = QU.CovarianceState.fromSlaterFrame(orbital)
        jx, jy, jz = mv.declared_spin_matrices(modes)
        expectation = complex(
            state.wickSpinSquaredExpectation(jx, jy, jz).value).real
        variance = complex(
            state.wickSpinSquaredVariance(jx, jy, jz).value).real
        self.assertLess(abs(expectation - 0.375), 1e-12)   # (3/4) * 1/2
        self.assertGreater(variance, 0.1)

    def test_the_pairing_offset_is_a_different_operator(self):
        """The offset arm must actually differ, or the reported pairing
        spread would be a vacuous zero."""
        jx0, _, _ = mv.declared_spin_matrices(6, 0)
        jx1, _, _ = mv.declared_spin_matrices(6, 1)
        self.assertGreater(float(np.abs(np.array(jx0) - np.array(jx1)).max()),
                           0.1)

    def test_the_spin_matrices_are_hermitian(self):
        for matrix in mv.declared_spin_matrices(7):
            array = np.array(matrix)
            self.assertLess(float(np.abs(array - array.conj().T).max()),
                            1e-15)


class NegativeControlTest(unittest.TestCase):
    """Every mandated control must FIRE; a silent pass is an instrument bug."""

    @classmethod
    def setUpClass(cls):
        config = mv.make_config(quick=True, sizes=[SMALL],
                                seeds=[SMALL_SEED])
        cls.controls = {control["name"]: control
                        for control in mv.negative_controls(SMALL, config)}

    def test_all_eight_mandated_controls_are_present(self):
        self.assertEqual(
            set(self.controls),
            {"shuffled_phases", "destroyed_modularity",
             "modularity_resolution_limit", "unanchored_rank_three_band",
             "closed_spectral_and_rank_gaps", "cube_root_branch_change",
             "uncancelled_berry_loop", "disabled_grading"})

    def test_every_control_fires(self):
        for name, control in self.controls.items():
            with self.subTest(control=name):
                self.assertTrue(
                    control["fired"],
                    f"{name} did NOT fire: {json.dumps(control['detail'])}")

    def test_the_resolution_limit_control_really_merges_cliques(self):
        detail = self.controls["modularity_resolution_limit"]["detail"]
        self.assertLess(detail["components_at_gamma_1"], detail["cliques"])
        self.assertGreater(detail["components_at_gamma_4"],
                           detail["components_at_gamma_1"])

    def test_the_grading_control_separates_pauli_from_the_permanent(self):
        detail = self.controls["disabled_grading"]["detail"]
        self.assertLess(detail["graded_amplitude"], 1e-12)
        self.assertGreater(detail["ungraded_amplitude"], 1e-6)

    def test_the_berry_control_shows_the_raw_determinant_is_not_a_sign(self):
        detail = self.controls["uncancelled_berry_loop"]["detail"]
        raw = complex(*detail["raw_determinant"])
        self.assertGreater(min(abs(raw - 1.0), abs(raw + 1.0)), 0.1)
        self.assertEqual(detail["character_sign"], -1)


class FitHelperTest(unittest.TestCase):
    """The fits must refuse to invent uncertainty they do not have."""

    def test_a_two_point_fit_is_refused(self):
        self.assertIsNone(mv.linear_fit([1.0, 2.0], [1.0, 2.0]))

    def test_an_exact_line_fits_with_zero_residual(self):
        fit = mv.linear_fit([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(fit["slope"], 2.0, places=12)
        self.assertAlmostEqual(fit["intercept"], 1.0, places=12)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=12)

    def test_a_constant_series_has_no_correlation(self):
        self.assertIsNone(mv.pearson([1.0, 2.0, 3.0, 4.0],
                                     [5.0, 5.0, 5.0, 5.0]))

    def test_fewer_than_four_points_have_no_correlation(self):
        self.assertIsNone(mv.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))

    def test_inverse_size_fit_reports_a_conservative_verdict(self):
        # Perfect y = 4 - 12/N: converging, extrapolating to 4.
        sizes = [6, 12, 20, 30, 44]
        values = [4.0 - 12.0 / size for size in sizes]
        result = mv.inverse_size_fit(sizes, values)
        self.assertEqual(result["verdict"], "converging")
        self.assertAlmostEqual(result["extrapolated_limit"], 4.0, places=9)

    def test_unknown_is_none_never_zero(self):
        self.assertIsNone(mv._finite(float("nan")))
        self.assertIsNone(mv._finite(float("inf")))
        self.assertIsNone(mv._finite(None))
        self.assertEqual(mv._finite(0.0), 0.0)


class QuickPathTest(unittest.TestCase):
    """The whole --quick path, end to end, on the smallest declared host."""

    @classmethod
    def setUpClass(cls):
        cls.result = mv.main([
            "--quick", "--sizes", str(SMALL), "--seeds", str(SMALL_SEED)])
        # The document a findings report is written from must round-trip.
        cls.round_tripped = json.loads(json.dumps(cls.result))

    def test_the_document_round_trips_through_json(self):
        self.assertEqual(self.result["schema_version"], mv.SCHEMA_VERSION)
        self.assertEqual(self.round_tripped["ticket"], 777)
        self.assertEqual(self.round_tripped["epic"], 763)

    def test_every_point_carries_its_provenance(self):
        for run in self.result["runs"]:
            self.assertEqual(run["config_hash"], self.result["config_hash"])
            self.assertEqual(run["commit"], self.result["commit"])
            self.assertIn("size", run)
            self.assertIn("seed", run)

    def test_the_config_hash_is_a_function_of_the_config_alone(self):
        self.assertEqual(mv.config_hash_of(self.result["config"]),
                         self.result["config_hash"])

    def test_the_run_is_an_unforced_strict_emergence_run(self):
        for run in self.result["runs"]:
            checkpoint = run["checkpoint"]
            self.assertEqual(checkpoint["mode"], "emergence")
            self.assertEqual(checkpoint["emergence_submode"], "strict")
            # Nothing entered the objective through the state channel.
            self.assertEqual(
                checkpoint["objective"]["carried_state_energy"], 0.0)
            self.assertEqual(
                checkpoint["objective"]["carried_state_energy_weight"], 0.0)

    def test_no_threshold_is_per_size(self):
        """One declared threshold set for the whole ensemble."""
        config = self.result["config"]
        for key in ("degrees", "analysis_resolution", "resolution_scan",
                    "shift_fractions", "window_half_width_fraction",
                    "amls_mode_cutoff", "degeneracy_tolerance",
                    "degeneracy_tolerances", "threshold_scan",
                    "candidate_moves", "stage2_iters", "krylov_dim"):
            self.assertIn(key, config)
            self.assertNotIsInstance(config[key], dict,
                                     f"{key} is keyed per size")

    def test_every_named_measurement_block_is_present(self):
        run = self.result["runs"][0]
        for block in ("stationarity", "hierarchy", "analysis_modularity",
                      "components", "bands", "reduction", "amplitudes",
                      "gauge", "statistics", "particles",
                      "particles_resolution_scan", "covariance",
                      "spectral_dimension"):
            self.assertIn(block, run, f"missing measurement block {block}")

    def test_the_dichotomy_returns_one_of_the_three_declared_outcomes(self):
        self.assertIn(self.result["dichotomy"]["classification"],
                      {"covariance_only_proton",
                       "quasi_free_sharp_spin_obstruction",
                       "inconclusive"})
        self.assertTrue(self.result["dichotomy"]["reason"])

    def test_an_uncertified_read_is_reported_as_data_not_hidden(self):
        run = self.result["runs"][0]
        self.assertGreater(run["particles"]["quark_reads"], 0)
        for classification, count in run["particles"][
                "classifications"].items():
            self.assertGreater(count, 0)
            if classification == "none":
                # A "none" read must NAME what failed.
                self.assertTrue(run["particles"]["first_failing_certificate"])

    def test_rank_one_spin_reads_are_labelled_by_their_pairing(self):
        for run in self.result["runs"]:
            for state in run["covariance"]["states"]:
                if not state.get("available") or state.get("rank") != 1:
                    continue
                if state["modes"] % 2 == 0:
                    self.assertTrue(state["trivial_rank1"])
                    self.assertFalse(state["rank1_with_unpaired_mode"])
                else:
                    self.assertFalse(state["trivial_rank1"])
                    self.assertTrue(state["rank1_with_unpaired_mode"])
        for entry in self.result["dichotomy"]["var_j2_on_nontrivial_bands"]:
            self.assertNotEqual(entry["rank"], 1)

    def test_the_spin_convention_dominance_is_reported(self):
        dominance = self.result["dichotomy"]["spin_convention_dominance"]
        self.assertIn("reads_total", dominance)
        self.assertIn("var_j2_pairing_spread", dominance)
        self.assertIn("not supplied by", dominance["meaning"])

    def test_the_stationarity_correlation_is_labelled_conjectural(self):
        correlation = self.result["stationarity_defect_correlation"]
        self.assertIn("CONJECTURAL", correlation["status"])
        self.assertIn("envelope", correlation["rejected_argument"])

    def test_the_spectral_dimension_reuses_the_existing_estimator(self):
        verdict = self.result["spectral_dimension_verdict"]
        self.assertIn("getSpectralDimensionOnSkeleton", verdict["estimator"])
        self.assertEqual(verdict["pinned_baseline"], mv.PINNED_DS_BASELINE)
        run = self.result["runs"][0]
        curve = run["spectral_dimension"]["curve"]
        self.assertEqual(len(curve), len(run["spectral_dimension"]["sigmas"]))
        self.assertIsNotNone(run["spectral_dimension"]["peak"])

    def test_the_degeneracy_report_does_not_name_a_mechanism(self):
        report = self.result["degeneracy"]
        self.assertIn("NOT automatically", report["interpretation"])
        self.assertIsNotNone(report["clusters_total"])

    def test_no_seed_is_dropped(self):
        expected = {(size, seed)
                    for size in self.result["config"]["sizes"]
                    for seed in self.result["config"]["seeds"]}
        actual = {(run["size"], run["seed"]) for run in self.result["runs"]}
        self.assertEqual(actual, expected)

    def test_every_checkpoint_replays_cold_to_the_same_verdicts(self):
        replay = self.result["replay"]
        self.assertTrue(replay["all_replayed"])
        self.assertTrue(replay["all_discrete_verdicts_identical"],
                        "a cold replay changed a DISCRETE verdict")
        for entry in replay["entries"]:
            self.assertEqual(entry["mode"], "replay")
            for block, within in entry["blocks_within_tolerance"].items():
                self.assertTrue(
                    within,
                    f"replayed block {block} differs by "
                    f"{entry['worst_relative_difference'][block]} at "
                    f"{entry['worst_difference_at'].get(block)}")


class ResponseNetworkEmptyReductionTest(unittest.TestCase):
    """The #777 fix: an empty reduction is reported, never a fault.

    A partition whose single component covers every cell keeps no interface
    coordinate, so the reduced operator is 0 x 0. That used to segfault in a
    Release build; it is now the exactly-empty network.
    """

    def test_a_single_covering_component_yields_the_empty_network(self):
        spacetime = mv.build_host(SMALL)
        vertices = sorted(v.getId()
                          for v in spacetime.getVertexList().toVector())
        quotient = T.cobordism.RecursiveQuotient.overVertexSupports(
            spacetime, 1, [vertices],
            T.cobordism.RecursiveQuotient.Options())
        self.assertEqual(len(quotient.interfaceIndices), 0)
        network = quotient.responseNetwork()
        self.assertEqual(list(network.stalkDimensions), [0])
        self.assertEqual(len(network.edges), 0)
        self.assertEqual(network.coverageResidual, 0.0)
        self.assertTrue(network.certificate.holds())


if __name__ == "__main__":
    unittest.main()
