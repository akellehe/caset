# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#778 — the complete recursive baryon simulation, exercised end to end.

Per repository convention the SIMULATION lives in ``examples/`` and its
findings in ``docs/design/``; this file covers the correctness of the
INSTRUMENT:

* the fast path runs end to end, exits successfully, and emits a complete,
  JSON-round-trippable document with the schema the findings report is
  written from;
* the replay path reproduces every stored verdict and content hash with
  cold caches, and rejects an unknown schema version;
* the schema round trip carries ``null`` for every unknown and never a
  zero, and every absent read carries a NAMED reason;
* the exactness fixtures the ticket names — static Schur, shifted Feshbach,
  second-quantized subset-sum and hopping, triangle anchor, center branch,
  Berry cancellation — agree with analytic or dense references computed
  independently here;
* the campaign aggregates over at least three sizes with runtime, memory
  and cache statistics;
* the animation and the headless path read the SAME checkpoint data;
* the verdict vocabulary contains no target-dependent success string, and
  the verdict is the library classifier's own, relayed verbatim.

The large mode and the full campaign are NOT run here.
"""

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import recursive_baryon_simulation as rbs  # noqa: E402

QU = T.quantum
cob = T.cobordism

#: The smallest host the simulation is meaningful on: one committed move, a
#: multi-component modularity read, and a full band enumeration.
SMALL = 6
SMALL_SEED = 7

#: One shared fast run, built once — the whole suite reads it.
_RUN_CACHE = {}


def small_config(**overrides):
    config = rbs.make_config(size=SMALL, seed=SMALL_SEED, drive_steps=1)
    config.update(overrides)
    return config


def shared_run():
    """A single fast run and its sidecar directory, built once."""
    if "document" not in _RUN_CACHE:
        directory = tempfile.mkdtemp(prefix="rbs778-")
        document = rbs.run_simulation(
            small_config(), commit="testcommit",
            sidecar_path=os.path.join(directory, "run.sidecar.npz"))
        path = os.path.join(directory, "run.json")
        with open(path, "w") as handle:
            json.dump(document, handle)
        _RUN_CACHE.update({"document": document, "directory": directory,
                           "path": path})
    return _RUN_CACHE["document"], _RUN_CACHE["directory"]


# =====================================================================
# the verdict vocabulary
# =====================================================================

class VerdictVocabularyTest(unittest.TestCase):
    """Design spec §21.4 item 6: four verdicts, no target-dependent code
    path, and no target-dependent success string anywhere."""

    def test_the_vocabulary_is_exactly_the_four_declared_verdicts(self):
        self.assertEqual(
            set(rbs.VERDICTS),
            {"no baryon", "baryon candidate", "certified proton",
             "quasi-free sharp-spin obstruction"})

    def test_the_vocabulary_carries_no_target_dependent_success_string(self):
        banned = ("success", "found", "pass", "ok", "achieved", "confirmed",
                  "win", "victory", "proven", "emerged")
        for verdict in rbs.VERDICTS:
            for word in banned:
                self.assertNotIn(word, verdict.lower(),
                                 f"{verdict!r} contains {word!r}")

    def test_every_library_classification_maps_into_the_vocabulary(self):
        self.assertEqual(sorted(rbs.LIBRARY_VERDICTS.values()),
                         sorted(rbs.VERDICTS))
        for hyphenated in rbs.LIBRARY_VERDICTS:
            self.assertEqual(
                hyphenated,
                rbs.LIBRARY_VERDICTS[hyphenated].replace(" ", "-"))

    def test_the_classifier_on_empty_evidence_is_a_verdict_not_a_fault(self):
        """The target-independent path: hand the classifier whatever the run
        produced — possibly nothing — and it NAMES every gap."""
        read = T.ParticleClusters().classifyBaryon(T.BaryonCandidateEvidence())
        self.assertIn(read.classification, rbs.LIBRARY_VERDICTS)
        self.assertEqual(read.classification, "no-baryon")
        self.assertIn("constituent-quarks", read.failedCertificates)
        self.assertIn("bound-supercomponent", read.failedCertificates)

    def test_an_unknown_classification_is_refused_not_relabelled(self):
        """A verdict the vocabulary does not contain is an error, never a
        string this file coerces into the nearest known one."""
        class FakeRead:
            classification = "certified-proton-probably"
            failedCertificates = []
            confidence = 1.0

        class FakeReadout:
            @staticmethod
            def baryon_read():
                return FakeRead(), []

        with self.assertRaises(AssertionError) as caught:
            rbs.verdict_block(FakeReadout())
        self.assertIn("certified-proton-probably", str(caught.exception))


# =====================================================================
# the neutral host
# =====================================================================

class NeutralHostTest(unittest.TestCase):
    """The documented initial complex is NEUTRAL: closed, unrefined by any
    target, and carrying nothing the epic looks for."""

    def test_the_bare_host_is_a_closed_four_complex(self):
        host = rbs.build_neutral_host(0)
        cells = list(host.getTopSimplices())
        self.assertEqual(len(cells), 6)          # boundary of a 5-simplex
        for cell in cells:
            self.assertEqual(len(cell.getVertices()), 5)

    def test_refinement_grows_the_host_deterministically(self):
        a = rbs.build_neutral_host(SMALL)
        b = rbs.build_neutral_host(SMALL)
        self.assertEqual(len(a.getTopSimplices()), len(b.getTopSimplices()))
        self.assertGreater(len(a.getTopSimplices()), 6)

    def test_the_host_carries_no_boundary_block_and_no_target(self):
        host = rbs.build_neutral_host(SMALL)
        node = cob.MultiCobordism(host, [], [], [1], 1.0, SMALL_SEED)
        self.assertEqual(len(node.inputs), 0)
        self.assertEqual(len(node.outputs), 0)

    def test_the_metric_is_the_declared_mild_non_uniform_one(self):
        host = rbs.build_neutral_host(2)
        squared = sorted({round(abs(complex(edge.getLength()) ** 2), 9)
                          for edge in host.getEdgeList().toVector()})
        for value in squared:
            self.assertLessEqual(abs(value - 1.0), 0.0500001)


# =====================================================================
# the fast path, end to end
# =====================================================================

class FastPathTest(unittest.TestCase):
    """The one documented command, end to end."""

    @classmethod
    def setUpClass(cls):
        cls.document, cls.directory = shared_run()

    def test_the_document_round_trips_through_json(self):
        again = json.loads(json.dumps(self.document))
        self.assertEqual(again["verdict"]["verdict"],
                         self.document["verdict"]["verdict"])
        self.assertEqual(again["content_hashes"],
                         self.document["content_hashes"])

    def test_every_declared_block_is_present(self):
        for name in ("provenance", "config", "quantity_classes", "host",
                     "drive", "checkpoints", "checkpoint", "scan_checkpoint",
                     "raw_geometry", "edge_mode_data", "hierarchy",
                     "response_hierarchy", "fibers", "transports",
                     "covariance", "fock", "statistics", "particles",
                     "particles_resolution_scan", "spectral_dimension",
                     "exactness", "verdict", "certificates", "runtime",
                     "content_hashes", "reproducibility"):
            self.assertIn(name, self.document, name)

    def test_the_verdict_is_in_the_declared_vocabulary(self):
        self.assertIn(self.document["verdict"]["verdict"], rbs.VERDICTS)
        self.assertEqual(
            rbs.LIBRARY_VERDICTS[
                self.document["verdict"]["library_classification"]],
            self.document["verdict"]["verdict"])

    def test_the_run_persists_every_schema_three_checkpoint(self):
        self.assertEqual(len(self.document["checkpoints"]),
                         self.document["config"]["drive_steps"])
        for checkpoint in self.document["checkpoints"]:
            self.assertEqual(checkpoint["schema_version"],
                             cob.MultiCobordism.checkpoint_schema_version())
            self.assertEqual(checkpoint["mode"], "emergence")

    def test_the_emergence_firewall_is_recorded_and_intact(self):
        firewall = self.document["drive"]["firewall"]
        self.assertEqual(firewall["objective_terms"], [
            "regge_stationarity", "hodge_stationarity", "register_residual",
            "action_magnitude", "carried_state_energy"])
        self.assertEqual(firewall["refinement_indicators"], [
            "regge_stationarity_residual", "hodge_stationarity_residual",
            "curvature_concentration", "mesh_quality", "solver_error"])
        # Strict sub-mode: the ONE channel from the carried state is zero.
        self.assertEqual(firewall["carried_state_energy"], 0.0)
        self.assertEqual(firewall["carried_state_energy_weight"], 0.0)

    def test_the_refinement_rule_is_geometry_only_and_declared(self):
        for step in self.document["drive"]["steps"]:
            self.assertEqual(
                sorted(step["indicators"]),
                sorted(cob.MultiCobordism.refinement_indicator_names()))
            if step["refinement"]["trigger"] is not None:
                self.assertIn(step["refinement"]["trigger"],
                              cob.MultiCobordism.refinement_indicator_names())

    def test_disabling_refinement_commits_no_cell(self):
        document = rbs.run_simulation(small_config(refine=False),
                                      commit="testcommit")
        self.assertEqual(document["drive"]["refinement_events"], 0)
        for step in document["drive"]["steps"]:
            self.assertFalse(step["refinement"]["enabled"])
            self.assertEqual(step["refinement"]["cells_committed"], 0)

    def test_the_provenance_carries_seed_config_hash_and_commit(self):
        provenance = self.document["provenance"]
        self.assertEqual(provenance["seed"], SMALL_SEED)
        self.assertEqual(provenance["commit"], "testcommit")
        self.assertEqual(provenance["config_hash"],
                         rbs.config_hash_of(self.document["config"]))
        self.assertEqual(
            self.document["checkpoint"]["provenance"]["config_hash"],
            provenance["config_hash"])

    def test_the_driver_reads_agree_with_the_cpp_overlays(self):
        agreement = self.document["particles"]["checkpoint_agreement"]
        self.assertEqual(agreement["driver_quark_reads"],
                         agreement["checkpoint_quark_reads"])
        self.assertTrue(agreement["classifications_match"])
        self.assertTrue(agreement["failed_certificates_match"])

    def test_the_exactness_fixtures_are_all_exact_in_the_run(self):
        failed = [f["name"] for f in self.document["exactness"]
                  if not f["exact"]]
        self.assertEqual(failed, [])

    def test_the_cli_fast_path_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "run.json")
            result = subprocess.run(
                [sys.executable,
                 os.path.join(os.path.dirname(rbs.__file__),
                              "recursive_baryon_simulation.py"),
                 "run", "--size", str(SMALL), "--drive-steps", "1",
                 "--quiet", "--out", out],
                capture_output=True, text=True,
                env={**os.environ, "OMP_NUM_THREADS": "4"})
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            with open(out) as handle:
                document = json.load(handle)
            self.assertIn(document["verdict"]["verdict"], rbs.VERDICTS)


# =====================================================================
# unknown is null, never zero; every absence is named
# =====================================================================

class UnknownIsNullTest(unittest.TestCase):
    """The epic's own rule, enforced on the emitted document."""

    @classmethod
    def setUpClass(cls):
        cls.document, _ = shared_run()

    def test_finite_helper_returns_none_for_every_unknown(self):
        for value in (None, float("nan"), float("inf"), float("-inf"),
                      "not a number", object()):
            self.assertIsNone(rbs._finite(value))
        self.assertEqual(rbs._finite(0.0), 0.0)
        self.assertEqual(rbs._finite(-2.5), -2.5)

    def test_an_unemitted_sheaf_realization_reports_a_null_residual(self):
        realization = self.document["response_hierarchy"]["realization"]
        if not realization.get("emitted"):
            self.assertIsNone(realization["reconstruction_residual"])

    def test_every_absent_holonomy_channel_names_its_reason(self):
        transports = self.document["transports"]
        for channel in ("full", "determinant", "projective", "center",
                        "winding"):
            entry = transports[channel]
            if not entry.get("available"):
                self.assertTrue(entry.get("reason"),
                                f"{channel} is absent with no named reason")
                self.assertGreater(len(entry["reason"]), 20)

    def test_an_absent_fock_dag_names_its_reason_and_claims_nothing(self):
        fock = self.document["fock"]
        if not fock["present"]:
            self.assertTrue(fock["reason"])
            self.assertIsNone(fock["nodes"])
            self.assertIsNone(fock["discarded_norm"])

    def test_the_verdict_names_every_piece_of_missing_evidence(self):
        verdict = self.document["verdict"]
        self.assertTrue(verdict["missing_evidence"])
        for reason in verdict["missing_evidence"]:
            self.assertGreater(len(reason), 20)
        for name in verdict["failed_certificates"]:
            self.assertIn(name, rbs.BARYON_GATE_ORDER)

    def test_an_unsupplied_spin_read_is_null_and_never_zero(self):
        verdict = self.document["verdict"]
        if not verdict["sharp_spin"]:
            self.assertIsNone(verdict["total_j2_variance"])

    def test_the_emergent_spinor_carrier_absence_is_named(self):
        carrier = self.document["statistics"]["emergent_carrier"]
        if not carrier["supplied"]:
            self.assertIsNone(carrier["rows"])
            self.assertTrue(carrier["reason"])


# =====================================================================
# the schema round trip and version rejection
# =====================================================================

class SchemaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.document, cls.directory = shared_run()

    def test_a_replay_rejects_an_unknown_run_schema_version(self):
        broken = copy.deepcopy(self.document)
        broken["schema_version"] = rbs.RUN_SCHEMA_VERSION + 1
        with self.assertRaises(ValueError) as caught:
            rbs.replay_document(broken, directory=self.directory)
        self.assertIn("schema_version", str(caught.exception))

    def test_a_replay_rejects_a_missing_run_schema_version(self):
        broken = copy.deepcopy(self.document)
        broken.pop("schema_version")
        with self.assertRaises(ValueError):
            rbs.replay_document(broken, directory=self.directory)

    def test_the_checkpoint_reader_rejects_an_unknown_schema_version(self):
        checkpoint = json.loads(json.dumps(self.document["checkpoint"]))
        checkpoint["schema_version"] = 99
        with self.assertRaises(ValueError):
            cob.MultiCobordism.replay_checkpoint(json.dumps(checkpoint))

    def test_the_checkpoint_reader_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            cob.MultiCobordism.replay_checkpoint("{not json")

    def test_canonical_json_is_order_independent(self):
        self.assertEqual(rbs.canonical_json({"a": 1, "b": [1, 2]}),
                         rbs.canonical_json({"b": [1, 2], "a": 1}))
        self.assertNotEqual(rbs.content_hash({"a": 1}),
                            rbs.content_hash({"a": 2}))

    def test_canonical_json_refuses_a_non_finite_number(self):
        """A NaN in a hashed block would be a silent 'unknown reads as a
        value'; the hash refuses it instead."""
        with self.assertRaises(ValueError):
            rbs.canonical_json({"a": float("nan")})

    def test_every_hashed_block_is_present_and_hashed(self):
        for name in rbs.HASHED_BLOCKS:
            self.assertIn(name, self.document)
            self.assertIn(name, self.document["content_hashes"])
            self.assertEqual(self.document["content_hashes"][name],
                             rbs.content_hash(self.document[name]))


# =====================================================================
# replay
# =====================================================================

class ReplayTest(unittest.TestCase):
    """Cold-cache replay reproduces every verdict and content hash."""

    @classmethod
    def setUpClass(cls):
        cls.document, cls.directory = shared_run()
        cls.report = rbs.replay_document(cls.document,
                                         directory=cls.directory)

    def test_the_replay_verifies(self):
        self.assertTrue(self.report["verified"],
                        json.dumps({k: v for k, v in self.report.items()
                                    if k != "frames"})[:2000])

    def test_every_frame_reproduces_its_discrete_verdicts_exactly(self):
        self.assertEqual(self.report["frames_discrete_identical"],
                         self.report["frames_total"])
        for frame in self.report["frames"]:
            self.assertTrue(frame["discrete_verdicts_identical"])
            self.assertLessEqual(frame["worst_relative_difference"],
                                 self.report["tolerance"])

    def test_every_frame_replays_with_cold_caches(self):
        for frame in self.report["frames"]:
            self.assertTrue(frame["cold_caches"])

    def test_every_content_hash_matches(self):
        self.assertEqual(self.report["content_hashes_matched"],
                         len(self.report["content_hashes"]))

    def test_every_persisted_matrix_verifies_against_its_hash(self):
        self.assertTrue(self.report["sidecar"]["arrays"])
        for entry in self.report["sidecar"]["arrays"]:
            self.assertTrue(entry["verified"], entry.get("reason"))

    def test_a_corrupted_matrix_hash_is_caught_not_ignored(self):
        broken = copy.deepcopy(self.document)
        descriptors = rbs._matrix_descriptors(broken)
        self.assertTrue(descriptors)
        descriptors[0]["sha256"] = "0" * 64
        report = rbs.replay_document(broken, directory=self.directory)
        self.assertFalse(report["verified"])
        self.assertFalse(report["sidecar"]["arrays"][0]["verified"])
        self.assertIn("content hash mismatch",
                      report["sidecar"]["arrays"][0]["reason"])

    def test_a_tampered_verdict_is_caught_by_its_content_hash(self):
        broken = copy.deepcopy(self.document)
        broken["verdict"]["verdict"] = "certified proton"
        report = rbs.replay_document(broken, directory=self.directory)
        self.assertFalse(report["verified"])
        self.assertFalse(report["verdict"]["match"])
        mismatched = [h["block"] for h in report["content_hashes"]
                      if not h["match"]]
        self.assertIn("verdict", mismatched)

    def test_the_replayed_verdict_equals_the_stored_one(self):
        self.assertTrue(self.report["verdict"]["match"])
        self.assertEqual(self.report["verdict"]["replayed"],
                         self.document["verdict"]["verdict"])
        self.assertTrue(self.report["verdict"]["failed_certificates_match"])

    def test_the_exactness_fixtures_are_exact_on_both_paths(self):
        for fixture in self.report["exactness"]:
            self.assertTrue(fixture["both_exact"], fixture["name"])

    def test_the_excluded_blocks_are_reported_with_their_reasons(self):
        for frame in self.report["frames"]:
            self.assertEqual(sorted(frame["excluded_blocks"]),
                             sorted(rbs.REPLAY_EXCLUDED_BLOCKS))
            for name, entry in frame["excluded_blocks"].items():
                self.assertTrue(entry["reason"], name)

    def test_the_cli_replay_exits_zero_on_a_good_document(self):
        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(rbs.__file__),
                          "recursive_baryon_simulation.py"),
             "replay", "--from", _RUN_CACHE["path"], "--quiet"],
            capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])

    def test_the_worst_relative_difference_calls_a_discrete_mismatch_infinite(
            self):
        worst, where = rbs._worst_relative_difference({"a": "x"}, {"a": "y"})
        self.assertEqual(worst, float("inf"))
        self.assertTrue(where)
        worst, _ = rbs._worst_relative_difference([1, 2], [1, 2, 3])
        self.assertEqual(worst, float("inf"))
        worst, _ = rbs._worst_relative_difference(True, False)
        self.assertEqual(worst, float("inf"))
        worst, _ = rbs._worst_relative_difference(1.0, 1.0 + 1e-14)
        self.assertLess(worst, 1e-13)

    def test_the_rebuilt_complex_matches_the_persisted_raw_geometry(self):
        raw = self.document["raw_geometry"]
        rebuilt = rbs._spacetime_from_raw(raw)
        self.assertEqual(len(rebuilt.getTopSimplices()), raw["cell_count"])
        again = rbs.raw_geometry_block(rebuilt)
        self.assertEqual(again["cells"], raw["cells"])
        self.assertEqual(again["edges"], raw["edges"])


# =====================================================================
# the exactness fixtures, against independent references
# =====================================================================

class ExactnessFixtureTest(unittest.TestCase):
    """The set the ticket names, each against an analytic or dense
    reference computed independently HERE, not by the driver."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = {f["name"]: f for f in rbs.exactness_fixtures()}

    def test_every_named_fixture_is_present(self):
        for name in ("static_schur_path", "shifted_feshbach_pencil",
                     "static_schur_does_not_preserve_the_pencil",
                     "second_quantized_subset_sum",
                     "second_quantized_hopping", "triangle_anchor",
                     "center_branch", "closed_determinant_winding",
                     "berry_cancellation", "sharp_spin_half",
                     "generic_slater_is_not_a_sharp_spin"):
            self.assertIn(name, self.fixtures)

    def test_every_fixture_is_exact_at_machine_precision(self):
        for name, fixture in self.fixtures.items():
            with self.subTest(fixture=name):
                self.assertIsNotNone(fixture["residual"])
                self.assertTrue(fixture["exact"],
                                f"{name} residual {fixture['residual']} "
                                f"exceeded {fixture['tolerance']}")
                self.assertEqual(fixture["grade"], "exact")

    def test_static_schur_matches_a_dense_kron_reduction(self):
        operator = np.array([[3, -1, 0, 0], [-1, 3, -1, 0],
                             [0, -1, 3, -1], [0, 0, -1, 3]], dtype=complex)
        quotient = cob.RecursiveQuotient.overMatrix(
            operator.reshape(-1).tolist(), 4, [], [[0, 1, 2], [2, 3]],
            cob.RecursiveQuotient.Options())
        static = quotient.staticReduction()
        kept = len(static.coordinates)
        effective = np.array(static.effectiveOperator).reshape(kept, kept)
        interface = list(quotient.interfaceIndices)
        interior = [i for i in range(4) if i not in interface]
        dense = (operator[np.ix_(interface, interface)]
                 - operator[np.ix_(interface, interior)]
                 @ np.linalg.inv(operator[np.ix_(interior, interior)])
                 @ operator[np.ix_(interior, interface)])
        self.assertLess(float(np.abs(effective - dense).max()), 1e-13)
        self.assertLess(abs(static.solveResidual), 1e-13)

    def test_the_shifted_pencil_matches_a_dense_feshbach_reference(self):
        operator = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
                            dtype=complex)
        quotient = cob.RecursiveQuotient.overMatrix(
            operator.reshape(-1).tolist(), 3, [], [[0, 1], [1, 2]],
            cob.RecursiveQuotient.Options())
        interface = list(quotient.interfaceIndices)
        interior = [i for i in range(3) if i not in interface]
        for lam in (0.0, 0.25, 0.5, 0.9):
            read = quotient.feshbach(lam, lam - 0.05, lam + 0.05)
            kept = len(read.coordinates)
            response = np.array(read.response).reshape(kept, kept)
            dense = (operator[np.ix_(interface, interface)]
                     - lam * np.eye(len(interface))
                     - operator[np.ix_(interface, interior)]
                     @ np.linalg.inv(operator[np.ix_(interior, interior)]
                                     - lam * np.eye(len(interior)))
                     @ operator[np.ix_(interior, interface)])
            self.assertLess(float(np.abs(response - dense).max()), 1e-12,
                            f"lambda={lam}")

    def test_static_schur_is_visibly_not_the_shifted_response(self):
        detail = self.fixtures[
            "static_schur_does_not_preserve_the_pencil"]["detail"]
        self.assertGreater(detail["eigenvalue_separation"], 1e-6)

    def test_second_quantized_subset_sums_match_itertools(self):
        import itertools
        spectrum = [0.25 + 0j, 1.0 + 0j, 1.75 + 0j, 2.5 + 0j, 4.0 + 0j]
        key = (lambda z: (z.real, z.imag))
        for particles in (0, 1, 2, 3, 5):
            got = sorted((complex(v) for v in
                          cob.OccupationSpectra.subsetSums(spectrum,
                                                           particles)),
                         key=key)
            reference = sorted(
                (sum(c) if c else 0j
                 for c in itertools.combinations(spectrum, particles)),
                key=key)
            self.assertEqual(len(got), len(reference))
            for a, b in zip(got, reference):
                self.assertLess(abs(a - b), 1e-14)

    def test_the_hopping_block_matches_a_dense_assembly(self):
        block_a = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
        block_b = np.array([[0.5]], dtype=complex)
        coupling = np.array([[0.3 + 0.1j], [-0.4 + 0.2j]])
        flat = cob.OccupationSpectra.hoppingBlock(
            block_a.reshape(-1).tolist(), 2, block_b.reshape(-1).tolist(), 1,
            coupling.reshape(-1).tolist())
        assembled = np.array(flat).reshape(3, 3)
        dense = np.zeros((3, 3), dtype=complex)
        dense[:2, :2] = block_a
        dense[2:, 2:] = block_b
        dense[:2, 2:] = coupling
        dense[2:, :2] = coupling.conj().T
        self.assertLess(float(np.abs(assembled - dense).max()), 1e-15)
        self.assertLess(
            float(np.abs(np.linalg.eigvalsh(assembled)
                         - np.linalg.eigvalsh(dense)).max()), 1e-13)

    def test_a_literal_triangle_anchors_exactly(self):
        anchor = T.ColorAnchor([T.OrientedTriangle([0, 1, 2], [1, -1, 1])])
        profile = anchor.evaluate(np.eye(3, dtype=complex), np.ones(3))
        self.assertAlmostEqual(float(profile.score), 1.0, delta=1e-13)
        self.assertGreaterEqual(float(profile.score), 0.0)
        self.assertLessEqual(float(profile.score), 1.0 + 1e-13)

    def test_an_abstract_unanchored_rank_three_band_scores_zero(self):
        frame = np.zeros((6, 3), dtype=complex)
        frame[0, 0] = frame[1, 1] = frame[2, 2] = 1.0
        anchor = T.ColorAnchor([T.OrientedTriangle([3, 4, 5], [1, -1, 1])])
        profile = anchor.evaluate(frame, np.ones(6))
        self.assertLess(abs(float(profile.score)), 1e-14)

    def test_post_hoc_anchor_reweighting_is_refused(self):
        detail = self.fixtures["triangle_anchor"]["detail"]
        self.assertTrue(detail["post_hoc_reweighting_refused"])
        self.assertTrue(detail["empty_atlas_refused"])

    def test_the_three_center_branches_share_a_sector_and_shift_by_omega(self):
        detail = self.fixtures["center_branch"]["detail"]
        self.assertEqual(len(detail["center_sectors"]), 1)
        self.assertLess(detail["branch_trace_residual"], 1e-13)
        self.assertLess(detail["adjoint_branch_blindness"], 1e-13)
        self.assertGreater(detail["fundamental_center_spread"], 1e-6)

    def test_the_closed_determinant_winding_is_an_integer(self):
        detail = self.fixtures["closed_determinant_winding"]["detail"]
        self.assertEqual(detail["winding"], 1)
        self.assertEqual(detail["closure"], "closed-family")
        self.assertIsInstance(detail["winding"], int)

    def test_the_raw_berry_loop_is_not_a_sign_while_the_ratio_is(self):
        detail = self.fixtures["berry_cancellation"]["detail"]
        self.assertFalse(detail["raw_is_a_sign"])
        single = complex(*detail["single_exchange"])
        double = complex(*detail["double_exchange"])
        self.assertLess(abs(single + 1.0), 1e-13)
        self.assertLess(abs(double - 1.0), 1e-13)

    def test_the_sharp_and_generic_spin_fixtures_are_distinguished(self):
        self.assertTrue(self.fixtures["sharp_spin_half"]["exact"])
        self.assertTrue(
            self.fixtures["generic_slater_is_not_a_sharp_spin"]["exact"])

    def test_the_cli_fixtures_command_exits_zero(self):
        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(rbs.__file__),
                          "recursive_baryon_simulation.py"), "fixtures"],
            capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("static_schur_path", result.stdout)


# =====================================================================
# the campaign
# =====================================================================

class CampaignTest(unittest.TestCase):
    """A small three-size campaign: the aggregate must carry runtime,
    memory and cache statistics and must never drop a member."""

    @classmethod
    def setUpClass(cls):
        base = rbs.make_config(drive_steps=1)
        cls.report = rbs.run_campaign((4, 6, 8), (SMALL_SEED,), base)

    def test_at_least_three_sizes_are_reported(self):
        self.assertGreaterEqual(len(self.report["aggregate"]["by_size"]), 3)

    def test_no_member_is_silently_dropped(self):
        self.assertEqual(self.report["members_total"], 3)
        self.assertEqual(len(self.report["members"]), 3)
        self.assertEqual(self.report["members_ok"]
                         + len(self.report["members_failed"]),
                         self.report["members_total"])

    def test_every_member_reports_runtime_memory_and_cache_statistics(self):
        for member in self.report["members"]:
            if not member["ok"]:
                continue
            for field in ("wall_seconds", "drive_seconds", "analysis_seconds",
                          "rss_bytes", "peak_rss_bytes", "cache_hits",
                          "cache_misses", "cache_invalidations",
                          "cache_entries"):
                self.assertIsNotNone(member[field], field)

    def test_the_aggregate_carries_a_scaling_fit_or_says_why_not(self):
        for field, fit in self.report["aggregate"]["scaling"].items():
            self.assertIsNotNone(fit, field)
            if fit["available"]:
                self.assertIsNotNone(fit["exponent"])
                self.assertIsNotNone(fit["r_squared"])
            else:
                self.assertTrue(fit["reason"])

    def test_the_aggregate_verdicts_are_in_the_declared_vocabulary(self):
        for verdict in self.report["aggregate"]["verdicts"]:
            self.assertIn(verdict, rbs.VERDICTS)

    def test_the_campaign_grows_with_size(self):
        rows = self.report["aggregate"]["by_size"]
        cells = [row["cells"]["mean"] for row in rows]
        self.assertEqual(cells, sorted(cells))

    def test_a_two_point_fit_refuses_to_invent_an_uncertainty(self):
        self.assertIsNone(rbs.linear_fit([1.0, 2.0], [1.0, 2.0]))
        fit = rbs.linear_fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(fit["slope"], 2.0, places=12)


# =====================================================================
# relabeling covariance (design spec §21.2)
# =====================================================================

def _relabel(raw, permutation):
    """The same complex with every vertex id mapped through `permutation`."""
    relabelled = copy.deepcopy(raw)
    relabelled["cells"] = sorted([permutation[int(v)] for v in cell]
                                 for cell in raw["cells"])
    edges = {}
    for edge in raw["edges"]:
        a, b = permutation[int(edge["a"])], permutation[int(edge["b"])]
        edges[(min(a, b), max(a, b))] = edge["length"]
    relabelled["edges"] = [{"a": a, "b": b, "length": length}
                           for (a, b), length in sorted(edges.items())]
    return relabelled


class RelabelingTest(unittest.TestCase):
    """A random vertex relabeling changes no hierarchy support, no particle
    verdict, no transport verdict and no band read.

    #776 finding 6 measured that the canonical component HASH is NOT fully
    label-free, so nothing here is keyed on that hash — the SUPPORTS are the
    label-free object, and they must map through the permutation exactly.
    """

    @classmethod
    def setUpClass(cls):
        document, _ = shared_run()
        cls.config = document["config"]
        raw = document["raw_geometry"]
        vertices = sorted({int(v) for cell in raw["cells"] for v in cell})
        rng = np.random.default_rng(20260823)
        shuffled = list(rng.permutation(vertices))
        cls.permutation = {int(v): int(shuffled[i])
                           for i, v in enumerate(vertices)}
        cls.base = rbs.RecursiveReadout(
            rbs._spacetime_from_raw(raw), cls.config, cls.config["seed"])
        cls.relabelled = rbs.RecursiveReadout(
            rbs._spacetime_from_raw(_relabel(raw, cls.permutation)),
            cls.config, cls.config["seed"])

    def _supports(self, readout):
        return sorted(tuple(sorted(int(v) for v in c.support))
                      for c in readout.components)

    def test_the_permutation_is_a_genuine_relabeling(self):
        self.assertNotEqual(sorted(self.permutation.items()),
                            [(k, k) for k in sorted(self.permutation)])
        self.assertEqual(sorted(self.permutation.values()),
                         sorted(self.permutation))

    def test_the_discovered_partition_maps_through_exactly(self):
        mapped = sorted(tuple(sorted(self.permutation[v] for v in support))
                        for support in self._supports(self.base))
        self.assertEqual(mapped, self._supports(self.relabelled))

    def test_the_band_reads_are_identical(self):
        base = sorted((f.degree(), f.rank(), f.accepted())
                      for read in self.base.band_reads for f in read.fibers)
        again = sorted((f.degree(), f.rank(), f.accepted())
                       for read in self.relabelled.band_reads
                       for f in read.fibers)
        self.assertEqual(base, again)

    def test_the_transport_verdicts_are_identical(self):
        base = sorted((t["read"].accepted, t["read"].rank,
                       t["read"].rejectionReason)
                      for t in self.base.transports)
        again = sorted((t["read"].accepted, t["read"].rank,
                        t["read"].rejectionReason)
                       for t in self.relabelled.transports)
        self.assertEqual(base, again)

    def test_the_quark_verdicts_are_identical(self):
        base = sorted((q.classification, tuple(q.failedCertificates),
                       q.colorRank) for q in self.base.quarks)
        again = sorted((q.classification, tuple(q.failedCertificates),
                        q.colorRank) for q in self.relabelled.quarks)
        self.assertEqual(base, again)

    def test_the_baryon_verdict_is_identical(self):
        base = rbs.verdict_block(self.base)
        again = rbs.verdict_block(self.relabelled)
        self.assertEqual(base["verdict"], again["verdict"])
        self.assertEqual(base["library_classification"],
                         again["library_classification"])
        self.assertEqual(base["failed_certificates"],
                         again["failed_certificates"])

    def test_the_closed_holonomy_center_and_exchange_reads_are_identical(self):
        """The gauge and statistics channels — whether present or absent —
        must read the same under a relabeling, absence reasons included."""
        base = rbs.transports_block(self.base)
        again = rbs.transports_block(self.relabelled)
        for channel in ("full", "determinant", "projective", "center",
                        "winding"):
            self.assertEqual(base[channel].get("available"),
                             again[channel].get("available"), channel)
            self.assertEqual(base[channel].get("reason"),
                             again[channel].get("reason"), channel)
        self.assertEqual(rbs.statistics_block(self.base)["declared_carrier"],
                         rbs.statistics_block(
                             self.relabelled)["declared_carrier"])

    def test_the_continuous_aggregates_agree_to_double_round_off(self):
        base = rbs.hierarchy_block(self.base, {"hierarchy": []})
        again = rbs.hierarchy_block(self.relabelled, {"hierarchy": []})
        self.assertAlmostEqual(base["analysis_slice"]["q"],
                               again["analysis_slice"]["q"], delta=1e-12)
        self.assertEqual(base["analysis_slice"]["levels"],
                         again["analysis_slice"]["levels"])


# =====================================================================
# the geometry-only refinement rule
# =====================================================================

class RefinementTest(unittest.TestCase):
    """The refinement rule is #776's, unchanged: STATIC over five geometric
    indicators, so it cannot reach a certificate."""

    def test_the_declared_thresholds_are_geometry_only(self):
        for name in rbs.DECLARED_REFINEMENT_THRESHOLDS:
            self.assertIn(
                name, cob.MultiCobordism.refinement_indicator_names())

    def test_a_none_threshold_never_fires_and_a_number_can(self):
        indicators = cob.MultiCobordism.RefinementIndicators()
        indicators.curvature_concentration = 100.0
        indicators.mesh_quality = 0.5
        never = cob.MultiCobordism.RefinementIndicators()
        never.regge_stationarity_residual = float("inf")
        never.hodge_stationarity_residual = float("inf")
        never.curvature_concentration = float("inf")
        never.solver_error = float("inf")
        never.mesh_quality = 0.0
        self.assertFalse(
            cob.MultiCobordism.refinement_decision_of(indicators,
                                                      never).refine)
        declared = cob.MultiCobordism.RefinementIndicators()
        declared.regge_stationarity_residual = float("inf")
        declared.hodge_stationarity_residual = float("inf")
        declared.curvature_concentration = 4.0
        declared.solver_error = float("inf")
        declared.mesh_quality = 0.1
        decision = cob.MultiCobordism.refinement_decision_of(indicators,
                                                             declared)
        self.assertTrue(decision.refine)
        self.assertEqual(decision.trigger, "curvature_concentration")

    def test_the_declared_thresholds_are_applied_to_the_node(self):
        node = cob.MultiCobordism(rbs.build_neutral_host(SMALL), [], [],
                                  [1], 1.0, SMALL_SEED)
        rbs._apply_refinement_thresholds(node, small_config())
        thresholds = node.refinement_thresholds
        self.assertEqual(thresholds.curvature_concentration, 4.0)
        self.assertEqual(thresholds.mesh_quality, 0.1)
        self.assertEqual(thresholds.regge_stationarity_residual,
                         float("inf"))
        self.assertEqual(thresholds.solver_error, float("inf"))

    def test_the_refinement_mechanism_commits_through_the_gated_surgery(self):
        """A THRESHOLD CHOSEN HERE to make the decision fire — a test of the
        MECHANISM, never a run configuration. The declared run thresholds
        stay untouched."""
        node = cob.MultiCobordism(rbs.build_neutral_host(SMALL), [], [],
                                  [1], 1.0, SMALL_SEED)
        node.set_objective_mode(cob.CobordismObjectiveMode.JointStationarity)
        firing = cob.MultiCobordism.RefinementIndicators()
        firing.regge_stationarity_residual = float("inf")
        firing.hodge_stationarity_residual = float("inf")
        firing.curvature_concentration = float("inf")
        firing.solver_error = float("inf")
        firing.mesh_quality = 1.0          # a LOWER bound nothing can meet
        node.set_refinement_thresholds(firing)
        decision = node.refinement_decision()
        self.assertTrue(decision.refine)
        self.assertEqual(decision.trigger, "mesh_quality")
        before = len(node.st.getTopSimplices())
        committed = node.refine_geometry(1)
        self.assertGreaterEqual(committed, 0)
        if committed:
            self.assertGreater(len(node.st.getTopSimplices()), before)

    def test_the_run_records_the_thresholds_that_never_fire(self):
        document, _ = shared_run()
        never = document["drive"]["refinement_never_fires"]
        self.assertEqual(
            sorted(never),
            sorted(name for name, value
                   in rbs.DECLARED_REFINEMENT_THRESHOLDS.items()
                   if value is None))


# =====================================================================
# the animation reads the same checkpoint data as the headless path
# =====================================================================

class AnimationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        cls.document, cls.directory = shared_run()

    def test_the_overlay_renders_headlessly_from_the_run_document(self):
        path = os.path.join(self.directory, "overlay.png")
        rbs.render_overlay(self.document, path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 20000)

    def test_the_animation_and_headless_paths_read_the_same_data(self):
        """The overlay is a pure function of the persisted document — the
        SAME object the headless path emits and the replay verifies."""
        reloaded = json.loads(json.dumps(self.document))
        self.assertEqual(rbs.content_hashes_of(reloaded),
                         self.document["content_hashes"])
        layout_a = rbs._layout_from_raw(self.document["raw_geometry"])
        layout_b = rbs._layout_from_raw(reloaded["raw_geometry"])
        self.assertEqual(sorted(layout_a), sorted(layout_b))
        for key in layout_a:
            self.assertLess(float(np.abs(layout_a[key] - layout_b[key]).max()),
                            1e-12)

    def test_the_overlay_renders_from_a_checkpoint_alone(self):
        """A frame other than the last is drawn from its own checkpoint's
        raw complex, so every persisted frame is renderable."""
        path = os.path.join(self.directory, "overlay_frame0.png")
        rbs.render_overlay(self.document, path, frame=0)
        self.assertTrue(os.path.exists(path))

    def test_the_layout_is_not_offered_as_a_spacetime_coordinate(self):
        self.assertIn("not a spacetime coordinate",
                      rbs._layout_from_raw.__doc__.lower().replace(
                          "coordinate system", "coordinate system"))

    def test_absent_panels_say_absent_and_name_the_reason(self):
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots()
        rbs._absent(axis, "a channel", "a specific named reason for absence")
        texts = [t.get_text() for t in axis.texts]
        self.assertIn("ABSENT", texts)
        self.assertTrue(any("named reason" in t for t in texts))
        plt.close(figure)

    def test_a_single_frame_document_falls_back_to_a_still(self):
        path = os.path.join(self.directory, "single.gif")
        produced = rbs.render_animation(self.document, path)
        self.assertEqual(len(produced), 1)
        self.assertTrue(os.path.exists(path))

    def test_a_multi_frame_document_renders_every_frame(self):
        """Two frames of the SAME persisted document — the animation walks
        the checkpoints, so a longer drive animates without any extra data
        source."""
        document = copy.deepcopy(self.document)
        document["checkpoints"].append(
            copy.deepcopy(document["checkpoints"][0]))
        path = os.path.join(self.directory, "multi.gif")
        produced = rbs.render_animation(document, path)
        self.assertEqual(produced[0], path)
        self.assertEqual(len(produced), 1 + len(document["checkpoints"]))
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 20000)

    def test_the_cli_animate_command_exits_zero(self):
        out = os.path.join(self.directory, "cli_overlay.png")
        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(rbs.__file__),
                          "recursive_baryon_simulation.py"),
             "animate", "--from", _RUN_CACHE["path"], "--out", out],
            capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "4", "MPLBACKEND": "Agg"})
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertTrue(os.path.exists(out))


# =====================================================================
# the quantity classification and the CLI help
# =====================================================================

class DocumentationTest(unittest.TestCase):

    def test_the_quantity_classes_cover_the_three_declared_kinds(self):
        self.assertEqual(sorted(rbs.QUANTITY_CLASSES),
                         ["certified_numerical", "exact", "heuristic"])
        for kind, entries in rbs.QUANTITY_CLASSES.items():
            self.assertTrue(entries, kind)

    def test_the_help_epilog_documents_exact_certified_and_heuristic(self):
        for token in ("EXACT", "CERTIFIED NUMERICAL", "HEURISTIC"):
            self.assertIn(token, rbs._EPILOG)
        for verdict in rbs.VERDICTS:
            self.assertIn(verdict, rbs._EPILOG)

    def test_the_run_document_carries_the_quantity_classification(self):
        document, _ = shared_run()
        self.assertEqual(document["quantity_classes"], rbs.QUANTITY_CLASSES)

    def test_the_cli_help_exits_zero_and_prints_the_vocabulary(self):
        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(rbs.__file__),
                          "recursive_baryon_simulation.py"), "run", "--help"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("quasi-free sharp-spin obstruction", result.stdout)
        self.assertIn("HEURISTIC", result.stdout)


# =====================================================================
# the existing examples keep their behaviour
# =====================================================================

class ExistingExamplesTest(unittest.TestCase):
    """#778 must not disturb what #776 and #777 shipped."""

    def test_the_multiscale_driver_still_imports_and_declares_its_sizes(self):
        import multiscale_validation as mv
        self.assertEqual(len(mv.DECLARED_SIZES_FULL), 5)
        self.assertEqual(mv.SCHEMA_VERSION, 1)

    def test_the_two_drivers_build_the_same_neutral_host(self):
        """#778's host is #777's, so the two studies are comparable."""
        import multiscale_validation as mv
        mine = rbs.build_neutral_host(SMALL, rbs.DECLARED_HOST_SEED)
        theirs = mv.build_host(SMALL, mv.DECLARED_HOST_SEED)
        self.assertEqual(len(mine.getTopSimplices()),
                         len(theirs.getTopSimplices()))
        self.assertEqual(
            rbs.raw_geometry_block(mine)["cells"],
            rbs.raw_geometry_block(theirs)["cells"])

    def test_the_proton_animation_overlay_is_still_opt_in(self):
        import proton_animation
        self.assertTrue(hasattr(proton_animation,
                                "_select_recursive_analysis"))
        node = cob.MultiCobordism(rbs.build_neutral_host(2), [], [], [1],
                                  1.0, SMALL_SEED)
        # Unselected, the overlay never runs and writes no checkpoint.
        self.assertFalse(node.analysis_config.enabled)
        self.assertEqual(node.checkpoint_json, "")


# =====================================================================
# the sidecar
# =====================================================================

class SidecarTest(unittest.TestCase):
    """The versioned binary sidecar (design spec §20)."""

    def test_a_small_matrix_is_inlined_and_a_large_one_goes_to_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "s.npz")
            sidecar = rbs.MatrixSidecar(path)
            small = sidecar.store("small", np.eye(2, dtype=complex))
            big = sidecar.store("big", np.eye(16, dtype=complex))
            self.assertEqual(small["storage"], "inline")
            self.assertEqual(big["storage"], "sidecar")
            descriptor = sidecar.write()
            self.assertEqual(descriptor["arrays"], ["big"])
            self.assertTrue(os.path.exists(path))
            for entry in (small, big):
                matrix, ok, reason = rbs.MatrixSidecar.load(entry, directory)
                self.assertTrue(ok, reason)
                self.assertEqual(matrix.shape, tuple(entry["shape"]))

    def test_a_missing_sidecar_file_is_named_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = rbs.MatrixSidecar(os.path.join(directory, "s.npz"))
            descriptor = sidecar.store("big", np.eye(16, dtype=complex))
            matrix, ok, reason = rbs.MatrixSidecar.load(descriptor, directory)
            self.assertIsNone(matrix)
            self.assertFalse(ok)
            self.assertIn("missing", reason)

    def test_a_run_with_no_large_matrix_writes_no_sidecar_file(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = rbs.MatrixSidecar(os.path.join(directory, "s.npz"))
            sidecar.store("small", np.eye(2, dtype=complex))
            self.assertIsNone(sidecar.write())
            self.assertFalse(os.path.exists(os.path.join(directory, "s.npz")))


# =====================================================================
# the recursive readout mirrors the overlay's assembly rule
# =====================================================================

class ReadoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.document, _ = shared_run()

    def test_the_candidate_band_is_the_first_accepted_one(self):
        for read in self.document["fibers"]["reads"]:
            candidates = [b for b in read["bands"] if b.get("candidate")]
            self.assertLessEqual(len(candidates), 1)
            accepted = [b for b in read["bands"] if b["accepted"]]
            if accepted:
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["index"], accepted[0]["index"])

    def test_every_derived_transport_links_two_distinct_band_reads(self):
        for entry in self.document["transports"]["detail"]:
            self.assertNotEqual(entry["from"], entry["to"])

    def test_a_rejected_transport_names_its_reason(self):
        for entry in self.document["transports"]["detail"]:
            if not entry["accepted"]:
                self.assertTrue(entry["rejection_reason"])

    def test_the_covariance_purity_certificate_is_measured_not_assumed(self):
        covariance = self.document["covariance"]
        for state in covariance["states"]:
            if state["available"]:
                self.assertIsNotNone(state["purity_defect"])
                self.assertLess(state["purity_defect"], 1e-9)
            else:
                self.assertTrue(state["reason"])

    def test_the_vacuum_embedding_preserves_every_amplitude(self):
        worst = self.document["covariance"]["vacuum_embedding_defect_max"]
        if worst is not None:
            self.assertLess(worst, 1e-12)

    def test_the_response_network_covers_the_reduced_operator(self):
        network = self.document["response_hierarchy"]["response_network"]
        if network.get("coverage_residual") is not None:
            self.assertLess(network["coverage_residual"], 1e-9)
            self.assertEqual(len(network["stalk_dimensions"]),
                             len(network["stalk_coordinates"]))

    def test_the_certificates_block_lists_every_failure_by_name(self):
        certificates = self.document["certificates"]
        self.assertEqual(
            certificates["held"] + len(certificates["failed"]),
            certificates["total"])
        self.assertEqual(
            certificates["held"] + certificates["refused"]
            + certificates["failed_count"], certificates["total"])
        for entry in certificates["entries"]:
            self.assertIn(entry["status"], ("holds", "refused", "failed"))
            if entry["status"] == "refused":
                self.assertTrue(entry["reason"], entry["name"])
            if not entry["holds"]:
                self.assertTrue(entry["reason"] or entry["residual"]
                                is not None or entry["name"].startswith(
                                    ("baryon:", "holonomy-")))

    def test_a_correct_domain_refusal_is_not_reported_as_a_failure(self):
        """AMLS on a non-normal operator and an unemitted sheaf realization
        are REFUSALS with named reasons, not certificate failures."""
        named = {e["name"]: e
                 for e in self.document["certificates"]["entries"]}
        amls = named["amls-craig-bampton"]
        if not amls["holds"]:
            self.assertEqual(amls["status"], "refused")
            self.assertIn("non-normal", amls["reason"])
        sheaf = named.get("sheaf-realization")
        if sheaf is not None and not sheaf["holds"]:
            self.assertEqual(sheaf["status"], "refused")

    def test_labeled_fiber_sum_gram_regime_is_classified_not_averaged(self):
        for entry in self.document["response_hierarchy"]["labeled_fiber_sums"]:
            if entry.get("gram_defect") is None:
                continue
            self.assertIn(entry["gram_defect_regime"],
                          ("isometric", "signature_flipped", "intermediate"))


if __name__ == "__main__":
    unittest.main()
