# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The observable measurement layer's framework (#583): Register validation,
requires/skip logic, the gate harness and its self-test, record shape.

Fast — everything runs on the composite_spin fixtures (74–105 top cells) and
hand-built complexes. The adapters' migration equivalence lives in
test_observe_adapters.py; the ProtonIngredients construction in
test_observe_proton_ingredients.py.
"""
import itertools
import json
import math
import os
import unittest
import warnings

import tessera
from tessera.observe import (
    GEOMETRY_SCHEMA,
    Battery,
    Observable,
    Register,
    battery,
    build_complex,
    ensure_jsonable,
    load_geometry_dump,
    measure_all,
    rebuild_spacetime,
    register_holes,
    report_delta,
    split_complex,
    verify_rebuild,
    write_geometry_dump,
)
from tessera.observe.pair_loop_flavor import build_spacetime, load_fixture

cob = tessera.cobordism


def _fixture(name="synthetic_b3_3.json"):
    meta, cells, edges = load_fixture(name)
    return meta, cells, edges


def _quiet_register(st, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Register(st, **kwargs)


class RegisterValidationTest(unittest.TestCase):
    """Hole selection is validated at ONE entry point: deficit raises,
    surplus warns NAMING the dropped holes; the census fields are recorded."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells, cls.edges = _fixture()
        # The stored fixture holes (removed top cells) let us FILL holes to
        # produce genuine deficit / divergence specimens.
        cls.holes_meta = [list(h) for h in cls.meta["holes"]]

    def test_surplus_warns_naming_dropped_hole_and_records_census(self):
        # synthetic_b3_3 stores FOUR emergent holes with b3 = 3 — the ticket's
        # 4-hole surplus fixture.
        st = build_spacetime(self.cells, self.edges)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg = Register(st, count=3)
        messages = [str(w.message) for w in caught
                    if "register selection" in str(w.message)]
        self.assertEqual(len(messages), 1)
        self.assertIn("4 emergent holes", messages[0])
        self.assertIn("(1, 4, 5, 6, 12)", messages[0])  # the dropped hole, NAMED
        self.assertEqual(len(reg.holes), 3)
        self.assertEqual(reg.holes_total, 4)
        self.assertEqual(reg.dropped, [(1, 4, 5, 6, 12)])
        self.assertEqual(reg.b3, 3)
        self.assertEqual(reg.summary()["holes_used"], 3)
        self.assertEqual(reg.summary()["holes_total"], 4)
        self.assertEqual(reg.summary()["b3"], 3)

    def test_deficit_raises_with_clear_message_on_two_hole_specimen(self):
        # Fill two of the four stored holes: a genuine 2-hole specimen.
        st = build_spacetime(self.cells + self.holes_meta[2:], self.edges)
        self.assertEqual(
            len(cob.MultiCobordism.emergent_holes(st, 3)), 2)
        with self.assertRaises(ValueError) as ctx:
            Register(st, count=3)
        self.assertIn("need >= 3 register holes", str(ctx.exception))
        self.assertIn("found 2", str(ctx.exception))

    def test_register_holes_returns_selection_and_dropped(self):
        st = build_spacetime(self.cells, self.edges)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holes, dropped = register_holes(st, 3)
        self.assertEqual(len(holes), 3)
        self.assertEqual(dropped, [(1, 4, 5, 6, 12)])

    def test_divergence_flag_on_holes_3_b3_2_specimen(self):
        # Fill ONE stored hole: 3 emergent holes remain but b3 drops to 2 —
        # exactly the campaign's holes-vs-b3 divergence.
        st = build_spacetime(self.cells + self.holes_meta[3:], self.edges)
        reg = Register(st, count=3)
        self.assertEqual(reg.holes_total, 3)
        self.assertEqual(reg.b3, 2)
        self.assertTrue(reg.holes_vs_b3_divergent)
        self.assertTrue(reg.summary()["holes_vs_b3_divergent"])

    def test_divergence_flag_false_on_matching_specimen(self):
        # converged_b3_3 carries exactly 3 emergent holes with b3 = 3.
        meta, cells, edges = _fixture("converged_b3_3.json")
        reg = Register(build_spacetime(cells, edges), count=3)
        self.assertEqual((reg.holes_total, reg.b3), (3, 3))
        self.assertFalse(reg.holes_vs_b3_divergent)

    def test_supplied_holes_validated_with_same_semantics(self):
        st = build_spacetime(self.cells, self.edges)
        all_holes = [tuple(h) for h in cob.MultiCobordism.emergent_holes(st, 3)]
        with self.assertRaises(ValueError):
            Register(st, holes=all_holes[:2], count=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg = Register(st, holes=all_holes, count=3)
        self.assertTrue(any("dropping" in str(w.message) for w in caught))
        self.assertEqual(len(reg.holes), 3)

    def test_eps_are_unit_signs_from_the_one_orientation_convention(self):
        st = build_spacetime(self.cells, self.edges)
        reg = _quiet_register(st, count=3)
        self.assertEqual(len(reg.eps), 3)
        self.assertTrue(all(s in (-1, 1) for s in reg.eps))

    def test_build_complex_fails_loudly_on_missing_edge_length(self):
        edges = dict(self.edges)
        edges.pop(next(iter(edges)))
        with self.assertRaises(KeyError):
            build_complex(self.cells, edges, dimensions=4)


class RegisterGateTransformsTest(unittest.TestCase):
    """The GAUGE/RELABEL transforms the gate harness runs on."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells, cls.edges = _fixture()
        cls.st = build_spacetime(cls.cells, cls.edges)
        cls.reg = _quiet_register(cls.st, count=3)

    def test_gauged_rotates_target_and_shares_the_complex(self):
        gauged = self.reg.gauged(0.25)
        self.assertIs(gauged.st, self.reg.st)
        self.assertEqual(gauged.holes, self.reg.holes)
        ratio = gauged.target[0] / self.reg.target[0]
        self.assertAlmostEqual(abs(ratio), 1.0, places=12)
        self.assertNotEqual(gauged.target, self.reg.target)
        # caches are shared (the gauge knob only rotates the target)
        self.assertIs(gauged.es, self.reg.es)

    def test_relabeled_matches_holes_by_permuted_vertex_set(self):
        reg2, perm = self.reg.relabeled(seed=5)
        self.assertEqual(len(reg2.holes), 3)
        for original, image in zip(self.reg.holes, reg2.holes):
            self.assertEqual({perm[v] for v in original}, set(image))
        # census invariants carry over
        self.assertEqual(reg2.holes_total, self.reg.holes_total)
        self.assertEqual(reg2.b3, self.reg.b3)
        self.assertEqual(reg2.dimensions, self.reg.dimensions)
        # the relabeled complex is a genuine rebuild
        self.assertIsNot(reg2.st, self.reg.st)


class SkipLogicTest(unittest.TestCase):
    """requires -> skipped-with-reason, never a crash."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells, cls.edges = _fixture()
        cls.st = build_spacetime(cls.cells, cls.edges)
        cls.reg = _quiet_register(cls.st, count=3)

    def test_min_holes_skip_reason(self):
        class NeedsFive(Observable):
            name = "needs_five"
            requires = {"min_holes": 5}

            def measure(self, register, provenance=None):
                return {}

        self.assertEqual(NeedsFive().skip_reason(self.reg),
                         "holes=3 < min_holes=5")

    def test_needs_provenance_skip_reasons(self):
        class NeedsBlocks(Observable):
            name = "needs_blocks"
            requires = {"needs_provenance": ("blocks",)}

            def measure(self, register, provenance=None):
                return {}

        obs = NeedsBlocks()
        self.assertEqual(obs.skip_reason(self.reg, None), "no_provenance")
        self.assertEqual(obs.skip_reason(self.reg, {"diquark_pair": [0, 1]}),
                         "provenance_missing:blocks")
        self.assertIsNone(obs.skip_reason(self.reg, {"blocks": []}))

    def test_needs_causal_content_skips_all_spacelike_specimen(self):
        class NeedsCausal(Observable):
            name = "needs_causal"
            requires = {"needs_causal_content": True}

            def measure(self, register, provenance=None):
                return {}

        # The fixture is all-spacelike: at initialization no time has passed,
        # causal structure may only emerge — so this honestly skips.
        self.assertFalse(self.reg.causal_content)
        self.assertEqual(NeedsCausal().skip_reason(self.reg),
                         "no_causal_content")

    def test_dimensions_requirement(self):
        cells3 = [list(c) for c in itertools.combinations(range(5), 4)]
        st3 = tessera.Spacetime.fromCells(3, cells3, 1.0, 0.0)
        reg3 = Register(st3, count=0)

        class Needs4D(Observable):
            name = "needs_4d"
            requires = {"dimensions": 4}

            def measure(self, register, provenance=None):
                return {}

        self.assertEqual(Needs4D().skip_reason(reg3), "dimensions=3 != 4")

    def test_battery_reports_skips_with_reason_per_observable(self):
        # A sub-3-hole register: the default battery skips the register-only
        # observables WITH REASONS and still measures the rest.
        holes_meta = [list(h) for h in self.meta["holes"]]
        st = build_spacetime(self.cells + holes_meta[2:], self.edges)
        reg = Register(st, count=2)
        record = measure_all(reg, provenance=None, gates=False)
        status = {k: v["status"] for k, v in record["observables"].items()}
        self.assertEqual(status["pair_loop_flavor"],
                         "skipped(holes=2 < min_holes=3)")
        self.assertEqual(status["block_residuals"], "skipped(no_provenance)")
        self.assertEqual(status["singlet_diagnostic"], "measured")
        self.assertEqual(status["mass_radius"], "measured")
        self.assertEqual(record["observables"]["pair_loop_flavor"]["reason"],
                         "holes=2 < min_holes=3")


class ReportDeltaSelfTest(unittest.TestCase):
    """The gate metric flags perturbed channels — the self-test pattern: a
    passing gate can never be a comparison that silently skipped a leaf."""

    BASE = {"a": 1.0, "nested": {"xs": [0.5, 0.25], "flag": True},
            "label": "ok", "maybe": None, "nan": float("nan")}

    def test_identical_records_have_zero_delta(self):
        self.assertEqual(report_delta(self.BASE, json.loads(
            json.dumps(self.BASE))), 0.0)

    def test_perturbed_deep_leaf_is_flagged(self):
        perturbed = json.loads(json.dumps(self.BASE))
        perturbed["nested"]["xs"][1] += 1e-3
        self.assertAlmostEqual(report_delta(self.BASE, perturbed), 1e-3,
                               places=12)

    def test_flipped_bool_is_flagged(self):
        perturbed = json.loads(json.dumps(self.BASE))
        perturbed["nested"]["flag"] = False
        self.assertEqual(report_delta(self.BASE, perturbed), 1.0)

    def test_changed_string_and_none_are_flagged_inf(self):
        perturbed = json.loads(json.dumps(self.BASE))
        perturbed["label"] = "changed"
        self.assertEqual(report_delta(self.BASE, perturbed), float("inf"))
        perturbed = json.loads(json.dumps(self.BASE))
        perturbed["maybe"] = 0.0
        self.assertEqual(report_delta(self.BASE, perturbed), float("inf"))

    def test_shape_mismatches(self):
        with self.assertRaises(KeyError):
            report_delta({"a": 1.0}, {"b": 1.0})
        self.assertEqual(report_delta([1.0, 2.0], [1.0]), float("inf"))

    def test_nan_agrees_with_nan_but_flags_against_a_number(self):
        self.assertEqual(report_delta(float("nan"), float("nan")), 0.0)
        # an appeared/vanished reading is a FLAGGED channel, never a silent
        # NaN poisoning the max
        self.assertEqual(report_delta(float("nan"), 1.0), float("inf"))

    def test_gate_harness_flags_a_label_sensitive_observable(self):
        # An observable that (wrongly) reports a labeling-dependent quantity:
        # the RELABEL gate must flag it.
        class LabelLeak(Observable):
            name = "label_leak"
            gate_tol = 1e-9

            def measure(self, register, provenance=None):
                return {"vertex_id_sum": float(sum(
                    v for hole in register.holes for v in hole))}

        meta, cells, edges = _fixture()
        reg = _quiet_register(build_spacetime(cells, edges), count=3)
        entry = LabelLeak().gated_measure(reg)
        self.assertFalse(entry["gates"]["relabel_ok"])
        self.assertGreater(entry["gates"]["relabel_delta"], 1.0)

    def test_gate_harness_flags_a_gauge_sensitive_observable(self):
        # An observable that (wrongly) reports the raw target phase: the
        # GAUGE gate must flag it.
        class GaugeLeak(Observable):
            name = "gauge_leak"
            gate_tol = 1e-9

            def measure(self, register, provenance=None):
                return split_complex("raw_target0", register.target[0])

        meta, cells, edges = _fixture()
        reg = _quiet_register(build_spacetime(cells, edges), count=3)
        entry = GaugeLeak().gated_measure(reg)
        self.assertFalse(entry["gates"]["gauge_ok"])
        self.assertGreater(entry["gates"]["gauge_delta"], 0.1)


class RecordShapeTest(unittest.TestCase):
    """Records are JSON-able; complex values travel as explicit re/im pairs."""

    def test_split_complex_scalar_and_sequence(self):
        self.assertEqual(split_complex("z", 1 + 2j),
                         {"z_re": 1.0, "z_im": 2.0})
        pairs = split_complex("w", [1 + 2j, 3 - 4j])
        self.assertEqual(pairs, {"w_re": [1.0, 3.0], "w_im": [2.0, -4.0]})
        json.dumps(pairs)

    def test_ensure_jsonable_rejects_raw_complex(self):
        with self.assertRaises(TypeError):
            ensure_jsonable({"z": 1 + 2j})

    def test_battery_record_is_json_serializable(self):
        meta, cells, edges = _fixture()
        reg = _quiet_register(build_spacetime(cells, edges), count=3)
        record = measure_all(reg, provenance={"diquark_pair": [0, 2]})
        text = json.dumps(record)  # includes complex-as-re/im channels
        self.assertIn("w_re", text)
        self.assertIn("w_im", text)
        round_tripped = json.loads(text)
        self.assertEqual(sorted(round_tripped["observables"]),
                         ["block_residuals", "mass_radius",
                          "pair_loop_flavor", "singlet_diagnostic"])

    def test_battery_registry_semantics(self):
        class One(Observable):
            name = "one"

            def measure(self, register, provenance=None):
                return {}

        b = Battery([One()])
        with self.assertRaises(ValueError):
            b.register(One())  # duplicate name
        self.assertEqual([o.name for o in b.observables], ["one"])
        self.assertEqual([o.name for o in battery.observables],
                         ["singlet_diagnostic", "block_residuals",
                          "mass_radius", "pair_loop_flavor"])


class GeometryDumpTest(unittest.TestCase):
    """The faithful record: schema-1 dumps round-trip byte-identically and a
    tampered rebuild is caught."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells, cls.edges = _fixture()
        cls.st = build_spacetime(cls.cells, cls.edges)

    def test_round_trip_is_byte_identical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "a.json")
            p2 = os.path.join(tmp, "b.json")
            write_geometry_dump(self.st, p1, meta={"base_seed": 7})
            dump = load_geometry_dump(p1)
            self.assertEqual(dump["schema"], GEOMETRY_SCHEMA)
            self.assertEqual(dump["base_seed"], 7)
            st2 = rebuild_spacetime(dump)
            self.assertEqual(verify_rebuild(st2, dump), {})
            write_geometry_dump(st2, p2, meta={"base_seed": 7})
            with open(p1) as f1, open(p2) as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_unknown_schema_and_missing_keys_raise(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as fh:
                json.dump({"schema": 99, "cells": []}, fh)
            with self.assertRaises(ValueError):
                load_geometry_dump(path)
            with open(path, "w") as fh:
                json.dump({"schema": GEOMETRY_SCHEMA, "cells": []}, fh)
            with self.assertRaises(ValueError):
                load_geometry_dump(path)

    def test_verify_rebuild_catches_a_tampered_length(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.json")
            write_geometry_dump(self.st, path)
            dump = load_geometry_dump(path)
            st2 = rebuild_spacetime(dump)
            dump_tampered = dict(dump)
            dump_tampered["edges"] = [list(row) for row in dump["edges"]]
            dump_tampered["edges"][0][2] += 0.5
            mismatches = verify_rebuild(st2, dump_tampered)
            self.assertIn("edge_lengths", mismatches)

    def test_writer_is_byte_identical_to_the_frozen_campaign_writer(self):
        # tessera.observe.geometry_dump is the IMPORTABLE home of the schema;
        # the frozen campaign scripts (examples/cobordism/proton_campaign,
        # sha256-manifested by #579) are its provenance. One schema, no
        # divergence: both writers must serialize the same state to the same
        # bytes. (The frozen scripts land in this tree when the stack rebases
        # onto main >= #579; until then this guard is pending, verified
        # out-of-band against origin/main's copy.)
        import importlib.util
        import tempfile
        worker_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "cobordism",
            "proton_campaign", "worker.py")
        if not os.path.exists(worker_path):
            self.skipTest("frozen campaign scripts not in this tree yet "
                          "(they arrive with the stack's rebase onto "
                          "main >= #579)")
        spec = importlib.util.spec_from_file_location("campaign_worker",
                                                      worker_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.assertEqual(worker.GEOMETRY_SCHEMA, GEOMETRY_SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            p_frozen = os.path.join(tmp, "frozen.json")
            p_ours = os.path.join(tmp, "ours.json")
            worker.dump_geometry(self.st, p_frozen, {"base_seed": 9})
            write_geometry_dump(self.st, p_ours, {"base_seed": 9})
            with open(p_frozen) as f1, open(p_ours) as f2:
                self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()
