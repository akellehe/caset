# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ONE battery construction (#583): observe_proton_ingredients.

Fast: the geometry-dump path on a synthetic b3=3 fixture — the round trip the
campaign relies on (dump-format record -> Spacetime.fromCells rebuild ->
Register -> full battery), provenance flowing from dump metadata, the
sub-3-hole skip reporting, the dump-verification wiring, and the readable
table.

Slow: ONE bounded real ProtonIngredients end-to-end at smoke budgets (the
joint arm: it exercises the output-block provenance and the per-block
residual equivalence against the C++ oracle). Assertions are answer-agnostic
— hole counts, residual values and convergence are REPORTED observables of
the experiment, never requirements.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
import warnings

import pytest

import tessera
from tessera.observe import battery, load_geometry_dump, write_geometry_dump
from tessera.observe import mass_radius as mass_radius_module
from tessera.observe.pair_loop_flavor import build_spacetime, load_fixture

cob = tessera.cobordism

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                   "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


observe = _load("observe_proton_ingredients")


class GeometryDumpConstructionTest(unittest.TestCase):
    """The faithful path: schema-1 dump -> rebuild -> Register -> battery."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells, cls.edges = load_fixture("synthetic_b3_3.json")
        cls.st = build_spacetime(cls.cells, cls.edges)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dump_path = os.path.join(cls.tmp.name, "seed_42_geometry.json")
        write_geometry_dump(cls.st, cls.dump_path, meta={
            "base_seed": 42,
            "betti": [1, 0, 0, 3, 0],
            "holes": 4,
        })

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_round_trip_runs_the_full_battery(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = observe.observe_geometry_dump(self.dump_path)
        reg = record["register"]
        self.assertEqual(reg["holes_used"], 3)
        self.assertEqual(reg["holes_total"], 4)
        self.assertEqual(reg["b3"], 3)
        self.assertTrue(reg["holes_vs_b3_divergent"])
        status = {k: v["status"] for k, v in record["observables"].items()}
        self.assertEqual(status["singlet_diagnostic"], "measured")
        self.assertEqual(status["mass_radius"], "measured")
        self.assertEqual(status["pair_loop_flavor"], "measured")
        self.assertEqual(status["block_residuals"], "skipped(no_provenance)")
        source = record["input"]
        self.assertEqual(source["mode"], "geometry_dump")
        self.assertTrue(source["dump_verified"])
        self.assertEqual(source["base_seed"], 42)
        self.assertIn("not process-deterministic", source["note"])
        json.dumps(record)

    def test_provenance_flows_from_dump_metadata(self):
        path = os.path.join(self.tmp.name, "seed_43_geometry.json")
        write_geometry_dump(self.st, path, meta={
            "base_seed": 43,
            "diquark_pair": [0, 2],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = observe.observe_geometry_dump(path)
        self.assertEqual(record["provenance_keys"], ["diquark_pair"])
        flavor = record["observables"]["pair_loop_flavor"]["record"]
        self.assertEqual(flavor["odd_is_diquark_loop_status"], "evaluated")
        self.assertIs(flavor["odd_is_diquark_loop"], True)  # odd loop is (0,2)

    def test_provenance_file_overrides_and_extends_dump_metadata(self):
        provenance_path = os.path.join(self.tmp.name, "provenance.json")
        with open(provenance_path, "w") as fh:
            json.dump({"diquark_pair": [0, 1]}, fh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = observe.observe_geometry_dump(
                self.dump_path, provenance_path=provenance_path)
        flavor = record["observables"]["pair_loop_flavor"]["record"]
        self.assertIs(flavor["odd_is_diquark_loop"], False)

    def test_mismatching_dump_metadata_fails_the_verification(self):
        path = os.path.join(self.tmp.name, "seed_44_geometry.json")
        write_geometry_dump(self.st, path, meta={
            "base_seed": 44,
            "holes": 12,  # not this specimen's census
        })
        with self.assertRaises(RuntimeError) as ctx:
            observe.observe_geometry_dump(path)
        self.assertIn("does not match", str(ctx.exception))

    def test_sub_three_hole_dump_reports_skip_reasons(self):
        # fill two stored holes -> a genuine 2-hole specimen
        holes_meta = [list(h) for h in self.meta["holes"]]
        st2 = build_spacetime(self.cells + holes_meta[2:], self.edges)
        path = os.path.join(self.tmp.name, "seed_45_geometry.json")
        write_geometry_dump(st2, path, meta={"base_seed": 45})
        record = observe.observe_geometry_dump(path)
        self.assertEqual(record["register"]["holes_used"], 2)
        self.assertEqual(record["register"]["holes_total"], 2)
        status = {k: v["status"] for k, v in record["observables"].items()}
        self.assertEqual(status["pair_loop_flavor"],
                         "skipped(holes=2 < min_holes=3)")
        self.assertEqual(status["singlet_diagnostic"], "measured")
        self.assertEqual(status["mass_radius"], "measured")

    def test_render_table_is_readable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = observe.observe_geometry_dump(self.dump_path)
        table = observe.render_table(record)
        self.assertIn("observable battery", table)
        self.assertIn("holes_used=3 of holes_total=4", table)
        self.assertIn("divergent=True", table)
        self.assertIn("singlet_diagnostic: measured", table)
        self.assertIn("skipped(no_provenance)", table)
        self.assertIn("order-of-magnitude only", table)

    def test_dump_loader_rejects_a_non_dump_record(self):
        path = os.path.join(self.tmp.name, "not_a_dump.json")
        with open(path, "w") as fh:
            json.dump({"base_seed": 1}, fh)
        with self.assertRaises(ValueError):
            load_geometry_dump(path)


class FreshAttemptContractTest(unittest.TestCase):
    """The --seed mode is a FRESH SAMPLE, never a reproduction — the #578
    finding (the engine build is not process-deterministic) is part of the
    construction's documented contract."""

    def test_docstring_and_helpers_carry_the_non_reproduction_language(self):
        doc = " ".join(observe.__doc__.split())
        self.assertIn("NOT process-deterministic", doc)
        self.assertIn("never a reproduction", doc)
        self.assertIn("faithful", doc)
        self.assertIn("from tessera.observe import", doc)  # analyzer surface
        self.assertIn("NEW SAMPLE",
                      " ".join(observe.observe_fresh_attempt.__doc__.split()))

    def test_make_register_policy(self):
        meta, cells, edges = load_fixture("synthetic_b3_3.json")
        st = build_spacetime(cells, edges)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = observe.make_register(st)
        self.assertEqual(len(reg.holes), 3)   # 3-hole register when available
        holes_meta = [list(h) for h in meta["holes"]]
        st2 = build_spacetime(cells + holes_meta[2:], edges)
        reg2 = observe.make_register(st2)
        self.assertEqual(len(reg2.holes), 2)  # otherwise all it has


@pytest.mark.slow
class ProtonIngredientsBatteryEndToEndTest(unittest.TestCase):
    """ONE bounded real build at smoke budgets (#574's joint arm), measured
    through the battery. Answer-agnostic: every physical value is a reported
    observable; the assertions check the RECORD's coherence and the
    per-block/singlet equivalence against the C++ oracle."""

    @classmethod
    def setUpClass(cls):
        cls.ingredients = cob.ProtonIngredients(seed=1)
        cls.ingredients.build_joint(max_restarts=1, init_steps=40,
                                    evolve_steps=20, stage2_max_iters=10)
        cls.blocks = cls.ingredients.output_blocks()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.register = observe.make_register(cls.ingredients.block())
            cls.provenance = {
                "blocks": observe.blocks_provenance(cls.blocks)}
            cls.record = battery.measure_all(cls.register,
                                             provenance=cls.provenance)

    def test_joint_build_exposes_its_two_output_blocks(self):
        self.assertEqual(len(self.blocks), 2)
        for block in self.blocks:
            self.assertTrue(len(block.vertices) >= 1)
            self.assertEqual(len(block.target), 3)

    def test_record_is_json_serializable_and_census_coherent(self):
        text = json.dumps(self.record)
        self.assertIn("holes_vs_b3_divergent", text)
        reg = self.record["register"]
        self.assertEqual(reg["holes_used"], len(self.register.holes))
        self.assertEqual(reg["b3"], self.register.b3)
        # the flag IS the comparison, whatever this attempt produced
        self.assertEqual(reg["holes_vs_b3_divergent"],
                         reg["holes_total"] != reg["b3"])

    def test_singlet_diagnostic_equals_the_cpp_oracle(self):
        entry = self.record["observables"]["singlet_diagnostic"]
        self.assertEqual(entry["status"], "measured")
        self.assertAlmostEqual(
            entry["record"]["singlet_residual"],
            self.ingredients.singlet_residual(), places=12)

    def test_block_residuals_equal_the_cpp_oracle(self):
        entry = self.record["observables"]["block_residuals"]
        self.assertEqual(entry["status"], "measured")
        rows = entry["record"]["blocks"]
        self.assertEqual([r["label"] for r in rows],
                         ["baryon", "antibaryon"])
        oracle = (self.ingredients.baryon_residual(),
                  self.ingredients.antibaryon_residual())
        for row, expected in zip(rows, oracle):
            self.assertTrue(math.isfinite(row["residual"]))
            self.assertAlmostEqual(
                row["residual"], expected,
                delta=1e-12 * max(1.0, abs(expected)),
                msg=f"{row['label']} residual diverged from the C++ read")

    def test_mass_radius_matches_the_source_reader_on_the_same_block(self):
        entry = self.record["observables"]["mass_radius"]
        self.assertEqual(entry["status"], "measured")
        source = mass_radius_module.measure(self.ingredients.block(),
                                            self.register.holes)
        record = entry["record"]
        self.assertEqual(record["census"]["n_hinges_interior"],
                         source["census"]["n_hinges_interior"])
        self.assertEqual(record["mass"]["m_sum"], source["mass"]["m_sum"])
        self.assertEqual(record["radius"]["r_dual"],
                         source["radius"]["r_dual"])

    def test_pair_loop_flavor_measures_or_skips_with_reason(self):
        entry = self.record["observables"]["pair_loop_flavor"]
        if len(self.register.holes) >= 3:
            self.assertEqual(entry["status"], "measured")
            flavor = entry["record"]
            # no diquark provenance in the joint arm: reported, not guessed
            self.assertIsNone(flavor["odd_is_diquark_loop"])
            self.assertEqual(flavor["odd_is_diquark_loop_status"],
                             "not_evaluable(no_provenance)")
        else:
            self.assertEqual(
                entry["status"],
                f"skipped(holes={len(self.register.holes)} < min_holes=3)")

    def test_gates_hold_on_the_real_specimen(self):
        for name, entry in self.record["observables"].items():
            if entry["status"] != "measured":
                continue
            with self.subTest(observable=name):
                self.assertTrue(entry["gates"]["gauge_ok"], entry["gates"])
                self.assertTrue(entry["gates"]["relabel_ok"], entry["gates"])

    def test_readable_table_renders(self):
        table = observe.render_table(self.record)
        self.assertIn("observable battery", table)
        for name in ("singlet_diagnostic", "block_residuals", "mass_radius",
                     "pair_loop_flavor"):
            self.assertIn(name, table)
        print("\n" + table)


if __name__ == "__main__":
    unittest.main()
