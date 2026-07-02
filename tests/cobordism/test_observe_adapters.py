# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Migration equivalence of the battery adapters (#583): each adapter's
values EQUAL its source example's values on the same fixtures.

The sources are the three readout surfaces this layer migrates by
composition: the #576 pair-loop flavor read, the #575 mass/radius battery,
and the #574 singlet/per-block reads. The example scripts are loaded through
the repo's example-loading pattern and must expose the SAME functions the
package owns (moved-shared-helper — one home, no second copy). The published
per-fixture table of PR #576 anchors the values non-circularly.

Fast throughout; the real-build (ProtonIngredients) equivalence runs in the
@slow end-to-end of test_observe_proton_ingredients.py.
"""
import importlib.util
import json
import os
import sys
import unittest
import warnings

import tessera
from tessera.observe import Register, battery, measure_all
from tessera.observe import mass_radius as mass_radius_module
from tessera.observe import pair_loop_flavor as pair_loop_module
from tessera.observe.adapters import (
    BlockResiduals,
    MassRadius,
    PairLoopFlavor,
    SingletDiagnostic,
)

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


def _quiet_register(st, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Register(st, **kwargs)


class AdapterFixtureBase(unittest.TestCase):
    fixture = "synthetic_b3_3.json"

    @classmethod
    def setUpClass(cls):
        cls.plf = _load("pair_loop_flavor")
        cls.meta, cls.cells, cls.edges = pair_loop_module.load_fixture(
            cls.fixture)
        cls.st = pair_loop_module.build_spacetime(cls.cells, cls.edges)
        cls.register = _quiet_register(cls.st, count=3)


class MovedSharedHelperTest(AdapterFixtureBase):
    """The examples import the package machinery — one home, no second copy."""

    def test_pair_loop_example_reexports_the_package_functions(self):
        for name in ("joint_read", "odd_one_out", "evaluate_criteria",
                     "complement_hole", "register_holes", "build_spacetime",
                     "load_fixture", "gauge_gate", "relabel_gate",
                     "read_structure", "_facet_indices"):
            self.assertIs(getattr(self.plf, name),
                          getattr(pair_loop_module, name),
                          f"{name} is duplicated instead of shared")
        self.assertEqual(self.plf.SINGLET, pair_loop_module.SINGLET)
        self.assertEqual(self.plf.PAIR_LOOPS, pair_loop_module.PAIR_LOOPS)

    def test_mass_radius_example_reexports_the_package_functions(self):
        emr = _load("emergent_mass_radius")
        for name in ("build_skeleton", "interior_hinges", "masses", "radii",
                     "localization", "rm_table", "measure"):
            self.assertIs(getattr(emr, name),
                          getattr(mass_radius_module, name),
                          f"{name} is duplicated instead of shared")


class PairLoopFlavorEquivalenceTest(AdapterFixtureBase):
    """The adapter's values equal the source read's on the same fixture."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source_read = pair_loop_module.joint_read(
            cls.st, cls.register.holes)
        cls.source_verdict = pair_loop_module.evaluate_criteria(
            cls.source_read)
        cls.record = PairLoopFlavor().measure(cls.register)

    def test_direct_channels_equal_the_source(self):
        self.assertEqual(self.record["r_u"], float(self.source_read["r_u"]))
        self.assertEqual(self.record["q"],
                         [float(x) for x in self.source_read["q"]])
        self.assertEqual(self.record["loop_q"],
                         [float(x) for x in self.source_read["loop_q"]])
        self.assertEqual(self.record["dual_residual"],
                         [float(x) for x in self.source_read["dual_residual"]])

    def test_verdict_channels_equal_the_source(self):
        self.assertEqual(tuple(self.record["odd_loop"]),
                         self.source_verdict["odd_loop"])
        self.assertEqual(self.record["dual_hole"],
                         self.source_verdict["dual_hole"])
        self.assertEqual(self.record["rho"], self.source_verdict["rho"])
        self.assertEqual(self.record["multiplicity_2_1"],
                         self.source_verdict["multiplicity_2_1"])

    def test_periods_equal_the_source_in_the_root_fixed_convention(self):
        w = self.source_read["w"]
        phase0 = w[0] / abs(w[0])
        for record_re, record_im, source in zip(
                self.record["w_re"], self.record["w_im"],
                [wi / phase0 for wi in w]):
            self.assertEqual(complex(record_re, record_im), source)
        for record_re, record_im, source in zip(
                self.record["loop_w_re"], self.record["loop_w_im"],
                [wi / phase0 for wi in self.source_read["loop_w"]]):
            self.assertEqual(complex(record_re, record_im), source)

    def test_register_cache_injection_changes_nothing(self):
        # joint_read with the Register's cached synthesis/signs/weights ==
        # the from-scratch read, bit for bit (the read is deterministic).
        reg = self.register
        cached = pair_loop_module.joint_read(
            self.st, reg.holes, reg.target,
            es=reg.es, sigma=reg.eps,
            weights=cob.HodgeLaplacian(self.st).weights(3),
            cell_index={frozenset(t): i
                        for i, t in enumerate(reg.es.cellSimplices())})
        fresh = pair_loop_module.joint_read(self.st, reg.holes, reg.target)
        self.assertEqual(cached["r_u"], fresh["r_u"])
        for key in ("q", "loop_q", "dual_residual", "w", "loop_w", "sigma"):
            self.assertEqual(list(cached[key]), list(fresh[key]), key)

    def test_provenance_decides_criterion_b_and_absence_reports_not_evaluable(self):
        record_none = PairLoopFlavor().measure(self.register, None)
        self.assertIsNone(record_none["odd_is_diquark_loop"])
        self.assertEqual(record_none["odd_is_diquark_loop_status"],
                         "not_evaluable(no_provenance)")
        # the odd loop on this fixture is (0, 2) — supplying that pair as the
        # recorded diquark decides True; a different pair decides False
        record_hit = PairLoopFlavor().measure(
            self.register, {"diquark_pair": [2, 0]})
        self.assertIs(record_hit["odd_is_diquark_loop"], True)
        self.assertEqual(record_hit["odd_is_diquark_loop_status"], "evaluated")
        record_miss = PairLoopFlavor().measure(
            self.register, {"diquark_pair": [0, 1]})
        self.assertIs(record_miss["odd_is_diquark_loop"], False)
        self.assertEqual(record_miss["odd_is_diquark_loop_status"],
                         "evaluated")


class PublishedTableAnchorTest(unittest.TestCase):
    """The #576 per-fixture table, re-measured through the battery adapter —
    the non-circular anchor that migration preserved the physics."""

    # fixture -> (loop_q to 5 decimals, odd_loop, dual_hole, rho to 3
    # decimals, multiplicity_2_1)
    TABLE = {
        "synthetic_b3_3.json": ((0.06054, 0.05971, 0.06063), (0, 2), 1,
                                0.103, True),
        "synthetic_b3_4.json": ((0.05726, 0.05971, 0.06074), (0, 1), 2,
                                0.348, True),
        "synthetic_b3_5.json": ((0.05542, 0.05722, 0.05772), (0, 1), 2,
                                0.248, True),
        "converged_b3_3.json": ((0.05153, 0.05165, 0.05177), (0, 1), 2,
                                0.665, False),
    }

    def test_adapter_reproduces_the_published_rows(self):
        for name, (loop_q, odd, dual, rho, verdict) in self.TABLE.items():
            meta, cells, edges = pair_loop_module.load_fixture(name)
            register = _quiet_register(
                pair_loop_module.build_spacetime(cells, edges), count=3)
            record = PairLoopFlavor().measure(register)
            with self.subTest(fixture=name):
                for measured, published in zip(record["loop_q"], loop_q):
                    self.assertAlmostEqual(measured, published, places=5)
                self.assertEqual(tuple(record["odd_loop"]), odd)
                self.assertEqual(record["dual_hole"], dual)
                self.assertAlmostEqual(record["rho"], rho, places=3)
                self.assertEqual(record["multiplicity_2_1"], verdict)
                self.assertLess(record["r_u"], 1e-20)


class MassRadiusEquivalenceTest(AdapterFixtureBase):
    """The adapter record equals the source reader's blocks on the same
    complex and holes (shell keys stringified, the vertex-id-carrying
    boundary_tets list dropped — reporting shape, not values)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = mass_radius_module.measure(cls.st, cls.register.holes)
        cls.record = MassRadius().measure(cls.register)

    def test_census_equals_the_source(self):
        expected = {k: v for k, v in self.source["census"].items()
                    if k != "boundary_tets"}
        self.assertEqual(self.record["census"], expected)

    def test_masses_equal_the_source(self):
        expected = dict(self.source["mass"])
        expected["shell_means"] = {
            ("unshelled" if k is None else str(int(k))): v
            for k, v in expected["shell_means"].items()}
        self.assertEqual(self.record["mass"], expected)

    def test_radius_rm_and_localization_equal_the_source(self):
        self.assertEqual(self.record["radius"], self.source["radius"])
        self.assertEqual(self.record["rm"], self.source["rm"])
        expected = dict(self.source["localization"])
        expected["shell_profile"] = {
            ("unshelled" if k is None else str(int(k))): v
            for k, v in expected["shell_profile"].items()}
        self.assertEqual(self.record["localization"], expected)
        self.assertEqual(self.record["n_holes"], self.source["n_holes"])


class SingletDiagnosticEquivalenceTest(AdapterFixtureBase):
    """The adapter is the #574 diagnostic: the relabeling-invariant singlet
    r_state of Proton.singlet() against the whole, plus the census."""

    def test_residual_equals_the_direct_r_state_read(self):
        record = SingletDiagnostic().measure(self.register)
        self.assertEqual(
            record["singlet_residual"],
            float(cob.MultiCobordism.r_state(self.st, 3, cob.Proton.singlet())))
        self.assertEqual(record["holes_total"], 4)
        self.assertEqual(record["holes_used"], 3)
        self.assertEqual(record["b3"], 3)
        self.assertEqual(record["betti"], [1, 0, 0, 3, 0])
        self.assertTrue(record["holes_vs_b3_divergent"])

    def test_divergence_flag_follows_the_register(self):
        meta, cells, edges = pair_loop_module.load_fixture(
            "converged_b3_3.json")
        register = _quiet_register(
            pair_loop_module.build_spacetime(cells, edges), count=3)
        record = SingletDiagnostic().measure(register)
        self.assertFalse(record["holes_vs_b3_divergent"])
        self.assertEqual((record["holes_total"], record["b3"]), (3, 3))


class BlockResidualsEquivalenceTest(AdapterFixtureBase):
    """The Python mirror of ProtonIngredients::outputBlockResidual, branch by
    branch: empty region -> the full leak ||target||^2; occupied region ->
    r_state on the block's own UNIFORM-metric fromCells sub-complex. (The
    real-build equivalence against baryon_residual()/antibaryon_residual()
    runs in the @slow end-to-end.)"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.singlet = list(cob.Proton.singlet())

    def test_empty_region_reports_the_full_leak(self):
        record = BlockResiduals().measure(
            self.register,
            {"blocks": [{"label": "empty", "vertices": [0],
                         "target": self.singlet}]})
        (row,) = record["blocks"]
        self.assertTrue(row["full_leak"])
        self.assertEqual(row["n_cells_in_region"], 0)
        self.assertEqual(row["residual"], row["target_norm2"])
        self.assertAlmostEqual(row["residual"], 3.0, places=12)

    def test_whole_region_equals_direct_r_state_on_the_uniform_rebuild(self):
        vertices = sorted({v for c in self.cells for v in c})
        record = BlockResiduals().measure(
            self.register,
            {"blocks": [{"label": "whole", "vertices": vertices,
                         "target": self.singlet}]})
        (row,) = record["blocks"]
        self.assertFalse(row["full_leak"])
        self.assertEqual(row["n_cells_in_region"], len(self.cells))
        # the mirror's own construction, driven directly
        cells_inside = [[v.getId() for v in c.getVertices()]
                        for c in self.st.getTopSimplices()]
        sub = tessera.Spacetime.fromCells(4, cells_inside, 1.0, 0.0)
        expected = float(cob.MultiCobordism.r_state(sub, 3, self.singlet))
        self.assertEqual(row["residual"], expected)

    def test_sub_region_scores_only_its_own_cells(self):
        # one ambient cell's own vertex set: a genuine strict sub-region —
        # only the cells living entirely on those five vertices qualify
        region = sorted(self.cells[0])
        record = BlockResiduals().measure(
            self.register,
            {"blocks": [{"label": "one_cell", "vertices": region,
                         "target": self.singlet}]})
        (row,) = record["blocks"]
        self.assertGreater(row["n_cells_in_region"], 0)
        self.assertLess(row["n_cells_in_region"], len(self.cells))
        self.assertFalse(row["full_leak"])
        self.assertGreaterEqual(row["residual"], 0.0)

    def test_target_re_im_pairs_equal_complex_targets(self):
        vertices = sorted({v for c in self.cells for v in c})
        by_complex = BlockResiduals().measure(
            self.register,
            {"blocks": [{"label": "b", "vertices": vertices,
                         "target": self.singlet}]})
        by_pairs = BlockResiduals().measure(
            self.register,
            {"blocks": [{"label": "b", "vertices": vertices,
                         "target_re": [t.real for t in self.singlet],
                         "target_im": [t.imag for t in self.singlet]}]})
        self.assertEqual(by_complex, by_pairs)

    def test_orphaned_region_ids_are_inert_and_survive_the_relabel_gate(self):
        # An emergent block region can reference vertices no longer in any
        # top cell (surgical moves orphan them — observed on a real joint
        # build). They are inert in the residual, and the relabel gate must
        # carry them without a KeyError while preserving the region size.
        region = sorted(self.cells[0]) + [10 ** 6]  # one id not in the complex
        provenance = {"blocks": [{"label": "with_orphan", "vertices": region,
                                  "target": self.singlet}]}
        observable = BlockResiduals()
        entry = observable.gated_measure(self.register, provenance)
        self.assertEqual(entry["status"], "measured")
        (row,) = entry["record"]["blocks"]
        self.assertEqual(row["n_region_vertices"], len(region))
        # the orphan changed nothing vs the same region without it
        (clean_row,) = observable.measure(
            self.register, {"blocks": [{"label": "with_orphan",
                                        "vertices": region[:-1],
                                        "target": self.singlet}]})["blocks"]
        self.assertEqual(row["residual"], clean_row["residual"])
        self.assertEqual(row["n_cells_in_region"],
                         clean_row["n_cells_in_region"])
        self.assertTrue(entry["gates"]["gauge_ok"], entry["gates"])
        self.assertTrue(entry["gates"]["relabel_ok"], entry["gates"])

    def test_block_order_and_labels_are_preserved(self):
        vertices = sorted({v for c in self.cells for v in c})
        record = BlockResiduals().measure(
            self.register,
            {"blocks": [
                {"label": "baryon", "vertices": vertices,
                 "target": self.singlet},
                {"label": "antibaryon", "vertices": [0],
                 "target": self.singlet},
            ]})
        self.assertEqual([b["label"] for b in record["blocks"]],
                         ["baryon", "antibaryon"])
        self.assertEqual(record["n_blocks"], 2)


class BatteryGatesOnFixtureTest(AdapterFixtureBase):
    """The full gated battery on the fixture: every measured observable's
    GAUGE and RELABEL residuals sit below its gate tolerance (BlockResiduals
    exercises transform_provenance — its regions are vertex-id sets)."""

    def test_full_battery_gates_hold(self):
        vertices = sorted({v for c in self.cells for v in c})
        provenance = {
            "diquark_pair": [0, 2],
            "blocks": [{"label": "whole", "vertices": vertices,
                        "target": list(cob.Proton.singlet())}],
        }
        record = measure_all(self.register, provenance=provenance)
        for name, entry in record["observables"].items():
            with self.subTest(observable=name):
                self.assertEqual(entry["status"], "measured")
                self.assertTrue(entry["gates"]["gauge_ok"],
                                entry["gates"])
                self.assertTrue(entry["gates"]["relabel_ok"],
                                entry["gates"])
        json.dumps(record)

    def test_default_battery_lineup(self):
        self.assertEqual([o.name for o in battery.observables],
                         ["singlet_diagnostic", "block_residuals",
                          "mass_radius", "pair_loop_flavor"])


if __name__ == "__main__":
    unittest.main()
