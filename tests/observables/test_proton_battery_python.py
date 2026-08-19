# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Migration equivalence + gates + validation for the C++ emergent-proton battery
(#593, part of #559).

The battery (``tessera.observables``) migrates the Python ``tessera/observe``
framework of PRs #574/#575/#576/#584 into C++ Observables. Every expected value
below is anchored non-circularly:

* the pair-loop rows are the PUBLISHED #576 table (``test_pair_loop_flavor`` /
  ``test_observe_adapters.PublishedTableAnchorTest``);
* the mass/radius values on ``∂Δ⁵`` are the closed forms
  ``m_sum = 20·(2π − 3·arccos¼)`` and ``V = 6√5/96`` (``test_emergent_mass_radius``);
* the fixture-specific values (singlet residual, census, mass/radius on
  ``synthetic_b3_3``) were computed by the Python ``tessera/observe`` oracle
  against this same built ``tessera`` and frozen here — the C++ observable must
  reproduce them to each observable's gate tolerance (1e-9 singlet/block,
  1e-6 mass/radius, 1e-9 pair-loop).
"""
import importlib.util
import itertools
import json
import math
import os
import unittest
import warnings

import tessera

obs = tessera.observables
cob = tessera.cobordism

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
_CS = os.path.join(_FIX, "composite_spin")
_CAUSAL = os.path.join(_FIX, "causal_specimens")
_KEEPER = "/home/andrew/campaign-keepers/gen3-joint-real-manifold/geometry"
_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")

# ∂Δ⁵: every triangle's fan closes with three pentatopes, so every deficit is
# 2π − 3·arccos(¼); the 6 unit pentatopes have total 4-volume 6·√5/96.
S4_DEFICIT = 2.0 * math.pi - 3.0 * math.acos(0.25)
S4_VOLUME = 6.0 * math.sqrt(5.0) / 96.0


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "observe_proton_ingredients",
        os.path.join(_EX, "observe_proton_ingredients.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _composite_spin(name):
    """A live complex from a composite_spin fixture (edges as {"a,b": [re, im]})."""
    with open(os.path.join(_CS, name)) as fh:
        meta = json.load(fh)
    edges = {}
    for key, (re, im) in meta["edges"].items():
        a, b = (int(x) for x in key.split(","))
        edges[(min(a, b), max(a, b))] = complex(re, im)
    st = obs.LiveComplex.load([[int(v) for v in c] for c in meta["cells"]],
                              edges, {}, 4)
    return meta, st


def _register(st, count=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return obs.RegisterContext(st, count, 3, cob.Proton.singlet())


def _boundary_delta5():
    return tessera.Spacetime.fromCells(
        4, [list(c) for c in itertools.combinations(range(6), 5)], 1.0, 0.0)


def _star_of_apex():
    return tessera.Spacetime.fromCells(
        4, [list(c) for c in itertools.combinations(range(6), 5) if 5 in c],
        1.0, 0.0)


class SingletResidualEquivalenceTest(unittest.TestCase):
    """SingletResidual == the #574 diagnostic on synthetic_b3_3."""

    def test_record_matches_the_oracle_and_the_direct_read(self):
        meta, st = _composite_spin("synthetic_b3_3.json")
        ctx = _register(st)
        record = obs.SingletResidual().record(ctx)
        # the direct C++ r_state read is the definition
        direct = float(cob.MultiCobordism.r_state(st, 3, cob.Proton.singlet()))
        self.assertAlmostEqual(record["singlet_residual"], direct, places=15)
        # oracle value (tessera/observe.SingletDiagnostic on this fixture): ~0
        self.assertLess(record["singlet_residual"], 1e-9)
        self.assertEqual(record["holes_used"], 3)
        self.assertEqual(record["holes_total"], 4)
        self.assertEqual(record["b3"], 3)
        self.assertEqual(record["betti"], [1, 0, 0, 3, 0])
        self.assertTrue(record["holes_vs_b3_divergent"])

    def test_conjugate_residual_accessor(self):
        _meta, st = _composite_spin("synthetic_b3_3.json")
        ctx = _register(st)
        conj = [c.conjugate() for c in cob.Proton.singlet()]
        self.assertAlmostEqual(
            obs.SingletResidual().conjugate_residual(ctx),
            float(cob.MultiCobordism.r_state(st, 3, conj)), places=15)


class PairLoopFlavorPublishedTableTest(unittest.TestCase):
    """PairLoopFlavor reproduces the recorded per-fixture table.

    Re-frozen for the complexified-Regge engine (#644): the V^2 Hodge weights
    move every spectral magnitude, so the #576-era numbers are historical. The
    TOPOLOGICAL reads survived the migration bit-for-bit — odd_loop and
    dual_hole below are identical to the #576 table on every fixture, and
    r_u stays < 1e-20 — while loop_q/q/rho rescaled and the multiplicity_2_1
    verdict flipped on three fixtures (#576: True/True/True/False; now
    False/False/True/True). The flips are a REAL readout change under the new
    inner product, recorded here deliberately, not smoothed over.
    """

    # fixture -> (loop_q to 5dp, odd_loop, dual_hole, rho to 3dp, multiplicity)
    TABLE = {
        "synthetic_b3_3.json": ((0.00798, 0.00791, 0.00803), (0, 2), 1, 0.559,
                                False),
        "synthetic_b3_4.json": ((0.00731, 0.00749, 0.00765), (0, 1), 2, 0.618,
                                False),
        "synthetic_b3_5.json": ((0.00733, 0.00773, 0.00775), (0, 1), 2, 0.036,
                                True),
        "converged_b3_3.json": ((0.00626, 0.00629, 0.00629), (0, 1), 2, 0.175,
                                True),
    }

    # These two fixtures do not have reproducible loop_q or rho: a relative
    # 1e-12 perturbation of ONE edge moves loop_q by ~3%, an amplification of
    # about 3e10, and rebuilding the identical source for a different
    # instruction set moves rho from 0.036 to 0.186 on synthetic_b3_5 (#732).
    # The values are deterministic WITHIN a build — identical to 12 decimals
    # across 1, 8 and 16 threads — so the published digits record one build's
    # rounding rather than a property of the geometry, and asserting them
    # pins noise. The other two fixtures are bitwise identical across the same
    # builds, so their rows stay tight. Only the quantities that survive are
    # checked here; #732 is where the amplification gets diagnosed.
    ILL_CONDITIONED = ("synthetic_b3_4.json", "synthetic_b3_5.json")

    def test_published_rows(self):
        for name, (loop_q, odd, dual, rho, verdict) in self.TABLE.items():
            _meta, st = _composite_spin(name)
            record = obs.PairLoopFlavor().record(_register(st))
            with self.subTest(fixture=name):
                self.assertEqual(tuple(record["odd_loop"]), odd)
                self.assertEqual(record["dual_hole"], dual)
                if name in self.ILL_CONDITIONED:
                    # Still worth something: the loop charges stay in the band
                    # the table reports, which survives the amplification.
                    for measured in record["loop_q"]:
                        self.assertGreater(measured, 0.0)
                        self.assertLess(measured, 0.02)
                    continue
                for measured, published in zip(record["loop_q"], loop_q):
                    self.assertAlmostEqual(measured, published, places=5)
                self.assertAlmostEqual(record["rho"], rho, places=3)
                self.assertEqual(record["multiplicity_2_1"], verdict)
                self.assertLess(record["r_u"], 1e-20)

    def test_frozen_oracle_high_precision(self):
        # The full-precision pair-loop read frozen from the tessera/observe
        # pair_loop_flavor oracle on synthetic_b3_3 — the C++ read must match to
        # the pair-loop gate tolerance (1e-9), tighter than the 5-dp table above.
        _meta, st = _composite_spin("synthetic_b3_3.json")
        record = obs.PairLoopFlavor().record(_register(st))
        # Re-frozen from this engine on 2026-08-16 (#644, V^2 weights); the
        # previous constants were the #576-era |vol|-weight read.
        q = [0.0039256984597337875, 0.00405312793847565, 0.003981372983158158]
        loop_q = [0.007978826398209437, 0.007907071442891945,
                  0.008034500921633807]
        for measured, oracle in zip(record["q"], q):
            self.assertAlmostEqual(measured, oracle, places=9)
        for measured, oracle in zip(record["loop_q"], loop_q):
            self.assertAlmostEqual(measured, oracle, places=9)
        self.assertAlmostEqual(record["rho"], 0.559024842350687, places=9)
        self.assertLess(record["r_u"], 1e-20)

    def test_oriented_weights_pin_the_singlet_root_fixed(self):
        # w_h == the induced-orientation singlet; in the root-fixed convention
        # w0 has zero phase so w_re/w_im are [1, -1/2, -1/2] / [0, ±√3/2].
        _meta, st = _composite_spin("synthetic_b3_3.json")
        record = obs.PairLoopFlavor().record(_register(st))
        for re, ex in zip(record["w_re"], (1.0, -0.5, -0.5)):
            self.assertAlmostEqual(re, ex, places=9)
        for im, ex in zip(record["w_im"], (0.0, math.sqrt(3) / 2,
                                           -math.sqrt(3) / 2)):
            self.assertAlmostEqual(im, ex, places=9)

    def test_diquark_provenance_decides_criterion_b(self):
        _meta, st = _composite_spin("synthetic_b3_3.json")
        ctx = _register(st)
        none = obs.PairLoopFlavor().record(ctx)
        self.assertIsNone(none["odd_is_diquark_loop"])
        self.assertEqual(none["odd_is_diquark_loop_status"],
                         "not_evaluable(no_provenance)")
        # the odd loop is (0, 2): that pair decides True, another decides False
        hit = obs.PairLoopFlavor((2, 0)).record(ctx)
        self.assertIs(hit["odd_is_diquark_loop"], True)
        self.assertEqual(hit["odd_is_diquark_loop_status"], "evaluated")
        miss = obs.PairLoopFlavor((0, 1)).record(ctx)
        self.assertIs(miss["odd_is_diquark_loop"], False)

    def test_odd_one_out_and_complement(self):
        odd, rho = obs.PairLoopFlavor.odd_one_out([1.0, 1.01, 2.0])
        self.assertEqual(odd, 2)
        self.assertLess(rho, 0.02)
        self.assertEqual([obs.PairLoopFlavor.complement_hole(p)
                          for p in ((0, 1), (0, 2), (1, 2))], [2, 1, 0])


class MassRadiusEquivalenceTest(unittest.TestCase):
    """EmergentMass / EmergentRadius on the hand-checkable ∂Δ⁵ and star, plus the
    frozen oracle values on synthetic_b3_3."""

    def test_closed_s4_mass_radius(self):
        ctx = _register(_boundary_delta5_live(), count=0)
        mass = obs.EmergentMass().record(ctx)
        radius = obs.EmergentRadius().record(ctx)
        self.assertEqual(mass["census"]["n_hinges_interior"], 20)
        self.assertEqual(mass["census"]["n_hinges_boundary"], 0)
        self.assertAlmostEqual(mass["mass"]["m_sum"], 20 * S4_DEFICIT, places=9)
        self.assertAlmostEqual(mass["mass"]["m_shell"], S4_DEFICIT, places=9)
        self.assertEqual(mass["mass"]["n_im_nonzero"], 0)
        self.assertAlmostEqual(mass["localization"]["PR"], 1.0, places=9)
        self.assertAlmostEqual(radius["radius"]["Vprimal"], S4_VOLUME, places=12)
        self.assertAlmostEqual(radius["radius"]["Vdual"], S4_VOLUME, places=12)
        self.assertEqual(radius["radius"]["n_interior_vertices"], 6)
        self.assertAlmostEqual(radius["radius"]["r_dual"],
                               S4_VOLUME ** 0.25, places=12)
        # the r.m table: 6 finite combos, honest spread first
        self.assertEqual(len(mass["rm"]["combos"]), 6)
        self.assertLessEqual(mass["rm"]["spread_min"], mass["rm"]["spread_max"])

    def test_star_of_apex_hole_seeded_shells(self):
        # the dropped pentatope {0..4} is the register hole seeding the BFS.
        st = _star_of_apex()
        st.materializeFacets()
        ctx = obs.RegisterContext(st, [[0, 1, 2, 3, 4]], 1, 3,
                                  cob.Proton.singlet())
        mass = obs.EmergentMass().record(ctx)
        self.assertEqual(mass["census"]["n_hinges_interior"], 10)
        self.assertEqual(mass["census"]["n_hinges_boundary"], 10)
        self.assertEqual(mass["census"]["n_boundary_tets"], 5)
        self.assertEqual(sorted(mass["localization"]["shell_profile"]), ["0"])
        self.assertAlmostEqual(mass["localization"]["frac_within_shell1"], 1.0,
                               places=9)
        self.assertAlmostEqual(mass["mass"]["m_shell"], S4_DEFICIT, places=9)

    def test_synthetic_b3_3_frozen_oracle_values(self):
        # frozen from the tessera/observe mass_radius oracle on this fixture.
        _meta, st = _composite_spin("synthetic_b3_3.json")
        ctx = _register(st)
        mass = obs.EmergentMass().record(ctx)
        radius = obs.EmergentRadius().record(ctx)
        self.assertEqual(mass["census"]["n_hinges_interior"], 164)
        self.assertEqual(mass["census"]["n_hinges_boundary"], 36)
        self.assertEqual(mass["census"]["n_boundary_tets"], 20)
        self.assertAlmostEqual(mass["mass"]["m_sum"], 254.10203119677297,
                               places=6)
        self.assertAlmostEqual(mass["mass"]["m_shell"], 3.7730789734335914,
                               places=6)
        self.assertAlmostEqual(mass["mass"]["m_action"], 24.066705999226475,
                               places=6)
        self.assertAlmostEqual(radius["radius"]["r_dual"], 0.8827130424237054,
                               places=6)
        self.assertAlmostEqual(radius["radius"]["Vdual"], 0.6071250804215924,
                               places=6)
        self.assertAlmostEqual(radius["radius"]["Vprimal"], 1.9484067394045557,
                               places=6)
        self.assertEqual(radius["radius"]["n_interior_vertices"], 15)


def _boundary_delta5_live():
    st = _boundary_delta5()
    st.materializeFacets()
    return st


class BlockResidualsEquivalenceTest(unittest.TestCase):
    """BlockResiduals mirrors ProtonIngredients::outputBlockResidual branch by
    branch (empty region -> full leak ‖target‖²; occupied -> r_state on the
    uniform-metric sub-complex)."""

    def setUp(self):
        self.meta, self.st = _composite_spin("synthetic_b3_3.json")
        self.ctx = _register(self.st)
        self.singlet = list(cob.Proton.singlet())
        self.allv = sorted({v for c in self.meta["cells"] for v in c})

    def _row(self, label, vertices):
        record = obs.BlockResiduals(
            [obs.Block(label, vertices, self.singlet)]).record(self.ctx)
        return record["blocks"][0]

    def test_empty_region_is_the_full_leak(self):
        row = self._row("empty", [0])
        self.assertTrue(row["full_leak"])
        self.assertEqual(row["n_cells_in_region"], 0)
        self.assertAlmostEqual(row["residual"], 3.0, places=12)
        self.assertEqual(row["residual"], row["target_norm2"])

    def test_whole_region_equals_direct_r_state(self):
        row = self._row("whole", self.allv)
        self.assertFalse(row["full_leak"])
        self.assertEqual(row["n_cells_in_region"], len(self.meta["cells"]))
        sub = tessera.Spacetime.fromCells(
            4, [list(c) for c in self.meta["cells"]], 1.0, 0.0)
        expected = float(cob.MultiCobordism.r_state(sub, 3, self.singlet))
        self.assertAlmostEqual(row["residual"], expected, places=12)

    def test_sub_region_scores_only_its_cells(self):
        row = self._row("one_cell", sorted(self.meta["cells"][0]))
        self.assertEqual(row["n_cells_in_region"], 1)
        self.assertFalse(row["full_leak"])
        self.assertGreaterEqual(row["residual"], 0.0)

    def test_orphan_region_ids_are_inert_and_survive_relabel(self):
        region = sorted(self.meta["cells"][0]) + [10 ** 6]
        observable = obs.BlockResiduals(
            [obs.Block("with_orphan", region, self.singlet)])
        row = observable.record(self.ctx)["blocks"][0]
        self.assertEqual(row["n_region_vertices"], len(region))
        clean = obs.BlockResiduals(
            [obs.Block("clean", region[:-1], self.singlet)]).record(
                self.ctx)["blocks"][0]
        self.assertEqual(row["residual"], clean["residual"])
        gate = obs.ObservableGates.evaluate(observable, self.ctx)
        self.assertTrue(gate.gauge_ok)
        self.assertTrue(gate.relabel_ok)


class RegisterValidationTest(unittest.TestCase):
    """Hole selection validated at one entry point: deficit raises, surplus warns
    naming the dropped holes."""

    def test_surplus_warns_naming_the_dropped_hole(self):
        # synthetic_b3_3 stores FOUR emergent holes at b3 = 3 (the surplus case).
        _meta, st = _composite_spin("synthetic_b3_3.json")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx = obs.RegisterContext(st, 3, 3, cob.Proton.singlet())
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("4 emergent holes" in m for m in messages))
        self.assertTrue(any("(1, 4, 5, 6, 12)" in m for m in messages))
        self.assertEqual(ctx.holes_used(), 3)
        self.assertEqual(ctx.holes_total(), 4)
        self.assertEqual([list(h) for h in ctx.dropped_holes()],
                         [[1, 4, 5, 6, 12]])

    def test_deficit_raises_naming_the_count(self):
        # fill two of the four stored holes: a genuine 2-hole specimen.
        meta, _st = _composite_spin("synthetic_b3_3.json")
        holes_meta = [list(h) for h in meta["holes"]]
        edges = {}
        for key, (re, im) in meta["edges"].items():
            a, b = (int(x) for x in key.split(","))
            edges[(min(a, b), max(a, b))] = complex(re, im)
        st = obs.LiveComplex.load(meta["cells"] + holes_meta[2:], edges, {}, 4)
        self.assertEqual(len(cob.MultiCobordism.emergent_holes(st, 3)), 2)
        with self.assertRaises(ValueError) as ctx:
            obs.RegisterContext(st, 3, 3, cob.Proton.singlet())
        self.assertIn("need >= 3 register holes", str(ctx.exception))
        self.assertIn("found 2", str(ctx.exception))

    def test_eps_are_unit_signs(self):
        _meta, st = _composite_spin("synthetic_b3_3.json")
        ctx = _register(st)
        # induced-orientation signs feed the pair-loop read; here we assert the
        # register carries exactly 3 holes with the singlet pinned.
        self.assertEqual(ctx.holes_used(), 3)
        self.assertLess(obs.PairLoopFlavor().record(ctx)["r_u"], 1e-20)


class GateTest(unittest.TestCase):
    """GAUGE/RELABEL gates hold on the fixture battery; the deliberately
    leaky probes are flagged."""

    def setUp(self):
        _meta, self.st = _composite_spin("synthetic_b3_3.json")
        self.ctx = _register(self.st)

    def test_all_observables_pass_both_gates(self):
        vertices = sorted({v for c in json.load(
            open(os.path.join(_CS, "synthetic_b3_3.json")))["cells"]
            for v in c})
        observables = [
            obs.SingletResidual(),
            obs.BlockResiduals([obs.Block("whole", vertices,
                                          list(cob.Proton.singlet()))]),
            obs.EmergentMass(),
            obs.EmergentRadius(),
            obs.PairLoopFlavor((0, 2)),
        ]
        for observable in observables:
            with self.subTest(observable=observable.record_key()):
                result = obs.ObservableGates.evaluate(observable, self.ctx)
                self.assertTrue(result.gauge_ok,
                                (result.gauge_delta, result.gate_tol))
                self.assertTrue(result.relabel_ok,
                                (result.relabel_delta, result.gate_tol))

    def test_self_test_flags_both_probes(self):
        self.assertTrue(obs.ObservableGates.self_test(self.ctx))
        # the label probe leaks vertex ids -> RELABEL flags it
        self.assertGreater(
            obs.ObservableGates.relabel_delta(obs.LabelLeakProbe(), self.ctx),
            1.0)
        # the gauge probe leaks the raw target phase -> GAUGE flags it
        self.assertGreater(
            obs.ObservableGates.gauge_delta(obs.GaugeLeakProbe(), self.ctx),
            0.1)

    def test_report_delta_semantics(self):
        rd = obs.ObservableGates.report_delta
        self.assertEqual(rd({"a": 1.0, "b": [0.5, 0.25]},
                            {"a": 1.0, "b": [0.5, 0.25]}), 0.0)
        self.assertAlmostEqual(rd({"a": 1.0}, {"a": 1.001}), 1e-3, places=9)
        self.assertEqual(rd(True, False), 1.0)
        self.assertEqual(rd("ok", "changed"), float("inf"))
        self.assertEqual(rd(None, 0.0), float("inf"))
        self.assertEqual(rd(float("nan"), float("nan")), 0.0)
        self.assertEqual(rd(float("nan"), 1.0), float("inf"))
        self.assertEqual(rd([1.0, 2.0], [1.0]), float("inf"))
        with self.assertRaises(ValueError):
            rd({"a": 1}, {"b": 1})


class DriverTest(unittest.TestCase):
    """The classless driver: geometry-dump rehydration, full battery, skips."""

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver()

    def _b3_dump(self, tmpdir, **meta):
        with open(os.path.join(_CS, "synthetic_b3_3.json")) as fh:
            fixture = json.load(fh)
        verts = sorted({v for c in fixture["cells"] for v in c})
        dump = dict(meta)
        dump.update({
            "schema": 1, "dimensions": 4, "cells": fixture["cells"],
            "edges": [[int(a), int(b), re, im]
                      for k, (re, im) in fixture["edges"].items()
                      for a, b in [k.split(",")]],
            "vertex_times": [[v, 0.0] for v in verts],
        })
        path = os.path.join(tmpdir, "seed_42_geometry.json")
        with open(path, "w") as fh:
            json.dump(dump, fh)
        return path

    def test_geometry_dump_full_battery(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._b3_dump(tmp, base_seed=42, betti=[1, 0, 0, 3, 0],
                                 holes=4, diquark_pair=[0, 2])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = self.driver.observe_geometry_dump(path)
        reg = record["register"]
        self.assertEqual((reg["holes_used"], reg["holes_total"], reg["b3"]),
                         (3, 4, 3))
        self.assertTrue(reg["holes_vs_b3_divergent"])
        status = {k: v["status"] for k, v in record["observables"].items()}
        self.assertEqual(status["singlet_residual"], "measured")
        self.assertEqual(status["emergent_mass"], "measured")
        self.assertEqual(status["emergent_radius"], "measured")
        self.assertEqual(status["pair_loop_flavor"], "measured")
        self.assertEqual(status["block_residuals"], "skipped(no_provenance)")
        # diquark provenance flowed from the dump metadata
        flavor = record["observables"]["pair_loop_flavor"]["record"]
        self.assertIs(flavor["odd_is_diquark_loop"], True)
        self.assertEqual(record["input"]["mode"], "geometry_dump")
        self.assertTrue(record["input"]["dump_verified"])
        json.dumps(record)  # JSON-able

    def test_dump_verification_catches_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._b3_dump(tmp, base_seed=44, holes=12)  # wrong census
            with self.assertRaises(RuntimeError):
                self.driver.observe_geometry_dump(path)

    def test_readable_table(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._b3_dump(tmp, base_seed=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = self.driver.observe_geometry_dump(path)
        table = self.driver.render_table(record)
        self.assertIn("observable battery", table)
        self.assertIn("holes_used=3 of holes_total=4", table)
        self.assertIn("divergent=True", table)
        self.assertIn("order-of-magnitude only", table)


class KeeperDumpSmokeTest(unittest.TestCase):
    """A real campaign keeper dump rehydrates and reads (skip path — the kept
    specimens carry < 3 holes, so the register-only observables measure and the
    3-hole ones report skips)."""

    def _dumps(self, directory):
        if not os.path.isdir(directory):
            self.skipTest(f"{directory} not present")
        return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
                if f.endswith(".json")]

    def test_causal_specimen_dumps_read(self):
        driver = _load_driver()
        any_causal = False
        for path in self._dumps(_CAUSAL):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = driver.observe_geometry_dump(path)
            self.assertEqual(record["register"]["dimensions"], 4)
            self.assertEqual(
                record["observables"]["singlet_residual"]["status"], "measured")
            any_causal = any_causal or record["register"]["causal_content"]
        # at least one causal specimen honestly reports emergent causal content
        self.assertTrue(any_causal)

    def test_keeper_dumps_read(self):
        driver = _load_driver()
        for path in self._dumps(_KEEPER):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = driver.observe_geometry_dump(path)
            self.assertIn(record["observables"]["emergent_mass"]["status"],
                          ("measured", "skipped(dimensions=4 != 4)"))
            json.dumps(record)


if __name__ == "__main__":
    unittest.main()
