# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Mass/radius reader on converged emergent 4D interiors (#566).

Three layers, mirroring the #451 methodology the example ports to 4D:

  * closed-fan hinge selection proved on hand-checkable complexes — the boundary
    of Δ⁵ (the minimal closed S⁴: every one of its 20 triangles has a closed
    3-pentatope fan) and the star of one vertex in it (a solid 4-ball where
    exactly the 10 triangles containing the apex are interior);
  * the C++-skeleton rule — after ``build_skeleton`` the readings are finite and
    nonzero, dual and primal 4-volumes agree to machine precision (the guard that
    ``dualVolume`` sees ALL its cofaces), and the deficit matches the closed form
    2π − 3·arccos(¼) with Im = 0; without the skeleton the reader fails loudly;
  * determinism — independent same-parameter builds read bit-identically, and
    re-reading one complex is bit-identical;

plus the bounded canonical ``Proton`` build validation (@slow, the fast-test
budget: seed=1, max_restarts=1, defaults otherwise).
"""
import importlib.util
import itertools
import math
import os
import sys
import unittest

import pytest

import tessera

cob = tessera.cobordism

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


emr = _load("emergent_mass_radius")

# Regular 4-simplex dihedral angle at a triangle hinge is arccos(1/4); three
# pentatopes close each fan of dDelta^5, so every deficit is 2pi - 3 arccos(1/4).
S4_DEFICIT = 2.0 * math.pi - 3.0 * math.acos(0.25)
# Volume of the regular unit-side 4-simplex is sqrt(5)/96; dDelta^5 has 6 of them.
S4_VOLUME = 6.0 * math.sqrt(5.0) / 96.0


def _boundary_delta5():
    """The closed S^4 = dDelta^5 (6 pentatopes on {0..5}), uniform l^2 = 1."""
    cells = [list(c) for c in itertools.combinations(range(6), 5)]
    return tessera.Spacetime.fromCells(4, cells, 1.0, 0.0)


def _star_of_apex():
    """The star of vertex 5 in dDelta^5: the 5 pentatopes containing vertex 5 —
    a solid 4-ball whose boundary is the dropped cell's dDelta^4. Hand count:
    the 10 triangles containing the apex 5 have closed fans (each fan
    tetrahedron contains 5, so both its pentatopes survive the drop); the 10
    triangles inside {0..4} touch a once-shared tetrahedron and are boundary."""
    cells = [list(c) for c in itertools.combinations(range(6), 5) if 5 in c]
    return tessera.Spacetime.fromCells(4, cells, 1.0, 0.0)


class ClosedFanSelectionTest(unittest.TestCase):
    """Interior selection + census on hand-checkable 4-complexes."""

    def test_closed_s4_all_hinges_interior(self):
        st = _boundary_delta5()
        emr.build_skeleton(st)
        hinges, census = emr.interior_hinges(st)
        self.assertEqual(census["n_hinges_total"], 20)
        self.assertEqual(census["n_hinges_interior"], 20)
        self.assertEqual(census["n_hinges_boundary"], 0)
        self.assertEqual(census["n_tops"], 6)
        self.assertEqual(census["n_tets"], 15)
        self.assertEqual(census["n_boundary_tets"], 0)
        self.assertEqual(len(hinges), 20)

    def test_star_of_apex_census_matches_hand_count(self):
        st = _star_of_apex()
        emr.build_skeleton(st)
        hinges, census = emr.interior_hinges(st)
        self.assertEqual(census["n_hinges_total"], 20)
        self.assertEqual(census["n_hinges_interior"], 10)
        self.assertEqual(census["n_hinges_boundary"], 10)
        self.assertEqual(census["n_boundary_tets"], 5)
        # exactly the triangles containing the apex vertex 5 are interior
        for h in hinges:
            self.assertIn(5, h["vids"])
        # the boundary tetrahedra are getBoundary()'s (independent C++ count)
        self.assertEqual({tuple(sorted(t)) for t in st.getBoundary()},
                         set(census["boundary_tets"]))

    def test_single_pentatope_has_no_interior_hinge(self):
        st = tessera.Spacetime.fromCells(4, [list(range(5))], 1.0, 0.0)
        emr.build_skeleton(st)
        hinges, census = emr.interior_hinges(st)
        self.assertEqual(census["n_hinges_total"], 10)
        self.assertEqual(census["n_hinges_interior"], 0)
        self.assertEqual(census["n_hinges_boundary"], 10)
        self.assertEqual(hinges, [])

    def test_non_4d_complex_fails_loudly(self):
        # dDelta^4 is a 3-complex — hinges would be edges there, not triangles.
        cells = [list(c) for c in itertools.combinations(range(5), 4)]
        st = tessera.Spacetime.fromCells(3, cells, 1.0, 0.0)
        with self.assertRaises(ValueError):
            emr.interior_hinges(st)


class SkeletonRuleTest(unittest.TestCase):
    """The C++-built skeleton rule: readings finite/nonzero, dual == primal."""

    def test_readings_on_closed_s4(self):
        st = _boundary_delta5()
        o = emr.measure(st, label="dDelta^5")
        census, mass, rad, loc = (o["census"], o["mass"], o["radius"],
                                  o["localization"])
        self.assertEqual(census["n_hinges_interior"], 20)
        # every deficit is the closed form, purely real
        self.assertAlmostEqual(mass["m_sum"], 20 * S4_DEFICIT, places=10)
        self.assertEqual(mass["n_im_nonzero"], 0)
        self.assertLess(mass["max_abs_im"], 1e-12)
        # m_shell with no holes reduces to the plain mean deficit
        self.assertAlmostEqual(mass["m_shell"], S4_DEFICIT, places=10)
        self.assertGreater(mass["m_action"], 0.0)
        # dual 4-volume == primal 4-volume == 6*sqrt(5)/96 to machine precision:
        # the "dualVolume sees ALL its cofaces" guard (a corrupt skeleton halves it)
        self.assertAlmostEqual(rad["Vdual"], rad["Vprimal"], places=12)
        self.assertAlmostEqual(rad["Vprimal"], S4_VOLUME, places=12)
        self.assertEqual(rad["n_interior_vertices"], 6)
        # the dimension-correct FOURTH root on a 4-complex (#451 used the cube
        # root on its 3D event)
        self.assertAlmostEqual(rad["r_dual"], rad["Vdual"] ** 0.25, places=15)
        self.assertGreater(rad["r_dual"], 0.0)
        # a perfectly uniform complex IS the round reference: PR = 1 exactly
        self.assertAlmostEqual(loc["PR"], 1.0, places=12)
        self.assertGreater(loc["mean_re"], 0.0)  # positive curvature
        # the r.m table: 6 finite combos and an honest spread
        self.assertEqual(len(o["rm"]["combos"]), 6)
        self.assertTrue(all(math.isfinite(v) for v in o["rm"]["combos"].values()))
        self.assertLessEqual(o["rm"]["spread_min"], o["rm"]["spread_max"])

    def test_hole_seeded_shells_on_star_fixture(self):
        # Treat the dropped pentatope {0..4} as the register hole: its vertices
        # seed the BFS, every interior hinge touches shell 0, and the whole
        # curvature weight sits within shell <= 1.
        st = _star_of_apex()
        o = emr.measure(st, holes=[(0, 1, 2, 3, 4)], label="star")
        self.assertEqual(o["census"]["n_hole_vertices"], 5)
        self.assertEqual(sorted(o["localization"]["shell_profile"]), [0])
        self.assertAlmostEqual(o["localization"]["frac_within_shell1"], 1.0,
                               places=12)
        self.assertAlmostEqual(o["mass"]["m_shell"], S4_DEFICIT, places=10)

    def test_reader_fails_loudly_without_skeleton(self):
        # fromCells registers only the top cells; without the C++ skeleton the
        # canonical triangle Simplex objects don't exist and the reader must
        # refuse rather than silently reading nothing.
        st = _boundary_delta5()
        with self.assertRaises(RuntimeError):
            emr.interior_hinges(st)


class DeterminismTest(unittest.TestCase):
    """Same parameters => identical numbers, and re-reads are bit-identical."""

    @staticmethod
    def _flat(o):
        """Every numeric leaf of a measurement dict, in a stable order — down to
        the per-hinge deficits, so equality is bit-for-bit (NaN normalized to a
        sentinel so an absent reading equals an absent reading)."""
        def norm(v):
            return "nan" if isinstance(v, float) and math.isnan(v) else v

        flat = []
        for section in ("census", "mass", "radius", "localization", "rm"):
            for key, value in sorted(o[section].items()):
                if isinstance(value, (int, float)):
                    flat.append((section, key, norm(value)))
        flat.extend(("combo", k, v) for k, v in sorted(o["rm"]["combos"].items()))
        flat.extend(("shell_mean", k, v)
                    for k, v in sorted(o["mass"]["shell_means"].items(),
                                       key=lambda kv: (kv[0] is None, kv[0])))
        flat.extend(("shell_profile", k, tuple(sorted(p.items())))
                    for k, p in sorted(o["localization"]["shell_profile"].items()))
        for h in o["hinges"]:
            flat.append(("hinge", tuple(h["vids"]),
                         (h["re"], h["im"], h["dv"], h["shell"])))
        return flat

    def test_independent_builds_read_identically(self):
        a = emr.measure(_boundary_delta5())
        b = emr.measure(_boundary_delta5())
        self.assertEqual(self._flat(a), self._flat(b))

    def test_rereading_one_complex_is_identical(self):
        st = _star_of_apex()
        holes = [(0, 1, 2, 3, 4)]
        a = emr.measure(st, holes)
        b = emr.measure(st, holes)
        self.assertEqual(self._flat(a), self._flat(b))


@pytest.mark.slow
class ProtonMassRadiusValidationTest(unittest.TestCase):
    """The bounded canonical build (the fast-test budget): one real emergent
    proton, read end-to-end. The numeric values are REPORTED findings (printed
    for the record); the assertions pin structure, finiteness, and reader
    determinism — never a particular mass."""

    SEED = 1
    MAX_RESTARTS = 1

    @classmethod
    def setUpClass(cls):
        cls.proton = cob.Proton(seed=cls.SEED)
        cls.proton.build(max_restarts=cls.MAX_RESTARTS)
        cls.holes = cls.proton.quark_holes()
        cls.reading = emr.measure(cls.proton.block(), cls.holes,
                                  "canonical Proton().block()")

    def test_census_is_consistent_and_reported(self):
        census = self.reading["census"]
        self.assertEqual(census["n_hinges_interior"] + census["n_hinges_boundary"],
                         census["n_hinges_total"])
        self.assertGreater(census["n_hinges_total"], 0)
        self.assertGreater(census["n_tops"], 0)
        emr.report(self.reading)  # the readings table, for the record

    def test_relaxed_interior_readings_are_finite(self):
        # The emergent block must expose a relaxed interior at all — the whole
        # point of the reader — and every reading off it must be finite.
        census, mass, rad = (self.reading["census"], self.reading["mass"],
                             self.reading["radius"])
        self.assertGreater(census["n_hinges_interior"], 0,
                           "no closed-fan hinge — nothing interior to read")
        for name in ("m_shell", "m_sum", "m_action"):
            self.assertTrue(math.isfinite(mass[name]), name)
        self.assertTrue(math.isfinite(mass["max_abs_im"]))
        self.assertGreater(rad["Vprimal"], 0.0)
        self.assertTrue(math.isfinite(rad["r_primal"]))
        pr = self.reading["localization"]["PR"]
        self.assertTrue(math.isfinite(pr))
        self.assertGreater(pr, 0.0)
        self.assertLessEqual(pr, 1.0 + 1e-12)

    def test_rm_table_covers_all_definitions(self):
        rm = self.reading["rm"]
        self.assertEqual(
            sorted(rm["combos"]),
            sorted(f"{r} x {m}" for m in ("m_shell", "m_sum", "m_action")
                   for r in ("r_dual", "r_primal")))
        self.assertLessEqual(rm["spread_min"], rm["spread_max"])

    def test_reader_is_deterministic_on_the_built_block(self):
        # Same complex, same holes => bit-identical numbers. (The BUILD itself is
        # documented FP-thread-variable — test_proton_cpp_python.py — so reader
        # determinism, not build determinism, is the assertable contract.)
        again = emr.measure(self.proton.block(), self.holes,
                            "canonical Proton().block()")
        self.assertEqual(DeterminismTest._flat(self.reading),
                         DeterminismTest._flat(again))


if __name__ == "__main__":
    unittest.main()
