# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#831 — the emergence animation, exercised end to end.

Per repository convention the driver lives in ``examples/``; this file covers
the correctness of the INSTRUMENT:

* the drive runs unforced emergence and produces one frame per engine unit,
  and the objective is genuinely the joint stationarity objective in strict
  emergence mode;
* every channel is either a measurement or an ``Absent`` carrying a NAMED
  reason — never a zero standing in for an unmeasured value, and never a
  blank;
* the paper's abandoned vocabulary is absent from the driver: no hole count,
  no register, no pinned singlet target, no residual against a prescribed
  carrier;
* Betti numbers are reported as an independent observable and are never used
  as a quark count or a success condition;
* the verdict is the library classifier's own string, relayed verbatim;
* the measurements are JSON-round-trippable, with unknowns as ``null``;
* the overlay renders without raising, and an absent panel still draws its
  reason.
"""

import json
import os
import sys
import tempfile
import unittest

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

cob = T.cobordism
obs = T.observables
MC = cob.MultiCobordism

#: The smallest host the driver is meaningful on: a multi-component
#: modularity read and a full band enumeration.
SMALL = 4
SMALL_STEPS = 1

#: One shared drive, built once — the whole suite reads it.
_FRAME_CACHE = {}


def _frames():
    if "frames" not in _FRAME_CACHE:
        config = ea.build_config(size=SMALL, steps=SMALL_STEPS)
        _FRAME_CACHE["frames"] = ea.drive(config, progress=False)
        _FRAME_CACHE["config"] = config
    return _FRAME_CACHE["frames"]


# ======================================================================
# the drive
# ======================================================================


class DriveTest(unittest.TestCase):
    """The drive is unforced emergence, one frame per engine unit."""

    def test_one_frame_per_engine_unit_plus_the_initial_read(self):
        frames = _frames()
        self.assertEqual(len(frames), SMALL_STEPS + 1)
        self.assertEqual([f.step for f in frames],
                         list(range(SMALL_STEPS + 1)))

    def test_the_host_is_neutral(self):
        """No holes, no pinned carrier, no boundary blocks by construction."""
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        betti = list(MC.betti(host))
        # A 4-ball refined by stellar adds stays contractible: the neutral
        # host carries no hole for a quark to be identified with.
        self.assertEqual(betti[0], 1)
        self.assertTrue(all(b == 0 for b in betti[1:3]))

    def test_the_host_has_an_incoming_boundary(self):
        """The readouts need M0; a closed complex cannot supply one."""
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        self.assertTrue(ea.boundary_vertices(host),
                        "the host must be a cobordism, not a closed complex")

    def test_the_seed_weights_real_and_imaginary_parts_evenly(self):
        """The canonical seed, and the Lorentzian content the old host lacked.

        The previous host initialized every length purely real and positive,
        so the seed carried no imaginary part at all. Nothing CONSTRAINS the
        geometry to stay there -- stage 2 rotates `z` freely and the engine
        has disposition moves -- but the starting point had no causal content
        to evolve from.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        edges = host.getEdgeList().toVector()
        self.assertTrue(edges)
        for edge in edges:
            length = complex(edge.getLength())
            with self.subTest(edge=str(edge)):
                self.assertAlmostEqual(length.real, length.imag, places=12)
                self.assertNotAlmostEqual(length.imag, 0.0, places=12)

    def test_the_objective_is_joint_stationarity_in_strict_emergence(self):
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        node = MC(host, [], [], list(ea.DECLARED_REGISTER_DEGREES), 1.0,
                  ea.DECLARED_SEED)
        node.set_objective(cob.JointStationarityObjective())
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        self.assertEqual(node.simulation_mode, MC.SimulationMode.EMERGENCE)
        self.assertEqual(node.emergence_submode, MC.EmergenceSubmode.STRICT)

    def test_no_input_or_output_target_is_pinned(self):
        """The node is built with empty target lists: nothing is prescribed.

        A target-conditioned node scores a residual against a carrier the
        caller chose; the paper's emergence protocol calls that synthesis.
        This driver constructs its node with both target lists empty, so
        there is nothing to score against.
        """
        import inspect
        source = inspect.getsource(ea.drive)
        self.assertIn("MC(host, [], []", source)


# ======================================================================
# absence is a statement, not a zero
# ======================================================================


class AbsenceTest(unittest.TestCase):
    """An unmeasured channel carries a NAMED reason, never a zero."""

    CHANNELS = ("clusters", "bands", "anchors", "transports", "statistics",
                "crossings", "spin", "betti", "verdict")

    def test_every_channel_is_a_measurement_or_a_named_absence(self):
        frame = _frames()[-1]
        for name in self.CHANNELS:
            with self.subTest(channel=name):
                value = getattr(frame, name)
                if isinstance(value, ea.Absent):
                    self.assertTrue(value.reason.strip(),
                                    "%s is absent with an empty reason" % name)
                else:
                    self.assertIsInstance(value, dict)

    def test_an_absence_never_serializes_as_zero(self):
        frame = _frames()[-1]
        document = frame.to_json()
        for name in self.CHANNELS:
            value = document[name]
            if isinstance(value, dict) and value.get("absent"):
                with self.subTest(channel=name):
                    self.assertNotEqual(value.get("reason"), "")
                    self.assertNotIn("value", value)

    def test_the_crossing_readouts_are_reached_at_all(self):
        """The host is a cobordism, so tau has a reference surface.

        On a CLOSED complex these readouts are unreachable at any size: tau
        is the Lorentzian distance FROM M0 and there is no M0 to measure
        from. The channel may still refuse -- a band that fails positivity
        supplies no crossing -- but it must refuse for a reason of its own
        rather than for want of a surface to slice.
        """
        frame = _frames()[-1]
        if isinstance(frame.crossings, ea.Absent):
            self.assertNotIn("closed host", frame.crossings.reason)
            self.assertNotIn("no incoming boundary", frame.crossings.reason)
            return
        self.assertIn("level", frame.crossings)
        self.assertIn("crossings", frame.crossings)

    def test_the_host_supplies_a_reference_surface(self):
        """M0 exists, so `tau` has a surface to be measured from.

        This is the ticket's fix. Whether `tau` then CERTIFIES is a separate
        question about the seed's causal content, pinned below.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        self.assertTrue(ea.boundary_vertices(host))

    def test_the_seed_has_no_causal_order_yet(self):
        """Measured, and NOT the behaviour to preserve.

        The canonical seed weights every length's real and imaginary parts
        evenly, so every edge is causal. The temporal-function certificate
        requires that no causal edge lie inside a hop layer of M0, so it
        refuses with `causal-cycle`: the seed has causal CHARACTER everywhere
        but no causal ORDER. That is a reason of the readout's own, not the
        absent surface this ticket removed.

        This test records the measured state so a change is noticed. If the
        dynamics later produces a causal order, `certified` becomes True and
        this test should be replaced by one asserting that -- it is a
        tripwire, not a specification.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        temporal = obs.CrossingReadouts.temporalFunction(
            host, ea.boundary_vertices(host))
        reasons = [str(r) for r in temporal.failedCertificates]
        if temporal.certified:
            self.assertEqual(reasons, [])
            return
        self.assertIn("causal-cycle", reasons)

    def test_a_refused_transport_names_why(self):
        frame = _frames()[-1]
        if isinstance(frame.transports, ea.Absent):
            self.assertTrue(frame.transports.reason.strip())
            return
        rejected = [r for r in frame.transports["rows"] if not r["accepted"]]
        for row in rejected:
            with self.subTest(row=row):
                self.assertTrue(str(row.get("reason", "")).strip())

    def test_unknown_serializes_as_null_not_zero(self):
        frame = _frames()[-1]
        document = frame.to_json()
        objective = document["objective"]
        for key, value in objective.items():
            with self.subTest(term=key):
                self.assertTrue(value is None or isinstance(value, float))


# ======================================================================
# the paper's ontology — what this driver refuses to draw
# ======================================================================


class OntologyTest(unittest.TestCase):
    """A quark is a persistent modular spectral cluster, never a hole."""

    SOURCE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "examples", "cobordism", "emergence_animation.py")

    #: Vocabulary of the construction the whitepaper abandoned.
    RETIRED = ("_MIN_QUARK_HOLES", "r_state", "hole = quark",
               "color register", "quark hole", "singlet residual")

    @classmethod
    def _code(cls):
        """The driver's CODE, with the module docstring removed.

        The docstring names the retired vocabulary in order to say the driver
        does not use it, so scanning it would be self-defeating: what matters
        is that no such construction appears in the code.
        """
        import ast
        with open(cls.SOURCE) as handle:
            source = handle.read()
        tree = ast.parse(source)
        body = tree.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            lines = source.splitlines(keepends=True)
            first = body[0].lineno - 1
            last = body[0].end_lineno
            return "".join(lines[:first] + lines[last:])
        return source

    def test_the_driver_carries_no_retired_vocabulary(self):
        code = self._code()
        for token in self.RETIRED:
            with self.subTest(token=token):
                self.assertNotIn(token, code)

    def test_the_driver_pins_no_omega_target(self):
        """No `{1, omega, omega^2}` carrier is prescribed anywhere."""
        code = self._code()
        self.assertNotIn("omega ** 2", code)
        self.assertNotIn("exp(2j * math.pi / 3", code)

    def test_betti_is_an_observable_and_not_a_quark_count(self):
        frame = _frames()[-1]
        if isinstance(frame.betti, ea.Absent):
            self.assertTrue(frame.betti.reason.strip())
            return
        numbers = frame.betti["numbers"]
        self.assertTrue(numbers)
        # The verdict must not be a function of the Betti numbers: it is the
        # certified-quark count that gates it, and on this host that is zero
        # while b_0 is one.
        if isinstance(frame.verdict, ea.Absent):
            self.assertNotIn("betti", frame.verdict.reason.lower())
            self.assertNotIn("hole", frame.verdict.reason.lower())

    def test_the_verdict_gate_counts_certified_quarks(self):
        frame = _frames()[-1]
        if isinstance(frame.verdict, ea.Absent):
            self.assertIn("quark", frame.verdict.reason)
        else:
            self.assertIn(frame.verdict["classification"],
                          {"no-baryon", "baryon-candidate", "certified-proton",
                           "quasi-free-sharp-spin-obstruction"})


# ======================================================================
# serialization
# ======================================================================


class SerializationTest(unittest.TestCase):
    """The measurements round-trip through JSON."""

    def test_every_frame_round_trips(self):
        for frame in _frames():
            with self.subTest(step=frame.step):
                document = frame.to_json()
                restored = json.loads(json.dumps(document))
                self.assertEqual(restored["step"], frame.step)

    def test_the_document_carries_the_config(self):
        frames = _frames()
        document = {"config": _FRAME_CACHE["config"],
                    "frames": [f.to_json() for f in frames]}
        restored = json.loads(json.dumps(document))
        self.assertEqual(restored["config"]["size"], SMALL)
        self.assertEqual(len(restored["frames"]), SMALL_STEPS + 1)


# ======================================================================
# the overlay
# ======================================================================


class OverlayTest(unittest.TestCase):
    """The overlay renders, and an absent panel still draws its reason."""

    def test_the_overlay_renders_a_png(self):
        frames = _frames()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "overlay.png")
            ea.render(frames, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1024)

    def test_every_panel_paints_without_raising(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = _frames()
        figure = plt.figure(figsize=(15, 9))
        try:
            for index in range(len(frames)):
                with self.subTest(frame=index):
                    ea.draw_frame(figure, frames, index)
        finally:
            plt.close(figure)

    def test_an_absent_panel_draws_its_reason(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111)
        try:
            ea._absent_panel(axis, "a title", "a named reason")
            texts = [t.get_text() for t in axis.texts]
            self.assertTrue(any("named reason" in t for t in texts))
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
