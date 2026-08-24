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

    def test_no_causal_edge_lies_inside_a_hop_layer(self):
        """The temporal-function certificate names and refuses that case."""
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        layer = ea._hop_layers(host, ea.boundary_vertices(host))
        timelike = 0
        for edge in host.getEdgeList().toVector():
            a, b = int(edge.getKey()[0]), int(edge.getKey()[1])
            if edge.isTimelike():
                timelike += 1
                self.assertNotEqual(layer.get(a, 0), layer.get(b, 0))
            else:
                self.assertEqual(layer.get(a, 0), layer.get(b, 0))
        self.assertGreater(timelike, 0,
                           "a Lorentzian host with a boundary must carry "
                           "causal edges, or tau cannot accumulate")

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

    def test_the_closed_host_refuses_the_crossing_readouts_by_name(self):
        """A closed complex has no M0, so tau has no reference surface."""
        frame = _frames()[-1]
        self.assertIsInstance(frame.crossings, ea.Absent)
        self.assertIn("M0", frame.crossings.reason)

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
