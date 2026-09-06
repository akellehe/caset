# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The qubit input mode of ``examples/cobordism/emergence_animation.py``
(#963, T4 of ``docs/design/qubit_cobordism_spec.md``, delta D4).

The qubit cobordism runs through the SAME drive as the neutral mode (spec R5):
``drive`` takes its node from a factory selected by ``config["inputs"]`` --
the neutral host by default, ``build_qubit_node`` for ``--inputs qubit`` --
and the loop, the engine calls and the frames are the same either way. The
qubit factory is the T1-T3 setup: two ``SimplicialQubit.flat_torus`` inputs,
their collar as the host, each torus one input block with its holomorphic
form attached as a degree-1 fiber on the harmonic contour, its period frame
with the dual on the block's own pencil, chi of spec S5 as the two-body
target, fiber residuals at the input weight, the node's default objective.

Coverage:

* the drive produces one frame per engine unit plus the seed, publishing
  each to the live callback as it completes, and every channel of spec S6
  is a measurement or an ``Absent`` with a named reason;
* frame 0 is the collar seed: each block's tau-hat equals tau_in to 1e-9
  with both distances at zero, the own-Laplacian residuals at their floor,
  the monodromy the identity, Betti numbers [1, 2, 1, 0], the boundary two
  tori (Euler characteristic 0 each), the two-body read in the period frames
  (2 x 2, Schmidt rank 2), the restricted leaks recorded (they are NOT at
  rounding on the collar seed -- 3.1e-3 and 9.3e-3 on 3x3 tori -- because the
  whole's harmonic representative restricted to a torus differs from the
  torus's own; a finding for T5, not a rule here);
* after synthesis every channel is still read, the block residuals hold
  below the weight's level and the surfaces are still the tori;
* the algebra: chi of spec S5 against an explicit computation, the exact
  block evolution against first order at small J t, the selection rule;
* ``to_json`` round-trips through ``json.dumps`` with complex numbers as
  [re, im] pairs, and the run document carries the inputs once;
* ``render`` writes a PNG and a two-frame GIF under Agg;
* the neutral mode is unchanged: its config defaults, its record schema
  (no qubit key), and its tests, which run untouched next to this file;
* the config refuses a modulus off the upper half plane, a grid below 3, a
  non-positive weight and an unknown mode by name, and the CLI parses the
  qubit flags, ``--tau-b=-0.2+0.8j`` included.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import tessera as T  # noqa: F401  (the driver imports the package)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
GRID = 3
STEPS = 2
FLOOR = 1e-24

_CACHE = {}


def _run():
    """One shared qubit drive on 3x3 tori, built once for the whole file."""
    if "result" not in _CACHE:
        seen = []
        config = ea.build_config(steps=STEPS, stage1_iters=1, stage2_iters=2,
                                 tolerance=1e-30, inputs=ea.InputMode.QUBIT,
                                 tau_a=TAU_A, tau_b=TAU_B, grid=GRID)
        result = ea.drive(
            config, progress=False,
            on_frame=lambda frames, index: seen.append((index, len(frames))))
        _CACHE.update(result=result, config=config, seen=seen)
    return _CACHE


def _present(value):
    return not isinstance(value, ea.Absent)


def _read(frame, index):
    read = frame.blocks[index]["read"]
    assert _present(read), read.reason
    return read


# --------------------------------------------------------------------------- #
# the drive and the channels
# --------------------------------------------------------------------------- #
def test_one_frame_per_unit_published_live():
    run = _run()
    frames = run["result"].frames
    assert len(frames) == STEPS + 1
    assert [f.step for f in frames] == list(range(STEPS + 1))
    assert run["result"].terminator == ea.Terminator.STEPS
    assert run["seen"] == [(i, i + 1) for i in range(STEPS + 1)]
    assert run["result"].inputs is not None
    assert all(f.inputs is run["result"].inputs for f in frames)


def test_every_channel_is_a_measurement_or_a_named_absence():
    for frame in _run()["result"].frames:
        for name in ea.EmergenceFrame.QUBIT_CHANNELS + ("betti", "objective"):
            value = getattr(frame, name)
            if isinstance(value, ea.Absent):
                assert value.reason.strip(), "%s is absent without a reason" % name
            else:
                assert isinstance(value, (dict, list)), name
        assert isinstance(frame.blocks, list) and len(frame.blocks) == 2
        for row in frame.blocks:
            assert row["label"] in ea.DECLARED_TORUS_LABELS
            assert isinstance(row["reversed"], bool)
            for key in ("residual", "read"):
                value = row[key]
                assert _present(value) or value.reason.strip(), key
        # the neutral mode's certificates are absent by name here
        for name in ea.EmergenceFrame.CERTIFICATE_CHANNELS:
            value = getattr(frame, name)
            assert isinstance(value, ea.Absent) and "qubit input mode" in value.reason
        assert isinstance(frame.dual, ea.Absent) and "3-dimensional" in frame.dual.reason
        # the tori highlighted in the layout
        assert _present(frame.layout)
        assert [(m["label"], m["missing"]) for m in frame.layout["highlight"]] == [("A", 0), ("B", 0)]
        assert all(len(m["edges"]) == 27 for m in frame.layout["highlight"])


def test_frame_zero_is_the_collar_seed():
    run = _run()
    frame = run["result"].frames[0]
    inputs = run["result"].inputs
    assert inputs.tau_in == [TAU_A, TAU_B]
    assert inputs.objective_name == "legacy"
    assert inputs.weight == ea.DECLARED_INPUT_WEIGHT and inputs.regge
    assert inputs.seed == {"cells": 54, "vertices": 18, "edges": 63}
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        row = frame.blocks[index]
        assert row["tau_in"] == tau_in and row["weight"] == ea.DECLARED_INPUT_WEIGHT
        assert row["residual"] < FLOOR, row["residual"]
        read = _read(frame, index)
        assert abs(read["tau"] - tau_in) < 1e-9, read["tau"]
        assert read["fubini_study_distance"] < 1e-9 and read["weil_petersson_distance"] < 1e-9
        assert (read["vertices"], read["edges"], read["faces"]) == (9, 27, 18)
        assert not read["marking_swapped"] and not read["near_degenerate"]
        assert read["j_residual"] < 1e-12
        bloch = np.asarray(read["bloch"])
        assert abs(np.linalg.norm(bloch) - 1.0) < 1e-12 and bloch[1] > 0
        assert np.allclose(bloch, np.asarray(inputs.tori[index].bloch()), atol=1e-12)
    # the whole: Betti numbers, boundary, completion, monodromy, zero mode
    assert frame.betti["numbers"] == {0: 1, 1: 2, 2: 1, 3: 0}
    assert _present(frame.boundary)
    assert frame.boundary["count"] == 2 and frame.boundary["faces"] == 36
    assert [(c["euler_characteristic"], c["faces"], c["block"]) for c in frame.boundary["components"]] == [(0, 18, 0), (0, 18, 1)]
    assert frame.completion == {"bridge_phase_complete": True, "uncovered_faces": 0}
    assert _present(frame.monodromy), frame.monodromy
    assert frame.monodromy["betti"] == [1, 2, 1, 0] and frame.monodromy["harmonic_rank"] == 2
    assert frame.monodromy["rounded"] == [[1, 0], [0, 1]]
    assert frame.monodromy["rounding_residual"] < 1e-9 and frame.monodromy["fit_residual"] < 1e-9
    assert _present(frame.leaks) and frame.leaks["harmonic_rank"] == 2
    for row in frame.leaks["per_block"]:
        assert _present(row) and 0.0 <= row["leak"] <= 1.0 and row["rank"] == 2
    # the two-body read in the period frames against chi of spec S5
    two_body = frame.two_body
    assert _present(two_body), two_body
    assert two_body["in_frames"] and two_body["choi_decomposed"] and two_body["shape"] == [2, 2]
    assert 0.0 <= two_body["residual"] <= 1.0
    assert two_body["schmidt_rank"] == 2 and len(two_body["singular_values"]) == 2
    assert two_body["reversal_residual"] < 1e-8
    assert all(r < FLOOR for r in two_body["input_fiber_residuals"])
    transfer = np.asarray(two_body["transfer"])
    assert transfer.shape == (2, 2) and np.abs(transfer.imag).max() < 1e-12 * np.abs(transfer).max()
    # the objective: the node's default, Regge stationarity plus Gamma r_U
    assert frame.objective["total"] is not None
    assert frame.objective["regge_stationarity"] > 0
    assert frame.objective["register_residual"] == pytest.approx(two_body["residual"], rel=1e-9)
    print("\n[T4] seed: leaks %s, two-body %.6f, T %s" % (
        ["%.3e" % r["leak"] for r in frame.leaks["per_block"]], two_body["residual"],
        np.round(transfer.real, 6).tolist()))


def test_synthesis_reads_every_channel_and_holds_the_tori():
    run = _run()
    frames = run["result"].frames
    last = frames[-1]
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        row = last.blocks[index]
        assert isinstance(row["residual"], float) and row["residual"] < 1e-3, row["residual"]
        read = _read(last, index)
        assert (read["vertices"], read["edges"], read["faces"]) >= (9, 27, 18), "the surface keeps the torus"
        assert read["weil_petersson_distance"] >= 0.0 and read["fubini_study_distance"] >= 0.0
        assert abs(np.linalg.norm(read["bloch"]) - 1.0) < 1e-12
    assert _present(last.monodromy) and _present(last.two_body) and _present(last.leaks)
    assert _present(last.boundary) and _present(last.completion) and _present(last.betti)
    assert last.two_body["in_frames"] and last.two_body["shape"] == [2, 2]
    totals = [f.objective["total"] for f in frames]
    assert all(t is not None for t in totals) and totals[-1] < totals[0]
    print("\n[T4] after %d units: objective %s, blocks %s, two-body %s, tau-hat %s, monodromy %s" % (
        STEPS, ["%.4f" % t for t in totals],
        ["%.2e" % f.blocks[0]["residual"] + "/%.2e" % f.blocks[1]["residual"] for f in frames],
        ["%.4f" % f.two_body["residual"] for f in frames],
        [str(_read(last, i)["tau"]) for i in range(2)], last.monodromy["rounded"]))


# --------------------------------------------------------------------------- #
# the algebra of spec S5
# --------------------------------------------------------------------------- #
def test_the_flip_flop_algebra():
    rng = np.random.default_rng(963)
    psi, phi = (rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(2))
    psi, phi = psi / np.linalg.norm(psi), phi / np.linalg.norm(phi)
    chi = ea.two_qubit_flip_flop(psi, phi)
    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    expected = np.outer(lowering @ psi, lowering.T @ phi) + np.outer(lowering.T @ psi, lowering @ phi)
    assert np.allclose(chi, expected, atol=1e-15)
    assert chi[0, 0] == 0 and chi[1, 1] == 0, "the selection rule: chi lives in the |01>, |10> sector"
    bilinear = (np.kron(lowering, lowering.T) + np.kron(lowering.T, lowering)) @ np.kron(psi, phi)
    assert np.allclose(chi, bilinear.reshape(2, 2)), "chi is the bilinear map's image of the product"
    small = ea.flip_flop_evolution(psi, phi, 2.0, 5e-5)
    product = np.kron(psi, phi).reshape(2, 2)
    assert np.allclose(small["product_state"], product)
    assert np.allclose(small["first_order_amplitudes"], -1j * 1e-4 * chi)
    velocity = (small["exact_amplitudes"] - product) / (-1j * 1e-4)
    assert np.abs(velocity - chi).max() < 1e-3, "the exact evolution's first order is -i J t chi"
    large = ea.flip_flop_evolution(psi, phi, 1.0, 0.7)
    exact = large["exact_amplitudes"]
    assert abs(np.linalg.norm(exact) - 1.0) < 1e-12, "unitary"
    assert exact[0, 0] == product[0, 0] and exact[1, 1] == product[1, 1], "the size-1 blocks are untouched"
    c, s = np.cos(0.7), np.sin(0.7)
    assert np.allclose(exact[0, 1], c * product[0, 1] - 1j * s * product[1, 0])
    assert np.allclose(exact[1, 0], c * product[1, 0] - 1j * s * product[0, 1])
    assert large["Jt"] == 0.7


# --------------------------------------------------------------------------- #
# serialization and rendering
# --------------------------------------------------------------------------- #
def test_to_json_round_trips_with_complex_numbers_as_pairs():
    run = _run()
    for frame in run["result"].frames:
        document = json.loads(json.dumps(frame.to_json()))
        assert document["step"] == frame.step
        for name in ea.EmergenceFrame.QUBIT_CHANNELS:
            assert name in document
        read = document["blocks"][0]["read"]
        assert read["tau"] == [frame.blocks[0]["read"]["tau"].real, frame.blocks[0]["read"]["tau"].imag]
        assert document["blocks"][0]["tau_in"] == [TAU_A.real, TAU_A.imag]
        assert len(document["two_body"]["transfer"]) == 2 and len(document["two_body"]["transfer"][0][0]) == 2
        assert document["monodromy"]["rounded"] == frame.monodromy["rounded"]
    inputs = json.loads(json.dumps(run["result"].inputs.to_json()))
    assert inputs["tau_in"] == [[TAU_A.real, TAU_A.imag], [TAU_B.real, TAU_B.imag]]
    assert inputs["objective"] == "legacy" and "register_residual" in inputs["objective_terms"]
    assert inputs["reversed"] == run["result"].inputs.reversed
    assert len(inputs["algebra"]["chi"]) == 2 and inputs["algebra"]["Jt"] == ea.DECLARED_COUPLING * ea.DECLARED_TIME
    assert inputs["markings"][0][0][0] == [0, 1] or len(inputs["markings"][0][0]) == GRID
    whole = json.dumps({"config": run["config"], "inputs": inputs,
                        "frames": [f.to_json() for f in run["result"].frames]}, sort_keys=True)
    assert json.loads(whole)["config"]["inputs"] == "qubit"


def test_render_png_and_gif_under_agg():
    frames = _run()["result"].frames
    with tempfile.TemporaryDirectory() as directory:
        png = os.path.join(directory, "qubit.png")
        gif = os.path.join(directory, "qubit.gif")
        ea.render(frames, png)
        ea.render(frames[:2], gif)
        assert os.path.getsize(png) > 1024 and os.path.getsize(gif) > 1024
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(18, 10))
    try:
        ea.draw_frame(figure, frames, len(frames) - 1)
        assert "qubit cobordism" in figure._suptitle.get_text()
        assert "legacy" in figure._suptitle.get_text()
        titles = [axis.get_title() for axis in figure.axes]
        for token in ("residuals", "tau-hat", "Bloch", "|T|", "the whole"):
            assert any(token in title for title in titles), token
    finally:
        plt.close(figure)
    assert [name for name, _ in ea.panels_for(_run()["config"])][:6] == list(ea._QUBIT_PANEL_ORDER[:6])
    rows, columns = ea.DECLARED_PANEL_GRID
    assert len(ea._PANELS) == len(ea._QUBIT_PANELS) <= rows * columns
    assert {name for name, _ in ea._PANELS} == {name for name, _ in ea._QUBIT_PANELS}


# --------------------------------------------------------------------------- #
# the neutral mode is unchanged
# --------------------------------------------------------------------------- #
def test_the_neutral_mode_is_the_default_and_keeps_its_record():
    config = ea.build_config()
    assert config["inputs"] == ea.InputMode.NEUTRAL == ea.DECLARED_INPUTS
    assert config["betti_degrees"] == list(ea.DECLARED_BETTI_DEGREES)
    assert ea.NODE_FACTORIES == {ea.InputMode.QUBIT: ea.build_qubit_node}
    result = ea.drive(ea.build_config(size=4, steps=0), progress=False)
    assert result.inputs is None and len(result.frames) == 1
    frame = result.frames[0]
    document = frame.to_json()
    for name in ea.EmergenceFrame.QUBIT_CHANNELS:
        assert name not in document
        assert isinstance(getattr(frame, name), ea.Absent)
    assert frame.layout["highlight"] == []
    assert [name for name, _ in ea.panels_for(config)] == [name for name, _ in ea._PANELS]
    assert [name for name, _ in ea._PANELS[:14]] == [
        "objective", "layout", "dual_spatial", "dual_temporal", "clusters", "bands", "anchors",
        "transports", "statistics", "crossings", "mass", "spin", "betti", "verdict"]


# --------------------------------------------------------------------------- #
# the config and the CLI
# --------------------------------------------------------------------------- #
def test_config_refusals_by_name():
    with pytest.raises(ValueError, match="tau_a must lie in the upper half plane"):
        ea.build_config(inputs=ea.InputMode.QUBIT, tau_a=0.3 - 1.1j)
    with pytest.raises(ValueError, match="tau_b must lie in the upper half plane"):
        ea.build_config(inputs=ea.InputMode.QUBIT, tau_b=0.5)
    with pytest.raises(ValueError, match="grid must be at least 3"):
        ea.build_config(inputs=ea.InputMode.QUBIT, grid=2)
    with pytest.raises(ValueError, match="input weight must be a positive finite number"):
        ea.build_config(inputs=ea.InputMode.QUBIT, input_weight=0.0)
    with pytest.raises(ValueError, match="J must be finite"):
        ea.build_config(inputs=ea.InputMode.QUBIT, coupling=float("nan"))
    with pytest.raises(ValueError, match="unknown input mode 'qbit'"):
        ea.build_config(inputs="qbit")
    config = ea.build_config(inputs=ea.InputMode.QUBIT, tau_a="0.3+1.1j", tau_b=[-0.2, 0.8])
    assert config["tau_a"] == [0.3, 1.1] and config["tau_b"] == [-0.2, 0.8]
    assert config["layers"] == ea.DECLARED_COLLAR_LAYERS == 1


def test_the_cli_parses_the_qubit_flags():
    parser = ea.build_parser()
    args = parser.parse_args(["run", "--inputs", "qubit", "--tau-a", "0.3+1.1j", "--tau-b=-0.2+0.8j",
                              "--grid", "4", "--J", "2.0", "--time", "0.1", "--input-weight", "1e3",
                              "--no-regge"])
    assert args.inputs == "qubit" and args.tau_a == TAU_A and args.tau_b == TAU_B
    assert args.grid == 4 and args.coupling == 2.0 and args.time == 0.1
    assert args.input_weight == 1e3 and args.regge is False
    defaults = parser.parse_args(["run"])
    assert defaults.inputs == ea.InputMode.NEUTRAL and defaults.regge is ea.DECLARED_REGGE
    assert defaults.tau_a == ea.DECLARED_TAU_A and defaults.grid == ea.DECLARED_GRID
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--inputs", "qbit"])
    with pytest.raises(argparse.ArgumentTypeError):
        ea._complex_argument("abc")


def test_main_writes_the_run_document():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "run.json")
        ea.main(["run", "--inputs", "qubit", "--tau-a", "0.3+1.1j", "--tau-b=-0.2+0.8j",
                 "--grid", "3", "--steps", "1", "--stage-two-iterations", "1",
                 "--tolerance", "1e-30", "--json", path, "--out", "", "--quiet"])
        with open(path) as handle:
            document = json.load(handle)
    assert document["config"]["inputs"] == "qubit" and document["config"]["grid"] == 3
    assert document["terminator"] in ea.Terminator.ALL
    assert document["inputs"]["objective"] == "legacy"
    assert len(document["frames"]) == 2
    for frame in document["frames"]:
        for name in ea.EmergenceFrame.QUBIT_CHANNELS:
            assert name in frame
        assert frame["blocks"][0]["read"]["tau"][1] > 0
