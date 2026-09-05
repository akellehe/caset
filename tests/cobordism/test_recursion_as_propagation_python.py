# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Plumbing of the recursion-as-propagation chains (#943): rank-two Schmidt
fibers attach to an interaction node, a two-layer velocity chain records the
three bookkeepings against the exact N-block exponentials, and the algebraic
Euler chain's error is the discretization error (first order in Δt)."""
import importlib.util
import math
import pathlib

import numpy as np
import pytest

from tessera import cobordism as cob

HL = cob.HodgeLaplacian
SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cobordism" / "recursion_as_propagation.py"


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


@pytest.fixture(scope="module")
def rap():
    spec = importlib.util.spec_from_file_location("recursion_as_propagation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAlgebra:
    def test_velocity_is_the_generator_on_the_pair_frame(self, rap):
        rng = np.random.default_rng(0)
        pair = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        expected = (rap.bilinear_generator() @ pair.reshape(-1)).reshape(4, 4)
        np.testing.assert_allclose(rap.apply_b(pair), expected, atol=1e-12)

    def test_euler_chain_error_is_first_order_in_dt(self, rap):
        rng = np.random.default_rng(1)
        pair0 = np.outer(rap._unit(rng.normal(size=4) + 1j * rng.normal(size=4)),
                         rap._unit(rng.normal(size=4) + 1j * rng.normal(size=4)))
        errors = []
        for layers in (4, 8, 16):
            dt, pair = 0.2 / layers, pair0.copy()
            for _ in range(layers):
                pair = pair - 1j * dt * rap.apply_b(pair)
            exact = rap.exact_state(pair0, 0.2)
            errors.append(np.linalg.norm(pair - exact) / np.linalg.norm(exact))
        assert errors[0] > errors[1] > errors[2]
        assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.3)  # halving Δt halves the error

    def test_schmidt_fibers_reconstruct_the_pair(self, rap):
        rng = np.random.default_rng(2)
        pair = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        fa, fb, sigma = rap.schmidt_fibers(pair, rank=4)
        A, B = np.asarray(fa.images), np.asarray(fb.images)
        np.testing.assert_allclose(A @ B.T, pair, atol=1e-10)
        assert fa.degree == 0 and [list(c) for c in fa.cells] == [[0], [1], [2], [3]]


class TestChains:
    def test_two_layer_velocity_chain_records_every_bookkeeping(self, rap, whitney_default):
        args = rap.build_parser().parse_args(["--chain", "velocity", "--layers", "2", "--rounds", "1",
                                              "--stage1-steps", "1", "--candidates", "2", "--stage2-iters", "3",
                                              "--precone", "8"])
        record = rap.run(args)
        assert len(record["layers_record"]) == 2
        for layer in record["layers_record"]:
            assert 0.0 <= layer["leak"] <= 1.0 and layer["reversal_residual"] < 1e-8
            assert len(layer["input_schmidt"]) == 4 and len(layer["read_schmidt"]) == 4
            for key in ("distance_geometric_vs_exact", "distance_algebraic_euler_vs_exact",
                        "distance_geometric_vs_algebraic_euler"):
                assert math.isfinite(layer[key])
        assert set(record["checks"]) == {"every_layer_carried_its_velocity",
                                         "geometric_chain_tracks_exact_as_well_as_euler", "reversal_identity"}

    def test_extend_chain_continues_the_same_node(self, rap, whitney_default):
        args = rap.build_parser().parse_args(["--chain", "extend", "--layers", "2", "--rounds", "1",
                                              "--stage1-steps", "1", "--candidates", "2", "--stage2-iters", "3",
                                              "--precone", "8"])
        record = rap.run(args)
        assert record["layers_record"][0]["cone_ins"] is not None
        assert record["layers_record"][1]["cone_ins"] is None  # the same cobordism, continued
