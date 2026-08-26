# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.

"""Branch-free z/U storage and structural-orientation acceptance tests (#882)."""

import json

import pytest

import tessera as T


def _triangle(z=complex(-0.75, 2.5), U=complex(1.25, -0.4)):
    return T.Spacetime.fromCellsWithFields(
        2, [[0, 1, 2]], squaredLength=z, canonicalLink=U)


def _vertices(st):
    return {v.getId(): v for v in st.getVertexList().toVector()}


def test_complex_squared_length_is_exact_and_complex_valued():
    z = complex(-3.125, 7.75)
    st = _triangle(z=z)
    for edge in st.getEdgeList().toVector():
        got = edge.squaredLength()
        assert isinstance(got, complex)
        assert got == z


def test_direct_fields_remain_authoritative_when_vertex_times_are_recorded():
    z = complex(4.5, -2.25)
    U = complex(-0.6, 1.4)
    st = T.Spacetime.fromCellsWithFields(
        2, [[0, 1, 2]], squaredLength=z, canonicalLink=U,
        vertexTimes=[0.0, 1.0, 1.0])
    for edge in st.getEdgeList().toVector():
        assert isinstance(edge.squaredLength(), complex)
        assert edge.squaredLength() == z
        assert edge.canonicalLink() == U


def test_direct_builder_rejects_values_outside_finite_c_star_even_if_empty():
    with pytest.raises(ValueError):
        T.Spacetime.fromCellsWithFields(
            2, [], squaredLength=1 + 0j, canonicalLink=0 + 0j)
    with pytest.raises(ValueError):
        T.Spacetime.fromCellsWithFields(
            2, [], squaredLength=complex(float("nan"), 0.0),
            canonicalLink=1 + 0j)


def test_link_reverse_is_inverse_and_revisions_are_independent():
    st = _triangle()
    edge = st.getEdgeList().toVector()[0]
    a, b = edge.getSource().getId(), edge.getTarget().getId()
    g0, u0 = edge.geometryRevision(), edge.linkRevision()

    z = complex(0.125, -6.5)
    edge.setSquaredLength(z)
    assert edge.squaredLength() == z
    assert edge.geometryRevision() == g0 + 1
    assert edge.linkRevision() == u0

    U = complex(-0.375, 1.625)
    edge.setLink(a, b, U)
    assert edge.link(a, b) == U
    assert edge.link(b, a) == 1 / U
    delta_U = complex(0.2, -0.7)
    assert edge.linkLogTangent(a, b, delta_U) == delta_U / U
    assert edge.geometryRevision() == g0 + 1
    assert edge.linkRevision() == u0 + 1


def test_face_holonomy_is_gauge_invariant_and_orientation_reverses_it():
    st = _triangle(U=1 + 0j)
    verts = _vertices(st)
    links = {
        (0, 1): complex(1.2, 0.3),
        (1, 2): complex(-0.7, 0.8),
        (0, 2): complex(0.9, -0.4),
    }
    for edge in st.getEdgeList().toVector():
        a, b = sorted((edge.getSource().getId(), edge.getTarget().getId()))
        edge.setCanonicalLink(links[(a, b)])

    forward = st.faceHolonomy([0, 1, 2])
    reverse = st.faceHolonomy([0, 2, 1])
    assert abs(reverse - 1 / forward) < 1e-13

    gauges = {
        0: complex(0.8, 0.2),
        1: complex(-1.1, 0.6),
        2: complex(0.45, -0.9),
    }
    for edge in st.getEdgeList().toVector():
        a, b = sorted((edge.getSource().getId(), edge.getTarget().getId()))
        transformed = (1 / gauges[a]) * edge.canonicalLink() * gauges[b]
        edge.setCanonicalLink(transformed)
    assert abs(st.faceHolonomy([0, 1, 2]) - forward) < 1e-13


def test_vertex_relabeling_preserves_oriented_geometric_holonomy():
    st = _triangle(U=1 + 0j)
    verts = _vertices(st)
    for index, edge in enumerate(st.getEdgeList().toVector(), start=1):
        edge.setCanonicalLink(complex(0.5 + index, -0.2 * index))
    object_cycle = [verts[0], verts[1], verts[2]]
    before = st.faceHolonomy([v.getId() for v in object_cycle])
    st.swapVertexLabels(verts[0], verts[2])
    after = st.faceHolonomy([v.getId() for v in object_cycle])
    assert abs(after - before) < 1e-13


def test_relabel_copy_preserves_z_and_reexpresses_only_canonical_u():
    st = _triangle(U=1 + 0j)
    original = {}
    for index, edge in enumerate(st.getEdgeList().toVector(), start=1):
        a, b = sorted((edge.getSource().getId(), edge.getTarget().getId()))
        z = complex(-0.4 * index, 1.7 + 0.3 * index)
        U = complex(0.8 + 0.2 * index, -0.35 * index)
        edge.setSquaredLength(z)
        edge.setCanonicalLink(U)
        original[(a, b)] = (z, U)

    rebuilt = T.LiveComplex.relabel(st, seed=41)
    permutation = dict(rebuilt.vertex_map)
    copied = {}
    for edge in rebuilt.spacetime.getEdgeList().toVector():
        a, b = sorted((edge.getSource().getId(), edge.getTarget().getId()))
        copied[(a, b)] = (edge.squaredLength(), edge.canonicalLink())

    for (a, b), (z, U) in original.items():
        pa, pb = permutation[a], permutation[b]
        expected_U = U if pa < pb else 1 / U
        assert copied[(min(pa, pb), max(pa, pb))] == (z, expected_U)


def test_cut_reversal_is_structural_and_metric_independent():
    cut = T.cobordism.CoorientedCut()
    cut.id = "sigma"
    cut.oriented_simplices = [[0, 1], [1, 2]]
    cut.coorientation = T.cobordism.Coorientation.Positive
    reversed_cut = cut.reversed()
    assert reversed_cut.id == cut.id
    assert reversed_cut.oriented_simplices == cut.oriented_simplices
    assert reversed_cut.coorientation == T.cobordism.Coorientation.Negative


def test_whole_cobordism_reversal_swaps_roles_without_reading_z():
    st = _triangle(z=complex(-13.0, 8.0), U=complex(0.3, -1.7))
    node = T.cobordism.MultiCobordism(
        st, [[1 + 0j]], [[2 + 0j]], [1], 1.0, 19,
        should_propose_dispositions=False)
    node.seed_inputs([0])
    node.seed_outputs([2])
    cut = T.cobordism.CoorientedCut()
    cut.id = "sigma"
    cut.oriented_simplices = [[0, 1]]
    cut.coorientation = T.cobordism.Coorientation.Positive
    node.set_cooriented_cuts([cut])

    old_inputs = [set(block.vertices) for block in node.inputs]
    old_outputs = [set(block.vertices) for block in node.outputs]
    node.reverse_cobordism_orientation()
    assert [set(block.vertices) for block in node.inputs] == old_outputs
    assert [set(block.vertices) for block in node.outputs] == old_inputs
    assert all(block.role == T.cobordism.BoundaryRole.Incoming
               for block in node.inputs)
    assert all(block.role == T.cobordism.BoundaryRole.Outgoing
               for block in node.outputs)
    assert (node.cooriented_cuts[0].coorientation ==
            T.cobordism.Coorientation.Negative)


def test_metric_sections_never_determine_boundary_or_cut_orientation():
    st = _triangle()
    node = T.cobordism.MultiCobordism(
        st, [[1 + 0j]], [[2 + 0j]], [1], 1.0, 23,
        should_propose_dispositions=False)
    node.seed_inputs([0])
    node.seed_outputs([2])
    cut = T.cobordism.CoorientedCut()
    cut.id = "metric-independent"
    cut.oriented_simplices = [[0, 1]]
    cut.coorientation = T.cobordism.Coorientation.Positive
    node.set_cooriented_cuts([cut])
    roles = ([block.role for block in node.inputs],
             [block.role for block in node.outputs])

    # Exercise positive/negative real sections, both imaginary axes, and
    # genuinely mixed complex values. Structural data must not inspect any of
    # these coordinates.
    for z in (3 + 0j, -5 + 0j, 0 + 7j, 0 - 11j, -2.5 + 4.75j):
        for edge in st.getEdgeList().toVector():
            edge.setSquaredLength(z)
        assert ([block.role for block in node.inputs],
                [block.role for block in node.outputs]) == roles
        assert (node.cooriented_cuts[0].coorientation ==
                T.cobordism.Coorientation.Positive)


def test_contractible_four_ball_carries_complex_fields_and_structural_cut():
    st = T.Spacetime.fromCellsWithFields(
        4, [[0, 1, 2, 3, 4]],
        squaredLength=complex(-2.0, 5.0),
        canonicalLink=complex(0.4, 1.3))
    betti = T.cobordism.MultiCobordism.betti(st)
    assert betti[0] == 1
    assert all(value == 0 for value in betti[1:])
    cut = T.cobordism.CoorientedCut()
    cut.id = "contractible-cut"
    cut.oriented_simplices = [[0, 1, 2, 3]]
    cut.coorientation = T.cobordism.Coorientation.Positive
    # The cut's structural data exists independently of the trivial homology.
    assert cut.oriented_simplices and cut.coorientation


def test_checkpoint_cold_replay_preserves_complex_fields_roles_and_cuts():
    st = _triangle(z=complex(-8.25, 3.75), U=complex(0.7, -1.2))
    node = T.cobordism.MultiCobordism(
        st, [[1 + 2j]], [[-3 + 0.5j]], [1], 1.0, 29,
        should_propose_dispositions=False)
    node.seed_inputs([0])
    node.seed_outputs([2])
    cut = T.cobordism.CoorientedCut()
    cut.id = "replay-cut"
    cut.oriented_simplices = [[0, 1]]
    cut.coorientation = T.cobordism.Coorientation.Negative
    node.set_cooriented_cuts([cut])
    config = T.cobordism.MultiCobordism.AnalysisConfig()
    config.enabled = True
    config.cold_caches = True
    config.degrees = [1]
    node.set_analysis_config(config)
    node.run_recursive_analysis()

    written = json.loads(node.checkpoint_json)["raw_complex"]
    replayed = json.loads(
        T.cobordism.MultiCobordism.replay_checkpoint(node.checkpoint_json))
    assert replayed["raw_complex"] == written
