# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The bridge move: drawing the bulk between two boundary blocks (#960, T1 of
``docs/design/qubit_cobordism_spec.md``).

Two qubit tori (``SimplicialQubit.flat_torus``) are seeded into one
3-dimensional host as their own triangles with their lengths and no 3-cell
(``MultiCobordism.seed_from_surfaces``), each torus one input block. The bulk
is DRAWN: ``SurgicalCone.bridge`` creates a tetrahedron on existing vertices
split across the two blocks (1+3, 2+2, 3+1), gated by the manifold check and
nothing else, rolled back bit-exactly on refusal; ``draw_bridges`` repeats it
until every torus face has exactly one 3-cell on it and no other boundary face
exists, so the boundary of W is exactly the two tori. The drawn topology is
emergent: its Betti numbers and the monodromy of the whole's degree-1 zero
mode between the two markings are recorded, never prescribed.

Coverage:

* the seed holds each torus exactly (faces, edges, lengths, zero phases,
  disjoint id ranges, no 3-cell) and the blocks' own surfaces are the tori;
* the gate refuses a non-manifold bridge (a cell sharing only a vertex or only
  an edge with the drawing, a third cell on a face) and malformed input,
  leaving the complex unchanged;
* an accepted bridge round-trips bit-exactly, including after the cell's face
  lattice was materialized by a read, and LIFO with a second bridge;
* a drawing from a 3x3 and a 4x4 torus is gated (a manifold-with-boundary),
  chordless (each block's sub-complex on its vertex set is still exactly its
  torus), every cell straddles the blocks, no vertex is buried, the tori's
  lengths never move, and the Betti numbers and monodromy read are reported;
  completion itself is a strict expected failure carrying the finding that
  random gated bridging does not complete (``COMPLETION_FINDING``);
* the monodromy read is checked on a prism collar of known topology (identity,
  and the swap matrix under a re-marked far end);
* stage 1 never sees the bridge kind on an ordinary node, and a rebuild after
  a committed move keeps the uncovered torus faces.
"""
import math
import os

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def torus(tau, n):
    return obs.SimplicialQubit.flat_torus(tau, n, n)


def mapped_faces(q, ids):
    return sorted(tuple(sorted(ids[int(v)] for v in f)) for f in q.faces())


def mapped_edges(q, ids):
    return sorted(tuple(sorted((ids[int(i)], ids[int(j)]))) for i, j in q.edges())


def mapped_lengths(q, ids):
    return {tuple(sorted((ids[int(i)], ids[int(j)]))): l for (i, j), l in zip(q.edges(), q.lengths())}


def host_marking(q, ids):
    """The torus's marking as cycles of directed host steps (u -> v)."""
    edges = q.edges()

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = edges[int(e)]
            out.append((ids[int(i)], ids[int(j)]) if sign > 0 else (ids[int(j)], ids[int(i)]))
        return out

    return [cycle(q.cycle_A()), cycle(q.cycle_B())]


def tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices())) for s in st.getTopSimplices())


def registered(st, size):
    return sorted(
        tuple(sorted(v.getId() for v in s.getVertices()))
        for s in st.getSimplices() if len(s.getVertices()) == size
    )


def edge_geometry(st):
    out = {}
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        out[tuple(sorted((u, v)))] = (complex(e.getLength()), complex(e.getPhase()))
    return out


def vertex_ids(st):
    return sorted(v.getId() for v in st.getVertexList().toVector())


def boundary(st):
    return sorted(tuple(f) for f in st.getBoundary())


def surface_cells(st):
    """Registered triangles that are facets of no top cell: the surfaces' own
    cells. Facets of surviving top cells are lazily materialized bookkeeping a
    read re-creates identically, so a rollback leaves them as it found them."""
    covered = {tuple(sorted(set(cell) - {v})) for cell in tops(st) for v in cell}
    return [f for f in registered(st, 3) if f not in covered]


def state(st):
    return (tops(st), surface_cells(st), edge_geometry(st), vertex_ids(st))


def seeded(seed_value=0, einstein_hilbert=False):
    qa, qb = torus(TAU_A, 3), torus(TAU_B, 4)
    seed = MC.seed_from_surfaces([qa.spacetime(), qb.spacetime()])
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value,
              einstein_hilbert=einstein_hilbert)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    return qa, qb, seed, node


def block_split(cell, regions):
    return tuple(sum(1 for v in cell if v in region) for region in regions)


# --------------------------------------------------------------------------- #
# 1. the seed holds the tori exactly
# --------------------------------------------------------------------------- #
def test_seed_from_surfaces_holds_the_tori():
    qa, qb, seed, node = seeded()
    st = seed.host
    assert tops(st) == [], "the seed must hold no 3-cell"
    assert registered(st, 4) == []
    assert boundary(st) == []
    ids_a, ids_b = seed.vertex_ids
    assert sorted(ids_a.values()) == list(range(9))
    assert sorted(ids_b.values()) == list(range(9, 25))
    assert vertex_ids(st) == list(range(25))
    assert registered(st, 3) == sorted(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    geometry = edge_geometry(st)
    assert sorted(geometry) == sorted(mapped_edges(qa, ids_a) + mapped_edges(qb, ids_b))
    for q, ids in ((qa, ids_a), (qb, ids_b)):
        for edge, length in mapped_lengths(q, ids).items():
            assert geometry[edge] == (complex(length), 0j), edge
    # each torus is one input block, marked as a surface; its own surface is the torus
    assert len(node.inputs) == 2 and node.has_surface_inputs()
    for block, q, ids in zip(node.inputs, (qa, qb), (ids_a, ids_b)):
        assert block.surface
        assert sorted(block.vertices) == sorted(ids.values())
        surface = MC.block_surface(block, st)
        assert sorted(tuple(f) for f in surface.faces) == mapped_faces(q, ids)
        assert sorted(tuple(e) for e in surface.edges) == mapped_edges(q, ids)
    assert len(node.uncovered_input_faces()) == 18 + 32
    assert not node.bridge_phase_complete()
    with pytest.raises(ValueError):
        MC.seed_from_surfaces([])
    with pytest.raises(ValueError, match="differ in dimension"):
        MC.seed_from_surfaces([qa.spacetime(), MC.seed_simplex(3)])


# --------------------------------------------------------------------------- #
# 2. the gate refuses a non-manifold bridge and leaves the complex unchanged
# --------------------------------------------------------------------------- #
def test_bridge_gate_refuses_non_manifold_cells():
    qa, qb, seed, node = seeded()
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    faces_a = mapped_faces(qa, ids_a)
    b0 = ids_b[0]
    sc = cob.SurgicalCone(st)
    # malformed input, nothing applied
    for bad in ([0, 1, 2], [0, 0, 1, b0], [0, 1, 2, 999]):
        ok, reason = sc.bridge(bad)
        assert not ok, reason
    assert state(st) == state(seed.host) and sc.depth == 0
    before = state(st)
    face = faces_a[0]
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    assert sc.depth == 1 and sc.validate()[0]
    assert tops(st) == [tuple(sorted(face + (b0,)))]
    ok, reason = sc.bridge(list(face) + [b0])
    assert not ok and "already exists" in reason
    # a cell sharing only the vertex b0 with the drawing: b0's link is two
    # disjoint triangles, a pinch
    disjoint = next(f for f in faces_a if not set(f) & set(face))
    ok, reason = sc.bridge(list(disjoint) + [b0])
    assert not ok, "a cell sharing only a vertex is not a manifold"
    assert "pinch" in reason or "disconnected" in reason, reason
    # a cell sharing only an edge: that edge's link is two disjoint paths
    shared_edge = next(f for f in faces_a if len(set(f) & set(face)) == 2)
    other = ids_b[5]
    ok, reason = sc.bridge(list(shared_edge) + [other])
    assert not ok, "a cell sharing only an edge is not a manifold"
    assert sc.depth == 1 and tops(st) == [tuple(sorted(face + (b0,)))]
    # a third cell on one face: three cofaces
    b1 = next(v for (u, v) in [tuple(sorted(e)) for e in mapped_edges(qb, ids_b)] if u == b0)
    ok, reason = sc.bridge([face[0], face[1], b0, b1])
    assert ok, reason
    ok, reason = sc.bridge([face[0], face[1], b0, ids_b[7]])
    assert not ok and "cofaces" in reason, reason
    assert sc.depth == 2
    assert sc.rollbackAll() == 2
    assert state(st) == before, "the refused and rolled-back bridges must leave the complex unchanged"


# --------------------------------------------------------------------------- #
# 3. bit-exact rollback, also after the lattice was materialized, and LIFO
# --------------------------------------------------------------------------- #
def test_bridge_rollback_is_bit_exact():
    qa, qb, seed, node = seeded()
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    face = mapped_faces(qa, ids_a)[4]
    b0 = ids_b[3]
    before = state(st)
    sc = cob.SurgicalCone(st)
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    after = edge_geometry(st)
    fresh = sorted(set(after) - set(before[2]))
    assert fresh == sorted(tuple(sorted((v, b0))) for v in face), "a bridge wires exactly the missing edges"
    for edge in fresh:
        assert after[edge] == (1 + 0j, 0j), "auto-wired spacelike unit length, zero phase"
    for edge, geometry in before[2].items():
        assert after[edge] == geometry, "a bridge never touches an existing edge"
    # materialize the cell's face lattice through the reads the engine makes
    K = cob.ChainComplex.fromSpacetime(st)
    assert K.bettiNumbers() == [1, 0, 0, 0]  # the drawn 3-ball alone (fromSpacetime seeds from the top cells)
    assert sc.bettiNumbers() == [1, 0, 0, 0]
    assert sc.rollback()
    assert state(st) == before, "round trip after materialization"
    # LIFO with two bridges sharing a face, materialized in between
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    mid = state(st)
    b1 = next(v for (u, v) in [tuple(sorted(e)) for e in mapped_edges(qb, ids_b)] if u == b0)
    ok, reason = sc.bridge([face[0], face[1], b0, b1])
    assert ok, reason
    cob.ChainComplex.fromSpacetime(st).bettiNumbers()
    assert sc.rollback()
    assert state(st) == mid
    assert sc.rollback()
    assert state(st) == before
    assert registered(st, 3) == before[1], "no materialized face survives the full unwind"
    assert not sc.rollback()


# --------------------------------------------------------------------------- #
# 4. the drawing: gated, chordless, straddling; completion (the finding)
# --------------------------------------------------------------------------- #
COMPLETION_FINDING = (
    "random gated bridging does not complete on 3x3 vs 4x4 (nor 3x3 vs 3x3) tori: with the "
    "manifold gate and the no-buried-vertex condition, a depth-first search over frontier-"
    "adjacent split cells (close-most-faces first, random ties) stalls at 50-98 cells with "
    "~24 surface faces uncovered after 3e5 gated attempts per seed, 0/80 greedy restarts "
    "complete, and a memoized search over ~1e3 distinct partial drawings finds no completion. "
    "A finished drawing is a cellular correspondence between the two surfaces (a degree-one "
    "map T_B -> T_A^*), which local random coning does not discover; the spec's S3 process "
    "needs a coherence principle the spec does not provide (owner's decision, see the PR)."
)


def drawn(seed_value=0):
    qa, qb, seed, node = seeded(seed_value=seed_value)
    drawn_cells = node.draw_bridges(max_attempts=int(os.environ.get("TESSERA_BRIDGE_ATTEMPTS", "20000")))
    return qa, qb, seed, node, drawn_cells


def test_drawing_is_gated_chordless_and_straddling(whitney_default):
    qa, qb, seed, node, drawn_cells = drawn(int(os.environ.get("TESSERA_BRIDGE_SEED", "0")))
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    st = node.spacetime()
    assert drawn_cells == len(tops(st)) > 0
    # the gate was never bypassed: the drawn cells are a manifold-with-boundary
    ok, reason = cob.SurgicalCone(st).validate()
    assert ok, reason
    # no chord: each block's sub-complex on its vertex set is exactly its torus
    torus_edges = set(mapped_edges(qa, ids_a) + mapped_edges(qb, ids_b))
    for block, q, ids in zip(node.inputs, (qa, qb), (ids_a, ids_b)):
        surface = MC.block_surface(block, st)
        assert sorted(tuple(f) for f in surface.faces) == mapped_faces(q, ids)
        assert sorted(tuple(e) for e in surface.edges) == mapped_edges(q, ids)
        assert not any(set(cell) <= set(block.vertices) for cell in tops(st)), "no 3-cell inside a block"
    edges = edge_geometry(st)
    for (u, v), (length, phase) in edges.items():
        if any(u in r and v in r for r in regions):
            assert (u, v) in torus_edges, "a chord edge"
        else:
            assert length == 1 + 0j and phase == 0j, "bridge edges carry the auto-wired length"
    # every cell straddles the two blocks with one of the three splits
    splits = {block_split(cell, regions) for cell in tops(st)}
    assert splits <= {(1, 3), (2, 2), (3, 1)}, splits
    # every vertex is still a boundary vertex (none was buried)
    covered = boundary(st)
    for v in range(25):
        assert any(v in f for f in covered) or all(v not in c for c in tops(st))
    # the tori's own lengths never moved
    for q, ids in ((qa, ids_a), (qb, ids_b)):
        for edge, length in mapped_lengths(q, ids).items():
            assert edges[edge] == (complex(length), 0j)
    # every boundary facet is a torus face or a mixed face of the front
    torus_faces = set(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    for f in covered:
        assert f in torus_faces or block_split(f, regions) in {(1, 2), (2, 1)}
    betti = cob.ChainComplex.fromSpacetime(st).bettiNumbers()
    read = MC.monodromy(st, host_marking(qa, ids_a), host_marking(qb, ids_b))
    print(f"\n[bridge T1] drawn cells={drawn_cells} uncovered faces={len(node.uncovered_input_faces())} "
          f"complete={node.bridge_phase_complete()} betti(drawn region)={betti} "
          f"zero-mode rank={read.harmonic_rank} obstruction={read.obstruction!r}")
    assert read.betti == betti


@pytest.mark.xfail(strict=True, reason=COMPLETION_FINDING)
def test_drawing_completes_and_reports_topology(whitney_default):
    qa, qb, seed, node, drawn_cells = drawn(int(os.environ.get("TESSERA_BRIDGE_SEED", "0")))
    ids_a, ids_b = seed.vertex_ids
    st = node.spacetime()
    assert node.bridge_phase_complete(), f"{drawn_cells} cells drawn, {len(node.uncovered_input_faces())} faces uncovered"
    assert boundary(st) == sorted(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b)), "the boundary of W is T_A and T_B"
    assert node.uncovered_input_faces() == []
    betti = cob.ChainComplex.fromSpacetime(st).bettiNumbers()
    read = MC.monodromy(st, host_marking(qa, ids_a), host_marking(qb, ids_b))
    print(f"\n[bridge T1] complete: cells={drawn_cells} betti={betti} zero-mode rank={read.harmonic_rank} "
          f"monodromy={np.asarray(read.monodromy)} rounded={read.rounded} residual={read.rounding_residual}")
    assert read.betti == betti


# --------------------------------------------------------------------------- #
# 5. the monodromy read on a collar of known topology
# --------------------------------------------------------------------------- #
def test_monodromy_read_on_a_prism_collar(whitney_default):
    q = torus(TAU_A, 3)
    base = [[int(v) for v in f] for f in q.faces()]
    cells = T.Spacetime.prismCells(base, 1)
    st = T.Spacetime.fromCells(3, cells, 1.0, 0j)
    assert boundary(st) == sorted(mapped_faces(q, {v: v for v in range(9)}) + mapped_faces(q, {v: v + 9 for v in range(9)}))
    near = {v: v for v in range(9)}
    far = {v: v + 9 for v in range(9)}
    read = MC.monodromy(st, host_marking(q, near), host_marking(q, far))
    assert read.obstruction == "", read.obstruction
    assert read.betti == [1, 2, 1, 0]
    assert read.harmonic_rank == 2
    np.testing.assert_allclose(np.asarray(read.monodromy), np.eye(2), atol=1e-9)
    assert read.rounded == [[1, 0], [0, 1]]
    assert read.rounding_residual < 1e-9 and read.fit_residual < 1e-9
    # re-mark the far end with the cycles swapped: the read is the swap matrix
    a, b = host_marking(q, far)
    swapped = MC.monodromy(st, host_marking(q, near), [b, a])
    assert swapped.rounded == [[0, 1], [1, 0]] and swapped.rounding_residual < 1e-9
    # reverse a far cycle: the row changes sign
    reversed_b = [(v, u) for (u, v) in reversed(b)]
    flipped = MC.monodromy(st, host_marking(q, near), [a, reversed_b])
    assert flipped.rounded == [[1, 0], [0, -1]] and flipped.rounding_residual < 1e-9
    # a marking edge that is not an edge of the whole is refused by name
    bad = MC.monodromy(st, host_marking(q, near), [[(0, 14)], b])  # lo 0 -- hi 5: 5 is no neighbour of 0
    assert "not an edge of the whole" in bad.obstruction


# --------------------------------------------------------------------------- #
# 6. ordinary nodes never see the bridge kind; a rebuild keeps the tori
# --------------------------------------------------------------------------- #
def test_ordinary_nodes_are_untouched_and_rebuilds_keep_the_tori(whitney_default):
    ordinary = MC(MC.seed_simplex(3), [[1.0 + 0j]], [], degrees=[1], einstein_hilbert=False)
    ordinary.seed_inputs([0])
    assert not ordinary.has_surface_inputs()
    assert not ordinary.inputs[0].surface
    assert not ordinary.bridge_phase_complete()
    assert ordinary.uncovered_input_faces() == []
    with pytest.raises(RuntimeError, match="surface"):
        ordinary.draw_bridges()
    assert hasattr(MC.BuildAction, "BRIDGE")
    with pytest.raises(ValueError, match="not a vertex of the host"):
        ordinary.seed_inputs([[0, 99]])

    # Under the Einstein-Hilbert term the first cell raises the Regge
    # stationarity norm from zero (six boundary hinges with deficits), so the
    # bridge phase keeps nothing: the objective's verdict, recorded here.
    qa, qb, seed, node = seeded(seed_value=3, einstein_hilbert=True)
    ids_a, ids_b = seed.vertex_ids
    assert node.draw_bridges(max_cells=4, max_attempts=200) == 0
    assert tops(node.spacetime()) == []
    # With the term weighted out the objective is flat along the drawing and
    # cells are kept.
    node.set_regge_weight(0.0)
    assert node.draw_bridges(max_cells=4) == 4
    faces = [sorted(tuple(f) for f in MC.block_surface(b, node.spacetime()).faces) for b in node.inputs]
    assert faces == [mapped_faces(qa, ids_a), mapped_faces(qb, ids_b)]
    uncovered = sorted(tuple(f) for f in node.uncovered_input_faces())
    assert len(uncovered) == 50 - 4
    # A committed move rebuilds the complex from a snapshot; the uncovered torus
    # faces must ride along. refine_geometry commits a gated cone-in (a fresh
    # apex on a boundary facet) through exactly that rebuild.
    thresholds = MC.RefinementIndicators()
    thresholds.mesh_quality = 2.0  # a lower bound every mesh crosses: refine now
    node.set_refinement_thresholds(thresholds)
    assert node.refine_geometry(1) == 1
    st = node.spacetime()
    assert len(tops(st)) == 5 and len(vertex_ids(st)) == 26
    assert sorted(tuple(f) for f in node.uncovered_input_faces()) == uncovered
    assert set(uncovered) <= set(registered(st, 3)), "the uncovered torus faces are registered cells of the rebuilt host"
    assert [sorted(tuple(f) for f in MC.block_surface(b, st).faces) for b in node.inputs] == faces
    # Stage 1 with the Regge term restored: a cone-out that lowers the Regge
    # norm may commit. Removing a bridge cell that sat on a torus face is the
    # spec's dent (section 6): that face leaves the block's own surface (it is
    # no face of W any more) while the face's edges persist. Nothing else may
    # change on the surfaces.
    node.set_regge_weight(1.0)
    before = tops(st)
    node.run_stage1(max_steps=4, n_candidate_moves=8)
    st = node.spacetime()
    removed = set(before) - set(tops(st))
    dented = {f for cell in removed for f in
              (tuple(sorted(set(cell) - {v})) for v in cell) if f in set(faces[0] + faces[1])}
    after = [sorted(tuple(f) for f in MC.block_surface(b, st).faces) for b in node.inputs]
    for block_faces, block_after, q, ids in zip(faces, after, (qa, qb), (ids_a, ids_b)):
        lost = set(block_faces) - set(block_after)
        assert lost <= dented, f"a surface face vanished without a dent: {lost - dented}"
        assert set(block_after) <= set(block_faces), "no chord face"
        assert sorted(tuple(e) for e in MC.block_surface(node.inputs[faces.index(block_faces)], st).edges) == mapped_edges(q, ids)
    print(f"[bridge T1] stage 1 on a partial drawing with the Regge term: cells {len(before)} -> {len(tops(st))}, "
          f"removed={len(removed)} dented faces={len(dented)} lookahead={node.last_stage1_lookahead}")
