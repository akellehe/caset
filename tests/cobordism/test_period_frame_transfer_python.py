# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The two-body transfer read in the blocks' period frames (#962, T3 of
``docs/design/qubit_cobordism_spec.md``, delta D3).

A qubit torus's PERIOD FRAME (``SimplicialQubit.period_frame``) is the basis
(f_A, f_B) of its harmonic space with periods (1, 0) and (0, 1) over its
marking: the harmonic basis times the inverse period matrix, an n_E x 2 real
matrix in the torus's edge order. Every harmonic form is P_A f_A + P_B f_B with
its own periods as coefficients, so the holomorphic form is P_A * F @ (1, tau):
the qubit |0> + tau|1> read as a 1-form, f_A <-> |0>, f_B <-> |1>. That is the
basis the two-body target chi of spec S5 is written in.

On the engine each input block may carry a FRAME (``MultiCobordism.BlockFrame``
via ``set_input_frame``): images Z and dual images Z^vee on the block's
attached fiber cells, stated by the caller at attachment (the torus's period
frame through the collar's id map) and held constant by the engine. With both
blocks framed, ``read_two_body``, ``two_body_residual`` and the analytic
two-body gradient read the transfer in the frames,
T = Z_A^vee.T @ A~_AB @ Z_B (``PencilSchur.transfer``, the whole's pencil
operator block between the frames), r_A x r_B - 2 x 2 for two qubits - and
``set_two_body_target`` checks chi's shape against the frames' ranks (against
the cell counts without frames). Without frames everything is bit-identical
to before: identity frames on the cells. The state fibers stay rank one; the
frames are separate data and nothing about a marking enters the relaxation.

The dual-frame contract found in the code: ``PencilSchur.transfer`` pairs one
fiber's dual images against the pencil operator applied to the other's images
by the TRANSPOSE and normalizes nothing (``PencilLayer.read_boundary_fiber``
supplies the dual band's images, which equal the band's own at U = 1). The
dual of a supplied frame is therefore its left partner under that pairing on
the block's own pencil, Z^vee = Z @ inv(B).T with B = Z.T @ M_1 @ Z the frame's
pairing under the Whitney mass matrix M_1 of the block's own surface, so that
Z^vee.T @ M_1 @ Z = I (``MultiCobordism.dual_frame`` / ``input_frame_dual``;
the whitepaper's canonical left frame at U = 1). Under it a change of frame
(g_A, g_B) sends T to g_A^-1 T g_B - the matrix of the operator block in the
frames' coordinates - which is what a target written in those coordinates
compares with; paired with the plain images instead, T would transform as a
bilinear form (g_A.T T g_B).

The fixture is T2's: collar nodes from two ``SimplicialQubit.flat_torus``
inputs (3x3 and 4x4; tau_A = 0.3 + 1.1i, tau_B = -0.2 + 0.8i), Whitney pencil
metric source, state fibers = holomorphic forms with the block's harmonic
contour, plus the degree-0 tetrahedral setup of
``test_two_body_cobordism_map_python.py`` for bit-identity.
"""
import hashlib
import itertools
import math
import warnings

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
FLOOR = 1e-24
# The identity-frame two-body residual against a fixed chi (rng 962, nE x nE)
# on the collar, recorded on the build before this change (origin/main fea4ed1;
# process-deterministic: two runs byte-identical). Asserted to rounding; the
# byte hashes of T and the gradient are in the docstring of (b).
PRECHANGE_RESIDUAL = {3: 0.9965969445169056, 4: 0.999399418478379}


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return obs.SimplicialQubit.flat_torus(tau, n, n)


def remarked(q, cycle_A, cycle_B):
    """The same torus (its Spacetime, its lengths) read over another marking."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return obs.SimplicialQubit(q.spacetime(), cycle_A, cycle_B)


def rescaled(q, factor):
    """The same torus with every length multiplied by ``factor`` (the section-2 constructor)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return obs.SimplicialQubit(list(q.vertices()), list(q.edges()), list(q.faces()),
                                   [factor * l for l in q.lengths()], list(q.cycle_A()), list(q.cycle_B()))


def reversed_cycle(cycle):
    return [(e, -s) for e, s in reversed(list(cycle))]


def period(F, cycle):
    """The period of each column of ``F`` over a (edge index, sign) cycle."""
    return sum(sign * F[int(e)] for e, sign in cycle)


def state_fiber(q, ids, contour_on):
    f = cob.BoundaryFiber()
    f.degree = 1
    f.cells = [sorted((ids[int(i)], ids[int(j)])) for i, j in q.edges()]
    f.images = np.asarray(q.holomorphic_form()).reshape(-1, 1)
    f.contour = cob.PencilLayer.harmonic_contour(cob.PencilLayer.assemble([contour_on]), 1)
    return f


def collar(n, attach=(0, 1), einstein_hilbert=False, real_squared_lengths_only=False, weight=None, seed_value=0):
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), 1)
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value,
              einstein_hilbert=einstein_hilbert, real_squared_lengths_only=real_squared_lengths_only)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    if weight is not None:
        node.set_input_residual_weight(weight)
    for i in attach:
        q, ids = (qa, qb)[i], seed.vertex_ids[i]
        f = state_fiber(q, ids, q.spacetime())
        node.attach_input_fiber(i, f, f.cells)
    return qa, qb, seed, node


def period_frame(q):
    return np.asarray(q.period_frame()).astype(complex)


def frame_on(node, index, q):
    """(cells, Z, Z^vee): the torus's period frame on the block's attached cells
    (the torus's edge order carried through the id map) with its dual on the
    block's own pencil."""
    cells = [list(c) for c in node.inputs[index].fiber.cells]
    Z = period_frame(q)
    return cells, Z, np.asarray(node.input_frame_dual(index, Z))


def set_frames(node, qa, qb):
    frames = []
    for index, q in enumerate((qa, qb)):
        cells, Z, Zd = frame_on(node, index, q)
        node.set_input_frame(index, cells, Z, Zd)
        frames.append((cells, Z, Zd))
    return frames


def own_mass_matrix(node, index):
    """The Whitney mass matrix M_1 of the block's OWN surface on its fiber cells."""
    own = MC.block_surface_subcomplex(node.inputs[index], node.spacetime())
    assembled = cob.PencilLayer.assemble([own])
    idx = cob.PencilLayer.indices_of(assembled, 1, node.inputs[index].fiber.cells)
    return np.asarray(cob.PencilLayer.pencil(assembled, 1).B)[np.ix_(idx, idx)]


def identity_transfer(st, degree, cells_a, cells_b):
    """The transfer with identity frames through PencilLayer alone (what the
    engine read before frames existed)."""
    assembled = cob.PencilLayer.assemble([st])
    fa, fb = cob.BoundaryFiber(), cob.BoundaryFiber()
    for f, cells in ((fa, cells_a), (fb, cells_b)):
        f.degree = degree
        f.cells = [list(c) for c in cells]
        f.images = np.eye(len(cells), dtype=complex)
        f.dualImages = np.eye(len(cells), dtype=complex)
    return cob.PencilLayer.transfer(assembled, degree, fa, fb)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=complex)).tobytes()).hexdigest()[:16]


def fixed_chi(nE):
    rng = np.random.default_rng(962)
    return rng.normal(size=(nE, nE)) + 1j * rng.normal(size=(nE, nE))


def spin_half_chi(psi, phi):
    """chi of spec S5 for two spin-1/2: H_int = hbar J (s1+ s2- + s1- s2+),
    chi = (s- psi)(s+ phi)^T + (s+ psi)(s- phi)^T in the |0>, |1> bases."""
    lowering = np.array([[0, 0], [1, 0]], dtype=complex)  # s-|0> = |1>
    raising = lowering.T
    return np.outer(lowering @ psi, raising @ phi) + np.outer(raising @ psi, lowering @ phi)


def projective_leak(T, chi):
    tt = np.linalg.norm(T) ** 2
    return 1.0 - abs(np.vdot(T, chi)) ** 2 / (tt * np.linalg.norm(chi) ** 2)


def holomorphic(packed):
    """dF from the packed (2 Re dF, -2 Im dF)."""
    packed = np.asarray(packed)
    return 0.5 * (packed.real - 1j * packed.imag)


def squared_lengths(st):
    return np.array([complex(e.getLength()) ** 2 for e in st.getEdgeList().toVector()])


def euler_defect(st, gradient):
    """|sum_e s_e dF_e| relative to the gradient and length scales: zero when
    the residual is invariant under a common scale of the squared lengths."""
    s = squared_lengths(st)
    g = np.asarray(gradient)
    return abs(np.sum(s * holomorphic(g))) / (np.abs(g).max() * np.abs(s).max())


def scale_all(st, factor):
    for e in st.getEdgeList().toVector():
        e.setLength(complex(e.getLength()) * factor)


def residuals(node):
    return [node.fiber_residual_for_input_block(i) for i in range(len(node.inputs))]


def spacelike_real(st):
    s = squared_lengths(st)
    return bool(np.all(s.real > 0) and np.all(np.abs(s.imag) < 1e-15))


# degree-0 tetrahedral setup of test_two_body_cobordism_map_python.py
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])


def flip_flop(psi, phi):
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return np.outer(D @ psi, D.T @ phi) + np.outer(D.T @ psi, D @ phi)


def degree0_fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = [[0], [1], [2], [3]]
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def interaction_node(psi, phi, seed=0):
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=seed, precone=8, einstein_hilbert=False)
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(node.spacetime()).kSimplexVertices(3)]
    a, b = next((x, y) for x, y in itertools.combinations(tets, 2) if not set(x) & set(y))
    node.seed_inputs([0, 1])
    node.attach_input_fiber(0, degree0_fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, degree0_fiber(phi), [[v] for v in b])
    node.set_two_body_target(flip_flop(psi, phi))
    node.use_fiber_residuals(True)
    return node, a, b


# --------------------------------------------------------------------------- #
# (a) the period frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_period_frame(n, whitney_default):
    for tau in (TAU_A, TAU_B):
        q = torus(tau, n)
        F = np.asarray(q.period_frame())
        assert F.dtype.kind == "f" and F.shape == (len(q.edges()), 2), "real, one row per edge, two columns"
        # the columns have periods (1, 0) and (0, 1) over the marking
        periods = np.array([[period(F[:, c], cycle) for cycle in (q.cycle_A(), q.cycle_B())] for c in range(2)])
        assert np.abs(periods - np.eye(2)).max() < 1e-13, periods
        assert not q.marking_swapped()
        # the frame spans the harmonic space: F = H times a 2 x 2 matrix
        H = np.asarray(q.harmonic_basis())
        assert np.abs(F - H @ np.linalg.lstsq(H, F, rcond=None)[0]).max() < 1e-13
        # the holomorphic form is the column combination (P_A, P_B), i.e. P_A (1, tau)
        omega = np.asarray(q.holomorphic_form())
        PA, PB = q.periods()
        assert np.abs(omega - F @ np.array([PA, PB])).max() < 1e-14 * np.abs(omega).max()
        assert np.abs(omega / PA - F @ np.array([1.0, q.tau()])).max() < 1e-13
        # scale invariance carries over from tau: a common scale of the lengths leaves the frame
        scaled = rescaled(q, 2.5)
        assert abs(scaled.tau() - q.tau()) < 1e-12
        assert np.abs(np.asarray(scaled.period_frame()) - F).max() < 1e-12
        # remarking: swapping the cycles swaps the columns, reversing a cycle negates its column
        swapped = remarked(q, q.cycle_B(), q.cycle_A())
        assert np.abs(np.asarray(swapped.period_frame()) - F[:, [1, 0]]).max() < 1e-12
        flipped = remarked(q, q.cycle_A(), reversed_cycle(q.cycle_B()))
        assert np.abs(np.asarray(flipped.period_frame()) - F * np.array([1.0, -1.0])).max() < 1e-12
        print(f"\n[T3] {n}x{n} tau={tau}: frame periods {periods.tolist()}, |omega - F(P_A,P_B)| "
              f"{np.abs(omega - F @ np.array([PA, PB])).max():.1e}, |omega/P_A - F(1,tau)| "
              f"{np.abs(omega / PA - F @ np.array([1.0, q.tau()])).max():.1e}")


# --------------------------------------------------------------------------- #
# (b) without frames: bit-identical to before
# --------------------------------------------------------------------------- #
def framed_reads(node):
    read = node.read_two_body()
    g_l, g_p = node.two_body_residual_gradient()
    return read, np.asarray(read.transfer), node.two_body_residual(), np.asarray(g_l), np.asarray(g_p)


@pytest.mark.parametrize("n", [3, 4])
def test_without_frames_the_collar_reads_as_before(n, whitney_default):
    """No frames: the transfer is the identity-frame block on the tori's edges
    (nE x nE), its residual against the fixed chi is the pre-change value, and
    explicit identity frames give the same bytes. Recorded on this machine
    against the build before this change (OMP_NUM_THREADS=8): T bytes
    sha256[:16] cbe38b29eef48f33 (3x3), 39b9e9fb64ba1f14 (4x4); gradient bytes
    9d02b9d5a15a6b55 (3x3), 64402542aa6e42f9 (4x4); residuals
    0.9965969445169056 and 0.999399418478379 - all equal on the framed build."""
    qa, qb, seed, node = collar(n)
    nE = len(qa.edges())
    node.set_two_body_target(fixed_chi(nE), True)
    read0, T0, r0, g0, p0 = framed_reads(node)
    assert not read0.in_frames and T0.shape == (nE, nE) and p0.size == 0
    assert r0 == pytest.approx(PRECHANGE_RESIDUAL[n], rel=1e-12, abs=0)
    assert read0.residual == r0
    # the identity-frame transfer through PencilLayer alone: the same bytes
    replica = identity_transfer(node.spacetime(), 1, node.inputs[0].fiber.cells, node.inputs[1].fiber.cells)
    assert np.array_equal(np.asarray(replica.forward), T0)
    # explicit identity frames on the cells: bit-identical to no frames
    for index in range(2):
        cells = [list(c) for c in node.inputs[index].fiber.cells]
        node.set_input_frame(index, cells, np.eye(nE, dtype=complex), np.eye(nE, dtype=complex))
        assert node.inputs[index].frame is not None and node.input_frame(index).rank() == nE
    read1, T1, r1, g1, _ = framed_reads(node)
    assert read1.in_frames
    assert np.array_equal(T1, T0) and r1 == r0 and np.array_equal(g1, g0)
    assert read1.singular_values == read0.singular_values and read1.reversal_residual == read0.reversal_residual
    # and back
    node.clear_input_frame(0)
    node.clear_input_frame(1)
    assert node.inputs[0].frame is None and node.input_frame(1) is None
    read2, T2, r2, g2, _ = framed_reads(node)
    assert not read2.in_frames and np.array_equal(T2, T0) and r2 == r0 and np.array_equal(g2, g0)
    print(f"\n[T3] {n}x{n} no frames: T {T0.shape} sha {sha(T0)}, residual {r0!r}, gradient sha {sha(g0)} "
          f"(|g|max {np.abs(g0).max():.6e}); explicit identity frames bit-identical")


def test_without_frames_the_degree0_node_reads_as_before(whitney_default):
    """The degree-0 tetrahedral setup of the existing two-body test (grown by
    cone-ins, so a seed labels an attempt): no frames equals the PencilLayer
    replica byte for byte, and explicit identity frames equal no frames."""
    rng = np.random.default_rng(1)
    psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
    node, a, b = interaction_node(psi, phi)
    read0, T0, r0, g0, p0 = framed_reads(node)
    assert not read0.in_frames and T0.shape == (4, 4) and p0.size == len(g0)
    replica = identity_transfer(node.spacetime(), 0, [[v] for v in a], [[v] for v in b])
    assert np.array_equal(np.asarray(replica.forward), T0)
    assert r0 == pytest.approx(projective_leak(T0, flip_flop(psi, phi)), rel=1e-12)
    for index, cells in enumerate((a, b)):
        node.set_input_frame(index, [[v] for v in cells], np.eye(4, dtype=complex), np.eye(4, dtype=complex))
    read1, T1, r1, g1, p1 = framed_reads(node)
    assert read1.in_frames and np.array_equal(T1, T0) and r1 == r0
    assert np.array_equal(g1, g0) and np.array_equal(p1, p0)
    print(f"\n[T3] degree 0: T sha {sha(T0)}, residual {r0!r}, gradient sha {sha(g0)}; identity frames bit-identical")


# --------------------------------------------------------------------------- #
# (c) the transfer in the period frames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_transfer_in_the_period_frames(n, whitney_default):
    qa, qb, seed, node = collar(n)
    nE = len(qa.edges())
    node.set_two_body_target(fixed_chi(nE), True)
    T_full = np.asarray(node.read_two_body().transfer)
    (cells_a, Za, Zda), (cells_b, Zb, Zdb) = set_frames(node, qa, qb)
    # a target of the old shape no longer fits: the frames give 2x2
    with pytest.raises(ValueError, match="attached frames give 2x2"):
        node.set_two_body_target(fixed_chi(nE), True)
    chi = spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state()))
    node.set_two_body_target(chi, True)
    read = node.read_two_body()
    T = np.asarray(read.transfer)
    assert read.in_frames and T.shape == (2, 2) and np.asarray(read.choi_state).shape == (4,)
    assert read.reversal_residual < 1e-8
    assert len(read.singular_values) == 2 and read.schmidt_rank >= 1
    assert read.residual == node.two_body_residual() and 0.0 <= read.residual <= 1.0
    # the framed transfer is the identity-frame block contracted with the frames
    assert np.abs(T - Zda.T @ T_full @ Zb).max() < 1e-13 * np.abs(T).max()
    # the frames are real on real tori, so the transfer is real
    assert np.abs(T.imag).max() < 1e-14 * np.abs(T).max()
    # swapping torus B's marking (A <-> B) permutes the columns; reversing a
    # cycle flips the sign of its column; on torus A the same acts on the rows
    swapped_b = remarked(qb, qb.cycle_B(), qb.cycle_A())
    cells, Z, Zd = frame_on(node, 1, swapped_b)
    assert np.abs(Z - Zb[:, [1, 0]]).max() < 1e-12
    node.set_input_frame(1, cells, Z, Zd)
    T_swap = np.asarray(node.read_two_body().transfer)
    assert np.abs(T_swap - T[:, [1, 0]]).max() < 1e-12 * np.abs(T).max()
    flipped_b = remarked(qb, qb.cycle_A(), reversed_cycle(qb.cycle_B()))
    cells, Z, Zd = frame_on(node, 1, flipped_b)
    node.set_input_frame(1, cells, Z, Zd)
    T_flip = np.asarray(node.read_two_body().transfer)
    assert np.abs(T_flip - T * np.array([1.0, -1.0])).max() < 1e-12 * np.abs(T).max()
    node.set_input_frame(1, cells_b, Zb, Zdb)
    swapped_a = remarked(qa, qa.cycle_B(), qa.cycle_A())
    cells, Z, Zd = frame_on(node, 0, swapped_a)
    node.set_input_frame(0, cells, Z, Zd)
    T_swap_a = np.asarray(node.read_two_body().transfer)
    assert np.abs(T_swap_a - T[[1, 0], :]).max() < 1e-12 * np.abs(T).max()
    flipped_a = remarked(qa, reversed_cycle(qa.cycle_A()), qa.cycle_B())
    cells, Z, Zd = frame_on(node, 0, flipped_a)
    node.set_input_frame(0, cells, Z, Zd)
    T_flip_a = np.asarray(node.read_two_body().transfer)
    assert np.abs(T_flip_a - T * np.array([[-1.0], [1.0]])).max() < 1e-12 * np.abs(T).max()
    node.set_input_frame(0, cells_a, Za, Zda)
    assert np.array_equal(np.asarray(node.read_two_body().transfer), T)
    # a common scale of all lengths (frames held) scales T by one power of the
    # length and leaves the projective residual against any chi unchanged
    residual_before = node.two_body_residual()
    scale_all(node.spacetime(), 1.7)
    T_scaled = np.asarray(node.read_two_body().transfer)
    ratio = T_scaled / T
    assert np.abs(ratio - ratio[0, 0]).max() < 1e-10 * abs(ratio[0, 0])
    assert node.two_body_residual() == pytest.approx(residual_before, rel=1e-12, abs=1e-15)
    rng = np.random.default_rng(3)
    for _ in range(3):
        any_chi = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        scale_all(node.spacetime(), 1.0 / 1.7)
        node.set_two_body_target(any_chi, True)
        unscaled = node.two_body_residual()
        scale_all(node.spacetime(), 1.7)
        assert node.two_body_residual() == pytest.approx(unscaled, rel=1e-12, abs=1e-15)
    scale_all(node.spacetime(), 1.0 / 1.7)
    # what T is on the collar: recorded, not ruled (spec C3 is a prediction)
    off_diagonal = max(abs(T[0, 1]), abs(T[1, 0])) / max(abs(T[0, 0]), abs(T[1, 1]))
    sv = read.singular_values
    print(f"\n[T3] {n}x{n} period frames: T = {np.round(T.real, 8).tolist()}, reversal residual "
          f"{read.reversal_residual:.2e}, Schmidt spectrum {sv} (sigma2/sigma1 {sv[1] / sv[0]:.3e}, rank "
          f"{read.schmidt_rank}), off-diagonal/diagonal {off_diagonal:.3f} (diagonal: {off_diagonal < 1e-8}), "
          f"scale x1.7 -> T x {ratio[0, 0].real:.10f} (1/1.7 = {1 / 1.7:.10f}); residual against chi(S5) "
          f"{read.residual:.6e}")


# --------------------------------------------------------------------------- #
# (d) the dual-frame contract and the refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_dual_frame_contract(n, whitney_default):
    qa, qb, seed, node = collar(n)
    for index, q in enumerate((qa, qb)):
        cells, Z, Zd = frame_on(node, index, q)
        M = own_mass_matrix(node, index)
        pairing = Z.T @ M @ Z
        assert np.abs(Zd.T @ M @ Z - np.eye(2)).max() < 1e-14, "the contract Z^vee.T M_1 Z = I on the block's own pencil"
        assert np.abs(Zd - Z @ np.linalg.inv(pairing).T).max() < 1e-13 * np.abs(Zd).max()
        # the same dual from the static builder on the standalone torus (its own ids, the same edge order)
        own_cells = [sorted((int(i), int(j))) for i, j in q.edges()]
        alone = np.asarray(MC.dual_frame(q.spacetime(), 1, own_cells, Z))
        assert np.abs(alone - Zd).max() < 1e-13 * np.abs(Zd).max()
        # the pairing of a flat torus's period frame is the continuum Gram of
        # its unit-period forms, Im(tau) * [[1 + a^2/b^2, -a/b^2], [-a/b^2, 1/b^2]]
        t1, t2 = q.tau().real, q.tau().imag
        gram = t2 * np.array([[1 + t1 ** 2 / t2 ** 2, -t1 / t2 ** 2], [-t1 / t2 ** 2, 1 / t2 ** 2]])
        assert np.abs(pairing.real - gram).max() < 1e-12
        print(f"\n[T3] {n}x{n} block {index}: |Z^vee.T M Z - I| = {np.abs(Zd.T @ M @ Z - np.eye(2)).max():.2e}, "
              f"pairing {np.round(pairing.real, 6).tolist()}")


def test_frame_refusals(whitney_default):
    qa, qb, seed, node = collar(3)
    (cells_a, Za, Zda), (cells_b, Zb, Zdb) = [frame_on(node, i, q) for i, q in enumerate((qa, qb))]
    with pytest.raises(IndexError):
        node.set_input_frame(2, cells_a, Za, Zda)
    with pytest.raises(ValueError, match="is not input block 0's attached fiber cell"):
        node.set_input_frame(0, cells_b, Zb, Zdb)  # the other torus's cells
    with pytest.raises(ValueError, match="is not input block 0's attached fiber cell"):
        node.set_input_frame(0, list(reversed(cells_a)), Za, Zda)  # the right cells in the wrong order
    with pytest.raises(ValueError, match="names 26 cells but input block 0's fiber is attached to 27"):
        node.set_input_frame(0, cells_a[:-1], Za[:-1], Zda[:-1])
    with pytest.raises(ValueError, match="one row per cell"):
        node.set_input_frame(0, cells_a, Za[:-1], Zda)
    with pytest.raises(ValueError, match="share the rank"):
        node.set_input_frame(0, cells_a, Za, np.hstack([Zda, Zda[:, :1]]))
    with pytest.raises(ValueError, match="no columns"):
        node.set_input_frame(0, cells_a, Za[:, :0], Zda[:, :0])
    bad = Za.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        node.set_input_frame(0, cells_a, bad, Zda)
    assert node.inputs[0].frame is None
    # a block without an attached fiber has no cells to frame
    _, _, _, half = collar(3, attach=(0,))
    with pytest.raises(RuntimeError, match="carries no attached fiber"):
        half.set_input_frame(1, cells_b, Zb, Zdb)
    with pytest.raises(RuntimeError, match="carries no attached fiber"):
        half.input_frame_dual(1, Zb)
    # the dual builder's refusals
    with pytest.raises(ValueError, match="null complex"):
        MC.dual_frame(None, 1, cells_a, Za)
    with pytest.raises(ValueError, match="is not a degree-1 cell of the assembled complex"):
        MC.dual_frame(qa.spacetime(), 1, [[0, 5]] + [sorted((int(i), int(j))) for i, j in qa.edges()][1:], Za)
    with pytest.raises(ValueError, match="outside the complex's dimension"):
        MC.dual_frame(qa.spacetime(), 3, [sorted((int(i), int(j))) for i, j in qa.edges()], Za)
    with pytest.raises(ValueError, match="one image row per cell"):
        MC.dual_frame(qa.spacetime(), 1, [sorted((int(i), int(j))) for i, j in qa.edges()], Za[:-1])
    with pytest.raises(RuntimeError, match="singular"):
        node.input_frame_dual(0, np.hstack([Za[:, :1], Za[:, :1]]))  # an isotropic (rank-deficient) frame
    # one frame is not a reading: both blocks or neither
    node.set_input_frame(0, cells_a, Za, Zda)
    with pytest.raises(RuntimeError, match="only when both"):
        node.read_two_body()
    node.set_two_body_target(np.eye(2, dtype=complex))  # one frame set: the shape is settled at read time
    with pytest.raises(RuntimeError, match="only when both"):
        node.two_body_residual()
    with pytest.raises(RuntimeError, match="only when both"):
        node.two_body_residual_gradient()
    node.set_input_frame(1, cells_b, Zb, Zdb)
    assert node.read_two_body().transfer.shape == (2, 2)
    # a target set against the frames refuses the cells' shape and vice versa
    with pytest.raises(ValueError, match="the target is 27x27 but the attached frames give 2x2"):
        node.set_two_body_target(fixed_chi(27))
    node.clear_input_frame(0)
    node.clear_input_frame(1)
    with pytest.raises(ValueError, match="the target is 2x2 but the attached cells give 27x27"):
        node.set_two_body_target(np.eye(2, dtype=complex))
    # a target set before the frames is caught at read time by name
    node.set_two_body_target(fixed_chi(27))
    node.set_input_frame(0, cells_a, Za, Zda)
    node.set_input_frame(1, cells_b, Zb, Zdb)
    with pytest.raises(RuntimeError, match="the two-body target is 27x27 but the attached frames give 2x2"):
        node.two_body_residual()
    # re-attaching a fiber clears the block's frame (its rows were the previous attachment's)
    f = state_fiber(qa, seed.vertex_ids[0], qa.spacetime())
    node.attach_input_fiber(0, f, f.cells)
    assert node.inputs[0].frame is None and node.inputs[1].frame is not None


# --------------------------------------------------------------------------- #
# (e) chi of spec S5 for two spin-1/2 and the analytic gradient
# --------------------------------------------------------------------------- #
def test_spin_half_target_and_gradient(whitney_default):
    qa, qb, seed, node = collar(3)
    set_frames(node, qa, qb)
    psi, phi = np.asarray(qa.state()), np.asarray(qb.state())
    chi = spin_half_chi(psi, phi)
    assert chi.shape == (2, 2) and np.linalg.matrix_rank(chi, tol=1e-10) == 2
    node.set_two_body_target(chi, True)
    st = node.spacetime()
    residual = node.two_body_residual()
    assert 0.0 <= residual <= 1.0
    read = node.read_two_body()
    assert read.residual == residual and read.in_frames
    assert residual == pytest.approx(projective_leak(np.asarray(read.transfer), chi), rel=1e-12)
    # r_U = the two block residuals (at their floor) + the two-body residual
    assert node.r_u(st) == pytest.approx(sum(residuals(node)) + residual, rel=1e-12)
    g_l, g_p = node.two_body_residual_gradient()
    g = np.asarray(g_l)
    assert np.asarray(g_p).size == 0, "degree 1 carries no phase gradient"
    assert np.abs(g).max() > 0
    # Euler identity: the residual is invariant under a common scale of the squared lengths
    defect = euler_defect(st, g)
    assert defect < 1e-10, f"Euler identity violated: {defect:.3e}"
    # the analytic ascent of r_U includes it (the block terms vanish at the flat point)
    total_l, _ = node.fiber_mode_ascent()
    assert np.abs(np.asarray(total_l) - g).max() < 1e-10 * np.abs(g).max()
    # one central-difference probe on the edge of largest sensitivity (a sanity
    # check of this test only; the engine never uses a finite difference)
    edges = st.getEdgeList().toVector()
    i = int(np.argmax(np.abs(g)))
    e, h = edges[i], 1e-6
    s0 = complex(e.getLength()) ** 2
    e.setLength(np.sqrt(s0 + h))
    plus = node.two_body_residual()
    e.setLength(np.sqrt(s0 - h))
    minus = node.two_body_residual()
    e.setLength(np.sqrt(s0))
    fd = (plus - minus) / (2 * h)
    analytic = g[i].real  # d/dRe s of the packed gradient
    assert fd != 0 and np.sign(fd) == np.sign(analytic)
    assert abs(fd - analytic) < 1e-5 * abs(analytic)
    key = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
    print(f"\n[T3] chi(S5) for tau_A, tau_B: singular values {np.linalg.svd(chi, compute_uv=False)}; residual "
          f"{residual:.6e}; Euler defect {defect:.2e}; edge {key}: analytic {analytic:.6e} central difference "
          f"{fd:.6e}; |g|max {np.abs(g).max():.3e}")


# --------------------------------------------------------------------------- #
# (f) one stage-2 pass with fibers, frames and chi
# --------------------------------------------------------------------------- #
def test_stage2_pass_with_frames(whitney_default):
    qa, qb, seed, node = collar(3, einstein_hilbert=True, real_squared_lengths_only=True, weight=1e6)
    set_frames(node, qa, qb)
    chi = spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state()))
    node.set_two_body_target(chi, True)
    st = node.spacetime()
    assert spacelike_real(st)
    blocks_before = residuals(node)
    two_body_before = node.two_body_residual()
    objective_before = node.objective()
    terms_before = node.objective_terms()
    trace = node.run_stage2(beta=1.0, max_iters=20, tolerance=1e-15)
    st = node.spacetime()
    blocks_after = residuals(node)
    two_body_after = node.two_body_residual()
    terms_after = node.objective_terms()
    read = node.read_two_body()
    assert len(trace) > 1 and trace[-1] < objective_before
    assert spacelike_real(st)
    assert read.in_frames and np.asarray(read.transfer).shape == (2, 2)
    assert two_body_after <= two_body_before, f"two-body residual rose: {two_body_before:.6e} -> {two_body_after:.6e}"
    for index in range(2):
        assert blocks_after[index] < 1e-5, f"block {index} left its floor: {blocks_after[index]:.3e}"
    assert node.inputs[0].frame is not None and node.inputs[1].frame is not None, "frames are held constant"
    print(f"\n[T3] stage 2 at weight 1e6 ({len(trace) - 1} accepted steps): objective {objective_before:.6e} -> "
          f"{trace[-1]:.6e}, Regge stationarity {terms_before.regge_stationarity:.4e} -> "
          f"{terms_after.regge_stationarity:.4e}; two-body residual {two_body_before:.6e} -> {two_body_after:.6e}; "
          f"block residuals {blocks_before[0]:.3e} {blocks_before[1]:.3e} -> {blocks_after[0]:.3e} "
          f"{blocks_after[1]:.3e}; T after = {np.round(np.asarray(read.transfer).real, 8).tolist()}, "
          f"Schmidt {read.singular_values}")
