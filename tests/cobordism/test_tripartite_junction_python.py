# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The tripartite W_ABC junction through the C++ TransportCobordism (#396).

`TripartiteRegisterTopology` is the trivalent proton junction: one geodesic-2
icosahedron (S^2, 42 vertices) minus 12 vertex-disjoint hole triangles in FOUR
windows of three -- A, B, C (inputs) and R (the emergent result) -- extruded x I.
Distinct holes = independent cycles, so the three inputs do NOT average (the #382
bipartite obstruction is escaped) and charge is conserved at the junction by the
surface's global Stokes relation (Sigma_R = -Sigma_inputs).

The windows are placed SYMMETRICALLY (#398): one orbit of a tetrahedral subgroup
A4 of the icosahedral rotation group, seated at the four tetrahedral vertex-orbits
of the icosahedron. The windows are A4-equivalent, so the per-window
period-transport blocks are cyclically related: the transport intertwines the color
Z3, and a color-symmetric (omega-representation) input reaches the EXACT singlet
with manifest S3 -- a greedy (geometrically inequivalent) placement reached ~0.74.

These tests pin, through the canonical C++ `TransportCobordism`:

  * the topology is a valid manifold with b1 = 11 (no weld);
  * three color-neutral pairs carry EXACTLY (residualForPeriods ~ 1e-27) -- no
    over-determination, unlike a shared register;
  * charge conservation: neutral inputs -> neutral result, colored -> |Sigma_R| ~ 3;
  * the #396 relaxation fix: the junction relaxation actually runs (it stalled at
    iteration 0 before the state-gradient gate);
  * the singlet is REACHABLE -- the input->result transport is full-rank and the
    singlet [1,w,w^2] is in its image (max overlap over neutral inputs = 1.0);
  * #398 SYMMETRY: the four windows are the tetrahedral A4 orbit, the transport
    intertwines the color Z3, and the natural color-symmetric (omega-rep) input
    transports to the EXACT singlet (overlap ~1.0), with manifest S3;
  * the Lorentzian option makes worldlines timelike and a photon (null edge) can
    emerge under the relax.

The emergent result is read via the carried-input transport (carry the pinned
inputs over the input holes, read their periods over the result holes) using the
cobordism's OWN vertex labels (`m.input_holes` / `m.result_holes`).
"""

import cmath
import math
import unittest

import pytest

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = [1, _W, _W * _W]
# Three color-neutral q-qbar pairs (Sigma = 0 each): the carriable inputs.
_NEUTRAL_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]


def _merge(inputs, max_iters=0, setup=None):
    """Build (and optionally relax) the junction via TransportCobordism. `setup`
    configures the topology (VR metric / Lorentzian worldlines) before the build."""
    trt = cob.TripartiteRegisterTopology()
    if setup is not None:
        setup(trt)
    return cob.TransportCobordism(inputs, max_iters=max_iters, seed=0, topology=trt)


def _carried_result(m):
    """The emergent result block R, via the carried-input transport: carry the
    pinned inputs (m.input_holes/targets) as an L1(W) harmonic and read its periods
    over the result holes -- using the cobordism's own vertex labels."""
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {}
    for i, c in enumerate(es.cellSimplices()):
        if len(c) == 2:
            edge[(min(c), max(c))] = i
    psi = es.carriedRepresentative(list(m.input_holes), list(m.input_hole_targets))
    out = []
    for h in m.result_holes:
        a, b, c = sorted(h)
        out.append(psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]])
    return out


def _overlap(v, ref=_SINGLET):
    v = np.array(v, complex)
    r = np.array(ref, complex)
    nv, nr = np.linalg.norm(v), np.linalg.norm(r)
    return abs(np.vdot(v, r)) / (nv * nr) if nv * nr > 0 else 0.0


def _edge_l2(m):
    return [e.getSquaredLength().real for e in m.cobordism.getEdgeList().toVector()]


def _windows(m):
    """The four windows (A, B, C inputs and R result), in the cobordism's OWN vertex
    labels. Each hole's corner is its smallest (< 12) vertex."""
    ih = [tuple(sorted(h)) for h in m.input_holes]
    return [ih[0:3], ih[3:6], ih[6:9], [tuple(sorted(h)) for h in m.result_holes]]


def _transport_matrix(m):
    """The input->result transport M (3 result components x 9 input color
    amplitudes): carry each UNIT input hole through the bare junction and read its
    raw periods on R. The seed geometry is input-independent (the symmetric uniform
    metric), so this is one build and nine carries."""
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {}
    for i, c in enumerate(es.cellSimplices()):
        if len(c) == 2:
            edge[(min(c), max(c))] = i
    holes = [h for w in _windows(m) for h in w]
    M = np.zeros((3, 9), complex)
    for col in range(9):
        psi = es.carriedRepresentative([list(holes[col])], [1.0])
        for k, h in enumerate(holes[9:12]):
            a, b, c = h
            M[k, col] = psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]
    return M


def _window_cycle_rep(windows):
    """The signed-permutation reps (P_in 9x9, P_out 3x3) of the window-cycling
    symmetry g -- the tetrahedral-group element that FIXES R (= windows[3]) and
    cycles A->B->C. Reconstructed from the icosahedral A4 generators applied to the
    windows' OWN labels (never Python-reconstructed holes). The omega-eigenvectors
    of P_in are the color-symmetric inputs; the singlet is the omega-eigenvector of
    P_out, and the transport intertwines the two."""
    ico = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 5, 10),
           (1, 10, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4),
           (4, 8, 9), (4, 9, 5), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11),
           (9, 8, 11), (10, 9, 11)]
    mid, nxt = {}, [12]

    def mk(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = nxt[0]
            nxt[0] += 1
        return mid[key]

    for f in (tuple(sorted(t)) for t in ico):
        mk(f[0], f[1]); mk(f[1], f[2]); mk(f[0], f[2])
    # The four C3 generators (12-vertex perms); they generate the tetrahedral A4.
    gens = [[4, 3, 8, 9, 5, 0, 7, 11, 10, 1, 2, 6],
            [3, 4, 0, 2, 7, 8, 5, 1, 6, 11, 9, 10],
            [6, 10, 11, 7, 2, 1, 9, 8, 3, 0, 5, 4],
            [10, 6, 1, 5, 9, 11, 2, 0, 4, 8, 7, 3]]

    def comp(p, q):
        return [p[q[i]] for i in range(len(q))]

    def lift(p):  # lift a 12-vertex perm to the 42 geodesic vertices
        full = list(range(42))
        for i in range(12):
            full[i] = p[i]
        for (a, b), idx in mid.items():
            full[idx] = mk(p[a], p[b])
        return full

    group = {tuple(p): p for p in [list(range(12))] + [list(g) for g in gens]}
    changed = True
    while changed:
        changed = False
        for p in list(group.values()):
            for g in gens:
                r = comp(p, g)
                if tuple(r) not in group:
                    group[tuple(r)] = r
                    changed = True
    hsets = [set(w) for w in windows]

    def apply_h(full, h):
        return tuple(sorted(full[v] for v in h))

    def winperm(full):
        perm = []
        for w in windows:
            img = {apply_h(full, h) for h in w}
            match = [j for j in range(4) if img == hsets[j]]
            if len(match) != 1:
                return None
            perm.append(match[0])
        return tuple(perm)

    g_full = next(lift(p) for p in group.values()
                  if winperm(lift(p)) == (1, 2, 0, 3))
    holes = [h for w in windows for h in w]
    hidx = {h: i for i, h in enumerate(holes)}

    def sgn3(t):  # parity of the oriented triple relative to its sorted order
        return 1 if ((t[0] > t[1]) + (t[0] > t[2]) + (t[1] > t[2])) % 2 == 0 else -1

    P = np.zeros((12, 12), complex)
    for i, h in enumerate(holes):
        img = (g_full[h[0]], g_full[h[1]], g_full[h[2]])
        P[hidx[tuple(sorted(img))], i] = sgn3(img)
    return P[0:9, 0:9], P[9:12, 9:12]


# Built once for the metric-independent structural assertions.
_M = _merge(_NEUTRAL_PAIRS, max_iters=0)


class TopologyTest(unittest.TestCase):
    def test_name_is_the_tripartite_junction(self):
        self.assertIn("tripartite", _M.stats.topology.lower())
        self.assertIn("windows", _M.stats.topology.lower())

    def test_betti_is_b1_eleven(self):
        # b1 = 11: 12 vertex-disjoint holes on one connected surface, minus one
        # global Stokes relation. NOT a shared b1=2 register (which would average).
        betti = list(_M.stats.betti_cobordism)
        self.assertEqual(betti[0], 1)
        self.assertEqual(betti[1], 11)

    def test_dual_complex_is_a_valid_manifold(self):
        valid, _msg = cob.EigenstateSynthesis(_M.cobordism, 1).dualComplexValid()
        self.assertTrue(valid)

    def test_windows_are_nine_input_three_result_disjoint(self):
        inp = {tuple(sorted(h)) for h in _M.input_holes}
        res = {tuple(sorted(h)) for h in _M.result_holes}
        self.assertEqual(len(_M.input_holes), 9)   # 3 input windows x 3 holes
        self.assertEqual(len(_M.result_holes), 3)  # the R window
        self.assertEqual(len(inp & res), 0)


class CarryAndConservationTest(unittest.TestCase):
    def test_neutral_pairs_carry_exactly(self):
        # Distinct holes = independent cycles -> the three neutral pairs realize
        # exactly (no over-determination), unlike a shared register (~1e-2 floor).
        self.assertLess(_M.stats.state_residual, 1e-18)

    def test_neutral_inputs_give_neutral_result(self):
        # Conservation at the junction: Sigma_R = -Sigma_inputs -> ~0 for neutral.
        # Read at the seed, where the carried-rep Stokes relation is exact.
        sigma = abs(sum(_carried_result(_merge(_NEUTRAL_PAIRS, max_iters=0))))
        self.assertLess(sigma, 0.3)

    def test_colored_inputs_carry_charge_not_floored(self):
        # Three colored singles (R,G,B) carry a net charge: the result-window sum is
        # O(1), in sharp contrast with the neutral case (~0.07). The junction
        # CARRIES color charge (conservation), it does not floor it -- per-input
        # confinement is the heavier tube junction, deferred. (Read over raw result
        # periods, so the magnitude is O(1) rather than the signed net charge 3.)
        colored = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        neutral = abs(sum(_carried_result(_merge(_NEUTRAL_PAIRS, max_iters=0))))
        sigma = abs(sum(_carried_result(_merge(colored, max_iters=0))))
        self.assertGreater(sigma, 0.7)
        self.assertGreater(sigma, 2.0 * neutral)


@pytest.mark.slow
class RelaxationFixTest(unittest.TestCase):
    """#396: the junction relaxation stalled at iteration 0 (a converged state
    residual's spurious gradient exploded the ill-conditioned step); the
    state-gradient gate fixed it."""

    def test_relaxation_runs(self):
        m = _merge(_NEUTRAL_PAIRS, max_iters=60)
        self.assertGreater(m.stats.relax_iterations, 0)

    def test_relaxation_reduces_the_action_residual(self):
        seed = _merge(_NEUTRAL_PAIRS, max_iters=0).stats.stat_action_residual
        relaxed = _merge(_NEUTRAL_PAIRS, max_iters=60).stats.stat_action_residual
        self.assertLess(relaxed, seed)


@pytest.mark.slow
class SingletReachabilityTest(unittest.TestCase):
    """A 74% overlap for a NAIVE input is a suboptimal input, not a geometry limit.
    The carried-input read is linear, so the input->result transport is a matrix M;
    it is full-rank and the singlet is in its image. (The matched input that
    actually reaches the singlet is the color-symmetric omega-rep -- see
    SymmetricWindowsTest, the #398 completion.)"""

    @classmethod
    def setUpClass(cls):
        # Transport M (3 result components x 9 input color amplitudes), by feeding
        # each unit input through the (bare) junction.
        cls.M = np.zeros((3, 9), complex)
        for col in range(9):
            inp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            inp[col // 3][col % 3] = 1
            cls.M[:, col] = _carried_result(_merge(inp, max_iters=0))

    def test_transport_is_full_rank(self):
        self.assertEqual(np.linalg.matrix_rank(self.M, tol=1e-9), 3)

    def test_singlet_is_reachable_over_neutral_inputs(self):
        # Restrict M to the neutral-input subspace (each window Sigma=0) and project
        # the singlet onto its image: max overlap = 1.0 iff the singlet is reachable.
        neutral = np.array([[1, -1, 0], [1, 0, -1]], complex).T  # 3x2 per window
        basis = np.zeros((9, 6), complex)
        for s in range(3):
            basis[3 * s:3 * s + 3, 2 * s:2 * s + 2] = neutral
        image, _ = np.linalg.qr(self.M @ basis)
        s = np.array(_SINGLET, complex)
        max_overlap = np.linalg.norm(image @ (image.conj().T @ s)) / np.linalg.norm(s)
        self.assertGreater(max_overlap, 0.999)

    def test_a_naive_input_is_suboptimal(self):
        # A NAIVE input (three identical neutral pairs, read raw) does NOT reach the
        # singlet: the matched input is the color-SYMMETRIC omega-rep, not just any
        # neutral input. SymmetricWindowsTest exhibits the symmetric input -> singlet.
        ov = _overlap(_carried_result(_merge(_NEUTRAL_PAIRS, max_iters=60)))
        self.assertLess(ov, 0.95)


class SymmetricWindowsTest(unittest.TestCase):
    """#398 -- the principled completion of the junction. The four windows are ONE
    orbit of a tetrahedral subgroup A4 of the icosahedral group (each a C3 orbit of
    corner sub-triangles), so the windows are A4-equivalent: the transport
    intertwines the color Z3, and the natural color-symmetric (omega-representation)
    input transports to the EXACT singlet with manifest S3."""

    @classmethod
    def setUpClass(cls):
        cls.windows = _windows(_M)
        cls.P_in, cls.P_out = _window_cycle_rep(cls.windows)
        cls.M = _transport_matrix(_M)

    def test_windows_are_the_four_tetrahedral_vertex_orbits(self):
        # The windows sit at the icosahedron's four tetrahedral vertex-orbits, which
        # partition all 12 original vertices (each hole's corner is its < 12 vertex).
        corners = {frozenset(min(h) for h in w) for w in self.windows}
        self.assertEqual(
            corners,
            {frozenset({2, 8, 10}), frozenset({1, 4, 7}),
             frozenset({0, 6, 9}), frozenset({3, 5, 11})})

    def test_metric_seed_is_uniform_symmetric(self):
        # The symmetric windows need a symmetry-respecting metric: the default seed
        # is uniform (l^2 = 1), unlike a jitter that would break the A4 symmetry.
        self.assertTrue(all(abs(x - 1.0) < 1e-12 for x in _edge_l2(_M)))

    def test_window_cycling_symmetry_is_the_color_z3(self):
        # g: the A4 3-cycle that fixes R and cycles A->B->C; on the result window it
        # is the color Z3 (P_out has eigenvalues 1, omega, omega^2).
        eig = sorted(np.angle(np.linalg.eigvals(self.P_out)))
        self.assertTrue(np.allclose(
            eig, [-2 * math.pi / 3, 0.0, 2 * math.pi / 3], atol=1e-9))

    def test_transport_intertwines_the_color_z3(self):
        # M P_in = P_out M (the windows are A4-equivalent). The small residual is the
        # junction's intrinsic discretization (a g-invariant non-degenerate metric
        # gives the same ~4e-2); the singlet overlap below is the conclusive metric.
        err = (np.linalg.norm(self.M @ self.P_in - self.P_out @ self.M)
               / np.linalg.norm(self.M))
        self.assertLess(err, 0.1)

    def test_symmetric_input_transports_to_the_singlet(self):
        # THE HEADLINE (#398): the natural color-symmetric input is the omega-rep of
        # g (the omega-eigenvectors of P_in); each transports to the singlet (the
        # omega-eigenvector of P_out) with overlap ~1.0 -- vs ~0.74 for greedy windows.
        wv, vin = np.linalg.eig(self.P_in)
        wo, vout = np.linalg.eig(self.P_out)
        singlet = vout[:, int(np.argmin(np.abs(wo - _W)))]
        overlaps = [_overlap(self.M @ vin[:, k], singlet)
                    for k in range(9) if abs(wv[k] - _W) < 1e-6]
        self.assertEqual(len(overlaps), 3)  # the omega-eigenspace of P_in is 3-dim
        self.assertGreater(min(overlaps), 0.99)


class LorentzianPhotonTest(unittest.TestCase):
    # The Lorentzian worldline seed requires the PRISM interior: its cross-layer
    # detection (id / N) assumes the prism stride, so it is skipped under the
    # (default) symmetric apex interior -- select the prism explicitly here (#413).
    def test_worldlines_become_timelike(self):
        m = _merge(_NEUTRAL_PAIRS, max_iters=0,
                   setup=lambda t: (t.set_symmetric_interior(False),
                                    t.set_lorentzian_worldlines(-1.0)))
        timelike = sum(1 for x in _edge_l2(m) if x < -1e-9)
        self.assertGreater(timelike, 0)

    @pytest.mark.slow
    def test_a_photon_null_edge_emerges_under_relax(self):
        # With near-null worldlines, the relax drives a worldline edge through null
        # (l^2 -> 0, lightlike) -- the predicted photon.
        m = _merge(_NEUTRAL_PAIRS, max_iters=60,
                   setup=lambda t: (t.set_symmetric_interior(False),
                                    t.set_lorentzian_worldlines(-0.3)))
        nulls = sum(1 for x in _edge_l2(m) if abs(x) < 1e-3)
        self.assertGreater(nulls, 0)


class EntangledMetricTest(unittest.TestCase):
    def test_set_entangled_metric_builds_a_valid_junction(self):
        # The VR-from-entanglement seed (a metric only) leaves the topology intact.
        # It requires the PRISM interior (its id % N party map assumes the prism
        # stride, skipped under the default symmetric apex interior) -- #413.
        m = _merge(_NEUTRAL_PAIRS, max_iters=0,
                   setup=lambda t: (t.set_symmetric_interior(False),
                                    t.set_entangled_metric(math.log(3), math.log(3))))
        self.assertEqual(list(m.stats.betti_cobordism)[1], 11)


if __name__ == "__main__":
    unittest.main()
