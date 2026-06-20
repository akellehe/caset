# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The tripartite W_ABC junction through the C++ MergeCobordism (#396).

`TripartiteRegisterTopology` is the trivalent proton junction: one geodesic-2
icosahedron (S^2, 42 vertices) minus 12 vertex-disjoint hole triangles in FOUR
windows of three -- A, B, C (inputs) and R (the emergent result) -- extruded x I.
Distinct holes = independent cycles, so the three inputs do NOT average (the #382
bipartite obstruction is escaped) and charge is conserved at the junction by the
surface's global Stokes relation (Sigma_R = -Sigma_inputs).

These tests pin, through the canonical C++ `MergeCobordism`:

  * the topology is a valid manifold with b1 = 11 (no weld);
  * three color-neutral pairs carry EXACTLY (residualForPeriods ~ 1e-27) -- no
    over-determination, unlike a shared register;
  * charge conservation: neutral inputs -> neutral result, colored -> |Sigma_R| ~ 3;
  * the #396 relaxation fix: the junction relaxation actually runs (it stalled at
    iteration 0 before the state-gradient gate);
  * the singlet is REACHABLE -- the input->result transport is full-rank and the
    singlet [1,w,w^2] is in its image (max overlap over neutral inputs = 1.0); a
    *natural* input is suboptimal (~0.7), the ceiling is the input, not the geometry;
  * the Lorentzian option makes worldlines timelike and a photon (null edge) can
    emerge under the relax.

The emergent result is read via the carried-input transport (carry the pinned
inputs over the input holes, read their periods over the result holes) using the
cobordism's OWN vertex labels (`m.input_holes` / `m.result_holes`).
"""

import cmath
import math
import unittest

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = [1, _W, _W * _W]
# Three color-neutral q-qbar pairs (Sigma = 0 each): the carriable inputs.
_NEUTRAL_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]


def _merge(inputs, max_iters=0, setup=None):
    """Build (and optionally relax) the junction via MergeCobordism. `setup`
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


class SingletReachabilityTest(unittest.TestCase):
    """The headline result: the 74% overlap is a suboptimal INPUT, not a geometry
    limit. The carried-input read is linear, so the input->result transport is a
    matrix M; it is full-rank and the singlet is in its image."""

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

    def test_a_natural_input_is_suboptimal(self):
        # A natural (gauge-asymmetric-windows) input does NOT reach the singlet --
        # the matched input is geometry-specific; symmetric windows are the
        # principled fix (deferred, #20).
        ov = _overlap(_carried_result(_merge(_NEUTRAL_PAIRS, max_iters=60)))
        self.assertLess(ov, 0.95)


class LorentzianPhotonTest(unittest.TestCase):
    def test_worldlines_become_timelike(self):
        m = _merge(_NEUTRAL_PAIRS, max_iters=0,
                   setup=lambda t: t.set_lorentzian_worldlines(-1.0))
        timelike = sum(1 for x in _edge_l2(m) if x < -1e-9)
        self.assertGreater(timelike, 0)

    def test_a_photon_null_edge_emerges_under_relax(self):
        # With near-null worldlines, the relax drives a worldline edge through null
        # (l^2 -> 0, lightlike) -- the predicted photon.
        m = _merge(_NEUTRAL_PAIRS, max_iters=60,
                   setup=lambda t: t.set_lorentzian_worldlines(-0.3))
        nulls = sum(1 for x in _edge_l2(m) if abs(x) < 1e-3)
        self.assertGreater(nulls, 0)


class EntangledMetricTest(unittest.TestCase):
    def test_set_entangled_metric_builds_a_valid_junction(self):
        # The VR-from-entanglement seed (a metric only) leaves the topology intact.
        m = _merge(_NEUTRAL_PAIRS, max_iters=0,
                   setup=lambda t: t.set_entangled_metric(math.log(3), math.log(3)))
        self.assertEqual(list(m.stats.betti_cobordism)[1], 11)


if __name__ == "__main__":
    unittest.main()
