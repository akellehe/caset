# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The symmetric apex interior of the W_ABC junction (#413).

The Freudenthal prism extrusion cuts each cell's tetrahedralization diagonal by
`std::sort` of vertex ids -- a label-dependent, symmetry-breaking choice that is the
documented source of the intertwining residual `||M P_in - P_out M|| / ||M|| ~ 4.26e-2`
that pins the color singlet at 0.999 instead of 1. `set_symmetric_interior(True)`
replaces it with `Spacetime::symmetricStackCells`: each base triangle cones up to a
face-apex and down to the top copy, and the gap octahedron over a shared edge is split
along the CANONICAL dual edge f1-f2 (the two face-centres) -- no vertex-label sort
chooses any diagonal. The carried transport then intertwines the color Z3 EXACTLY, so
the residual collapses to machine zero and the singlet overlap is 1.

These tests pin, through the production C++ `TransportCobordism`:
  * the symmetric interior is a valid manifold with b1 = 11 (same topology as the prism);
  * its intertwining residual is far below the prism's (< 5e-3, in fact ~machine zero);
  * its singlet overlap is essentially exact (>= 0.9999);
  * charge conservation (the signed Stokes Sigma_R) and the color Z3 are preserved;
  * the default (prism) path is unchanged.
"""

import cmath
import math
import unittest

import pytest

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_NEUTRAL = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]


def _build(symmetric):
    trt = cob.TripartiteRegisterTopology()
    trt.set_symmetric_interior(symmetric)  # symmetric apex is now the DEFAULT (#413)
    return cob.TransportCobordism(_NEUTRAL, max_iters=0, seed=0, topology=trt)


def _windows(m):
    ih = [tuple(sorted(h)) for h in m.input_holes]
    return [ih[0:3], ih[3:6], ih[6:9], [tuple(sorted(h)) for h in m.result_holes]]


def _transport(m):
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es.cellSimplices()) if len(c) == 2}
    holes = [h for w in _windows(m) for h in w]
    M = np.zeros((3, 9), complex)
    for col in range(9):
        psi = es.carriedRepresentative([list(holes[col])], [1.0])
        for k, (a, b, c) in enumerate(holes[9:12]):
            M[k, col] = psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]
    return M


def _window_cycle_rep(windows):
    ico = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 5, 10),
           (1, 10, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4),
           (4, 8, 9), (4, 9, 5), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11),
           (9, 8, 11), (10, 9, 11)]
    mid, nxt = {}, [12]

    def mk(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = nxt[0]; nxt[0] += 1
        return mid[k]

    for f in (tuple(sorted(t)) for t in ico):
        mk(f[0], f[1]); mk(f[1], f[2]); mk(f[0], f[2])
    gens = [[4, 3, 8, 9, 5, 0, 7, 11, 10, 1, 2, 6],
            [3, 4, 0, 2, 7, 8, 5, 1, 6, 11, 9, 10],
            [6, 10, 11, 7, 2, 1, 9, 8, 3, 0, 5, 4],
            [10, 6, 1, 5, 9, 11, 2, 0, 4, 8, 7, 3]]
    comp = lambda p, q: [p[q[i]] for i in range(len(q))]

    def lift(p):
        full = list(range(42))
        for i in range(12):
            full[i] = p[i]
        for (a, b), idx in mid.items():
            full[idx] = mk(p[a], p[b])
        return full

    grp = {tuple(p): p for p in [list(range(12))] + [list(g) for g in gens]}
    ch = True
    while ch:
        ch = False
        for p in list(grp.values()):
            for g in gens:
                r = comp(p, g)
                if tuple(r) not in grp:
                    grp[tuple(r)] = r; ch = True
    hs = [set(w) for w in windows]
    ah = lambda f, h: tuple(sorted(f[v] for v in h))

    def wp(f):
        perm = []
        for w in windows:
            img = {ah(f, h) for h in w}
            mm = [j for j in range(4) if img == hs[j]]
            if len(mm) != 1:
                return None
            perm.append(mm[0])
        return tuple(perm)

    gf = next(lift(p) for p in grp.values() if wp(lift(p)) == (1, 2, 0, 3))
    hl = [h for w in windows for h in w]
    hi = {h: i for i, h in enumerate(hl)}
    sgn3 = lambda t: 1 if ((t[0] > t[1]) + (t[0] > t[2]) + (t[1] > t[2])) % 2 == 0 else -1
    P = np.zeros((12, 12), complex)
    for i, h in enumerate(hl):
        img = (gf[h[0]], gf[h[1]], gf[h[2]])
        P[hi[tuple(sorted(img))], i] = sgn3(img)
    return P[0:9, 0:9], P[9:12, 9:12]


def _carried_sigma_r(m):
    """The signed carried color charge Sigma_R: carry the (signed) inputs as an L1
    harmonic and read its result-hole periods, weighted by the induced signs (#412).
    Input-dependent (unlike m.result's seed cyclePeriods), so ~0 for neutral inputs."""
    es = cob.EigenstateSynthesis(m.cobordism, 1)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es.cellSimplices()) if len(c) == 2}
    psi = es.carriedRepresentative(list(m.input_holes), list(m.input_hole_targets))
    total = 0.0 + 0.0j
    for sign, h in zip(list(m.result_signs), m.result_holes):
        a, b, c = sorted(h)
        total += sign * (psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]])
    return total


def _residual_and_overlap(m):
    windows = _windows(m)
    P_in, P_out = _window_cycle_rep(windows)
    M = _transport(m)
    resid = np.linalg.norm(M @ P_in - P_out @ M) / np.linalg.norm(M)
    wo, vout = np.linalg.eig(P_out)
    wv, vin = np.linalg.eig(P_in)
    singlet = vout[:, int(np.argmin(np.abs(wo - _W)))]
    ov = [abs(np.vdot(M @ vin[:, k], singlet))
          / (np.linalg.norm(M @ vin[:, k]) * np.linalg.norm(singlet))
          for k in range(9) if abs(wv[k] - _W) < 1e-6]
    return resid, min(ov), P_out


class SymmetricInteriorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prism = _build(symmetric=False)
        cls.sym = _build(symmetric=True)
        cls.r_prism, cls.ov_prism, _ = _residual_and_overlap(cls.prism)
        cls.r_sym, cls.ov_sym, cls.p_out = _residual_and_overlap(cls.sym)

    def test_symmetric_interior_is_a_valid_manifold_b1_eleven(self):
        betti = list(cob.ChainComplex.fromSpacetime(self.sym.cobordism).bettiNumbers())
        self.assertEqual(betti[0], 1)
        self.assertEqual(betti[1], 11)   # same topology as the prism (12 holes - 1)
        valid, _msg = cob.EigenstateSynthesis(self.sym.cobordism, 1).dualComplexValid()
        self.assertTrue(valid)

    def test_residual_collapses_vs_the_prism(self):
        # The prism is the documented ~4.26e-2 artifact; the symmetric interior drives
        # the residual far below it (in fact to ~machine zero).
        self.assertGreater(self.r_prism, 1e-2)          # the prism baseline
        self.assertLess(self.r_sym, 5e-3)               # the #413 acceptance bar
        self.assertLess(self.r_sym, self.r_prism / 100)  # >= two orders of magnitude

    def test_singlet_overlap_is_exact(self):
        self.assertGreaterEqual(self.ov_sym, 0.9999)    # vs the 0.999 prism floor

    def test_color_z3_eigenvalues_preserved(self):
        eig = sorted(np.angle(np.linalg.eigvals(self.p_out)))
        self.assertTrue(np.allclose(
            eig, [-2 * math.pi / 3, 0.0, 2 * math.pi / 3], atol=1e-9))

    def test_signed_stokes_charge_is_neutral(self):
        # Charge conservation: the signed CARRIED Sigma_R (the #412 endSignCovector-
        # weighted, input-dependent charge) is ~0 for the neutral input on the
        # symmetric uniform interior -- Sigma_R = -(Sigma_A+Sigma_B+Sigma_C), exact.
        self.assertEqual(len(list(self.sym.result_signs)), 3)
        self.assertLess(abs(_carried_sigma_r(self.sym)), 1e-6)

    def test_prism_path_unchanged(self):
        # Default off: the prism build still reproduces its ~4.26e-2 residual and
        # ~0.999 overlap (no regression to the existing path).
        self.assertAlmostEqual(self.r_prism, 4.26e-2, delta=5e-3)
        self.assertGreaterEqual(self.ov_prism, 0.999)


class SymmetricLorentzianTest(unittest.TestCase):
    """Parity for the prism's LorentzianPhotonTest on the (default) symmetric interior:
    the Lorentzian worldline seed works there too -- the surface<->apex and bottom<->top
    edges are the timelike worldlines (#413)."""

    @staticmethod
    def _l2(m):
        return [e.getSquaredLength().real
                for e in m.cobordism.getEdgeList().toVector()]

    def test_worldlines_become_timelike(self):
        trt = cob.TripartiteRegisterTopology()  # symmetric apex is the default
        trt.set_lorentzian_worldlines(-1.0)
        m = cob.TransportCobordism(_NEUTRAL, max_iters=0, seed=0, topology=trt)
        self.assertGreater(sum(1 for x in self._l2(m) if x < -1e-9), 0)

    @pytest.mark.slow
    def test_worldlines_stay_timelike_no_spurious_photon(self):
        # The symmetric counterpart of the prism's photon test, and a finding: on the
        # symmetric interior the relax keeps ALL worldlines uniformly timelike (-0.3) --
        # NO worldline spontaneously relaxes through null. The prism's "photon" (a null
        # edge emerging under relax) was triggered by the triangulation asymmetry
        # singling out one worldline; the symmetric interior removes that artifact, so a
        # photon needs a genuine symmetry-breaking SOURCE, not the bare uniform seed.
        trt = cob.TripartiteRegisterTopology()
        trt.set_lorentzian_worldlines(-0.3)
        m = cob.TransportCobordism(_NEUTRAL, max_iters=60, seed=0, topology=trt)
        l2 = self._l2(m)
        self.assertGreater(sum(1 for x in l2 if x < -1e-9), 0)    # worldlines timelike
        self.assertEqual(sum(1 for x in l2 if abs(x) < 1e-3), 0)  # no spurious photon


class SymmetricMetricSeedParityTest(unittest.TestCase):
    """Parity for the prism's EntangledMetricTest: the van Raamsdonk seed is prism-only
    (its id%N party map assumes the prism stride), so on the (default) symmetric
    interior set_entangled_metric is a no-op and the topology stays intact -- the same
    'a metric seed leaves the topology intact' property, on the symmetric default."""

    def test_topology_intact_under_metric_seed(self):
        trt = cob.TripartiteRegisterTopology()  # symmetric default
        trt.set_entangled_metric(math.log(3), math.log(3))  # skipped on symmetric
        m = cob.TransportCobordism(_NEUTRAL, max_iters=0, seed=0, topology=trt)
        self.assertEqual(list(m.stats.betti_cobordism)[1], 11)
        valid, _ = cob.EigenstateSynthesis(m.cobordism, 1).dualComplexValid()
        self.assertTrue(valid)
        # the seed is skipped, so the metric stays uniform (the symmetric requirement)
        l2 = [e.getSquaredLength().real for e in m.cobordism.getEdgeList().toVector()]
        self.assertTrue(all(abs(x - 1.0) < 1e-12 for x in l2))


if __name__ == "__main__":
    unittest.main()
