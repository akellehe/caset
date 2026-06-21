# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Orientation signs come from the standard formula, not vertex-label sort (#412).

The color charge of a window is a SUM of per-hole periods. Read in each hole's own
sorted-tuple reference orientation, that sum depends on the arbitrary vertex
NUMBERING: relabeling the base vertices flips an individual hole's term with no
conjugating partner, scrambling the sum. The fix reads the emergent result in the
SAME global induced orientation the inputs already use --- the standard spatial
orientation `ChainComplex::endSignCovector` --- so the signed sum is the geometric
Stokes charge `Sigma_R = -Sigma_inputs`, a property of the geometry, not the labels.

Note the math: with `sign_k` the induced orientation and `raw_k` the period read in
hole `k`'s sorted order, `sign_k * raw_k = G * p_k` where `p_k` is the (label-free)
geometric period and `G` is one global orientation sign for the whole connected
surface. So `sum_k sign_k * raw_k = G * (geometric charge)` --- invariant under a
relabeling up to the single global `G`. The raw sum `sum_k raw_k` is not.

(The intertwining residual and singlet overlap were ALREADY relabeling-invariant ---
bilinear in the transport, so a per-hole sign change conjugates out --- and are left
untouched.)
"""

import cmath
import unittest
from collections import Counter

import numpy as np

import tessera

cob = tessera.cobordism
st = tessera.spacetime
_W = cmath.exp(2j * cmath.pi / 3)
_NEUTRAL = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
_COLORED = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
_SEED = 412


def _build(inputs):
    return cob.TransportCobordism(inputs, max_iters=0, seed=0,
                                  topology=cob.TripartiteRegisterTopology())


def _geometry(m):
    """Reconstruct the base surface (the 80 geodesic-2 icosahedron faces), the 12
    holes, the tetrahedra, and the per-layer stride N from a built junction --- the
    inputs `endSignCovector` needs, and the cells to rebuild under a relabeling."""
    tets = [tuple(sorted(v.getId() for v in t.getVertices()))
            for t in m.cobordism.getTopSimplices()]
    nverts = max(v for c in tets for v in c) + 1
    N = nverts // 3  # three prism layers (base x I, two intervals)
    incidence = Counter()
    for t in tets:
        for i in range(4):
            incidence[tuple(v for j, v in enumerate(t) if j != i)] += 1
    holed = [list(f) for f, c in incidence.items()
             if c == 1 and all(v < N for v in f)]  # t=0 cap (the holed base surface)
    holes = [sorted(h) for h in list(m.input_holes) + list(m.result_holes)]
    faces = holed + [h for h in holes if list(h) not in holed]
    return faces, holes, tets, N


def _signed_sigma_r(cobordism, input_holes, input_targets, result_holes, result_signs):
    """The carried color charge Sigma_R: carry the (signed) inputs as an L1 harmonic
    and read its periods over the result holes, each weighted by its induced sign."""
    es = cob.EigenstateSynthesis(cobordism, 1)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es.cellSimplices()) if len(c) == 2}
    psi = es.carriedRepresentative(input_holes, input_targets)
    total = 0.0 + 0.0j
    for sign, h in zip(result_signs, result_holes):
        a, b, c = sorted(h)
        total += sign * (psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]])
    return total


class SpatialOrientationPrimitiveTest(unittest.TestCase):
    """`endSignCovector` is the standard induced spatial orientation, and it is
    exactly what the production read-out applies."""

    def test_reconstructed_orientation_matches_the_applied_result_signs(self):
        # Independently recompute the induced orientation over the 12 holes and check
        # the result block (last three) equals the signs the C++ TransportCobordism
        # actually applies (m.result_signs == signTable_[R]).
        m = _build(_NEUTRAL)
        faces, holes, _tets, _N = _geometry(m)
        cov = cob.ChainComplex.endSignCovector(
            [list(f) for f in faces], [list(h) for h in holes])
        self.assertEqual(len(faces), 80)  # the geodesic-2 icosahedron
        self.assertEqual(cov[9:12], list(m.result_signs))

    def test_non_orientable_input_throws(self):
        # A facet with three cofaces is not a pseudomanifold: endSignCovector must
        # raise, never return a silent covector.
        bad = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]  # edge (0,1) in three faces
        with self.assertRaises(Exception):
            cob.ChainComplex.endSignCovector(bad, [[0, 1, 2]])


class ResultSigningTest(unittest.TestCase):
    """The production read-out applies the result block's induced orientation, so the
    signed charge is the geometric Stokes charge, not a bare per-hole sorted sum."""

    def test_result_signs_present_and_pm_one(self):
        m = _build(_NEUTRAL)
        signs = list(m.result_signs)
        self.assertEqual(len(signs), 3)
        self.assertTrue(all(s in (1, -1) for s in signs))

    def test_signed_charge_is_the_neutral_stokes_charge(self):
        # Neutral inputs (Sigma=0 each) -> signed Sigma_R ~ 0 (confinement), while the
        # raw (unsigned) sum is O(0.1)-O(1) and NOT the geometric charge.
        m = _build(_NEUTRAL)
        faces, holes, _t, _N = _geometry(m)
        cov = cob.ChainComplex.endSignCovector(
            [list(f) for f in faces], [list(h) for h in holes])
        targets = [cov[i] * _NEUTRAL[i // 3][i % 3] for i in range(9)]
        signed = _signed_sigma_r(m.cobordism, [list(h) for h in holes[:9]],
                                 targets, [list(h) for h in holes[9:12]], cov[9:12])
        raw = _signed_sigma_r(m.cobordism, [list(h) for h in holes[:9]],
                              targets, [list(h) for h in holes[9:12]], [1, 1, 1])
        self.assertLess(abs(signed), 1e-6)        # the Stokes charge, exactly ~0
        self.assertGreater(abs(raw), 0.05)        # the unsigned sum is not the charge

    def test_signed_charge_is_the_colored_net_charge(self):
        # Colored singles (R,G,B): each window Sigma=1, so the signed net charge is 3.
        m = _build(_COLORED)
        faces, holes, _t, _N = _geometry(m)
        cov = cob.ChainComplex.endSignCovector(
            [list(f) for f in faces], [list(h) for h in holes])
        targets = [cov[i] * _COLORED[i // 3][i % 3] for i in range(9)]
        signed = _signed_sigma_r(m.cobordism, [list(h) for h in holes[:9]],
                                 targets, [list(h) for h in holes[9:12]], cov[9:12])
        self.assertAlmostEqual(abs(signed), 3.0, delta=1e-6)

    def test_bipartite_register_result_is_left_unsigned(self):
        # The single (bipartite) register is out of scope and emits no result signs,
        # so its emergent result is unchanged (behavior preserved).
        m = cob.TransportCobordism(_NEUTRAL, max_iters=0, seed=0,
                                   topology=cob.RegisterTopology())
        self.assertEqual(len(list(m.result_signs)), 0)


class RelabelingInvarianceTest(unittest.TestCase):
    """G6 (this ticket owns it): permuting the base-surface vertex ids and rebuilding
    the SAME geometry leaves the signed charge invariant (up to one global sign),
    while the unsigned charge is scrambled."""

    def test_signed_charge_is_relabeling_invariant_raw_is_not(self):
        m = _build(_COLORED)
        faces, holes, tets, N = _geometry(m)

        def sigma(cobordism, perm, signed):
            pi = {v: perm[v] for v in range(len(perm))}
            pf = [[pi[v] for v in f] for f in faces]
            ph = [[pi[v] for v in h] for h in holes]
            cov = cob.ChainComplex.endSignCovector(pf, ph)
            targets = [cov[i] * _COLORED[i // 3][i % 3] for i in range(9)]
            rsigns = cov[9:12] if signed else [1, 1, 1]
            return _signed_sigma_r(cobordism, ph[:9], targets, ph[9:12], rsigns)

        identity = list(range(3 * N))
        base_signed = abs(sigma(m.cobordism, identity, True))
        self.assertAlmostEqual(base_signed, 3.0, delta=1e-6)

        rng = np.random.default_rng(_SEED)
        raw_values = []
        for _ in range(5):
            p = rng.permutation(N)
            perm = [int(p[v % N]) + (v // N) * N for v in range(3 * N)]  # per-layer
            ptets = [[perm[v] for v in t] for t in tets]
            pcob = st.Spacetime.fromCells(3, ptets, 1.0, 0.0)  # uniform l^2 = 1
            signed = abs(sigma(pcob, perm, True))
            raw = abs(sigma(pcob, perm, False))
            # SIGNED charge: invariant (up to the single global sign, hence abs).
            self.assertAlmostEqual(signed, base_signed, delta=1e-6)
            raw_values.append(round(raw, 4))
        # RAW charge: label-dependent -- it does NOT stay at the geometric 3.0, and it
        # is not constant across relabelings (the bug this ticket fixes).
        self.assertTrue(any(abs(r - 3.0) > 1e-3 for r in raw_values))


if __name__ == "__main__":
    unittest.main()
