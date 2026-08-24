# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""U(1) connection holonomy mode of observables::WilsonLoop (#114, T4).

WilsonMode.U1_CONNECTION accumulates the U(1) connection (``Edge.phase``)
around a closed cycle on the primal 1-skeleton, honoring each edge's stored
source->target orientation (``+phase`` forward, ``-phase`` reversed), and
returns the total holonomy mod 2*pi.

Validated against two independent references:

  * a numpy/hand oracle of the oriented phase sum (``_cycle_flux``), the same
    helper the Hodge-Laplacian tests use; and
  * the Stage-1 cycle flux carried by ``cobordism.HodgeLaplacian`` — recovered
    from the complex adjacency ``A[i,j] = squaredLength * exp(i*phase)`` it
    assembles (``_hodge_flux``), the cross-layer consistency T4 asserts.

The Z2 case (phases restricted to {0, pi}) checks that the bulk holonomy lands
in {0, pi} and matches the Stage-1 flux there.

Fixtures (shared idiom with tests/cobordism): build a HERMITIAN_WEIGHTED
spacetime directly from explicit simplex vertex tuples.
"""

import math
import types
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _triangle():
    """S^1: the 1-skeleton cycle 0-1-2-0 (b1 = 1)."""
    return _from_simplices(3, [(0, 1), (1, 2), (2, 0)])


def _testbed():
    """Square 0-1-2-3-0 plus the diagonal 0-2 (b1 = 2)."""
    return _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def _vertices_by_id(st):
    return {v.getId(): v for v in st.getVertexList().toVector()}


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _set_phases(st, phases):
    """Assign edge phases from a {frozenset({a, b}): phi} map; unit positive
    weights everywhere, phase 0 on edges not in the map."""
    for e in st.getEdgeList().toVector():
        key = frozenset({e.getSource().getId(), e.getTarget().getId()})
        e.setLength(cmath.sqrt(complex(1.0)))
        e.setPhase(phases.get(key, 0.0))


# --------------------------------------------------------------------------- #
# Oracles and mod-2*pi helpers
# --------------------------------------------------------------------------- #
def _cycle_flux(st, cycle):
    """Hand oracle: directed holonomy sum of phase around a closed vertex-id
    cycle, honoring each edge's stored source->target orientation.

    The COMPACT part only. The connection phase is complex (C* = U(1) x R+),
    and a Wilson loop measures the U(1) winding: only Re has winding, while
    Im is a local scale that would turn the holonomy into an unbounded
    modulus."""
    total = 0.0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.getSource().getId() == a and e.getTarget().getId() == b:
            total += e.getPhase().real
        else:
            total -= e.getPhase().real
    return total


def _hodge_flux(st, cycle):
    """Stage-1 cycle flux read off cobordism.HodgeLaplacian's complex
    adjacency: arg(A[i,j]) is +phase along source->target and -phase reversed
    (unit weights), so the directed sum around the cycle is the same oriented
    holonomy the operator encodes in L = D - A."""
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.array(cob.HodgeLaplacian(st).adjacency(), dtype=complex).reshape(n, n)
    total = 0.0
    m = len(cycle)
    for k in range(m):
        a, b = cycle[k], cycle[(k + 1) % m]
        total += np.angle(A[idx[a], idx[b]])
    return float(total)


def _wrap(theta):
    """Principal value in (-pi, pi] (matches WilsonLoop::principalAngle)."""
    r = math.remainder(theta, TWO_PI)
    if r <= -math.pi:
        r += TWO_PI
    return r


def _close_mod_2pi(a, b, tol=1e-9):
    """True iff a == b modulo 2*pi (robust to the +-pi boundary)."""
    return abs(math.remainder(a - b, TWO_PI)) < tol


def _holonomy(st, cycle):
    wl = tessera.WilsonLoop(st)
    by_id = _vertices_by_id(st)
    r = wl.evaluateU1Connection([by_id[c] for c in cycle])
    if not hasattr(r, "loopSize"):  # degenerate/open cycles return {} unchanged
        return r
    # WilsonResult.value is complex-typed for the deficit-angle mode's sake; a
    # U(1) holonomy is a real phase, and this asserts that rather than assuming.
    v = complex(r.value)
    assert v.imag == 0.0, f"U(1) holonomy grew an imaginary part: {v}"
    return types.SimpleNamespace(value=v.real, loopSize=r.loopSize)


# --------------------------------------------------------------------------- #
# Holonomy vs the hand oracle, and total flux Phi (mod 2*pi)
# --------------------------------------------------------------------------- #
class TestHolonomyVsOracle(unittest.TestCase):

    def test_zero_phase_is_trivial(self):
        st = _triangle()
        _set_phases(st, {})
        r = _holonomy(st, [0, 1, 2])
        self.assertEqual(r.loopSize, 3)
        self.assertAlmostEqual(r.value, 0.0, places=12)

    def test_matches_hand_oracle_single_edge(self):
        # All of Phi on one edge of the triangle; the holonomy equals the
        # oriented oracle sum exactly (both honor the stored orientation).
        for phi in (0.3, 1.0, 2.0, -0.7, math.pi / 2):
            with self.subTest(phi=phi):
                st = _triangle()
                _set_phases(st, {frozenset({0, 1}): phi})
                r = _holonomy(st, [0, 1, 2])
                self.assertEqual(r.loopSize, 3)
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_the_non_compact_part_carries_no_winding(self):
        """The connection is C* = U(1) x R+ and only the compact factor has
        winding, so a Wilson loop must read Re(phase) alone. Adding an
        arbitrary imaginary part -- a local scale, no quantum number -- cannot
        move the holonomy (#804)."""
        for phi, scale in ((0.3, 2.5), (-1.1, -0.75), (math.pi / 2, 4.0)):
            with self.subTest(phi=phi, scale=scale):
                compact = _triangle()
                _set_phases(compact, {frozenset({0, 1}): phi})
                twisted = _triangle()
                _set_phases(twisted, {frozenset({0, 1}): complex(phi, scale)})
                self.assertTrue(_close_mod_2pi(
                    _holonomy(twisted, [0, 1, 2]).value,
                    _holonomy(compact, [0, 1, 2]).value))

    def test_a_purely_non_compact_connection_is_trivial(self):
        # An imaginary-only phase is pure scale: zero winding.
        st = _triangle()
        _set_phases(st, {frozenset({0, 1}): complex(0.0, 1.7),
                         frozenset({1, 2}): complex(0.0, -0.4)})
        self.assertAlmostEqual(_holonomy(st, [0, 1, 2]).value, 0.0, places=12)

    def test_equals_total_flux_mod_2pi(self):
        # Orient the cycle to hit the phased edge forward, so the holonomy is
        # exactly +Phi (mod 2*pi). Covers Phi outside (-pi, pi] to exercise the
        # mod-2*pi reduction.
        for phi in (0.0, 1.0, math.pi, 4.0, 2 * math.pi, -5.0, 3 * math.pi):
            with self.subTest(phi=phi):
                st = _triangle()
                _set_phases(st, {frozenset({0, 1}): phi})
                e01 = _edge(st, 0, 1)
                s, t = e01.getSource().getId(), e01.getTarget().getId()
                third = ({0, 1, 2} - {s, t}).pop()
                r = _holonomy(st, [s, t, third])
                self.assertTrue(_close_mod_2pi(r.value, phi),
                                f"holonomy {r.value} != Phi {phi} (mod 2pi)")
                # Reduced into the principal interval (-pi, pi].
                self.assertGreater(r.value, -math.pi - 1e-12)
                self.assertLessEqual(r.value, math.pi + 1e-12)

    def test_distributed_phases_sum(self):
        # Distinct phases on every edge of a testbed cycle; the holonomy is the
        # oriented sum over the three traversed edges.
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): 0.4,
            frozenset({1, 2}): -1.1,
            frozenset({0, 2}): 0.9,   # the (2, 0) leg of the cycle
        })
        r = _holonomy(st, [0, 1, 2])
        self.assertEqual(r.loopSize, 3)
        self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_orientation_reversal_negates(self):
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): 0.4,
            frozenset({1, 2}): -1.1,
            frozenset({0, 2}): 0.9,
        })
        fwd = _holonomy(st, [0, 1, 2]).value
        rev = _holonomy(st, [2, 1, 0]).value
        self.assertTrue(_close_mod_2pi(fwd, -rev))


# --------------------------------------------------------------------------- #
# Cross-layer: equals the Stage-1 HodgeLaplacian cycle flux
# --------------------------------------------------------------------------- #
class TestMatchesHodgeStageOneFlux(unittest.TestCase):

    def test_triangle_random_phase_matches_hodge(self):
        rng = np.random.default_rng(114)
        for _ in range(8):
            st = _triangle()
            _set_phases(st, {
                frozenset({0, 1}): float(rng.uniform(-math.pi, math.pi)),
                frozenset({1, 2}): float(rng.uniform(-math.pi, math.pi)),
                frozenset({2, 0}): float(rng.uniform(-math.pi, math.pi)),
            })
            r = _holonomy(st, [0, 1, 2])
            self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, [0, 1, 2])))
            self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_testbed_both_cycles_match_hodge(self):
        # The b1 = 2 testbed has two independent cycles; each holonomy matches
        # the operator's Stage-1 flux for that signed-edge cycle.
        rng = np.random.default_rng(2024)
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({1, 2}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({2, 3}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({3, 0}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({0, 2}): float(rng.uniform(-math.pi, math.pi)),
        })
        for cycle in ([0, 1, 2], [0, 2, 3]):
            with self.subTest(cycle=cycle):
                r = _holonomy(st, cycle)
                self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, cycle)))
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, cycle)))


# --------------------------------------------------------------------------- #
# Z2 case: phases in {0, pi} -> holonomy in {0, pi}, matching Stage-1 flux
# --------------------------------------------------------------------------- #
class TestZ2Holonomy(unittest.TestCase):

    def test_parity_lands_in_zero_or_pi(self):
        # k edges at phase pi; the bulk holonomy is pi for odd k, 0 for even k
        # (a pi edge contributes +-pi == pi mod 2*pi regardless of orientation).
        edges = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 0})]
        saw_zero = saw_pi = False
        for k in range(len(edges) + 1):
            with self.subTest(num_pi_edges=k):
                st = _triangle()
                _set_phases(st, {e: math.pi for e in edges[:k]})
                r = _holonomy(st, [0, 1, 2])
                expected = math.pi if (k % 2 == 1) else 0.0

                # Holonomy restricted to {0, pi}.
                self.assertTrue(
                    abs(r.value) < 1e-9 or abs(r.value - math.pi) < 1e-9,
                    f"Z2 holonomy {r.value} not in {{0, pi}}")
                self.assertTrue(_close_mod_2pi(r.value, expected))

                # Cross-layer: matches the Stage-1 flux there.
                self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, [0, 1, 2])))
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

                saw_zero |= (expected == 0.0)
                saw_pi |= (expected == math.pi)
        self.assertTrue(saw_zero and saw_pi)  # both Z2 classes exercised


# --------------------------------------------------------------------------- #
# Degenerate / open cycles return an empty result
# --------------------------------------------------------------------------- #
class TestDegenerateCycles(unittest.TestCase):

    def test_too_few_vertices_is_empty(self):
        st = _triangle()
        _set_phases(st, {})
        by_id = _vertices_by_id(st)
        wl = tessera.WilsonLoop(st)
        self.assertEqual(wl.evaluateU1Connection([]).loopSize, 0)
        self.assertEqual(wl.evaluateU1Connection([by_id[0]]).loopSize, 0)

    def test_open_path_is_empty(self):
        # On the testbed, vertices 1 and 3 are not adjacent (no edge 1-3), so a
        # cycle through that pair is open and yields an empty result.
        st = _testbed()
        _set_phases(st, {})
        with self.assertRaises(KeyError):
            _edge(st, 1, 3)  # confirm the gap exists
        r = _holonomy(st, [0, 1, 3])
        self.assertEqual(r.loopSize, 0)
        self.assertEqual(r.value, 0.0)


if __name__ == "__main__":
    unittest.main()
