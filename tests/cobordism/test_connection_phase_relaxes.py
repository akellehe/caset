# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#853 — the connection phase is a dynamical field.

`phi` was declared in the ontology and moved by nothing: every `L_k` is
certified blind to it, so the objective's gradient with respect to `phi` was
identically zero. The term added here is built on the operator the connection
actually acts on — the degree-zero Aharonov-Bohm operator, whose zero mode a
nonzero flux lifts and which `ker L_0 = b_0` can never register.

Two properties carry the design and each is asserted rather than argued.

The term is read from the EIGENVALUES alone. A gauge transformation acts on the
operator by the similarity `diag(g)^-1 (.) diag(g)`, which fixes eigenvalues for
every `g: K_0 -> C*`, so the functional is constant along gauge orbits. Gauge
invariance is therefore a property of the construction, not a correction — and
the exact consequence, which stands in for an Euler identity here, is that the
`phi` gradient has NO component along any gauge direction, for every complex
vertex function.

The word EIGENVALUES is load-bearing and these tests are what hold it. An
entropy built the way the Hodge term builds one, on `A = M^dag M`, is a
functional of the SINGULAR values instead, and those survive only UNITARY
similarity. `C* = U(1) x R^+`, so that form is gauge-invariant for real `chi`
and measurably not for complex `chi` — this operator is non-normal under complex
phase, which is where the two spectra part company. `_chi` below is complex on
purpose: with the `M^dag M` form the entropy drifts 4.9e-3 and the identity
fails at 1.2e-2, against the 1e-16 and 1e-15 asserted here.

Gauge orthogonality alone would certify only the gauge subspace, so a second
exact identity — `S` is EVEN in `phi` at real weights, hence its gradient is
ODD — carries the physical directions the first one cannot see.

And `laplacian(k)` must stay blind. Making the geometric operator see `phi`
would be the error the two-field split exists to prevent, so its bitwise
invariance is re-asserted here alongside the new dependence.
"""

import cmath
import math
import unittest

import tessera as T

cob = T.cobordism
MC = cob.MultiCobordism
MODE = cob.HodgeEntropyPhaseMode.IncludeComplexPhase


def _host(jitter=True):
    """A closed 4-manifold with a mild non-degenerate metric."""
    spacetime = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                            1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(4))
    spacetime.build()
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        squared = 1.0 + (0.01 * (index % 6) if jitter else 0.0)
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


def _set_flux(spacetime, scale=1.0):
    """A connection with nonzero flux, in BOTH components of `phi`.

    Not a gauge transform of the flat one: the values do not come from any
    vertex function, so a holonomy around some cycle is necessarily nontrivial.
    """
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setPhase(complex(scale * 0.37 * ((index % 5) - 2),
                              scale * 0.11 * ((index % 3) - 1)))


def _flatten(spacetime):
    for edge in spacetime.getEdgeList().toVector():
        edge.setPhase(complex(0.0, 0.0))


def _gauge(spacetime, chi):
    """Apply `U_xy -> g_x^-1 U_xy g_y` with `g = e^{i chi}`.

    On the stored phase that is `phi -> phi + chi_t - chi_s`, since the stored
    orientation carries `e^{i phi}` and the reverse its inverse.
    """
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(edge.getPhase() + chi[target] - chi[source])


def _chi(spacetime, seed=0):
    """A complex vertex function — the full C* gauge group, not just U(1)."""
    values = {}
    for index, vertex in enumerate(spacetime.getVertexList().toVector()):
        step = index + seed
        values[int(vertex.getId())] = complex(0.29 * ((step % 7) - 3),
                                              0.13 * ((step % 4) - 1.5))
    return values


def _phases(spacetime):
    return [complex(edge.getPhase())
            for edge in spacetime.getEdgeList().toVector()]


class ConnectionEntropySeesThePhaseTest(unittest.TestCase):
    """The new operator depends on `phi`; every `L_k` still does not."""

    def test_the_connection_entropy_moves_when_the_phase_does(self):
        spacetime = _host()
        _flatten(spacetime)
        flat = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        _set_flux(spacetime)
        fluxed = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        self.assertNotAlmostEqual(
            flat, fluxed, places=9,
            msg="the connection entropy must SEE the connection")

    def test_every_hodge_laplacian_stays_bitwise_blind_to_the_phase(self):
        # The trap this whole design avoids. If a phase ever reaches the metric
        # weight, the geometry becomes gauge-variant and the derived form of
        # L_k is destroyed. Assert equality, not closeness.
        spacetime = _host()
        _flatten(spacetime)
        before = {k: cob.HodgeLaplacian(spacetime).laplacian(k, True)
                  for k in (0, 1, 2)}
        _set_flux(spacetime)
        for k in (0, 1, 2):
            with self.subTest(degree=k):
                self.assertEqual(
                    list(cob.HodgeLaplacian(spacetime).laplacian(k, True)),
                    list(before[k]),
                    "laplacian(%d) must be built from the lengths alone" % k)

    def test_the_phase_gradient_is_nonzero_under_flux(self):
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        self.assertEqual(len(gradient),
                         len(spacetime.getEdgeList().toVector()))
        self.assertGreater(sum(abs(component) ** 2 for component in gradient),
                           0.0,
                           "a fluxed connection must have a phi gradient")

    def test_both_components_of_the_phase_are_differentiated(self):
        # The owner's rule: never exclude Re or Im by construction. If a
        # component does not matter it must CANCEL, measurably, not be dropped.
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        self.assertGreater(max(abs(component.real) for component in gradient),
                           1e-12, "the compact part must be differentiated")
        self.assertGreater(max(abs(component.imag) for component in gradient),
                           1e-12, "the non-compact part must be too")


class GaugeInvarianceIsStructuralTest(unittest.TestCase):
    """Built from the EIGENVALUES, so gauge invariance is not a correction."""

    def test_a_gauge_transformation_leaves_the_entropy_unchanged(self):
        # Machine precision, not "close": the M^dag M form this replaced passes
        # a loose bar for real `chi` and fails at 4.9e-3 for complex `chi`, so a
        # slack tolerance here would stop distinguishing the two.
        spacetime = _host()
        _set_flux(spacetime)
        before = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        _gauge(spacetime, _chi(spacetime))
        after = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        self.assertAlmostEqual(before, after, delta=1e-13)

    def test_the_phase_gradient_is_orthogonal_to_every_gauge_direction(self):
        """The exact identity this term is certified by.

        `S` is constant along gauge orbits, so its directional derivative along
        any gauge displacement vanishes identically. In the `h = S_x - i S_y`
        convention that derivative is `sum_e Re(h_e v_e)`, and a gauge
        displacement is `v_e = chi_t - chi_s`. Holds for COMPLEX `chi`, so each
        vertex function gives two independent exact zeros.
        """
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        edges = spacetime.getEdgeList().toVector()
        scale = math.sqrt(sum(abs(component) ** 2 for component in gradient))
        self.assertGreater(scale, 0.0, "a zero gradient would pass vacuously")
        for seed in range(4):
            chi = _chi(spacetime, seed)
            with self.subTest(seed=seed):
                directional = 0.0
                for index, edge in enumerate(edges):
                    displacement = (chi[int(edge.getTarget().getId())] -
                                    chi[int(edge.getSource().getId())])
                    directional += (gradient[index] * displacement).real
                self.assertLess(
                    abs(directional) / scale, 1e-12,
                    "the phi gradient must have no gauge component")

    def test_a_flat_connection_relaxes_only_by_gauge(self):
        # With no flux there is nothing physical for the phase to do, so the
        # holonomy must stay trivial however the phases themselves move.
        spacetime = _host()
        _flatten(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        for index, component in enumerate(gradient):
            with self.subTest(edge=index):
                self.assertLess(abs(component), 1e-9)


class TheGradientIsCertifiedInEveryDirectionTest(unittest.TestCase):
    """A second exact identity, because gauge orthogonality is not enough.

    Gauge displacements span the image of the coboundary, which is `V - 1`
    complex dimensions out of `E`. On these hosts that is a small subspace, so a
    gradient could be wrong in every PHYSICAL direction and still satisfy the
    orthogonality identity exactly. That gap is closed here.

    For real weights, negating the connection transposes the operator:
    `L_ij(-phi) = -w e^{-i phi} = L_ji(phi)`. Transposition preserves
    eigenvalues, so `S` is EVEN in `phi` and its gradient is ODD — an exact
    identity that constrains every edge direction at once, not a subspace.
    """

    def test_the_gradient_is_odd_under_reversing_the_connection(self):
        spacetime = _host()
        _set_flux(spacetime)
        edges = spacetime.getEdgeList().toVector()
        forward = [complex(edge.getPhase()) for edge in edges]

        hodge = cob.HodgeLaplacian(spacetime)
        entropy = hodge.connectionSpectralEntropy()
        gradient = hodge.connectionSpectralEntropyPhaseGradient()
        scale = math.sqrt(sum(abs(component) ** 2 for component in gradient))
        self.assertGreater(scale, 0.0, "a zero gradient would pass vacuously")

        for edge, phase in zip(edges, forward):
            edge.setPhase(-phase)
        reversed_hodge = cob.HodgeLaplacian(spacetime)
        reversed_entropy = reversed_hodge.connectionSpectralEntropy()
        reversed_gradient = (
            reversed_hodge.connectionSpectralEntropyPhaseGradient())

        self.assertAlmostEqual(entropy, reversed_entropy, delta=1e-13,
                               msg="S must be even in phi at real weights")
        residual = math.sqrt(
            sum(abs(a + b) ** 2 for a, b in zip(reversed_gradient, gradient)))
        self.assertLess(residual / scale, 1e-12,
                        "the phi gradient must be odd in phi")


class StageTwoMovesThePhaseTest(unittest.TestCase):
    """The headline claim: a geometric update changes the connection."""

    def _node(self, spacetime, connection_weight):
        node = MC(spacetime, [], [], [1], 1.0, 7)
        node.set_objective(cob.JointStationarityObjective())
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        node.set_connection_entropy_weight(connection_weight)
        return node

    def test_stage_two_moves_the_phase_when_the_term_is_declared(self):
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 1.0)
        before = _phases(node.spacetime())
        list(node.run_stage2(max_iters=12))
        after = _phases(node.spacetime())
        self.assertEqual(len(before), len(after))
        moved = max(abs(a - b) for a, b in zip(after, before))
        self.assertGreater(moved, 0.0,
                           "declaring the term must make the phase dynamical")

    def test_the_phase_is_inert_when_the_term_is_not_declared(self):
        # Zero by default: a node acquires phase dynamics only when asked, and
        # every existing caller keeps the behaviour it had.
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 0.0)
        before = _phases(node.spacetime())
        list(node.run_stage2(max_iters=12))
        self.assertEqual(_phases(node.spacetime()), before)

    def test_the_term_appears_in_the_declared_objective_record(self):
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 1.0)
        self.assertIn(cob.ObjectiveTermName.CONNECTION_STATIONARITY,
                      node.objective_term_names())
        self.assertGreater(node.objective_terms().connection_stationarity, 0.0)


if __name__ == "__main__":
    unittest.main()
