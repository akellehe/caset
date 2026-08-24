# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for the world-tube crossing readouts
(:class:`tessera.observables.CrossingReadouts`), ticket #807 and the
whitepaper section "Mass, charge, and form factor from world-tube
crossings".

The whitepaper is the specification these tests hold the code to:

* ``tau`` is the COMPLEX Lorentzian distance from the incoming boundary M0,
  built from the 1-skeleton and the stored complex edge lengths and never
  from a vertex coordinate; ``Re tau`` carries a temporal-function
  certificate and every failure is NAMED;
* ``pi_perp(c) = sum_e mu_c(e) dtau(e)`` stays COMPLEX, with ``mu_c`` the
  band density from the matched left/right frames (gauge-invariant by
  left/right cancellation) normalized to sum one over the crossing set;
* admissibility = the band's positivity certificate AND ``Re pi_perp``
  nonvanishing with a single sign; then ``sgn pi_perp := sgn Re pi_perp``;
* ``m_x = kappa_m sum |pi_perp|`` (INCOHERENT) and
  ``B = (1/3) sum sgn pi_perp`` over certified quark tubes (COHERENT), both
  as DIFFERENCES against the same sum at M0;
* the crossing sign and the determinant-line winding must agree on every
  certified tube -- disagreement is a DEFECT SIGNAL, never resolved;
* ``S(lambda)`` uses EIGENSPACE PROJECTORS, so it is basis- and
  phase-invariant and handles degeneracies; it is an incoherent structure
  factor and is NEVER the electromagnetic form factor.

Closed-form anchors (verified, not fitted):

* the causal ladder below has ``tau = 0`` on M0 and ``tau = 1`` on the
  upper layer exactly, because a timelike edge of length ``l = i`` has
  ``l^2 = -1`` and proper time ``sqrt(-l^2) = 1``;
* a tube localized on one timelike rung therefore has ``pi_perp = 1 + 0j``
  exactly, and a reversed tube ``-1 + 0j``;
* the degenerate slice-Laplacian eigenspace of a 3-cycle (spec {0, 3, 3})
  gives a closed-form eigenprojector ``I - J/3`` whose power is reproduced
  here from an INDEPENDENTLY ROTATED numpy eigenbasis of the same
  eigenspace -- the basis-invariance the projector formulation buys.
"""
import cmath
import math
import unittest

import numpy as np

import tessera

obs = tessera.observables

MACHINE = 1e-12  # machine-precision claims on exact closed forms


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None):
    """Explicit-complex idiom shared with the spectral-fiber and Hodge
    Laplacian suites."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.createVertex(i) for i in ids]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1.0 + 0j)
        e.setPhase(0.0)
    return st


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _ladder():
    """The causal ladder: a spacelike triangle 0-1-2 (M0), a spacelike
    triangle 3-4-5 above it, and three TIMELIKE rungs 0-3, 1-4, 2-5 with
    l = i, so l^2 = -1 and the proper time of each rung is exactly 1."""
    st = _from_simplices(
        6,
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
         (0, 3), (1, 4), (2, 5)],
    )
    for a, b in ((0, 3), (1, 4), (2, 5)):
        _edge(st, a, b).setLength(1j)
    return st


M0 = [0, 1, 2]
RUNGS = ((0, 3), (1, 4), (2, 5))


# --------------------------------------------------------------------------- #
# a real, certified, positive band -- reused as the certificate of every tube
# --------------------------------------------------------------------------- #
def _certified_rank_one_record():
    """The record of a REAL rank-one accepted band with positive signature,
    produced by the real tracker on the Euclidean 3-cycle (spec {0, 3, 3},
    so the lambda = 0 band is rank one and isolated).

    The tests below re-point this record's CELLS at the ladder's timelike
    rungs through the sanctioned `fromRecord` replay path.  The CERTIFICATE
    is never fabricated: it stays exactly the one the tracker issued.
    """
    st = _from_simplices(3, [(0, 1), (1, 2), (0, 2)])
    tracker = obs.SpectralFiberTracker(st, obs.SpectralFiberConfig())
    read = tracker.enumerateBands([0, 1, 2], 0)
    for fiber in read.fibers:
        cert = fiber.certificate()
        if (fiber.rank() == 1 and cert.accepted
                and cert.positiveSignature == 1 and cert.negativeSignature == 0):
            return fiber.toRecord()
    raise AssertionError("no certified rank-one positive band on the 3-cycle")


_BASE_RECORD = None


def _band_on(cells, amplitudes):
    """A rank-one band supported on `cells` with the given complex frame
    amplitudes, carrying the real certificate of `_certified_rank_one_record`.

    With unit weights the projector is `P = Phi Psi^dagger`, so the density
    on row i is |a_i|^2; `Psi^dagger W Phi = I` requires sum |a_i|^2 = 1.
    """
    global _BASE_RECORD
    if _BASE_RECORD is None:
        _BASE_RECORD = _certified_rank_one_record()
    record = {k: (list(v) if isinstance(v, list) else v)
              for k, v in _BASE_RECORD.items()}
    rows = len(cells)
    record["cells"] = [list(cell) for cell in cells]
    record["rows"] = rows
    record["rank"] = 1
    record["right_frame_re"] = [complex(a).real for a in amplitudes]
    record["right_frame_im"] = [complex(a).imag for a in amplitudes]
    record["left_frame_re"] = [complex(a).real for a in amplitudes]
    record["left_frame_im"] = [complex(a).imag for a in amplitudes]
    record["weights_re"] = [1.0] * rows
    record["weights_im"] = [0.0] * rows
    record["eigenvalues_re"] = [0.0]
    record["eigenvalues_im"] = [0.0]
    return obs.SpectralFiber.fromRecord(record)


def _tube(tube_id, cells, amplitudes, orientation=1, quark=True,
          winding=None):
    tube = obs.WorldTubeInput()
    tube.tubeId = tube_id
    tube.band = _band_on(cells, amplitudes)
    tube.orientation = orientation
    tube.certifiedQuarkTube = quark
    if winding is not None:
        tube.determinantWinding = winding
    return tube


def _rung_tube(tube_id, rung, **kwargs):
    return _tube(tube_id, [list(rung)], [1.0 + 0j], **kwargs)


# --------------------------------------------------------------------------- #
# the temporal function and its certificate
# --------------------------------------------------------------------------- #
class TestTemporalFunction(unittest.TestCase):
    def test_ladder_tau_is_exact_and_certified(self):
        """tau = 0 on M0 and exactly 1 on the upper layer: the proper time
        sqrt(-l^2) of a rung with l = i.  The certificate holds."""
        read = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        self.assertTrue(read.certified, read.failedCertificates)
        self.assertEqual(list(read.failedCertificates), [])
        for vertex in M0:
            self.assertAlmostEqual(abs(read.at(vertex)), 0.0, delta=MACHINE)
        for vertex in (3, 4, 5):
            self.assertAlmostEqual(read.at(vertex).real, 1.0, delta=MACHINE)
            self.assertAlmostEqual(read.at(vertex).imag, 0.0, delta=MACHINE)
        self.assertAlmostEqual(read.minCausalIncrement, 1.0, delta=MACHINE)
        self.assertEqual(read.causalEdgeCount, 3)
        self.assertEqual(read.unreachableCount, 0)

    def test_tau_is_complex_valued(self):
        """tau is complex like the geometry that defines it: a rung with a
        complex squared length carries a complex proper time."""
        st = _ladder()
        # l^2 = -(1 + i) -> proper time sqrt(1 + i), genuinely complex.
        _edge(st, 0, 3).setLength(cmath.sqrt(complex(-1.0, -1.0)))
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        expected = cmath.sqrt(complex(1.0, 1.0))
        self.assertAlmostEqual(read.at(3).real, expected.real, delta=1e-9)
        self.assertAlmostEqual(read.at(3).imag, expected.imag, delta=1e-9)
        self.assertGreater(abs(read.at(3).imag), 1e-3)

    def test_layers_are_intrinsic_not_coordinates(self):
        """The time orientation is the one M0 induces combinatorially; no
        vertex coordinate is read.  Moving a vertex's stored time cannot
        change the layering."""
        st = _ladder()
        before = obs.CrossingReadouts.temporalFunction(st, M0)
        for vertex_id in (3, 4, 5):
            st.getVertexList().get(vertex_id).setTime(-99.0)
        after = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertEqual(list(before.layer), list(after.layer))
        self.assertEqual(before.certified, after.certified)

    def test_empty_boundary_refuses_by_name(self):
        read = obs.CrossingReadouts.temporalFunction(_ladder(), [])
        self.assertFalse(read.certified)
        self.assertIn("empty-boundary", list(read.failedCertificates))

    def test_null_causal_edge_refuses_by_name(self):
        """A null edge is refused, never counted at zero."""
        st = _ladder()
        _edge(st, 0, 3).setLength(0.0 + 0j)
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertFalse(read.certified)
        self.assertIn("null-causal-edge", list(read.failedCertificates))

    def test_causal_edge_inside_one_layer_refuses_by_name(self):
        """A causal edge joining two vertices of the same layer cannot be
        ordered by the induced time orientation."""
        st = _ladder()
        _edge(st, 0, 1).setLength(1j)   # a timelike edge inside M0
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertFalse(read.certified)
        self.assertIn("causal-cycle", list(read.failedCertificates))

    def test_uncertified_temporal_function_blocks_every_crossing(self):
        read = obs.CrossingReadouts.temporalFunction(_ladder(), [])
        crossing = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), read, 0.5)
        self.assertFalse(crossing.admissible)
        self.assertIn("uncertified-temporal-function",
                      list(crossing.failedCertificates))


# --------------------------------------------------------------------------- #
# the band density and its gauge invariance
# --------------------------------------------------------------------------- #
class TestBandDensity(unittest.TestCase):
    def test_density_matches_the_projector_diagonal(self):
        band = _band_on([[0, 3], [1, 4]],
                        [math.sqrt(0.25), math.sqrt(0.75)])
        density = obs.CrossingReadouts.bandEdgeDensity(band)
        self.assertAlmostEqual(density[(0, 3)], 0.25, delta=MACHINE)
        self.assertAlmostEqual(density[(1, 4)], 0.75, delta=MACHINE)

    def test_density_is_gauge_invariant(self):
        """A local C* gauge factor acts on right frames by g^-1 and on left
        frames by g and CANCELS in the bilinear product, so the density is
        unchanged.  Verified through the whole crossing: pi_perp is
        identical."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        plain = _tube("q", [list(RUNGS[0]), list(RUNGS[1])],
                      [math.sqrt(0.25), math.sqrt(0.75)])
        phase = cmath.exp(0.7j)
        gauged = _tube("q", [list(RUNGS[0]), list(RUNGS[1])],
                       [phase * math.sqrt(0.25), phase * math.sqrt(0.75)])
        a = obs.CrossingReadouts.crossing(plain, temporal, 0.5)
        b = obs.CrossingReadouts.crossing(gauged, temporal, 0.5)
        self.assertTrue(a.admissible and b.admissible)
        self.assertAlmostEqual(abs(a.perpendicular - b.perpendicular), 0.0,
                               delta=MACHINE)
        for x, y in zip(a.density, b.density):
            self.assertAlmostEqual(x, y, delta=MACHINE)

    def test_degree_zero_band_has_no_edge_density(self):
        band = _band_on([[0]], [1.0 + 0j])
        self.assertEqual(band.degree(), 0)
        self.assertEqual(dict(obs.CrossingReadouts.bandEdgeDensity(band)), {})


# --------------------------------------------------------------------------- #
# the crossing decomposition
# --------------------------------------------------------------------------- #
class TestCrossing(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)

    def test_pi_perp_is_exact_on_one_rung(self):
        """A tube localized on one rung crosses with pi_perp = dtau = 1."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 0.5)
        self.assertTrue(read.admissible, read.failedCertificates)
        self.assertAlmostEqual(read.perpendicular.real, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.perpendicular.imag, 0.0, delta=MACHINE)
        self.assertEqual(read.sign, 1)
        self.assertEqual([tuple(e) for e in read.crossingEdges], [(0, 3)])
        self.assertAlmostEqual(read.density[0], 1.0, delta=MACHINE)

    def test_density_normalizes_over_the_crossing_set(self):
        """mu is normalized to sum one over the CROSSING SET, so a band with
        support off the surface still yields a normalized crossing."""
        band_cells = [list(RUNGS[0]), [3, 4]]     # one rung + one edge above
        tube = _tube("q", band_cells, [math.sqrt(0.5), math.sqrt(0.5)])
        read = obs.CrossingReadouts.crossing(tube, self.temporal, 0.5)
        self.assertTrue(read.admissible, read.failedCertificates)
        self.assertEqual([tuple(e) for e in read.crossingEdges], [(0, 3)])
        self.assertAlmostEqual(sum(read.density), 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.perpendicular.real, 1.0, delta=MACHINE)

    def test_orientation_reversal_flips_the_sign(self):
        """Reversing the tube sends the crossing past-directed: the sign
        flips while the modulus is untouched."""
        forward = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 0.5)
        reversed_ = obs.CrossingReadouts.crossing(
            _rung_tube("qbar", RUNGS[0], orientation=-1), self.temporal, 0.5)
        self.assertEqual(forward.sign, 1)
        self.assertEqual(reversed_.sign, -1)
        self.assertAlmostEqual(abs(forward.perpendicular),
                               abs(reversed_.perpendicular), delta=MACHINE)
        self.assertAlmostEqual(
            forward.perpendicular.real + reversed_.perpendicular.real, 0.0,
            delta=MACHINE)

    def test_level_outside_the_tube_refuses_by_name(self):
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 7.5)
        self.assertFalse(read.admissible)
        self.assertIn("empty-crossing", list(read.failedCertificates))
        self.assertEqual(read.sign, 0)

    def test_nonregular_level_refuses_by_name(self):
        """A level passing exactly through a vertex is not transversal."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 1.0)
        self.assertFalse(read.admissible)
        self.assertIn("nonregular-level", list(read.failedCertificates))

    def test_inadmissible_crossing_has_no_sign(self):
        """An inadmissible crossing reports sign 0 = UNKNOWN, never a silent
        zero that could be summed."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 7.5)
        self.assertEqual(read.sign, 0)
        self.assertTrue(math.isnan(read.perpendicular.real))


# --------------------------------------------------------------------------- #
# the two sums
# --------------------------------------------------------------------------- #
class TestMassAndBaryon(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)

    def _tubes(self, orientations, windings=None):
        windings = windings or [None] * len(orientations)
        return [
            _rung_tube(f"t{i}", RUNGS[i], orientation=o, winding=w)
            for i, (o, w) in enumerate(zip(orientations, windings))
        ]

    def test_three_forward_quark_tubes_give_baryon_one(self):
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1, 1, 1]), self.temporal, 0.5, 0.0)
        self.assertIsNotNone(read.baryonNumber)
        self.assertAlmostEqual(read.baryonNumber, 1.0, delta=MACHINE)
        self.assertEqual(read.quarkTubes, 3)

    def test_conjugate_pair_cancels_baryon_and_doubles_mass(self):
        """Moduli add while signs cancel: the pair carries twice the crossing
        mass of one constituent and zero net baryon number."""
        single = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        pair_tubes = self._tubes([1, -1])
        pair_mass = obs.CrossingReadouts.crossingMass(
            pair_tubes, self.temporal, 0.5, 0.0)
        pair_baryon = obs.CrossingReadouts.baryonNumber(
            pair_tubes, self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(pair_mass.crossingMass,
                               2.0 * single.crossingMass, delta=MACHINE)
        self.assertIsNotNone(pair_baryon.baryonNumber)
        self.assertAlmostEqual(pair_baryon.baryonNumber, 0.0, delta=MACHINE)

    def test_baryon_thirds_are_the_normalization(self):
        """One certified quark tube carries exactly B = 1/3, consistent with
        the independent determinant-line proposal B = nu/3."""
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(read.baryonNumber, 1.0 / 3.0, delta=MACHINE)
        reversed_read = obs.CrossingReadouts.baryonNumber(
            self._tubes([-1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(reversed_read.baryonNumber, -1.0 / 3.0,
                               delta=MACHINE)

    def test_non_quark_tubes_carry_mass_but_no_baryon_number(self):
        tubes = [_rung_tube("g", RUNGS[0], quark=False)]
        mass = obs.CrossingReadouts.crossingMass(
            tubes, self.temporal, 0.5, 0.0)
        baryon = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(mass.crossingMass, 1.0, delta=MACHINE)
        self.assertIsNone(baryon.baryonNumber)   # unknown, never zero

    def test_mass_ships_uncalibrated(self):
        read = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertFalse(read.calibrated)
        self.assertEqual(read.units, "uncalibrated")
        self.assertAlmostEqual(read.kappaMass, 1.0, delta=MACHINE)

    def test_kappa_scales_the_mass_only(self):
        cfg = obs.CrossingReadoutsConfig()
        cfg.kappaMass = 3.5
        scaled = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0, cfg)
        plain = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(scaled.crossingMass,
                               3.5 * plain.crossingMass, delta=MACHINE)
        baryon = obs.CrossingReadouts.baryonNumber(
            self._tubes([1]), self.temporal, 0.5, 0.0, cfg)
        self.assertAlmostEqual(baryon.baryonNumber, 1.0 / 3.0, delta=MACHINE)

    def test_readout_at_m0_itself_is_zero(self):
        """Every readout is the difference against the same sum at M0, so
        reading M0 against itself is zero by construction."""
        tubes = self._tubes([1, 1, 1])
        mass = obs.CrossingReadouts.crossingMass(
            tubes, self.temporal, 0.5, 0.5)
        baryon = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.5)
        self.assertAlmostEqual(mass.crossingMass, 0.0, delta=MACHINE)
        self.assertAlmostEqual(baryon.baryonNumber, 0.0, delta=MACHINE)

    def test_winding_disagreement_is_a_defect_signal(self):
        """A tube whose crossing sign disagrees with its certified winding is
        NAMED, and it is not dropped from the sum."""
        tubes = self._tubes([-1], windings=[+1])
        read = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.0)
        self.assertEqual(list(read.signDefects), ["t0"])
        self.assertEqual(read.windingAgreements, 0)
        self.assertAlmostEqual(read.baryonNumber, -1.0 / 3.0, delta=MACHINE)

    def test_winding_agreement_is_counted(self):
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1, -1], windings=[+1, -1]), self.temporal, 0.5, 0.0)
        self.assertEqual(list(read.signDefects), [])
        self.assertEqual(read.windingAgreements, 2)


# --------------------------------------------------------------------------- #
# the spectral charge-power profile
# --------------------------------------------------------------------------- #
class TestChargePowerProfile(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)

    def test_profile_matches_an_independent_rotated_eigenbasis(self):
        """S(lambda) is built from EIGENSPACE PROJECTORS, so it must agree
        with <rho, P_lambda rho> computed from a DIFFERENT (randomly rotated)
        orthonormal basis of the same eigenspace.

        The three rungs pairwise share no vertex, so the slice Laplacian is
        the 3-node empty graph: a single eigenvalue 0 of multiplicity three.
        Its projector is the identity, and the closed-form power is
        ||rho||^2 -- reproduced below through a random rotation of that
        degenerate eigenspace.
        """
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertEqual(read.sliceNodes, 3)

        rho = np.ones(3)                       # three forward unit crossings
        rng = np.random.default_rng(20260824)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))   # a rotated eigenbasis
        projector = q @ q.T                            # = I, basis-independent
        expected = float(rho @ projector @ rho)

        self.assertEqual(len(read.eigenvalues), 1)
        self.assertAlmostEqual(read.eigenvalues[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(read.power[0], expected, delta=1e-9)

    def test_degenerate_eigenspace_power_is_basis_independent(self):
        """The same invariance on a slice whose Laplacian has a genuinely
        degenerate NONZERO eigenspace: a 3-cycle of crossing edges (spec
        {0, 3, 3}), whose lambda = 3 eigenprojector is the closed form
        I - J/3."""
        # A slice whose crossing edges pairwise share vertices: three tubes
        # crossing on rungs that meet at the shared upper triangle.
        tubes = [
            _tube("a", [[0, 3]], [1.0 + 0j]),
            _tube("b", [[1, 4]], [1.0 + 0j]),
            _tube("c", [[2, 5]], [1.0 + 0j]),
        ]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        rho = np.ones(read.sliceNodes)
        total = sum(read.power)
        self.assertAlmostEqual(total, float(rho @ rho), delta=1e-9)

    def test_neutral_system_refuses_normalization_and_reports_power(self):
        """The normalizing monopole vanishes for a conjugate pair: the
        NORMALIZED profile refuses by name while the unnormalized power stays
        reported."""
        tubes = [
            _rung_tube("q", RUNGS[0]),
            _rung_tube("qbar", RUNGS[1], orientation=-1),
        ]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertFalse(read.normalized)
        self.assertIn("neutral-system", list(read.failedCertificates))
        self.assertEqual(list(read.normalizedPower), [])
        self.assertTrue(len(read.power) >= 1)
        self.assertGreater(sum(read.power), 0.0)

    def test_charged_system_normalizes_to_one_at_zero(self):
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertTrue(read.normalized, list(read.failedCertificates))
        zero_index = min(range(len(read.eigenvalues)),
                         key=lambda i: abs(read.eigenvalues[i]))
        self.assertAlmostEqual(read.normalizedPower[zero_index], 1.0,
                               delta=1e-9)

    def test_empty_slice_refuses_by_name(self):
        read = obs.CrossingReadouts.chargePowerProfile(
            [_rung_tube("q", RUNGS[0])], self.temporal, 7.5)
        self.assertFalse(read.normalized)
        self.assertIn("empty-slice", list(read.failedCertificates))
        self.assertEqual(read.sliceNodes, 0)


# --------------------------------------------------------------------------- #
# the conditional form factor
# --------------------------------------------------------------------------- #
class TestFormFactor(unittest.TestCase):
    def test_form_factor_is_a_refusal_scaffold(self):
        """G_E needs a certified conserved current, certified momentum-
        transfer states, and a refinement extrapolation.  None exist here, so
        the radius is UNAVAILABLE with every missing certificate named."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        profile = obs.CrossingReadouts.chargePowerProfile(
            tubes, temporal, 0.5)
        read = obs.CrossingReadouts.formFactor(profile)
        self.assertFalse(read.available)
        self.assertIsNone(read.chargeRadiusSquared)
        self.assertEqual(
            set(read.failedCertificates),
            {"no-certified-conserved-current", "no-certified-momentum-states",
             "no-refinement-extrapolation"})

    def test_spectral_power_is_never_substituted_for_g_e(self):
        """Even on a fully normalized profile the form factor refuses: the
        incoherent structure factor is not the coherent matrix element."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        profile = obs.CrossingReadouts.chargePowerProfile(
            tubes, temporal, 0.5)
        self.assertTrue(profile.normalized)
        read = obs.CrossingReadouts.formFactor(profile)
        self.assertFalse(read.available)
        self.assertIn("never substituted", read.note)


# --------------------------------------------------------------------------- #
# the overlay block
# --------------------------------------------------------------------------- #
class TestOverlayRecord(unittest.TestCase):
    def test_overlay_is_versioned_and_complete(self):
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        record = obs.CrossingReadouts.overlayRecord(
            tubes, temporal, 0.5, 0.0)
        self.assertEqual(record["schema_version"],
                         obs.CrossingReadouts.kSchemaVersion)
        for key in ("level", "reference_level", "thresholds",
                    "temporal_function", "crossings", "crossing_mass",
                    "baryon_number", "charge_power_profile", "form_factor"):
            self.assertIn(key, record)
        self.assertEqual(len(record["crossings"]), 3)
        self.assertAlmostEqual(record["baryon_number"]["baryon_number"], 1.0,
                               delta=MACHINE)
        self.assertFalse(record["form_factor"]["available"])

    def test_complex_channels_carry_both_parts(self):
        """pi_perp is complex and both parts always travel: nothing is
        silently .real-ed."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        record = obs.CrossingReadouts.overlayRecord(
            [_rung_tube("q", RUNGS[0])], temporal, 0.5, 0.0)
        crossing = record["crossings"][0]
        self.assertIn("perpendicular_re", crossing)
        self.assertIn("perpendicular_im", crossing)
        self.assertIn("tau_re", record["temporal_function"])
        self.assertIn("tau_im", record["temporal_function"])

    def test_thresholds_are_echoed(self):
        cfg = obs.CrossingReadoutsConfig()
        cfg.kappaMass = 2.25
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        record = obs.CrossingReadouts.overlayRecord(
            [_rung_tube("q", RUNGS[0])], temporal, 0.5, 0.0, cfg)
        self.assertAlmostEqual(record["thresholds"]["kappa_mass"], 2.25,
                               delta=MACHINE)
        self.assertFalse(record["thresholds"]["mass_calibrated"])


if __name__ == "__main__":
    unittest.main()
