# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for quark/antiquark discovery and the emergent
flavor-charge reads (:class:`tessera.observables.ParticleClusters`),
ticket #773 / design spec sections 6.8 and 16.1 (Algorithm I, quark
classifier) and the whitepaper "Quarks as modular clusters".

Covers every ticket acceptance bullet:

* synthetic anchored quark/antiquark fixtures differ by orientation, have
  certified winding nu = +-1, and provisionally carry B = +-1/3;
* a two-quark anti-triplet is NOT mislabeled an antiquark (distinguished
  by total occupation and determinant-line data, not color alone);
* a certified gap-preserving conjugate-pair creation path has zero total
  determinant winding/baryon flux and even total parity; a singular
  (gap-closing) path returns UNKNOWN flux;
* a missing/unstable flavor doublet yields unknown flavor AND charge;
* certified u/d fixtures return +2/3 / -1/3 Gauss-consistent charge, and
  Q = I3 + B/2 is tested only when both baryon flux and the doublet are
  certified (the proposed u/d identification);
* unknown fields are None and every missing certificate is NAMED in
  failedCertificates (negative control per certificate);
* relabeling and refinement preserve accepted classifications;
* cached classification equals cold recomputation under the #764 cache;
* no quark-specific quantity enters the emergence objective.

Fixtures are built by composing the MERGED public APIs (#765 ComponentId,
#769 SpectralFiber/ComponentBandRead record synthesis, #767 ColorAnchor,
#770 FiberConnection transports/windings, #780 CovarianceState Wick
reads, and the existing EigenstateSynthesis.gaussLawCharge) — the
classifier consumes them; nothing is faked past its own public surface.
"""
import math
import time
import unittest
from pathlib import Path

import numpy as np

import tessera

obs = tessera.observables
cob = tessera.cobordism
qm = tessera.quantum

MACHINE = 1e-12
NAN = float("nan")
TWO_PI = 2.0 * math.pi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------- #
# fiber fixtures (the sanctioned #769 record-rehydration route)
# --------------------------------------------------------------------------- #
def _cert_record(regime, grade="certified-numerical"):
    return {"grade": grade, "domain": "band-window", "regime": regime,
            "residual": 1e-15, "conditioning": 1.0,
            "dense_reference_error": NAN, "tolerance": 1e-9}


def _split(name, values, record):
    arr = np.asarray(values, dtype=complex).reshape(-1)
    record[name + "_re"] = [float(v.real) for v in arr]
    record[name + "_im"] = [float(v.imag) for v in arr]


def _fiber(cells, right, left=None, weights=None, *, degree=1, accepted=True,
           regime="positive-semidefinite", pos=None, neg=0, lower_gap=1.0,
           upper_gap=1.0, cond=1.0, self_adjoint=True, gram_defect=0.0,
           localization=0.5, eigenvalues=None):
    """Rehydrate a SpectralFiber from its record (the #769 replay route)."""
    right = np.asarray(right, dtype=complex)
    n, r = right.shape
    left = right if left is None else np.asarray(left, dtype=complex)
    weights = (np.ones(n, dtype=complex) if weights is None
               else np.asarray(weights, dtype=complex))
    pos = r if pos is None else pos
    eigenvalues = ([1.0 + 0j] * r if eigenvalues is None else eigenvalues)
    record = {
        "schema_version": 1, "record_type": "spectral_fiber",
        "cells": [[int(v) for v in cell] for cell in cells],
        "rows": int(n), "rank": int(r),
        "certificate": {
            "degree": int(degree), "rank": int(r),
            "lower_gap": float(lower_gap), "upper_gap": float(upper_gap),
            "localization": float(localization),
            "projector_residual": 1e-16,
            "eigen_residual": 1e-16, "left_residual": 1e-16,
            "gram_defect": float(gram_defect),
            "condition_number": float(cond),
            "positive_signature": int(pos), "negative_signature": int(neg),
            "frequency_lower": 0.0, "frequency_upper": 2.0,
            "self_adjoint": bool(self_adjoint), "accepted": bool(accepted),
            "certificate": _cert_record("positive-semidefinite"
                                        if regime is None else regime)}}
    _split("eigenvalues", eigenvalues, record)
    _split("right_frame", right, record)
    _split("left_frame", left, record)
    _split("weights", weights, record)
    return obs.SpectralFiber.fromRecord(record)


def _unit_fiber(base_id, r, **kw):
    """Rank-r fiber with identity frame on r synthetic cells."""
    return _fiber([[base_id + i] for i in range(r)], np.eye(r), **kw)


def _band_read(fibers, degree=1, support=None):
    """Synthesize a ComponentBandRead carrying `fibers` (the #769 replay
    route for a whole enumeration frame)."""
    cells = []
    for f in fibers:
        cells.extend(f.cellVertices())
    record = {
        "schema_version": 1, "record_type": "spectral_band_read",
        "support": [int(v) for v in (support or [])],
        "degree": int(degree),
        "dimension": len(cells),
        "cell_vertices": [[int(v) for v in cell] for cell in cells],
        "regime": "positive-semidefinite",
        "solver_path": "dense-self-adjoint",
        "truncated": False,
        "fibers": [f.toRecord() for f in fibers],
        "solve_certificate": _cert_record("positive-semidefinite"),
    }
    _split("covered_eigenvalues", [], record)
    return obs.ComponentBandRead.fromRecord(record)


def _phase_link(conn, A, B, phi):
    """Accepted rank-3 transport whose determinant phase is exactly phi."""
    v = np.diag([np.exp(1j * phi), 1.0, 1.0]).astype(complex)
    return conn.transport(A, B, v)


def _winding_family(conn, A, B, turns=1, samples=8):
    """A closed transport family with certified integer winding `turns`."""
    return [_phase_link(conn, A, B, TWO_PI * turns * k / samples)
            for k in range(samples)]


def _parity_occupation(occupations):
    """#780 Wick parity/total-number reads of a diagonal covariance."""
    state = qm.CovarianceState.fromOccupations(np.asarray(occupations,
                                                          dtype=float))
    return state.wickParity(), state.wickTotalNumber()


def _anchor_profile(seed=61):
    """#767 single-triangle oracle: score 1, coherence 1, held certificate."""
    rng = np.random.default_rng(seed)
    w = np.array([2.0, 0.5, 1.25])
    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    phi = obs.ColorAnchor.orthonormalizeFrame(z, w)
    anchor = obs.ColorAnchor([tessera.OrientedTriangle([0, 1, 2], [1, 1, 1])])
    return anchor.evaluate(phi, w)


def _doublet_frames(frames=3, base=200, ranks=(1, 2, 3), drop_rank2_at=None,
                    unaccept_rank2_at=None, extra_rank2=False):
    """Synthetic per-frame band enumerations for the flavor search: one
    band per rank in `ranks`, identical cells across frames (overlap 1)."""
    out = []
    for t in range(frames):
        fibers = []
        for r in ranks:
            if r == 2 and drop_rank2_at == t:
                # moved to disjoint cells: no positive overlap into frame t
                fibers.append(_unit_fiber(base + 50 + 10 * t, 2))
                continue
            accepted = not (r == 2 and unaccept_rank2_at == t)
            fibers.append(_unit_fiber(base + 10 * r, r, accepted=accepted))
        if extra_rank2:
            fibers.append(_unit_fiber(base + 80, 2))
        out.append(_band_read(fibers))
    return out


# --------------------------------------------------------------------------- #
# spacetime fixtures (shared explicit-complex idiom)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None, timelike=True):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.createVertex(i) for i in ids]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1j if timelike else (1.0 + 0j))
        e.setPhase(0.0)
    return st


_TETRA_CHAIN = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]


def _gauss_fixture(target, ids=None):
    """A tetrahedron-chain complex (all edges timelike -> every plaquette
    electric) plus a field-strength cochain whose electric flux equals
    `target` on BOTH nested enclosing surfaces (solved through the real
    gaussLawCharge by linearity)."""
    st = _from_simplices(7, _TETRA_CHAIN, ids=ids)
    base = 0 if ids is None else ids[0]
    es = cob.EigenstateSynthesis(st, 2)
    surfaces = obs.ParticleClusters.nestedEnclosures(st, [base], 2)
    n = es.order()

    def probe(vset):
        rows = []
        for c in range(n):
            f = [0j] * n
            f[c] = 1.0 + 0j
            rows.append(es.gaussLawCharge(f, vset, True).real)
        return np.array(rows)

    A = np.vstack([probe(surfaces[0]), probe(surfaces[1])])
    F, *_ = np.linalg.lstsq(A, np.array([target, target]), rcond=None)
    return st, [complex(v) for v in F], surfaces


# --------------------------------------------------------------------------- #
# the full certified evidence bundle
# --------------------------------------------------------------------------- #
def _certified_evidence(turns=1, *, occupations=(1.0, 0.0, 0.0),
                        band_base=1, with_flavor=False, occupancy=None,
                        orientation=1, with_charge=None, conn=None,
                        anchor=None):
    conn = conn or obs.FiberConnection()
    A = _unit_fiber(band_base, 3)
    B = _unit_fiber(band_base + 10, 3)
    family = _winding_family(conn, A, B, turns=turns)
    winding = conn.closedFamilyWinding(family)

    parity, occupation = _parity_occupation(list(occupations))

    ev = obs.QuarkCandidateEvidence()
    ev.component = obs.ComponentId("ab" * 16, 1)
    ev.colorBand = A
    ev.anchor = anchor if anchor is not None else _anchor_profile()
    ev.lifetimeTransports = family
    ev.winding = winding
    ev.parityRead = parity
    ev.occupationRead = occupation
    ev.persistenceLifetime = 3.0
    ev.persistenceMinOverlap = 1.0
    ev.refinementOverlap = 1.0
    if with_flavor:
        pc = obs.ParticleClusters()
        ev.flavor = pc.flavorDoubletSearch(_doublet_frames())
        assert ev.flavor.found
        ev.doubletOccupancy = (np.array([1.0, 0.0], dtype=complex)
                               if occupancy is None
                               else np.asarray(occupancy, dtype=complex))
        ev.doubletOrientation = orientation
    if with_charge is not None:
        pc = obs.ParticleClusters()
        st, F, surfaces = _gauss_fixture(with_charge)
        ev.charge = pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        assert ev.charge.consistent
    return ev


CORE = ["persistence", "localization", "parity-odd", "occupation-one",
        "color-rank-three", "anchor", "transport-leakage", "winding",
        "winding-unit", "refinement-stability"]


# =========================================================================== #
# core classification
# =========================================================================== #
class TestCoreClassification(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_quark(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=1))
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.determinantWinding, 1)
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.exteriorParity, -1)
        self.assertEqual(read.colorRank, 3)
        self.assertEqual(read.confidence, 1.0)
        self.assertEqual(read.windingClosure, "closed-family")
        for name in CORE:
            self.assertNotIn(name, read.failedCertificates)
        self.assertTrue(read.certificate.holds())

    def test_certified_antiquark_is_the_orientation_reverse(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=-1))
        self.assertEqual(read.classification, "antiquark")
        self.assertEqual(read.determinantWinding, -1)
        self.assertAlmostEqual(read.baryonFlux, -1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.confidence, 1.0)

    def test_quark_and_antiquark_differ_only_by_orientation(self):
        q = self.pc.classifyQuark(_certified_evidence(turns=1))
        aq = self.pc.classifyQuark(_certified_evidence(turns=-1))
        # identical anchored/parity/persistence evidence; opposite line
        self.assertEqual(q.triangleAnchorScore, aq.triangleAnchorScore)
        self.assertEqual(q.exteriorParity, aq.exteriorParity)
        self.assertEqual(q.occupationTotal, aq.occupationTotal)
        self.assertEqual(q.determinantWinding, -aq.determinantWinding)
        self.assertEqual(q.baryonFlux, -aq.baryonFlux)

    def test_reversed_family_is_the_antiquark(self):
        # reversing the tube = traversing the same transport family in the
        # opposite parameter order (the #770 orientation convention)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B, turns=1)
        ev = _certified_evidence(turns=1)
        ev.winding = conn.closedFamilyWinding(list(reversed(family)))
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "antiquark")

    def test_unknown_winding_leaves_baryon_flux_unknown(self):
        ev = _certified_evidence()
        # open segment with NO declared closure: raw endpoint phase is
        # never promoted to baryon-flux evidence
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.4, 0.8)]
        ev.winding = conn.openSegmentWinding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.determinantWinding)
        self.assertIsNone(read.baryonFlux)
        self.assertEqual(read.windingClosure, "none")
        self.assertIn("winding", read.failedCertificates)
        self.assertEqual(read.classification, "none")
        self.assertFalse(read.certificate.holds())

    def test_open_segment_with_declared_closure_carries_its_specification(self):
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, TWO_PI * k / 4) for k in range(5)]
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        spec.referenceId = "co-moving-reference"
        spec.referenceTransports = [np.eye(3)] * len(segment)
        ev = _certified_evidence()
        ev.winding = conn.openSegmentWinding(segment, spec)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.windingClosure, "matched-reference")
        self.assertEqual(read.windingReferenceId, "co-moving-reference")
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)

    def test_certified_zero_winding_is_a_certified_zero_flux_not_a_quark(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=0))
        self.assertEqual(read.determinantWinding, 0)
        self.assertEqual(read.baryonFlux, 0.0)  # certified zero, not unknown
        self.assertIn("winding-unit", read.failedCertificates)
        self.assertNotIn("winding", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_anchor_profile_travels_on_the_read(self):
        profile = _anchor_profile()
        read = self.pc.classifyQuark(_certified_evidence(anchor=profile))
        self.assertAlmostEqual(read.triangleAnchorScore, profile.score,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangleAnchorMaxTerm, profile.max_term,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangleAnchorParticipation,
                               profile.participation_ratio, delta=MACHINE)
        self.assertAlmostEqual(read.anchorPhaseDispersion,
                               profile.phase_dispersion, delta=MACHINE)
        self.assertEqual(read.anchorWeightingId, "uniform")

    def test_confidence_is_the_passed_core_fraction(self):
        ev = _certified_evidence()
        ev.refinementOverlap = NAN  # remove exactly one core certificate
        read = self.pc.classifyQuark(ev)
        self.assertAlmostEqual(read.confidence, 9.0 / 10.0, delta=MACHINE)
        self.assertEqual(read.classification, "none")

    def test_thresholds_are_recorded_on_every_read(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minAnchorScore = 0.75
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertEqual(read.thresholds.minAnchorScore, 0.75)
        rec = read.toRecord()
        self.assertEqual(rec["thresholds"]["min_anchor_score"], 0.75)

    def test_classify_quarks_stream_preserves_order(self):
        reads = self.pc.classifyQuarks(
            [_certified_evidence(turns=1), _certified_evidence(turns=-1)])
        self.assertEqual([r.classification for r in reads],
                         ["quark", "antiquark"])


# =========================================================================== #
# negative controls: a candidate missing ANY one certificate is NOT a quark
# and the failed certificate is NAMED
# =========================================================================== #
class TestNegativeControls(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def _assert_named_failure(self, ev, name):
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn(name, read.failedCertificates)
        self.assertLess(read.confidence, 1.0)
        self.assertFalse(read.certificate.holds())
        return read

    def test_missing_anchor(self):
        ev = _certified_evidence()
        ev.anchor = obs.AnchorProfile()
        read = self._assert_named_failure(ev, "anchor")
        self.assertTrue(math.isnan(read.triangleAnchorScore))

    def test_low_anchor_score(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minAnchorScore = 1.5  # unreachable: a^2 <= 1
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertIn("anchor", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_even_parity(self):
        ev = _certified_evidence(occupations=(1.0, 1.0, 0.0))
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exteriorParity, +1)

    def test_uncertified_parity_never_emits_a_sign(self):
        ev = _certified_evidence()
        ev.parityRead = qm.WickCertificateRead()  # default: never holds
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exteriorParity, 0)

    def test_rank_two_band(self):
        ev = _certified_evidence()
        ev.colorBand = _unit_fiber(1, 2)
        read = self._assert_named_failure(ev, "color-rank-three")
        self.assertEqual(read.colorRank, 2)

    def test_unaccepted_band_gap_closed(self):
        ev = _certified_evidence()
        ev.colorBand = _unit_fiber(1, 3, accepted=False)
        self._assert_named_failure(ev, "color-rank-three")

    def test_leaking_transport(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        leaky = conn.transport(_unit_fiber(1, 3), _unit_fiber(11, 3),
                               np.diag([0.1, 1.0, 1.0]))
        self.assertFalse(leaky.accepted)
        ev.lifetimeTransports = list(ev.lifetimeTransports) + [leaky]
        self._assert_named_failure(ev, "transport-leakage")

    def test_missing_transports(self):
        ev = _certified_evidence()
        ev.lifetimeTransports = []
        read = self._assert_named_failure(ev, "transport-leakage")
        self.assertEqual(read.transportCount, 0)
        self.assertTrue(math.isnan(read.transportLeakageMax))

    def test_insufficient_persistence(self):
        ev = _certified_evidence()
        ev.persistenceLifetime = 1.0
        self._assert_named_failure(ev, "persistence")

    def test_missing_persistence(self):
        ev = _certified_evidence()
        ev.persistenceLifetime = NAN
        self._assert_named_failure(ev, "persistence")

    def test_low_track_overlap(self):
        ev = _certified_evidence()
        ev.persistenceMinOverlap = 0.2
        self._assert_named_failure(ev, "persistence")

    def test_low_localization(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minLocalization = 0.9  # fixture band carries 0.5
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertIn("localization", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_missing_refinement_stability(self):
        ev = _certified_evidence()
        ev.refinementOverlap = NAN
        self._assert_named_failure(ev, "refinement-stability")

    def test_unstable_refinement(self):
        ev = _certified_evidence()
        ev.refinementOverlap = 0.3
        self._assert_named_failure(ev, "refinement-stability")

    def test_gap_closed_winding_family_invalidates(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closedFamilyWinding(family + [bad])
        read = self._assert_named_failure(ev, "winding")
        self.assertIsNone(read.determinantWinding)
        self.assertIsNone(read.baryonFlux)


# =========================================================================== #
# the two-quark anti-triplet is not an antiquark
# =========================================================================== #
class TestAntiTriplet(unittest.TestCase):
    def test_anti_triplet_not_mislabeled_antiquark(self):
        # Two quarks in the Lambda^2 C^3 anti-triplet: the COLOR
        # representation looks like an antiquark (3bar) and the tube even
        # carries nu = -1 here — but total occupation is TWO and parity is
        # EVEN, and the classifier refuses on exactly those channels.
        pc = obs.ParticleClusters()
        ev = _certified_evidence(turns=-1, occupations=(1.0, 1.0, 0.0))
        read = pc.classifyQuark(ev)
        self.assertEqual(read.classification, "none")
        self.assertEqual(read.exteriorParity, +1)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertIn("parity-odd", read.failedCertificates)
        self.assertIn("occupation-one", read.failedCertificates)
        # the color-alone channels would NOT have refused:
        self.assertNotIn("color-rank-three", read.failedCertificates)
        self.assertNotIn("winding", read.failedCertificates)

    def test_top_wedge_triple_occupation_is_not_a_quark(self):
        # N = 3 (odd parity!) still fails: single-fermion occupation is a
        # separate certificate from parity.
        pc = obs.ParticleClusters()
        ev = _certified_evidence(occupations=(1.0, 1.0, 1.0))
        read = pc.classifyQuark(ev)
        self.assertEqual(read.exteriorParity, -1)
        self.assertIn("occupation-one", read.failedCertificates)
        self.assertNotIn("parity-odd", read.failedCertificates)
        self.assertEqual(read.classification, "none")


# =========================================================================== #
# conjugate-pair conservation
# =========================================================================== #
class TestConjugatePair(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_conjugate_pair_conserves(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        anti = self.pc.classifyQuark(_certified_evidence(turns=-1))
        pair = self.pc.conjugatePair(quark, anti)
        self.assertEqual(pair.totalWinding, 0)
        self.assertEqual(pair.totalBaryonFlux, 0.0)
        self.assertEqual(pair.totalParity, +1)
        self.assertTrue(pair.parityEven)
        self.assertTrue(pair.conserved)
        self.assertEqual(pair.failedCertificates, [])
        self.assertTrue(pair.certificate.holds())

    def test_singular_path_returns_unknown_flux(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=-1) + [bad])
        singular = self.pc.classifyQuark(ev)
        self.assertIsNone(singular.determinantWinding)
        pair = self.pc.conjugatePair(quark, singular)
        self.assertIsNone(pair.totalWinding)
        self.assertIsNone(pair.totalBaryonFlux)  # UNKNOWN, never zero
        self.assertFalse(pair.conserved)
        self.assertIn("winding-second", pair.failedCertificates)
        self.assertFalse(pair.certificate.holds())

    def test_non_conjugate_pair_fails_conservation(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        pair = self.pc.conjugatePair(quark, quark)
        self.assertEqual(pair.totalWinding, 2)
        self.assertAlmostEqual(pair.totalBaryonFlux, 2.0 / 3.0,
                               delta=MACHINE)
        self.assertFalse(pair.conserved)
        self.assertIn("winding-conservation", pair.failedCertificates)

    def test_uncertified_parity_leaves_total_parity_unknown(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        ev.parityRead = qm.WickCertificateRead()
        anti = self.pc.classifyQuark(ev)
        pair = self.pc.conjugatePair(quark, anti)
        self.assertEqual(pair.totalParity, 0)
        self.assertFalse(pair.parityEven)
        self.assertIn("parity-second", pair.failedCertificates)
        self.assertFalse(pair.conserved)

    def test_pair_of_odd_clusters_is_even(self):
        # whitepaper parity table: quark + antiquark -> even composite
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        anti = self.pc.classifyQuark(_certified_evidence(turns=-1))
        self.assertEqual(quark.exteriorParity * anti.exteriorParity, +1)
        self.assertEqual(self.pc.conjugatePair(quark, anti).totalParity, +1)


# =========================================================================== #
# the emergent flavor doublet (no requested dimension)
# =========================================================================== #
class TestFlavorDoublet(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_planted_doublet_emerges(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames())
        self.assertTrue(read.found)
        self.assertEqual(read.rank, 2)
        self.assertEqual(read.framesTracked, 3)
        self.assertEqual(read.twoStateCount, 1)
        self.assertAlmostEqual(read.minContinuationOverlap, 1.0,
                               delta=MACHINE)
        self.assertTrue(read.certificate.holds())
        # the search never requested a dimension: every stable rank is
        # reported, not only the two-state one
        self.assertEqual(sorted(read.stableSubclassRanks), [1, 2, 3])

    def test_no_two_state_subclass_is_unknown(self):
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(ranks=(1, 3)))
        self.assertFalse(read.found)
        self.assertIn("flavor-doublet", read.failedCertificates)
        self.assertEqual(read.invalidationReason,
                         "no-stable-two-state-subclass")
        self.assertFalse(read.certificate.holds())

    def test_doublet_dropping_out_is_unstable(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(drop_rank2_at=2))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidationReason,
                         "no-stable-two-state-subclass")

    def test_gap_closing_doublet_is_uncertified(self):
        # an unaccepted (gap-closed) band anywhere on the track breaks the
        # certified continuation — the #769 semantics
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(unaccept_rank2_at=1))
        self.assertFalse(read.found)

    def test_single_frame_is_insufficient(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(frames=1))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidationReason, "insufficient-frames")

    def test_ambiguous_two_doublets_stay_uncertified(self):
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(extra_rank2=True))
        self.assertFalse(read.found)
        self.assertEqual(read.twoStateCount, 2)
        self.assertEqual(read.invalidationReason,
                         "ambiguous-two-state-subclasses")

    def test_merging_chains_invalidate_each_other(self):
        # two rank-2 bands spanning the SAME cells at frame 0 both best-
        # match the single frame-1 band: the continuation is ambiguous
        f0 = _band_read([_unit_fiber(300, 2),
                         _fiber([[300], [301]], np.eye(2)[:, ::-1])])
        f1 = _band_read([_unit_fiber(300, 2)])
        read = self.pc.flavorDoubletSearch([f0, f1])
        self.assertFalse(read.found)

    def test_doublet_carries_the_recorded_trivialization(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(base=400))
        self.assertTrue(read.found)
        self.assertEqual(read.doublet.rank(), 2)
        self.assertEqual(read.doublet.cellVertices(), [[420], [421]])

    def test_search_is_deterministic(self):
        a = self.pc.flavorDoubletSearch(_doublet_frames())
        b = self.pc.flavorDoubletSearch(_doublet_frames())
        self.assertEqual(a.found, b.found)
        self.assertEqual(a.stableSubclassRanks, b.stableSubclassRanks)
        self.assertEqual(a.minContinuationOverlap, b.minContinuationOverlap)


# =========================================================================== #
# isospin, Gauss charge, and the proposed u/d identification
# =========================================================================== #
class TestIsospinCharge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_u_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.isospin, 0.5)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)
        self.assertTrue(read.udIdentificationProposed)
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertEqual(read.failedCertificates, [])

    def test_d_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=-1.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electricFlux, -1.0 / 3.0, delta=1e-9)
        self.assertTrue(read.udIdentificationProposed)

    def test_missing_doublet_yields_unknown_flavor_and_charge(self):
        # even with a CONSISTENT Gauss read, the quark charge stays
        # unknown without the doublet (ticket acceptance)
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electricFlux)
        self.assertIn("flavor-doublet", read.failedCertificates)
        self.assertEqual(read.classification, "quark")  # quark-ness intact

    def test_unstable_doublet_yields_unknown_flavor_and_charge(self):
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        ev.flavor = self.pc.flavorDoubletSearch(
            _doublet_frames(drop_rank2_at=1))
        ev.doubletOccupancy = np.array([1.0, 0.0], dtype=complex)
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electricFlux)
        self.assertIn("flavor-doublet", read.failedCertificates)

    def test_superposition_occupancy_yields_unknown_isospin(self):
        ev = _certified_evidence(with_flavor=True,
                                 occupancy=[1.0, 1.0])
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIn("isospin", read.failedCertificates)

    def test_inconsistent_gauss_yields_unknown_charge(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0])
        ev.charge = self.pc.gaussFluxConsistency(
            [complex(2.0 / 3.0), complex(1.0)])
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)
        # Q = I3 + B/2 is NOT tested without a certified charge
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)

    def test_violated_ud_relation_is_named(self):
        # occupancy says d (I3 = -1/2) but the Gauss flux says +2/3: the
        # proposed identification fails BY NAME; the independently
        # certified fields stay reported
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)

    def test_declared_orientation_is_recorded(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 orientation=-1, with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        # orientation -1: member 2 carries +1/2 under the declared
        # convention -> the SAME occupancy now reads as the u member
        self.assertEqual(read.isospin, 0.5)
        self.assertEqual(read.doubletOrientation, -1)
        self.assertTrue(read.udIdentificationProposed)

    def test_ud_not_tested_without_baryon_flux(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.3)]
        ev.winding = conn.openSegmentWinding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.baryonFlux)
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)


# =========================================================================== #
# the reused Gauss-flux read on nested enclosing surfaces
# =========================================================================== #
class TestGaussFlux(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_nested_surfaces_consistent_read(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertEqual(len(read.fluxes), 2)
        self.assertLess(read.maxDeviation, 1e-12)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.surfaceVertexCounts, [1, 4])

    def test_inconsistent_flux_is_unknown(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        # perturb one electric plaquette between the two surfaces
        es = cob.EigenstateSynthesis(st, 2)
        n = es.order()
        # find a cell seen by exactly one surface
        for c in range(n):
            probe = [0j] * n
            probe[c] = 1.0 + 0j
            s1 = es.gaussLawCharge(probe, surfaces[0], True)
            s2 = es.gaussLawCharge(probe, surfaces[1], True)
            if abs(s1 - s2) > 0.5:
                F2 = list(F)
                F2[c] += 1.0
                break
        read = self.pc.gaussFluxOnSurfaces(st, F2, surfaces, True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)
        self.assertFalse(read.certificate.holds())

    def test_all_spacelike_complex_reads_certified_zero(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        es = cob.EigenstateSynthesis(st, 2)
        surfaces = obs.ParticleClusters.nestedEnclosures(st, [0], 2)
        F = [0.7 + 0j] * es.order()
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertEqual(read.electricFlux, 0.0)  # a CERTIFIED zero
        self.assertEqual(read.failedCertificates, [])

    def test_single_surface_makes_no_consistency_claim(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gaussFluxOnSurfaces(st, F, [surfaces[0]], True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)

    def test_exact_field_strength_has_zero_total_flux(self):
        # topological protection: F = dA is exact, so the FULL (electric +
        # magnetic) closed-surface flux vanishes on every surface
        st = _from_simplices(7, _TETRA_CHAIN)
        es1 = cob.EigenstateSynthesis(st, 1)
        es2 = cob.EigenstateSynthesis(st, 2)
        rng = np.random.default_rng(7)
        A = [complex(v) for v in rng.normal(size=es1.order())]
        F = es2.curvatureFromConnection(A)
        surfaces = obs.ParticleClusters.nestedEnclosures(st, [0], 2)
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces,
                                           electric_only=False)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electricFlux, 0.0, delta=1e-12)

    def test_pure_combination_equals_spacetime_path(self):
        st, F, surfaces = _gauss_fixture(0.25)
        es = cob.EigenstateSynthesis(st, 2)
        fluxes = [es.gaussLawCharge(F, s, True) for s in surfaces]
        via_st = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        pure = self.pc.gaussFluxConsistency(fluxes, [1, 4], True)
        self.assertEqual(via_st.consistent, pure.consistent)
        self.assertEqual(via_st.electricFlux, pure.electricFlux)
        self.assertEqual(via_st.fluxes, pure.fluxes)

    def test_nested_enclosures_are_strictly_growing_here(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        sets_ = obs.ParticleClusters.nestedEnclosures(st, [0], 3)
        self.assertEqual(len(sets_), 3)
        self.assertEqual(sets_[0], [0])
        for a, b in zip(sets_, sets_[1:]):
            self.assertTrue(set(a) < set(b))

    def test_nested_enclosures_validates(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [], 2)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [0], 0)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [999], 2)

    def test_imaginary_leakage_is_reported_not_discarded(self):
        read = self.pc.gaussFluxConsistency([complex(0.5, 0.3),
                                             complex(0.5, 0.3)])
        self.assertFalse(read.consistent)  # |Im| above tolerance
        self.assertAlmostEqual(read.imagLeakage, 0.3, delta=MACHINE)
        self.assertIsNone(read.electricFlux)

    def test_gauss_read_is_read_only(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        before = st.metricRevisionKey()
        self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertEqual(st.metricRevisionKey(), before)


# =========================================================================== #
# relabeling / refinement / gauge invariance
# =========================================================================== #
class TestRelabelRefinementGauge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.delta = staticmethod(obs.ObservableGates.report_delta)

    def test_relabeling_preserves_accepted_classification(self):
        base = self.pc.classifyQuark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        shifted_ev = _certified_evidence(band_base=1001, with_flavor=True,
                                         occupancy=[1.0, 0.0],
                                         with_charge=2.0 / 3.0)
        shifted = self.pc.classifyQuark(shifted_ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(base.toRecord(),
                                             shifted.toRecord()), 0.0)

    def test_vertex_relabeling_of_the_gauss_complex(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        ids = [i + 500 for i in range(7)]
        st2, F2, surfaces2 = _gauss_fixture(2.0 / 3.0, ids=ids)
        a = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        b = self.pc.gaussFluxOnSurfaces(st2, F2, surfaces2, True)
        self.assertEqual(a.consistent, b.consistent)
        self.assertAlmostEqual(a.electricFlux, b.electricFlux, delta=1e-9)

    def test_refinement_preserves_accepted_classification(self):
        # the refined band adds cells while keeping the original subspace:
        # the measured overlap feeds the refinement certificate
        band = _unit_fiber(1, 3)
        refined = _fiber([[1], [2], [3], [77]],
                         np.vstack([np.eye(3), np.zeros((1, 3))]))
        overlap = obs.SpectralFiber.overlap(band, refined).subspaceOverlap
        ev = _certified_evidence()
        ev.refinementOverlap = overlap
        read = self.pc.classifyQuark(ev)
        self.assertAlmostEqual(overlap, 1.0, delta=1e-12)
        self.assertEqual(read.classification, "quark")
        self.assertNotIn("refinement-stability", read.failedCertificates)

    def test_in_band_gauge_rotation_leaves_the_verdict(self):
        # an SU(3) in-band frame change of the anchored frame leaves the
        # anchor profile (hence the classification evidence) invariant
        rng = np.random.default_rng(11)
        w = np.array([2.0, 0.5, 1.25])
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        phi = obs.ColorAnchor.orthonormalizeFrame(z, w)
        tri = [tessera.OrientedTriangle([0, 1, 2], [1, 1, 1])]
        p1 = obs.ColorAnchor(tri).evaluate(phi, w)
        # a special-unitary in-band rotation
        g = np.linalg.qr(rng.normal(size=(3, 3))
                         + 1j * rng.normal(size=(3, 3)))[0]
        g = g / np.linalg.det(g) ** (1.0 / 3.0)
        p2 = obs.ColorAnchor(tri).evaluate(phi @ g, w)
        r1 = self.pc.classifyQuark(_certified_evidence(anchor=p1))
        r2 = self.pc.classifyQuark(_certified_evidence(anchor=p2))
        self.assertEqual(r1.classification, r2.classification)
        self.assertAlmostEqual(r1.triangleAnchorScore,
                               r2.triangleAnchorScore, delta=1e-12)

    def test_transport_order_does_not_matter_for_leakage(self):
        ev = _certified_evidence()
        base = self.pc.classifyQuark(ev)
        ev.lifetimeTransports = list(reversed(list(ev.lifetimeTransports)))
        permuted = self.pc.classifyQuark(ev)
        self.assertEqual(base.transportLeakageMax,
                         permuted.transportLeakageMax)
        self.assertEqual(base.classification, permuted.classification)


# =========================================================================== #
# tracking, checkpointing, and the #764 cache
# =========================================================================== #
class TestTrackingCheckpointCache(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_track_candidates_across_frames(self):
        a1 = _certified_evidence(band_base=1)
        a2 = _certified_evidence(band_base=101)
        b1 = _certified_evidence(band_base=1)
        matches = obs.ParticleClusters.trackCandidates([a1, a2], [b1])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].fromIndex, 0)
        self.assertEqual(matches[0].toIndex, 0)
        self.assertTrue(matches[0].certifiedContinuation)

    def test_record_roundtrip_is_exact(self):
        read = self.pc.classifyQuark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        rec = read.toRecord()
        back = obs.QuarkRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)

    def test_unknown_fields_serialize_as_null_never_zero(self):
        read = self.pc.classifyQuark(obs.QuarkCandidateEvidence())
        rec = read.toRecord()
        self.assertIsNone(rec["baryon_flux"])
        self.assertIsNone(rec["isospin"])
        self.assertIsNone(rec["electric_flux"])
        self.assertIsNone(rec["determinant_winding"])
        self.assertEqual(rec["exterior_parity"], 0)
        self.assertTrue(math.isnan(rec["triangle_anchor_score"]))

    def test_full_evidence_is_checkpointed(self):
        read = self.pc.classifyQuark(_certified_evidence())
        rec = read.toRecord()
        for key in ("component_hash", "winding_closure",
                    "failed_certificates", "thresholds", "certificate",
                    "transport_leakage_max", "persistence_lifetime",
                    "localization", "refinement_overlap",
                    "occupation_total", "confidence"):
            self.assertIn(key, rec)
        self.assertEqual(rec["classification"], "quark")
        self.assertEqual(rec["winding_closure"], "closed-family")

    def test_from_record_rejects_unknown_schema(self):
        read = self.pc.classifyQuark(_certified_evidence())
        rec = read.toRecord()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.QuarkRead.fromRecord(rec)

    def test_describe_smoke(self):
        read = self.pc.classifyQuark(_certified_evidence())
        text = read.describe()
        self.assertIn("quark", text)
        self.assertIn("nu=1", text)

    def _spacetime_backed_evidence(self):
        # a real spacetime band so the cache has a live star to publish
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerateBands(list(range(7)), 0)
        fiber = bands.fibers[0]
        ev = _certified_evidence()
        ev.colorBand = fiber  # rank/acceptance of the REAL band applies
        return st, ev

    def test_cached_classification_equals_cold(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        cold = self.pc.classifyQuark(ev)
        first = self.pc.classifyQuarkCached(cache, ev)
        served = self.pc.classifyQuarkCached(cache, ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.toRecord(),
                                             first.toRecord()), 0.0)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.toRecord(),
                                             served.toRecord()), 0.0)
        self.assertGreaterEqual(cache.hits, 1)

    def test_touched_star_invalidates_only_the_touching_candidate(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classifyQuarkCached(cache, ev)
        self.assertEqual(cache.size, 1)
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)  # touches the band's support
        cache.publish(star)
        self.assertEqual(cache.size, 0)

    def test_disjoint_star_keeps_the_entry(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classifyQuarkCached(cache, ev)
        star = cob.TouchedStar()
        star.addChangedEdge(9001, 9002)  # disjoint from the band support
        cache.publish(star)
        self.assertEqual(cache.size, 1)

    def test_changed_evidence_never_serves_a_stale_read(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        quark = self.pc.classifyQuarkCached(cache, ev)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=-1))
        anti = self.pc.classifyQuarkCached(cache, ev)
        self.assertNotEqual(quark.determinantWinding,
                            anti.determinantWinding)

    def test_different_thresholds_have_different_fingerprints(self):
        ev = _certified_evidence()
        loose = obs.ParticleClusters()
        strict_cfg = obs.ParticleClustersConfig()
        strict_cfg.minAnchorScore = 0.99
        strict = obs.ParticleClusters(strict_cfg)
        self.assertNotEqual(loose.evidenceFingerprint(ev),
                            strict.evidenceFingerprint(ev))


# =========================================================================== #
# the emergence objective stays particle-blind; performance contract
# =========================================================================== #
class TestObjectiveGuardAndBenchmark(unittest.TestCase):
    def test_no_quark_quantity_enters_the_emergence_objective(self):
        # The emergence objective lives in the cobordism optimizer and the
        # RL harness; neither may reference the particle classifier.
        objective_homes = [REPO_ROOT / "src" / "cobordism",
                           REPO_ROOT / "include" / "cobordism",
                           REPO_ROOT / "src" / "rl",
                           REPO_ROOT / "include" / "rl",
                           REPO_ROOT / "src" / "simulations",
                           REPO_ROOT / "include" / "simulations"]
        offenders = []
        for home in objective_homes:
            if not home.exists():
                continue
            for path in home.rglob("*"):
                if path.suffix not in (".h", ".cpp", ".cu", ".hpp"):
                    continue
                text = path.read_text(errors="ignore")
                if "ParticleClusters" in text or "QuarkRead" in text:
                    offenders.append(str(path))
        self.assertEqual(offenders, [])

    def test_classification_is_read_only_on_the_spacetime(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerateBands(list(range(7)), 0)
        ev = _certified_evidence()
        ev.colorBand = bands.fibers[0]
        before = st.metricRevisionKey()
        obs.ParticleClusters().classifyQuark(ev)
        self.assertEqual(st.metricRevisionKey(), before)

    def test_classification_cost_per_candidate(self):
        # merge-gate benchmark: classification cost per candidate, cold
        # versus cache-served (numbers reported in the PR body)
        pc = obs.ParticleClusters()
        ev = _certified_evidence()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyQuark(ev)
        cold = (time.perf_counter() - t0) / n

        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        cache = cob.AnalyticCache(st)
        tracker = obs.SpectralFiberTracker(st)
        ev.colorBand = tracker.enumerateBands(list(range(7)), 0).fibers[0]
        pc.classifyQuarkCached(cache, ev)  # warm
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyQuarkCached(cache, ev)
        cached = (time.perf_counter() - t0) / n
        print(f"\n[benchmark] classifyQuark cold: {cold * 1e6:.1f} us; "
              f"cache-served: {cached * 1e6:.1f} us per candidate")
        self.assertLess(cold, 0.05)  # generous ceiling: 50 ms/candidate


if __name__ == "__main__":
    unittest.main()
