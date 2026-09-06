# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The simplicial qubit (#955): a marked torus's harmonic zero mode read as a
point of CP^1. The flat torus C/(Z + tau Z) is the exact reference (the
construction is exact there, so those are equality tests to rounding); the
remaining statements are the specification's invariances, certificates and
refusals (docs/design/simplicial_qubit_spec.md, sections 12-15)."""
import cmath
import math

import numpy as np
import pytest

import tessera
from tessera import observables as obs

SimplicialQubit = obs.SimplicialQubit

# (name, tau, expected Bloch vector) -- specification section 12.
REFERENCE = [
    ("square", 1j, (0.0, 1.0, 0.0)),
    ("rectangle r=2", 2j, (0.0, 4.0 / 5.0, -3.0 / 5.0)),
    ("shear s=0.3", 0.3 + 1j, (0.6 / 2.09, 2.0 / 2.09, -0.09 / 2.09)),
    ("hexagonal", cmath.exp(1j * math.pi / 3), (0.5, math.sqrt(3) / 2, 0.0)),
]


def _read(tau, nx=4, ny=4):
    torus = SimplicialQubit.flatTorus(tau, nx, ny)
    return torus, torus.qubit.read(torus.spacetime)


def _edges(st):
    return list(st.getEdgeList().toVector())


# ---------------------------------------------------------------------------
# The flat torus: exact reference cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,tau,bloch", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_flat_torus_reference_table(name, tau, bloch):
    torus, r = _read(tau)
    assert r.holds(), r.refusal
    assert r.harmonicRank == 2 and r.twistedHarmonicRank == 2
    assert r.betti == [1, 2, 1] and r.eulerCharacteristic == 0
    assert (r.vertices, r.edges, r.faces) == (16, 48, 32)
    assert abs(r.intersectionNumber - 1) < 1e-9
    assert not r.markingSwapped
    # Exact on a flat torus: J^2 = -I and tau to rounding.
    assert r.complexStructureResidual < 1e-10
    assert abs(r.tau - tau) < 1e-10
    assert np.allclose(r.bloch, bloch, atol=1e-10)
    assert abs(r.blochNorm - 1.0) < 1e-12
    rho = np.asarray(r.density)
    assert np.allclose(rho, rho.conj().T) and abs(np.trace(rho) - 1) < 1e-12
    assert np.allclose(rho @ rho, rho, atol=1e-12)
    psi = np.asarray(r.state)
    assert np.allclose(np.outer(psi, psi.conj()), rho)
    assert abs(SimplicialQubit.periodRatioOf(psi) - tau) < 1e-10
    # The headline scalar is the residual.
    assert torus.qubit.compute(torus.spacetime) == pytest.approx(r.complexStructureResidual)


def test_tau_is_refinement_invariant_on_a_flat_torus():
    tau = 0.3 + 1.2j
    for nx, ny in [(3, 3), (4, 6), (8, 8)]:
        _, r = _read(tau, nx, ny)
        assert r.holds(), r.refusal
        assert abs(r.tau - tau) < 1e-9, (nx, ny, r.tau)
        assert r.complexStructureResidual < 1e-9


def test_tau_is_invariant_under_uniform_scaling():
    tau = 0.2 + 0.9j
    torus, before = _read(tau)
    for e in _edges(torus.spacetime):
        e.setLength(2.7 * e.getLength())
    after = torus.qubit.read(torus.spacetime)
    assert after.holds()
    assert abs(after.tau - before.tau) < 1e-10
    # The Gram scales, the intersection form does not.
    assert np.allclose(np.asarray(after.intersection), np.asarray(before.intersection), atol=1e-12)


def test_modular_transformations_of_the_marking():
    tau = 0.35 + 1.1j
    torus, base = _read(tau)
    a, b, reversed_ = torus.qubit.cycleA(), torus.qubit.cycleB(), torus.qubit.reversed()
    # A' = B, B' = -A  ->  tau' = -1/tau (A'.B' = +1 still).
    swapped = SimplicialQubit(list(b), list(reversed(a)), reversed_)
    r = swapped.read(torus.spacetime)
    assert r.holds(), r.refusal
    assert abs(r.tau - (-1.0 / tau)) < 1e-9
    # B' = A + B (the concatenated closed walk)  ->  tau' = tau + 1.
    shifted = SimplicialQubit(list(a), list(a) + list(b), reversed_)
    r = shifted.read(torus.spacetime)
    assert r.holds(), r.refusal
    assert abs(r.tau - (tau + 1.0)) < 1e-9
    assert abs(SimplicialQubit.fubiniStudyDistance(base.tau, base.tau)) < 1e-12


# ---------------------------------------------------------------------------
# A non-flat metric: the residual is a discretization error
# ---------------------------------------------------------------------------

def _conformal_square_torus(n, amplitude=0.3):
    """The square torus with lengths scaled by exp(phi) at each edge midpoint,
    phi = amplitude sin(2 pi x) cos(2 pi y): a conformally flat metric, whose
    conformal structure is still that of tau = i."""
    torus = SimplicialQubit.flatTorus(1j, n, n)

    def reduce(delta):
        if delta == n - 1:
            return -1
        if delta == -(n - 1):
            return 1
        return delta

    for e in _edges(torus.spacetime):
        u, v = e.getSource().getId(), e.getTarget().getId()
        iu, ju = divmod(u, n)
        iv, jv = divmod(v, n)
        di, dj = reduce(iv - iu), reduce(jv - ju)
        x = (iu + 0.5 * di) / n
        y = (ju + 0.5 * dj) / n
        phi = amplitude * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
        e.setLength(e.getLength() * math.exp(phi))
    return torus


def test_residual_and_tau_converge_under_refinement_for_a_conformal_metric():
    residuals, errors = [], []
    for n in (4, 8, 16):
        torus = _conformal_square_torus(n)
        r = torus.qubit.read(torus.spacetime)
        assert r.holds(), r.refusal
        residuals.append(r.complexStructureResidual)
        errors.append(abs(r.tau - 1j))
    assert residuals[0] > 1e-6  # a genuinely non-flat mesh
    assert residuals[0] > residuals[1] > residuals[2]
    assert errors[0] > errors[1] > errors[2]


# ---------------------------------------------------------------------------
# Phases on: pure gauge is invisible, holonomy and flux refuse by name
# ---------------------------------------------------------------------------

def test_pure_gauge_phases_leave_tau_invariant():
    tau = 0.1 + 1.3j
    torus, before = _read(tau)
    rng = np.random.default_rng(3)
    g = {v.getId(): rng.uniform(-1.5, 1.5) for v in torus.spacetime.getVertexList().toVector()}
    for e in _edges(torus.spacetime):  # phi_e = g(target) - g(source): a gauge
        e.setPhase(complex(g[e.getTarget().getId()] - g[e.getSource().getId()], 0.0))
    after = torus.qubit.read(torus.spacetime)
    assert after.holds(), after.refusal
    assert after.twistedHarmonicRank == 2
    assert abs(after.tau - before.tau) < 1e-10


def test_flux_refuses_by_name():
    torus, _ = _read(1j)
    edges = _edges(torus.spacetime)
    edges[0].setPhase(complex(0.7, 0.0))  # curvature on the two faces of one edge
    r = torus.qubit.read(torus.spacetime)
    assert not r.holds()
    assert "holonomy or flux" in r.refusal
    assert r.twistedHarmonicRank < 2
    assert math.isnan(torus.qubit.compute(torus.spacetime))


def test_flat_holonomy_refuses_by_name():
    n = 4
    torus, _ = _read(1j, n, n)
    # A flat connection with holonomy around the row loop: the same phase on
    # every edge crossing the seam between columns n-1 and 0.
    for e in _edges(torus.spacetime):
        iu, iv = e.getSource().getId() // n, e.getTarget().getId() // n
        if {iu, iv} == {0, n - 1}:
            e.setPhase(complex(0.9 if iu == 0 else -0.9, 0.0))
    r = torus.qubit.read(torus.spacetime)
    assert not r.holds()
    assert "holonomy or flux" in r.refusal


# ---------------------------------------------------------------------------
# Degeneration and distances
# ---------------------------------------------------------------------------

def test_pinching_cycle_warns_and_moves_the_state_to_a_pole():
    torus = SimplicialQubit.flatTorus(0.05j, 4, 4)
    strict = SimplicialQubit(torus.qubit.cycleA(), torus.qubit.cycleB(), torus.qubit.reversed(),
                             degeneracy_threshold=10.0)
    r = strict.read(torus.spacetime)
    assert r.holds(), r.refusal          # a warning, never a failure
    assert r.nearDegenerate and "pinching" in r.warning
    assert abs(r.tau - 0.05j) < 1e-9
    assert r.bloch[2] > 0.99             # towards |0>
    assert math.isfinite(SimplicialQubit.fubiniStudyDistance(r.tau, 1j))
    assert SimplicialQubit.weilPeterssonDistance(r.tau, 1j) > SimplicialQubit.weilPeterssonDistance(0.5j, 1j)


def test_distances():
    assert SimplicialQubit.weilPeterssonDistance(1j, 2j) == pytest.approx(math.log(2))
    assert SimplicialQubit.weilPeterssonDistance(1j, 1j) == 0.0
    assert SimplicialQubit.fubiniStudyDistance(1j, 1j) == 0.0
    # |0>-like against |1>-like: antipodal states are pi/2 apart.
    assert SimplicialQubit.fubiniStudyDistance(1e-9j, 1e9j) == pytest.approx(math.pi / 2, abs=1e-6)
    assert SimplicialQubit.fubiniStudyDistance(1j, 2j) == SimplicialQubit.fubiniStudyDistance(2j, 1j)
    with pytest.raises(ValueError):
        SimplicialQubit.weilPeterssonDistance(1j, -1j)
    with pytest.raises(ValueError):
        SimplicialQubit.periodRatioOf(np.array([0.0, 1.0], dtype=complex))


# ---------------------------------------------------------------------------
# Certificates and refusals
# ---------------------------------------------------------------------------

def test_marking_certificates():
    torus, _ = _read(1j)
    a, b, rev = torus.qubit.cycleA(), torus.qubit.cycleB(), torus.qubit.reversed()
    r = SimplicialQubit(list(a), list(a), rev).read(torus.spacetime)
    assert "dependent" in r.refusal
    r = SimplicialQubit(list(b), list(a), rev).read(torus.spacetime)
    assert "A·B = -1" in r.refusal
    r = SimplicialQubit(list(a), list(b), not rev).read(torus.spacetime)
    assert "A·B = -1" in r.refusal
    with pytest.raises(ValueError):
        SimplicialQubit([0, 10], list(b), rev).read(torus.spacetime)  # (0,0)-(2,2) is no edge
    with pytest.raises(ValueError):
        SimplicialQubit([0], list(b), rev).read(torus.spacetime)


def test_not_a_torus_refuses_by_name():
    cells = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]  # the 2-sphere
    st = tessera.Spacetime.fromCells(2, cells, 1.0, 0.0)
    r = SimplicialQubit([0, 1, 2], [0, 1, 3]).read(st)
    assert "not a torus" in r.refusal
    assert r.betti == [1, 0, 1]
    # A disk: boundary.
    st = tessera.Spacetime.fromCells(2, [[0, 1, 2], [0, 2, 3]], 1.0, 0.0)
    r = SimplicialQubit([0, 1, 2], [0, 2, 3]).read(st)
    assert "boundary" in r.refusal
    with pytest.raises(ValueError):
        SimplicialQubit.flatTorus(-1j, 4, 4)
    with pytest.raises(ValueError):
        SimplicialQubit.flatTorus(1j, 2, 4)


def test_record_splits_complex_channels():
    _, r = _read(0.4 + 1.5j)
    rec = r.toRecord()
    assert rec["refusal"] == "" and rec["harmonic_rank"] == 2
    assert rec["tau_re"] == pytest.approx(0.4) and rec["tau_im"] == pytest.approx(1.5)
    assert len(rec["holomorphic_form_re"]) == r.edges
    assert len(rec["state_re"]) == 2 and len(rec["density_im"]) == 4
    assert rec["bloch"] == pytest.approx(list(r.bloch))


# ---------------------------------------------------------------------------
# The building blocks
# ---------------------------------------------------------------------------

def test_polygon_circle_product_is_the_grid_torus():
    with pytest.raises(ValueError):
        tessera.PolygonCircle(2).build(tessera.Spacetime(), 0)
    _, r = _read(1j, 5, 3)
    assert (r.vertices, r.edges, r.faces) == (15, 45, 30)
    assert r.holds()


def test_cup_product_form_is_the_intersection_form():
    torus, r = _read(0.3 + 1j)
    K = tessera.chainhodge.WhitneyMass.complexOf(torus.spacetime)
    form = K.cupProductForm(1)
    assert form.degree == 1 and form.rows == form.cols == r.edges
    assert len(form.front) == r.faces and set(form.orientation) <= {-1, 1}
    Z = np.asarray(r.harmonicImages)
    z0, z1 = [complex(v) for v in Z[:, 0]], [complex(v) for v in Z[:, 1]]
    r01 = form.evaluate(z0, z1)
    r10 = form.evaluate(z1, z0)
    assert abs(r01 + r10) < 1e-12 and abs(form.evaluate(z0, z0)) < 1e-12
    sign = -1.0 if torus.qubit.reversed() else 1.0
    assert abs(sign * r01 - np.asarray(r.intersection)[0, 1]) < 1e-12
    with pytest.raises(ValueError):
        form.evaluate(z0[:-1], z1)
    with pytest.raises(RuntimeError):  # a disk has no fundamental class
        tessera.cobordism.ChainComplex.fromTopCells([[0, 1, 2], [0, 2, 3]]).cupProductForm(1)
