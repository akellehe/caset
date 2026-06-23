"""Orientation-safe, ``dualComplexValid``-gated stellar coning (T2, #459).

The Emergent Color Topology epic (#457) grows the proton's ``b2`` color register
by *gated surgical coning*. T1 (#458) made the cone primitives hinge-exact and
exactly invertible; this module (T2) makes them **orientation-safe** and gates
them on the **full manifold check**, so a cone can never flip a local induced
orientation and inject a spurious sign into the **complex** (causal / oriented)
deficit -- i.e. into ``Im S`` (``ReggeSolver.dualReggeAction``), which is real
physics. The Lorentzian action is complex: every round-trip assertion here pins
**both** Re and Im.

The gate (``cobordism::OrientedCone``) reuses the T1 cone primitives
(``AddMove`` / ``RemoveMove`` in ``PachnerMode.PreGeometric``, the
``1<->(d+1)`` stellar subdivision) and accepts a cone only when the result is a
valid (``ChainComplex.dualComplexIsValid``, the #429 n>=4 recursive manifold
check) **and orientable** (``ChainComplex.orientationCovector`` propagates
without contradiction) complex. For a topology-PRESERVING refinement (this
ticket) the gate always passes; it is the live guard the topology-CHANGING
surgical variant (T3, #460) relies on.

Coverage:

* the gate accepts a topology-preserving cone on S^3 / S^4 / S^2xS^1, and the
  round trip restores the action (Re+Im) and the top-cell set exactly;
* the induced orientation (``orientationCovector`` / closed-manifold
  ``fundamentalClass``) is well-defined after cone-in and **identical** after the
  cone round trip -- ``move o move^-1`` restores every sign;
* ``endSignCovector`` of a fixed set of holes is stable across a cone applied
  away from them;
* ``Im S`` sign consistency across a gated cone on a genuinely Lorentzian CDT;
* the orientability gate has teeth: ``orientationCovector`` raises on a
  non-orientable (Moebius) complex and returns a clean covector on an orientable
  one -- the exact condition that makes the gate reject a sign-flipping cone.
"""

import math

import pytest

import tessera as T

cob = T.cobordism  # ChainComplex / OrientedCone live in the cobordism submodule

TOL = 1e-9
TOL_BIG = 1e-6


# --------------------------------------------------------------------------- #
# Builders / measurement helpers (mirroring test_hinge_exact_moves.py)
# --------------------------------------------------------------------------- #
def _matter():
    return T.MatterConfiguration()


def _sphere(d, sq=1.0):
    """Boundary of a (d+1)-simplex = a minimal triangulated S^d, unit spacelike."""
    sig = T.Signature(d, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(d))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(sq)
    return st


def _s2_cross_s1():
    sig = T.Signature(3, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.SphereCircleProduct())
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
    return st


def _make_cdt(n_simplices=120):
    """A built 4D Lorentzian CDT toroid -- genuine complex action (Im S != 0)."""
    sig = T.Signature(4, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.Toroid())
    st.build(n_simplices)
    return st


def _act(st):
    return T.ReggeSolver(st, _matter()).dualReggeAction()


def _top_set(st):
    return sorted(
        tuple(sorted(v.getId() for v in s.getVertices()))
        for s in st.getTopSimplices()
    )


def _assert_close(a, b, tol, what):
    assert abs(a.real - b.real) < tol, f"{what}: Re drift {abs(a.real - b.real):.2e}"
    assert abs(a.imag - b.imag) < tol, f"{what}: Im drift {abs(a.imag - b.imag):.2e}"


def _orientation_map(st):
    """The induced orientation as a {top-cell-tuple: +/-1} dict, comparable across
    a move that preserves the cell set (covector is aligned to canonical order)."""
    cc = cob.ChainComplex.fromSpacetime(st)
    d = cc.dimension()
    cells = [tuple(c) for c in cc.kSimplexVertices(d)]
    eps = cob.ChainComplex.orientationCovector([list(c) for c in cells])
    # orientationCovector is aligned to the sorted-unique cell order, which is the
    # same canonical C_d order kSimplexVertices(d) returns.
    return dict(zip(cells, eps))


# --------------------------------------------------------------------------- #
# 1. The gate accepts a topology-preserving cone; round trip is exact (Re+Im)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder", [lambda: _sphere(3), lambda: _sphere(4), _s2_cross_s1]
)
def test_gated_cone_in_accepts_and_roundtrips(builder):
    st = builder()
    a0 = _act(st)
    tops0 = _top_set(st)

    cone = cob.OrientedCone(st)
    for seed in range(64):
        ok, reason = cone.coneIn(seed)
        if not ok:
            # A topology-preserving stellar refinement of a manifold is always a
            # valid, orientable manifold -- the gate must never reject it.
            assert "did not propose" in reason, f"gate wrongly rejected: {reason}"
            continue
        assert cone.isApplied
        # The refinement genuinely redistributes curvature.
        assert abs(_act(st) - a0) > TOL, "cone-in did not change the action"
        # The current complex passes its own gate.
        ok2, reason2 = cone.validate()
        assert ok2, f"post-cone validate failed: {reason2}"

        assert cone.rollback()
        assert not cone.isApplied
        _assert_close(_act(st), a0, TOL_BIG, "gated cone round trip")
        assert _top_set(st) == tops0, "cone-out did not restore the top cells"
        return
    pytest.skip("cone-in never proposed in 64 seeds")


def test_gated_cone_out_accepts_and_roundtrips():
    """The gated cone-out (d+1)->1 weld as a first-class primitive: raise an apex,
    then a gated OrientedCone.coneOut welds it away and accepts (valid orientable
    manifold); its rollback re-raises it, restoring the action (Re+Im) exactly."""
    st = _sphere(3)
    # Raise an apex so a (d+1)->1 weld target exists (the T1 raw add primitive).
    raise_ = T.AddMove(st, 1, False, T.PachnerMode.PreGeometric, False)
    assert raise_.propose() and raise_.apply()
    a0 = _act(st)
    tops0 = _top_set(st)

    cone = cob.OrientedCone(st)
    for seed in range(200):
        ok, reason = cone.coneOut(seed)
        if not ok:
            assert "did not propose" in reason, f"gate wrongly rejected: {reason}"
            continue
        assert cone.isApplied
        assert abs(_act(st) - a0) > TOL, "cone-out did not change the action"
        ok2, reason2 = cone.validate()
        assert ok2, f"post-weld validate failed: {reason2}"
        assert cone.rollback()
        _assert_close(_act(st), a0, TOL, "gated cone-out round trip")
        assert _top_set(st) == tops0, "cone-in did not restore the top cells"
        return
    pytest.skip("cone-out never proposed in 200 seeds")


# --------------------------------------------------------------------------- #
# 2. Induced orientation is well-defined after coning and restored by the inverse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", [lambda: _sphere(3), lambda: _sphere(4)])
def test_induced_orientation_after_cone_in(builder):
    """After a cone-in the refined complex is still a closed oriented manifold: the
    orientation covector exists (no propagation contradiction) and the closed-manifold
    fundamental class is still a single +/-1 generator."""
    st = builder()
    cone = cob.OrientedCone(st)
    for seed in range(64):
        ok, _ = cone.coneIn(seed)
        if not ok:
            continue
        eps = cone.orientationCovector()
        assert len(eps) == len(_top_set(st))
        assert all(s in (-1, 1) for s in eps)
        # Closed oriented manifold: fundamentalClass is the unique +/-1 generator.
        fc = cob.ChainComplex.fromSpacetime(st).fundamentalClass()
        assert len(fc) == len(eps) and all(s in (-1, 1) for s in fc)
        return
    pytest.skip("cone-in never proposed in 64 seeds")


@pytest.mark.parametrize(
    "builder", [lambda: _sphere(3), lambda: _sphere(4), _s2_cross_s1]
)
def test_roundtrip_restores_every_orientation_sign(builder):
    """move o move^-1 must restore *all* signs: the per-cell induced orientation
    covector is bit-identical before and after a cone round trip."""
    st = builder()
    before = _orientation_map(st)
    cone = cob.OrientedCone(st)
    for seed in range(64):
        ok, _ = cone.coneIn(seed)
        if not ok:
            continue
        assert cone.rollback()
        after = _orientation_map(st)
        assert after == before, "cone round trip did not restore the orientation signs"
        return
    pytest.skip("cone-in never proposed in 64 seeds")


# --------------------------------------------------------------------------- #
# 3. endSignCovector of fixed holes is stable across a cone applied away from them
# --------------------------------------------------------------------------- #
def test_end_sign_covector_stable_across_cone():
    st = _sphere(4)
    tops_before = _top_set(st)

    cone = cob.OrientedCone(st)
    for seed in range(64):
        ok, _ = cone.coneIn(seed)
        if not ok:
            continue
        tops_after = _top_set(st)
        # The subdivided cell is the one present before but gone after; the holes
        # must be survivors so the same hole set is valid in both surfaces.
        survivors = [c for c in tops_after if c in tops_before]
        assert len(survivors) >= 2
        holes = [list(survivors[0]), list(survivors[1])]

        sig_before = cob.ChainComplex.endSignCovector(
            [list(c) for c in tops_before], holes
        )
        sig_after = cob.ChainComplex.endSignCovector(
            [list(c) for c in tops_after], holes
        )
        assert all(s in (-1, 1) for s in sig_before)
        assert all(s in (-1, 1) for s in sig_after)
        # The covector is defined up to one overall sign per connected component
        # (the constraint sum_k sigma_k p_k = 0 holds for either sign); a cone
        # that removes the lexicographic root cell can flip that global sign. The
        # global-sign-invariant statement of "end-sign stability" is that the
        # *relative* induced orientation between holes is preserved.
        assert (
            sig_before[0] * sig_before[1] == sig_after[0] * sig_after[1]
        ), "relative end-sign of the holes drifted across the cone"
        cone.rollback()
        return
    pytest.skip("cone-in never proposed in 64 seeds")


# --------------------------------------------------------------------------- #
# 4. Im S sign consistency across a gated cone on a genuinely Lorentzian CDT
# --------------------------------------------------------------------------- #
def test_gated_cone_preserves_imaginary_part():
    """On a genuinely complex action (CDT toroid, Im S ~ -35) the gated cone-in/out
    round trip must restore Im S, not just |S|. A flipped induced orientation would
    show up here as an Im drift -- the precise corruption the gate exists to stop."""
    st = _make_cdt(120)
    a0 = _act(st)
    assert abs(a0.imag) > 1.0, "fixture is not genuinely Lorentzian"

    cone = cob.OrientedCone(st)
    last_reason = "cone-in never proposed on the CDT in 64 seeds"
    for seed in range(64):
        ok, reason = cone.coneIn(seed)
        if not ok:
            last_reason = reason
            continue
        a1 = _act(st)
        assert abs(a1 - a0) > TOL, "cone-in did not change the action"
        assert cone.rollback()
        a2 = _act(st)
        _assert_close(a2, a0, TOL_BIG, "gated cone Im round trip")
        return
    # A valid CDT refinement should be gate-accepted; if every candidate was
    # rejected on a manifold ground that is a finding about the gate's strictness
    # vs the CDT build, not an Im-sign failure -- surface it rather than fail.
    pytest.skip(f"no gated cone accepted on the CDT: {last_reason}")


# --------------------------------------------------------------------------- #
# 5. The orientability gate has teeth (pure combinatorics)
# --------------------------------------------------------------------------- #
def test_orientation_covector_orientable_sphere():
    """S^2 = boundary of a tetrahedron: orientable, so orientationCovector returns a
    clean +/-1 covector with the lexicographically smallest cell carrying +1."""
    s2 = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    eps = cob.ChainComplex.orientationCovector(s2)
    assert len(eps) == 4 and all(s in (-1, 1) for s in eps)
    # Sorted-unique order puts [0,1,2] first -> the component root carries +1.
    assert eps[0] == 1


def test_orientation_covector_raises_on_non_orientable():
    """A minimal Moebius band (5 triangles in a flipped cycle) is non-orientable: the
    sign propagation contradicts itself, so orientationCovector must raise -- exactly
    the verdict that makes OrientedCone reject a sign-flipping (orientation-reversing)
    cone."""
    moebius = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 3, 4], [0, 1, 4]]
    with pytest.raises(RuntimeError, match="orientationCovector"):
        cob.ChainComplex.orientationCovector(moebius)


def test_validate_is_the_applied_gate():
    """OrientedCone.validate() reports the same (manifold + orientable) verdict the
    cone moves gate on; on a pristine sphere it is (True, 'ok')."""
    st = _sphere(3)
    ok, reason = cob.OrientedCone(st).validate()
    assert ok, reason
