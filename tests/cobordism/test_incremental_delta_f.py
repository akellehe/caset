# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Incremental, hinge-local ΔS_Regge accounting (#461, T4).

The Emergent Color Topology optimizer (#457) evaluates a candidate objective for
every move, every step; a full recompute is the #418 cost spike. The geometry term
of `F` is `‖∇S_Regge‖²` (extremize the action, δS=0), built hinge-locally from the
dual (Sorkin) Regge action `S = Σ_h |★h|·ε_h`. This module pins the foundational
accounting: the localized dual action over a FIXED affected-hinge set, evaluated
across a move, reproduces the full `dualReggeAction` delta to machine precision —
for an edge-length perturbation and for a Pachner move alike.

`ReggeSolver.hingeFacesOfCells` builds the affected-hinge set (the (d-2)-faces of a
move's touched cells); `ReggeSolver.dualReggeActionOverHinges` sums `|★h|·ε_h` over
exactly those genuine hinges, term-for-term identical to `dualReggeAction`.
"""
import unittest

import tessera as T

_TOL = 1e-12


def _sphere4(jitter=True):
    """A minimal triangulated S⁴ (boundary of a 5-simplex), unit spacelike, lightly
    jittered so the dual Regge action is nontrivial."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + (0.013 * (i % 5) if jitter else 0.0))
    return st


def _tops(st):
    return {tuple(sorted(v.getId() for v in s.getVertices()))
            for s in st.getTopSimplices()}


class IncrementalDeltaSReggeTest(unittest.TestCase):
    def test_over_all_hinges_equals_full_action(self):
        # The localized action over the full hinge set IS the full action: same
        # per-term measure (circumcentric dualVolume), genuine-only.
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        tops = [list(c) for c in _tops(st)]
        full = rs.dualReggeAction()
        over_all = rs.dualReggeActionOverHinges(rs.hingeFacesOfCells(tops))
        self.assertLess(abs(full - over_all), _TOL)
        # the action is genuinely complex (Lorentzian/Sorkin) on this build
        self.assertGreater(abs(full), 1e-6)

    def test_delta_under_edge_perturbation_is_exact(self):
        # ΔS over the FIXED affected-hinge set == full Δ, to machine precision.
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        e = st.getEdgeList().toVector()[3]
        ev = {e.getSource().getId(), e.getTarget().getId()}
        aff = [list(c) for c in _tops(st) if ev.issubset(set(c))]
        hinges = rs.hingeFacesOfCells(aff)
        self.assertTrue(hinges)

        before_full = rs.dualReggeAction()
        before_loc = rs.dualReggeActionOverHinges(hinges)
        orig = e.getSquaredLength()
        e.setSquaredLength(orig * 1.07)
        after_full = rs.dualReggeAction()
        after_loc = rs.dualReggeActionOverHinges(hinges)
        e.setSquaredLength(orig)

        self.assertLess(abs((after_full - before_full) - (after_loc - before_loc)),
                        _TOL)

    def test_delta_under_pachner_move_is_exact(self):
        # Across a combinatorial Pachner move (a PreGeometric 1→(d+1) stellar cone-in),
        # the affected region = the symmetric difference of the top-cell set; ΔS over
        # its hinges == full Δ exactly. Two *fresh* deterministic complexes give the
        # before/after states, so the accounting check never leans on rollback fidelity
        # (the move's invertibility is #458's concern, not T4's).
        st_before = _sphere4()
        rs_before = T.ReggeSolver(st_before, T.MatterConfiguration())

        st_after = _sphere4()                       # identical fresh copy
        rs_after = T.ReggeSolver(st_after, T.MatterConfiguration())
        mv = T.AddMove(st_after, 5, False, T.PachnerMode.PreGeometric, False)
        self.assertTrue(mv.propose(), "stellar cone-in did not propose")
        self.assertTrue(mv.apply())

        affected = [list(c) for c in (_tops(st_before) ^ _tops(st_after))]
        self.assertTrue(affected, "a Pachner move must change the top-cell set")
        # pure topology ⇒ the same hinge tuples resolve on either complex (absent
        # tuples — e.g. ones using the fresh apex vertex — contribute 0 on `before`).
        hinges = rs_after.hingeFacesOfCells(affected)

        d_full = rs_after.dualReggeAction() - rs_before.dualReggeAction()
        d_loc = (rs_after.dualReggeActionOverHinges(hinges)
                 - rs_before.dualReggeActionOverHinges(hinges))
        self.assertLess(abs(d_full - d_loc), _TOL)


def _cdt_toroid(n_simplices=200):
    """A built 4D Lorentzian CDT toroid — a genuinely large complex (Im S ≠ 0),
    where the hinge-local update touches far fewer than all edges."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED, T.Toroid())
    st.build(n_simplices)
    return st


def _grad_norm2(rs):
    """‖∇S_Regge‖² = Σ_e |∂S/∂ℓ²_e|² from the exact analytic gradient."""
    return sum(abs(z) ** 2 for z in rs.actionGradientExact())


class GradientNormObjectiveTest(unittest.TestCase):
    """The geometry term of `F = ‖∇S_Regge‖² + Γ·r_U` (extremize, δS=0), localized."""

    def test_over_all_edges_equals_action_gradient_exact(self):
        # The localized gradient norm over the full edge set IS ‖∇S_Regge‖².
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        all_edges = [(e.getSource().getId(), e.getTarget().getId())
                     for e in st.getEdgeList().toVector()]
        self.assertLess(abs(rs.gradientNorm2OverEdges(all_edges) - _grad_norm2(rs)),
                        _TOL)

    def test_delta_under_edge_perturbation_is_exact(self):
        # Δ‖∇S_Regge‖² over the FIXED affected-edge set == full Δ, machine precision.
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        e = st.getEdgeList().toVector()[3]
        ev = {e.getSource().getId(), e.getTarget().getId()}
        aff = [list(c) for c in _tops(st) if ev.issubset(set(c))]
        E = rs.affectedEdgesOfCells(aff)
        self.assertTrue(E)

        before_full, before_loc = _grad_norm2(rs), rs.gradientNorm2OverEdges(E)
        orig = e.getSquaredLength()
        e.setSquaredLength(orig * 1.07)
        after_full, after_loc = _grad_norm2(rs), rs.gradientNorm2OverEdges(E)
        e.setSquaredLength(orig)

        self.assertLess(abs((after_full - before_full) - (after_loc - before_loc)),
                        _TOL)

    def test_delta_under_pachner_move_is_exact(self):
        # Across a PreGeometric stellar cone-in, Δ‖∇S_Regge‖² over the affected-edge
        # set (union of the before/after evaluations, fixed across the move) == full Δ.
        st_before = _sphere4()
        rs_before = T.ReggeSolver(st_before, T.MatterConfiguration())
        st_after = _sphere4()
        rs_after = T.ReggeSolver(st_after, T.MatterConfiguration())
        mv = T.AddMove(st_after, 5, False, T.PachnerMode.PreGeometric, False)
        self.assertTrue(mv.propose() and mv.apply())

        affected = [list(c) for c in (_tops(st_before) ^ _tops(st_after))]
        # cells are added/removed ⇒ union the affected-edge set on both complexes
        E = sorted(set(map(tuple, rs_before.affectedEdgesOfCells(affected)))
                   | set(map(tuple, rs_after.affectedEdgesOfCells(affected))))
        self.assertTrue(E)

        d_full = _grad_norm2(rs_after) - _grad_norm2(rs_before)
        d_loc = (rs_after.gradientNorm2OverEdges(E)
                 - rs_before.gradientNorm2OverEdges(E))
        self.assertLess(abs(d_full - d_loc), _TOL)


class LocalityTest(unittest.TestCase):
    """On a large complex the update is genuinely local — it touches far fewer than
    all edges — while staying exact. This is the whole point of T4 (the #418 cost)."""

    def test_edge_perturbation_is_local_and_exact(self):
        st = _cdt_toroid()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        edges = st.getEdgeList().toVector()
        n_total = len(edges)
        e = edges[n_total // 2]
        ev = {e.getSource().getId(), e.getTarget().getId()}
        aff = [list(c) for c in _tops(st) if ev.issubset(set(c))]
        E = rs.affectedEdgesOfCells(aff)

        # genuinely local: the perturbed edge touches a small neighborhood
        self.assertLess(len(E), n_total // 2, "update is not local on a large mesh")

        before_full, before_loc = _grad_norm2(rs), rs.gradientNorm2OverEdges(E)
        orig = e.getSquaredLength()
        e.setSquaredLength(orig * 1.05)
        after_full, after_loc = _grad_norm2(rs), rs.gradientNorm2OverEdges(E)
        e.setSquaredLength(orig)
        self.assertLess(abs((after_full - before_full) - (after_loc - before_loc)),
                        1e-9)  # larger complex ⇒ looser abs tol, still ~machine


if __name__ == "__main__":
    unittest.main()
