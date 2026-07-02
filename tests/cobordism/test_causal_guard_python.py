# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Causal-aware degeneracy guard on run_stage2 + signature-change verification (#565).

Guard semantics: `setCausalGuard` in MultiCobordism.h is THE authoritative statement
(OFF by default = the pre-guard spacelike clamp; ON = both cone sides admissible, only
the light-cone degeneracy band forbidden). The projection itself is probed directly
through `bounded_trial_real_part` — the single owner of both trial-bound families —
so the semantics are pinned unit-level, not inferred from stage-2 dynamics.

Epic #559's rule: NO timelike initialization — causal content may only EMERGE. The
timelike edge hand-set below is a verification of the READERS (the complex Sorkin
deficit, the dual Regge action, the exact-gradient objective) across a signature
change — not an initialization policy.
"""
import cmath
import math
import os
import sys
import unittest

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402  (the shared host fixture)

cob = T.cobordism


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


def _hinges(st):
    """The (d-2)-simplices, exactly as ``ReggeSolver::collectHinges``: top simplices
    have d+1 vertices, so a hinge has (top_verts - 2) vertices (triangles in 4D)."""
    sims = list(st.getSimplices())
    hinge_nverts = max(len(s.getVertices()) for s in sims) - 2
    return [s for s in sims if len(s.getVertices()) == hinge_nverts]


def _edge_census(st):
    """(min |Re l^2|, #timelike Re<0, #at the ±20 cap, all-finite?) over the edges."""
    min_abs_re, timelike, at_cap, finite = float("inf"), 0, 0, True
    for e in st.getEdgeList().toVector():
        sq = complex(e.getSquaredLength())
        finite = finite and cmath.isfinite(sq)
        min_abs_re = min(min_abs_re, abs(sq.real))
        timelike += 1 if sq.real < 0.0 else 0
        at_cap += 1 if abs(sq.real) >= 20.0 - 1e-9 else 0
    return min_abs_re, timelike, at_cap, finite


class TrialProjectionTest(unittest.TestCase):
    """Direct unit probe of the projection (`bounded_trial_real_part`), straddling
    0, ±epsilon, and ±20 — so a sign-flipped push-out or a deleted band branch
    fails HERE, without relying on stage-2 dynamics to wander into the band."""

    def test_guard_on_projection_is_exact(self):
        f = cob.MultiCobordism.bounded_trial_real_part
        eps = 0.05
        for trial, expected in [
            (0.0, +eps),         # exactly 0 lands at +epsilon (documented)
            (+0.01, +eps),       # in-band + pushes out to +epsilon
            (-0.01, -eps),       # in-band - pushes out to -epsilon (sign preserved)
            (+eps, +eps),        # the band boundary is admissible
            (-eps, -eps),
            (+0.3, +0.3),        # outside the band keeps its value ...
            (-0.3, -0.3),        # ... on the timelike side too (admissible)
            (+1.0, +1.0),
            (-19.0, -19.0),
            (+20.0, +20.0),      # the cap is admissible
            (-20.0, -20.0),
            (+25.0, +20.0),      # symmetric magnitude cap
            (-25.0, -20.0),
        ]:
            self.assertEqual(f(trial, eps), expected,
                             f"guard-ON projection of {trial} (eps={eps})")

    def test_guard_off_projection_pins_the_clamp_constants(self):
        # The OFF path (epsilon <= 0) is the pre-guard spacelike clamp — an
        # EXECUTABLE pin of the 0.05 floor and the 20 cap, for epsilon 0 and negative.
        f = cob.MultiCobordism.bounded_trial_real_part
        for eps in (0.0, -1.0):
            for trial, expected in [
                (0.01, 0.05),    # the floor
                (0.0, 0.05),
                (-5.0, 0.05),    # OFF forbids the whole timelike half-line
                (-25.0, 0.05),
                (1.0, 1.0),
                (20.0, 20.0),
                (25.0, 20.0),    # the cap
            ]:
                self.assertEqual(f(trial, eps), expected,
                                 f"guard-OFF projection of {trial} (eps={eps})")

    def test_epsilon_validation(self):
        # set_causal_guard / bounded_trial_real_part reject NaN and epsilon > 20: a
        # band wider than the cap would contradict it (push-outs past the cap, caps
        # landing inside the forbidden band) and inf would forbid every trial (see
        # MultiCobordism.h). epsilon <= 0 (OFF) and epsilon == 20 remain valid.
        w = cmath.exp(2j * math.pi / 3)
        opt = cob.MultiCobordism(_sphere4(), [[1, w, w * w]], [], degrees=[3],
                                 gamma=1.0, seed=0)
        for bad in (float("nan"), 25.0, float("inf")):
            with self.assertRaises(ValueError):
                opt.set_causal_guard(bad)
            with self.assertRaises(ValueError):
                cob.MultiCobordism.bounded_trial_real_part(1.0, bad)
        self.assertEqual(opt.causal_guard_epsilon, 0.0)  # rejects left no trace
        opt.set_causal_guard(-1.0)   # <= 0 is valid: OFF
        opt.set_causal_guard(20.0)   # the cap itself is a valid band half-width
        self.assertEqual(opt.causal_guard_epsilon, 20.0)


class SignatureChangeReadersTest(unittest.TestCase):
    """The load-bearing physics check: every reader the stage-2 objective is built
    from stays sane when one edge is hand-set timelike (Re l^2 < 0)."""

    def test_readers_survive_one_timelike_edge(self):
        w = cmath.exp(2j * math.pi / 3)
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        action_before = complex(rs.dualReggeAction())
        self.assertTrue(cmath.isfinite(action_before))

        # Hand-set ONE edge timelike — a reader verification, not initialization.
        st.getEdgeList().toVector()[3].setSquaredLength(complex(-0.8, 0.0))

        # (a) The complex Sorkin/Asante–Dittrich deficit is sane on EVERY hinge.
        for h in _hinges(st):
            eps = complex(h.lorentzianDeficitAngle())
            self.assertTrue(cmath.isfinite(eps),
                            f"non-finite deficit {eps} on hinge "
                            f"{[v.getId() for v in h.getVertices()]}")

        # (b) The dual Regge action is finite and its Im part RESPONDS to the
        # signature change (the boost branch wakes up; Lorentzian action is complex —
        # never Re-only).
        action_after = complex(rs.dualReggeAction())
        self.assertTrue(cmath.isfinite(action_after))
        self.assertGreater(abs(action_after.imag - action_before.imag), 1e-9,
                           "Im(dual Regge action) did not respond to the "
                           f"signature change: {action_before} -> {action_after}")

        # (c) The exact-gradient term and the full objective stay finite.
        grad_sq = cob.MultiCobordism.regge_action_gradient(st)
        self.assertTrue(math.isfinite(grad_sq))
        opt = cob.MultiCobordism(st, [[1, w, w * w], [1, w * w, w]], [[1, w, w * w]],
                                 degrees=[3], gamma=1.0, seed=0)
        opt.seed_inputs([v.getId() for v in st.getVertexList().toVector()][:2])
        self.assertTrue(math.isfinite(opt.objective()))


class CausalGuardStage2Test(unittest.TestCase):
    """The guard inside run_stage2: default OFF, and a guarded step neither NaNs nor
    collapses a simplex (the projection semantics themselves are pinned unit-level
    in TrialProjectionTest)."""

    @classmethod
    def setUpClass(cls):
        cls.w = cmath.exp(2j * math.pi / 3)

    def _node(self, host, seed=3):
        w = self.w
        opt = cob.MultiCobordism(host, [[1, w, w * w], [1, w * w, w]],
                                 [[1, w, w * w]], degrees=[3], gamma=1.0, seed=seed)
        opt.seed_inputs([v.getId() for v in host.getVertexList().toVector()][:2])
        return opt

    def test_guard_default_off_keeps_spacelike_clamp(self):
        # DEFAULT OFF: causal_guard_epsilon == 0 and a stage-2 run keeps every edge on
        # the spacelike clamp Re l^2 in [0.05, 20] — the pre-guard behavior (the
        # byte-identical drift guard is the golden-constant suite; the OFF-path
        # constants are pinned executable in TrialProjectionTest).
        host = _closed_s4(n_refine=8, seed=3)
        opt = self._node(host)
        self.assertEqual(opt.causal_guard_epsilon, 0.0)
        trace = opt.run_stage2(beta=1.0, max_iters=2, alpha0=0.05, rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        for e in opt.st.getEdgeList().toVector():
            sq = complex(e.getSquaredLength())
            self.assertGreaterEqual(sq.real, 0.05 - 1e-12)
            self.assertLessEqual(sq.real, 20.0 + 1e-12)

    def test_guarded_step_with_timelike_edge_no_nan_no_collapse(self):
        # A guarded stage-2 run on a host carrying one hand-set timelike edge (the
        # reader-verification allowance) neither NaNs nor collapses: the F trace stays
        # finite, every edge stays finite, and no edge sits inside the forbidden
        # degeneracy band |Re l^2| < epsilon.
        host = _closed_s4(n_refine=8, seed=3)
        host.getEdgeList().toVector()[5].setSquaredLength(complex(-0.8, 0.0))
        opt = self._node(host)
        opt.set_causal_guard(0.05)
        self.assertEqual(opt.causal_guard_epsilon, 0.05)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05, rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        min_abs_re, timelike, at_cap, finite = _edge_census(opt.st)
        self.assertTrue(finite)
        self.assertGreaterEqual(min_abs_re, 0.05 - 1e-12,
                                "an edge collapsed into the degeneracy band")
        # The hand-set edge may keep its cone side or relax back (dynamics, not the
        # guard's business) — but at this budget any FURTHER crossing would be an
        # anomaly, and any edge at the ±20 cap is the conformal runaway this census
        # exists to catch.
        self.assertLessEqual(timelike, 1)
        self.assertEqual(at_cap, 0)

    def test_guard_on_all_spacelike_seed_respects_band(self):
        # Guard ON from an ALL-SPACELIKE seed (no timelike initialization — the epic's
        # rule): a short bounded stage-2 run stays finite and outside the degeneracy
        # band on BOTH cone sides. Per #541's basin analysis the descent never points
        # across the cone from an all-spacelike seed, so at this budget a crossing
        # would be an anomaly worth failing on (the long-horizon run where a crossing
        # would be a FINDING is the reported bounded experiment on #565).
        host = _closed_s4(n_refine=8, seed=3)
        opt = self._node(host)
        opt.set_causal_guard(0.05)
        trace = opt.run_stage2(beta=1.0, max_iters=6, alpha0=0.05, rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        self.assertTrue(all(trace[i + 1] <= trace[i] for i in range(len(trace) - 1)))
        min_abs_re, timelike, at_cap, finite = _edge_census(opt.st)
        self.assertTrue(finite)
        self.assertGreaterEqual(min_abs_re, 0.05 - 1e-12)
        self.assertEqual(timelike, 0)  # nothing crossed at this budget (see above)
        self.assertEqual(at_cap, 0)    # no conformal runaway to the ±20 cap


if __name__ == "__main__":
    unittest.main()
