# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Causal-aware degeneracy guard on run_stage2 + signature-change verification (#565).

#541 found the relaxed geometry stays all-spacelike by DYNAMICS (the Euclidean basin
of an all-spacelike seed), not by the stage-2 clamp `Re l^2 in [0.05, 20]` — but that
clamp forbids the entire timelike half-line, and the campaign's machine-precision
descent does bind its floor. `MultiCobordism.set_causal_guard(epsilon)` replaces it,
flag-gated: OFF by default (the clamp, byte-identical — the golden-constant suite in
test_multi_cobordism_python.py is the drift guard); ON forbids only the light-cone
degeneracy band `|Re l^2| < epsilon` (both cone sides admissible, the trial's sign
preserved, symmetric `|Re l^2| <= 20` cap, `Im l^2` handling unchanged).

Epic #559's rule: NO timelike initialization — causal content may only EMERGE. The
timelike edge hand-set below is a verification of the READERS (the complex Sorkin
deficit, the dual Regge action, the exact-gradient objective) across a signature
change — not an initialization policy.
"""
import cmath
import math
import unittest

import tessera as T

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


def _closed_s4(n_refine=12, seed=3):
    """A refined closed S⁴ host (the test_multi_cobordism_python.py fixture): the bare
    ∂Δ⁵ sphere refined by `n_refine` PreGeometric stellar Pachner adds, then a mild
    deterministic non-uniform metric."""
    st = _sphere4(jitter=False)
    applied = 0
    for s in range(seed, seed + n_refine * 4):
        mv = T.AddMove(st, s, False, T.PachnerMode.PreGeometric, False)
        if mv.propose() and mv.apply():
            applied += 1
        if applied >= n_refine:
            break
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.01 * (i % 6))
    return st


def _hinges(st):
    """The (d-2)-simplices, exactly as ``ReggeSolver::collectHinges``: top simplices
    have d+1 vertices, so a hinge has (top_verts - 2) vertices (triangles in 4D)."""
    sims = list(st.getSimplices())
    hinge_nverts = max(len(s.getVertices()) for s in sims) - 2
    return [s for s in sims if len(s.getVertices()) == hinge_nverts]


def _edge_census(st, band_epsilon):
    """(min |Re l^2|, #timelike Re<0, #at the ±20 cap, all-finite?) over the edges."""
    min_abs_re, timelike, at_cap, finite = float("inf"), 0, 0, True
    for e in st.getEdgeList().toVector():
        sq = complex(e.getSquaredLength())
        finite = finite and cmath.isfinite(sq)
        min_abs_re = min(min_abs_re, abs(sq.real))
        timelike += 1 if sq.real < 0.0 else 0
        at_cap += 1 if abs(sq.real) >= 20.0 - 1e-9 else 0
    return min_abs_re, timelike, at_cap, finite


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
    """The guard itself: default OFF (spacelike clamp), ON = only the light-cone band
    forbidden; a guarded step neither NaNs nor collapses a simplex."""

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
        # byte-identical drift guard is the golden-constant suite; this pins the flag
        # default and the OFF-path floor).
        host = _closed_s4(n_refine=12, seed=3)
        opt = self._node(host)
        self.assertEqual(opt.causal_guard_epsilon, 0.0)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05, rel_tol=1e-9)
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
        host = _closed_s4(n_refine=12, seed=3)
        host.getEdgeList().toVector()[5].setSquaredLength(complex(-0.8, 0.0))
        opt = self._node(host)
        opt.set_causal_guard(0.05)
        self.assertEqual(opt.causal_guard_epsilon, 0.05)
        trace = opt.run_stage2(beta=1.0, max_iters=5, alpha0=0.05, rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        min_abs_re, timelike, at_cap, finite = _edge_census(opt.st, 0.05)
        self.assertTrue(finite)
        self.assertGreaterEqual(min_abs_re, 0.05 - 1e-12,
                                "an edge collapsed into the degeneracy band")
        # The hand-set timelike edge may keep its side of the cone or relax back —
        # both are dynamics, not the guard's business. The cap census is the honest
        # conformal-runaway watch: edges racing to ±20 would show up here.
        self.assertLessEqual(at_cap, 1)

    def test_guard_on_all_spacelike_seed_respects_band(self):
        # Guard ON from an ALL-SPACELIKE seed (no timelike initialization — the epic's
        # rule): a short bounded stage-2 run stays finite and outside the degeneracy
        # band on BOTH cone sides. Per #541's basin analysis no edge is expected to
        # cross the cone from an all-spacelike seed (the Euclidean basin is where the
        # descent lives) — that absence is a fine result, so the census asserts the
        # guard's CONTRACT (any crossing lands outside the band), not a crossing count.
        host = _closed_s4(n_refine=12, seed=3)
        opt = self._node(host)
        opt.set_causal_guard(0.05)
        trace = opt.run_stage2(beta=1.0, max_iters=10, alpha0=0.05, rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        self.assertTrue(all(trace[i + 1] <= trace[i] for i in range(len(trace) - 1)))
        min_abs_re, timelike, at_cap, finite = _edge_census(opt.st, 0.05)
        self.assertTrue(finite)
        self.assertGreaterEqual(min_abs_re, 0.05 - 1e-12)
        self.assertEqual(at_cap, 0)   # no conformal runaway to the ±20 cap


if __name__ == "__main__":
    unittest.main()
