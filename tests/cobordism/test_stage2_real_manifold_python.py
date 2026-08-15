# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2 is real-manifold dynamics, correct by construction (#589).

The chosen configuration space is real signed l^2 (ordinary Lorentzian Regge;
the complexified theory is unbuilt). ``runStage2`` therefore descends the
exact gradient of F restricted to that manifold — for real F of a complex
variable on the real axis, dF/dx = 2 Re(dF/dz̄), i.e. ``Re(2β·H̄·g)`` — and
constructs every trial exactly real. The #582 interim interventions (the
Gram/CM ``domain_error`` guard, the ``catch(...) → +inf`` line-search backoff)
are deleted: nothing polices the invariant at runtime, because no writer of
``Im l^2`` exists anywhere in the dynamics. These tests are where the
invariant lives now:

* the real-axis direction IS the derivative — it matches finite differences
  of the objective along real perturbations on a host with mixed
  (light-cone-crossing) hinges, where the Wirtinger direction is genuinely
  complex and its real part is load-bearing;
* after any runStage1 + runStage2 sequence on cone-crossing hosts (and on a
  rebuilt causal-specimen fixture of the #562 campaign), max |Im l^2| == 0.0
  EXACTLY — bit-zero, not small — and no exception surfaces from any trial
  the optimizer can construct;
* descent genuinely proceeds through the mixed-hinge regime (the #582
  deferral — every complex trial backed off, no accepted step — is over).
"""

import cmath
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _causal_specimen import load_dump, rebuild_joint_node  # noqa: E402
from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402

cob = T.cobordism


def _sphere4():
    """A minimal triangulated S⁴ (boundary of a 5-simplex), unit spacelike,
    lightly jittered — small enough for per-edge FD of the objective."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setLength(cmath.sqrt(complex(1.0 + 0.013 * (i % 5))))
    return st


def _hinges(st):
    sims = list(st.getSimplices())
    hinge_nverts = max(len(s.getVertices()) for s in sims) - 2
    return [s for s in sims if len(s.getVertices()) == hinge_nverts]


def _assert_mixed_hinge_regime(test, st):
    """The host must genuinely cross the light cone: at least one hinge's
    Sorkin deficit carries a nonzero imaginary part, so the Wirtinger
    direction is complex and the real-axis restriction is load-bearing."""
    T.ReggeSolver(st, T.MatterConfiguration())  # materialize the hinges
    n_complex = sum(1 for h in _hinges(st)
                    if abs(complex(h.lorentzianDeficitAngle()).imag) > 1e-9)
    test.assertGreater(n_complex, 0, "fixture lost its mixed-hinge regime")


def _max_abs_im(st):
    return max(abs(complex(e.getLength()**2).imag)
               for e in st.getEdgeList().toVector())


class RealAxisDirectionTest(unittest.TestCase):
    """The stage-2 descent direction equals the FD gradient of the objective
    along real perturbations — the item-1 physics, tested directly."""

    def test_direction_matches_fd_on_mixed_hinge_host(self):
        beta = 1.0
        st = _sphere4()
        # Hand-set one edge timelike: every base triangle wedge against it has
        # a cofactor pair straddling zero — the m=1 crossing branch (#582) —
        # so the action gradient/Hessian are genuinely complex here.
        st.getEdgeList().toVector()[3].setLength(cmath.sqrt(complex(complex(-0.8, 0.0))))
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        _assert_mixed_hinge_regime(self, st)

        g = np.asarray(rs.actionGradientExact(), dtype=complex)
        H = np.asarray(rs.actionHessianExact(), dtype=complex)
        self.assertGreater(np.max(np.abs(g.imag)), 1e-9,
                           "gradient must be genuinely complex here")
        # The engine's direction (runStage2): Re(2*beta*conj(H)@g) — the exact
        # restriction of the Wirtinger direction to the real axis.
        direction = (2.0 * beta * (np.conj(H) @ g)).real

        edges = st.getEdgeList().toVector()

        def objective():
            solver = T.ReggeSolver(st, T.MatterConfiguration())
            return beta * sum(abs(c) ** 2
                              for c in solver.actionGradientExact())

        h = 1e-6
        fd = np.zeros(len(edges))
        for i, e in enumerate(edges):
            re0 = complex(e.getLength()**2).real
            e.setLength(cmath.sqrt(complex(complex(re0 + h, 0.0))))
            fp = objective()
            e.setLength(cmath.sqrt(complex(complex(re0 - h, 0.0))))
            fm = objective()
            e.setLength(cmath.sqrt(complex(complex(re0, 0.0))))
            fd[i] = (fp - fm) / (2 * h)

        scale = np.max(np.abs(fd))
        self.assertGreater(scale, 0.0)
        self.assertLess(np.max(np.abs(direction - fd)) / scale, 1e-5,
                        f"direction != dF/dx:\n{direction}\nvs FD\n{fd}")


class ImExactlyZeroInvariantTest(unittest.TestCase):
    """max |Im l^2| == 0.0 EXACTLY after any stage-1 + stage-2 sequence: the
    by-construction invariant the deleted #582 guard used to police."""

    def _node(self, host, seed=3):
        w = cmath.exp(2j * math.pi / 3)
        opt = cob.MultiCobordism(host, [[1, w, w * w], [1, w * w, w]],
                                 [[1, w, w * w]], degrees=[3], gamma=1.0,
                                 seed=seed)
        opt.seed_inputs([v.getId()
                         for v in host.getVertexList().toVector()][:2])
        return opt

    def test_stage1_stage2_on_cone_crossing_host(self):
        host = _closed_s4(n_refine=8, seed=3)
        edges = host.getEdgeList().toVector()
        edges[5].setLength(cmath.sqrt(complex(complex(-0.8, 0.0))))   # timelike
        edges[9].setLength(cmath.sqrt(complex(complex(0.01, 0.0))))   # inside the null band
        opt = self._node(host)
        _assert_mixed_hinge_regime(self, host)
        # No exception may surface from any trial either stage constructs.
        opt.run_stage1(max_steps=4, n_candidate_moves=4,
                       grow_boundaries=True)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05,
                               rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        if len(trace) == 1:
            # No accepted step is legitimate ONLY as the variational verdict
            # (real-manifold stationarity), never as a backed-off error path.
            self.assertTrue(opt.last_stage2_stationary)
        self.assertEqual(_max_abs_im(opt.st), 0.0)

    def test_descent_descends_through_the_crossing_regime(self):
        # The objective genuinely decreases while hinges cross the cone — the
        # regime the interim guard could only veto.
        host = _closed_s4(n_refine=8, seed=3)
        host.getEdgeList().toVector()[5].setLength(cmath.sqrt(complex(complex(-0.8, 0.0))))
        opt = self._node(host)
        trace = opt.run_stage2(beta=1.0, max_iters=6, alpha0=0.05,
                               rel_tol=1e-9)
        self.assertGreaterEqual(len(trace), 2)
        self.assertLess(trace[-1], trace[0])
        self.assertEqual(_max_abs_im(opt.st), 0.0)


class CausalSpecimenContinuationTest(unittest.TestCase):
    """A rebuilt #562 causal specimen continues under the real-axis dynamics:
    finite F, exactly-real geometry, no exception. (The full re-verification
    of all three causal specimens is reported on PR #590 — this pins the
    rebuild path and the invariant on the smallest one.)"""

    def test_specimen_11001000_stage2_continuation(self):
        dump = load_dump(11001000)
        node = rebuild_joint_node(dump, seed=11001000)
        st = node.st
        # The specimen is causal: its recorded re_min < 0 must survive rebuild.
        re_min = min(complex(e.getLength()**2).real
                     for e in st.getEdgeList().toVector())
        self.assertLess(re_min, 0.0)
        self.assertEqual(_max_abs_im(st), 0.0)  # dumps carry Im == 0
        trace = node.run_stage2(beta=1.0, max_iters=3, alpha0=0.05,
                                rel_tol=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        self.assertEqual(_max_abs_im(node.st), 0.0)
        # Timelike content is dynamics, not policy — but the reader must agree
        # the geometry stayed finite and classifiable.
        for e in node.st.getEdgeList().toVector():
            self.assertTrue(cmath.isfinite(complex(e.getLength()**2)))


if __name__ == "__main__":
    unittest.main()
