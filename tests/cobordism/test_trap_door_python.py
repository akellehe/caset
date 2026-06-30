# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The trap door + in-run stall recovery (#503).

When greedy stalls (no move lowers F = ‖∇S‖² + Γ·r_U), runStage1 takes a gated move
from the FULL range (Pachner add/remove/flip/iflip + surgical cone_out/cone_in) so it
escapes a too-small complex instead of halting. That is what lets a register grow out
of a single Δ⁴ simplex (the proton's actual seed), and — with the revert/reseed/retry
on both stall paths — what lets a single long run_stage1 self-recover (no caller-side
chunking).
"""
import cmath
import math
import unittest

import tessera as T

cob = T.cobordism
W = cmath.exp(2j * math.pi / 3)


def single_simplex():
    """The minimal seed: a single Δ⁴ simplex (one pentatope — 5 verts, 1 cell, Betti
    [1,0,0,0,0], a contractible 4-ball) with a uniform metric — the proton's seed."""
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT, 1.0, 1.0,
                     T.PREFERRED, T.SolidSimplex(4))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
    return st


def _opt(st, seed):
    opt = cob.MultiCobordism(st, [[1, W]], [[1, W, W * W]], degrees=[3], gamma=50.0,
                             seed=seed)
    vb = [v.getId() for v in st.getVertexList().toVector()]
    opt.seed_inputs(vb[:1])
    return opt


class TrapDoorTest(unittest.TestCase):
    def test_minimal_seed_is_one_simplex(self):
        st = single_simplex()
        self.assertEqual(len(st.getTopSimplices()), 1)                        # one pentatope
        self.assertEqual(len(st.getVertexList().toVector()), 5)               # 5 vertices
        self.assertEqual(list(cob.MultiCobordism.betti(st)), [1, 0, 0, 0, 0])  # contractible 4-ball

    def test_trap_door_grows_out_of_the_single_simplex(self):
        # Greedy alone makes no move from one simplex (nothing lowers F yet — the
        # chicken-and-egg). The trap door takes a gated full-range move regardless, so
        # the complex grows past its single starting cell instead of halting at step 0.
        #
        # The greedy ΔF tie-breaks read FP values from the OpenMP-reduced Regge gradient,
        # whose summation order varies run-to-run, so a single (seed, budget) can — even
        # at a fixed RNG seed — occasionally net no growth within the budget. The PROPERTY
        # under test is that the trap door grows out of the seed, not that one fixed seed
        # does so every run; so try a few seeds with a healthy budget and require growth.
        # Short-circuits on the first seed that grows (the usual case is the first).
        grew = False
        for seed in range(1, 9):
            st = single_simplex()
            n0 = len(st.getTopSimplices())
            opt = _opt(st, seed=seed)
            opt.run_stage1(40, 8, 15, grow_boundaries=True)
            if len(opt.st.getTopSimplices()) > n0:
                grew = True
                break
        self.assertTrue(grew,
                        "trap door failed to grow the single-simplex seed for any of "
                        "seeds 1..8")

    def test_input_weight_scales_an_uncarried_input_residual(self):
        # Before an input carries, its residual contributes to r_U; weighting it up
        # raises r_U (the lever that keeps inputs from dissolving). On the single-simplex
        # seed the input cannot carry yet, so a heavier weight strictly increases r_U.
        st = single_simplex()
        opt = _opt(st, seed=1)
        opt.set_input_residual_weight(1.0)
        low = opt.r_u(opt.st)
        opt.set_input_residual_weight(20.0)
        high = opt.r_u(opt.st)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
