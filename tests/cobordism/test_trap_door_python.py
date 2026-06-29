# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The trap door + in-run stall recovery (#503).

When greedy stalls (no move lowers F = ‖∇S‖² + Γ·r_U), runStage1 takes a gated move
from the FULL range (Pachner add/remove/flip/iflip + surgical cone_out/cone_in) so it
escapes a too-small complex instead of halting. That is what lets a register grow out
of the minimal ∂Δ⁵ seed, and — with the revert/reseed/retry on both stall paths — what
lets a single long run_stage1 self-recover (no caller-side chunking).
"""
import cmath
import math
import unittest

import tessera as T

cob = T.cobordism
W = cmath.exp(2j * math.pi / 3)


def bare_s4():
    """The minimal closed seed: bare ∂Δ⁵ (S⁴, Betti [1,0,0,0,1]) = six pentatopes."""
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT, 1.0, 1.0,
                     T.PREFERRED, T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.01 * (i % 6))
    return st


def _opt(st, seed):
    opt = cob.MultiCobordism(st, [[1, W]], [[1, W, W * W]], degrees=[3], gamma=50.0,
                             seed=seed)
    vb = [v.getId() for v in st.getVertexList().toVector()]
    opt.construct_inputs(vb[:1], rounds=8)
    return opt


class TrapDoorTest(unittest.TestCase):
    def test_minimal_seed_is_six_pentatopes(self):
        st = bare_s4()
        self.assertEqual(len(st.getTopSimplices()), 6)                       # ∂Δ⁵ facets
        self.assertEqual(list(cob.MultiCobordism.betti(st)), [1, 0, 0, 0, 1])  # closed S⁴

    def test_trap_door_grows_out_of_the_minimal_seed(self):
        # Greedy alone makes no move from the bare seed (nothing lowers F yet — the
        # chicken-and-egg). The trap door takes a gated full-range move regardless, so
        # the complex grows past its six starting cells instead of halting at step 0.
        st = bare_s4()
        n0 = len(st.getTopSimplices())
        opt = _opt(st, seed=1)
        opt.run_stage1(25, 8, 15, grow_boundaries=True)
        self.assertGreater(len(opt.st.getTopSimplices()), n0,
                           "trap door failed to grow the minimal seed")

    def test_input_weight_scales_an_uncarried_input_residual(self):
        # Before an input carries, its residual contributes to r_U; weighting it up
        # raises r_U (the lever that keeps inputs from dissolving). On the bare seed the
        # input cannot carry yet, so a heavier weight strictly increases r_U.
        st = bare_s4()
        opt = _opt(st, seed=1)
        opt.set_input_residual_weight(1.0)
        low = opt.r_u(opt.st)
        opt.set_input_residual_weight(20.0)
        high = opt.r_u(opt.st)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
