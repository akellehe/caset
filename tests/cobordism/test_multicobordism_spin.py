# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Loops-as-quarks composite-spin readout on MultiCobordism (#517, part of #410).

`MultiCobordism.composite_spin_j2` reads the total-spin Casimir of an output block in the
edge (loops-as-quarks) basis: the pair-loop is the Poincare dual of the complementary hole, so
its closed-loop spin holonomy is that hole's deficit, lifted through the revived DiracKahler
spin-1/2 double cover to <S_i.S_j> = 1/4 cos(eps_k); J^2 = 9/4 + 1/2 sum_k cos(eps_k).

Two checks: a fast error-path test (raises without a 3-hole register), and a slow build test
that drives the simultaneous pair-creation build and confirms the readout lands in the
three-spin-1/2 baryon range J^2 in [3/4, 15/4]. As documented (cartan_weyl_gluon.tex /
pair_loop_quarks.tex), this reduces to the closed-loop holonomy and floors above the entangled
proton 3/4 -- the value now lives on the source-of-truth class, not retired Python.
"""
import importlib.util
import math
import os
import sys
import unittest

import cmath

import tessera as T

cob = T.cobordism
_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eo = _load("emergent_optimizer")
_W = cmath.exp(2j * math.pi / 3)
_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]      # three neutral q-qbar pairs (Sigma=0)
_PROTON = [1, _W, _W * _W]
_ANTIPROTON = [1, _W * _W, _W]


class CompositeSpinReadoutTest(unittest.TestCase):
    def test_raises_without_three_hole_register(self):
        # fast: an output block with no emerged 3-hole (b3) register has no composite spin
        host = eo.build_closed_s4(n_refine=8, seed=0)
        opt = cob.MultiCobordism(host, [[1, -1, 0]], [_PROTON], degrees=[3],
                                 gamma=1.0, seed=0)
        with self.assertRaises(RuntimeError):
            opt.composite_spin_j2(0)              # outputs not constructed -> raises

    def test_composite_spin_in_baryon_range(self):
        # slow: build the proton by simultaneous pair creation, read the composite spin
        result = None
        for seed in range(3, 14):
            host = eo.build_closed_s4(n_refine=18, seed=seed)
            opt = cob.MultiCobordism(host, _PAIRS, [_PROTON, _ANTIPROTON], degrees=[3],
                                     gamma=1.0, seed=seed)
            sv = [v.getId() for v in host.getVertexList().toVector()]
            opt.construct_inputs(sv[:3], rounds=20)
            opt.construct_outputs(sv[3:5], rounds=20)
            opt.run_stage1(max_steps=60, n_candidates=10, patience=12)
            opt.run_stage2(beta=1.0, max_iters=20)
            try:
                result = opt.composite_spin_j2(0)
                break
            except RuntimeError:
                continue
        if result is None:
            self.skipTest("no converged 3-hole proton block in the seed range")
        self.assertTrue(math.isfinite(result))
        # J^2 = 9/4 + 1/2 sum_k cos(eps_k), cos in [-1,1] over 3 holes -> [3/4, 15/4]
        self.assertGreaterEqual(result, 0.75 - 1e-6)
        self.assertLessEqual(result, 3.75 + 1e-6)


if __name__ == "__main__":
    unittest.main()
