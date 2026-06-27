# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The C++ EmergentOptimizer is the source-of-truth port of the Python reference (#491).

Fast: the C++ engine's deterministic objective core (betti, emergent_holes, grad_norm2,
r_state, objective) must equal `examples/cobordism/emergent_optimizer.py` to machine precision
on an identical host. Slow: the two-stage run grows the emergent b₃ register, and a CobordismDAG
chains merges output->input.
"""
import cmath
import importlib.util
import math
import os
import sys
import unittest

import tessera

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _eo():
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(
        "emergent_optimizer", os.path.join(_EX, "emergent_optimizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EmergentOptimizerCxxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _eo()
        cls.CXX = tessera.cobordism.EmergentOptimizer
        cls.w = cmath.exp(2j * math.pi / 3)
        cls.host = cls.eo.build_closed_s4(n_refine=20, seed=3)

    def test_cxx_objective_matches_python_reference(self):
        eo, CXX, w = self.eo, self.CXX, self.w
        tgt = [1, w, w * w]
        self.assertEqual(eo.betti(self.host), list(CXX.betti(self.host)))
        self.assertEqual({tuple(h) for h in eo.emergent_holes(self.host, 3)},
                         {tuple(h) for h in CXX.emergent_holes(self.host, 3)})
        self.assertAlmostEqual(eo._grad_norm2(self.host), CXX.grad_norm2(self.host),
                               places=8)
        self.assertAlmostEqual(eo.r_state(self.host, 3, tgt),
                               CXX.r_state(self.host, 3, tgt), places=10)
        po = eo.EmergentOptimizer(self.host, [[1, w, w * w], [1, w * w, w]], tgt,
                                  degrees=(3,), gamma=1.0, seed=0).objective()
        co = CXX(self.host, [[1, w, w * w], [1, w * w, w]], tgt, degrees=[3],
                 gamma=1.0, seed=0).objective()
        self.assertAlmostEqual(po, co, places=6)

    def test_two_stage_grows_emergent_register(self):
        CXX, w = self.CXX, self.w
        host = self.eo.build_closed_s4(n_refine=20, seed=3)
        opt = CXX(host, [[1, w, w * w], [1, w * w, w]], [1, w, w * w], degrees=[3],
                  gamma=1.0, seed=3)
        self.assertEqual(list(CXX.betti(opt.st))[4], 1)        # bare closed S⁴: b₄=1
        sv = [v.getId() for v in host.getVertexList().toVector()][:2]
        opt.construct_inputs(sv, rounds=12)
        opt.run_stage1(max_steps=20, n_candidates=8, patience=8)
        self.assertGreaterEqual(list(CXX.betti(opt.st))[3], 1)  # a b₃ register emerged

    def test_dag_chains_output_to_input(self):
        cob, eo, w = tessera.cobordism, self.eo, self.w
        dag = cob.CobordismDAG()
        h0 = eo.build_closed_s4(n_refine=14, seed=3)
        h1 = eo.build_closed_s4(n_refine=14, seed=4)
        n0 = dag.add_node(h0, [[1, -1, 0], [1, 0, -1]], [], [1, w, w * w],
                          degrees=[3], seed=3)
        n1 = dag.add_node(h1, [[0, 1, -1]], [n0], [1, w, w * w], degrees=[3], seed=4)
        self.assertEqual(len(dag), 2)
        dag.run(stage1_max_steps=8, stage1_candidates=4, stage1_patience=4,
                stage2_max_iters=10)
        self.assertEqual(len(dag.output(n1)), 3)               # threaded + ran
        self.assertTrue(math.isfinite(dag.residual(n0)))
        self.assertTrue(math.isfinite(dag.residual(n1)))


if __name__ == "__main__":
    unittest.main()
