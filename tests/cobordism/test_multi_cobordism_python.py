# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The C++ MultiCobordism is the source-of-truth port of the Python reference (#491).

Fast: the C++ engine's deterministic objective core (betti, emergent_holes, regge_action_gradient,
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


class MultiCobordismCxxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _eo()
        cls.CXX = tessera.cobordism.MultiCobordism
        cls.w = cmath.exp(2j * math.pi / 3)
        cls.host = cls.eo.build_closed_s4(n_refine=20, seed=3)

    def test_cxx_objective_matches_python_reference(self):
        eo, CXX, w = self.eo, self.CXX, self.w
        tgt = [1, w, w * w]
        self.assertEqual(eo.betti(self.host), list(CXX.betti(self.host)))
        self.assertEqual({tuple(h) for h in eo.emergent_holes(self.host, 3)},
                         {tuple(h) for h in CXX.emergent_holes(self.host, 3)})
        self.assertAlmostEqual(eo._grad_norm2(self.host), CXX.regge_action_gradient(self.host),
                               places=8)
        self.assertAlmostEqual(eo.r_state(self.host, 3, tgt),
                               CXX.r_state(self.host, 3, tgt), places=10)
        po = eo.EmergentOptimizer(self.host, [[1, w, w * w], [1, w * w, w]], tgt,
                                  degrees=(3,), gamma=1.0, seed=0).objective()
        # output_targets is a list (the full cobordism); pre-construction the
        # single-output fallback reproduces the reference's r_state(output).
        co = CXX(self.host, [[1, w, w * w], [1, w * w, w]], [tgt], degrees=[3],
                 gamma=1.0, seed=0).objective()
        self.assertAlmostEqual(po, co, places=6)

    def test_two_stage_grows_emergent_register(self):
        CXX, w = self.CXX, self.w
        host = self.eo.build_closed_s4(n_refine=20, seed=3)
        opt = CXX(host, [[1, w, w * w], [1, w * w, w]], [[1, w, w * w]], degrees=[3],
                  gamma=1.0, seed=3)
        self.assertEqual(list(CXX.betti(opt.st))[4], 1)        # bare closed S⁴: b₄=1
        sv = [v.getId() for v in host.getVertexList().toVector()][:2]
        opt.construct_inputs(sv, rounds=12)
        opt.run_stage1(max_steps=20, n_candidate_moves=8, patience=8)
        self.assertGreaterEqual(list(CXX.betti(opt.st))[3], 1)  # a b₃ register emerged

    def test_two_step_proton_via_canonical_class(self):
        # Retrofit of the old hand-rolled proton-shaped DAG smoke (#503): the
        # canonical two-step proton build now goes through tessera.cobordism.Proton
        # (Step A recombination -> a *colored* diquark, Step B formation -> the
        # color singlet) instead of a hand-wired CobordismDAG with the physically
        # wrong all-singlet recipe. A fast smoke that both steps run end-to-end and
        # expose the 3-vector proton singlet; the thorough convergence test lives in
        # tests/cobordism/test_proton_cpp_python.py. (CobordismDAG's output->input
        # threading and output() stay covered by test_dag_recombination_routes_two_outputs.)
        Proton = tessera.cobordism.Proton
        self.assertEqual(len(Proton.singlet()), 3)            # the proton is a 3-vector
        p = Proton(seed=3)
        p.build(max_restarts=1, construct_rounds=8, init_steps=8, evolve_steps=4,
                stage1_candidate_moves=4, stage1_patience=4, stage2_max_iters=6,
                min_quark_holes=1)
        # both steps ran: Step A (diquark recombination) and Step B (proton formation)
        self.assertTrue(math.isfinite(p.diquark_residual()))
        self.assertTrue(math.isfinite(p.color_residual()))
        # block() is the carved formation sub-complex (None only if nothing emerged)
        block = p.block()
        if block is not None:
            self.assertGreater(len(block.getEdgeList().toVector()), 0)

    def test_recombination_two_in_two_out(self):
        # 2->2 recombination in ONE co-optimized node: 2 input pairs, 2 outputs.
        CXX, w = self.CXX, self.w
        host = self.eo.build_closed_s4(n_refine=14, seed=3)
        opt = CXX(host, [[1, -1, 0], [1, 0, -1]], [[1, w, w * w], [1, w * w, w]],
                  degrees=[3], gamma=1.0, seed=3)
        sv = [v.getId() for v in host.getVertexList().toVector()]
        opt.construct_inputs(sv[:2], rounds=8)
        opt.construct_outputs(sv[2:4], rounds=8)   # two output blocks, co-optimized
        opt.run_stage1(max_steps=6, n_candidate_moves=4, patience=4)
        self.assertTrue(math.isfinite(opt.r_u(opt.st)))   # all 4 blocks scored, no crash

    def test_dag_recombination_routes_two_outputs(self):
        # recombination node (2 outputs) -> two independent legs via output index.
        cob, eo, w = tessera.cobordism, self.eo, self.w
        dag = cob.CobordismDAG()
        hr = eo.build_closed_s4(n_refine=12, seed=3)
        hp = eo.build_closed_s4(n_refine=12, seed=5)
        ha = eo.build_closed_s4(n_refine=12, seed=6)
        rec = dag.add_node(hr, [[1, -1, 0], [1, 0, -1]],
                           [], [[1, w, w * w], [1, w * w, w]], degrees=[3], seed=3)
        pro = dag.add_node(hp, [[0, 1, -1]], [(rec, 0)], [[1, w, w * w]],
                           degrees=[3], seed=5)
        apr = dag.add_node(ha, [[0, -1, 1]], [(rec, 1)], [[1, w, w * w]],
                           degrees=[3], seed=6)
        dag.run(stage1_max_steps=6, stage1_candidate_moves=3, stage1_patience=3,
                stage2_max_iters=6)
        self.assertEqual(dag.num_outputs(rec), 2)
        self.assertEqual(len(dag.output(rec, 0)), 3)   # CobordismDAG.output() threading
        for nd in (rec, pro, apr):
            self.assertTrue(math.isfinite(dag.residual(nd)))


if __name__ == "__main__":
    unittest.main()
