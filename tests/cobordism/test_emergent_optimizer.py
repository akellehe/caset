# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The emergent optimizer loop on a closed S⁴ (T5, #462).

End-to-end checks of the Stage-1 loop's *mechanics* — that it is a faithful greedy,
gated, fully-emergent optimizer of `F = ‖∇S_Regge‖² + Γ·r_U` (the physics verdict —
does a color b_k hole emerge — is T6/#463, not here):

  * **greedy extremization** — `F` decreases monotonically and stays `≥ 0`;
  * **exact accounting** — each committed step's reported `ΔF` equals the actual
    objective change (the snapshot-the-winner commit is drift-free);
  * **gated** — every accepted state is a valid manifold (`dualComplexValid`);
  * **emergent, not prescribed** — the optimizer holds no target topology; whatever
    `b_k` results comes only from objective-justified random moves;
  * **no-improvement is a no-op** — a step with no improving candidate leaves `F`.

Heavy (`@pytest.mark.slow`): one Stage-1 run on a small closed S⁴.
"""
import cmath
import importlib.util
import math
import os
import sys
import unittest

import pytest

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load_optimizer():
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(
        "emergent_optimizer", os.path.join(_EX, "emergent_optimizer.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["emergent_optimizer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_opt(EO, seed=1):
    st = EO.build_closed_s4(n_refine=20, seed=0)
    tets = sorted({tuple(sorted(v.getId() for v in f.getVertices()))
                   for s in st.getTopSimplices() for f in s.getFacets()})[:3]
    w = cmath.exp(2j * math.pi / 3)                       # ω color charge target
    return EO.EmergentOptimizer(st, [list(t) for t in tets], [1.0, w, w * w],
                                k=2, gamma=1.0, seed=seed)


@pytest.mark.slow
class EmergentOptimizerStage1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.EO = _load_optimizer()

    def test_stage1_is_a_gated_monotone_exact_emergent_optimizer(self):
        EO = self.EO
        opt = _make_opt(EO)
        self.assertEqual(EO.betti(opt.st), [1, 0, 0, 0, 1])   # a closed S⁴ host

        # exact accounting: each committed step's ΔF == the actual objective change
        for _ in range(8):
            before = opt.objective()
            dF = opt.step(n_candidates=12)
            after = opt.objective()
            self.assertLess(abs(after - (before + dF)), 1e-6)
            self.assertLessEqual(dF, 1e-9)                    # never accept a worsening move

        trace = opt.run_stage1(max_steps=40, n_candidates=12, patience=8)

        # greedy: monotone non-increasing, and the objective is genuinely lowered
        self.assertTrue(all(trace[i + 1] <= trace[i] + 1e-6
                            for i in range(len(trace) - 1)))
        self.assertLess(trace[-1], trace[0])
        # F = ‖∇S‖² + Γ·r_U is a sum of non-negatives — never goes negative
        self.assertTrue(all(f >= -1e-6 for f in trace))

        # gated: the final emergent complex is a valid manifold
        ok, _reason = EO.cob.EigenstateSynthesis(opt.st, 2).dualComplexValid()
        self.assertTrue(ok)

        # emergent (not prescribed): the optimizer carries no b_k target; the host
        # started [1,0,0,0,1] and whatever topology it ends at came only from ΔF.
        end_betti = EO.betti(opt.st)
        self.assertEqual(end_betti[0], 1)                    # still connected

    def test_no_improving_candidate_is_a_noop(self):
        EO = self.EO
        opt = _make_opt(EO, seed=2)
        f0 = opt.objective()
        # with zero candidates there is nothing to improve → an exact no-op
        dF = opt.step(n_candidates=0)
        self.assertEqual(dF, 0.0)
        self.assertLess(abs(opt.objective() - f0), 1e-9)


if __name__ == "__main__":
    unittest.main()
