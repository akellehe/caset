# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#353 color confinement + emergent result through the C++ MergeCobordism (#379).

The reworked `RegisterTopology` is the #353 color register built as a
**b₁=2 holed-icosahedron staircase** (`(S²−3 color holes) × [0,2]`): one shared
color register across the three blocks. That b₁=2 is the confinement — a colored
(Σ≠0) configuration cannot be carried, so its realizability residual floors,
while a color-neutral (Σ=0) configuration realizes. (The earlier tube-merge was
b₁=8 and carried *any* config — no confinement.) These tests pin, through the
canonical C++ `MergeCobordism`:

  * the topology is a valid manifold with **b₁=2** (not 8);
  * **confinement**: neutral/singlet states score orders below colored states;
  * the color singlet `[1,ω,ω²]` realizes (the induced-sign `kColorSign` fix —
    raw components would mis-floor it);
  * **S₃ gauge invariance** of the score;
  * the **emergent result block** is read after relaxation (never inserted); the
    two-pair bipartite merge is *not* a color singlet (|σ_R| ≫ 0) — #353's
    documented result that the proton needs three pairs (the #382 tripartite/
    sequence merge), not a single bipartite merge.

Through the merge the register is scored over the EXACT `residualForPeriods` (the
#353 period path; the merge no longer uses the soft `residualForLoops` edge-loop
encoding for the register, and pins INPUTS only — the result block EMERGES). The
~1e-2 neutral floor here is intrinsic to the *shared-register* staircase: a single
block's three holes span the b₁=2 register only approximately, so neutral floors at
~1e-2 — both encodings agree on this, so it was never the encoding (the machine-zero
~1e-29 realizability map lives on the single-register `cob.Register`). Colored
configs floor at ~10, so the confinement *split* (ratio ~10³) is intact.
"""

import cmath
import itertools
import unittest

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)

_NEUTRAL = {"[1,-1,0]": [1, -1, 0], "[1,0,-1]": [1, 0, -1],
            "singlet [1,w,w^2]": [1, _W, _W * _W]}
_COLORED = {"[1,0,0]": [1, 0, 0], "[1,1,0]": [1, 1, 0], "[1,1,1]": [1, 1, 1]}


def _merge(states_in, max_iters=0):
    # The #353 register pins INPUTS only and reads the EMERGENT result block, so
    # no output states are supplied (emergesResult()).
    return cob.TransportCobordism(states_in, max_iters=max_iters, seed=0,
                              topology=cob.RegisterTopology())


def _score(cfg):
    # Pin the config on one input register block; the post-build r_U
    # (state_residual, the EXACT residualForPeriods) is the realizability score.
    # max_iters=0 => bare metric (the confinement is a topology property,
    # max_iters-independent — the staircase floor does not relax away).
    return _merge([cfg]).stats.state_residual


# Built once for the structural assertions (topology is max_iters-independent).
_M = _merge([[1, -1, 0]])


class TopologyIsB1EqualsTwoTest(unittest.TestCase):
    def test_name_is_the_b1_2_staircase(self):
        self.assertIn("register", _M.stats.topology)
        self.assertIn("b1=2", _M.stats.topology)

    def test_betti_is_b1_two(self):
        # b₁(W)=2: one shared color register (the confinement), not the b₁=8 tube.
        betti = list(_M.stats.betti_cobordism)
        self.assertEqual(betti[0], 1)
        self.assertEqual(betti[1], 2)


class ConfinementThroughTheMergeTest(unittest.TestCase):
    """Neutral/singlet realize (low r_U); colored floor — the confinement split."""

    def test_neutral_below_colored(self):
        worst_neutral = max(_score(c) for c in _NEUTRAL.values())
        best_colored = min(_score(c) for c in _COLORED.values())
        self.assertLess(worst_neutral, 1.0, "neutral should be small")
        self.assertGreater(best_colored, 5.0, "colored should floor")
        self.assertGreater(best_colored / worst_neutral, 50.0)

    def test_singlet_realizes_not_floored(self):
        # The sign fix: [1,ω,ω²] scores like a neutral pair, NOT like a colored
        # state. Without kColorSign it would floor (raw n·a ≠ 0).
        self.assertLess(_score([1, _W, _W * _W]), 1.0)
        self.assertGreater(min(_score(c) for c in _COLORED.values()), 5.0)


class S3GaugeInvarianceThroughTheMergeTest(unittest.TestCase):
    def test_singlet_score_is_s3_invariant(self):
        base = [1, _W, _W * _W]
        scores = [_score([base[i] for i in p])
                  for p in itertools.permutations(range(3))]
        self.assertLess(max(scores) - min(scores), 1e-6)


class EmergentResultBlockTest(unittest.TestCase):
    """The result block emerges after relaxation (never inserted); the two-pair
    merge is not a singlet (the proton needs three pairs — #382)."""

    @classmethod
    def setUpClass(cls):
        # Pin two neutral q-qbar pairs as INPUTS (blocks A, B); the result block R
        # emerges, read EXACTLY over its color holes (cyclePeriods), never pinned.
        cls.m = _merge([[1, -1, 0], [1, 0, -1]], max_iters=40)

    def test_result_is_a_color_triple(self):
        # The transport reads a color rep (a 3-vector), carried to the result block.
        self.assertEqual(len(self.m.result), 3)

    def test_two_pair_merge_is_not_a_singlet(self):
        # |σ_R| is O(1), not < 1e-3: the bipartite (2-pair) merge does NOT yield a
        # color singlet — #353's finding that the proton needs the 3-pair merge.
        sigma_r = abs(sum(self.m.result))
        self.assertGreater(sigma_r, 0.1)


if __name__ == "__main__":
    unittest.main()
