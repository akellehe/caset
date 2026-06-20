# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Compose bipartite TransportCobordism transports into a sequence (#382).

The compose mechanism: the EMERGENT result of one merge is fed as a boundary
INPUT of the next (result-state = next-merge boundary), never a hand-welded
interior. Each merge pins its inputs over the EXACT period path
(`residualForPeriods`, inputs only) and reads the emergent result block over
`cyclePeriods`; each step is a genuine valid manifold (b1 = 2). These tests pin:

  * the compose is real -- step 2's input IS step 1's emergent output_state, and
    the chained merge produces a result (the result emerges, never inserted);
  * each step is the b1 = 2 shared-color-register manifold;
  * the structural obstruction (#353's documented finding): a bipartite sequence
    does NOT reach the color singlet -- |sigma| stays O(1) at every step, while
    the singlet target has |sigma| = 0. The proton needs the tripartite W_ABC
    junction (three neutral-pair inputs into one bulk), not a bipartite sequence.
"""

import cmath
import unittest

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)

# Three color-neutral q-qbar pairs (Sigma = 0): the carriable inputs.
_A = [1, -1, 0]
_B = [1, 0, -1]
_C = [0, 1, -1]
_SINGLET = [1, _W, _W * _W]


def _merge(inputs, max_iters=60):
    return cob.TransportCobordism(inputs, max_iters=max_iters, seed=0,
                              topology=cob.RegisterTopology())


class ComposeSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m1 = _merge([_A, _B])
        cls.ab = list(cls.m1.result)        # emergent result of step 1
        cls.m2 = _merge([cls.ab, _C])             # ...fed as a boundary input
        cls.abc = list(cls.m2.result)

    def test_each_step_reads_an_emergent_color_triple(self):
        # The result is READ out of the relaxed geometry (a 3-vector): the transport
        # carries the inputs to the result block.
        self.assertEqual(len(self.ab), 3)
        self.assertEqual(len(self.abc), 3)

    def test_compose_is_real_not_welded(self):
        # Step 2's input boundary IS step 1's emergent result -- the transports are
        # composed (result-state = next boundary), and the chained transport
        # produces a result. Composition, never a hand-welded interior.
        self.assertEqual(list(self.m2.input_states[0]), self.ab)
        self.assertTrue(self.abc)  # the second merge produced an emergent result

    def test_each_step_is_the_b1_two_manifold(self):
        # b1(W) = 2: the shared color register (the build() manifold gate already
        # asserts dualComplexValid; b1 = 2 confirms the register topology).
        self.assertEqual(list(self.m1.stats.betti_cobordism)[1], 2)
        self.assertEqual(list(self.m2.stats.betti_cobordism)[1], 2)

    def test_bipartite_sequence_is_not_a_singlet(self):
        # The structural obstruction (#353): |sigma| stays O(1) -- the bipartite
        # sequence does NOT reach the singlet (|sigma| = 0). Proton needs W_ABC.
        self.assertLess(abs(sum(_SINGLET)), 1e-9)          # the target is neutral
        self.assertGreater(abs(sum(self.ab)), 0.1)         # AB is color-charged
        self.assertGreater(abs(sum(self.abc)), 0.1)        # ABC is color-charged


if __name__ == "__main__":
    unittest.main()
