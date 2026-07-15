# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2 fails loudly on off-axis l^2 instead of projecting it (#597).

runStage2's base point used to read getSquaredLength().real() — a silent
projection that would mask any upstream Im producer forever (every trial is
then constructed exactly real, #589, so the final state always reads
Im == 0). With the checked read, a planted Im l^2 reaching the physics loop
raises instead.
"""
import unittest

import tessera

cob = tessera.cobordism


class Stage2OnAxisInvariant(unittest.TestCase):
    def test_planted_im_l2_raises_in_stage2(self):
        node = cob.ProtonIngredients(seed=0).joint_node(0)
        node.st.getEdgeList().toVector()[0].setSquaredLength(complex(1.0, 0.25))
        with self.assertRaisesRegex(RuntimeError, "Im l\\^2"):
            node.run_stage2(beta=1.0, max_iters=1)

    def test_real_geometry_relaxes_normally(self):
        node = cob.ProtonIngredients(seed=0).joint_node(0)
        node.run_stage2(beta=1.0, max_iters=1)   # must not raise
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
