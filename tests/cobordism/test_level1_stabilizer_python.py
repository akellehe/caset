# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The level-1 stabilizer law
(``examples/cobordism/level1_stabilizer_law.py``).

The law: the level-1 realizable transports are the END's mapping classes --
the image of its hole-triple stabilizer in Sym(holes), read as actions on
the carried plane V. Icosahedron ends give C_3 (#279); hexagon-join ends,
whose stabilizer is the full S_3, transport all six hole permutations --
including the transpositions the icosahedron provably cannot. These tests
pin the 4-dimensional instance:

  1. Every twist used is verified to be an automorphism of the holed join
     preserving the hole set.
  2. The straight 4d fill carries a 2-dim register (b_2 by homotopy) and
     transports exactly the identity's V-class.
  3. A transposition twist (rho) transports exactly CNOT -- impossible on
     icosahedron ends.
  4. The V-class structure: gates equal on the charge-zero plane
     co-transport, and the only nontrivial class among the 13 candidates is
     {H(x)H, SWAP} (H(x)H = SWAP - J/2 with J the all-ones matrix, and J
     annihilates V).
  5. The factor swap -- a nontrivial twist inducing the identity on holes --
     transports the identity: mapping classes, not vertex maps, transport.
"""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "level1_stabilizer_law.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("level1_stabilizer_law",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["level1_stabilizer_law"] = module
    spec.loader.exec_module(module)
    return module


LAW = _load_example()


class TwistsAreEndAutomorphismsTest(unittest.TestCase):
    """1: the twist family is the hole-triple stabilizer, verifiably."""

    def test_every_twist_preserves_cells_and_holes(self):
        for name, perm in LAW.TWISTS:
            self.assertTrue(LAW._is_end_automorphism(perm), msg=name)


class VClassStructureTest(unittest.TestCase):
    """4: transport sees only the action on V."""

    def test_the_only_nontrivial_class_is_hxh_swap(self):
        classes = LAW._v_classes()
        nontrivial = {m for m in classes.values() if len(m) > 1}
        self.assertEqual(nontrivial, {("H(x)H", "SWAP")})


class StraightFillAnchorTest(unittest.TestCase):
    """2: the 4d level-1 anchor."""

    @classmethod
    def setUpClass(cls):
        cls.fill = LAW.Level1FillS3()
        cls.rows = LAW.battery(cls.fill)

    def test_register_is_two_dimensional(self):
        self.assertEqual(self.fill.dim, 2)
        self.assertEqual(self.fill.rank, 2)

    def test_ends_conserve_signed_charge(self):
        self.assertLess(self.fill.end_charge_leak(), 1e-9)

    def test_transports_exactly_the_identity(self):
        realized = sorted(r["gate"] for r in self.rows if r["realizable"])
        self.assertEqual(realized, ["Identity"])
        self.assertEqual(LAW.L1.match_gate(self.fill.emergent_gate()),
                         "Identity")

    def test_floored_transports_are_leak_certified(self):
        for row in self.rows:
            if not row["realizable"]:
                self.assertGreater(row["leak"], 1e-6, msg=row["gate"])


class TranspositionTransportTest(unittest.TestCase):
    """3 + 5: the transposition twist and the factor-swap control."""

    def test_rho_transports_exactly_cnot(self):
        fill = LAW.Level1FillS3(twist=LAW._RHO)
        self.assertEqual(fill.dim, 2)
        self.assertEqual(LAW.L1.match_gate(fill.emergent_gate()), "CNOT")
        realized = sorted(r["gate"] for r in LAW.battery(fill)
                          if r["realizable"])
        self.assertEqual(realized, ["CNOT"])

    def test_sigma_rho_transports_the_swap_v_class(self):
        fill = LAW.Level1FillS3(twist=LAW.L1._compose(LAW._SIGMA, LAW._RHO))
        self.assertEqual(LAW.L1.match_gate(fill.emergent_gate()), "SWAP")
        realized = sorted(r["gate"] for r in LAW.battery(fill)
                          if r["realizable"])
        self.assertEqual(realized, ["H(x)H", "SWAP"])

    def test_factor_swap_transports_the_identity(self):
        fill = LAW.Level1FillS3(twist=LAW._TAU)
        realized = sorted(r["gate"] for r in LAW.battery(fill)
                          if r["realizable"])
        self.assertEqual(realized, ["Identity"])


if __name__ == "__main__":
    unittest.main()
