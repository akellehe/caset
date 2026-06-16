# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Emergent-bulk realizability: surgery makes b_1 a pure output (the #196 capability).

Independent re-derivation of the claims the example
(``examples/cobordism/emergent_bulk_realizability.py``) makes, plus a self-verify
that the committed example exits 0:

  1. **The surgery remove primitive moves b_1, holding the boundary bit-exact.**
     ``EigenstateSynthesis.removeInteriorCell`` on a disk filling opens the handle
     (b_1: 0 -> 1) with the pinned boundary untouched; ``restoreLastRemoval`` puts it
     back bit-exactly; a boundary-touching cell is refused.
  2. **The two-boundary meridian 2x2 (the heart of the experiment).** The meridian
     is carried on BOTH boundary circles. With MATCHED periods (p_A = p_B, the
     annulus's own H_1 generator) it FLOORS on the disk filling (b_1=0) and REALIZES
     as a bulk harmonic on the annulus (b_1=1). With one period sign-FLIPPED
     (p_A = -p_B, the cobordism conjugation) it FLOORS on BOTH fillings -- opposite
     periods are not the restriction of any closed form. Realizable iff the filling
     has b_1=1 AND the periods match; the realizable set is image(H_1(dW)->H_1(W)).
  3. **Surgery moves b_1 on its own.** From the disk seed the SURGERY search commits
     a removal scored purely by the harmonic residual, so b_1 moves 0 -> 1 for both
     targets. The matched meridian then REALIZES; the flipped one still FLOORS on the
     opened handle -- the period mismatch is a separate, cohomological obstruction.
  4. **Only removal moves b_1.** The additive attach is boundary-locked at k>=1, so
     it is refused and b_1 stays frozen -- removal is the load-bearing move.
"""

import os
import subprocess
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

DEEP_EPS = 1e-7
REALIZE = 1e-3
CERT_FLOOR = 1e-2
RESTARTS = 64

# The octahedron surface (a triangulated S^2), built generically from a face list.
_OCT = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
        (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
HOLE_A, HOLE_B = (0, 1, 2), (3, 4, 5)
CYCLE_A, CYCLE_B = [(0, 1), (0, 2), (1, 2)], [(3, 4), (3, 5), (4, 5)]


def _surface(faces, weight=1.0, phase=0.0):
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(weight)
        e.setPhase(phase)
    return st


def _delete(*faces):
    drop = {tuple(sorted(f)) for f in faces}
    return [f for f in _OCT if tuple(sorted(f)) not in drop]


def _disk():
    return _surface(_delete(HOLE_A))


def _annulus():
    return _surface(_delete(HOLE_A, HOLE_B))


def _b1(st):
    return int(cob.ChainComplex.fromSpacetime(st).bettiNumbers()[1])


def _periods(vals):
    """The two boundary periods of a 1-form on CYCLE_A + CYCLE_B: p_A around
    0->1->2->0, p_B around 3->4->5->3."""
    v = {e: c for e, c in zip(CYCLE_A + CYCLE_B, vals)}
    p_a = v[(0, 1)] + v[(1, 2)] - v[(0, 2)]
    p_b = v[(3, 4)] + v[(4, 5)] - v[(3, 5)]
    return complex(p_a), complex(p_b)


def _meridian_target(flip=False):
    """The meridian carried on both boundary circles, read off the annulus's own
    harmonic 1-form. flip=False keeps equal periods (matched); flip=True negates
    circle B's period (the sign-flipped cobordism conjugation)."""
    h = cob.HodgeLaplacian(_annulus()).harmonics(1)[0]
    edges = CYCLE_A + CYCLE_B
    vals = [complex(h.amplitudeFor(list(e))) for e in edges]
    if flip:
        vals = vals[:3] + [-v for v in vals[3:]]
    return cob.Cochain(1, edges, np.asarray(vals, dtype=complex)), vals


def _decide(st, target, *, max_cones, seed=1):
    return cob.RealizabilityOracle(st).decideHarmonic(
        target, epsilon=DEEP_EPS, restarts=RESTARTS, max_cones=max_cones,
        seed=seed, growth_mode=SURGERY, connectivity_candidates=8, harmonic=True)


# --------------------------------------------------------------------------- #
class SurgeryRemovePrimitiveTest(unittest.TestCase):
    """1: removeInteriorCell moves b_1 with the boundary held bit-exact."""

    def test_remove_opens_handle_and_restore_is_bit_exact(self):
        st = _disk()
        es = cob.EigenstateSynthesis(st, 1)
        self.assertEqual(_b1(st), 0)                       # disk
        self.assertEqual(es.interiorTopCells(), [[3, 4, 5]])
        weights_before = list(es.weights())
        bnd_before = sorted(es.boundaryEdges())

        self.assertTrue(es.removeInteriorCell([3, 4, 5]))  # open the handle
        self.assertEqual(_b1(st), 1)                       # annulus: b_1 moved 0->1

        self.assertTrue(es.restoreLastRemoval())
        self.assertEqual(_b1(st), 0)                       # disk again
        self.assertEqual(list(es.weights()), weights_before)
        self.assertEqual(sorted(es.boundaryEdges()), bnd_before)

    def test_boundary_touching_cell_is_refused(self):
        st = _disk()
        es = cob.EigenstateSynthesis(st, 1)
        # {0,1,4} contains the boundary vertices 0,1 -> not an interior cell.
        self.assertFalse(es.removeInteriorCell([0, 1, 4]))
        self.assertEqual(_b1(st), 0)


# --------------------------------------------------------------------------- #
class MeridianTwoBoundaryTest(unittest.TestCase):
    """2: the two-boundary meridian 2x2 -- matched realizes only on the annulus;
    the sign-flipped conjugation floors on both fillings."""

    def test_matched_and_flipped_have_equal_and_opposite_periods(self):
        # The matched target is the annulus harmonic's restriction: equal periods
        # on the two circles. The flip negates circle B, leaving circle A intact.
        _, matched = _meridian_target(flip=False)
        _, flipped = _meridian_target(flip=True)
        pa_m, pb_m = _periods(matched)
        pa_f, pb_f = _periods(flipped)
        self.assertGreater(abs(pa_m), 0.5)               # a genuine nonzero meridian
        self.assertLess(abs(pa_m - pb_m), 1e-6)          # matched: p_A = p_B
        self.assertLess(abs(pa_f - pa_m), 1e-6)          # flip leaves circle A alone
        self.assertLess(abs(pa_f + pb_f), 1e-6)          # flipped: p_A = -p_B

    def test_matched_meridian_floors_on_disk_realizes_on_annulus(self):
        target, _ = _meridian_target(flip=False)
        disk = _decide(_disk(), target, max_cones=0)
        ann = _decide(_annulus(), target, max_cones=0)
        self.assertGreater(disk.residual, CERT_FLOOR)    # b_1=0: it bounds -> floored
        self.assertLess(ann.residual, REALIZE)           # b_1=1: survives -> realized
        self.assertLess(abs(ann.eigenvalue), 1e-2)       # carried as a harmonic

    def test_flipped_meridian_floors_on_both_fillings(self):
        target, _ = _meridian_target(flip=True)
        disk = _decide(_disk(), target, max_cones=0)
        ann = _decide(_annulus(), target, max_cones=0)
        # Opposite periods are not the restriction of any closed form, so the flip
        # floors even on the annulus, where the topology (b_1=1) is right.
        self.assertGreater(disk.residual, CERT_FLOOR)
        self.assertGreater(ann.residual, CERT_FLOOR)

    def test_the_two_fillings_have_different_b1(self):
        self.assertEqual(_b1(_disk()), 0)
        self.assertEqual(_b1(_annulus()), 1)


# --------------------------------------------------------------------------- #
class SurgeryMovesB1Test(unittest.TestCase):
    """3: the SURGERY search moves b_1 0 -> 1 on its own; the matched meridian then
    realizes, the flipped one still floors on the opened handle."""

    def test_surgery_opens_handle_and_matched_meridian_realizes(self):
        target, _ = _meridian_target(flip=False)
        for seed in (0, 1, 2):
            st = _disk()
            self.assertEqual(_b1(st), 0)                    # seed b_1
            v = _decide(st, target, max_cones=3, seed=seed)
            self.assertEqual(_b1(st), 1, f"seed {seed}: b_1 did not move")
            self.assertGreaterEqual(v.surgery_removals, 1, f"seed {seed}")
            self.assertLess(v.residual, REALIZE, f"seed {seed}: meridian floored")

    def test_surgery_opens_handle_but_flipped_meridian_still_floors(self):
        target, _ = _meridian_target(flip=True)
        for seed in (0, 1, 2):
            st = _disk()
            v = _decide(st, target, max_cones=3, seed=seed)
            # Surgery still opens the handle (the annulus floor < the disk floor, so
            # the removal improves the residual and is committed) ...
            self.assertEqual(_b1(st), 1, f"seed {seed}: b_1 did not move")
            self.assertGreaterEqual(v.surgery_removals, 1, f"seed {seed}")
            # ... but the flipped meridian never realizes: the period mismatch is a
            # cohomological obstruction surgery cannot fix.
            self.assertGreater(v.residual, CERT_FLOOR, f"seed {seed}")

    def test_surgery_is_residual_driven_not_indiscriminate(self):
        # On the annulus (already b_1=1, no interior top cell) the search has no
        # removal to make: b_1 stays 1, zero removals -- it removes only what helps.
        target, _ = _meridian_target(flip=False)
        st = _annulus()
        self.assertEqual(cob.EigenstateSynthesis(st, 1).interiorTopCells(), [])
        v = _decide(st, target, max_cones=3)
        self.assertEqual(v.surgery_removals, 0)
        self.assertEqual(_b1(st), 1)


# --------------------------------------------------------------------------- #
class OnlyRemovalMovesB1Test(unittest.TestCase):
    """4: without the remove move b_1 is frozen and the matched meridian floors."""

    def test_no_growth_leaves_b1_frozen(self):
        target, _ = _meridian_target(flip=False)
        st = _disk()
        v = _decide(st, target, max_cones=0)
        self.assertEqual(_b1(st), 0)
        self.assertGreater(v.residual, CERT_FLOOR)

    def test_additive_attach_is_boundary_locked(self):
        # Wiring a fresh triangle over an interior edge would grow dW -> the bit-
        # exact boundary guard refuses it, so additive growth cannot move b_1.
        st = _disk()
        es = cob.EigenstateSynthesis(st, 1)
        self.assertFalse(es.attachInteriorVertex([[3, 4]]))
        self.assertEqual(_b1(st), 0)


# --------------------------------------------------------------------------- #
class HarmonicCriterionTest(unittest.TestCase):
    """The harmonic residual (carried-by-H_1) is what separates the topologies;
    the eigenvalue-agnostic residual is under-constrained on a small boundary."""

    def test_eigenvalue_agnostic_residual_does_not_separate(self):
        target, _ = _meridian_target(flip=False)
        # Without harmonic=True a non-harmonic eigenvector is accepted, so even the
        # disk filling can host the meridian at a nonzero eigenvalue -> no clean
        # obstruction. (Contrast test_matched_meridian_floors_on_disk_realizes_on_annulus.)
        st = _disk()
        v = cob.RealizabilityOracle(st).decideHarmonic(
            target, epsilon=DEEP_EPS, restarts=RESTARTS, max_cones=0, seed=1,
            growth_mode=SURGERY, connectivity_candidates=8, harmonic=False)
        # It can be driven well below the harmonic floor (an eigenvector at some
        # lambda != 0), confirming the harmonic criterion is the discriminating one.
        self.assertLess(v.residual, CERT_FLOOR)
        self.assertGreater(abs(v.eigenvalue), 1e-2)


# --------------------------------------------------------------------------- #
class ExampleSelfVerifiesTest(unittest.TestCase):
    """The committed example runs end-to-end and exits 0 (its own assertions)."""

    def test_example_exits_zero(self):
        here = os.path.dirname(os.path.abspath(__file__))
        example = os.path.join(here, "..", "..", "examples", "cobordism",
                               "emergent_bulk_realizability.py")
        self.assertTrue(os.path.exists(example))
        result = subprocess.run(
            [sys.executable, example, "--no-write"],
            capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0,
                         msg=f"example exited {result.returncode}\n"
                             f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
