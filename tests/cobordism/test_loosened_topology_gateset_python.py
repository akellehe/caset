# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Re-characterize the realizable gate set under fully-loosened topology (#216).

Independent re-derivation of the claims the example
(``examples/cobordism/loosened_topology_gateset.py``) makes, plus a self-verify
that the committed example exits 0. The headline the prior loosened attempt
(#215) could not establish -- because its k=0 Choi-vector construction floored
the S_3 positive controls (CNOT, SWAP, ...) at r ~ 0.40 -- is here pinned down:

  1. **The validity anchor: the six S_3 controls realize at gap 0.** Each is the
     genuine ``DijkgraafWitten.map()`` of an emergent mapping cylinder, read
     through the metric-free state sum -- so it realizes exactly, not floored.
  2. **Loosening the topology EXPANDS the realizable operation image beyond S_3.**
     The disconnected ``disjointUnion`` cap-and-create cobordism (b_0 = 2) gives
     the rank-1 ``|st><st|`` -- a non-invertible map outside S_3 -- and the
     realizable matrix algebra grows 5 -> 10 dim.
  3. **No superposition/entangling gate realizes, at any topology.** Every DW map
     is integer-quantized; the superposition/phase/entangling gates floor at
     gap > 1e-2 to the realizable image, hole open or closed.
  4. **The boundary grows b_1 holes on its own (surgery).** From a closed
     icosahedron (b_1 = 0) the remove move opens three pairwise-disjoint faces,
     b_1: 0 -> 1 -> 2, so dim Z = 2^{b_1} scales 1 -> 2 -> 4 to the 2-qubit space.
  5. **The emergent-b_1 freedom carries STATES, not gates.** The superposed
     meridian floors at b_1 = 0 and realizes once surgery opens b_1 = 1; the
     S_3-control state does NOT floor; the period-mismatched control floors.
"""

import os
import subprocess
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism
DijkgraafWitten = cob.DijkgraafWitten
Cocycle = cob.Cocycle
Cobordism = cob.Cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

DIM = 4
REALIZE = 1e-7
FLOOR = 1e-2
STATE_REALIZE = 1e-3
STATE_FLOOR = 1e-2


# --------------------------------------------------------------------------- #
# Fixtures (the established cobordism-test idiom).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)


def _product_torus():
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _solid_torus():
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


_SWAP = [v * 3 + u for u in range(3) for v in range(3)]
_SEVEN = sorted({tuple(sorted(((i) % 7, (i + step) % 7, (i + 3) % 7)))
                 for i in range(7) for step in (1, 2)})


def _seven_vertex_torus():
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    verts = [st.createVertex(i) for i in range(7)]
    for tri in _SEVEN:
        st.createSimplex([verts[i] for i in tri])
    return st


def _dw_map(W, cocycle=Cocycle.Trivial):
    return np.asarray(DijkgraafWitten(W, cocycle).map()).real


def _s3_image():
    swap = _dw_map(Cobordism.twistedCylinder(_product_torus(), _SWAP))
    cycle = _dw_map(Cobordism.twistedCylinder(_seven_vertex_torus(),
                                              [(2 * i) % 7 for i in range(7)]))

    def key(m):
        return tuple(np.round(m).astype(int).reshape(-1))

    group = {key(np.eye(DIM)): np.eye(DIM)}
    frontier = [np.eye(DIM)]
    while frontier:
        element = frontier.pop()
        for gen in (swap, cycle):
            prod = gen @ element
            if key(prod) not in group:
                group[key(prod)] = prod
                frontier.append(prod)
    return list(group.values())


def _cap():
    return _dw_map(Cobordism.disjointUnion(_solid_torus(), _solid_torus()))


def _algebra_dimension(generators):
    def flat(m):
        return np.asarray(m, dtype=complex).reshape(-1)

    basis = [flat(np.eye(DIM))]

    def independent(v):
        return np.linalg.matrix_rank(np.array(basis + [v]), tol=1e-9) > len(basis)

    for g in generators:
        if independent(flat(g)):
            basis.append(flat(g))
    changed = True
    while changed:
        changed = False
        mats = [b.reshape(DIM, DIM) for b in basis]
        for a in mats:
            for b in mats:
                if independent(flat(a @ b)):
                    basis.append(flat(a @ b))
                    changed = True
    return len(basis)


def _h2():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _gap(U, image):
    return float(min(np.linalg.norm(np.asarray(U) - g) for g in image))


# --------------------------------------------------------------------------- #
# 1: the validity anchor -- the six S_3 controls realize at gap 0.
# --------------------------------------------------------------------------- #
class ValidityAnchorTest(unittest.TestCase):
    """The exact check the prior attempt failed: S_3 must realize."""

    def setUp(self):
        self.s3 = _s3_image()

    def test_s3_is_six_holonomy_permutations_fixing_trivial_class(self):
        self.assertEqual(len(self.s3), 6)
        for g in self.s3:
            self.assertTrue(np.allclose(g.sum(0), 1) and np.allclose(g.sum(1), 1))
            self.assertAlmostEqual(g[0, 0], 1.0)  # fixes the trivial class

    def test_each_s3_control_realizes_at_gap_zero(self):
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=float)
        rcnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                         dtype=float)
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=float)
        controls = [np.eye(DIM), swap, cnot, rcnot,
                    _dw_map(Cobordism.twistedCylinder(_seven_vertex_torus(),
                                                      [(2 * i) % 7 for i in range(7)]))]
        for U in controls:
            self.assertLess(_gap(U, self.s3), REALIZE,
                            "an S_3 control floored -- construction invalid")


# --------------------------------------------------------------------------- #
# 2: loosening EXPANDS the realizable operation image beyond S_3.
# --------------------------------------------------------------------------- #
class LooseningExpandsOperationImageTest(unittest.TestCase):

    def test_cap_is_rank_one_and_outside_s3(self):
        cap = _cap()
        self.assertEqual(np.linalg.matrix_rank(cap, tol=1e-9), 1)
        self.assertFalse(any(np.allclose(cap, g) for g in _s3_image()))

    def test_disjoint_union_bulk_is_disconnected(self):
        W = Cobordism.disjointUnion(_solid_torus(), _solid_torus())
        betti = cob.ChainComplex.fromSpacetime(W).bettiNumbers()
        self.assertEqual(betti[0], 2)   # b_0 = 2: not a cylinder

    def test_realizable_algebra_grows_five_to_ten(self):
        s3 = _s3_image()
        self.assertEqual(_algebra_dimension(s3), 5)
        self.assertEqual(_algebra_dimension(s3 + [_cap()]), 10)

    def test_every_realizable_map_is_integer_quantized(self):
        for g in _s3_image() + [_cap()]:
            np.testing.assert_allclose(g, np.round(g), atol=1e-9)


# --------------------------------------------------------------------------- #
# 3: no superposition/entangling gate realizes, at any topology.
# --------------------------------------------------------------------------- #
class NoSuperpositionGateRealizesTest(unittest.TestCase):

    def setUp(self):
        s3 = _s3_image()
        cap = _cap()
        # The realizable image: S_3 + the cap's S_3-translates (each a glue map).
        self.image = s3 + [g @ cap @ h for g in s3 for h in s3]

    def test_hadamard_family_floors(self):
        i2 = np.eye(2, dtype=complex)
        for U in (np.kron(_h2(), i2), np.kron(i2, _h2()), np.kron(_h2(), _h2())):
            self.assertGreater(_gap(U, self.image), FLOOR)

    def test_phase_and_entangler_family_floors(self):
        import cmath
        cz = np.diag([1, 1, 1, -1]).astype(complex)
        cphase = np.diag([1, 1, 1, cmath.exp(1j * np.pi / 4)]).astype(complex)
        iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                         dtype=complex)
        t1 = np.kron(np.diag([1, cmath.exp(1j * np.pi / 4)]), np.eye(2))
        for U in (cz, cphase, iswap, t1):
            self.assertGreater(_gap(U, self.image), FLOOR)

    def test_integer_gates_outside_s3_also_floor(self):
        # On the integer lattice yet still unrealizable: X(x)X moves the trivial
        # class, Z(x)Z is a diagonal sign -- being integer is necessary, not
        # sufficient.
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        for U in (np.kron(x, x), np.kron(z, z)):
            self.assertGreater(_gap(U, self.image), FLOOR)


# --------------------------------------------------------------------------- #
# 4: the boundary grows b_1 holes on its own (surgery).
# --------------------------------------------------------------------------- #
_ICOSA = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
          (1, 2, 6), (2, 3, 7), (3, 4, 8), (4, 5, 9), (1, 5, 10),
          (2, 6, 7), (3, 7, 8), (4, 8, 9), (5, 9, 10), (1, 6, 10),
          (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]


def _surface(faces):
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _b1(st):
    return int(cob.ChainComplex.fromSpacetime(st).bettiNumbers()[1])


class BoundaryGrowsB1Test(unittest.TestCase):

    def test_icosahedron_is_a_sphere(self):
        betti = cob.ChainComplex.fromSpacetime(_surface(_ICOSA)).bettiNumbers()
        self.assertEqual([betti[0], betti[1], betti[2]], [1, 0, 1])  # S^2

    def test_surgery_grows_b1_zero_to_two_and_dim_z_one_to_four(self):
        st = _surface(_ICOSA)
        es = cob.EigenstateSynthesis(st, 1)
        self.assertEqual(_b1(st), 0)
        observed = []
        for face in [(0, 1, 2), (3, 4, 8), (9, 10, 11)]:  # pairwise disjoint
            self.assertTrue(es.removeInteriorCell(list(face)))
            observed.append((_b1(st), 2 ** _b1(st)))
        # b_1: 0 (disk) -> 1 (annulus) -> 2 (pair of pants); Z = 2^{b_1}: 1,2,4.
        self.assertEqual(observed, [(0, 1), (1, 2), (2, 4)])
        self.assertEqual(len(cob.HodgeLaplacian(st).harmonics(1)), 2)  # ker L_1 = b_1


# --------------------------------------------------------------------------- #
# 5: the emergent-b_1 freedom carries STATES, not gates.
# --------------------------------------------------------------------------- #
_OCT = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
        (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
_EDGES = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]


def _without(*drop):
    gone = {tuple(sorted(f)) for f in drop}
    return [f for f in _OCT if tuple(sorted(f)) not in gone]


def _meridian(flip=False):
    annulus = _surface(_without((0, 1, 2), (3, 4, 5)))
    h = cob.HodgeLaplacian(annulus).harmonics(1)[0]
    vals = [complex(h.amplitudeFor(list(e))) for e in _EDGES]
    if flip:
        vals = vals[:3] + [-v for v in vals[3:]]
    return cob.Cochain(1, _EDGES, np.asarray(vals, dtype=complex))


def _decide_state(st, target, *, max_cones, seed=1):
    return cob.RealizabilityOracle(st).decideHarmonic(
        target, epsilon=1e-9, restarts=64, max_cones=max_cones, seed=seed,
        growth_mode=SURGERY, connectivity_candidates=8, harmonic=True)


class EmergentB1CarriesStatesTest(unittest.TestCase):

    def test_superposed_state_floors_at_b1_zero_realizes_after_surgery(self):
        matched = _meridian(flip=False)
        v_disk = _decide_state(_surface(_without((0, 1, 2))), matched, max_cones=0)
        self.assertGreater(v_disk.residual, STATE_FLOOR)   # b_1 = 0: floors
        st = _surface(_without((0, 1, 2)))
        v = _decide_state(st, matched, max_cones=3, seed=1)
        self.assertEqual(_b1(st), 1)                        # surgery opened b_1
        self.assertGreaterEqual(v.surgery_removals, 1)
        self.assertLess(v.residual, STATE_REALIZE)          # superposed state realizes

    def test_s3_control_state_does_not_floor(self):
        # The carried meridian on the annulus (the identity transition on the
        # surviving mode -- the S_3-control state analogue) realizes: the state
        # level does not floor the controls (the prior attempt's other failure).
        v = _decide_state(_surface(_without((0, 1, 2), (3, 4, 5))),
                          _meridian(flip=False), max_cones=0)
        self.assertLess(v.residual, STATE_REALIZE)

    def test_period_mismatched_state_floors_even_with_handle_open(self):
        v = _decide_state(_surface(_without((0, 1, 2), (3, 4, 5))),
                          _meridian(flip=True), max_cones=0)
        self.assertGreater(v.residual, STATE_FLOOR)


# --------------------------------------------------------------------------- #
# 6: the committed example runs end-to-end and exits 0.
# --------------------------------------------------------------------------- #
class ExampleSelfVerifiesTest(unittest.TestCase):

    def test_example_exits_zero(self):
        here = os.path.dirname(os.path.abspath(__file__))
        example = os.path.join(here, "..", "..", "examples", "cobordism",
                               "loosened_topology_gateset.py")
        self.assertTrue(os.path.exists(example))
        result = subprocess.run(
            [sys.executable, example, "--no-write"],
            capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0,
                         msg=f"example exited {result.returncode}\n"
                             f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
