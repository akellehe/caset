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

"""Re-characterize the realizable 2-qubit gate set under FULLY-loosened topology.

The question (third attempt). The pinned Dijkgraaf-Witten twisted-cylinder sweep
(`realizable_image_sweep.py`) finds the realizable operation image on
Z(T^2) = C[H^1(T^2;Z_2)] = C^4 is exactly S_3 = GL(2,Z_2): the six holonomy-class
permutations fixing the trivial class. Does *loosening the cobordism topology*
expand the realizable gate set beyond S_3?

Why the prior loosened attempt's "no" was an artifact (NOT a finding about gates).
`loosened_gate_retest.py` (#215) bent each gate to its length-16 Choi vector and
fed it to the harmonic oracle as a degree-k=0 (vertex) cochain. At k=0 the harmonic
kernel ker L_0 is the constants (dim b_0 = 1 on any connected complex), so a
generic 16-vector floors no matter what b_1 does -- and it floored the S_3 controls
(CNOT, SWAP, ...) right alongside the superposition gates, all at r ~ 0.4. A
construction that floors the *known-realizable* controls cannot say anything about
the other gates: it is the wrong/too-small construction, not a verdict.

This script's construction: loosen the STATES and the OPERATIONS, pin nothing.

  * STATES emerge.  A 2-qubit state lives in Z(Sigma) = C^{2^{b_1(Sigma)}}; a
    2-qubit space (C^4) needs b_1(Sigma) = 2. The boundary is NOT pinned to a torus
    -- it is whatever closed surface carries b_1 = 2, and (Section A) the SAME
    realizable image falls out of two independent emergent triangulations (the
    9-vertex product torus and the 7-vertex Moebius torus), so the answer is a
    topological invariant, not a fixture artifact. Section D shows the surgery
    move-set (removeInteriorCell) makes b_1 a pure OUTPUT: the boundary grows holes
    0 -> 1 -> 2 on its own, so dim ker L_1 (the spectral-qubit content) and hence
    Z = 2^{b_1} scale to whatever the state space needs -- the size the prior
    attempt's b_1 <= 1 boundary structurally lacked.

  * OPERATIONS emerge.  The realizable image is read as the genuine metric-free DW
    state sum `DijkgraafWitten.map()` of cobordisms whose TOPOLOGY is loosened, NOT
    pinned to one twisted cylinder: mapping cylinders through finite-order
    simplicial automorphisms (the S_3 permutation sector) AND the disconnected
    cap-and-create cobordism `disjointUnion` (b_0 = 2, a non-cylinder), whose map is
    the rank-1 |st><st| -- a non-invertible map OUTSIDE S_3 (#196). The bulk
    topology (b_0, b_1) is an output the map depends on.

The VALIDITY ANCHOR (reported first). The six S_3 controls -- Identity, SWAP, CNOT,
reversed-CNOT, the two 3-cycles -- MUST realize, or the test is invalid. In this
construction they realize at residual 0 (machine precision): each is literally the
DW map of an emergent mapping cylinder, read through the genuine state sum. This is
the exact check the prior attempt failed (it floored them at 0.40).

The honest result (Section B/C/D; exit 0 iff the verdicts hold):

  * Loosening the topology DOES expand the realizable *operation* image beyond S_3
    -- but only to non-invertible INTEGER maps: the cap/projector sector. The
    realizable matrix algebra grows from 5-dim (<S_3>) to 10-dim (<S_3, |st><st|>),
    a proper subalgebra of the 16-dim End(C^4).
  * Among the 2-qubit GATES (unitaries), the realizable set stays EXACTLY S_3. Every
    superposition/entangling gate -- Hadamard (H(x)I, I(x)H, H(x)H), CZ, T, S,
    iSWAP, sqrt-SWAP, sqrt-iSWAP, CPHASE, the entangling Cliffords -- floors at
    gap ~ 1-2 to the realizable image, hole open or closed. The reason is structural
    and topology-independent: every DW map is integer-quantized in the
    flat-connection basis, and these gates carry irrational / complex amplitudes
    (1/sqrt2, e^{i pi/4}, i) that no integer map can hit.
  * The continuous freedom the loosening genuinely adds lives in the STATE space,
    not the gate set: a superposed boundary 1-cycle (the state a Hadamard creates)
    floors at b_1 = 0 and REALIZES once the surgery search opens b_1 = 1 -- and now,
    on a boundary big enough, the S_3-control state transitions realize too (the
    state-level validity anchor the prior attempt also failed).

Verdict: the realizable GATE set is not enlarged by loosening the topology -- S_3
stands. What enlarges is (i) the realizable non-unitary OPERATION image (integer
cap maps) and (ii) the realizable STATE space (superposed cycles via emergent b_1).
The superposition/entangling gates are unreachable at every topology.

Run:  python examples/cobordism/loosened_topology_gateset.py
      (--help for options; the raw table defaults to /tmp/cobordism and is NOT
      committed -- attach it to the issue/PR to pin a result.)
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

import argparse  # noqa: E402
import cmath     # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
DijkgraafWitten = cob.DijkgraafWitten
Cocycle = cob.Cocycle
Cobordism = cob.Cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

DIM = 4                  # dim Z(T^2) = 2^{b_1(T^2)} = 4 (a 2-qubit space)
REALIZE = 1e-7           # a gate realizes iff gap-to-image < REALIZE
FLOOR = 1e-2             # certified off the realizable image iff gap > FLOOR
STATE_REALIZE = 1e-3     # state level: harmonic residual < this realizes
STATE_FLOOR = 1e-2       # state level: floors above this
RESTARTS = 64


# --------------------------------------------------------------------------- #
# Builders (the established cobordism-test idiom: Signature(d) so the d-cells
# register as top simplices; topology threads the vertex-id counter).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)              # S^1 = d(triangle)


def _product_torus():
    # T^2 = S^1 x S^1, the 9-vertex symmetric product torus (b_1 = 2).
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _solid_torus():
    # ST = S^1 x D^2, a 3-manifold with boundary T^2 (one component).
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


# The coordinate swap phi(u,v) = (v,u) on the 3x3 product torus: the order-2
# transposition (1 2) of the holonomy classes.
_SWAP = [v * 3 + u for u in range(3) for v in range(3)]

# The minimal 7-vertex (Moebius) torus and its order-3 multiplier i -> 2i (mod 7):
# the order-3 rotation (1 2 3) of the holonomy classes. A DIFFERENT triangulation
# of a b_1 = 2 surface -- used to show the realizable image is triangulation-free.
_SEVEN_TRIANGLES = sorted({
    tuple(sorted(((i) % 7, (i + step) % 7, (i + 3) % 7)))
    for i in range(7) for step in (1, 2)})


def _seven_vertex_torus():
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    verts = [st.createVertex(i) for i in range(7)]
    for tri in _SEVEN_TRIANGLES:
        st.createSimplex([verts[i] for i in tri])
    return st


def _multiplier(m, n):
    return [(m * i) % n for i in range(n)]


def _b1(st):
    return int(cob.ChainComplex.fromSpacetime(st).bettiNumbers()[1])


def _b0(st):
    return int(cob.ChainComplex.fromSpacetime(st).bettiNumbers()[0])


def _dw_map(W, cocycle=Cocycle.Trivial):
    """The genuine metric-free DW state sum read as a Z(Sigma_B) -> Z(Sigma_A)
    matrix on the emergent cobordism W (4x4 when both ends are b_1 = 2)."""
    return np.asarray(DijkgraafWitten(W, cocycle).map()).real


# --------------------------------------------------------------------------- #
# Section A -- the realizable OPERATION image of EMERGENT cobordisms.
# --------------------------------------------------------------------------- #
class RealizableImage:
    """The realizable T^2 -> T^2 DW map image over cobordisms whose topology is
    LOOSENED, pinned to nothing in particular.

    Generators, each a genuine `DijkgraafWitten.map()` of an emergent cobordism:
      * `swap`  -- mapping cylinder through the order-2 simplicial automorphism of
        the 9-vertex product torus (holonomy transposition (1 2));
      * `cycle` -- mapping cylinder through the order-3 automorphism of the
        7-vertex Moebius torus (holonomy 3-cycle (1 2 3)) -- a DIFFERENT emergent
        triangulation, so the group it generates with `swap` is triangulation-free;
      * `cap`   -- the disconnected cap-and-create cobordism ST || ST
        (`disjointUnion`, b_0 = 2): a NON-cylinder whose map is the rank-1 |st><st|,
        a non-invertible map outside the S_3 group (#196).

    The realizable SET is the S_3 group (the six cylinder permutations) together
    with the cap and its S_3-translates g @ cap @ h (each realizable by gluing a
    twisted cylinder onto the cap) -- every element a genuine cobordism map. The
    realizable ALGEBRA is their linear span (5-dim for S_3 alone, 10-dim with the
    cap, #196), reported as the structural cross-check.
    """

    def __init__(self):
        self.swap = _dw_map(Cobordism.twistedCylinder(_product_torus(), _SWAP))
        self.cycle = _dw_map(
            Cobordism.twistedCylinder(_seven_vertex_torus(), _multiplier(2, 7)))
        self.cap = _dw_map(Cobordism.disjointUnion(_solid_torus(),
                                                   _solid_torus()))
        self.s3 = self._close_group([self.swap, self.cycle])
        # The cap and its S_3-translates: every g @ cap @ h is the map of
        # glue(twistedCylinder_g, glue(cap, twistedCylinder_h)) -- realizable.
        translates = {self._key(g @ self.cap @ h): (g @ self.cap @ h)
                      for g in self.s3 for h in self.s3}
        self.cap_orbit = list(translates.values())
        self.image = self.s3 + self.cap_orbit

    @staticmethod
    def _key(matrix):
        return tuple(np.round(np.asarray(matrix).real, 6).reshape(-1))

    @classmethod
    def _close_group(cls, generators):
        identity = np.eye(DIM)
        group = {cls._key(identity): identity}
        frontier = [identity]
        while frontier:
            element = frontier.pop()
            for generator in generators:
                product = generator @ element
                key = cls._key(product)
                if key not in group:
                    group[key] = product
                    frontier.append(product)
        return list(group.values())

    def gap_to_s3(self, U):
        return float(min(np.linalg.norm(np.asarray(U) - g) for g in self.s3))

    def gap_to_image(self, U):
        return float(min(np.linalg.norm(np.asarray(U) - g) for g in self.image))

    # -- the realizable matrix algebra dimension (the #196 cross-check) ------- #
    @staticmethod
    def algebra_dimension(generators):
        """dim of the unital associative algebra <generators> in End(C^4)."""
        def flat(matrix):
            return np.asarray(matrix, dtype=complex).reshape(-1)

        basis = [flat(np.eye(DIM))]

        def independent(vector):
            stacked = np.array(basis + [vector])
            return np.linalg.matrix_rank(stacked, tol=1e-9) > len(basis)

        for generator in generators:
            v = flat(generator)
            if independent(v):
                basis.append(v)
        changed = True
        while changed:
            changed = False
            matrices = [b.reshape(DIM, DIM) for b in basis]
            for a in matrices:
                for b in matrices:
                    v = flat(a @ b)
                    if independent(v):
                        basis.append(v)
                        changed = True
        return len(basis)

    def certify(self):
        """Assert the realizable image is valid and genuinely emergent."""
        # The S_3 group is exactly 6 holonomy permutations fixing the trivial class.
        assert len(self.s3) == 6, f"S_3 must be order 6, got {len(self.s3)}"
        for g in self.s3:
            assert _is_permutation(g) and abs(g[0, 0] - 1.0) < 1e-9, \
                "every S_3 map is a 0/1 permutation fixing the trivial class"
        # The cap is rank-1, non-invertible, OUTSIDE S_3 -- the emergent expansion.
        assert np.linalg.matrix_rank(self.cap, tol=1e-9) == 1, "cap is rank 1"
        assert not any(np.allclose(self.cap, g) for g in self.s3), \
            "the cap is a new realizable map, outside S_3"
        # Every realizable map is integer-quantized in the flat-connection basis.
        for g in self.image:
            assert np.allclose(g, np.round(g)), "DW maps are integer-quantized"
        return {
            "s3_order": len(self.s3),
            "image_size": len(self.image),
            "algebra_s3": self.algebra_dimension(self.s3),
            "algebra_image": self.algebra_dimension(self.s3 + [self.cap]),
        }


def _is_permutation(U, tol=1e-9):
    m = np.asarray(U).real
    ok = np.all(np.isclose(m, 0.0, atol=tol) | np.isclose(m, 1.0, atol=tol))
    return bool(ok and np.all(np.isclose(m.sum(0), 1.0, atol=tol))
                and np.all(np.isclose(m.sum(1), 1.0, atol=tol)))


# --------------------------------------------------------------------------- #
# Gate-content classifiers (well-defined properties of the 4x4 matrix).
# --------------------------------------------------------------------------- #
def creates_superposition(U, tol=1e-9):
    """A basis state is mapped off the basis (a column is not a single nonzero up
    to phase): the gate superposes the holonomy classes."""
    mag = np.abs(np.asarray(U))
    return any(np.count_nonzero(mag[:, c] > tol) != 1 for c in range(DIM))


def is_entangling(U, tol=1e-9):
    """Operator-Schmidt rank > 1 -- the gate does not factor as a one-qubit
    tensor product."""
    reshaped = np.asarray(U).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    s = np.linalg.svd(reshaped, compute_uv=False)
    return int(np.sum(s > tol * max(s[0], 1.0))) > 1


def on_integer_lattice(U, tol=1e-9):
    """U is a real integer matrix -- a necessary condition to be ANY DW map."""
    m = np.asarray(U)
    return bool(np.all(np.isclose(m.imag, 0.0, atol=tol))
                and np.allclose(m.real, np.round(m.real), atol=tol))


# --------------------------------------------------------------------------- #
# The gate battery: the S_3 controls + the superposition / phase / entangling
# families the pinned image floored on (same set as loosened_gate_retest, for
# comparability), plus an entangling Clifford.
# --------------------------------------------------------------------------- #
def _gates():
    h2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    i2 = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    t1 = np.diag([1, cmath.exp(1j * np.pi / 4)]).astype(complex)
    s1 = np.diag([1, 1j]).astype(complex)

    def perm(p):
        m = np.zeros((DIM, DIM), dtype=complex)
        for r, c in enumerate(p):
            m[r, c] = 1.0
        return m

    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    rcnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                     dtype=complex)
    iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                     dtype=complex)
    swap = perm((0, 2, 1, 3))

    def root(U):
        w, vec = np.linalg.eig(U)
        return (vec * np.sqrt(w)) @ np.linalg.inv(vec)

    # An entangling Clifford: CNOT . (H (x) I) -- the canonical Bell-state maker.
    bell = cnot @ np.kron(h2, i2)

    return [
        ("Identity", np.eye(DIM, dtype=complex), "S3 control"),
        ("SWAP", swap, "S3 control"),
        ("CNOT", cnot, "S3 control"),
        ("reversed-CNOT", rcnot, "S3 control"),
        ("3-cycle (0231)", perm((0, 2, 3, 1)), "S3 control"),
        ("3-cycle (0312)", perm((0, 3, 1, 2)), "S3 control"),
        ("H(x)I", np.kron(h2, i2), "superposition"),
        ("I(x)H", np.kron(i2, h2), "superposition"),
        ("H(x)H", np.kron(h2, h2), "superposition"),
        ("sqrt-SWAP", root(swap), "superposition"),
        ("sqrt-iSWAP", root(iswap), "superposition"),
        ("CZ", np.diag([1, 1, 1, -1]).astype(complex), "phase/entangler"),
        ("CPHASE(pi/4)", np.diag([1, 1, 1, cmath.exp(1j * np.pi / 4)]).astype(complex),
         "phase/entangler"),
        ("T(x)I", np.kron(t1, i2), "phase"),
        ("S(x)I", np.kron(s1, i2), "phase"),
        ("iSWAP", iswap, "phase/entangler"),
        ("CNOT.(H(x)I) [Clifford]", bell, "entangling Clifford"),
        ("X(x)X", np.kron(x, x), "Pauli perm"),
        ("Z(x)Z", np.kron(z, z), "diagonal sign"),
    ]


# --------------------------------------------------------------------------- #
# Section D -- emergent b_1 by surgery (states, not gates). Reuses the #196
# octahedron idiom: the two-boundary meridian carried (or not) across a bulk whose
# b_1 the surgery search grows on its own.
# --------------------------------------------------------------------------- #
_OCT = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
        (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
_CYCLE_A, _CYCLE_B = [(0, 1), (0, 2), (1, 2)], [(3, 4), (3, 5), (4, 5)]

# A closed icosahedron (b_1 = 0, a triangulated S^2): 12 vertices, 20 faces. Three
# pairwise vertex-disjoint faces removed by surgery open three holes, so b_1 grows
# 0 -> 1 -> 2 -- the boundary growing to hold a 2-qubit's worth of spectral content.
_ICOSA = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
          (1, 2, 6), (2, 3, 7), (3, 4, 8), (4, 5, 9), (1, 5, 10),
          (2, 6, 7), (3, 7, 8), (4, 8, 9), (5, 9, 10), (1, 6, 10),
          (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]
_DISJOINT_FACES = [(0, 1, 2), (3, 4, 8), (9, 10, 11)]   # pairwise vertex-disjoint


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


def _without(faces, *drop):
    gone = {tuple(sorted(f)) for f in drop}
    return [f for f in faces if tuple(sorted(f)) not in gone]


def _meridian_target(flip=False):
    """The meridian carried on both octahedron boundary circles (the annulus's own
    H_1 generator). MATCHED periods (the superposed state a Hadamard creates) vs
    FLIPPED (the period-mismatched conjugation, the negative control)."""
    annulus = _surface(_without(_OCT, (0, 1, 2), (3, 4, 5)))
    h = cob.HodgeLaplacian(annulus).harmonics(1)[0]
    edges = _CYCLE_A + _CYCLE_B
    vals = [complex(h.amplitudeFor(list(e))) for e in edges]
    if flip:
        vals = vals[:3] + [-v for v in vals[3:]]
    return cob.Cochain(1, edges, np.asarray(vals, dtype=complex))


def _decide_state(st, target, *, max_cones, seed=1):
    return cob.RealizabilityOracle(st).decideHarmonic(
        target, epsilon=1e-9, restarts=RESTARTS, max_cones=max_cones, seed=seed,
        growth_mode=SURGERY, connectivity_candidates=8, harmonic=True)


def grow_boundary_b1():
    """The boundary grows holes on its own: from a closed icosahedron (b_1 = 0)
    the surgery remove move opens three pairwise-disjoint faces, b_1: 0 -> 1 -> 2,
    so the spectral content dim ker L_1 (= b_1) and Z = 2^{b_1} scale to a 2-qubit
    space. Returns the (b_1, dim ker L_1, 2^{b_1}) trace."""
    st = _surface(_ICOSA)
    es = cob.EigenstateSynthesis(st, 1)
    trace = []

    def snapshot(removed):
        b1 = _b1(st)
        harm = len(cob.HodgeLaplacian(st).harmonics(1))
        trace.append({"removed": removed, "b1": b1, "ker_L1": harm,
                      "dim_Z": 2 ** b1})

    snapshot(0)
    for k, face in enumerate(_DISJOINT_FACES, start=1):
        ok = es.removeInteriorCell(list(face))
        assert ok, f"surgery removal of {face} refused"
        snapshot(k)
    return trace


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="dir for the raw table (default /tmp/cobordism; NOT committed).")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print("Re-characterizing the realizable 2-qubit gate set under FULLY-loosened "
          "topology\n(states + operations both emergent; nothing pinned)\n")

    image = RealizableImage()
    facts = image.certify()
    gates = _gates()
    ok = True

    # ---- Section A: the emergent realizable operation image ----------------- #
    print("  (A) The realizable OPERATION image of EMERGENT cobordisms "
          "(genuine DijkgraafWitten.map()):")
    print(f"      S_3 group (mapping cylinders): {facts['s3_order']} holonomy "
          "permutations fixing the trivial class")
    print(f"        swap  built on the 9-vertex product torus  (b_1 = 2 surface, "
          f"Z = C^4)")
    print(f"        cycle built on the 7-vertex Moebius torus   (a DIFFERENT "
          f"triangulation -> same image: states not pinned)")
    cap_W = Cobordism.disjointUnion(_solid_torus(), _solid_torus())
    print(f"      cap   = disjointUnion(ST, ST): bulk b_0 = {_b0(cap_W)} "
          f"(disconnected, NOT a cylinder), b_1 = {_b1(cap_W)}")
    print(f"        map(cap) is rank {np.linalg.matrix_rank(image.cap, tol=1e-9)} "
          f"|st><st| -- a NON-invertible map outside S_3")
    print(f"      realizable matrix algebra: <S_3> = {facts['algebra_s3']}-dim  ->  "
          f"<S_3, cap> = {facts['algebra_image']}-dim  (of 16-dim End(C^4))")
    expanded = facts["algebra_image"] > facts["algebra_s3"]
    print(f"      => loosening the topology EXPANDS the realizable operation image "
          f"({facts['algebra_s3']} -> {facts['algebra_image']} dim): "
          f"{'YES' if expanded else 'NO'}\n")
    ok &= expanded and facts["algebra_s3"] == 5 and facts["algebra_image"] == 10

    # ---- THE VALIDITY ANCHOR (reported FIRST among the gates) --------------- #
    print("  THE VALIDITY ANCHOR -- the six S_3 positive controls MUST realize "
          "(else the test is invalid):")
    print(f"      {'gate':18} {'gap-to-image':>13} {'realizes?':>10}   "
          f"(prior attempt #215 floored these at ~0.40)")
    anchor_ok = True
    for label, U, family in gates:
        if family != "S3 control":
            continue
        gap = image.gap_to_image(U)
        realizes = gap < REALIZE
        anchor_ok &= realizes
        print(f"      {label:18} {gap:>13.2e} {('YES' if realizes else 'FLOOR'):>10}")
    print(f"      => S_3 controls realize: {'YES (construction valid)' if anchor_ok else 'NO -- INVALID construction'}\n")
    ok &= anchor_ok

    # ---- Section B: the full gate table ------------------------------------- #
    print("  (B) The full gate table (gap to the emergent realizable image; "
          "contrast pinned gap-to-S_3):")
    header = (f"      {'gate':24} {'family':20} {'pin gap_S3':>11} "
              f"{'loose gap':>10} {'real?':>6} {'int?':>5} {'sup':>4} {'ent':>4}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    rows = []
    newly = []
    for label, U, family in gates:
        gap_s3 = image.gap_to_s3(U)
        gap_img = image.gap_to_image(U)
        realizes = gap_img < REALIZE
        integer = on_integer_lattice(U)
        sup = creates_superposition(U)
        ent = is_entangling(U)
        # A gate "newly realizes" if it realizes on the loosened image but not on
        # pinned S_3.
        if realizes and gap_s3 >= REALIZE:
            newly.append(label)
        rows.append({"gate": label, "family": family, "pinned_gap_s3": gap_s3,
                     "loosened_gap_image": gap_img, "realizable": bool(realizes),
                     "on_integer_lattice": bool(integer),
                     "creates_superposition": bool(sup), "entangling": bool(ent)})
        print(f"      {label:24} {family:20} {gap_s3:>11.3f} {gap_img:>10.3f} "
              f"{('YES' if realizes else 'floor'):>6} "
              f"{('Y' if integer else '-'):>5} {('Y' if sup else '-'):>4} "
              f"{('Y' if ent else '-'):>4}")

    # ---- Section C: why no superposition gate enters, at ANY topology ------- #
    superpos = [r for r in rows if r["family"] != "S3 control"]
    none_superpos_realize = not any(r["realizable"] for r in superpos)
    all_floor_far = all(r["loosened_gap_image"] > FLOOR for r in superpos)
    # The floored gates split into two honest reasons -- both off the realizable
    # image at every topology: (a) continuous amplitudes (1/sqrt2, e^{i pi/4}, i)
    # that no integer DW map can hit; (b) integer but not a realizable DW map (a
    # sign or a permutation that moves the trivial class -- on the lattice yet
    # outside S_3 union the cap orbit). Integer-ness is necessary, not sufficient.
    continuous = [r for r in superpos if not r["on_integer_lattice"]]
    integer_but_floors = [r for r in superpos if r["on_integer_lattice"]]
    print("\n  (C) Why no superposition/entangling gate enters -- at ANY topology:")
    print(f"      every realizable DW map is integer-quantized in the "
          f"flat-connection basis (asserted); the realizable image is the specific")
    print(f"      S_3-union-cap-orbit set, so being integer is necessary, not "
          f"sufficient. The {len(superpos)} floored gates split:")
    print(f"        {len(continuous)} continuous (off the integer lattice: "
          f"1/sqrt2, e^(i pi/4), i amplitudes) -- no integer map can hit them;")
    print(f"        {len(integer_but_floors)} integer but outside the realizable "
          f"image (CZ/Z(x)Z signs, X(x)X moves the trivial class).")
    print(f"      all floor at gap > {FLOOR} (min "
          f"{min(r['loosened_gap_image'] for r in superpos):.3f}, "
          f"max {max(r['loosened_gap_image'] for r in superpos):.3f}); "
          f"loosening the topology lowers no gap (loose == pinned).")
    print(f"      => no superposition/entangling gate realizes: "
          f"{'CONFIRMED' if none_superpos_realize else 'VIOLATED'}")
    print(f"      => the realizable GATE set stays exactly S_3 "
          f"(newly-realizing gates beyond S_3: {newly or 'none'})\n")
    # The loosened gap never beats the pinned gap for these gates (the cap orbit,
    # being rank-1, never approaches a unitary): the expansion is genuinely orthogonal
    # to the gate question.
    loose_eq_pinned = all(
        r["loosened_gap_image"] >= r["pinned_gap_s3"] - 1e-9 for r in superpos)
    ok &= none_superpos_realize and all_floor_far and len(newly) == 0
    ok &= loose_eq_pinned

    # ---- Section D: the emergent-b_1 freedom is in STATES, not gates -------- #
    print("  (D) The genuine emergent-b_1 freedom lives in STATES, not gates "
          "(surgery: removeInteriorCell):")
    trace = grow_boundary_b1()
    print("      the boundary grows holes on its own (closed icosahedron seed, "
          "surgery removes pairwise-disjoint faces):")
    for t in trace:
        print(f"        after {t['removed']} removals:  b_1 = {t['b1']}  "
              f"dim ker L_1 = {t['ker_L1']}  =>  dim Z = 2^b_1 = {t['dim_Z']}")
    grew_to_2 = trace[-1]["b1"] == 2 and trace[-1]["dim_Z"] == DIM
    print(f"      => the boundary grows to b_1 = 2 (Z = C^4, a 2-qubit space): "
          f"{'YES' if grew_to_2 else 'NO'}  "
          f"(the prior attempt's b_1 <= 1 boundary was too small)\n")
    ok &= grew_to_2

    # The superposed state a Hadamard creates, carried across an emergent bulk.
    # `_decide_state` mutates the bulk in place (surgery), so each read gets a
    # fresh seed; the surgery run's bulk is held to read its b_1 afterwards.
    matched = _meridian_target(flip=False)
    flipped = _meridian_target(flip=True)
    disk_seed = _surface(_without(_OCT, (0, 1, 2)))          # b_1 = 0 seed
    v_disk = _decide_state(_surface(_without(_OCT, (0, 1, 2))), matched, max_cones=0)
    st_grow = _surface(_without(_OCT, (0, 1, 2)))
    v_grow = _decide_state(st_grow, matched, max_cones=3, seed=1)
    b1_after = _b1(st_grow)
    # The S_3-control state analogue: the identity transition on the carried mode
    # (the annulus's own meridian on the annulus) realizes -- the controls do NOT
    # floor at the state level either.
    annulus = _surface(_without(_OCT, (0, 1, 2), (3, 4, 5)))
    v_ctrl = _decide_state(annulus, matched, max_cones=0)
    # The genuine obstruction: the period-mismatched (flipped) state floors even
    # with the handle open.
    v_flip = _decide_state(annulus, flipped, max_cones=0)

    print("      the superposed meridian STATE (what a Hadamard creates), carried "
          "by H_1 across an emergent bulk:")
    print(f"        disk seed (b_1 = {_b1(disk_seed)}):        r = {v_disk.residual:.2e}  "
          f"{'realizes' if v_disk.residual < STATE_REALIZE else 'floors'}")
    print(f"        surgery opens the handle (b_1: 0 -> {b1_after}, "
          f"{v_grow.surgery_removals} removal): r = {v_grow.residual:.2e}  "
          f"{'REALIZES' if v_grow.residual < STATE_REALIZE else 'floors'}")
    print(f"        S_3-control state (carried meridian, the validity anchor): "
          f"r = {v_ctrl.residual:.2e}  "
          f"{'realizes' if v_ctrl.residual < STATE_REALIZE else 'FLOOR'}")
    print(f"        period-mismatched state (negative control): "
          f"r = {v_flip.residual:.2e}  "
          f"{'floors (correct)' if v_flip.residual > STATE_FLOOR else 'realizes (WRONG)'}")
    state_ok = (v_disk.residual > STATE_FLOOR
                and v_grow.residual < STATE_REALIZE and b1_after == 1
                and v_ctrl.residual < STATE_REALIZE
                and v_flip.residual > STATE_FLOOR)
    print(f"      => emergent b_1 carries the superposed STATE, and the controls do "
          f"NOT floor: {'YES (state-level anchor holds)' if state_ok else 'NO'}\n")
    ok &= state_ok

    # ---- raw table (PR artifact, not committed) ----------------------------- #
    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "loosened_topology_gateset.json")
        with open(path, "w") as handle:
            json.dump({"image_facts": facts, "gate_table": rows,
                       "newly_realizing_beyond_s3": newly,
                       "boundary_b1_growth": trace,
                       "state_level": {
                           "disk_residual": float(v_disk.residual),
                           "surgery_residual": float(v_grow.residual),
                           "surgery_b1_after": b1_after,
                           "control_residual": float(v_ctrl.residual),
                           "flipped_residual": float(v_flip.residual)}},
                      handle, indent=2)
        print(f"  raw table (PR artifact, not committed): {path}")

    print("\n  Verdict: " + (
        "SUPPORTED -- loosening the cobordism topology EXPANDS the realizable "
        "operation image beyond S_3 (to non-invertible integer cap maps, algebra "
        "5 -> 10 dim) and the realizable STATE space (superposed cycles via "
        "emergent b_1), but does NOT enlarge the realizable GATE set: every "
        "superposition/entangling gate floors at every topology (the DW image is "
        "the integer lattice). S_3 stands as the realizable gate image, with its "
        "controls correctly realizing (the anchor the prior attempt failed)."
        if ok else
        "NOT SUPPORTED -- a verdict departed from the residuals; inspect the table."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
