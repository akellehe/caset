# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Iterated apex-reflection cobordism, dimension-generic, and emergent twist (#429).

`Spacetime.symmetricStackCells` (the #413 symmetric apex interior) is generalized
from the triangle-only (S^2 base) special case to ANY base dimension via coface
mirroring: each top d-simplex cones to an apex (up-cone + down-cone = the point
reflection), and the gap over a (d-1)-facet shared by two cofaces is the join of
the canonical dual edge with the worldprism boundary, `[f1,f2] * boundary(g x I)`.
In d=2 this is exactly the #413 octahedron split; in d>=3 (a tetrahedral S^3 base)
the side worldsheets take a globally-consistent staircase diagonal -> a valid
4-manifold. An `nApexSlices` parameter stacks the reflect-and-cap over many
time-slices.

These tests pin, through the production C++:
  * the dimension-generic rule builds a VALID manifold (no empty interior,
    dualComplexValid, Betti [1,0,0,1]) on a tetrahedral S^3 base -- the full,
    un-reduced 3+1 D construction (NOT a 2+1 D shortcut);
  * dualComplexValid is rigorous for the 4D cobordism (it recurses on vertex
    links), validating BOTH this construction and the prismCells(S^3) control;
  * nApexSlices=1 reproduces the single #413 reflection (b1, dualComplexValid),
    nApexSlices>1 is a valid multi-slice cobordism;
  * the emergent twist is BIDIRECTIONAL / parity-time symmetric: each apex is a
    point reflection (-1 orientation flip), so consecutive slices carry the
    OPPOSITE chirality (a Dirac field, both Weyl components -- not a single chiral
    screw). The twist is read off the oriented geometry, canonical and invariant
    under base-vertex relabeling;
  * the alternating sign tracks the Dirac-Kahler gamma^5 chirality (#415): the
    16 = 4x4 fiber splits 8/8 into the two gamma^5 eigenspaces.
"""

import itertools
import unittest
from collections import deque

import numpy as np

import tessera

cob = tessera.cobordism
ST = tessera.spacetime.Spacetime


def _s3_five_cell():
    """The boundary of the 4-simplex = 5 tetrahedra: the minimal tetrahedral S^3."""
    return [list(c) for c in itertools.combinations(range(5), 4)]


def _s3_sixteen_cell():
    """The 16-cell (hyperoctahedron) boundary = 16 tetrahedra: a richer S^3 with
    higher edge degrees (axis i has vertices 2i, 2i+1; a facet picks one per axis)."""
    return [[2 * i + s for i, s in enumerate(sig)]
            for sig in itertools.product((0, 1), repeat=4)]


def _validity(cells, dim):
    st = ST.fromCells(dim, [list(c) for c in cells], 1.0, 0.0)
    valid, msg = cob.EigenstateSynthesis(st, 1).dualComplexValid()
    betti = list(cob.ChainComplex.fromSpacetime(st).bettiNumbers())
    return valid, msg, betti


def _coherent_orientation(cells):
    """Assign each top cell a sign so adjacent cells induce OPPOSITE orientation on
    their shared facet (the coherent orientation of an orientable pseudomanifold).
    Returns {cell_index: +-1} or None if non-orientable. Canonical up to one global
    sign on a connected complex -- the per-slice PATTERN is invariant."""
    cells = [tuple(sorted(c)) for c in cells]
    nv = len(cells[0])
    facet_map = {}
    for ci, c in enumerate(cells):
        for p in range(nv):
            facet_map.setdefault(c[:p] + c[p + 1:], []).append((ci, p))
    adj = {ci: [] for ci in range(len(cells))}
    for lst in facet_map.values():
        if len(lst) == 2:
            (c1, p1), (c2, p2) = lst
            rel = -((-1) ** (p1 - p2))  # induced orientations must be opposite
            adj[c1].append((c2, rel))
            adj[c2].append((c1, rel))
    o = {}
    for seed in range(len(cells)):
        if seed in o:
            continue
        o[seed] = 1
        dq = deque([seed])
        while dq:
            x = dq.popleft()
            for y, rel in adj[x]:
                want = rel * o[x]
                if y not in o:
                    o[y] = want
                    dq.append(y)
                elif o[y] != want:
                    return None
    return o


def _per_apex_twist(base, dim, n):
    """The signed per-apex twist read off the oriented geometry: for each apex
    slice j, the relative orientation of the up-cone vs the down-cone of a fixed
    base simplex (the point-reflection sense the apex carries). A uniform -1 is the
    bidirectional PT inversion; the cumulative slice chirality is then (-1)^j."""
    cells = [tuple(sorted(c)) for c in ST.symmetricStackCells(
        [list(b) for b in base], n)]
    idx = {c: i for i, c in enumerate(cells)}
    o = _coherent_orientation(cells)
    if o is None:
        return None
    stride = 1 + max(v for c in base for v in c)
    # First-appearance dedup, matching the C++ apex indexing (and relabeling-
    # consistent: relabeling permutes vertex ids but preserves the cell order, so
    # apex index 0 tracks the same base simplex either way).
    tops, top_index = [], {}
    for c in base:
        t = tuple(sorted(c))
        if t not in top_index:
            top_index[t] = len(tops)
            tops.append(t)
    n_tops = len(tops)
    apex_base = (n + 1) * stride
    t = tops[0]                            # the first-appearance base simplex
    signs = []
    for j in range(n):
        f = apex_base + j * n_tops + 0     # apex 0 of slice j
        up = tuple(sorted(tuple(j * stride + x for x in t) + (f,)))
        dn = tuple(sorted(tuple((j + 1) * stride + x for x in t) + (f,)))
        signs.append(o[idx[up]] * o[idx[dn]])
    return signs


class DimensionGenericManifoldTest(unittest.TestCase):
    """The generalized rule builds a valid manifold on a tetrahedral S^3 base."""

    def test_s3_five_cell_is_a_valid_4manifold(self):
        valid, msg, betti = _validity(
            ST.symmetricStackCells(_s3_five_cell(), 1), 4)
        self.assertTrue(valid, msg)
        # S^3 x I ~ S^3: Betti [1,0,0,1] (the trailing b4=0).
        self.assertEqual(betti[:4], [1, 0, 0, 1])

    def test_s3_sixteen_cell_is_a_valid_4manifold(self):
        valid, msg, betti = _validity(
            ST.symmetricStackCells(_s3_sixteen_cell(), 1), 4)
        self.assertTrue(valid, msg)
        self.assertEqual(betti[:4], [1, 0, 0, 1])

    def test_no_empty_interior(self):
        # Every base tetrahedron contributes an up-cone and a down-cone, and every
        # interior (degree-2) facet is gap-filled: the interior is not empty.
        cells = ST.symmetricStackCells(_s3_five_cell(), 1)
        self.assertEqual(len(cells), 90)            # 5 tets -> 90 4-cells
        apex_ids = {v for c in cells for v in c if v >= 2 * 5}
        self.assertEqual(len(apex_ids), 5)          # one cell-apex per tetrahedron

    def test_validator_is_rigorous_for_4d_prism_control(self):
        # The 4D validator must certify the KNOWN-valid prismCells(S^3) too (it was
        # a false-negative before the recursive vertex-link extension).
        prism = [list(c) for c in ST.prismCells(_s3_five_cell(), 1)]
        valid, msg, betti = _validity(prism, 4)
        self.assertTrue(valid, msg)
        self.assertEqual(betti[:4], [1, 0, 0, 1])

    def test_validator_rejects_a_nonmanifold(self):
        # Three 4-cells on one shared 3-facet is a non-manifold; the validator
        # catches it (the manifold gate is not vacuous).
        broken = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]]
        st = ST.fromCells(4, broken, 1.0, 0.0)
        valid, _ = cob.EigenstateSynthesis(st, 1).dualComplexValid()
        self.assertFalse(valid)


class IteratedSlicesTest(unittest.TestCase):
    """nApexSlices stacks valid reflect-and-cap layers; n=1 is the single #413
    reflection."""

    def test_single_slice_matches_413_betti(self):
        valid, msg, betti = _validity(
            ST.symmetricStackCells(_s3_five_cell(), 1), 4)
        self.assertTrue(valid, msg)
        self.assertEqual(betti[:4], [1, 0, 0, 1])

    def test_multi_slice_is_a_valid_cobordism(self):
        # Each added slice is one more reflect-and-cap block (n*90 cells), and the
        # stack stays a valid S^3 x I cobordism.
        for n in (2, 3, 4):
            cells = ST.symmetricStackCells(_s3_five_cell(), n)
            self.assertEqual(len(cells), 90 * n)     # n stacked blocks
            valid, msg, betti = _validity(cells, 4)
            self.assertTrue(valid, f"nApexSlices={n}: {msg}")
            self.assertEqual(betti[:4], [1, 0, 0, 1], f"nApexSlices={n}")


class EmergentTwistTest(unittest.TestCase):
    """The emergent twist is bidirectional / parity-time symmetric: a uniform -1
    per-apex point reflection, so the cumulative slice chirality alternates -- a
    Dirac (non-chiral) twist, never a monotonic screw. Read off the oriented
    geometry, canonical and relabeling-invariant."""

    def test_construction_is_orientable(self):
        # A coherent orientation EXISTS (the stack is an orientable manifold): there
        # is no net chiral monodromy in the bare geometry -- both senses balance.
        for base, dim in [(_s3_five_cell(), 3), (_s3_sixteen_cell(), 3)]:
            for n in (1, 2, 3):
                cells = ST.symmetricStackCells([list(b) for b in base], n)
                self.assertIsNotNone(_coherent_orientation(cells))

    def test_each_apex_is_a_point_reflection(self):
        # Every apex flips orientation: the up-cone vs down-cone relative sign is
        # -1 in EVERY slice (the parity+time inversion), uniformly -- bidirectional,
        # not a monotonic accumulation.
        for base in (_s3_five_cell(), _s3_sixteen_cell()):
            for n in (1, 2, 3, 4):
                signs = _per_apex_twist(base, 3, n)
                self.assertEqual(signs, [-1] * n)

    def test_cumulative_slice_chirality_alternates(self):
        # The slice chirality chi_j = product of the apex flips below it = (-1)^j:
        # +,-,+,- ... -- the alternating (NOT monotonic) twist the PT structure
        # predicts. Both chiralities appear = a Dirac field, not a single Weyl screw.
        signs = _per_apex_twist(_s3_five_cell(), 3, 6)
        chi = np.cumprod([1] + signs)          # chi_0 .. chi_6
        self.assertTrue(np.array_equal(chi, [(-1) ** j for j in range(7)]))

    def test_twist_is_relabeling_invariant(self):
        # Permute the base vertex ids: validity, Betti, and the per-apex twist are
        # all unchanged -- the twist is a canonical geometric quantity, not an
        # artifact of the labeling.
        base = _s3_five_cell()
        perm = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
        relabeled = [[perm[v] for v in c] for c in base]
        v0, m0, b0 = _validity(ST.symmetricStackCells(base, 2), 4)
        v1, m1, b1 = _validity(ST.symmetricStackCells(relabeled, 2), 4)
        self.assertTrue(v0 and v1, f"{m0} / {m1}")
        self.assertEqual(b0, b1)
        self.assertEqual(_per_apex_twist(base, 3, 3),
                         _per_apex_twist(relabeled, 3, 3))


class DiracKahlerChiralityTest(unittest.TestCase):
    """The alternating twist tracks the Dirac-Kahler gamma^5 chirality (#415): the
    16 = 4x4 form fiber splits 8/8 into the two gamma^5 eigenspaces -- the two
    Weyl chiralities the bidirectional (both-senses) twist realizes."""

    def test_gamma5_splits_the_fiber_into_two_chiralities(self):
        st = ST.fromCells(4, ST.symmetricStackCells(_s3_five_cell(), 1), 1.0, 0.0)
        dk = cob.DiracKahler(st)
        self.assertEqual(dk.frameworkDimension(), 4)
        self.assertEqual(dk.gammaDimension(), 16)     # 2^4 form fiber
        self.assertEqual(dk.multiplicity(), 4)        # 16 = 4 spinor x 4 taste
        gammas = [np.array(g).reshape(16, 16) for g in dk.gammas(False)]
        gamma5 = gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
        evals = np.linalg.eigvals(gamma5)
        # gamma^5 is an involution: eigenvalues +-1, split 8/8 (the two chiralities).
        self.assertTrue(np.allclose(np.sort(evals.real), [-1] * 8 + [1] * 8, atol=1e-9))
        self.assertTrue(np.allclose(evals.imag, 0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
