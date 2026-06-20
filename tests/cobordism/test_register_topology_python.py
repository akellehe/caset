# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The #353-style RegisterTopology and the selectable-topology seam (#378).

`MergeCobordism` chooses its cobordism topology behind a `TopologyBuilder`. The
default is now the color `RegisterTopology` (the #353 register), selectable
against the `(T^2-3holes)xS^1` `TorusOperatorTopology`.

`RegisterTopology` extrudes the holed icosahedron (S²−3 color holes, b₁=2 on the
Σ=0 hyperplane) over a 3-layer staircase into one connected 3-complex with
**b₁(W)=2** — one shared color register across the three blocks (#379). That b₁=2
is the #353 confinement (a Σ≠0 config cannot be carried). It is a valid manifold
(every triangle in ≤2 tets, dualComplexValid) — not the welded shared-block
construction. These tests pin the topology + the selectable seam with a
topology-specific state dimension (register: d=3; operator: a power of two); the
color realizability map / S₃ invariance / emergent result are in
`test_register_merge_353_python.py` and `test_proton_realizability_python.py`.
"""

import cmath
import unittest

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)

# Color (d=3) states: neutral-pair inputs (plain Sigma=0) + a singlet result.
_A = [1.0, -1.0, 0.0]
_B = [1.0, 0.0, -1.0]
_R = [1.0, _W, _W * _W]
# Qubit (d=2) state, for the operator topology / the dimension guard.
_Q = [1.0, 0.0]


def _top_cells(st, k):
    return [tuple(int(v) for v in c)
            for c in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(k)]


def _max_facet_coface(top_cells):
    """The largest number of top cells sharing any codim-1 facet. A manifold has
    <= 2 everywhere; a weld (hand-identified blocks) shows > 2."""
    counts = {}
    for c in top_cells:
        vs = sorted(c)
        for drop in range(len(vs)):
            facet = tuple(v for i, v in enumerate(vs) if i != drop)
            counts[facet] = counts.get(facet, 0) + 1
    return max(counts.values()) if counts else 0


# The register relaxation is light (2-D, signature-blind); a tiny iteration
# budget keeps the suite fast. The topology invariants are seed/budget-independent.
_M = cob.MergeCobordism([_A, _B], [_R], max_iters=2, seed=0)


class RegisterIsDefaultTest(unittest.TestCase):
    def test_default_topology_is_register(self):
        # No topology= arg -> the #353-style register.
        self.assertIn("register", _M.stats.topology)


class RegisterIsAValidManifoldTest(unittest.TestCase):
    def test_no_triangle_in_more_than_two_tets(self):
        # The weld signature is a (codim-1) triangle in > 2 tets. The continuous
        # staircase is a clean 3-manifold (each triangle in 1 or 2 tets).
        tets = _top_cells(_M.cobordism, 3)
        self.assertGreater(len(tets), 0)
        self.assertLessEqual(_max_facet_coface(tets), 2)

    def test_dual_complex_gate_passes(self):
        ok, why = cob.EigenstateSynthesis(_M.cobordism, 1).dualComplexValid()
        self.assertTrue(ok, why)

    def test_connected_b1_is_two(self):
        # One connected complex (b0=1) with b1=2: the single shared color register
        # (the #353 confinement), NOT the b1=8 of independent per-block holes.
        betti = list(_M.stats.betti_cobordism)
        self.assertEqual(betti[0], 1)
        self.assertEqual(betti[1], 2)

    def test_boundary_is_nonempty(self):
        self.assertGreater(len(_M.boundary), 0)


class RegisterCarriesTheColorStatesTest(unittest.TestCase):
    def test_state_pinning_residual_is_finite(self):
        # The relaxation pins the color states over the hole-circles; r_psi is a
        # finite, real residual (a couple of iterations already reduce it).
        self.assertGreaterEqual(_M.stats.state_residual, 0.0)
        self.assertEqual(_M.stats.residual,
                         _M.stats.stat_action_residual + _M.stats.state_residual)


class TopologySelectionTest(unittest.TestCase):
    """The topology is user-selectable, with a topology-specific state dim."""

    def test_operator_topology_is_selectable(self):
        topo = cob.TorusOperatorTopology()
        self.assertIn("operator", topo.name())
        self.assertEqual(topo.carried_dim(2), 3)      # d^2 - 1 for a qubit
        topo.validate_state_dim(2)                     # power of two -> ok
        with self.assertRaises((ValueError, RuntimeError)):
            topo.validate_state_dim(3)                 # not a power of two

    def test_register_topology_dim_is_three(self):
        topo = cob.RegisterTopology()
        self.assertIn("register", topo.name())
        topo.validate_state_dim(3)                     # color triple -> ok
        with self.assertRaises((ValueError, RuntimeError)):
            topo.validate_state_dim(2)

    def test_register_default_rejects_non_color_dimension(self):
        # d=2 with the default (register) topology is rejected at construction.
        with self.assertRaises((ValueError, RuntimeError)):
            cob.MergeCobordism([_Q, _Q], [_Q], max_iters=1)


if __name__ == "__main__":
    unittest.main()
