# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The genuinely 3+1 D color-register foundation (#453).

Stage 1 of the 4D conversion of the proton experiments: a triangulated ``S^3``
spatial slice carrying symmetric, color-``Z_3``-equivariant windows (the faithful
4D analog of the ``S^2`` ``SymmetricWindowSurface`` / geodesic icosahedron),
stacked over time into a genuine **4-manifold** event. NEVER 2+1 D, no ``S^2``
fallback.

The crux: the color register degree **tracks the spatial dimension** -- it is
``ker L_{d-1}`` with the holes the removed **top d-cells**. On ``S^2`` (``d=2``) a
window hole is a removed TRIANGLE (the ``b_1`` / ``k=1`` register); on ``S^3``
(``d=3``) it is a removed TETRAHEDRON (the ``b_2`` / ``k=2`` register). ``S^3`` is
simply connected (``b_1 = 0``), so the register genuinely lives one degree up,
mirroring the ``S^2`` construction one dimension down. Removing ``n`` disjoint
tetrahedra from ``S^3`` gives ``b_2 = n - 1`` (the ``S^2`` analog: ``n`` disjoint
triangles give ``b_1 = n - 1``).

The slice is the join of two ``K``-cycles (a clean ``Z_3``-symmetric triangulated
3-sphere); each window is a ``Z_3`` orbit of three vertex-disjoint hole
tetrahedra; the holed slice is stacked over time by the dimension-generic
symmetric apex reflection (``Spacetime.symmetricStackCells``, #429) into a
pentatope 4-complex, gated by the rigorous ``n=4`` recursive ``dualComplexValid``.
"""

import cmath
import math
import unittest

import pytest

import tessera

cob = tessera.cobordism
S = tessera.Spacetime

# The color triple's omega phase and the three reps on the color triple.
_W = cmath.exp(2j * math.pi / 3)
_SINGLET = [1.0, 1.0, 1.0]          # the trivial rep -- NOT carried (confinement)
_OMEGA = [1.0, _W, _W * _W]         # the standard-rep color state -- carried
_OMEGABAR = [1.0, _W * _W, _W]      # its conjugate -- carried


def _holed(window_count, granularity=1):
    """The S^3 slice minus its window tetrahedra, and the flat hole list."""
    surf = cob.S3WindowSurface.build(window_count, granularity)
    faces = [list(t) for t in surf.faces]
    windows = [[list(h) for h in w] for w in surf.windows]
    holes = [h for w in windows for h in w]
    holeset = {tuple(h) for h in holes}
    holed = [t for t in faces if tuple(t) not in holeset]
    return faces, windows, holes, holed


def _betti(top_cells, dim):
    st = S.fromCells(dim, [list(c) for c in top_cells], 1.0, 0.0)
    return list(cob.ChainComplex.fromSpacetime(st).bettiNumbers()), st


class S3WindowSurfaceTest(unittest.TestCase):
    """The S^3 slice + Z_3 windows + the b_2 / k=2 color register."""

    def test_slice_is_a_clean_triangulated_three_sphere(self):
        surf = cob.S3WindowSurface.build(1, 1)
        betti, st = _betti(surf.faces, 3)
        self.assertEqual(betti, [1, 0, 0, 1])               # S^3
        self.assertEqual(len(st.getBoundary()), 0)          # closed
        ok, _msg = cob.EigenstateSynthesis(st, 2).dualComplexValid()
        self.assertTrue(ok)
        # the join of two K-cycles: K = 6, 2K = 12 vertices, K^2 = 36 tetrahedra
        self.assertEqual(len(surf.faces), 36)

    def test_windows_are_z3_orbits_of_disjoint_hole_tetrahedra(self):
        wc = 4
        surf = cob.S3WindowSurface.build(wc, 1)
        windows = [[tuple(h) for h in w] for w in surf.windows]
        self.assertEqual(len(windows), wc)
        # every hole is a TETRAHEDRON (4 vertices) -- the d=3 register's removed
        # top cell, not a triangle.
        for w in windows:
            self.assertEqual(len(w), 3)
            for h in w:
                self.assertEqual(len(h), 4)
        # all 3*wc holes are vertex-disjoint (mirrors the S^2 12-disjoint-holes).
        verts = [v for w in windows for h in w for v in h]
        self.assertEqual(len(verts), len(set(verts)))
        # the color Z_3 sigma = tau^{K/3} cyclically permutes each window's holes.
        K = 6 * wc
        step = K // 3

        def sigma(v):
            return (v + step) % K if v < K else K + ((v - K + step) % K)

        for w in windows:
            rotated = {tuple(sorted(sigma(v) for v in h)) for h in w}
            self.assertEqual(rotated, set(w))               # sigma preserves the window
            # and it is a genuine 3-cycle, not the identity
            self.assertNotEqual(tuple(sorted(sigma(v) for v in w[0])), w[0])

    def test_removing_windows_opens_the_b2_register(self):
        # b_2 = 3*windowCount - 1, the S^3 lift of the S^2 proton's b_1 = 11.
        for wc, expect_b2 in [(1, 2), (2, 5), (4, 11)]:
            _f, _w, _h, holed = _holed(wc)
            betti, st = _betti(holed, 3)
            self.assertEqual(betti[2], expect_b2, f"windowCount={wc}")
            self.assertEqual(betti[1], 0)                   # no spurious b_1
            ok, _msg = cob.EigenstateSynthesis(st, 2).dualComplexValid()
            self.assertTrue(ok, f"windowCount={wc} not a valid manifold")

    def test_color_register_carries_omega_not_singlet(self):
        # The crux: pin a color state on one window's three tetrahedral holes and
        # read its period at k=2. The omega (standard-rep) states are carried
        # (residual -> machine zero); the singlet is NOT (confinement / Sum=0).
        _f, windows, _h, holed = _holed(1)
        st = S.fromCells(3, [list(t) for t in holed], 1.0, 0.0)
        es = cob.EigenstateSynthesis(st, 2)
        holes = [list(h) for h in windows[0]]
        r_omega = es.residualForPeriods(holes, [complex(x) for x in _OMEGA])
        r_obar = es.residualForPeriods(holes, [complex(x) for x in _OMEGABAR])
        r_singlet = es.residualForPeriods(holes, [complex(x) for x in _SINGLET])
        self.assertLess(r_omega, 1e-12)
        self.assertLess(r_obar, 1e-12)
        self.assertGreater(r_singlet, 1.0)                  # confinement

    def test_deterministic(self):
        a = cob.S3WindowSurface.build(2, 1)
        b = cob.S3WindowSurface.build(2, 1)
        self.assertEqual([list(t) for t in a.faces], [list(t) for t in b.faces])
        self.assertEqual([[list(h) for h in w] for w in a.windows],
                         [[list(h) for h in w] for w in b.windows])


_EVENT_CACHE = {}

# The minimal genuinely-4D event: one Z_3 color window (b_2 = 2), stacked over two
# temporal apex slices. The like-resolution four-window event (b_2 = 11) is the
# 10^3-10^4x cost the #418 spike budgeted for; the b_2 = 11 headline is verified on
# the (cheap) holed S^3 3-complex in S3WindowSurfaceTest instead.
_EVENT_WINDOWS = 1
_EVENT_B2 = 3 * _EVENT_WINDOWS - 1


def _s3_event(n_layers=2):
    """Build (once) the S^3 EmergentEventTopology event cobordism, gated on the
    rigorous n=4 recursive dualComplexValid inside build()."""
    if n_layers not in _EVENT_CACHE:
        topo = cob.EmergentEventTopology()
        topo.set_s3_slice(True)
        topo.set_s3_windows(_EVENT_WINDOWS)
        topo.set_layers(n_layers)
        W, boundary = topo.build_cobordism(state_dim=3, seed=0)
        _EVENT_CACHE[n_layers] = (topo, W, boundary)
    return _EVENT_CACHE[n_layers]


class S3FourManifoldEventTest(unittest.TestCase):
    """The EmergentEventTopology S^3 path: a genuine 4-manifold carrying the register."""

    def test_register_degree_is_two(self):
        topo = cob.EmergentEventTopology()
        topo.set_s3_slice(True)
        self.assertTrue(topo.s3_slice())
        self.assertEqual(topo.register_degree(), 2)

    def test_s2_path_unchanged(self):
        # the default (S^2) path is untouched: register degree 1, a 3-complex.
        s2 = cob.EmergentEventTopology()
        self.assertFalse(s2.s3_slice())
        self.assertEqual(s2.register_degree(), 1)

    @pytest.mark.slow
    def test_event_is_a_genuine_four_manifold(self):
        _topo, W, _bd = _s3_event()
        # dimension 4 (pentatope top cells), not a 2+1 D thing in disguise.
        self.assertEqual(cob.CombinatorialDimension().compute(W), 4.0)
        cc = cob.ChainComplex.fromSpacetime(W)
        fvec = list(cc.fVector())
        self.assertEqual(len(fvec), 5)                      # V,E,tri,tet,pentatope
        # the b_2 color register survives the temporal stack.
        self.assertEqual(list(cc.bettiNumbers())[2], _EVENT_B2)
        # the rigorous n=4 recursive manifold gate (vertex links are 3-manifolds).
        ok, _msg = cob.EigenstateSynthesis(W, 2).dualComplexValid()
        self.assertTrue(ok)

    @pytest.mark.slow
    def test_carries_register_through_bilateral_pinning(self):
        # Read the windows at the BOTTOM (ell=0) and TOP (ell=nLayers) temporal
        # slices -- the bilaterally pinned endpoints -- at k=2: the omega color
        # state is carried at both ends, the singlet is not.
        topo, W, _bd = _s3_event()
        es = cob.EigenstateSynthesis(W, 2)
        n = topo.n_layers()
        for layer in (0, n):
            holes = [list(h) for h in topo.window_holes_at_layer(0, layer)]
            self.assertEqual(len(holes), 3)
            self.assertTrue(all(len(h) == 4 for h in holes))   # tetrahedral holes
            r_omega = es.residualForPeriods(holes, [complex(x) for x in _OMEGA])
            r_singlet = es.residualForPeriods(holes, [complex(x) for x in _SINGLET])
            self.assertLess(r_omega, 1e-9, f"omega not carried at layer {layer}")
            self.assertGreater(r_singlet, 1.0, f"no confinement at layer {layer}")


if __name__ == "__main__":
    unittest.main()
