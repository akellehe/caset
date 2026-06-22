"""Regge-mediated synthesis search (#248).

The realizability search scored by the mediated objective
``F_β(W) = r_U(W) + β·|S_Regge(W*)|``: the §4b realizability residual on the
primal Laplacian plus β times the magnitude of the dual Lorentzian Regge action
(``ReggeSolver.dualReggeAction``, #247) on the candidate's circumcentric dual.

What's covered here (all deterministic — the surgery move ``removeInteriorCell``
is deterministic; the additive Pachner cone ``growInterior`` is not, so these
tests never depend on a cone firing):

  * β=0 reproduces the base-layer verdict **bit-for-bit** (the |S| term is not
    even computed) — both that the default call and an explicit ``beta=0`` agree,
    and that the realizability decision/residual are unchanged.
  * ``regge_action`` (the realized |S_Regge(W*)|) is reported at every β, finite
    and ≥ 0, and varies with the geometry.
  * the ``maxVertices`` volume bound caps additive growth.
  * the boundary ∂W is byte-identical across the whole search.

Not covered here — deferred to the #249 gate-battery experiment: the
**selection** effect, β>0 choosing a *lower-|S_Regge|* filling among competing
realizing surgeries. It needs **inequivalent competing realizing fillings**: a
minimal hand fixture realizes through a single best surgery (the one that opens
the needed b_k), which also reduces |S|, so β reinforces rather than redirects it
and the outcome is β-inert. The diverse 13-gate battery supplies the competing
fillings; the machinery that selects among them (the F_β candidate scoring) is
exercised here through the β=0 byte-identical guarantee and the |S| reporting.

Fixtures reuse the octahedron-surface (a triangulated S²) idiom from
test_emergent_bulk_python.py: deleting a face opens a disk whose single interior
top cell the surgery search removes.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

try:
    import tessera
    cob = tessera.cobordism
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

# The octahedron surface (a triangulated S²), built generically from a face list.
_OCT = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
        (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
_CYCLE_A, _CYCLE_B = [(0, 1), (0, 2), (1, 2)], [(3, 4), (3, 5), (4, 5)]


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


def _disk(weight=1.0):
    """Octahedron minus face (0,1,2): a disk (b_1=0) with exactly one interior
    top cell, the opposite face (3,4,5)."""
    return _surface(_delete((0, 1, 2)), weight=weight)


def _annulus():
    return _surface(_delete((0, 1, 2), (3, 4, 5)))


def _bipyramid():
    """Two triangles 012, 013 sharing the interior edge 01; the four outer edges
    are ∂W. No interior top cell (every vertex is on ∂W), so the only way to grow
    is the additive cone — which the maxVertices cap can block deterministically."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(2))
    st.build()
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v3 = st.createVertex(3)
    st.createSimplex([v[0], v[1], v3])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _meridian_target():
    """The meridian 1-form read off the annulus's own harmonic, carried on both
    boundary circles — realizable on the disk only after the interior face is
    drilled out (b_1: 0→1)."""
    h = cob.HodgeLaplacian(_annulus()).harmonics(1)[0]
    edges = _CYCLE_A + _CYCLE_B
    vals = [complex(h.amplitudeFor(list(e))) for e in edges]
    return cob.Cochain(1, edges, np.asarray(vals, dtype=complex))


def _boundary_edge_geometry(st):
    """Snapshot every edge's (squared-length, phase) keyed by sorted endpoint
    ids — the ∂W byte-fixed check reads the boundary subset from this."""
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = (e.getSquaredLength().real, e.getPhase())
    return out


_SURG = lambda: cob.RealizabilityOracle.GrowthMode.SURGERY


def _decide_disk(beta=0.0, max_cones=3, max_vertices=16, explicit_beta=True,
                 weight=1.0):
    st = _disk(weight=weight)
    target = _meridian_target()
    kw = dict(epsilon=1e-9, restarts=64, max_cones=max_cones, seed=1,
              growth_mode=_SURG(), harmonic=True, max_vertices=max_vertices)
    if explicit_beta:
        kw["beta"] = beta
    return st, cob.RealizabilityOracle(st).decideHarmonic(target, **kw)


# --------------------------------------------------------------------------- #
@pytest.mark.slow
class Beta0ReproducesBaseLayer(unittest.TestCase):
    def test_explicit_beta0_equals_default(self):
        # The β=0 mediated path is the base-layer search bit-for-bit: residual,
        # witness state, and reported |S| are byte-identical to the default call.
        _, v_def = _decide_disk(explicit_beta=False)
        _, v_b0 = _decide_disk(beta=0.0)
        self.assertEqual(v_def.residual, v_b0.residual)
        self.assertEqual(v_def.regge_action, v_b0.regge_action)
        self.assertEqual(list(v_def.state), list(v_b0.state))
        self.assertEqual(v_def.surgery_removals, v_b0.surgery_removals)

    def test_beta0_decision_unchanged_by_beta(self):
        # The realizability decision is the primal r_U<ε regardless of β; β only
        # selects among fillings. The single-surgery disk commits its one removal
        # at every β, so the residual is β-independent here.
        _, v0 = _decide_disk(beta=0.0)
        _, v5 = _decide_disk(beta=5.0)
        self.assertEqual(v0.surgery_removals, 1)
        self.assertEqual(v0.residual, v5.residual)        # one option ⇒ β inert
        self.assertEqual(v0.regge_action, v5.regge_action)


@pytest.mark.slow
class ReggeActionReported(unittest.TestCase):
    def test_regge_action_finite_nonnegative(self):
        _, v = _decide_disk(beta=0.0)
        self.assertTrue(np.isfinite(v.regge_action))
        self.assertGreaterEqual(v.regge_action, 0.0)
        self.assertGreater(v.regge_action, 0.0)  # the drilled disk has hinges

    def test_regge_action_varies_with_geometry(self):
        # |S_Regge(W*)| reads the edge squared-lengths, so two different seed
        # geometries give different reported actions.
        _, v1 = _decide_disk(beta=0.0, weight=1.0)
        _, v2 = _decide_disk(beta=0.0, weight=2.0)
        self.assertNotAlmostEqual(v1.regge_action, v2.regge_action, places=6)


class MaxVerticesBound(unittest.TestCase):
    def test_cap_blocks_additive_growth(self):
        # The 1×3 gate on the bipyramid realizes only by coning in one interior
        # vertex (no interior top cell exists, so surgery can do nothing). With
        # maxVertices set to the seed vertex count the cone is never attempted —
        # the |atVertexCap| guard short-circuits before growInterior, so this is
        # immune to growInterior's nondeterminism — and |W| stays put.
        st = _bipyramid()
        n_seed = st.getVertexList().size()              # 4
        U = [complex(1), complex(0.3, 0.5), complex(-0.8, 0.2)]
        M = cob.RealizabilityOracle.GrowthMode.SURGERY_AND_CONE
        v = cob.RealizabilityOracle(st).decide(
            U, 1, 3, epsilon=1e-10, restarts=32, max_cones=5, seed=1,
            growth_mode=M, beta=0.0, max_vertices=n_seed)
        self.assertEqual(v.interior_vertex_count, 0)     # no vertex coned in
        self.assertEqual(st.getVertexList().size(), n_seed)  # |W| capped
        self.assertFalse(v.realizable)                   # so it floors


class BoundaryHeldFixed(unittest.TestCase):
    def test_boundary_byte_identical_across_search(self):
        st = _disk()
        # ∂W edges: the three sides of the drilled-out face (0,1,2).
        boundary_keys = {(0, 1), (0, 2), (1, 2)}
        before = {k: v for k, v in _boundary_edge_geometry(st).items()
                  if k in boundary_keys}
        target = _meridian_target()
        cob.RealizabilityOracle(st).decideHarmonic(
            target, epsilon=1e-9, restarts=64, max_cones=3, seed=1,
            growth_mode=_SURG(), harmonic=True, beta=2.0, max_vertices=16)
        after = {k: v for k, v in _boundary_edge_geometry(st).items()
                 if k in boundary_keys}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
