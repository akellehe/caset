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

"""§4b cone-and-retry synthesis loop → geo(ψ) (#134).

BoundaryStateSynthesis grows the simplest complex realizing a target qubit as a
k=0 Hodge-Laplacian eigenvector, building on EigenstateSynthesis (#133, the
fixed-complex residual/optimizer core, reused unmodified) and the pre-geometric
vertex insertion (#112). The qubit is embedded as ψ = (c0, c1, 0, ..., 0): the
two smallest-id (logical) vertices carry the amplitudes, the rest are
zero-amplitude auxiliaries (§4b.1).

Acceptance (numpy oracle; no scipy — the multi-restart optimizer is in C++):
  * a general-amplitude qubit (|c0| != |c1|) cannot be realized on the two-vertex
    seed — its residual floors at w_min²(|c0|²-|c1|²)² > 0 (#133) — but IS
    synthesized (r < ε) after coning in one auxiliary vertex; the accepted
    complex (the triangle K_3) is minimal: no smaller complex converges;
  * the realized ψ is genuinely an eigenvector of the synthesized Laplacian, with
    eigenvalue equal to the reported Rayleigh quotient λ;
  * coning preserves the homotopy type: the Betti numbers (b0 = 1, all higher 0,
    i.e. contractible) are unchanged across the loop;
  * the spec seed Δ⁴ (K_5) already supplies enough freedom, so a general qubit is
    realized there with zero cones.
"""

import math
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders (shared idiom with the Hodge / #133 tests)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _edge_seed():
    """The §4b.2 object: two vertices, one edge (K_2)."""
    return _from_simplices(2, [(0, 1)])


def _delta4_seed():
    """The §4b.4 seed: a single 4-simplex Δ⁴ (5 vertices, 1-skeleton K_5)."""
    return _from_simplices(5, [(0, 1, 2, 3, 4)])


def _cvec(v):
    return [complex(z) for z in v]


def _np_L(st):
    """Independent D - A reference (magnitude convention), sorted-vertex-id
    order — the oracle the Hodge / #133 tests use."""
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n), dtype=complex)
    D = np.zeros(n)
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        if s == t:
            continue
        i, j = idx[s], idx[t]
        w = e.getSquaredLength()
        z = w * np.exp(1j * e.getPhase())
        A[i, j] += z
        A[j, i] += np.conj(z)
        D[i] += abs(w)
        D[j] += abs(w)
    return np.diag(D).astype(complex) - A


# A general-amplitude qubit: |c0|^2 = 0.8, |c1|^2 = 0.2  (|c0| != |c1|).
A_AMP, B_AMP = math.sqrt(0.8), math.sqrt(0.2)
W_MIN = 0.1  # the search-box floor (matches the C++ optimizer)


# --------------------------------------------------------------------------- #
class StructureTest(unittest.TestCase):
    def test_seed_sizes(self):
        bss = cob.BoundaryStateSynthesis(_edge_seed())
        self.assertEqual(bss.num_vertices(), 2)
        self.assertEqual(bss.num_edges(), 1)

        bss4 = cob.BoundaryStateSynthesis(_delta4_seed())
        self.assertEqual(bss4.num_vertices(), 5)
        self.assertEqual(bss4.num_edges(), 10)

    def test_too_small_seed_raises(self):
        # One vertex cannot carry a two-amplitude qubit.
        st = _from_simplices(1, [])
        with self.assertRaises(Exception):
            cob.BoundaryStateSynthesis(st)


class TwoVertexFloorTest(unittest.TestCase):
    """§4b.2: a general-amplitude qubit cannot be a two-vertex eigenvector —
    the residual floors bounded away from 0 (the motivation for coning)."""

    def test_general_qubit_floors(self):
        bss = cob.BoundaryStateSynthesis(_edge_seed())
        r = bss.optimize(complex(A_AMP), complex(B_AMP), restarts=48, seed=1)
        # Closed-form box minimum: w_min^2 (|c0|^2 - |c1|^2)^2.
        d = A_AMP ** 2 - B_AMP ** 2
        floor = W_MIN ** 2 * d ** 2
        self.assertGreater(r, 1e-3)                       # bounded away from 0
        self.assertAlmostEqual(r, floor, delta=1.5e-3)   # ~ the closed form

    def test_balanced_qubit_is_realizable_on_two_vertices(self):
        bss = cob.BoundaryStateSynthesis(_edge_seed())
        amp = 1.0 / math.sqrt(2.0)
        r = bss.optimize(complex(amp), complex(amp), restarts=48, seed=2)
        self.assertLess(r, 1e-9)


class ConeAndRetryTest(unittest.TestCase):
    """The central acceptance: a general qubit is synthesized after coning, and
    the accepted complex is minimal."""

    def test_general_qubit_synthesized_after_coning(self):
        bss = cob.BoundaryStateSynthesis(_edge_seed())
        geo = bss.synthesize(complex(A_AMP), complex(B_AMP), epsilon=1e-9,
                             restarts=80, max_cones=4, seed=3)

        # Synthesized (r < ε) — only after coning in one auxiliary vertex.
        self.assertTrue(geo.converged)
        self.assertLess(geo.residual, 1e-9)
        self.assertEqual(geo.cones_applied, 1)

        # Minimal complex: the triangle K_3 (one auxiliary vertex). |V|=3, |E|=3.
        self.assertEqual(geo.num_vertices, 3)
        self.assertEqual(geo.num_edges, 3)

    def test_accepted_complex_is_minimal(self):
        # The accepted complex has 3 vertices; the only smaller complex (the
        # 2-vertex seed) floors — so 3 is minimal.
        bss = cob.BoundaryStateSynthesis(_edge_seed())
        geo = bss.synthesize(complex(A_AMP), complex(B_AMP), epsilon=1e-9,
                             restarts=80, max_cones=4, seed=5)
        self.assertEqual(geo.num_vertices, 3)

        smaller = cob.BoundaryStateSynthesis(_edge_seed())
        r2 = smaller.optimize(complex(A_AMP), complex(B_AMP), restarts=64, seed=5)
        self.assertGreater(r2, 1e-3)             # 2 vertices cannot converge
        self.assertLess(geo.residual, 1e-6 * r2)  # 3 vertices does

    def test_realized_state_is_an_eigenvector(self):
        seed = _edge_seed()
        bss = cob.BoundaryStateSynthesis(seed)
        geo = bss.synthesize(complex(A_AMP), complex(B_AMP), epsilon=1e-9,
                             restarts=80, max_cones=4, seed=7)
        self.assertTrue(geo.converged)

        # seed is mutated in place into the synthesized K_3; build its Laplacian.
        L = _np_L(seed)
        ids = sorted(v.getId() for v in seed.getVertexList().toVector())
        self.assertEqual(len(ids), 3)
        psi = np.zeros(3, dtype=complex)
        psi[0] = A_AMP   # logical vertex 0 (smallest id)
        psi[1] = B_AMP   # logical vertex 1
        psi /= np.linalg.norm(psi)

        # L psi || psi, with eigenvalue == the reported Rayleigh quotient λ.
        Lpsi = L @ psi
        lam = np.vdot(psi, Lpsi).real
        self.assertAlmostEqual(lam, geo.eigenvalue, places=6)
        self.assertTrue(np.allclose(Lpsi, lam * psi, atol=1e-4))

        # Independent eigh cross-check: psi matches one Laplacian eigenvector.
        evals, evecs = np.linalg.eigh(L)
        overlaps = [abs(np.vdot(evecs[:, k], psi)) for k in range(3)]
        k = int(np.argmax(overlaps))
        self.assertAlmostEqual(overlaps[k], 1.0, places=4)
        self.assertAlmostEqual(evals[k], geo.eigenvalue, places=5)


class Delta4SeedTest(unittest.TestCase):
    """The spec seed Δ⁴ (K_5) already has enough combinatorial freedom: a
    general qubit is realized there with zero cones."""

    def test_delta4_realizes_general_qubit_without_coning(self):
        bss = cob.BoundaryStateSynthesis(_delta4_seed())
        geo = bss.synthesize(complex(A_AMP), complex(B_AMP), epsilon=1e-9,
                             restarts=64, max_cones=0, seed=11)
        self.assertTrue(geo.converged)
        self.assertLess(geo.residual, 1e-9)
        self.assertEqual(geo.cones_applied, 0)
        self.assertEqual(geo.num_vertices, 5)
        self.assertEqual(geo.num_edges, 10)


class HomotopyPreservedTest(unittest.TestCase):
    """Coning preserves the homotopy type: Betti numbers unchanged (contractible
    throughout — b0 = 1, all higher 0)."""

    @staticmethod
    def _betti(st):
        return cob.ChainComplex.fromSpacetime(st).bettiNumbers()

    def test_betti_unchanged_across_coning(self):
        seed = _edge_seed()
        bss = cob.BoundaryStateSynthesis(seed)

        b = self._betti(seed)               # edge: contractible
        self.assertEqual(b[0], 1)
        self.assertEqual(sum(b[1:]), 0)

        for expected_v in (3, 4, 5):
            self.assertTrue(bss.cone_in_vertex())
            self.assertEqual(bss.num_vertices(), expected_v)
            b = self._betti(seed)
            self.assertEqual(b[0], 1)            # still connected
            self.assertEqual(sum(b[1:]), 0)      # still contractible

    def test_betti_unchanged_during_synthesis(self):
        seed = _edge_seed()
        b_before = self._betti(seed)
        bss = cob.BoundaryStateSynthesis(seed)
        bss.synthesize(complex(A_AMP), complex(B_AMP), epsilon=1e-9,
                       restarts=80, max_cones=4, seed=13)
        b_after = self._betti(seed)
        # Homotopy type preserved: contractible before and after.
        self.assertEqual(b_before[0], 1)
        self.assertEqual(sum(b_before[1:]), 0)
        self.assertEqual(b_after[0], 1)
        self.assertEqual(sum(b_after[1:]), 0)


if __name__ == "__main__":
    unittest.main()
