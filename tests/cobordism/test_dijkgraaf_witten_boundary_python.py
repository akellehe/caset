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

"""Dijkgraaf–Witten ℤ₂ state sum with boundary (#109) — Z(Σ_B) → Z(Σ_A), T1.

For a 3-manifold W with boundary the state sum is a vector in Z(∂W): holding the
boundary connection g|∂W fixed and summing the closed-case tetrahedron product
∏_t ω(g_01,g_12,g_23) over the interior gauge classes [g] ∈ H¹(W;ℤ₂) gives, for
each boundary flat-connection class, a complex amplitude. With ∂W = Σ_A ⊔ Σ_B
this reads as a map Z(Σ_B) → Z(Σ_A); each closed surface Σ carries the DW
Hilbert space Z(Σ) = ℂ[H¹(Σ;ℤ₂)] of dimension 2^{b₁(Σ)}.

Acceptance (spec §5.4, plan ticket 7):

* **T1 — cylinder is the identity.** For W = Σ×[0,T] (built as the simplicial
  product Σ × interval), Z(W) = id on Z(Σ); hence ⟨ψ_A|Z(W)|ψ_B⟩ = ⟨ψ_A|ψ_B⟩.
  The boundary states ψ are prepared from the harmonic 1-forms ker L₁(Σ) — the
  qubit of dimension b₁ = 2 for the torus, kept distinct from the 2^{b₁} = 4
  flat-connection count that indexes the state sum (spec §5.2).

* **numpy oracle.** On a small bounded fixture (the solid torus S¹×D², whose
  boundary is a single T²) an independent numpy reimplementation of the boundary
  sum reproduces the C++ amplitude vector.
"""

import itertools
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    # S¹ = ∂Δ², the minimal triangle-boundary circle (3 vertices).
    return tessera.SimplexBoundarySphere(1)


def _torus_topology():
    # T² = S¹ × S¹ via the simplicial (Eilenberg–Zilber) product.
    return tessera.SimplicialProduct(_circle(), _circle())


def _interval():
    return tessera.SolidSimplex(1)  # a single edge = [0, 1]


def _torus_cylinder():
    # W = T² × [0,T], the trivial cobordism T² → T² (∂W = T² ⊔ T²).
    return _build(tessera.SimplicialProduct(_torus_topology(), _interval()))


def _sphere_cylinder():
    # W = S² × [0,T], trivial cobordism S² → S² (Z(S²) is one-dimensional).
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            _interval()))


def _solid_torus():
    # W = S¹ × D² (solid torus); ∂W = S¹ × S¹ = T² (one boundary component).
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


def _three_sphere():
    return _build(tessera.SimplexBoundarySphere(3))  # closed: no boundary


# --------------------------------------------------------------------------- #
# Pure-numpy GF(2) linear algebra (independent of the C++ routines).
# --------------------------------------------------------------------------- #
def _gf2_nullspace(matrix, cols):
    """Basis of {x : matrix·x ≡ 0 (mod 2)} as length-`cols` rows, via RREF."""
    if matrix.size == 0:
        return [np.eye(cols, dtype=np.int64)[i] for i in range(cols)]
    a = (np.asarray(matrix, dtype=np.int64) & 1).copy()
    rows, _ = a.shape
    pivots, r = [], 0
    for col in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if a[i, col] & 1), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and (a[i, col] & 1):
                a[i] ^= a[r]
        pivots.append(col)
        r += 1
    is_pivot = [c in pivots for c in range(cols)]
    basis = []
    for free in range(cols):
        if is_pivot[free]:
            continue
        x = np.zeros(cols, dtype=np.int64)
        x[free] = 1
        for t, pc in enumerate(pivots):
            x[pc] = a[t, free] & 1
        basis.append(x)
    return basis


def _gf2_span(basis, cols):
    """All 2^k combinations of `basis` (each a length-`cols` 0/1 vector)."""
    k = len(basis)
    out = []
    for mask in range(1 << k):
        x = np.zeros(cols, dtype=np.int64)
        for b in range(k):
            if (mask >> b) & 1:
                x ^= basis[b]
        out.append(x)
    return out


def _gf2_independent_mod(vector, span_rows):
    """Reduce `vector` by the echelon `span_rows`; return the residue (0/1)."""
    v = (np.asarray(vector, dtype=np.int64) & 1).copy()
    for row in span_rows:
        piv = int(np.argmax(row)) if row.any() else -1
        if piv >= 0 and (v[piv] & 1):
            v ^= row
    return v & 1


def _gf2_echelon(generators, cols):
    """Row-echelon basis (sorted by pivot) of the span of `generators`."""
    rows = []
    for gen in generators:
        v = (np.asarray(gen, dtype=np.int64) & 1).copy()
        v = _gf2_independent_mod(v, rows)
        if v.any():
            rows.append(v)
            rows.sort(key=lambda row: int(np.argmax(row)))
    return rows


def _cohomology_reps(cocycles, coboundaries, cols):
    """Representatives of H¹ = Z¹/B¹: cocycles independent modulo B¹."""
    span_rows = _gf2_echelon(coboundaries, cols)
    reps = []
    for z in cocycles:
        if _gf2_independent_mod(z, span_rows).any():
            reps.append(np.asarray(z, dtype=np.int64) & 1)
            span_rows = _gf2_echelon(
                [r for r in span_rows] + [np.asarray(z, dtype=np.int64) & 1], cols)
    return reps


def _omega(kind, a, b, c):
    if kind == "trivial":
        return 1
    if kind == "sign":
        return -1 if (a & b & c) else 1
    raise ValueError(kind)


def _boundary_oracle(spacetime, kind):
    """Independent numpy recomputation of the boundary amplitude multiset.

    Returns the sorted list of amplitudes over the joint boundary classes — the
    indexing-convention-free fingerprint of the Z(∂W) vector, recomputed through
    a separate path (numpy GF(2) cohomology, holonomy binning, ω product).
    """
    chain = cobordism.ChainComplex.fromSpacetime(spacetime)
    num_edges = chain.numSimplices(1)
    num_triangles = chain.numSimplices(2)

    boundary2 = (np.asarray(chain.boundaryMatrix(2), dtype=np.int64)
                 .reshape(num_edges, num_triangles)) & 1     # edges × triangles
    coboundary1 = boundary2.T                                # triangles × edges
    z1 = _gf2_nullspace(coboundary1, num_edges)

    edges = [tuple(e) for e in chain.kSimplexVertices(1)]
    edge_index = {e: i for i, e in enumerate(edges)}
    vertex_ids = [int(v[0]) for v in chain.kSimplexVertices(0)]
    coboundary_basis = [
        np.array([1 if vid in edge else 0 for edge in edges], dtype=np.int64)
        for vid in vertex_ids]
    bulk_reps = _cohomology_reps(z1, coboundary_basis, num_edges)
    bulk_classes = _gf2_span(bulk_reps, num_edges)

    btris = [tuple(t) for t in cobordism.Cobordism.boundaryFaces(spacetime)]
    raw_components = cobordism.Cobordism.connectedComponents(
        [list(t) for t in btris])
    components = sorted(
        [sorted(tuple(t) for t in comp) for comp in raw_components])

    # Per component, an independent class index via holonomy around an H₁ basis.
    component_indexers = []
    for comp in components:
        comp_edges = sorted({pair for tri in comp
                             for pair in ((tri[0], tri[1]), (tri[0], tri[2]),
                                          (tri[1], tri[2]))})
        local_index = {e: i for i, e in enumerate(comp_edges)}
        n_local = len(comp_edges)
        # 1-boundary ∂₁ (vertices × local edges) for cycles, ∂₂ for filling.
        comp_verts = sorted({v for tri in comp for v in tri})
        vert_index = {v: i for i, v in enumerate(comp_verts)}
        d1 = np.zeros((len(comp_verts), n_local), dtype=np.int64)
        for (u, w) in comp_edges:
            e = local_index[(u, w)]
            d1[vert_index[u], e] ^= 1
            d1[vert_index[w], e] ^= 1
        cycles = _gf2_nullspace(d1, n_local)            # Z₁(Σ) = ker ∂₁
        d2 = np.zeros((n_local, len(comp)), dtype=np.int64)
        for j, tri in enumerate(comp):
            for pair in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
                d2[local_index[pair], j] ^= 1
        boundaries = [d2[:, j] for j in range(len(comp))]  # B₁(Σ) = im ∂₂
        h1_cycles = _cohomology_reps(cycles, boundaries, n_local)  # H₁ basis
        component_indexers.append((comp_edges, h1_cycles))

    # Bin every bulk class by its boundary holonomy signature; sum ω product.
    amplitudes = {}
    tets = [tuple(t) for t in chain.orientedTopSimplices()]
    for g in bulk_classes:
        signature = []
        for comp_edges, h1_cycles in component_indexers:
            local = np.array([g[edge_index[e]] for e in comp_edges],
                             dtype=np.int64)
            signature.extend(int(np.dot(local, cyc) & 1) for cyc in h1_cycles)
        weight = 1
        for tet in tets:
            g01 = int(g[edge_index[(tet[0], tet[1])]])
            g12 = int(g[edge_index[(tet[1], tet[2])]])
            g23 = int(g[edge_index[(tet[2], tet[3])]])
            weight *= _omega(kind, g01, g12, g23)
        key = tuple(signature)
        amplitudes[key] = amplitudes.get(key, 0) + weight

    total_classes = 1
    for _, h1_cycles in component_indexers:
        total_classes *= (1 << len(h1_cycles))
    values = list(amplitudes.values()) + [0] * (total_classes - len(amplitudes))
    return sorted(float(v) for v in values)


# --------------------------------------------------------------------------- #
# T1 — the cylinder is the identity.
# --------------------------------------------------------------------------- #
class TestCylinderIsIdentity(unittest.TestCase):

    def test_torus_cylinder_map_is_identity(self):
        matrix = np.asarray(DijkgraafWitten(_torus_cylinder(),
                                            Cocycle.Trivial).map())
        # Z(T²) has dimension 2^{b₁(T²)} = 4.
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-9)

    def test_sphere_cylinder_map_is_identity(self):
        # Z(S²) is one-dimensional, so Z(S²×I) is the 1×1 identity [[1]].
        matrix = np.asarray(DijkgraafWitten(_sphere_cylinder(),
                                            Cocycle.Trivial).map())
        self.assertEqual(matrix.shape, (1, 1))
        np.testing.assert_allclose(matrix, np.eye(1), atol=1e-9)

    def test_sign_cocycle_cylinder_is_also_identity(self):
        # The cup cube vanishes on the cylinder, so the Sign twist agrees.
        matrix = np.asarray(DijkgraafWitten(_torus_cylinder(),
                                            Cocycle.Sign).map())
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-9)

    def test_boundary_dimensions_are_two_qubits_worth(self):
        dims = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial).boundaryDimensions()
        self.assertEqual(list(dims), [4, 4])  # 2^{b₁(T²)} per boundary T²


# --------------------------------------------------------------------------- #
# T1 — ⟨ψ_A|Z(W)|ψ_B⟩ = ⟨ψ_A|ψ_B⟩, with states from the harmonic 1-forms.
# --------------------------------------------------------------------------- #
class TestCylinderReproducesInnerProduct(unittest.TestCase):

    def test_harmonic_one_forms_are_a_qubit(self):
        # ker L₁(T²) has dimension b₁ = 2 — the qubit. This is the continuous
        # harmonic count, deliberately distinct from the 2^{b₁} = 4 flat ℤ₂
        # connections that index Z(T²) (spec §5.2).
        torus = _build(_torus_topology())
        harmonics = cobordism.HodgeLaplacian(torus).harmonics(1)
        num_edges = cobordism.ChainComplex.fromSpacetime(torus).numSimplices(1)
        self.assertEqual(len(harmonics), 2)  # b₁ = 2 harmonic 1-forms (Cochains)
        self.assertTrue(all(h.degree() == 1 and h.size() == num_edges
                            for h in harmonics))

    def _boundary_states_from_harmonics(self):
        """Two Z(T²) states (length 2^{b₁}=4) seeded by the ker L₁ harmonics."""
        torus = _build(_torus_topology())
        harmonics = cobordism.HodgeLaplacian(torus).harmonics(1, 1e-9, True)
        # columns = the harmonic 1-form Cochains over the edge ordering.
        cols = (np.column_stack([np.asarray(h.coeffs()) for h in harmonics])
                if harmonics else np.zeros((0, 0), dtype=complex))
        # Map the two real harmonic 1-forms to two distinct unit vectors in the
        # 4-dim flat-connection Hilbert space Z(T²) (the precise embedding is
        # immaterial for T1: Z(W)=id makes the amplitude the inner product).
        psi_a = np.zeros(4, dtype=complex)
        psi_b = np.zeros(4, dtype=complex)
        seed_a = cols[:, 0]
        seed_b = cols[:, 1] if cols.shape[1] > 1 else cols[:, 0]
        for i in range(4):
            psi_a[i] = complex(seed_a[i % seed_a.size], seed_b[(i + 1) % seed_b.size])
            psi_b[i] = complex(seed_b[i % seed_b.size], -seed_a[(i + 2) % seed_a.size])
        psi_a /= np.linalg.norm(psi_a)
        psi_b /= np.linalg.norm(psi_b)
        return psi_a, psi_b

    def test_amplitude_equals_inner_product(self):
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        psi_a, psi_b = self._boundary_states_from_harmonics()
        amplitude = dw.amplitude(list(psi_a), list(psi_b))
        self.assertAlmostEqual(amplitude, np.vdot(psi_a, psi_b), places=9)

    def test_amplitude_equals_inner_product_random_states(self):
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        rng = np.random.default_rng(109)
        for _ in range(5):
            psi_a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi_b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            amplitude = dw.amplitude(list(psi_a), list(psi_b))
            self.assertAlmostEqual(amplitude, np.vdot(psi_a, psi_b), places=9)

    def test_diagonal_amplitudes_are_norms(self):
        # ⟨ψ|Z(W)|ψ⟩ = ‖ψ‖² for the identity cobordism.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        for basis in range(4):
            psi = [1.0 if i == basis else 0.0 for i in range(4)]
            self.assertAlmostEqual(dw.amplitude(psi, psi), 1.0, places=9)


# --------------------------------------------------------------------------- #
# numpy oracle for the boundary sum on a small bounded fixture.
# --------------------------------------------------------------------------- #
class TestBoundaryAgainstOracle(unittest.TestCase):

    def _check(self, spacetime):
        for cocycle, kind in ((Cocycle.Trivial, "trivial"), (Cocycle.Sign, "sign")):
            with self.subTest(cocycle=kind):
                vector = np.asarray(
                    DijkgraafWitten(spacetime, cocycle).boundaryVector())
                np.testing.assert_allclose(vector.imag, 0.0, atol=1e-9)
                got = sorted(float(x) for x in vector.real)
                self.assertEqual([round(x, 9) for x in got],
                                 [round(x, 9) for x in _boundary_oracle(spacetime, kind)])

    def test_solid_torus(self):
        self._check(_solid_torus())

    def test_solid_torus_pinned_values(self):
        # S¹×D² has H¹=ℤ₂ (2 bulk classes); the restriction to ∂=T² is injective,
        # so 2 of the 4 boundary classes get amplitude 1 and the rest 0.
        vector = np.asarray(
            DijkgraafWitten(_solid_torus(), Cocycle.Trivial).boundaryVector())
        self.assertEqual(vector.size, 4)
        self.assertAlmostEqual(vector.real.sum(), 2.0, places=9)
        self.assertEqual(int(np.count_nonzero(np.round(vector.real, 6))), 2)
        self.assertEqual(sorted(np.round(vector.real, 6)), [0.0, 0.0, 1.0, 1.0])

    def test_sphere_cylinder_oracle(self):
        # A two-component boundary fixture for the oracle as well (S² ⊔ S²).
        self._check(_sphere_cylinder())


# --------------------------------------------------------------------------- #
# Guards.
# --------------------------------------------------------------------------- #
class TestBoundaryGuards(unittest.TestCase):

    def test_closed_manifold_has_no_boundary_vector(self):
        with self.assertRaises(RuntimeError):
            DijkgraafWitten(_three_sphere(), Cocycle.Trivial).boundaryVector()

    def test_map_requires_two_components(self):
        # The solid torus has a single boundary component, so map() (which needs
        # Σ_A and Σ_B) must refuse it; boundaryVector() still works.
        dw = DijkgraafWitten(_solid_torus(), Cocycle.Trivial)
        self.assertEqual(len(dw.boundaryVector()), 4)
        with self.assertRaises(RuntimeError):
            dw.map()

    def test_amplitude_rejects_wrong_length(self):
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        with self.assertRaises((ValueError, RuntimeError)):
            dw.amplitude([1.0, 0.0], [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
