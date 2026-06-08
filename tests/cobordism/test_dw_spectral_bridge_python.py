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

"""The DW-spectral bridge: three independent readings of a cobordism value (#177,
the v0.4 bridge capstone of #174).

On a shared boundary surface Sigma = T^2 the cobordism value at a pair of prepared
boundary states (psi_A, psi_B) is computed three independent ways and tested for
agreement:

1. **Topological** Z_DW = DijkgraafWitten.amplitude(prepare(psi_A), prepare(psi_B))
   on the torus cylinder W = T^2 x [0,T] (Z(W) = id_4), the Z_2 state sum
   sandwiched between the prepared boundary states -- **metric-free**.
2. **Operation** <psi_A|U|psi_B> via ChoiJamiolkowski.transitionAmplitude.
3. **Spectral** Z_spec, the Hodge harmonic overlap (Cochain.innerProduct) of the
   two boundary states' coordinates along the longitude, the k=1 boundary harmonic
   the solid torus D^2 x S^1 carries (certified by RealizabilityOracle.decide
   Harmonic).

Each Z is cross-checked against an independent numpy computation (the DW state sum
via the GF(2)/holonomy path; the amplitude via vdot; the spectral value via the
ker L_1 Hodge projection). The honest finding: the Z_2 DW maps are a discrete,
integer-quantized family, so the bridge Z_DW = <psi_A|U|psi_B> holds only on the
DW-representable subset (U = id here) and **fails** for a generic U -- the
agree/disagree dichotomy. The spectral oracle (a continuum: the longitude is
realizable, only the meridian is obstructed) strictly extends the topological
theory.
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism
cj = tessera.quantum.ChoiJamiolkowski


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())


def _torus_cylinder():
    return _build(tessera.SimplicialProduct(_torus_topology(),
                                            tessera.SolidSimplex(1)))


def _solid_torus():
    return _build(tessera.SimplicialProduct(tessera.SolidSimplex(2), _circle()))


def _pin_uniform(st, w=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _pinned_solid_torus():
    W = _solid_torus()
    _pin_uniform(W)
    return W


def _cochain(simplices, coeffs):
    return cob.Cochain(1, simplices, np.asarray(coeffs, dtype=complex))


def _scaled(form, scalar):
    return _cochain(form.simplices(), scalar * np.asarray(form.coeffs()))


def _unit(form):
    c = np.asarray(form.coeffs())
    return _cochain(form.simplices(), c / np.linalg.norm(c))


def _longitude_and_meridian(W, space):
    """The longitude (carried by W, spectrally realizable) and the meridian (bounds
    a disk in W, spectrally obstructed), as unit degree-1 Cochains over Sigma."""
    sig_simpl = space.harmonics()[0].simplices()
    bulk_h = cob.HodgeLaplacian(W).harmonics(1)[0]
    restriction = _cochain(
        sig_simpl, [complex(bulk_h.amplitudeFor(list(e))) for e in sig_simpl])
    prepared = space.prepare(restriction)
    longitude = _unit(prepared.readout())
    coords = np.array([complex(prepared.generatorAmplitude(i)) for i in range(2)])
    coords = coords / np.linalg.norm(coords)
    harmonics = np.column_stack([np.asarray(h.coeffs()) for h in space.harmonics()])
    meridian = _unit(_cochain(sig_simpl,
                              harmonics @ np.array([coords[1], -coords[0]])))
    return longitude, meridian


def _embed_on_cells(form, cells):
    idx = {tuple(c): i for i, c in enumerate(cells)}
    out = np.zeros(len(cells), dtype=complex)
    for c, s in zip(np.asarray(form.coeffs()), form.simplices()):
        out[idx[tuple(s)]] = c
    return out


# --------------------------------------------------------------------------- #
# Independent numpy GF(2) Dijkgraaf-Witten state sum (the holonomy path).
# --------------------------------------------------------------------------- #
def _gf2_nullspace(matrix, cols):
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
    out = []
    for mask in range(1 << len(basis)):
        x = np.zeros(cols, dtype=np.int64)
        for b in range(len(basis)):
            if (mask >> b) & 1:
                x ^= basis[b]
        out.append(x)
    return out


def _gf2_independent_mod(vector, span_rows):
    v = (np.asarray(vector, dtype=np.int64) & 1).copy()
    for row in span_rows:
        piv = int(np.argmax(row)) if row.any() else -1
        if piv >= 0 and (v[piv] & 1):
            v ^= row
    return v & 1


def _gf2_echelon(generators, cols):
    rows = []
    for gen in generators:
        v = _gf2_independent_mod((np.asarray(gen, dtype=np.int64) & 1).copy(), rows)
        if v.any():
            rows.append(v)
            rows.sort(key=lambda row: int(np.argmax(row)))
    return rows


def _cohomology_reps(cocycles, coboundaries, cols):
    span_rows = _gf2_echelon(coboundaries, cols)
    reps = []
    for z in cocycles:
        if _gf2_independent_mod(z, span_rows).any():
            reps.append(np.asarray(z, dtype=np.int64) & 1)
            span_rows = _gf2_echelon(
                list(span_rows) + [np.asarray(z, dtype=np.int64) & 1], cols)
    return reps


def _omega(kind, a, b, c):
    return (-1 if (a & b & c) else 1) if kind == "sign" else 1


def _numpy_dw_boundary(spacetime, kind):
    """Independent numpy recomputation of the sorted Z(dW) amplitude multiset."""
    chain = cob.ChainComplex.fromSpacetime(spacetime)
    num_edges = chain.numSimplices(1)
    num_triangles = chain.numSimplices(2)
    boundary2 = (np.asarray(chain.boundaryMatrix(2), dtype=np.int64)
                 .reshape(num_edges, num_triangles)) & 1
    z1 = _gf2_nullspace(boundary2.T, num_edges)
    edges = [tuple(e) for e in chain.kSimplexVertices(1)]
    edge_index = {e: i for i, e in enumerate(edges)}
    vertex_ids = [int(v[0]) for v in chain.kSimplexVertices(0)]
    coboundary_basis = [
        np.array([1 if vid in edge else 0 for edge in edges], dtype=np.int64)
        for vid in vertex_ids]
    bulk_classes = _gf2_span(_cohomology_reps(z1, coboundary_basis, num_edges),
                             num_edges)
    btris = [tuple(t) for t in cob.Cobordism.boundaryFaces(spacetime)]
    components = sorted(
        [sorted(tuple(t) for t in comp)
         for comp in cob.Cobordism.connectedComponents([list(t) for t in btris])])
    component_indexers = []
    for comp in components:
        comp_edges = sorted({pair for tri in comp
                             for pair in ((tri[0], tri[1]), (tri[0], tri[2]),
                                          (tri[1], tri[2]))})
        local_index = {e: i for i, e in enumerate(comp_edges)}
        comp_verts = sorted({v for tri in comp for v in tri})
        vert_index = {v: i for i, v in enumerate(comp_verts)}
        d1 = np.zeros((len(comp_verts), len(comp_edges)), dtype=np.int64)
        for (u, w) in comp_edges:
            e = local_index[(u, w)]
            d1[vert_index[u], e] ^= 1
            d1[vert_index[w], e] ^= 1
        cycles = _gf2_nullspace(d1, len(comp_edges))
        d2 = np.zeros((len(comp_edges), len(comp)), dtype=np.int64)
        for j, tri in enumerate(comp):
            for pair in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
                d2[local_index[pair], j] ^= 1
        boundaries = [d2[:, j] for j in range(len(comp))]
        component_indexers.append(
            (comp_edges, _cohomology_reps(cycles, boundaries, len(comp_edges))))
    amplitudes = {}
    tets = [tuple(t) for t in chain.orientedTopSimplices()]
    for g in bulk_classes:
        signature = []
        for comp_edges, h1_cycles in component_indexers:
            local = np.array([g[edge_index[e]] for e in comp_edges], dtype=np.int64)
            signature.extend(int(np.dot(local, cyc) & 1) for cyc in h1_cycles)
        weight = 1
        for tet in tets:
            weight *= _omega(kind, int(g[edge_index[(tet[0], tet[1])]]),
                             int(g[edge_index[(tet[1], tet[2])]]),
                             int(g[edge_index[(tet[2], tet[3])]]))
        key = tuple(signature)
        amplitudes[key] = amplitudes.get(key, 0) + weight
    total = 1
    for _, h1 in component_indexers:
        total *= (1 << len(h1))
    values = list(amplitudes.values()) + [0] * (total - len(amplitudes))
    return sorted(float(v) for v in values)


# --------------------------------------------------------------------------- #
# numpy Hodge oracle for L_1.
# --------------------------------------------------------------------------- #
def _numpy_L1(st):
    chain = cob.ChainComplex.fromSpacetime(st)
    nv, ne, nt = (chain.numSimplices(0), chain.numSimplices(1),
                  chain.numSimplices(2))
    d1 = np.asarray(chain.boundaryMatrix(1), float).reshape(nv, ne)
    d2 = np.asarray(chain.boundaryMatrix(2), float).reshape(ne, nt)
    hodge = cob.HodgeLaplacian(st)
    w1 = np.asarray(hodge.weights(1), float)
    w2 = np.asarray(hodge.weights(2), float)
    b1 = d1 * (1.0 / np.sqrt(w1))[None, :]
    b2 = np.sqrt(w1)[:, None] * d2 * (1.0 / np.sqrt(w2))[None, :]
    return b1.T @ b1 + b2 @ b2.T


def _residual_agnostic(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    return float(np.vdot(Lp - lam * psi, Lp - lam * psi).real)


def _flat(M):
    return [complex(z) for z in np.asarray(M, dtype=complex).reshape(-1)]


def _haar_unitary(seed):
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


# --------------------------------------------------------------------------- #
class BridgeReadingsCrossCheckTest(unittest.TestCase):
    """Each of the three Z's reproduces an independent numpy computation."""

    def setUp(self):
        self.space = cob.BoundaryStateSpace(_build(_torus_topology()))
        self.sigma = _build(_torus_topology())
        W = _pinned_solid_torus()
        self.longitude, self.meridian = _longitude_and_meridian(W, self.space)
        self.dw = cob.DijkgraafWitten(_torus_cylinder(), cob.Cocycle.Trivial)

    def _prep(self, scalar):
        return self.space.prepare(_scaled(self.longitude, scalar))

    def test_Z_DW_cross_checks_gf2_holonomy_state_sum(self):
        # The C++ cylinder map equals id_4 and reproduces the independent
        # GF(2)/holonomy boundary state sum (sorted multiset, convention-free).
        cyl = _torus_cylinder()
        m = np.asarray(self.dw.map())
        self.assertEqual(m.shape, (4, 4))
        np.testing.assert_allclose(m, np.eye(4), atol=1e-9)
        self.assertEqual(
            sorted(round(x, 9) for x in m.real.flatten()),
            [round(x, 9) for x in _numpy_dw_boundary(cyl, "trivial")])
        # Z_DW = <psi_A| Z(W) |psi_B> equals the contraction of the prepared coeffs
        # against the (independently validated) map.
        pa, pb = self._prep(np.exp(1j)), self._prep(0.6 + 0.8j)
        z_dw = complex(self.dw.amplitude(pa, pb))
        z_dw_np = complex(np.vdot(np.asarray(pa.coeffs()),
                                  m @ np.asarray(pb.coeffs())))
        self.assertAlmostEqual(z_dw, z_dw_np, places=12)

    def test_operation_amplitude_cross_checks_vdot(self):
        pa, pb = self._prep(np.exp(1j)), self._prep(0.6 + 0.8j)
        for U in (np.eye(4, dtype=complex), _haar_unitary(7)):
            amp = complex(cj.transitionAmplitude(
                _flat(pa.coeffs()), _flat(U), _flat(pb.coeffs()), 4, 4))
            direct = complex(np.vdot(np.asarray(pa.coeffs()),
                                     U @ np.asarray(pb.coeffs())))
            self.assertAlmostEqual(amp, direct, places=12)

    def test_Z_spec_cross_checks_numpy_hodge_projection(self):
        # Z_spec (Cochain inner product of the boundary 1-forms) equals the
        # ker L_1(Sigma) projected overlap from an independent numpy eigendecomp.
        fa, fb = _scaled(self.longitude, np.exp(1j)), _scaled(self.longitude, 1j)
        z_sp = complex(fa.innerProduct(fb))
        L1 = _numpy_L1(self.sigma)
        evals, evecs = np.linalg.eigh(L1)
        kernel = evecs[:, np.abs(evals) < 1e-7]
        a, b = np.asarray(fa.coeffs()), np.asarray(fb.coeffs())
        pa = kernel @ (kernel.conj().T @ a)
        pb = kernel @ (kernel.conj().T @ b)
        # The longitude lies in ker L_1 (the projection is the identity on it).
        np.testing.assert_allclose(pa, a, atol=1e-7)
        self.assertAlmostEqual(z_sp, complex(np.vdot(pa, pb)), places=10)


# --------------------------------------------------------------------------- #
class ThreeWayAgreementTest(unittest.TestCase):
    """Z_DW = <psi_A|U=id|psi_B> = Z_spec on the DW-representable operation."""

    def setUp(self):
        self.space = cob.BoundaryStateSpace(_build(_torus_topology()))
        W = _pinned_solid_torus()
        self.longitude, _ = _longitude_and_meridian(W, self.space)
        self.dw = cob.DijkgraafWitten(_torus_cylinder(), cob.Cocycle.Trivial)

    def test_three_readings_agree(self):
        Uid = _flat(np.eye(4, dtype=complex))
        for a, b in [(1.0 + 0j, 1.0 + 0j),
                     (np.exp(1j * np.pi / 3), 0.6 + 0.8j),
                     (0.5 - 0.5j, 1j),
                     (2.0 + 0j, 0.25 + 0.97j)]:
            with self.subTest(a=a, b=b):
                fa, fb = _scaled(self.longitude, a), _scaled(self.longitude, b)
                pa, pb = self.space.prepare(fa), self.space.prepare(fb)
                z_dw = complex(self.dw.amplitude(pa, pb))
                z_op = complex(cj.transitionAmplitude(
                    _flat(pa.coeffs()), Uid, _flat(pb.coeffs()), 4, 4))
                z_sp = complex(fa.innerProduct(fb))
                # All three independent readings land on the same complex value,
                # which is conj(a) * b for the longitude-aligned states.
                self.assertAlmostEqual(z_dw, z_op, places=10)
                self.assertAlmostEqual(z_dw, z_sp, places=10)
                self.assertAlmostEqual(z_dw, complex(np.conj(a) * b), places=10)


# --------------------------------------------------------------------------- #
class AgreeDisagreeDichotomyTest(unittest.TestCase):
    """The bridge holds on the DW-representable U and FAILS for a generic U."""

    def setUp(self):
        self.space = cob.BoundaryStateSpace(_build(_torus_topology()))
        W = _pinned_solid_torus()
        self.longitude, _ = _longitude_and_meridian(W, self.space)
        self.dw = cob.DijkgraafWitten(_torus_cylinder(), cob.Cocycle.Trivial)
        self.pa = self.space.prepare(_scaled(self.longitude, np.exp(1j * np.pi / 3)))
        self.pb = self.space.prepare(_scaled(self.longitude, 0.6 + 0.8j))
        self.z_dw = complex(self.dw.amplitude(self.pa, self.pb))

    def _amp(self, U):
        return complex(cj.transitionAmplitude(
            _flat(self.pa.coeffs()), _flat(U), _flat(self.pb.coeffs()), 4, 4))

    def test_representable_U_agrees(self):
        # U = id is Z(T^2 x [0,T]) -- in the DW-representable set; the bridge holds.
        self.assertAlmostEqual(self._amp(np.eye(4, dtype=complex)), self.z_dw,
                               places=10)

    def test_generic_U_disagrees(self):
        hadamard = np.eye(4, dtype=complex)
        hadamard[1:3, 1:3] = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        for U in (hadamard, _haar_unitary(7), _haar_unitary(23)):
            with self.subTest(U=U[0, 0]):
                # A generic U is outside the discrete DW image: Z_DW != <A|U|B>.
                self.assertGreater(abs(self._amp(U) - self.z_dw), 1e-2)

    def test_dw_maps_are_integer_quantized(self):
        # The DW-representable maps are a discrete, integer-quantized family.
        triv = np.asarray(self.dw.map())
        sign = np.asarray(cob.DijkgraafWitten(_torus_cylinder(),
                                              cob.Cocycle.Sign).map())
        cap = np.asarray(cob.DijkgraafWitten(_solid_torus(),
                                             cob.Cocycle.Trivial).boundaryVector())
        for M in (triv, sign, cap.reshape(-1)):
            np.testing.assert_allclose(M.real, np.round(M.real), atol=1e-9)
            np.testing.assert_allclose(M.imag, 0.0, atol=1e-9)
        # And a Haar-random U(4) is bounded away from that lattice in the only
        # observable that matters here -- the bridge gap.
        gaps = [abs(self._amp(_haar_unitary(11 + k)) - self.z_dw)
                for k in range(64)]
        self.assertGreater(min(gaps), 1e-2)


# --------------------------------------------------------------------------- #
class SpectralRealizabilityDichotomyTest(unittest.TestCase):
    """The spectral oracle's continuum boundary: the longitude is realizable, the
    meridian is obstructed (cross-checked against an independent numpy floor)."""

    def setUp(self):
        self.space = cob.BoundaryStateSpace(_build(_torus_topology()))

    def test_longitude_is_spectrally_realizable(self):
        W = _pinned_solid_torus()
        longitude, _ = _longitude_and_meridian(W, self.space)
        v = cob.RealizabilityOracle(W).decideHarmonic(
            longitude, epsilon=1e-9, restarts=8, max_cones=0, seed=1)
        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-9)
        self.assertAlmostEqual(v.eigenvalue, 0.0, places=7)
        # The witness carries the target longitude on its boundary block.
        cells = [tuple(c) for c in cob.EigenstateSynthesis(W, 1).cellSimplices()]
        tgt = _embed_on_cells(longitude, cells)
        state = np.asarray(v.state)
        overlap = abs(np.vdot(state / np.linalg.norm(state),
                              tgt / np.linalg.norm(tgt)))
        self.assertAlmostEqual(overlap, 1.0, places=7)

    def test_meridian_is_obstructed_with_numpy_floor(self):
        W = _pinned_solid_torus()
        _, meridian = _longitude_and_meridian(W, self.space)
        v = cob.RealizabilityOracle(W).decideHarmonic(
            meridian, epsilon=1e-9, restarts=8, max_cones=0, seed=0)
        self.assertFalse(v.realizable)
        self.assertGreater(v.floor, 1e-2)
        self.assertEqual(v.floor, v.residual)
        # At the seed there are no interior cells, so the certified floor equals
        # the meridian residual on L_1(W) (an independent numpy Hodge oracle).
        cells = [tuple(c) for c in cob.EigenstateSynthesis(W, 1).cellSimplices()]
        mer_on_W = _embed_on_cells(meridian, cells)
        self.assertAlmostEqual(v.floor, _residual_agnostic(_numpy_L1(W), mer_on_W),
                               delta=1e-6)


if __name__ == "__main__":
    unittest.main()
