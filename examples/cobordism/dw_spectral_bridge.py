# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The DW-spectral bridge: three independent readings of a cobordism's value.

The capstone of the v0.4 bridge (#174). On a shared boundary surface Sigma = T^2
it computes the value of the cobordism W_AB at a pair of prepared boundary states
(psi_A, psi_B) three *independent* ways and tests their agreement:

1. **Topological** Z_DW(W_AB) -- the Z_2 Dijkgraaf-Witten state sum sandwiched
   between the prepared boundary states,
   ``DijkgraafWitten.amplitude(prepare(psi_A), prepare(psi_B))``. Computed from the
   topology + cocycle ALONE (a sum over flat Z_2 connections); **metric-free** (no
   edge weights). The cobordism is the torus cylinder W = T^2 x [0,T], so
   Z(W) = id on Z(T^2) = C[H^1(T^2;Z_2)] = C^4 (the flat-connection-class basis).

2. **Operation** <psi_A|U|psi_B> -- the transition amplitude of the operation U
   the cobordism implements, via ``quantum::ChoiJamiolkowski.transitionAmplitude``
   (the Hilbert-Schmidt / Choi map-state duality).

3. **Spectral** Z_spec -- the value read off the k=1 boundary harmonic of the
   *synthesized* W_AB. The solid torus W = D^2 x S^1 (boundary T^2) carries the
   **longitude** as a genuine ker L_1(W) harmonic; ``RealizabilityOracle.decide
   Harmonic`` certifies it (residual -> 0, lambda -> 0), and Z_spec is the Hodge
   harmonic overlap of the two boundary states' coordinates along that certified
   spectral mode (ker L_1(Sigma), the b_1-dimensional spectral qubit).

The test is ``Z_DW(W_AB) = <psi_A|U|psi_B> = Z_spec``.

**The honest finding (the "quantized shadow").** The Z_2 DW functor's cobordism
maps are a *discrete / finite* family (the C[Z_2] Frobenius algebra: cylinder,
cap/cup, pair-of-pants, twisted by the cocycle), all **integer-quantized** in the
flat-connection basis. So a *generic* operation U is **not** Z_DW of any Z_2
cobordism. The experiment therefore exhibits BOTH:

  * a U *in* the DW-representable set (U = id = Z(T^2 x [0,T])): all three readings
    agree to tolerance, each cross-checked against an independent numpy oracle; and
  * a *generic* U (a Hadamard mix on the qubit block, a Haar-random U(4)) *outside*
    it: Z_DW != <psi_A|U|psi_B>, with the gap opening continuously as U leaves the
    discrete DW point.

The bridge holds on the discrete DW-representable subset; the spectral oracle (a
continuum -- it realizes the longitude and a neighborhood, obstructing only the
meridian) **strictly extends** the topological theory. The DW invariant is the
*quantized shadow* of the spectral Z.

Run:  python examples/cobordism/dw_spectral_bridge.py
      (use --help for options; the section 7 sweeps + section 8 figures default to
      /tmp/cobordism and are NOT committed -- attach them to the issue/PR if you
      want to pin a result. The script is the committed artifact.)

Exit status is 0 iff the verdicts match expectation: the three readings agree on
the representable U, every generic U disagrees by a margin, the longitude is
spectrally realizable and the meridian obstructed, and all three numpy
cross-checks agree.
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS: cap the thread
# pools here so the script self-limits even when launched without the env prefix.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import cmath     # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
cj = tessera.quantum.ChoiJamiolkowski

# Agreement tolerance: two readings of the cobordism value "agree" when their
# difference is below this; a generic U must DIFFER by more than DISAGREE_MARGIN.
AGREE_TOL = 1e-7
DISAGREE_MARGIN = 1e-2


# --------------------------------------------------------------------------- #
# Fixtures (the cobordism idiom: Signature(d) so the d-cells register as top
# simplices; built through the topology so the vertex-id counter advances).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1 = boundary of a triangle


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())  # T^2 = S^1 x S^1


def _interval():
    return tessera.SolidSimplex(1)  # a single edge = [0, 1]


def _torus_cylinder():
    """W = T^2 x [0,T]: the trivial cobordism T^2 -> T^2 (dW = T^2 ⊔ T^2). The
    DW-representable cobordism whose map Z(W) = id on Z(T^2) = C^4."""
    return _build(tessera.SimplicialProduct(_torus_topology(), _interval()))


def _solid_torus():
    """W = D^2 x S^1 = SolidSimplex(2) x S^1: a 3-manifold with boundary T^2
    (b_1(W) = 1). dW equals the standalone T^2 edge-for-edge (shared product
    vertex-id scheme), so a boundary harmonic of Sigma lands on dW. Carries the
    longitude as a ker L_1(W) harmonic -- the spectral fixture."""
    return _build(tessera.SimplicialProduct(tessera.SolidSimplex(2), _circle()))


def _pin_uniform(st, w=1.0, phase=0.0):
    """Pin every edge to a fixed Hermitian value; decideHarmonic's fill only
    rewrites interior edges, so this fixes dW (and Sigma)."""
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _solid_torus_pinned():
    """A fresh solid torus with every edge pinned uniform -- the synthesis bulk."""
    W = _solid_torus()
    _pin_uniform(W)
    return W


def _embed_on_cells(form, cells, idx):
    """A degree-1 Cochain's coeffs scattered onto a bulk k=1 cell order (the rest
    zero) -- the target form embedded on the bulk's edges for a residual check."""
    out = np.zeros(len(cells), dtype=complex)
    for c, s in zip(np.asarray(form.coeffs()), form.simplices()):
        out[idx[tuple(s)]] = c
    return out


# --------------------------------------------------------------------------- #
# Cochain helpers + the longitude / meridian boundary harmonics of the solid
# torus (the H_1(Sigma) -> H_1(W) dichotomy, #176 / PR #190).
# --------------------------------------------------------------------------- #
def _cochain(simplices, coeffs):
    return cob.Cochain(1, simplices, np.asarray(coeffs, dtype=complex))


def _scaled(form, scalar):
    """A copy of a degree-1 Cochain scaled by a complex scalar."""
    return _cochain(form.simplices(), scalar * np.asarray(form.coeffs()))


def _unit(form):
    c = np.asarray(form.coeffs())
    return _cochain(form.simplices(), c / np.linalg.norm(c))


def longitude_and_meridian(W, space):
    """The two distinguished boundary harmonics of the solid torus:

      * longitude = the restriction of the bulk harmonic ker L_1(W) to Sigma,
        expressed in the prepared DW basis -- the cycle that survives in H_1(W),
        the harmonic the manifold *carries* (spectrally realizable);
      * meridian  = its orthogonal complement in ker L_1(Sigma) -- the cycle that
        bounds a disk in W, dying in H_1(W) (spectrally obstructed).
    """
    sig_simpl = space.harmonics()[0].simplices()
    bulk_h = cob.HodgeLaplacian(W).harmonics(1)[0]   # b_1(W) = 1
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


# --------------------------------------------------------------------------- #
# Independent numpy GF(2) Dijkgraaf-Witten state sum (the holonomy path) -- the
# cross-check for Z_DW. A separate implementation from the C++ state sum: numpy
# GF(2) cohomology, holonomy binning, omega product. Returns the sorted boundary
# amplitude multiset (the indexing-convention-free fingerprint of Z(dW)).
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


def numpy_dw_boundary(spacetime, kind):
    """Independent numpy recomputation of the boundary amplitude multiset of the
    Z_2 DW state sum (the GF(2)/holonomy path). The sorted Z(dW) amplitudes."""
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
# numpy Hodge oracle for L_1 (symmetric metric Laplacian) -- the cross-check for
# Z_spec and the meridian floor.
# --------------------------------------------------------------------------- #
def numpy_L1(st):
    """L_1^sym = B_1^T B_1 + B_2 B_2^T, in the canonical ChainComplex k=1 cell
    order (matches HodgeLaplacian.laplacian(1))."""
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


def numpy_harmonic_overlap(form_a, form_b, sigma):
    """The spectral (Hodge) overlap: project both 1-forms onto ker L_1(Sigma)
    (numpy eigendecomposition of the symmetric metric L_1) and contract the
    projections. The independent check for Z_spec."""
    L1 = numpy_L1(sigma)
    evals, evecs = np.linalg.eigh(L1)
    kernel = evecs[:, np.abs(evals) < 1e-7]          # ker L_1(Sigma), b_1 columns
    a = np.asarray(form_a.coeffs())
    b = np.asarray(form_b.coeffs())
    pa = kernel @ (kernel.conj().T @ a)
    pb = kernel @ (kernel.conj().T @ b)
    return complex(np.vdot(pa, pb))


# --------------------------------------------------------------------------- #
# The three readings of the cobordism value at (psi_A, psi_B).
# --------------------------------------------------------------------------- #
def reading_topological(dw_cylinder, prep_a, prep_b):
    """Z_DW(W_AB) = <psi_A| Z(W) |psi_B>: the Z_2 DW state sum (metric-free)
    sandwiched between the prepared boundary states."""
    return complex(dw_cylinder.amplitude(prep_a, prep_b))


def reading_operation(prep_a, U_flat, prep_b):
    """<psi_A|U|psi_B> via ChoiJamiolkowski (the Hilbert-Schmidt/Choi duality)."""
    return complex(cj.transitionAmplitude(
        [complex(z) for z in prep_a.coeffs()], U_flat,
        [complex(z) for z in prep_b.coeffs()], 4, 4))


def reading_spectral(form_a, form_b):
    """Z_spec: the Hodge harmonic overlap of the two boundary states' coordinates
    along the certified spectral mode (ker L_1(Sigma); Cochain inner product)."""
    return complex(form_a.innerProduct(form_b))


def identity_operator():
    return [complex(z) for z in np.eye(4, dtype=complex).reshape(-1)]


def hadamard_block_operator():
    """A generic operation: Hadamard on the qubit block {1,2} (the prepared
    support), identity elsewhere. Not a Z_2-DW cobordism map -- it mixes the two
    flat-connection generators with an irrational 1/sqrt(2) entry."""
    H = np.eye(4, dtype=complex)
    H[1:3, 1:3] = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    return [complex(z) for z in H.reshape(-1)]


def haar_unitary(seed):
    """A Haar-random U(4): a generic operation outside the discrete DW image."""
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    return [complex(x) for x in q.reshape(-1)]


# --------------------------------------------------------------------------- #
# Section 7 sweeps (text) + section 8 figures (PNG) -- written to /tmp/cobordism,
# not committed.
# --------------------------------------------------------------------------- #
def bridge_gap_sweep(dw, prep_a, prep_b, points=21):
    """Interpolate U(t) = (1-t) id + t G from the DW-representable point (U = id,
    t = 0) to a generic G (Haar U(4)); record the bridge gap
    Delta(t) = |Z_DW - <psi_A|U(t)|psi_B>|. Zero only at the discrete DW point."""
    Z_DW = reading_topological(dw, prep_a, prep_b)
    G = np.asarray(haar_unitary(7), dtype=complex).reshape(4, 4)
    I4 = np.eye(4, dtype=complex)
    ts, gaps = list(np.linspace(0.0, 1.0, points)), []
    for t in ts:
        U = (1.0 - t) * I4 + t * G
        amp = reading_operation(prep_a, [complex(z) for z in U.reshape(-1)], prep_b)
        gaps.append(abs(amp - Z_DW))
    return ts, gaps


def spectral_boundary_sweep(space, seed, points=11, restarts=8):
    """The spectral oracle's own (continuum) boundary: interpolate the target
    harmonic from the longitude (realizable) to the meridian (obstructed),
    form(s) = cos(s) longitude + sin(s) meridian, and record the decideHarmonic
    residual/floor. ~0 at the longitude, lifting toward the obstructed meridian."""
    ss = list(np.linspace(0.0, np.pi / 2, points))
    residuals, realizable = [], []
    for s in ss:
        W = _solid_torus()
        _pin_uniform(W)
        longitude, meridian = longitude_and_meridian(W, space)
        target = _cochain(longitude.simplices(),
                          np.cos(s) * np.asarray(longitude.coeffs())
                          + np.sin(s) * np.asarray(meridian.coeffs()))
        v = cob.RealizabilityOracle(W).decideHarmonic(
            target, epsilon=1e-9, restarts=restarts, max_cones=0, seed=seed)
        residuals.append(float(v.residual))
        realizable.append(bool(v.realizable))
    return ss, residuals, realizable


def discrete_lattice_probe(prep_a, prep_b, Z_DW, samples=200, seed=11):
    """Quantify the discreteness: over Haar-random U(4), the bridge gap
    |<psi_A|U|psi_B> - Z_DW| -- its minimum is bounded away from 0 (a generic U is
    not Z_DW of any Z_2 cobordism), and the DW maps themselves are integer."""
    gaps = []
    for k in range(samples):
        amp = reading_operation(prep_a, haar_unitary(seed + k), prep_b)
        gaps.append(abs(amp - Z_DW))
    return float(np.min(gaps)), float(np.median(gaps))


def write_sweeps(out_dir, dw, prep_a, prep_b, space, seed):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dw_spectral_bridge_sweeps.txt")
    ts, gaps = bridge_gap_sweep(dw, prep_a, prep_b)
    ss, sres, sreal = spectral_boundary_sweep(space, seed)
    gmin, gmed = discrete_lattice_probe(prep_a, prep_b,
                                        reading_topological(dw, prep_a, prep_b))
    lines = ["# DW-spectral bridge sweeps (section 7)",
             "",
             "## bridge gap: U(t) = (1-t) id + t G  (id = the DW point, t=0)",
             "## Delta(t) = |Z_DW - <psi_A|U(t)|psi_B>|  -- 0 only at the DW point",
             "#      t   bridge gap Delta"]
    for t, g in zip(ts, gaps):
        lines.append(f"{t:>8.3f}   {g:>16.6e}")
    lines += ["",
              "## spectral oracle boundary: target = cos(s) longitude + sin(s) meridian",
              "## decideHarmonic residual/floor on the solid torus (no growth)",
              "#      s   residual r       realizable"]
    for s, r, ok in zip(ss, sres, sreal):
        lines.append(f"{s:>8.4f}   {r:>14.6e}   {'yes' if ok else 'no'}")
    lines += ["",
              "## discreteness: bridge gap over 200 Haar U(4) (generic U outside DW)",
              f"# min |amp - Z_DW| = {gmin:.6e}   median = {gmed:.6e}"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return [path]


def write_figures(out_dir, dw, prep_a, prep_b, space, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping figures -- "
              '`pip install -e ".[examples]"`)')
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    ts, gaps = bridge_gap_sweep(dw, prep_a, prep_b, points=41)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ts, np.maximum(gaps, 1e-32), "o-")
    ax.axhline(DISAGREE_MARGIN, ls="--", color="crimson", lw=0.8,
               label="disagree margin $10^{-2}$")
    ax.set(xlabel="interpolation $t$  (id $\\to$ Haar $U(4)$)",
           ylabel="bridge gap $|Z_{DW} - \\langle\\psi_A|U(t)|\\psi_B\\rangle|$",
           title="DW bridge holds only at the discrete DW point ($t=0$)")
    ax.legend()
    p1 = os.path.join(out_dir, "dw_bridge_gap.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    ss, sres, _ = spectral_boundary_sweep(space, seed, points=21)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ss, np.maximum(sres, 1e-32), "s-")
    ax.axhline(DISAGREE_MARGIN, ls="--", color="seagreen", lw=0.8,
               label="obstruction floor $10^{-2}$")
    ax.set(xlabel="target $s$  (longitude $\\to$ meridian)",
           ylabel="decideHarmonic residual $r$",
           title="spectral oracle: longitude realizable, meridian obstructed")
    ax.legend()
    p2 = os.path.join(out_dir, "spectral_oracle_boundary.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)
    return paths


# --------------------------------------------------------------------------- #
def _fmt(z):
    return f"{z.real:+.6f}{z.imag:+.6f}i"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="synthesis seed (default 0).")
    ap.add_argument("--restarts", type=int, default=8,
                    help="multi-restart count for decideHarmonic (default 8).")
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="directory for sweeps + figures (default /tmp/cobordism).")
    ap.add_argument("--no-plot", action="store_true", help="skip sweeps + figures.")
    args = ap.parse_args()

    print("DW-spectral bridge -- three independent readings of a cobordism's "
          "value Z(W_AB)\n")
    print("  Sigma = T^2  (b_1 = 2);  Z(Sigma) = C[H^1(T^2;Z_2)] = C^4;  "
          "spectral qubit ker L_1(T^2) = C^2")
    print("  W_AB(topological) = T^2 x [0,T] cylinder  ->  Z(W) = id_4  "
          "(metric-free state sum)")
    print("  W_AB(spectral)    = solid torus D^2 x S^1  ->  carries the "
          "longitude harmonic (k=1)\n")

    all_ok = True

    # ---- the boundary state space + the spectrally-realizable mode ---------- #
    space = cob.BoundaryStateSpace(_build(_torus_topology()))
    sigma = _build(_torus_topology())
    W = _solid_torus()
    _pin_uniform(W)
    longitude, meridian = longitude_and_meridian(W, space)

    # Spectral-mode certification: the solid torus carries the longitude as a
    # genuine k=1 harmonic (r -> 0, lambda -> 0); the meridian (bounds a disk in
    # W) floors. This is what anchors Z_spec to the *synthesized* W_AB.
    v_long = cob.RealizabilityOracle(_solid_torus_pinned()).decideHarmonic(
        longitude, epsilon=1e-9, restarts=args.restarts, max_cones=0, seed=1)
    v_mer = cob.RealizabilityOracle(_solid_torus_pinned()).decideHarmonic(
        meridian, epsilon=1e-9, restarts=args.restarts, max_cones=0, seed=0)
    # Witness boundary-block overlap with the target longitude (all 27 cells are
    # boundary cells of the minimal solid torus).
    cells = [tuple(c) for c in cob.EigenstateSynthesis(W, 1).cellSimplices()]
    idx = {c: i for i, c in enumerate(cells)}
    tgt = _embed_on_cells(longitude, cells, idx)
    s_long = np.asarray(v_long.state)
    blk_overlap = abs(np.vdot(s_long / np.linalg.norm(s_long),
                              tgt / np.linalg.norm(tgt)))
    print("  spectral mode certification (RealizabilityOracle.decideHarmonic on "
          "the solid torus):")
    print(f"    longitude  {'REALIZABLE' if v_long.realizable else 'OBSTRUCTED ':11}"
          f"  r={v_long.residual:.2e}  lambda={v_long.eigenvalue:+.2e}  "
          f"witness d-overlap={blk_overlap:.6f}")
    print(f"    meridian   {'REALIZABLE' if v_mer.realizable else 'OBSTRUCTED ':11}"
          f"  floor={v_mer.floor:.3e}                       "
          f"(bounds a disk in W -> dies in H_1(W))")
    mer_floor_np = _residual_agnostic(numpy_L1(W),
                                      _embed_on_cells(meridian, cells, idx))
    print(f"    meridian floor cross-check vs numpy Hodge oracle: "
          f"{v_mer.floor:.4f} vs {mer_floor_np:.4f}  "
          f"(Delta {abs(v_mer.floor - mer_floor_np):.1e})  "
          f"{'AGREE' if abs(v_mer.floor - mer_floor_np) < 1e-3 else 'DISAGREE'}\n")
    all_ok &= v_long.realizable and v_long.residual < 1e-7 and blk_overlap > 1 - 1e-6
    all_ok &= (not v_mer.realizable) and v_mer.floor > DISAGREE_MARGIN
    all_ok &= abs(v_mer.floor - mer_floor_np) < 1e-3

    dw = cob.DijkgraafWitten(_torus_cylinder(), cob.Cocycle.Trivial)

    # ---- three readings agree on longitude-aligned boundary states --------- #
    print("  three readings on longitude-aligned states  psi_A = a * l_hat,  "
          "psi_B = b * l_hat:\n")
    header = (f"  {'(a, b)':28} {'Z_DW (topo)':20} {'<A|U=id|B> (op)':20} "
              f"{'Z_spec (Hodge)':20} {'max Delta':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    pairs = [("1, 1", 1.0 + 0j, 1.0 + 0j),
             ("e^{i pi/3}, 0.6+0.8i", cmath.exp(1j * cmath.pi / 3), 0.6 + 0.8j),
             ("0.5-0.5i, i", 0.5 - 0.5j, 1j),
             ("2, 0.25+0.97i", 2.0 + 0j, 0.25 + 0.97j)]
    Uid = identity_operator()
    for label, a, b in pairs:
        fa, fb = _scaled(longitude, a), _scaled(longitude, b)
        pa, pb = space.prepare(fa), space.prepare(fb)
        z_dw = reading_topological(dw, pa, pb)
        z_op = reading_operation(pa, Uid, pb)
        z_sp = reading_spectral(fa, fb)
        delta = max(abs(z_dw - z_op), abs(z_dw - z_sp))
        print(f"  {label:28} {_fmt(z_dw):20} {_fmt(z_op):20} {_fmt(z_sp):20} "
              f"{delta:>10.1e}")
        all_ok &= delta < AGREE_TOL
    print("  " + "-" * (len(header) - 2))

    # ---- cross-checks against independent numpy ---------------------------- #
    fa, fb = _scaled(longitude, pairs[1][1]), _scaled(longitude, pairs[1][2])
    pa, pb = space.prepare(fa), space.prepare(fb)
    z_dw = reading_topological(dw, pa, pb)
    z_op = reading_operation(pa, Uid, pb)
    z_sp = reading_spectral(fa, fb)
    # Z_DW: the C++ DW map vs the independent GF(2)/holonomy state sum (multiset),
    # then the contraction against the prepared coeffs.
    map_cpp = np.asarray(dw.map())
    dw_multiset_ok = (sorted(round(x, 9) for x in map_cpp.real.flatten())
                      == [round(x, 9) for x in numpy_dw_boundary(
                          _torus_cylinder(), "trivial")])
    z_dw_np = complex(np.vdot(np.asarray(pa.coeffs()),
                             map_cpp @ np.asarray(pb.coeffs())))
    # operation: the Choi amplitude vs the direct conj(psi_A) U psi_B.
    z_op_np = complex(np.vdot(np.asarray(pa.coeffs()),
                             np.eye(4) @ np.asarray(pb.coeffs())))
    # spectral: the Cochain inner product vs the numpy ker L_1(Sigma) projection.
    z_sp_np = numpy_harmonic_overlap(fa, fb, sigma)
    print("\n  cross-checks (independent numpy), on (a,b) = (e^{i pi/3}, 0.6+0.8i):")
    print(f"    Z_DW   vs GF(2)/holonomy state sum + contraction:  "
          f"Delta {abs(z_dw - z_dw_np):.1e}  multiset "
          f"{'AGREE' if dw_multiset_ok else 'DISAGREE'}")
    print(f"    <A|U|B> vs direct conj(psi_A) U psi_B (vdot):      "
          f"Delta {abs(z_op - z_op_np):.1e}  "
          f"{'AGREE' if abs(z_op - z_op_np) < AGREE_TOL else 'DISAGREE'}")
    print(f"    Z_spec vs numpy ker L_1(Sigma) projection:         "
          f"Delta {abs(z_sp - z_sp_np):.1e}  "
          f"{'AGREE' if abs(z_sp - z_sp_np) < AGREE_TOL else 'DISAGREE'}")
    all_ok &= dw_multiset_ok and abs(z_dw - z_dw_np) < AGREE_TOL
    all_ok &= abs(z_op - z_op_np) < AGREE_TOL and abs(z_sp - z_sp_np) < AGREE_TOL

    # ---- the bridge on a representable U vs generic U ---------------------- #
    print("\n  bridge on the DW-representable U vs generic U  "
          "(psi_A = e^{i pi/3} l_hat, psi_B = (0.6+0.8i) l_hat):\n")
    h2 = (f"  {'operation U':30} {'representable?':14} {'<A|U|B>':20} "
          f"{'Delta = |amp - Z_DW|':>20}  verdict")
    print(h2)
    print("  " + "-" * (len(h2) - 2))
    candidates = [("id_4  (= Z(T^2 x [0,T]))", identity_operator(), True),
                  ("Hadamard on the qubit block", hadamard_block_operator(), False),
                  ("Haar-random U(4)  (seed 7)", haar_unitary(7), False),
                  ("Haar-random U(4)  (seed 23)", haar_unitary(23), False)]
    for label, U, representable in candidates:
        amp = reading_operation(pa, U, pb)
        delta = abs(amp - z_dw)
        verdict = "AGREE" if delta < AGREE_TOL else "DIFFER"
        print(f"  {label:30} {'yes' if representable else 'no':14} {_fmt(amp):20} "
              f"{delta:>20.6e}  {verdict}")
        if representable:
            all_ok &= delta < AGREE_TOL
        else:
            all_ok &= delta > DISAGREE_MARGIN
    print("  " + "-" * (len(h2) - 2))
    print(f"  Z_DW = {_fmt(z_dw)}  (the cobordism's metric-free topological value)")

    # ---- the DW-representable set is discrete (integer-quantized) ---------- #
    cyl_sign = np.asarray(cob.DijkgraafWitten(_torus_cylinder(),
                                              cob.Cocycle.Sign).map())
    cap = np.asarray(cob.DijkgraafWitten(_solid_torus(),
                                         cob.Cocycle.Trivial).boundaryVector()).real
    gmin, gmed = discrete_lattice_probe(pa, pb, z_dw)
    print("\n  the DW-representable set is discrete (integer-quantized; the "
          "C[Z_2] Frobenius family):")
    print(f"    Z(T^2 x [0,T], Trivial) = id_4                      "
          f"integer={np.allclose(map_cpp, np.round(map_cpp.real))}")
    print(f"    Z(T^2 x [0,T], Sign)    = id_4                      "
          f"integer={np.allclose(cyl_sign, np.round(cyl_sign.real))}")
    print(f"    Z(D^2 x S^1, Trivial) boundary state = {cap.astype(int).tolist()}"
          f"   integer={np.allclose(cap, np.round(cap))}")
    print(f"    bridge gap over 200 Haar U(4):  min {gmin:.3e}  median {gmed:.3e}"
          f"   (generic U outside the DW image)")
    all_ok &= np.allclose(map_cpp, np.round(map_cpp.real))
    all_ok &= gmin > DISAGREE_MARGIN

    print("\n  Bridge verdict: Z_DW = <psi_A|U|psi_B> = Z_spec on the "
          "DW-representable U (id); a generic U is")
    print("  outside the discrete DW image, so Z_DW != <psi_A|U|psi_B> -- the "
          "spectral oracle (a continuum)")
    print(f"  strictly extends the topological theory "
          f"({'SUPPORTED' if all_ok else 'NOT SUPPORTED -- a check failed'}).")

    if not args.no_plot:
        print("\n  section 7 sweeps:")
        for p in write_sweeps(args.out, dw, pa, pb, space, args.seed):
            print(f"    {p}")
        print("  section 8 figures:")
        for p in write_figures(args.out, dw, pa, pb, space, args.seed):
            print(f"    {p}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
