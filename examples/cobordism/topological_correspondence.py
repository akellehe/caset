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

"""Stage-2 correctness oracle for the State-Operation-Cobordism correspondence.

Runs the topological-layer checks T1-T5 from
``docs/source/quantum-experiments/cobordism.md`` (the n=3 layer, where the sign
becomes a genuine topological invariant) and prints a pass/fail table with
residuals, then writes the parameter sweeps (sec 7) and figures (sec 8). It is
the Stage-2 capstone, mirroring the Stage-1 algebraic oracle
``algebraic_correspondence.py``.

The pieces under test, all consuming the merged + tested Stage-2 machinery (the
ℤ₂ Dijkgraaf-Witten state sum ``Z(W)``, the pre-geometric Pachner engine, the
gluing functor, and the U(1)/Lorentzian Hodge tooling):

* T1  cylinder = identity   -- DijkgraafWitten(Cylinder).map() == id; the
                               amplitude is the inner product (P1)
* T2  Pachner invariance    -- Z(S²×S¹) fixed across interior Pachner moves and
                               across distinct triangulations (P2, make-or-break)
* T3  sign is an invariant  -- Z_Sign(RP³) = 0 ≠ 1 = Z_Trivial(RP³); equal on the
                               g³ = 0 negatives S²×S¹ and T³ (P3)
* T4  cross-layer holonomy  -- WilsonLoop U(1) holonomy == Stage-1 cycle flux,
                               and the ℤ₂ restriction lands in {0, π}
* T5  functoriality         -- Z(glue(W₁,W₂)).map() == map(W₂)·map(W₁)
* §5.6 Lorentzian           -- (supplementary) the harmonic's indefinite norm
                               (2-α)/3 goes null at α = 2 on the timelike 3-cycle

The falsifiable predictions: P1, P2, P3 are *supported* iff T1, T2, T3 pass
(these three decide the hypothesis); P4, P5 are carried from Stage 1
(``algebraic_correspondence.py``). T4, T5 and the §5.6 row are supplementary
consistency checks reported alongside.

Run:  python examples/cobordism/topological_correspondence.py
      (use --help for options; sweeps + figures default to /tmp/cobordism and
      are not committed -- attach them to the issue/PR if you want to pin a
      result.)
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import tessera

cob = tessera.cobordism
DijkgraafWitten = cob.DijkgraafWitten
Cocycle = cob.Cocycle
Cobordism = cob.Cobordism

PRE = tessera.PachnerMode.PreGeometric
TWO_PI = 2.0 * math.pi

# The closed-form §5.6 alpha sweep (deliberately skips alpha = 2, the defective
# crossing): the harmonic null-norm (2-alpha)/3 is positive below, negative above.
ALPHA_SWEEP = (0.25, 0.5, 1.0, 1.5, 2.5, 3.0, 4.0)


# --------------------------------------------------------------------------- #
# Fixtures (closed 3-manifolds for the state sum; CDT-built simplicial mesh)
# --------------------------------------------------------------------------- #
def _build(topology, dimensions=None):
    if dimensions is None:
        dimensions = topology.dimension()
    sig = tessera.Signature(dimensions, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _chain(topology):
    return cob.ChainComplex.fromSpacetime(_build(topology))


def _circle():
    return tessera.SimplexBoundarySphere(1)               # S¹ = ∂Δ²


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())  # T² = S¹ × S¹


def _interval():
    return tessera.SolidSimplex(1)                        # [0, 1] (one edge)


def _torus_cylinder():
    # W = T² × [0,T], the trivial cobordism T² → T² (∂W = T² ⊔ T²).
    return _build(tessera.SimplicialProduct(_torus_topology(), _interval()))


def _rp3():
    return tessera.RealProjectiveSpace()                  # RP³ = L(2,1), t³ ≠ 0


def _s2_cross_s1():
    # S² × S¹: betti (1,1,1,1), no torsion, g³ = 0 (sign-cocycle negative).
    return tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2), _circle())


def _t3():
    # T³ = S¹×S¹×S¹: g³ = 0 too, but 27 vertices ⇒ dim Z¹ = 29 > 24, beyond the
    # brute-force state sum, so its negative status is read off the homology.
    return tessera.SimplicialProduct(_torus_topology(), _circle())


# --------------------------------------------------------------------------- #
# Hermitian-weighted 1-complex fixtures (Stage-1 idiom, for T4 and §5.6)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for s in simplices:
        st.createSimplex([verts[i] for i in s])
    return st


def _triangle_hw():
    """S¹ as the 1-skeleton cycle 0-1-2-0 (b₁ = 1)."""
    return _from_simplices(3, [(0, 1), (1, 2), (2, 0)])


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _set_phases(st, phases):
    """Unit positive weights everywhere; phase from {frozenset({a,b}): φ}."""
    for e in st.getEdgeList().toVector():
        key = frozenset({e.getSource().getId(), e.getTarget().getId()})
        e.setSquaredLength(1.0)
        e.setPhase(phases.get(key, 0.0))


def _set_all_spacelike(st, l2=1.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(l2)
        e.setPhase(0.0)
    return st


def _triangle_one_timelike(alpha):
    """The 3-cycle with edge (1,2) timelike (l² = -α²); the rest spacelike."""
    st = _set_all_spacelike(_triangle_hw(), 1.0)
    _edge(st, 1, 2).setSquaredLength(-(alpha ** 2))
    return st


# --------------------------------------------------------------------------- #
# Mod-2π and flux helpers (the same oracles the Stage-1 / Hodge tests use)
# --------------------------------------------------------------------------- #
def _modgap(a, b):
    """|a - b| reduced modulo 2π (robust at the ±π boundary)."""
    return abs(math.remainder(a - b, TWO_PI))


def _close_mod_2pi(a, b, tol=1e-9):
    return _modgap(a, b) < tol


def _cycle_flux(st, cycle):
    """Hand oracle: oriented holonomy sum of phase around a closed vertex cycle."""
    total = 0.0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.getSource().getId() == a and e.getTarget().getId() == b:
            total += e.getPhase()
        else:
            total -= e.getPhase()
    return total


def _hodge_flux(st, cycle):
    """Stage-1 cycle flux read off cobordism.HodgeLaplacian's complex adjacency
    A[i,j] = w·exp(i·θ): arg(A) is the oriented connection, so the directed sum
    around the cycle is the holonomy the operator encodes in L = D - A."""
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.array(cob.HodgeLaplacian(st).adjacency(), dtype=complex).reshape(n, n)
    total = 0.0
    m = len(cycle)
    for k in range(m):
        a, b = cycle[k], cycle[(k + 1) % m]
        total += np.angle(A[idx[a], idx[b]])
    return float(total)


def _holonomy(st, cycle):
    """WilsonLoop U(1)-connection holonomy around a vertex-id cycle (mod 2π)."""
    wl = tessera.WilsonLoop(st)
    by_id = {v.getId(): v for v in st.getVertexList().toVector()}
    return wl.evaluateU1Connection([by_id[c] for c in cycle]).value


# --------------------------------------------------------------------------- #
# Pre-geometric Pachner machinery (the T2 sweep)
# --------------------------------------------------------------------------- #
def _make(cls, st, seed, boundary_fixed=False):
    """A pre-geometric move (AddMove takes the extra `relabel` positional)."""
    if cls is tessera.AddMove:
        return cls(st, seed, False, PRE, boundary_fixed)
    return cls(st, seed, PRE, boundary_fixed)


def _tops(st):
    sizes = [len(s.getVertices()) for s in st.getSimplices()]
    if not sizes:
        return []
    top = max(sizes)
    return [tuple(sorted(v.getId() for v in s.getVertices()))
            for s in st.getSimplices() if len(s.getVertices()) == top]


def _is_closed_pseudomanifold(tops):
    """Every codim-1 face is shared by exactly two top cells."""
    counts = {}
    for t in tops:
        for drop in range(len(t)):
            facet = t[:drop] + t[drop + 1:]
            counts[facet] = counts.get(facet, 0) + 1
    return bool(counts) and all(c == 2 for c in counts.values())


def _pachner_z_series(max_depth, seed):
    """Apply interior pre-geometric Pachner moves to S²×S¹ and record Z(W) at
    every depth for both cocycles (the |V| cap keeps the flat space enumerable).

    The signature is 3D so the tetrahedra register as top cells for the moves;
    Z reads the chain complex, so the value is the same as the 4D build."""
    st = _build(tessera.SphereCircleProduct(), dimensions=3)
    st.setSeed(seed)
    zt0 = DijkgraafWitten(st, Cocycle.Trivial).partitionFunction()
    zs0 = DijkgraafWitten(st, Cocycle.Sign).partitionFunction()

    depths, zt, zs = [0], [zt0.real], [zs0.real]
    drift, closed = 0.0, True
    counts = {"FlipMove": 0, "IFlipMove": 0, "AddMove": 0}
    classes = (tessera.FlipMove, tessera.IFlipMove, tessera.AddMove)
    depth, seed_k = 0, 0
    while depth < max_depth and seed_k < 6000:
        for cls in classes:
            if cls is tessera.AddMove and st.getVertexCount() >= 16:
                continue                      # cap |V| so 2^{|V|-1+b1} stays small
            m = _make(cls, st, seed_k)
            if m.propose() and m.apply():
                depth += 1
                counts[cls.__name__] += 1
                closed &= _is_closed_pseudomanifold(_tops(st))
                zt_d = DijkgraafWitten(st, Cocycle.Trivial).partitionFunction()
                zs_d = DijkgraafWitten(st, Cocycle.Sign).partitionFunction()
                drift = max(drift, abs(zt_d - zt0), abs(zs_d - zs0))
                depths.append(depth)
                zt.append(zt_d.real)
                zs.append(zs_d.real)
                break
        seed_k += 1
    return {"depths": depths, "zt": zt, "zs": zs, "drift": drift,
            "depth": depth, "counts": counts, "closed": closed,
            "z0": (zt0.real, zs0.real)}


def _alpha_null_series(alphas):
    """Harmonic indefinite norm vs. the closed form (2-α)/3 on the timelike 3-cycle."""
    norms, expected = [], []
    for a in alphas:
        nn = cob.HodgeLaplacian(_triangle_one_timelike(a)).lorentzianNullNorms(
            1, 1e-9, True)
        norms.append(float(nn[0]))
        expected.append((2.0 - a) / 3.0)
    return list(alphas), norms, expected


def _boundary_states_from_harmonics():
    """Two Z(T²) states (length 2^{b₁}=4) seeded by the ker L₁(T²) harmonics —
    the b₁ = 2 qubit, distinct from the 2^{b₁} = 4 flat-connection count."""
    torus = _build(_torus_topology())
    harmonics = cob.HodgeLaplacian(torus).harmonics(1, 1e-9, True)
    # columns = the harmonic 1-form Cochains over the edge ordering.
    cols = (np.column_stack([np.asarray(h.coeffs()) for h in harmonics])
            if harmonics else np.zeros((0, 0), dtype=complex))
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


# --------------------------------------------------------------------------- #
# Checks: each returns (passed: bool, residual: float, detail: str)
# --------------------------------------------------------------------------- #
def check_t1(rng):
    """T1 (P1): the cylinder T²×[0,T] is the identity on Z(T²), so
    ⟨ψ_A|Z(W)|ψ_B⟩ = ⟨ψ_A|ψ_B⟩ for boundary states ψ ∈ ker L₁(T²)."""
    cyl = _torus_cylinder()
    matrix = np.asarray(DijkgraafWitten(cyl, Cocycle.Trivial).map())
    map_res = float(np.max(np.abs(matrix - np.eye(4))))   # Z(T²) is 4-dimensional

    dw = DijkgraafWitten(cyl, Cocycle.Trivial)
    psi_a, psi_b = _boundary_states_from_harmonics()
    amp_res = abs(dw.amplitude(list(psi_a), list(psi_b)) - np.vdot(psi_a, psi_b))
    for _ in range(8):                                     # random Z(T²) probes too
        a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        amp_res = max(amp_res, abs(dw.amplitude(list(a), list(b)) - np.vdot(a, b)))

    res = max(map_res, float(amp_res))
    return res < 1e-9, res, ("map(T²×I)=id₄; ⟨ψ_A|Z(W)|ψ_B⟩=⟨ψ_A|ψ_B⟩ "
                             "(ψ ∈ ker L₁(T²))")


def check_t2(seed, max_depth):
    """T2 (P2, make-or-break): Z(S²×S¹) is invariant across interior Pachner
    moves (2↔3 flips and 1→4 stellar adds) and across distinct triangulations."""
    # (a) retriangulation invariance: the product vs one stellar subdivision.
    product = _build(tessera.SphereCircleProduct())
    subdivided = _build(tessera.StellarSubdivision(tessera.SphereCircleProduct()))
    retri = 0.0
    for cocycle in (Cocycle.Trivial, Cocycle.Sign):
        z_p = DijkgraafWitten(product, cocycle).partitionFunction()
        z_s = DijkgraafWitten(subdivided, cocycle).partitionFunction()
        retri = max(retri, abs(z_p - z_s), abs(z_p - 1.0))   # anchor Z = 1

    # (b) the headline: Z fixed across an interior Pachner sweep, both cocycles.
    series = _pachner_z_series(max_depth, seed)
    depth, drift, counts = series["depth"], series["drift"], series["counts"]
    flips = counts["FlipMove"] + counts["IFlipMove"]
    adds = counts["AddMove"]

    ok = (retri < 1e-12 and drift < 1e-9 and series["closed"]
          and depth >= 15 and flips > 0 and adds > 0)
    res = max(retri, drift)
    detail = (f"Z(S²×S¹) drift {drift:.1e} over {depth} interior moves "
              f"(F/IF {flips}, 1→4 add {adds}); product==subdivided==1")
    return ok, res, detail


def check_t3():
    """T3 (P3): the sign cocycle carries a topological invariant.
    Z_Sign(RP³) = 0 ≠ 1 = Z_Trivial(RP³) (the cup cube t³ ≠ 0); on the g³ = 0
    negatives the sign cannot distinguish — S²×S¹ directly (Z_Sign = Z_Trivial),
    T³ via its torsion-free H₁ (too large to brute-force), both unlike RP³'s
    2-torsion."""
    rp3 = _build(_rp3())
    z_triv = DijkgraafWitten(rp3, Cocycle.Trivial).partitionFunction()
    z_sign = DijkgraafWitten(rp3, Cocycle.Sign).partitionFunction()
    gap = abs(z_triv - z_sign)

    s2s1 = _build(_s2_cross_s1())                          # computable negative
    neg = abs(DijkgraafWitten(s2s1, Cocycle.Trivial).partitionFunction()
              - DijkgraafWitten(s2s1, Cocycle.Sign).partitionFunction())

    tor_rp3 = list(_chain(_rp3()).torsion(1))             # H₁ 2-torsion fingerprint
    tor_s2s1 = list(_chain(_s2_cross_s1()).torsion(1))
    tor_t3 = list(_chain(_t3()).torsion(1))

    ok = (abs(z_triv.real - 1.0) < 1e-9 and abs(z_sign.real) < 1e-9 and gap > 0.5
          and neg < 1e-9 and tor_rp3 == [2] and tor_s2s1 == [] and tor_t3 == [])
    res = max(neg, abs(z_triv.real - 1.0), abs(z_sign.real))
    detail = (f"Z_Sign(RP³)={z_sign.real:.2f} ≠ Z_Triv(RP³)={z_triv.real:.2f} "
              f"(gap {gap:.2f}); g³=0 negs S²×S¹,T³ (torsion H₁ RP³={tor_rp3} vs "
              f"S²×S¹={tor_s2s1},T³={tor_t3})")
    return ok, res, detail


def check_t4(rng):
    """T4 (cross-layer): the WilsonLoop U(1) holonomy equals the Stage-1 cycle
    flux (read off the Hodge adjacency) mod 2π; with ℤ₂ phases in {0, π} it lands
    in {0, π} — π for an odd number of π-edges, 0 for even."""
    worst = 0.0
    z2_ok = True
    edges = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 0})]
    saw_zero = saw_pi = False
    for k in range(len(edges) + 1):                        # ℤ₂ parity sweep
        st = _triangle_hw()
        _set_phases(st, {e: math.pi for e in edges[:k]})
        val = _holonomy(st, [0, 1, 2])
        expected = math.pi if (k % 2 == 1) else 0.0
        in_z2 = abs(val) < 1e-9 or abs(abs(val) - math.pi) < 1e-9
        z2_ok &= in_z2 and _close_mod_2pi(val, expected)
        worst = max(worst, _modgap(val, _hodge_flux(st, [0, 1, 2])),
                    _modgap(val, _cycle_flux(st, [0, 1, 2])))
        saw_zero |= (expected == 0.0)
        saw_pi |= (expected == math.pi)

    for _ in range(8):                                     # generic-phase match
        st = _triangle_hw()
        _set_phases(st, {frozenset({a, b}): float(rng.uniform(-math.pi, math.pi))
                         for a, b in ((0, 1), (1, 2), (2, 0))})
        val = _holonomy(st, [0, 1, 2])
        worst = max(worst, _modgap(val, _hodge_flux(st, [0, 1, 2])),
                    _modgap(val, _cycle_flux(st, [0, 1, 2])))

    ok = z2_ok and saw_zero and saw_pi and worst < 1e-9
    return ok, worst, ("holonomy == Stage-1 flux Φ_γ (mod 2π); "
                       "ℤ₂ phases → {0,π} by parity")


def check_t5():
    """T5 (functoriality): gluing two cylinders composes their boundary maps,
    Z(glue(W₁,W₂)).map() == map(W₂)·map(W₁) (= id₄ here), for both cocycles."""
    worst = 0.0
    for cocycle in (Cocycle.Trivial, Cocycle.Sign):
        w1, w2 = _torus_cylinder(), _torus_cylinder()
        glued = Cobordism.glue(w1, w2)
        map1 = np.asarray(DijkgraafWitten(w1, cocycle).map())
        map2 = np.asarray(DijkgraafWitten(w2, cocycle).map())
        map_glued = np.asarray(DijkgraafWitten(glued, cocycle).map())
        worst = max(worst, float(np.max(np.abs(map_glued - map2 @ map1))),
                    float(np.max(np.abs(map_glued - np.eye(4)))))
    return worst < 1e-12, worst, ("map(glue(W₁,W₂)) == map(W₂)·map(W₁) == id₄ "
                                  "(T²×I ∘ T²×I)")


def check_lorentzian():
    """§5.6 (supplementary): on the timelike 3-cycle the harmonic's indefinite
    norm is ⟨h,h⟩_W = (2-α)/3 — positive below, null at, negative above α = 2 —
    so the harmonic representative becomes null exactly at the α = 2 crossing."""
    alphas, norms, expected = _alpha_null_series(ALPHA_SWEEP)
    worst = float(np.max(np.abs(np.array(norms) - np.array(expected))))
    monotone = all(x > y for x, y in zip(norms, norms[1:]))   # strictly down
    crosses = (any(a < 2.0 and n > 0 for a, n in zip(alphas, norms))
               and any(a > 2.0 and n < 0 for a, n in zip(alphas, norms)))
    ok = worst < 1e-6 and monotone and crosses
    return ok, worst, "harmonic ⟨h,h⟩_W=(2-α)/3: + below, null at, - above α=2"


# --------------------------------------------------------------------------- #
# §7 parameter sweeps (text) and §8 figures (PNG) -- not committed
# --------------------------------------------------------------------------- #
def write_sweeps(out_dir, seed, max_depth):
    """Write the sec-7 numeric sweeps (Pachner depth, Lorentzian α) to a table."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage2_sweeps.txt")
    series = _pachner_z_series(max_depth, seed)
    alphas, norms, expected = _alpha_null_series(ALPHA_SWEEP)
    lines = ["# State-Operation-Cobordism -- Stage 2 parameter sweeps (sec 7)",
             "",
             "## T2  bulk-refinement / Pachner-depth invariance on S2xS1",
             "# depth   Z_Trivial      Z_Sign"]
    for d, zt, zs in zip(series["depths"], series["zt"], series["zs"]):
        lines.append(f"{d:>6}   {zt:>12.9f}   {zs:>12.9f}")
    lines += ["",
              "## sec 5.6  Lorentzian alpha sweep on the timelike 3-cycle",
              "# alpha   null_norm      (2-alpha)/3"]
    for a, n, e in zip(alphas, norms, expected):
        lines.append(f"{a:>6.3f}   {n:>12.9f}   {e:>12.9f}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return [path]


def write_figures(out_dir, seed, max_depth, alpha_points):
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

    # (1) T2: Z(S²×S¹) flat across the Pachner depth sweep.
    series = _pachner_z_series(max_depth, seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(series["depths"], series["zt"], "o-", label="$Z_{\\mathrm{Trivial}}$")
    ax.plot(series["depths"], series["zs"], "s--", label="$Z_{\\mathrm{Sign}}$")
    ax.set(xlabel="interior Pachner depth", ylabel="$Z(S^2\\times S^1)$",
           title="T2: $Z(W)$ invariant under interior Pachner moves",
           ylim=(0.0, 2.0))
    ax.legend()
    p1 = os.path.join(out_dir, "pachner_depth_invariance.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    # (2) §5.6: harmonic null-norm (2-α)/3 crossing zero at α = 2.
    alphas = list(np.linspace(0.25, 4.0, max(alpha_points, 16)))
    alphas = [a for a in alphas if abs(a - 2.0) > 1e-6]   # skip the defective point
    _, norms, expected = _alpha_null_series(alphas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, norms, "o", label="$\\langle h,h\\rangle_W$ (computed)")
    ax.plot(alphas, expected, "-", color="crimson", label="$(2-\\alpha)/3$")
    ax.axhline(0.0, ls=":", color="gray", lw=0.8)
    ax.axvline(2.0, ls="--", color="gray", lw=0.8)
    ax.set(xlabel="CDT asymmetry $\\alpha$", ylabel="harmonic indefinite norm",
           title="§5.6: harmonic becomes null at $\\alpha=2$")
    ax.legend()
    p2 = os.path.join(out_dir, "lorentzian_null_norm_vs_alpha.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)

    # (3) §5.6: the closed-form Lorentzian spectrum {0, 3, 1 - 2/α} vs α.
    spectrum = np.array([np.sort(np.array(
        cob.HodgeLaplacian(_triangle_one_timelike(a)).lorentzianEigenvalues(1, True),
        dtype=complex).real) for a in alphas])
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(spectrum.shape[1]):
        ax.plot(alphas, spectrum[:, k], label=f"$\\lambda_{k}$")
    ax.axhline(0.0, ls=":", color="gray", lw=0.8)
    ax.axvline(2.0, ls="--", color="gray", lw=0.8)
    ax.set(xlabel="CDT asymmetry $\\alpha$", ylabel="eigenvalue",
           title="§5.6: Lorentzian d'Alembertian spectrum vs $\\alpha$")
    ax.legend()
    p3 = os.path.join(out_dir, "lorentzian_spectrum_vs_alpha.png")
    fig.tight_layout(); fig.savefig(p3, dpi=120); plt.close(fig); paths.append(p3)
    return paths


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="RNG / Pachner seed (default 0).")
    ap.add_argument("--pachner-depth", type=int, default=18,
                    help="interior moves in the T2 invariance sweep (default 18).")
    ap.add_argument("--alpha-points", type=int, default=16,
                    help="samples on alpha for the §5.6 figure (default 16).")
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="directory for sweeps + figures (default /tmp/cobordism).")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures.")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    checks = [
        ("T1  cylinder = identity", lambda: check_t1(rng)),
        ("T2  Pachner invariance", lambda: check_t2(args.seed, args.pachner_depth)),
        ("T3  sign is an invariant", check_t3),
        ("T4  cross-layer holonomy", lambda: check_t4(rng)),
        ("T5  composition / functoriality", check_t5),
        ("§5.6  Lorentzian null-harmonic", check_lorentzian),
    ]

    print("State-Operation-Cobordism correspondence -- Stage 2 (topological oracle)\n")
    print(f"  {'check':36} {'result':6} {'residual':>11}  detail")
    print("  " + "-" * 92)
    results = {}
    for name, fn in checks:
        ok, residual, detail = fn()
        results[name[:2]] = ok
        print(f"  {name:36} {'PASS' if ok else 'FAIL':6} {residual:11.2e}  {detail}")
    print("  " + "-" * 92)

    core_ok = results["T1"] and results["T2"] and results["T3"]
    all_ok = all(results.values())
    print(f"\n  P1, P2, P3 supported (cylinder / Pachner / sign decide the "
          f"hypothesis): {'YES' if core_ok else 'NO -- a core check FAILED'}")
    print("  P4, P5 carried from Stage 1 (examples/cobordism/algebraic_correspondence.py).")
    print(f"  Supplementary T4, T5, §5.6: "
          f"{'all PASS' if all_ok else 'a supplementary check FAILED'}")

    if not args.no_plot:
        print("\n  sec-7 sweeps:")
        for p in write_sweeps(args.out, args.seed, args.pachner_depth):
            print(f"    {p}")
        print("  sec-8 figures:")
        for p in write_figures(args.out, args.seed, args.pachner_depth,
                               args.alpha_points):
            print(f"    {p}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
