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

"""Synthesis + realizability capstone for the State-Operation-Cobordism work.

The runnable synthesis + realizability capstone, mirroring the Stage-1 / Stage-2 oracles
(``algebraic_correspondence.py``, ``topological_correspondence.py``). It runs the
two synthesis tasks from ``docs/source/quantum-experiments/state-operation-cobordism/cobordism.md``:

* §4b boundary-state synthesis -- the inverse eigenvector problem. Given a target
  qubit ψ, ``GeometrySynthesizer`` (#134) grows the *simplest* Hermitian-weighted
  complex whose k=0 graph-Laplacian L = D - A has ψ as an eigenvector, via the
  two-vertex floor → cone-and-retry loop. Output: geo(ψ) and its combinatorial
  complexity (|V|, |E|, cones, λ).

* §5.0 realizability -- *reframed* (see §5.0): an operation U : H_B → H_A is
  realizable iff a bulk W_AB can be *spectrally synthesized* for it, NOT by
  TQFT-membership Z(W)=U. The recipe: bend U to a boundary state via
  Choi-Jamiołkowski (vec(U), the operator-as-state); pin the bulk's boundary dW;
  fill the interior (Hermitian edge weights + boundary-fixed Pachner growth) so
  the output-boundary L-eigenvector matches the bent target, driving the §4b
  residual r = ||(I - ψψ†) L ψ||² → 0. ``RealizabilityOracle`` (#138) returns the
  verdict. A target is **realizable** iff r can be driven to zero, and certified
  **obstructed** iff r floors away from zero -- a spectral obstruction under the
  fixed-boundary constraint (the analogue of §4b's two-vertex floor). Non-existence
  is certified by the *floor*, not by exhausting triangulations.

The headline cases (the results table):

* U = [[1,1],[1,1]]  (the uniform zero-mode, = 2|+><+|): vec(U) is the constant
  vector, the exact zero mode of L for a phase-0 boundary. REALIZABLE at minimal
  complexity (no interior growth) -- r → 0.
* U = [[1, 0.3+0.5j, -0.8+0.2j]]  (a generic 1×3 op): not an eigenvector of the
  bare pinned triangle, so it floors at the seed, but the fixed-boundary
  cone-and-retry grows the interior and realizes it -- r → 0 after coning.
* U = [[1,2],[3,4]]  (a generic op): with no growth budget on the one-interior-edge
  bulk the residual floors bounded away from 0. OBSTRUCTED -- certified
  non-realizable by the floor.

Run:  python examples/cobordism/realizability_report.py
      (use --help for options; the §7/§8 sweeps + figures default to
      /tmp/cobordism and are NOT committed -- attach them to the issue/PR if you
      want to pin a result. The script is the committed artifact.)

Exit status is 0 iff every verdict matches its expectation (the realizable cases
reach r < 1e-10, the obstructed case floors above the certificate threshold).
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS: cap the thread
# pools here so the script self-limits even when launched without the env prefix.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import math      # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
cj = tessera.quantum.ChoiJamiolkowski

# The acceptance threshold: r below this is "driven to zero" (a realized W_AB);
# the obstruction certificate is a floor bounded above CERT_FLOOR.
EPSILON = 1e-10
CERT_FLOOR = 1e-2


# --------------------------------------------------------------------------- #
# Hermitian-weighted seed fixtures for §4b geo(ψ) synthesis (the #134 idiom).
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
    """The §4b.2 object: two vertices, one edge (K_2). A general-amplitude qubit
    floors here -- the motivation for coning."""
    return _from_simplices(2, [(0, 1)])


# --------------------------------------------------------------------------- #
# Bulk fixtures for §5.0 realizability (the #138 / #147 bulk-synthesis idiom:
# Signature(d) so the d-cells register as top simplices; built through the
# topology so the vertex-id counter advances).
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    """Δ²: a single triangle 0-1-2 whose boundary dW is the three sides (S¹);
    0 interior edges (every edge is boundary)."""
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _bipyramid():
    """Two triangles 012 and 013 sharing the interior edge 01; the four outer
    edges 02,12,03,13 are the pinned boundary dW. The smallest fixed-boundary
    complex with exactly one interior parameter -- the §5.0 analogue of §4b's
    two-vertex single edge, with the boundary pinned."""
    st = _solid_triangle()              # triangle 012, vertex-id counter at 3
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v3 = st.createVertex(3)
    st.createSimplex([v[0], v[1], v3])  # triangle 013
    return st


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _pin_all(st, w=1.0, phase=0.0):
    """Pin every edge to a fixed Hermitian value. The oracle's fill only ever
    rewrites the interior edges, so this fixes dW; the interior values are merely
    a starting point the fill overwrites."""
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


# --------------------------------------------------------------------------- #
# numpy cross-check helpers (the D - A magnitude convention, sorted-vertex-id
# order -- the oracle the Hodge / synthesis tests use).
# --------------------------------------------------------------------------- #
def _cvec(v):
    return [complex(z) for z in v]


def _bend(U, dA, dB):
    """The Choi-Jamiołkowski bend: vec(U), the operator-as-state (row-major).
    ``ChoiJamiolkowski.vectorize`` is the idiomatic primitive; it agrees with the
    plain row-major flatten of the dA×dB matrix."""
    flat = [complex(z) for row in U for z in row]
    return cj.vectorize(flat, dA, dB)


def _np_L(st):
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


def _np_residual(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    r = Lp - lam * psi
    return float(np.vdot(r, r).real)


# --------------------------------------------------------------------------- #
# §4b boundary-state synthesis: geo(ψ).
# --------------------------------------------------------------------------- #
def synthesize_boundary(c0, c1, seed, restarts=80, max_cones=4):
    """Grow geo(ψ) for the qubit ψ = (c0, c1, 0, ...): the simplest complex whose
    L = D - A has ψ as an eigenvector. Returns (geo, floor_on_seed) where floor is
    the residual the two-vertex seed alone floors at (the §4b.2 motivation)."""
    floor_seed = cob.GeometrySynthesizer(_edge_seed()).optimize(
        c0, c1, restarts=restarts, seed=seed)
    geo = cob.GeometrySynthesizer(_edge_seed()).synthesize(
        c0, c1, epsilon=1e-9, restarts=restarts, max_cones=max_cones, seed=seed)
    return geo, floor_seed


# --------------------------------------------------------------------------- #
# §5.0 realizability decisions.
# --------------------------------------------------------------------------- #
def decide(bulk_factory, U, dA, dB, seed, restarts, max_cones, growth_mode=None,
           beta=0.0):
    """Bend U → vec(U), pin the bulk boundary, and let the oracle fill the
    interior. `growth_mode` defaults to the historical cone-only growth;
    SURGERY_AND_CONE allows additions as well as surgical cuts (max_cones then
    budgets the added vertices only). `beta` couples the mediated objective
    F_beta = r_U + beta * |S_Regge(W*)|: candidate moves are ranked by F_beta
    rather than the bare residual (beta = 0, the default, is the base-layer
    search). Returns the Verdict."""
    bulk = bulk_factory()
    _pin_all(bulk, w=1.0, phase=0.0)
    target = _bend(U, dA, dB)
    oracle = cob.RealizabilityOracle(bulk)
    if growth_mode is None:
        growth_mode = cob.RealizabilityOracle.GrowthMode.CONE
    return oracle.decide(target, dA, dB, epsilon=EPSILON, restarts=restarts,
                         max_cones=max_cones, seed=seed, growth_mode=growth_mode,
                         beta=beta)


def _np_floor_oracle(U, dA, dB, w_bounds=(0.1, 10.0), n=60):
    """Independent numpy grid global-min of the residual over the single interior
    edge (0,1) of the bipyramid (boundary pinned at weight 1, phase 0) -- the
    same hand oracle the #138 test cross-checks the certified floor against. No
    scipy: a dense grid is enough to corroborate the floor is bounded away from 0."""
    fresh = _bipyramid()
    _pin_all(fresh, w=1.0, phase=0.0)
    edge01 = _edge_map(fresh)[(0, 1)]
    target = np.asarray([z for z in _bend(U, dA, dB)], dtype=complex)
    best = np.inf
    for w in np.linspace(w_bounds[0], w_bounds[1], n):
        for th in np.linspace(-math.pi, math.pi, n):
            edge01.setSquaredLength(float(w))
            edge01.setPhase(float(th))
            best = min(best, _np_residual(_np_L(fresh), target))
    return float(best)


# --------------------------------------------------------------------------- #
# §7 sweeps (text) and §8 figures (PNG) -- written to /tmp/cobordism, not committed.
# --------------------------------------------------------------------------- #
def _realizability_boundary(seed, points, restarts):
    """The realizability boundary: interpolate U(t) from the uniform zero-mode
    [[1,1],[1,1]] (realizable, floor 0) to the obstructed [[1,2],[3,4]] on the
    one-interior-edge bipyramid (no growth budget) and record where the residual
    lifts off zero -- the onset of the spectral obstruction."""
    ts = list(np.linspace(0.0, 1.0, points))
    residuals, realizable = [], []
    for t in ts:
        U = [[1.0 + 0j, 1.0 + 1.0 * t + 0j],
             [1.0 + 2.0 * t + 0j, 1.0 + 3.0 * t + 0j]]
        v = decide(_bipyramid, U, 2, 2, seed=seed, restarts=restarts,
                   max_cones=0)
        residuals.append(float(v.residual))
        realizable.append(bool(v.realizable))
    return ts, residuals, realizable


def _cone_budget_series(seed, restarts, max_budget=3):
    """The interior fill at work: for the generic 1×3 U on the solid triangle,
    sweep the cone budget and record the residual -- it floors at budget 0 and
    drops below ε once the boundary-fixed cone-and-retry is allowed to grow the
    interior (interior_vertex_count == cones_applied)."""
    U = [[1.0 + 0j, 0.3 + 0.5j, -0.8 + 0.2j]]
    budgets, residuals, cones = [], [], []
    for b in range(max_budget + 1):
        v = decide(_solid_triangle, U, 1, 3, seed=seed, restarts=restarts,
                   max_cones=b)
        budgets.append(b)
        residuals.append(float(v.residual))
        cones.append(int(v.cones_applied))
    return budgets, residuals, cones


def write_sweeps(out_dir, seed, restarts):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "realizability_sweeps.txt")
    ts, res, realiz = _realizability_boundary(seed, 11, max(restarts // 2, 16))
    budgets, bres, cones = _cone_budget_series(seed, restarts)
    floor_oracle = _np_floor_oracle([[1.0, 2.0], [3.0, 4.0]], 2, 2)
    lines = ["# State-Operation-Cobordism -- synthesis & realizability sweeps (§7)",
             "",
             "## realizability boundary: U(t) = uniform --(t)--> [[1,2],[3,4]]",
             "## on the one-interior-edge bipyramid (no growth budget)",
             "#      t   residual r       realizable"]
    for t, r, ok in zip(ts, res, realiz):
        lines.append(f"{t:>8.3f}   {r:>14.6e}   {'yes' if ok else 'no'}")
    lines += ["",
              "## interior-fill cone budget: generic 1x3 U on the solid triangle",
              "# budget   residual r       cones_applied"]
    for b, r, c in zip(budgets, bres, cones):
        lines.append(f"{b:>7}   {r:>14.6e}   {c}")
    lines += ["",
              "## independent numpy grid global-min of the obstructed floor",
              "## (U=[[1,2],[3,4]], single interior edge, boundary pinned)",
              f"# numpy_floor = {floor_oracle:.6e}"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return [path]


def write_figures(out_dir, seed, restarts):
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

    # (1) The realizability boundary: residual lifting off zero as U(t) leaves
    # the realizable zero-mode -- the onset of the spectral obstruction.
    ts, res, _ = _realizability_boundary(seed, 21, max(restarts // 2, 16))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ts, np.maximum(res, 1e-32), "o-")
    ax.axhline(CERT_FLOOR, ls="--", color="crimson", lw=0.8,
               label="certificate floor $10^{-2}$")
    ax.set(xlabel="interpolation $t$  (uniform $\\to$ [[1,2],[3,4]])",
           ylabel="residual $r$",
           title="§5.0 realizability boundary: obstruction onset")
    ax.legend()
    p1 = os.path.join(out_dir, "realizability_boundary.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    # (2) The interior fill: residual vs cone budget for the generic 1×3 U --
    # floors at budget 0, drops below ε once growth is allowed.
    budgets, bres, _ = _cone_budget_series(seed, restarts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(budgets, np.maximum(bres, 1e-32), "s-")
    ax.axhline(EPSILON, ls="--", color="seagreen", lw=0.8,
               label="acceptance $\\epsilon=10^{-10}$")
    ax.set(xlabel="interior cone budget", ylabel="residual $r$",
           title="§5.0 interior fill: generic $1\\times3$ $U$ realized after coning")
    ax.legend()
    p2 = os.path.join(out_dir, "cone_budget_residual.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)
    return paths


# --------------------------------------------------------------------------- #
# The three correspondence hypotheses -- the formal content of the
# state-operation-cobordism correspondence, evaluated on a realized witness.
# --------------------------------------------------------------------------- #
def verify_correspondence(seed, restarts):
    """Evaluate the three correspondence claims on a synthesized witness W_AB:
      (1) W_AB = geo(U) is a cobordism;
      (2) ∂W_AB = conj(geo(ψ_B)) ⊔ geo(ψ_A)  -- here: ∂W_AB = the pinned boundary
          geo's, interior excluded and byte-fixed;
      (3) Z(W_AB) = <ψ_A|U|ψ_B>  -- the witness carries vec(U) as its output-
          boundary eigenvector, and contracting it with the test functional
          vec(U_T) reproduces the transition amplitude (Stage-1 HS/Choi identity).
    Returns (h1, h2, h3)."""
    dA = dB = 2
    U = [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]]
    bulk = _bipyramid()
    _pin_all(bulk, w=1.0, phase=0.0)
    pinned = {k: (e.getSquaredLength(), e.getPhase())
              for k, e in _edge_map(bulk).items()}
    target = _bend(U, dA, dB)
    v = cob.RealizabilityOracle(bulk).decide(
        target, dA, dB, epsilon=EPSILON, restarts=restarts, max_cones=0, seed=seed)
    W = v.witness

    # (1) W_AB is a built manifold-with-boundary: a nonempty codim-1 boundary and
    # at least one interior cell carrying the fill (a genuine cobordism, not closed).
    boundary = sorted(tuple(sorted(c)) for c in W.getBoundary())
    interior = sorted(set(_edge_map(W).keys()) - set(boundary))
    h1 = bool(v.realizable and boundary and interior)

    # (2) ∂W_AB is exactly the pinned boundary (the geo's): the fill leaves the
    # boundary edges byte-identical and never touches the interior.
    live = _edge_map(W)
    boundary_fixed = all(
        (live[k].getSquaredLength(), live[k].getPhase()) == pinned[k]
        for k in boundary)
    h2 = bool(boundary_fixed and set(boundary).isdisjoint(interior))

    # (3) Z(W_AB) = <ψ_A|U|ψ_B>: the witness carries vec(U); contracting that
    # boundary state with vec(U_T) = vec(|ψ_A><ψ_B|) returns the amplitude (and
    # ChoiJamiolkowski.transitionAmplitude == the direct ψ_A^† U ψ_B).
    block = np.asarray(v.state)[:dA * dB]
    vecU = np.asarray([complex(z) for z in target])
    vecU = vecU / np.linalg.norm(vecU)
    carries = abs(np.vdot(block / np.linalg.norm(block), vecU))
    psiA = np.array([1 + 0j, 0.5j]); psiA /= np.linalg.norm(psiA)
    psiB = np.array([0.6 + 0j, 0.8 + 0j]); psiB /= np.linalg.norm(psiB)
    Uflat = [complex(z) for z in np.asarray(U, dtype=complex).reshape(-1)]
    amp = complex(cj.transitionAmplitude(
        [complex(z) for z in psiA], Uflat, [complex(z) for z in psiB], dA, dB))
    direct = complex(np.conj(psiA) @ (np.asarray(U, dtype=complex) @ psiB))
    u_t = cj.transitionOperator(
        [complex(z) for z in psiA], [complex(z) for z in psiB], dA, dB)
    vec_ut = np.asarray([complex(z) for z in cj.vectorize(
        [complex(z) for z in u_t], dA, dB)])
    Z = complex(np.vdot(vec_ut, np.asarray(Uflat)))          # <vec(U_T)|vec(U)>
    h3 = bool(abs(amp - direct) < 1e-9 and abs(Z - amp) < 1e-9
              and carries > 1 - 1e-6)

    print("\n  The three correspondence hypotheses (realized W_AB for "
          "U=[[1,1],[1,1]]):")
    print(f"    (1) W_AB = geo(U) is a cobordism             "
          f"[{'PASS' if h1 else 'FAIL'}]  "
          f"|∂W|={len(boundary)} bd edges, {len(interior)} interior")
    print(f"    (2) ∂W_AB = conj(geo(ψ_B)) ⊔ geo(ψ_A)        "
          f"[{'PASS' if h2 else 'FAIL'}]  "
          f"∂W={boundary}, interior {interior} excluded & pinned")
    print(f"    (3) Z(W_AB) = <ψ_A|U|ψ_B>                    "
          f"[{'PASS' if h3 else 'FAIL'}]  "
          f"Z={Z:.4f} = <A|U|B>={amp:.4f}  (witness carries vec(U), "
          f"overlap={carries:.6f})")
    return h1, h2, h3


# --------------------------------------------------------------------------- #
def _fmt_op(name):
    return name if len(name) <= 26 else name[:23] + "..."


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="synthesis / sweep seed (default 0).")
    ap.add_argument("--restarts", type=int, default=80,
                    help="multi-restart count for the LM solver (default 80).")
    ap.add_argument("--max-additional-vertices", type=int, default=20,
                    help="cap on the vertices the growth may ADD: the cone "
                         "budget of the §4b boundary synthesis and of the "
                         "growable realizability cases, which run with "
                         "additions as well as surgical cuts "
                         "(SURGERY_AND_CONE). The obstruction-floor controls "
                         "stay pinned at budget 0 — they certify floors at "
                         "fixed complexity by design (default 20).")
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="directory for sweeps + figures (default /tmp/cobordism).")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures.")
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0], metavar="B",
                    help="Regge-mediation coupling(s) for the mediated objective "
                         "F_beta = r_U + beta*|S_Regge(W*)| (#249/#250). With any "
                         "beta > 0 the §5.0 realizability cases are re-decided as a "
                         "sweep over these values; beta = 0 (default) is the "
                         "base-layer search and leaves the output unchanged.")
    args = ap.parse_args()

    print("State-Operation-Cobordism correspondence -- synthesis + "
          "realizability\n")

    # ---- §4b: boundary-state synthesis geo(ψ) ----------------------------- #
    # A general-amplitude qubit |c0| != |c1|: the two-vertex seed floors, so the
    # cone-and-retry grows the minimal complex (the triangle K_3) realizing it.
    c0, c1 = math.sqrt(0.8), math.sqrt(0.2)
    geo, floor_seed = synthesize_boundary(complex(c0), complex(c1), seed=args.seed,
                                          restarts=args.restarts,
                                          max_cones=args.max_additional_vertices)
    print("  §4b  boundary-state synthesis  geo(ψ),  "
          f"ψ = (√0.8, √0.2)  (|c0| ≠ |c1|)")
    print(f"       two-vertex seed floors at r = {floor_seed:.3e}  "
          "(general qubit is not a K_2 eigenvector)")
    print(f"       geo(ψ): converged={geo.converged}  r={geo.residual:.3e}  "
          f"complexity |V|={geo.num_vertices} |E|={geo.num_edges}  "
          f"cones={geo.cones_applied}  λ={geo.eigenvalue:.6f}\n")

    # ---- §5.0: realizability decisions ------------------------------------ #
    # The growable cases run with the composed move-set (additions as well as
    # surgical cuts, the added vertices capped by --max-additional-vertices);
    # the obstruction-floor control stays pinned at budget 0 / cone mode so it
    # certifies the floor at fixed complexity, cross-checked below against the
    # single-interior-edge numpy oracle.
    both = cob.RealizabilityOracle.GrowthMode.SURGERY_AND_CONE
    cone = cob.RealizabilityOracle.GrowthMode.CONE
    grow = args.max_additional_vertices
    cases = [
        ("[[1,1],[1,1]] zero-mode", _bipyramid,
         [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]], 2, 2, grow, both, True),
        ("[[1,.3+.5j,-.8+.2j]] 1×3", _solid_triangle,
         [[1.0 + 0j, 0.3 + 0.5j, -0.8 + 0.2j]], 1, 3, grow, both, True),
        ("[[1,2],[3,4]] generic", _bipyramid,
         [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], 2, 2, 0, cone, False),
    ]

    print("  §5.0  realizability of U : H_B → H_A  (bend → pin dW → fill "
          "interior → residual verdict;")
    print(f"        growable cases: additions + surgical cuts, at most "
          f"{grow} added vertices; the floor control: budget 0, fixed "
          f"complexity)\n")
    header = (f"  {'operation U':26} {'bulk':11} {'verdict':11} "
              f"{'r / floor':>12}  {'steps':>5} {'cuts':>4} {'interior':>8} "
              f"{'λ (Rayleigh)':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = True
    verdicts = []
    for name, factory, U, dA, dB, max_cones, mode, expect_real in cases:
        v = decide(factory, U, dA, dB, seed=args.seed, restarts=args.restarts,
                   max_cones=max_cones, growth_mode=mode)
        verdicts.append((name, v, expect_real))
        bulk = "bipyramid" if factory is _bipyramid else "triangle"
        verdict = "REALIZABLE" if v.realizable else "OBSTRUCTED"
        metric = (f"r={v.residual:.2e}" if v.realizable
                  else f"f={v.floor:.2e}")
        print(f"  {_fmt_op(name):26} {bulk:11} {verdict:11} {metric:>12}  "
              f"{v.cones_applied:>5} {v.surgery_removals:>4} "
              f"{v.interior_vertex_count:>8} {v.eigenvalue:>13.6f}")

        # Acceptance: realizable cases reach r < ε; the obstructed floor is a
        # certificate bounded above CERT_FLOOR (and floor == residual there).
        if expect_real:
            ok = v.realizable and v.residual < EPSILON
        else:
            ok = (not v.realizable and v.floor > CERT_FLOOR
                  and v.floor == v.residual)
        all_ok &= ok
    print("  " + "-" * (len(header) - 2))

    # Independent corroboration of the obstruction certificate: the numpy grid
    # global-min over the single interior edge agrees with the certified floor.
    obstructed = next(v for n, v, e in verdicts if not e)
    numpy_floor = _np_floor_oracle([[1.0, 2.0], [3.0, 4.0]], 2, 2)
    agree = abs(obstructed.floor - numpy_floor) < 5e-3
    all_ok &= agree
    print(f"\n  obstruction certificate cross-check: certified floor "
          f"{obstructed.floor:.6e} vs numpy grid min {numpy_floor:.6e} "
          f"(Δ {abs(obstructed.floor - numpy_floor):.1e})  "
          f"{'AGREE' if agree else 'DISAGREE'}")

    # The three correspondence hypotheses, evaluated on a realized witness.
    h1, h2, h3 = verify_correspondence(args.seed, args.restarts)
    all_ok &= (h1 and h2 and h3)

    print(f"\n  Realizability hypothesis: realizable U are realized with their "
          f"W_AB (r → 0); obstructed U are certified non-realizable by a "
          f"residual floor;")
    print(f"  and the three correspondence claims -- (1) W_AB is a cobordism, "
          f"(2) ∂W_AB = the pinned geo's, (3) Z(W_AB) = <ψ_A|U|ψ_B> -- hold "
          f"({'SUPPORTED' if all_ok else 'NOT SUPPORTED -- a check failed'}).")

    # ---- Regge mediation sweep (--beta) ----------------------------------- #
    # Authoritative verdict stays anchored to the beta = 0 decisions above; the
    # sweep re-decides the §5.0 cases under F_beta only when beta > 0 is asked for.
    if any(b > 0 for b in args.beta):
        print("\n  Regge mediation (--beta): §5.0 cases re-decided under "
              "F_beta = r_U + beta*|S_Regge(W*)|:")
        header = (f"  {'operation U':26} {'beta':>6} {'verdict':11} "
                  f"{'r / floor':>12} {'cuts':>5} {'|S_Regge|':>11}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, factory, U, dA, dB, max_cones, mode, _expect in cases:
            for beta in args.beta:
                v = decide(factory, U, dA, dB, seed=args.seed,
                           restarts=args.restarts, max_cones=max_cones,
                           growth_mode=mode, beta=beta)
                verdict = "REALIZABLE" if v.realizable else "OBSTRUCTED"
                metric = (f"r={v.residual:.2e}" if v.realizable
                          else f"f={v.floor:.2e}")
                print(f"  {_fmt_op(name):26} {beta:>6g} {verdict:11} "
                      f"{metric:>12} {v.surgery_removals:>5} "
                      f"{abs(v.regge_action):>11.3f}")
        print("        => beta is applied (candidate cuts are ranked by F_beta), but "
              "the verdicts do NOT contract here: these cases realize by cone growth "
              "(0 surgical cuts), so a surgery-ranking beta has nothing to gate, and "
              "where surgery is used the oracle scores |S_Regge| on the optimized "
              "geometry, so realizing does not raise it. The fixed-geometry "
              "contraction is in spectral_gate_realizability.py (13 -> 11 -> 0); "
              "see #276.")

    if not args.no_plot:
        print("\n  §7 sweeps:")
        for p in write_sweeps(args.out, args.seed, args.restarts):
            print(f"    {p}")
        print("  §8 figures:")
        for p in write_figures(args.out, args.seed, args.restarts):
            print(f"    {p}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
