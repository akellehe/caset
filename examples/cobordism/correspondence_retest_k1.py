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

"""k=1 re-test of the State-Operation-Cobordism hypotheses with the free engine.

The capstone for #202: re-runs the three correspondence hypotheses at the Hodge
qubit degree (k=1, harmonic 1-forms) using the free-interior-connectivity
realizability engine (#201), extended to k=1 (triangle/2-simplex candidates), and
compares the realizable set across three engines on the same boundary qubit:

  * **free-connectivity (k=1)** -- decideHarmonic(growth_mode=FREE_CONNECTIVITY);
  * **cone-only**               -- decideHarmonic(growth_mode=CONE), the #176 path;
  * **discrete-DW S_3 shadow**  -- the GL(2,Z_2) holonomy permutations the
    Dijkgraaf-Witten state sum realizes on Z(T^2)=C^4, rebuilt from
    Cobordism.twistedCylinder (#194).

The bulk is the solid torus W = D^2 x S^1 (a 3-manifold with boundary T^2); the
boundary qubit is ker L_1(T^2) (b_1 = 2): the **longitude** (the cycle the solid
torus carries, surviving in H_1(W)) and the **meridian** (the cycle that bounds a
disk in W, dying in H_1(W)).

The headline finding (the wall #202's stop-condition anticipates):

  At k=1 the **additive** free-connectivity engine cannot do spectrally-active,
  boundary-fixed growth. ChainComplex builds only the top cells' downward closure,
  so a dangling edge/triangle is DROPPED from L_k (spectrally inert); an additive
  top-cell attach is boundary-locked. The one boundary-fixed move that enriches
  L_k is the stellar Pachner subdivision (the cone). So the k=1 free search
  enumerates + logs the triangle candidates, finds them inert, and falls back to
  the cone -- **free == cone at k=1**. The realizable set is the topological image
  H_1(Sigma) -> H_1(W) (the longitude; not the meridian), and free connectivity
  does NOT enlarge it -- the opposite of the k=0 fan win (#201). The 2-cell IS
  spectrally real at k=1 (a filled vs hollow triangle shifts the k=1 residual); the
  additive engine simply cannot realize one under the pinned boundary.

  * **H1** -- W_AB = geo(U) is a cobordism: the longitude is realizable
    (r -> 0); the meridian and the raw ker-L_1(Sigma) basis float. Free == cone on
    every target; neither realizes what the discrete theory cannot.
  * **H2** -- dW = the two states, held bit-exact: the boundary edges are byte-
    identical through the whole (Pachner) growth.
  * **H3** -- Z(W_AB) = <psi_A|U|psi_B>: the realized longitude witness is a genuine
    L_1(W) harmonic (lambda ~ 0) whose boundary block reproduces the target form
    (overlap ~ 1); the Choi-Jamiolkowski transition-amplitude identity holds
    independently (the algebraic anchor reused from the v0.3 report).

Run:  python examples/cobordism/correspondence_retest_k1.py
      (--help for options; the raw comparison table defaults to /tmp/cobordism
      and is NOT committed -- attach it to the issue/PR to pin a result.)

Exit status is 0 iff the verdicts match: the longitude is realizable and the
meridian/basis float, identically under free and cone; the DW shadow is the
6-element S_3; H2 holds; and the H3 longitude overlap and the Choi identity hold.
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
cj = tessera.quantum.ChoiJamiolkowski
FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY
CONE = cob.RealizabilityOracle.GrowthMode.CONE

EPSILON = 1e-9
CERT_FLOOR = 1e-2          # a floored target sits above this (certified obstructed)
DIM = 4                    # dim Z(T^2) = 2^{b_1(T^2)}


# --------------------------------------------------------------------------- #
# Fixtures (the #176 / #194 idiom).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)


def _solid_torus():
    return _build(tessera.SimplicialProduct(tessera.SolidSimplex(2), _circle()))


def _torus():
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _pin_uniform(st, w=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _pinned_solid_torus():
    W = _solid_torus()
    _pin_uniform(W)
    return W


def _cvec(v):
    return [complex(z) for z in v]


def _Cochain(simplices, coeffs):
    return cob.Cochain(1, simplices, np.asarray(coeffs, dtype=complex))


def _boundary_snapshot(st):
    return {(min(a, b), max(a, b)): (e.getSquaredLength(), e.getPhase())
            for e in st.getEdgeList().toVector()
            for a, b in [(e.getSource().getId(), e.getTarget().getId())]}


# --------------------------------------------------------------------------- #
# The two distinguished boundary harmonics (the #176 construction).
# --------------------------------------------------------------------------- #
def _longitude_and_meridian(W, space):
    sig_simpl = space.harmonics()[0].simplices()
    bulk_h = cob.HodgeLaplacian(W).harmonics(1)[0]            # b_1(W) = 1
    restriction = _Cochain(
        sig_simpl, [complex(bulk_h.amplitudeFor(list(e))) for e in sig_simpl])
    prepared = space.prepare(restriction)
    longitude = prepared.readout()
    coords = np.array([complex(prepared.generatorAmplitude(i)) for i in range(2)])
    coords = coords / np.linalg.norm(coords)
    harmonics = np.column_stack([np.asarray(h.coeffs()) for h in space.harmonics()])
    meridian = _Cochain(sig_simpl, harmonics @ np.array([coords[1], -coords[0]]))
    return longitude, meridian


# --------------------------------------------------------------------------- #
# The discrete-DW S_3 shadow (rebuilt from twistedCylinder, #194).
# --------------------------------------------------------------------------- #
def _seven_vertex_torus():
    sig = tessera.Signature(2, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, None)
    verts = [st.createVertex(i) for i in range(7)]
    triangles = sorted({tuple(sorted(((i) % 7, (i + step) % 7, (i + 3) % 7)))
                        for i in range(7) for step in (1, 2)})
    for tri in triangles:
        st.createSimplex([verts[i] for i in tri])
    return st


def _twisted_map(surface, phi):
    twisted = cob.Cobordism.twistedCylinder(surface, phi)
    return np.asarray(cob.DijkgraafWitten(twisted, cob.Cocycle.Trivial).map()).real


def _dw_s3_shadow():
    """The realizable DW image: <swap, 3-cycle> closed, == the 6 GL(2,Z_2) perms."""
    swap = _twisted_map(
        _build(tessera.SimplicialProduct(_circle(), _circle())),
        [v * 3 + u for u in range(3) for v in range(3)])
    three_cycle = _twisted_map(_seven_vertex_torus(), [(2 * i) % 7 for i in range(7)])

    def key(m):
        return tuple(np.round(m.real).astype(int).reshape(-1))

    group = {key(np.eye(DIM)): np.eye(DIM)}
    frontier = [np.eye(DIM)]
    while frontier:
        element = frontier.pop()
        for generator in (swap, three_cycle):
            product = generator @ element
            if key(product) not in group:
                group[key(product)] = product
                frontier.append(product)
    return list(group.values())


# --------------------------------------------------------------------------- #
# H1: realizability of a boundary harmonic under free vs cone.
# --------------------------------------------------------------------------- #
def _decide(target, mode, max_cones, seed=1, restarts=8):
    W = _pinned_solid_torus()
    kwargs = dict(epsilon=EPSILON, restarts=restarts, max_cones=max_cones, seed=seed)
    if mode is FREE:
        kwargs.update(growth_mode=FREE, connectivity_candidates=8)
    else:
        kwargs.update(growth_mode=CONE)
    return cob.RealizabilityOracle(W).decideHarmonic(target, **kwargs)


def h1_realizable_set():
    """The realizable set on the boundary qubit, free vs cone (seed complex, so the
    verdict is exact + reproducible -- growth is non-deterministic, #201 AddMove
    global counter, but the floored verdicts hold regardless)."""
    space = cob.BoundaryStateSpace(_torus())
    longitude, meridian = _longitude_and_meridian(_pinned_solid_torus(), space)
    battery = [("longitude (carried)", longitude),
               ("meridian (bounds a disk)", meridian),
               ("ker L_1 basis #0", space.harmonics()[0]),
               ("ker L_1 basis #1", space.harmonics()[1])]
    rows = []
    for name, target in battery:
        free = _decide(target, FREE, max_cones=0)
        cone = _decide(target, CONE, max_cones=0)
        rows.append({
            "target": name,
            "free_realizable": bool(free.realizable),
            "free_residual": float(free.residual),
            "cone_realizable": bool(cone.realizable),
            "cone_residual": float(cone.residual),
            "triangle_candidates_at_growth":
                int(_decide(target, FREE, max_cones=1).triangle_candidates),
            "agree": (bool(free.realizable) == bool(cone.realizable)
                      and free.residual == cone.residual),
        })
    return rows, (longitude, meridian)


# --------------------------------------------------------------------------- #
# H2: the boundary is the two states, held bit-exact through growth.
# --------------------------------------------------------------------------- #
def h2_boundary_invariance(target):
    W = _pinned_solid_torus()
    before = _boundary_snapshot(W)
    v = cob.RealizabilityOracle(W).decideHarmonic(
        target, epsilon=EPSILON, restarts=8, max_cones=1, seed=2,
        growth_mode=FREE, connectivity_candidates=8)
    after = _boundary_snapshot(W)
    es = cob.EigenstateSynthesis(W, 1)
    boundary_keys = {(min(k), max(k)) for k in es.boundaryEdges()}
    bit_exact = all(after[k] == before[k] for k in boundary_keys)
    grew = v.cones_applied >= 1 and len(v.state) == 27 + 4 * v.cones_applied
    return bool(bit_exact), bool(grew), v


# --------------------------------------------------------------------------- #
# H3: Z(W_AB) = <psi_A|U|psi_B> on a realized witness.
# --------------------------------------------------------------------------- #
def h3_correspondence(longitude):
    """The realized longitude witness reproduces the target form (the spectral
    Z(W)), and the Choi-Jamiolkowski transition-amplitude identity holds (the
    algebraic anchor, bulk-independent)."""
    W = _pinned_solid_torus()
    v = cob.RealizabilityOracle(W).decideHarmonic(
        longitude, epsilon=EPSILON, restarts=8, max_cones=0, seed=1)
    state = np.asarray(v.state)
    cells = [tuple(c) for c in cob.EigenstateSynthesis(W, 1).cellSimplices()]
    index = {c: i for i, c in enumerate(cells)}
    target_on_W = np.zeros(len(cells), dtype=complex)
    for c, amp in zip(longitude.simplices(), np.asarray(longitude.coeffs())):
        target_on_W[index[tuple(c)]] = amp
    overlap = abs(np.vdot(state / np.linalg.norm(state),
                          target_on_W / np.linalg.norm(target_on_W)))

    # The Choi identity (reused from the v0.3 report): the transition amplitude
    # <psi_A|U|psi_B> equals the direct contraction and <vec(U_T)|vec(U)>.
    dA = dB = 2
    psiA = np.array([1 + 0j, 0.5j]); psiA /= np.linalg.norm(psiA)
    psiB = np.array([0.6 + 0j, 0.8 + 0j]); psiB /= np.linalg.norm(psiB)
    U = np.array([[1.0 + 0j, 0.0], [0.0, 1.0 + 0j]])
    Uflat = [complex(z) for z in U.reshape(-1)]
    amp = complex(cj.transitionAmplitude(_cvec(psiA), Uflat, _cvec(psiB), dA, dB))
    direct = complex(np.conj(psiA) @ (U @ psiB))
    u_t = cj.transitionOperator(_cvec(psiA), _cvec(psiB), dA, dB)
    vec_ut = np.asarray(_cvec(cj.vectorize(_cvec(u_t), dA, dB)))
    Z = complex(np.vdot(vec_ut, np.asarray(Uflat)))
    choi_ok = abs(amp - direct) < 1e-9 and abs(Z - amp) < 1e-9
    return float(overlap), float(v.eigenvalue), bool(v.realizable), bool(choi_ok)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="dir for the raw comparison table (default /tmp/cobordism; "
                         "NOT committed).")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print("State-Operation-Cobordism re-test at k=1 (the Hodge qubit) with the "
          "free-connectivity engine\n")

    ok = True

    # ---- DW S_3 shadow ---------------------------------------------------- #
    shadow = _dw_s3_shadow()
    print(f"  discrete-DW S_3 shadow: <swap, 3-cycle> closes to {len(shadow)} "
          f"maps on Z(T^2)=C^4 (the GL(2,Z_2) holonomy permutations)")
    ok &= (len(shadow) == 6)

    # ---- H1: realizable set, free vs cone vs DW shadow -------------------- #
    rows, (longitude, meridian) = h1_realizable_set()
    print("\n  H1  realizable set on the boundary qubit (free-k1 vs cone; the DW "
          "shadow realizes operations, not states):")
    header = (f"      {'target':28} {'free':>6} {'r (free)':>11} {'cone':>6} "
              f"{'r (cone)':>11} {'tri_cand':>8} {'agree':>6}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for r in rows:
        print(f"      {r['target']:28} "
              f"{'YES' if r['free_realizable'] else 'no':>6} "
              f"{r['free_residual']:>11.3e} "
              f"{'YES' if r['cone_realizable'] else 'no':>6} "
              f"{r['cone_residual']:>11.3e} "
              f"{r['triangle_candidates_at_growth']:>8} "
              f"{'yes' if r['agree'] else 'NO':>6}")
        ok &= r["agree"]
    # The longitude is the one realizable target; the rest float -- under both.
    longitude_row = rows[0]
    ok &= (longitude_row["free_realizable"] and longitude_row["cone_realizable"])
    ok &= all(not r["free_realizable"] for r in rows[1:])
    # The longitude is realized at the seed (no growth, so no candidates proposed);
    # every FLOORED target triggers a growth step that proposes triangle candidates.
    ok &= longitude_row["triangle_candidates_at_growth"] == 0
    ok &= all(r["triangle_candidates_at_growth"] > 0 for r in rows[1:])
    print("\n      Headline: free == cone on every target (additive growth is "
          "spectrally inert at k=1; it falls back to the cone Pachner). The "
          "realizable set is the carried longitude -- the topological image "
          "H_1(Sigma)->H_1(W) -- NOT enlarged by free connectivity (the opposite of "
          "the k=0 fan win). Triangles ARE proposed (tri_cand>0) but inert.")

    # ---- H2: boundary bit-exact through growth ---------------------------- #
    bit_exact, grew, _ = h2_boundary_invariance(meridian)
    print(f"\n  H2  dW = the two states, held bit-exact through growth: "
          f"{'PASS' if bit_exact else 'FAIL'}  "
          f"(Pachner growth applied: {grew}; every boundary edge byte-identical)")
    ok &= bit_exact and grew

    # ---- H3: Z(W) = <psi_A|U|psi_B> on the realized longitude ------------- #
    overlap, eigenvalue, realizable, choi_ok = h3_correspondence(longitude)
    h3 = realizable and overlap > 1 - 1e-6 and abs(eigenvalue) < 1e-6 and choi_ok
    print(f"\n  H3  Z(W_AB) = <psi_A|U|psi_B> on the realized longitude: "
          f"{'PASS' if h3 else 'FAIL'}")
    print(f"      witness is an L_1(W) harmonic (lambda={eigenvalue:.2e} ~ 0), its "
          f"boundary block reproduces the target form (overlap={overlap:.6f}); "
          f"the Choi transition-amplitude identity holds ({choi_ok}).")
    ok &= h3

    # ---- raw table (PR artifact, not committed) --------------------------- #
    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "correspondence_retest_k1.txt")
        with open(path, "w") as handle:
            handle.write("# k=1 State-Operation-Cobordism re-test (free vs cone vs "
                         "DW S_3 shadow)\n\n")
            handle.write(f"DW S_3 shadow size: {len(shadow)} (== 6 GL(2,Z_2) perms)\n\n")
            handle.write("H1 realizable set (free-k1 vs cone):\n")
            for r in rows:
                handle.write(f"  {r['target']:28} free={'Y' if r['free_realizable'] else 'n'} "
                             f"r={r['free_residual']:.3e}  cone={'Y' if r['cone_realizable'] else 'n'} "
                             f"r={r['cone_residual']:.3e}  tri_cand={r['triangle_candidates_at_growth']} "
                             f"agree={r['agree']}\n")
            handle.write(f"\nH2 boundary bit-exact through growth: {bit_exact} "
                         f"(grew: {grew})\n")
            handle.write(f"H3 longitude overlap={overlap:.6f} lambda={eigenvalue:.3e} "
                         f"choi_identity={choi_ok}\n")
        print(f"\n  raw table (PR artifact, not committed): {path}")

    print(f"\n  Verdict: {'SUPPORTED' if ok else 'NOT SUPPORTED -- a check failed'} "
          f"-- at k=1 the free engine realizes EXACTLY the carried harmonics that "
          f"cone growth does; additive free connectivity does not enlarge the "
          f"realizable set, and H2/H3 hold on the realized longitude.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
