# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Re-test the realizable gate set with the bulk topology LOOSENED (b_1 emergent).

The pinned construction (`realizable_image_sweep.py`) reads a 2-qubit operation
U on Z(T^2)=C^4 through the Dijkgraaf-Witten functor on a *twisted cylinder* whose
topology is fixed, and finds the realizable image is exactly S_3 = GL(2,Z_2): the
six holonomy-class permutations fixing the trivial class. Those are the operations
that PERMUTE the holonomy classes without superposing them. Superposition-creating
gates (Hadamard) and off-lattice phase/entanglers (CZ, T, S, iSWAP, sqrt-SWAP)
floor (gap_to_S3 ~ 1-2).

The emergent-bulk work (`emergent_bulk_realizability.py`) then *loosened* the
topology: it stopped pinning the cobordism and let a b_1=1 hole develop via the
boundary-fixed surgery remove move. That hole is the homology that carries a
superposed / entangled boundary state. The question this script answers: does the
loosened construction enlarge the realizable *gate* set -- do the
superposition/entangling gates that floored under the pinned DW image now realize
once the hole can develop?

It tests every gate two ways and reports the residuals, NOT a hoped-for answer:

  (A) OPERATION level. Bend each gate U to its Choi state vec(U) (length d_A d_B =
      16) and ask the surgery oracle to carry it as a bulk harmonic with b_1 free:
      `RealizabilityOracle.decide(vec(U), 4, 4, growth_mode=SURGERY,
      harmonic=True)` on a filled-disk bulk (b_1=0) whose interior core triangle the
      surgery search may remove (opening b_1:0->1). Realizable iff the harmonic
      residual r=||L psi||^2 is driven below REALIZE.

  (B) STATE level. The same boundary 1-cycle the emergent-bulk experiment uses --
      the meridian carried on both boundary circles -- on the disk (b_1=0) vs the
      annulus (b_1=1), via `decideHarmonic`. This is the superposed *state* the
      Hadamard-type gates would create, tested directly as a homology class.

What the residuals say (reproduced below; exit 0 iff the verdicts hold):

  * (A) EVERY gate floors -- the six S_3 controls (identity included) AND every
    superposition/entangling gate, r ~ 0.38-0.41, none below REALIZE. The surgery
    search opens b_1:0->1 for some gates (whenever a removal marginally lowers the
    residual) but the residual floors regardless, so b_1 development is DECOUPLED
    from realizability. The reason is structural: at the Choi-vec degree k=0 the
    harmonic kernel ker L_0 is the constants (dimension b_0=1) on any connected
    bulk, so opening a b_1 handle cannot enlarge it -- the hole is spectrally inert
    for an operation target. The eigenvalue-agnostic mode (harmonic=False) is
    under-constrained: it "realizes" e.g. H(x)H as a NON-harmonic eigenvector
    (lambda ~ 2.9), which is not a topological realization.

  * (B) The superposed meridian STATE floors on the disk (b_1=0, r~0.45) and
    REALIZES on the annulus (b_1=1, r~7e-8, lambda->0). The hole carries the
    superposition -- as a state.

Headline: the b_1 hole does NOT expand the realizable *gate* (operation) set. S_3
stays the pinned DW operation image; the loosened construction's expansion is in
the realizable *state* space (superposed boundary 1-cycles), not the gate set. The
engine realizes superposed/entangled STATES, not superposition/entangling GATES.

Run:  python examples/cobordism/loosened_gate_retest.py
      (--help for options; the raw table defaults to /tmp/cobordism and is NOT
      committed -- attach it to the issue/PR to pin a result.)
"""

from __future__ import annotations

# Honor the 10-CPU cap before numpy / the C++ ext pull in a BLAS.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

import argparse  # noqa: E402
import cmath     # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism
DijkgraafWitten = cob.DijkgraafWitten
Cocycle = cob.Cocycle
Cobordism = cob.Cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

DIM = 4                 # dim Z(T^2) = 2^{b_1(T^2)} = 4
PINNED_TOL = 1e-7       # pinned-DW realizable := gap_to_S3 < PINNED_TOL
REALIZE = 1e-3          # loosened realizable := harmonic residual < REALIZE
CERT_FLOOR = 1e-2       # certified obstructed := residual floors above this
RESTARTS = 32
GROW_STEPS = 3
N_OUT = 16              # the output-boundary support carries vec(U) (d_A d_B = 16)
_INNER = [16, 17, 18]   # the interior core triangle the surgery search may remove


# --------------------------------------------------------------------------- #
# The pinned realizable image S_3 = GL(2,Z_2), built from twisted cylinders --
# the #193/#194 idiom, inlined so this script stands alone (gap_to_S3 is the
# pinned-DW operation metric the old sweep scored against).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _product_torus():
    circle = tessera.SimplexBoundarySphere(1)
    return _build(tessera.SimplicialProduct(circle, circle))


_SEVEN_VERTEX_TRIANGLES = sorted({
    tuple(sorted(((i) % 7, (i + step) % 7, (i + 3) % 7)))
    for i in range(7) for step in (1, 2)})


def _seven_vertex_torus():
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    verts = [st.createVertex(i) for i in range(7)]
    for tri in _SEVEN_VERTEX_TRIANGLES:
        st.createSimplex([verts[i] for i in tri])
    return st


def _twisted_map(surface, phi):
    return np.asarray(DijkgraafWitten(Cobordism.twistedCylinder(surface, phi),
                                      Cocycle.Trivial).map()).real


def _s3_image():
    """<DW(swap), DW(3-cycle)> -- the 6-element S_3 holonomy-permutation group."""
    swap = _twisted_map(_product_torus(), [v * 3 + u for u in range(3)
                                           for v in range(3)])
    three = _twisted_map(_seven_vertex_torus(), [(2 * i) % 7 for i in range(7)])

    def key(m):
        return tuple(np.round(m).astype(int).reshape(-1))

    group = {key(np.eye(DIM)): np.eye(DIM)}
    frontier = [np.eye(DIM)]
    while frontier:
        element = frontier.pop()
        for generator in (swap, three):
            product = np.round(generator @ element)
            if key(product) not in group:
                group[key(product)] = product
                frontier.append(product)
    assert len(group) == 6, f"S_3 closure must be 6, got {len(group)}"
    return list(group.values())


def gap_to_s3(U, image):
    """min over the 6 realizable DW maps g of ||U - g||_F (the pinned metric)."""
    return float(min(np.linalg.norm(np.asarray(U) - g) for g in image))


# --------------------------------------------------------------------------- #
# Gate-content classifiers (well-defined properties of the 4x4 matrix). Pinned-DW
# realizability (gap_to_S3 < tol) is exactly holonomy-permutation membership, so it
# is read off the gap directly rather than re-derived here.
# --------------------------------------------------------------------------- #
def creates_superposition(U, tol=1e-9):
    """A computational-basis state is mapped off the basis (some column is not a
    single nonzero up to phase): the gate creates superposition."""
    mag = np.abs(np.asarray(U))
    return any(np.count_nonzero(mag[:, c] > tol) != 1 for c in range(DIM))


def operator_schmidt_rank(U, tol=1e-9):
    """1 iff U factors as a one-qubit tensor product (local); >1 iff entangling."""
    reshaped = np.asarray(U).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    # Operator-Schmidt singular values via the C++ SVD primitive.  The
    # realignment above is the operator bipartition (distinct from
    # ChoiJamiolkowski's vec(U) Choi bipartition), so only the singular-value
    # computation is shared; the threshold count stays here.
    s = np.asarray(tessera.quantum.ChoiJamiolkowski.singularValues(
        [complex(x) for x in reshaped.ravel()], 4, 4))
    return int(np.sum(s > tol * max(s[0], 1.0)))


# --------------------------------------------------------------------------- #
# (A) the operation-level loosened bulk: a filled triangulated disk (b_1=0) whose
# rim (vertices 0..15) is the pinned output-boundary support carrying vec(U), with
# an interior core triangle {16,17,18} the surgery search may remove (b_1:0->1).
# Built generically from a face list -- no pinned torus / solid torus anywhere.
# --------------------------------------------------------------------------- #
def _annulus_core(filled=True):
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in list(range(N_OUT)) + _INNER}

    def tri(a, b, c):
        t = sorted((a, b, c))
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])

    sector = [k * 3 // N_OUT for k in range(N_OUT)]      # rim vertex -> inner vertex
    for k in range(N_OUT):
        nxt = (k + 1) % N_OUT
        tri(k, nxt, _INNER[sector[k]])
        if sector[nxt] != sector[k]:
            tri(nxt, _INNER[sector[k]], _INNER[sector[nxt]])
    if filled:
        tri(16, 17, 18)
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _b1(st):
    return int(cob.ChainComplex.fromSpacetime(st).bettiNumbers()[1])


def _decide_operation(U, *, harmonic, seed=1):
    """decide(vec(U)) with the surgery move-set and b_1 free (the loosened test)."""
    st = _annulus_core(filled=True)
    before = _b1(st)
    flat = [complex(z) for z in np.asarray(U, dtype=complex).reshape(-1)]
    v = cob.RealizabilityOracle(st).decide(
        flat, DIM, DIM, epsilon=REALIZE, restarts=RESTARTS, max_cones=GROW_STEPS,
        seed=seed, growth_mode=SURGERY, connectivity_candidates=8,
        harmonic=harmonic)
    return v, before, _b1(st)


# --------------------------------------------------------------------------- #
# (B) the state-level loosened test: the superposed meridian on the two boundary
# circles, disk (b_1=0) vs annulus (b_1=1) -- the #196 octahedron idiom.
# --------------------------------------------------------------------------- #
_OCT = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
        (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 1, 4)]
_CYCLE_A, _CYCLE_B = [(0, 1), (0, 2), (1, 2)], [(3, 4), (3, 5), (4, 5)]


def _surface(faces):
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _without(*drop):
    gone = {tuple(sorted(f)) for f in drop}
    return [f for f in _OCT if tuple(sorted(f)) not in gone]


def _meridian():
    annulus = _surface(_without((0, 1, 2), (3, 4, 5)))
    h = cob.HodgeLaplacian(annulus).harmonics(1)[0]
    edges = _CYCLE_A + _CYCLE_B
    vals = [complex(h.amplitudeFor(list(e))) for e in edges]
    return cob.Cochain(1, edges, np.asarray(vals, dtype=complex))


def _decide_state(filling, target):
    return cob.RealizabilityOracle(filling).decideHarmonic(
        target, epsilon=1e-7, restarts=RESTARTS, max_cones=0, seed=1,
        growth_mode=SURGERY, connectivity_candidates=8, harmonic=True)


# --------------------------------------------------------------------------- #
# The gate battery: the S_3 controls + the superposition / phase / entangling
# families the pinned image floored on.
# --------------------------------------------------------------------------- #
def _gates():
    h2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    i2 = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    t1 = np.diag([1, cmath.exp(1j * np.pi / 4)]).astype(complex)
    s1 = np.diag([1, 1j]).astype(complex)

    def perm(p):
        m = np.zeros((DIM, DIM), dtype=complex)
        for r, c in enumerate(p):
            m[r, c] = 1.0
        return m

    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    rcnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                     dtype=complex)
    iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                     dtype=complex)
    swap = perm((0, 2, 1, 3))

    def root(U):
        w, vec = np.linalg.eig(U)
        return (vec * np.sqrt(w)) @ np.linalg.inv(vec)

    return [
        ("Identity", np.eye(DIM, dtype=complex), "S3 control"),
        ("SWAP", swap, "S3 control"),
        ("CNOT", cnot, "S3 control"),
        ("reversed-CNOT", rcnot, "S3 control"),
        ("3-cycle (0231)", perm((0, 2, 3, 1)), "S3 control"),
        ("3-cycle (0312)", perm((0, 3, 1, 2)), "S3 control"),
        ("DCNOT", cnot @ rcnot, "S3 control"),
        ("H(x)I", np.kron(h2, i2), "superposition"),
        ("I(x)H", np.kron(i2, h2), "superposition"),
        ("H(x)H", np.kron(h2, h2), "superposition"),
        ("sqrt-SWAP", root(swap), "superposition"),
        ("sqrt-iSWAP", root(iswap), "superposition"),
        ("CZ", np.diag([1, 1, 1, -1]).astype(complex), "phase/entangler"),
        ("CPHASE(pi/4)", np.diag([1, 1, 1, cmath.exp(1j * np.pi / 4)]).astype(complex),
         "phase/entangler"),
        ("T(x)I", np.kron(t1, i2), "phase"),
        ("S(x)I", np.kron(s1, i2), "phase"),
        ("iSWAP", iswap, "phase/entangler"),
        ("X(x)X", np.kron(x, x), "Pauli perm"),
        ("Z(x)Z", np.kron(z, z), "diagonal sign"),
    ]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="dir for the raw table (default /tmp/cobordism; NOT committed).")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print("Re-testing the realizable gate set with the topology LOOSENED "
          "(b_1 emergent via surgery)\n")

    image = _s3_image()
    gates = _gates()

    # ---- (A) operation level: decide(vec(U)) under surgery, b_1 free --------- #
    print("  (A) OPERATION level -- bend U to vec(U), carry as a bulk harmonic with "
          "b_1 free:")
    header = (f"      {'gate':16} {'family':16} {'gap_to_S3':>9} {'pinDW':>6} "
              f"{'sup':>4} {'ent':>4} {'r (loosened)':>13} {'real':>5} "
              f"{'b_1':>6} {'rm':>3}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    rows = []
    op_ok = True
    for label, U, family in gates:
        gap = gap_to_s3(U, image)
        pinned = gap < PINNED_TOL
        sup = creates_superposition(U)
        ent = operator_schmidt_rank(U) > 1
        v, b_before, b_after = _decide_operation(U, harmonic=True)
        realized = v.residual < REALIZE
        op_ok &= not realized                  # the headline: nothing realizes
        rows.append({"gate": label, "family": family, "gap_to_S3": gap,
                     "pinned_dw_realizable": bool(pinned),
                     "creates_superposition": bool(sup),
                     "entangling": bool(ent),
                     "loosened_residual": float(v.residual),
                     "loosened_realizable": bool(realized),
                     "b1_before": b_before, "b1_after": b_after,
                     "surgery_removals": int(v.surgery_removals)})
        print(f"      {label:16} {family:16} {gap:>9.3f} "
              f"{('yes' if pinned else 'no'):>6} {('Y' if sup else '-'):>4} "
              f"{('Y' if ent else '-'):>4} {v.residual:>13.2e} "
              f"{('YES' if realized else 'floor'):>5} "
              f"{str(b_before) + '->' + str(b_after):>6} {v.surgery_removals:>3}")

    print("        => EVERY gate floors at the operation level -- the six S_3 "
          "controls (identity included) AND every superposition/entangling gate. "
          "Surgery opens b_1:0->1 for some, but the residual floors regardless: "
          "b_1 development is decoupled from realizability (ker L_0 = the constants, "
          "dim b_0=1, so a b_1 handle cannot host an operation target).")

    # The eigenvalue-agnostic mode is under-constrained: it accepts a non-harmonic
    # eigenvector, so it is NOT a topological realization. H(x)H is the witness.
    hxh = next(U for n, U, _ in gates if n == "H(x)H")
    v_eig, _, _ = _decide_operation(hxh, harmonic=False)
    print(f"        (eigenvalue-agnostic harmonic=False on H(x)H: r={v_eig.residual:.2e} "
          f"at lambda={v_eig.eigenvalue:.2f} -- a non-harmonic eigenvector, not a "
          f"topological realization.)")

    # ---- (B) state level: the superposed meridian, disk vs annulus ---------- #
    target = _meridian()
    disk = _surface(_without((0, 1, 2)))
    annulus = _surface(_without((0, 1, 2), (3, 4, 5)))
    vd = _decide_state(disk, target)
    va = _decide_state(annulus, target)
    print("\n  (B) STATE level -- the superposed meridian 1-cycle "
          "(the state a Hadamard creates), carried by H_1:")
    print(f"      disk    (b_1={_b1(disk)}):  r={vd.residual:.2e}  "
          f"{'realizes' if vd.residual < REALIZE else 'floors'}")
    print(f"      annulus (b_1={_b1(annulus)}):  r={va.residual:.2e}  "
          f"lambda={va.eigenvalue:.2e}  "
          f"{'REALIZES' if va.residual < REALIZE else 'floors'}")
    print("        => the superposed STATE floors at b_1=0 and realizes at b_1=1: "
          "the hole carries the superposition -- as a state, not as a gate.")

    state_ok = (vd.residual > CERT_FLOOR and va.residual < REALIZE)
    # Sanity on the pinned controls: the S_3 set realizes in DW, the superposition
    # set floors (the result the loosened test re-frames, not overturns).
    pinned_ok = all(r["pinned_dw_realizable"] for r in rows
                    if r["family"] == "S3 control")
    pinned_ok &= all(not r["pinned_dw_realizable"] for r in rows
                     if r["family"] == "superposition")

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "loosened_gate_retest.json")
        with open(path, "w") as handle:
            json.dump({"operation_level": rows,
                       "state_level": {"disk_residual": float(vd.residual),
                                       "annulus_residual": float(va.residual),
                                       "annulus_eigenvalue": float(va.eigenvalue)}},
                      handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = op_ok and state_ok and pinned_ok
    print("\n  Verdict: " + (
        "SUPPORTED -- the b_1 hole does NOT enlarge the realizable gate set: every "
        "gate floors as an operation (S_3 stays the pinned DW image), while the "
        "superposed meridian STATE realizes once surgery opens b_1=1. The loosened "
        "construction expands the realizable STATE space, not the gate set."
        if ok else
        "NOT SUPPORTED -- a verdict departed from the residuals; inspect the table."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
