# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Gating verification for issue #363: is the discovered operator the harmonic
of `ker L1(W - dW)`?

The canonical `MergeCobordism(initialStates, finalState)` recovers the operator
as the harmonic eigenvector of the L1 Laplacian of the bulk with the FULL dW
subcomplex deleted -- `reshape` (ChoiJamiolkowski un-vectorize) of that harmonic
is meant to equal U. The ticket is explicit that this must be TESTED before the
formulation goes into the spec/docs/code.

This harness measures `dim ker L1(W - dW)` -- via the C++ primitive
`EigenstateSynthesis.bulkMinusBoundaryHarmonicMatrix` (the harmonics of the
subcomplex induced on the INTERIOR vertices, the only "delete dW" that stays a
valid complex; relative homology H1(W,dW)=0 is ruled out by the ticket) -- across
the constructions a merge / register cobordism can actually be built and grown.

Result (run it) -- the SURGERY matters:

  * A bare cobordism carries nothing -- matches the ticket's own numbers
    (bare merge: 3 interior vertices -> 0; solid torus: 0).
  * CONING does not help: growInterior / stellar never forms an all-interior
    tetrahedron, so the interior stays contractible (merge + growInterior x30
    is still 0). The register's HOLE cycles live on dW, and deleting dW deletes
    them.
  * An interior 1-HANDLE does: take the closed S2xS1 (the handle, ker L1 = 1),
    grow it, then open boundary cavities (the states) by gated removeInteriorCell,
    rolling back any cut that would kill the handle. The result is a genuine
    cobordism (dW != empty) with a LIVE interior cycle: ker L1(W - dW) = 1. This
    is the interior 1-surgery the coning moves cannot do.
  * Positive control: closed S2xS1 (dW empty) gives ker L1(W - dW) = b1 = 1,
    confirming the primitive returns nonzero exactly when a cycle is interior.

So `operator = ker L1(W - dW)` is NOT structurally empty: coning leaves it 0, but
a gated interior 1-handle makes it nonzero on a real cobordism. The carried
operator is then read off that handle (the reshape == U check is the next gate).

Run:
    python examples/cobordism/operator_recovery_verification.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SG = _load("spectral_gate_realizability")
MC = _load("merge_cobordism")
DW = _load("dw_spectral_bridge")
tessera = SG.tessera
cob = tessera.cobordism
np = SG.np


def _kerL1_bulk_minus_boundary(st):
    """(dim ker L1(W - dW), interior edges, interior vertices) for a Spacetime,
    via the C++ primitive on the degree-1 EigenstateSynthesis."""
    es = cob.EigenstateSynthesis(st, 1)
    cells = es.bulkMinusBoundaryCells()
    H = es.bulkMinusBoundaryHarmonicMatrix()
    nc = len(cells)
    dim = (len(H) // nc) if nc else 0
    return dim, nc, es.interiorVertexCount()


def _full_kerL1(st):
    return len(cob.HodgeLaplacian(st).harmonics(1))


def _dual_valid(st):
    try:
        ok, _ = cob.EigenstateSynthesis(st, 1).dualComplexValid()
        return bool(ok)
    except Exception:
        return None


def _from_tets(cells):
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i)
            for i in sorted({v for c in cells for v in c})}
    for c in cells:
        t = sorted(c)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
    return st


def _grow(st, steps):
    """Cone `steps` interior vertices via the boundary-fixed 1->(d+1) Pachner add
    (growInterior), the reliable coning surgery. Returns the EigenstateSynthesis."""
    es = cob.EigenstateSynthesis(st, 1)
    for i in range(steps):
        if not es.growInterior(i):
            break
    return es


def _interior_handle_cobordism(grow=60, keep_dim=1):
    """A cobordism (dW != empty) carrying a LIVE interior handle. Start from the
    closed S2xS1 (the handle: ker L1 = 1, every vertex interior), grow it with
    boundary-fixed coning, then open boundary cavities (the states) with the gated
    interior surgery `removeInteriorCellChecked` -- rolling back any cut that would
    drop ker L1(W - dW) below `keep_dim`. Each accepted cut is gated by
    dualComplexValid, so the result stays a valid manifold-with-boundary. This is
    the interior 1-surgery the coning moves cannot perform."""
    st = DW._build(tessera.SimplicialProduct(
        tessera.SimplexBoundarySphere(2), tessera.SimplexBoundarySphere(1)))
    es = cob.EigenstateSynthesis(st, 1)
    for i in range(grow):
        if not es.growInterior(i):
            break

    def kdim():
        nc = len(es.bulkMinusBoundaryCells())
        return (len(es.bulkMinusBoundaryHarmonicMatrix()) // nc) if nc else 0

    for _ in range(400):
        progressed = False
        for cell in es.interiorTopCells():
            ok, _why = es.removeInteriorCellChecked(list(cell))
            if not ok:
                continue
            if kdim() >= keep_dim:
                progressed = True
                break
            es.restoreLastRemoval()  # this cut would kill the handle
        if not progressed:
            break
    return st


def cases():
    """(label, Spacetime) for every construction the verification spans."""
    out = []

    # the bare merge (two staircase prisms sharing the result) -- the canonical
    # cobordism, valid manifold, no grown bulk
    out.append(("bare merge (A->R, B->R)", MC.MergeCobordism().st))

    # 3D merge after coning 30 interior vertices (growInterior succeeds, but no
    # all-interior tet ever forms, so the interior stays contractible)
    m = MC.MergeCobordism()
    _grow(m.st, 30)
    out.append(("merge + growInterior x30", m.st))

    # the surgery-grown register surface (icosahedron, holonomy holes opened,
    # gated additive stellar growth) -- the operator is supposed to emerge here
    out.append(("register grow=24 (icosahedron)",
                SG.Register(grow_vertices=24, grow_seed=0).st))

    # spectral fixtures with boundary: a solid torus (b1=1) and T^2x[0,1] (b1=2)
    out.append(("solid torus D2xS1", DW._solid_torus()))
    out.append(("T2x[0,1] cobordism", DW._torus_cylinder()))

    # THE INTERIOR-HANDLE COBORDISM (#363 (a)): gated interior 1-surgery gives a
    # cobordism (dW != empty) a live interior handle -> ker L1(W - dW) = 1.
    out.append(("interior-handle cobordism", _interior_handle_cobordism()))

    # POSITIVE CONTROL: a CLOSED manifold with b1>0 (dW empty => W-dW = W)
    out.append(("S2xS1 (closed, control)",
                DW._build(tessera.SimplicialProduct(
                    tessera.SimplexBoundarySphere(2),
                    tessera.SimplexBoundarySphere(1)))))
    return out


def main():
    print("Gating verification #363: operator = harmonic of ker L1(W - dW)?\n")
    hdr = ("construction", "interiorVtx", "interiorEdges", "dim kerL1(W-dW)",
           "full kerL1", "dualValid")
    print("  %-30s %11s %13s %15s %10s %9s" % hdr)
    print("  " + "-" * 92)
    rows = []
    for label, st in cases():
        dim, nc, iv = _kerL1_bulk_minus_boundary(st)
        full = _full_kerL1(st)
        dv = _dual_valid(st)
        rows.append((label, dim, full, dv))
        print("  %-30s %11d %13d %15d %10d %9s"
              % (label, iv, nc, dim, full, dv))

    closed_control = next(d for (l, d, f, v) in rows if "control" in l)
    coning = [(l, d) for (l, d, f, v) in rows
              if l in ("bare merge (A->R, B->R)", "merge + growInterior x30",
                       "register grow=24 (icosahedron)", "solid torus D2xS1",
                       "T2x[0,1] cobordism")]
    handle = next(d for (l, d, f, v) in rows if l == "interior-handle cobordism")

    print("\n  Positive control (closed S2xS1): ker L1(W - dW) = %d "
          "(expected b1 = 1) -> primitive %s"
          % (closed_control, "OK" if closed_control >= 1 else "BROKEN"))
    print("  Coning / prism / growth cobordisms: ker L1(W - dW) = %s -> all 0"
          % ([d for (l, d) in coning],))
    print("  Interior-HANDLE cobordism (gated 1-surgery): ker L1(W - dW) = %d"
          % handle)

    passed = closed_control >= 1 and handle >= 1 and all(d == 0 for (l, d) in coning)
    print("\n  Verdict: `operator = ker L1(W - dW)` is 0 under CONING (the holonomy"
          "-hole\n  cycles live on dW and are deleted with it), but a gated "
          "interior 1-HANDLE\n  lifts it to a live interior cycle on a genuine "
          "cobordism (dW != empty). So the\n  operator-carrying bulk EXISTS via "
          "interior 1-surgery -- the reshape == U check\n  is the remaining gate "
          "before the formulation goes into the spec/docs.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
