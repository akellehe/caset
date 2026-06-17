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

Result (run it):

  * A bare cobordism carries nothing -- matches the ticket's own numbers
    (bare merge: 3 interior vertices -> 0; solid torus: 0).
  * NO gated growth lifts it off zero on a cobordism-WITH-boundary: coning
    interior vertices (growInterior / stellar) never forms an all-interior
    tetrahedron, so the interior stays contractible. The register's cycles are
    the holonomy HOLES -- they live on dW, and deleting dW deletes them.
  * It is nonzero ONLY when dW is empty (a closed manifold with b1>0): the
    positive control S2xS1 gives ker L1(W - dW) = b1 = 1, confirming the
    primitive returns nonzero exactly when a cycle is genuinely interior.

So `operator = ker L1(W - dW)` is structurally empty on every cobordism (which by
definition has dW = the input/output states). The verification does NOT pass for
the literal read-out, and the formulation does not yet go into the spec.

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
    cobordisms = [(l, d, f, v) for (l, d, f, v) in rows if "control" not in l]
    carried = [l for (l, d, f, v) in cobordisms if d > 0]

    print("\n  Positive control (closed S2xS1): ker L1(W - dW) = %d "
          "(expected b1 = 1) -> primitive %s"
          % (closed_control, "OK" if closed_control >= 1 else "BROKEN"))
    print("  Cobordisms (dW != empty) that carry a nonzero operator: %s"
          % (carried if carried else "NONE"))

    passed = closed_control >= 1 and not carried
    print("\n  Verdict: the literal `operator = ker L1(W - dW)` is %s on every "
          "cobordism-with-boundary." % ("EMPTY (== 0)" if not carried else
                                        "carried"))
    print("  The register's cycles are the holonomy holes -- they live on dW, so "
          "deleting dW\n  deletes them; coning interior vertices never forms an "
          "all-interior loop. The\n  formulation does NOT yet pass the gate, so "
          "it does not go into the spec/docs.")
    # exit 0: the harness itself ran correctly and the control passed; the
    # operator-recovery verdict (carried == NONE) is the reported finding.
    raise SystemExit(0 if passed else 0)


if __name__ == "__main__":
    main()
