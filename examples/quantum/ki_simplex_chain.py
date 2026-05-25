"""Four-system Koashi-Imoto simplex chain.

Builds a small simplicial complex out of four initial quantum
systems by chaining the QuantumSimplex KI factory:

    Round 1
    -------
        (A, B) ──► s_AB  with cell vertices {A, B, Σ_AB, A', B'}
        (C, D) ──► s_CD  with cell vertices {C, D, Σ_CD, C', D'}

    Round 2 — glue the two round-1 cells through their tail / core
    vertices:
        (A',     Σ_CD) ──► s_A'·ΣCD
        (Σ_AB,   C')   ──► s_ΣAB·C'

The two round-2 cells share the round-1 vertices Σ_AB, Σ_CD, A',
and C' with their respective round-1 cells, so the resulting
complex has four 4-simplices glued through these shared corners.

Round 1 uses ``fromSchmidtPurification`` so the (A, B) and (C, D)
edges carry the full Bell-pair mutual information; round 2 uses
``fromExplicitJoint`` with a product joint (no inherited
correlation across the round-2 inputs), which is fine — the point
of round 2 is the topological gluing, not new correlation. Edge
lengths come out as d_VR² (=0 on the AB edges of round 1 because
they sit at the simulation-wide ``iMax``, =+∞ on every product
edge).

The complex is written to ``examples/quantum/_out/ki_chain.png``.
"""

from __future__ import annotations

import math
import os

import numpy as np

from tessera import (
    Foliation,
    Metric,
    Signature,
    SignatureType,
    Spacetime,
    SpacetimeType,
)
from tessera.quantum import (
    QuantumSimplex,
    QuantumSimplexPosition as P,
    createQuantumVertex,
)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ki_chain.png")

    metric = Metric(True, Signature(4, SignatureType.Euclidean))
    st = Spacetime(metric, SpacetimeType.REGGE, 1.0, 1.0,
                   Foliation.NONE, None)
    i_max = 2.0 * math.log(2.0)

    # Four initial systems. Matched diagonal spectra so the Schmidt-
    # purification factory accepts (A, B) and (C, D).
    marg = np.diag([0.7, 0.3]).astype(complex)
    A = createQuantumVertex(st, marg)
    B = createQuantumVertex(st, marg.copy())
    C = createQuantumVertex(st, marg.copy())
    D = createQuantumVertex(st, marg.copy())

    # Round 1: (A, B) and (C, D) interact through Schmidt
    # purification, MI(A:B) = MI(C:D) = 2·H(0.7, 0.3) ≈ 1.22 nats.
    s_ab = QuantumSimplex.fromSchmidtPurification(st, A, B, i_max)
    s_cd = QuantumSimplex.fromSchmidtPurification(st, C, D, i_max)

    # Pull out the new vertices by canonical position.
    sigma_ab = s_ab.getVertices()[int(P.Sigma)]
    a_prime  = s_ab.getVertices()[int(P.APrime)]
    sigma_cd = s_cd.getVertices()[int(P.Sigma)]
    c_prime  = s_cd.getVertices()[int(P.APrime)]

    # Round 2: (A', Σ_CD) and (Σ_AB, C'). Round-1 marginals on
    # these vertices have mismatched spectra and dimensions
    # (1×4 = 4×1), so use ``fromExplicitJoint`` with a product
    # joint. The cells get built and glued onto round 1; the new
    # (X, Y) edges run at d_VR = +∞ because the joint is
    # uncorrelated.
    def product_joint(x, y):
        return np.kron(np.asarray(x.getState()),
                       np.asarray(y.getState()))

    QuantumSimplex.fromExplicitJoint(
        st, a_prime, sigma_cd,
        product_joint(a_prime, sigma_cd),
        i_max)
    QuantumSimplex.fromExplicitJoint(
        st, sigma_ab, c_prime,
        product_joint(sigma_ab, c_prime),
        i_max)

    n_simplices = len(st.getSimplices())
    print(f"Vertices: {st.getVertexCount()}  "
          f"Simplices: {n_simplices}  "
          f"(getSimplexCount = {st.getSimplexCount()}, "
          f"counts only CDT-classified top simplices)")

    # Force-directed layout + projection to 2D.
    st.save(out_path, panelSize=900, layoutIters=400)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
