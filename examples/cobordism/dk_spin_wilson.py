# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Register-specific spin via the deficit-angle (spin-connection) Wilson loop (#477) — a
POST-HOC observable, never a loop condition.

The structural spin-½ (#483) lives in the DK fiber. The *register-specific* spin first
looked blocked: the spatial spin generators `Σ_ij` need a local frame, and a simplicial
complex has no global vielbein. **A Wilson loop removes that obstruction** — a holonomy is
frame-independent. The Regge **deficit angle `ε` is exactly the Lorentz rotation parallel
transport picks up around a hinge** (the spin-connection holonomy), and `WilsonLoop` with
`WilsonMode.DEFICIT_ANGLE` gives its **vector-rep** value, which in d=4 is

    W_vec = ((d-2) + 2 cos ε) / d = (1 + cos ε)/2 = cos²(ε/2).

Lifting the *same* holonomy to the **spinor (spin-½) rep** rotates a spinor by `ε/2` — the
double-cover half-angle — so its Wilson loop is

    W_spin = cos(ε/2) = √W_vec.

That `√` (spinor = square root of vector) **is** the Spin → SO double cover, i.e. the spin-½
signature, and it needs no frame. Read on the register-boundary hinges vs the bulk it gives
a frame-independent, register-specific spin holonomy.

Note: the skeleton must be materialized first (constructing a `ReggeSolver` does it), or the
`(d-2)`-hinges are not enumerable.
"""
import cmath

import numpy as np

import tessera as T


def _materialize(st):
    """Materialize the skeleton (hinges/facets) so `WilsonLoop` can enumerate hinges."""
    T.ReggeSolver(st, T.MatterConfiguration())


def spinor_wilson_loop(wl, hinge):
    """The spin-½ Wilson loop around `hinge`: the spin-connection holonomy in the spinor
    rep, `cos(ε/2) = √W_vec`, the half-angle of the Spin double cover. Frame-independent."""
    w_vec = wl.evaluateDeficitAngle(wl.hingeLoop(hinge)).value
    return cmath.sqrt(complex(w_vec))


def _hinges(st):
    return [s for s in st.getSimplices() if len(s.getVertices()) == 3]


def register_spin(st, holes):
    """Read the spin-½ Wilson loop on the **register-boundary** hinges (triangles whose
    vertices lie in a register hole) vs the **bulk** — the register-specific spin holonomy.
    A hole is a `(k+2)`-vertex tuple from `emergent_holes`. Frame-independent (a holonomy).
    Returns the per-group means/spreads and the register/bulk ratio."""
    _materialize(st)
    wl = T.WilsonLoop(st)
    tris = _hinges(st)
    hole_vsets = [set(h) for h in holes]
    reg, bulk = [], []
    for t in tris:
        vs = {v.getId() for v in t.getVertices()}
        (reg if any(vs <= hv for hv in hole_vsets) else bulk).append(t)
    rs = np.array([abs(spinor_wilson_loop(wl, t)) for t in reg]) if reg else np.array([])
    bs = np.array([abs(spinor_wilson_loop(wl, t)) for t in bulk]) if bulk else np.array([])
    return {
        "register_hinges": len(reg),
        "bulk_hinges": len(bulk),
        "register_spinor_W": float(rs.mean()) if rs.size else None,
        "register_spinor_W_std": float(rs.std()) if rs.size else None,
        "bulk_spinor_W": float(bs.mean()) if bs.size else None,
        "ratio": float(rs.mean() / (bs.mean() + 1e-30)) if rs.size and bs.size else None,
    }


def half_angle_residual(st):
    """Certify the spin-½ double cover on every hinge: `max |W_spin² − W_vec|`. → 0 means
    the spinor Wilson loop is exactly the square root of the vector one (the `ε/2` lift)."""
    _materialize(st)
    wl = T.WilsonLoop(st)
    worst = 0.0
    for t in _hinges(st):
        w_vec = wl.evaluateDeficitAngle(wl.hingeLoop(t)).value
        w_spin = spinor_wilson_loop(wl, t)
        worst = max(worst, abs(w_spin * w_spin - complex(w_vec)))
    return worst


if __name__ == "__main__":
    import importlib.util
    import os

    _here = os.path.dirname(__file__)
    _spec = importlib.util.spec_from_file_location(
        "eo", os.path.join(_here, "emergent_optimizer.py"))
    eo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(eo)
    import cmath as _c
    import math as _m
    w = _c.exp(2j * _m.pi / 3)
    for S in range(3, 40):
        host = eo.build_closed_s4(n_refine=20, seed=S % 997)
        opt = eo.EmergentOptimizer(host, [[1, w, w * w], [1, w * w, w]], [1, w, w * w],
                                   k=3, gamma=1.0, seed=S)
        sv = [v.getId() for v in host.getVertexList().toVector()][:2]
        opt.construct_inputs(sv, 12)
        opt.run_stage1(max_steps=30, n_candidates=8, patience=8)
        holes = eo.emergent_holes(opt.st, 3)
        if len(holes) >= 3:
            break
    print("converged betti", eo.betti(opt.st), "holes", len(holes))
    print("half-angle residual (spinor = sqrt(vector)):", half_angle_residual(opt.st))
    for k, v in register_spin(opt.st, holes).items():
        print(f"  {k}: {v}")
