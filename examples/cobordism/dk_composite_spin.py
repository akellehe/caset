# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Composite proton spin — total ½ (proton) vs 3/2 (Δ) from the three register holes (#485).

The *constituent* fermion is spin-½ by construction (#483, #477). This module reads the
**composite** total spin of the emergent bound state — the quantum number that actually
distinguishes a proton (J=½) from a Δ (J=3/2): three spin-½'s combine as
`2⊗2⊗2 = 2 ⊕ 2 ⊕ 4`, i.e. total `J² ∈ {¾ (spin-½), 15/4 (spin-3/2)}`.

A simplicial complex has no global frame, so per-hole spin polarizations aren't directly
comparable. The frame-free fix (as in #477) is **holonomy**: relate the three holes' spinor
frames by the spin-connection **Wilson line** (open-path holonomy), then combine.

## Construction (this file builds (1)–(2); (3)–(5) are the remaining geometry build)
1. **Spinor holonomy element** — `spinor_holonomy(ε, Σ)`: the `Spin(d)` group element
   `exp(ε·Σ)` of a rotation by `ε` in the plane whose generator is `Σ = Σ_ij` (from
   `dk_spin_readout.spin_generators`). Its eigenvalues are `e^{±iε/2}` — the `ε/2`
   half-angle of the Spin→SO double cover (spin-½). **DONE + tested.**
2. **Half-angle / double-cover certificate** — `is_double_cover(...)`: the eigenvalue
   phases are `±ε/2`, not `±ε`. **DONE + tested.**
3. **Per-hinge geometric generator** — from `Simplex.gramMatrix()`: embed the top cell
   (Cholesky of the Gram matrix), get the hinge's 2-plane and its orthogonal-complement
   normal 2-plane, map that bivector onto the `Σ_ij` basis. `deficitAngle()` gives `ε`.
   **TODO** (the heavy piece — the discrete spin connection).
4. **Inter-hole Wilson line** — order-compose the per-hinge holonomies along a dual path
   between two register holes (reuse `WilsonLoop.dualLatticeLoop`/`geodesicLoop` for the
   path). Transports one hole's spinor frame into another's. **TODO.**
5. **Combine → J²** — transport the three holes' carried-rep spinors to a common frame via
   (4), form the (anti)symmetrized 3-spinor bound state, and read its total-spin Casimir
   `J²` → `¾` (proton) or `15/4` (Δ). **TODO.** Honest negative allowed (indefinite mix).
"""
import numpy as np
import scipy.linalg


def spinor_holonomy(eps, sigma):
    """The `Spin(d)` holonomy of a rotation by angle `eps` in the plane with spin generator
    `sigma` (a `Σ_ij = ¼[γ_i,γ_j]` from `dk_spin_readout.spin_generators`, eigenvalues
    `±i/2`): `exp(eps · sigma)`. Eigenvalues are `e^{±i eps/2}` — the `eps/2` half-angle of
    the double cover. `eps` may be complex (Lorentzian deficit)."""
    return scipy.linalg.expm(complex(eps) * np.asarray(sigma, dtype=complex))


def holonomy_phases(eps, sigma):
    """The distinct eigenvalue phases of `spinor_holonomy(eps, sigma)`, sorted."""
    ev = np.linalg.eigvals(spinor_holonomy(eps, sigma))
    return sorted({round(float(np.angle(e)), 8) for e in ev})


def is_double_cover(eps, sigma, tol=1e-7):
    """True iff the holonomy phases are `±eps/2` (the spin-½ double cover) — not `±eps`
    (vector). Certifies the spinor lift is genuinely spin-½."""
    phases = set(np.round(holonomy_phases(eps, sigma), 7))
    want = {round(float(np.angle(np.exp(1j * eps / 2))), 7),
            round(float(np.angle(np.exp(-1j * eps / 2))), 7)}
    return want <= phases


# --- (3)-(5): the remaining geometry build (per-hinge generator from gramMatrix, the
# --- inter-hole Wilson line, and the J² combination) live here; see the module docstring.


if __name__ == "__main__":
    import importlib.util
    import os
    import math

    _here = os.path.dirname(__file__)
    _spec = importlib.util.spec_from_file_location(
        "sr", os.path.join(_here, "dk_spin_readout.py"))
    sr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sr)
    eo_spec = importlib.util.spec_from_file_location(
        "eo", os.path.join(_here, "emergent_optimizer.py"))
    eo = importlib.util.module_from_spec(eo_spec)
    eo_spec.loader.exec_module(eo)

    host = eo.build_closed_s4(n_refine=12, seed=0)
    dk = eo.cob.DiracKahler(host)
    sigma = sr.spin_generators(dk)[(1, 2)]            # a Σ_ij spin generator
    for eps in (0.3, 1.0, 2.0, math.pi / 2):
        ph = holonomy_phases(eps, sigma)
        print(f"ε={eps:.3f}: holonomy phases {ph}  (±ε/2 = ±{eps/2:.3f})  "
              f"double-cover={is_double_cover(eps, sigma)}")
