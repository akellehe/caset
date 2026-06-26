# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Spin readout from the Dirac–Kähler spinor sector (#477) — a POST-HOC observable on an
emergent register, never a loop condition.

The DK fiber `Λ(ℝᵈ)` is `2ᵈ`-dimensional; for d=4 it is `16 = 4 (Dirac spinor) × 4 (taste)`.
The **taste** factor is the flavor index (the per-hole DK charge, T5 findings); this module
reads the **spinor** factor — the spin. The spin of a field is carried by the Clifford
rotation generators

    Σ_ij = ¼ [γ_i, γ_j]          (i, j spatial)

whose eigenvalues are `±½` exactly when the field is **spin-½** (a Dirac spinor sits in
`(½,0) ⊕ (0,½)`, so every `Σ_ij` has eigenvalues `±½`; a scalar would give `0`, a vector
`0, ±1`). This reads off the `DiracKahler.gammas` of any d=4 mesh — the spin-½ capacity is a
structural property of the Kähler–Atiyah construction, the spin analog of `multiplicity = 4`
giving it the taste capacity.

The register-**specific** spin *state* (which spin sector the converged register populates,
the analog of the per-hole flavor charge) needs the discrete spin operator promoted to the
full DK total space (the fiber↔cells map), which the form degrees of a generic simplicial
complex do not align point-by-point — that is left as a follow-up rather than hacked here.
"""
import numpy as np

import tessera as T

cob = T.cobordism

# In the d = 1+3 convention, axis 0 is time and 1,2,3 are the spatial axes whose rotations
# generate spin; the proton is spin-½ under those spatial rotations.
_SPATIAL = (1, 2, 3)


def _gamma_matrices(dk, lorentzian=False):
    d = dk.gammaDimension()
    return [np.asarray(g, dtype=complex).reshape(d, d) for g in dk.gammas(lorentzian)]


def clifford_residual(dk, lorentzian=False):
    """Max deviation of the gammas from the Clifford algebra `{γ^a, γ^b} = 2 η^ab I` —
    `→ 0` certifies the generators close (so an `Σ_ij` eigenvalue of `±½` is meaningful)."""
    g = _gamma_matrices(dk, lorentzian)
    n = len(g)
    eta = np.asarray(dk.signature(lorentzian), dtype=float).reshape(n, n)
    eye = np.eye(dk.gammaDimension())
    worst = 0.0
    for a in range(n):
        for b in range(n):
            anti = g[a] @ g[b] + g[b] @ g[a]
            worst = max(worst, float(np.max(np.abs(anti - 2.0 * eta[a, b] * eye))))
    return worst


def spin_generators(dk, lorentzian=False):
    """The spatial Clifford rotation generators `Σ_ij = ¼[γ_i, γ_j]`, keyed by `(i, j)`."""
    g = _gamma_matrices(dk, lorentzian)
    return {(i, j): 0.25 * (g[i] @ g[j] - g[j] @ g[i])
            for a, i in enumerate(_SPATIAL) for j in _SPATIAL[a + 1:]}


def spin_eigenvalue_magnitudes(dk, lorentzian=False, decimals=6):
    """For each spatial rotation plane, the distinct `|eigenvalue|`s of `Σ_ij` — `{0.5}`
    for a spin-½ field."""
    out = {}
    for plane, sigma in spin_generators(dk, lorentzian).items():
        ev = np.linalg.eigvals(sigma)
        out[plane] = sorted({round(abs(e), decimals) for e in ev})
    return out


def is_spin_half(dk, lorentzian=False, tol=1e-6):
    """`True` iff every spatial rotation generator `Σ_ij` has eigenvalues exactly `±½` —
    the spin-½ signature (and nothing else: no `0`, no `±1`)."""
    return all(len(mags) == 1 and abs(mags[0] - 0.5) < tol
               for mags in spin_eigenvalue_magnitudes(dk, lorentzian).values())


def spin_report(st, lorentzian=False):
    """Read the spin-½ signature off the Dirac–Kähler structure of `st` (post-hoc)."""
    dk = cob.DiracKahler(st)
    return {
        "mesh_dim": dk.meshDimension(),
        "gamma_dim": dk.gammaDimension(),
        "taste_multiplicity": dk.multiplicity(),
        "clifford_residual": clifford_residual(dk, lorentzian),
        "spin_eigenvalue_magnitudes": spin_eigenvalue_magnitudes(dk, lorentzian),
        "spin_half": is_spin_half(dk, lorentzian),
    }


if __name__ == "__main__":
    import importlib.util
    import os

    _here = os.path.dirname(__file__)
    _spec = importlib.util.spec_from_file_location(
        "eo", os.path.join(_here, "emergent_optimizer.py"))
    eo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(eo)
    host = eo.build_closed_s4(n_refine=12, seed=0)
    rep = spin_report(host)
    for k, v in rep.items():
        print(f"{k}: {v}")
