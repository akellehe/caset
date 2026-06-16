# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Analytic gradient for the genuine per-edge backreaction relaxation.

The relaxation minimizes the matter-regulated free energy

    G(l^2) = Re S(l^2) + kappa * E(l^2) + lambda * |Im S(l^2)|

over the variable edge squared-lengths l^2 (result-surface spatial + timelike
bulk; the input states psi_A, psi_B fixed). A finite-difference gradient is both
an approximation (noisy -> the optimizer thrashes) and ~N_var G-evaluations per
gradient step (the per-edge eigendecomposition factor). This module computes
grad G EXACTLY and cheaply. NO physics change -- the same G; only its derivative,
computed analytically.

The gradient splits into two exact pieces:

1.  **Regge part -- the Schlafli identity.** For S = sum_e L_e * eps_e (3D Regge:
    hinges are edges, hinge volume = edge length, eps_e = deficit), the Schlafli
    identity sum_f L_f d(eps_f) = 0 collapses the derivative to

        dS/dL_e = eps_e          (the deficit on edge e, complex/Sorkin),

    which we already compute (`Simplex.lorentzianDeficitAngle`). Chain rule to the
    squared-length variable: dS/d(l^2_e) = eps_e / (2 L_e). This gives d(Re S) and
    d(Im S) together; d|Im S|/d(l^2_e) = sign(Im S) * d(Im S)/d(l^2_e).
    NOTE: tessera's action is the *dual* (Sorkin) action; this module VERIFIES the
    identity against finite differences and derives any dual-specific terms.

2.  **Matter part -- eigenvector perturbation theory.** E = <h, W1 h> with the
    carried representative h = c.H, H the harmonics (near-zero eigenvectors of the
    metric L1). dE/d(l^2_e) needs:
      * dW1/d(l^2_e): simplex volumes (Cayley-Menger) -- closed form, local.
      * dPi/d(l^2_e): the derivative of the **null-space projector** Pi = sum_i
        v_i v_i^dagger. Using the projector (basis-invariant) avoids the
        degenerate-ker-L1 blow-up: its derivative only divides by (0 - lambda_j)
        for j OUTSIDE the null space (nonzero). All from ONE eigendecomposition of
        L1 (the spectrum: eigenvalues + eigenvectors), plus the local dL1/d(l^2_e).
      * dL1/d(l^2_e): the metric Hodge Laplacian's dependence on l^2 is through the
        weights; computed analytically (or verified) from the assembly + the volume
        derivatives.

The full analytic grad G is VERIFIED against finite differences (match ~1e-6) on a
sample of edges before it is trusted -- finite differences become the test, not
the method.

This module is the gradient + the clean per-edge relaxation (live harmonics, no
binning, unconstrained with null->raise / overflow->warn, full complex deficit,
BFS graph-distance readout, L1-only energy) and the kappa sweep W*(kappa).

STATUS: scaffold. The construction/energy machinery and the two gradient pieces
are implemented incrementally on this branch (#336), each verified against finite
differences as it lands.
"""

from __future__ import annotations

# --- pieces, implemented + FD-verified incrementally on this branch ---------- #
# [ ] deep-merge construction + per-edge classification (ported, self-contained)
# [ ] exact #312 energy / dual Sorkin action (live, un-memoized)
# [ ] Regge gradient via Schlafli (dS/dl^2 = deficit) -- verify vs FD
# [ ] dW1/dl^2 via Cayley-Menger volume derivatives
# [ ] dL1/dl^2 (analytic from the assembly, or verified)
# [ ] dE/dl^2 via null-space-projector perturbation theory -- verify vs FD
# [ ] grad G assembled + FD-verified end to end (~1e-6)
# [ ] unconstrained relaxation with the analytic gradient; level 1 -> level 2
# [ ] kappa sweep W*(kappa); converged interior dual + curvature(d) profiles
