# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Read the proton's mass and radius off the relaxed emergent geometry (#480, part of #410).

The emergent proton of #489 is **built**, never hand-placed: a single co-optimized
`MultiCobordism` pair-creation event (`dk_joint_spin.build_converged_proton`) interacts
three neutral q-q̄ pairs into a [proton, antiproton] pair, and the proton's quark register
emerges as the three holes of the relaxed output block. This module reads two **honest,
dimensionless** observables straight off that relaxed geometry — post-hoc, nothing
fabricated, and explicitly NOT physical fm / MeV:

  * **radius** ``r = sqrt(mean(l²))`` over the relaxed **spacelike** edges (the edges whose
    complex squared length has positive real part). Reported for the proton block
    sub-complex (`rd['_sub']`) and, as context, for the full relaxed complex (`opt.st`),
    together with the spacelike / timelike edge counts.

  * **mass**, two handles, both read off the block sub-complex with the relaxed metric:
      A — ``|dualReggeAction|`` of `tessera.ReggeSolver(sub, MatterConfiguration())`. The
          Lorentzian action is genuinely **complex**, so the real and imaginary parts are
          reported separately (the Im part is real physics — spacelike-hinge boosts — and is
          kept, never dropped); ``|.|`` is the magnitude.
      B — a curvature-concentration proxy: ``Σ_hinge |deficitAngle| · |dualVolume|`` over the
          block's hinges (the (d−2)-cells), the matter's bending of the dual geometry. A free
          cross-check; reported when the bound deficit/dual-volume API is usable.

The dimensionless product ``r·|m|`` is reported for both handles and compared, honestly, to
the crude prior trajectory (~73) and the rough target (~4.0) — we report where it lands, not
where we wish it would.

Run:  python examples/cobordism/proton_mass_radius.py
"""

import importlib.util
import math
import os
import sys

import numpy as np

import tessera

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dj = _load("dk_joint_spin")


# ----- radius: RMS spacelike edge length off the relaxed metric -----
def radius_rms(st):
    """``r = sqrt(mean(l²))`` over the relaxed **spacelike** edges (Re(l²) > 0).

    Returns ``(r, n_spacelike, n_timelike)``; ``n_timelike`` counts the edges whose squared
    length has non-positive real part (timelike / null). ``r = 0`` if there is no spacelike
    edge (degenerate block)."""
    l2 = [e.getSquaredLength().real for e in st.getEdgeList().toVector()]
    spacelike = [x for x in l2 if x > 0.0]
    r = float(np.mean(spacelike)) ** 0.5 if spacelike else 0.0
    return r, len(spacelike), len(l2) - len(spacelike)


# ----- mass A: the full complex dual Regge action -----
def dual_action_mass(sub):
    """Mass handle A: the complex ``dualReggeAction`` of the block sub-complex.

    Returns ``(re, im, mag)`` — the Lorentzian action is genuinely complex, so the real and
    imaginary parts are both reported (the Im part is kept, never dropped) alongside its
    magnitude ``|S|``."""
    action = tessera.ReggeSolver(sub, tessera.MatterConfiguration()).dualReggeAction()
    return float(action.real), float(action.imag), float(abs(action))


# ----- mass B: curvature-concentration / shell-deficit proxy -----
def curvature_concentration_mass(sub):
    """Mass handle B: ``Σ_hinge |deficitAngle| · |dualVolume|`` over the block's hinges.

    Hinges are the (d−2)-cells (one below the facets); each carries a Lorentzian deficit
    angle (complex) and a circumcentric dual volume. The deficit-weighted dual content is a
    coordinate-free measure of how strongly the matter concentrates curvature. Returns
    ``(mass_b, n_hinges)``; ``(0.0, 0)`` if the dimension is too low to have hinges."""
    sub.materializeFacets()
    simplices = sub.getSimplices()
    if not simplices:
        return 0.0, 0
    top_vertices = max(len(s.getVertices()) for s in simplices)
    hinge_vertices = top_vertices - 2  # (d-2)-cell when top cell has d+1 vertices
    if hinge_vertices < 1:
        return 0.0, 0
    total = 0.0
    n = 0
    for s in simplices:
        if len(s.getVertices()) != hinge_vertices:
            continue
        deficit = abs(s.lorentzianDeficitAngle())
        dual = abs(s.dualVolume())
        if math.isfinite(deficit) and math.isfinite(dual):
            total += deficit * dual
            n += 1
    return float(total), n


# ----- the full read off a converged proton -----
def read_mass_radius(opt, rd):
    """Read radius + both mass handles off the relaxed geometry of a converged proton.

    `opt` / `rd` are the ``(opt, read)`` from `dk_joint_spin.build_converged_proton`;
    `rd['_sub']` is the proton block sub-complex (relaxed metric), `opt.st` the full complex.
    Returns a flat dict of finite dimensionless proxies (see module docstring)."""
    sub = rd["_sub"]
    r_sub, n_sp_sub, n_tl_sub = radius_rms(sub)
    r_full, n_sp_full, n_tl_full = radius_rms(opt.st)
    re, im, mag = dual_action_mass(sub)
    mass_b, n_hinges = curvature_concentration_mass(sub)
    return {
        "radius": r_sub,
        "n_spacelike": n_sp_sub,
        "n_timelike": n_tl_sub,
        "radius_full": r_full,
        "n_spacelike_full": n_sp_full,
        "n_timelike_full": n_tl_full,
        "mass_re": re,
        "mass_im": im,
        "mass_abs": mag,
        "mass_b": mass_b,
        "n_hinges": n_hinges,
        "rm_a": r_sub * mag,
        "rm_b": r_sub * mass_b,
        "block_residual": rd["block_residual"],
    }


def measure(seeds=range(5, 25), max_residual=0.6, n_refine=18,
            stage1_steps=60, stage2_iters=20):
    """Build a converged emergent proton and read its mass + radius. Returns ``(out, seed)``
    or ``None`` if no proton converges in the seed range (honest negative)."""
    found = dj.build_converged_proton(
        seeds=seeds, max_residual=max_residual, n_refine=n_refine,
        stage1_steps=stage1_steps, stage2_iters=stage2_iters)
    if not found:
        return None
    opt, rd, seed = found
    return read_mass_radius(opt, rd), seed


def main():
    print("The proton's mass & radius, read off the relaxed emergent geometry (#480)\n")
    result = measure()
    if not result:
        print("No converged 3-hole proton block in the seed range (honest negative).")
        return
    o, seed = result
    print(f"converged proton (seed {seed}): block_residual = {o['block_residual']:.3e}"
          f"  (singlet carried)\n")
    print("radius  r = sqrt(mean(l²)) over the relaxed spacelike edges:")
    print(f"  proton block : r = {o['radius']:.4f}   "
          f"({o['n_spacelike']} spacelike, {o['n_timelike']} timelike/null edges)")
    print(f"  full complex : r = {o['radius_full']:.4f}   "
          f"({o['n_spacelike_full']} spacelike, {o['n_timelike_full']} timelike/null edges)")
    print("\nmass A — the full complex dual Regge action (Im kept; it is real physics):")
    print(f"  Re S = {o['mass_re']:.4f}   Im S = {o['mass_im']:.4f}   |S| = {o['mass_abs']:.4f}")
    print("mass B — curvature concentration  Σ_hinge |deficit|·|dualVolume|:")
    print(f"  mass_B = {o['mass_b']:.4f}   ({o['n_hinges']} hinges)")
    print("\ndimensionless r·|m|  (crude prior ~73; rough target ~4.0):")
    print(f"  r·|S|   (A) = {o['rm_a']:.4f}")
    print(f"  r·mass_B (B) = {o['rm_b']:.4f}")
    print("\n(honest: these are dimensionless geometric proxies, NOT physical fm / MeV.)")


if __name__ == "__main__":
    main()
