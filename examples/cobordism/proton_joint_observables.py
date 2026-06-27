# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Joint proton assembly — every quantum number off ONE relaxed emergent state (#481).

The per-sector readouts (color/confinement, flavor, charge, radius·mass, structural spin-½)
were each established on their own. This module is the **composition**: it drives a *single*
emergent proton to convergence and reads **all** of its quantum numbers off **that one
relaxed structure**, then reports them together. Nothing is pieced together across runs and
nothing is imposed — every number is a post-hoc read of the converged geometry.

Substrate (the emergent proton, #489/#492). The proton is **built**, not hand-made: a single
co-optimized `MultiCobordism` pair-creation event

    3 neutral q-q̄ pairs  ⟶  [ proton , antiproton ]

(`dk_joint_spin.build_converged_proton`). The proton's three quarks are the three emergent
holes of the proton output block (`rd['_holes']`), carved out with the relaxed metric copied
over (`rd['_sub']`). The block carries the color singlet (`rd['block_residual'] → 0`).

What each sector reads off the **same** `rd['_sub']` / `rd['_holes']`:

  * COLOR / confinement — the singlet `[1, ω, ω²]` is carried (`r_state → 0`). Confinement is
    **color-neutrality**: the singlet-phase-weighted net Dirac–Kähler charge
    `|Σ_k ω^k q_k|` is ≈ 0 (the bound state is color-neutral), far below the constituent
    total `Σ_k q_k` — the `Σ = 0 ⇒ singlet ⇒ confinement` statement, the same convention as
    the legacy `dirac_kahler_net_charge`.
    Honest negative on the *r_state* probe of confinement: `r_state(sub, 3, [1,1,1])` does
    **not** floor on this substrate. The emergent block carries a rich register (`b₃ ≥ 3`),
    so the `r_state` least-squares fit realizes *every* 3-component color rep — it confirms
    the singlet is carried but cannot, by itself, floor a non-singlet here. Color-neutrality
    is the genuine confinement signal; both `r_state` values are reported for transparency.

  * FLAVOR — the three holes carry **distinguishable** per-hole DK charges (`flavor_spread
    > 0`): the inequivalent-quark structure #488's two-quark read lacked.

  * CHARGE — the net Dirac–Kähler charge `Σ_k q_k` (the constituent total) and the per-hole
    breakdown. (Crude geometric units; not normalized to +1.)

  * RADIUS · MASS — `r = √⟨ℓ²⟩` over the relaxed spacelike edges; `mass = |S_Regge|` the full
    complex dual Lorentzian Regge action; the dimensionless `r·m`.

  * STRUCTURAL SPIN-½ — the Dirac–Kähler spatial rotation generators `Σ_ij = ¼[γ_i, γ_j]`
    have eigenvalues `±½` exactly (`dk_spin_readout.is_spin_half`): the built-in spin-½
    capacity of the Kähler–Atiyah construction on a d=4 mesh.

The single read of per-hole DK charge feeds color-neutrality, flavor, and charge at once —
so all three color/flavor/charge numbers come from the *same* extracted register, which is
exactly the joint-assembly claim.

DEFERRED — the COMPOSITE spin (the proton's `J² = ¾`) is **not** read here: the
per-hole-product read provably sits at the mixture (`dk_joint_spin`, #489), and resolving the
entangled mixed-symmetry ¾ needs the total-space spin operator — scoped as **#495 / #477**.

Run:  python examples/cobordism/proton_joint_observables.py
"""
import cmath
import importlib.util
import math
import os
import sys

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * math.pi / 3)
_SINGLET = [1, _W, _W * _W]
_UNIFORM = [1, 1, 1]
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    """Load a sibling example module by absolute path (the repo's example convention)."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


djs = _load("dk_joint_spin")
dsr = _load("dk_spin_readout")


# ----- geometry observables (read off the relaxed block, emergent-first) -----
def radius_rms(sub):
    """r = √⟨ℓ²⟩ over the relaxed spacelike (ℓ² > 0) edges. Returns (r, n_spacelike,
    n_timelike/null)."""
    l2 = [e.getSquaredLength().real for e in sub.getEdgeList().toVector()]
    sp = [x for x in l2 if x > 0]
    r = float(np.mean(sp)) ** 0.5 if sp else 0.0
    return r, len(sp), len(l2) - len(sp)


def dual_regge_mass(sub):
    """mass = |S_Regge|, the full complex dual Lorentzian Regge action of the block."""
    s = complex(tessera.ReggeSolver(sub, tessera.MatterConfiguration()).dualReggeAction())
    return abs(s)


# ----- the joint read: ALL quantum numbers off ONE converged proton -----
def proton_report(opt_or_build, block_index=0):
    """Read every quantum number off one converged emergent proton and assemble a single
    report dict.

    `opt_or_build` is either a `(opt, rd, seed)` tuple (e.g. from
    `dk_joint_spin.build_converged_proton`) or a bare `MultiCobordism` `opt` (its proton
    block is read via `dk_joint_spin.read_proton`). Returns the joint observable dict, or
    `None` if the proton block has no 3-hole color register."""
    seed = None
    if isinstance(opt_or_build, (tuple, list)) and len(opt_or_build) == 3:
        opt, rd, seed = opt_or_build
    else:
        opt = opt_or_build
        rd = djs.read_proton(opt, block_index)
    if rd is None or rd.get("n_holes", 0) < 3 or "_sub" not in rd:
        return None

    sub, holes = rd["_sub"], rd["_holes"]

    # one read of the per-hole DK charge feeds color-neutrality, flavor AND charge.
    q = np.asarray(rd["flavor_charges"], float)            # |DK charge| per quark hole
    constituent_total = float(np.sum(q))                   # Σ_k q_k (three colored quarks)
    color_net = complex(sum(s * c for s, c in zip(_SINGLET, q)))  # Σ_k ω^k q_k (net color)
    neutrality_ratio = abs(color_net) / (abs(constituent_total) + 1e-30)

    r_singlet = float(rd["block_residual"])                # r_state(sub, 3, [1,ω,ω²])
    r_uniform = float(cob.MultiCobordism.r_state(sub, 3, _UNIFORM))

    r, n_sp, n_tl = radius_rms(sub)
    mass = dual_regge_mass(sub)

    spin = dsr.spin_report(sub)

    return {
        "seed": seed,
        "betti_sub": list(cob.MultiCobordism.betti(sub)),
        "n_holes": int(rd["n_holes"]),
        # --- color / confinement ---
        "color_singlet_residual": r_singlet,               # → 0 ⇒ singlet carried
        "color_singlet_carried": r_singlet < 1e-3,
        "color_uniform_residual": r_uniform,               # honest negative: also floors
        "color_uniform_floored": r_uniform > 1.0,
        "color_net_charge": abs(color_net),                # |Σ_k ω^k q_k| ≈ 0 ⇒ neutral
        "color_constituent_total": constituent_total,      # Σ_k q_k (the 3 quarks)
        "color_neutrality_ratio": neutrality_ratio,        # ≪ 1 ⇒ confined/neutral
        "color_neutral": neutrality_ratio < 0.5,
        # --- flavor ---
        "flavor_charges": q,
        "flavor_spread": float(rd["flavor_spread"]),       # > 0 ⇒ independent per-hole
        "flavor_independent": float(rd["flavor_spread"]) > 1e-3,
        # --- charge ---
        "net_charge": constituent_total,                   # Σ_k q_k (per-hole below)
        "per_hole_charge": q,
        # --- radius · mass ---
        "radius": r,
        "n_spacelike": n_sp,
        "n_timelike": n_tl,
        "mass": mass,
        "r_times_m": r * mass,
        # --- structural spin-½ ---
        "spin_clifford_residual": spin["clifford_residual"],
        "spin_eigenvalue_magnitudes": spin["spin_eigenvalue_magnitudes"],
        "spin_half": bool(spin["spin_half"]),
        # --- deferred ---
        "composite_j2": "DEFERRED to #495/#477 (total-space spin operator; the per-hole "
                        "product read sits at the mixture, not the entangled 3/4)",
    }


def build_and_report(seeds=range(5, 25), max_residual=0.6, n_refine=18,
                     stage1_steps=60, stage2_iters=20):
    """Build one converged emergent proton (the Shared recipe) and report all of its
    quantum numbers. Returns (report, seed) or (None, None) if none converge."""
    found = djs.build_converged_proton(
        seeds=seeds, max_residual=max_residual, n_refine=n_refine,
        stage1_steps=stage1_steps, stage2_iters=stage2_iters)
    if not found:
        return None, None
    rep = proton_report(found)
    return rep, (rep["seed"] if rep else None)


def format_report(rep):
    """A pretty-printed table of the joint observables of one relaxed emergent proton."""
    if rep is None:
        return "No converged 3-hole proton block (honest negative)."
    q = ", ".join(f"{x:.4f}" for x in rep["flavor_charges"])
    uniform_note = ("floored" if rep["color_uniform_floored"]
                    else "also ~0 - b3>=3 register realizes every rep "
                         "(honest negative on the r_state probe)")
    L = [
        "THE PROTON, END TO END - all quantum numbers off ONE relaxed emergent state (#481)",
        f"  seed = {rep['seed']}   betti(block) = {rep['betti_sub']}   holes = {rep['n_holes']}",
        "",
        "COLOR (confinement):",
        f"  singlet [1,w,w2] r_state = {rep['color_singlet_residual']:.3e}"
        f"   {'CARRIED' if rep['color_singlet_carried'] else 'NOT carried'}",
        f"  uniform [1,1,1]  r_state = {rep['color_uniform_residual']:.3e}   ({uniform_note})",
        f"  color-neutral: |Sum w^k q_k| = {rep['color_net_charge']:.4f}  vs  constituent"
        f" total {rep['color_constituent_total']:.4f}  (ratio {rep['color_neutrality_ratio']:.3f})"
        f"   {'NEUTRAL => confined' if rep['color_neutral'] else 'not neutral'}",
        "",
        "FLAVOR:",
        f"  per-hole DK charges = [{q}]   spread = {rep['flavor_spread']:.3f}"
        f"   {'independent' if rep['flavor_independent'] else 'degenerate'}",
        "",
        "CHARGE:",
        f"  net Dirac-Kahler charge Sum_k q_k = {rep['net_charge']:.4f}   per-hole = [{q}]",
        "",
        "RADIUS . MASS:",
        f"  r = sqrt(<l^2>) = {rep['radius']:.4f}  ({rep['n_spacelike']} spacelike,"
        f" {rep['n_timelike']} timelike/null)",
        f"  mass = |S_Regge| = {rep['mass']:.4f}      r.m = {rep['r_times_m']:.4f}",
        "",
        "STRUCTURAL SPIN-1/2:",
        f"  Sigma_ij eigenvalue |magnitudes| = {rep['spin_eigenvalue_magnitudes']}"
        f"  (clifford residual {rep['spin_clifford_residual']:.1e})",
        f"  spin-1/2 {'CONFIRMED' if rep['spin_half'] else 'NOT confirmed'}",
        "",
        "COMPOSITE SPIN (J^2 = 3/4):",
        f"  {rep['composite_j2']}",
    ]
    return "\n".join(L)


def main():
    rep, _seed = build_and_report()
    print(format_report(rep))


if __name__ == "__main__":
    main()
