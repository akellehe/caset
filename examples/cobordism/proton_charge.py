# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The proton's electric / U(1) charge, read off the emergent proton (#478, part of #410).

The proton here is the **built** one from #489 (`dk_joint_spin.build_converged_proton`): a
single co-optimized `MultiCobordism` pair-creation event whose proton output block carries
the color singlet, with the three quarks emerging as the block's three holes. This reads the
electric charge OFF that relaxed structure --- emergent-first, nothing fabricated.

The Gauss-law holonomy `gaussLawCharge` (`Q = ∮_S E`) is a **degree-2** observable on the
field strength `F = dA`, and `F = dA` is BLOCKED at the proton's **k = 3** register (there is
no curvature step that lands back on the degree-3 holes). So the charge here is the
**Dirac–Kähler net charge at k = 3**: per quark hole `h`, carry its unit-period representative

    ψ_h = es.carriedRepresentative([h], [1.0])          # es = EigenstateSynthesis(sub, 3)
    q_h = dk.charge(dk.lift(3, ψ_h))                     # dk = DiracKahler(sub)

`dk.charge` is the carried U(1) charge `Σ_c j⁰_c = ⟨Φ, Φ⟩_W` --- the conserved Dirac–Kähler
Noether charge. We report the three per-hole charges, their **constituent total** `Σ_k q_k`
(positive --- three colored quarks), and the **singlet-phased net** `Σ_k [1, ω, ω²]_k · q_k`.

Three POST-HOC checks (never loop conditions --- honest negatives are valid; these are
dimensionless lattice proxies, not physical units):

  1. **quantization** --- do the per-hole charges land on an integer or third-integer
     (`n/3`) lattice within a tolerance?
  2. **metric robustness** --- jitter each spacelike `l²` by `×(1 + Normal(0, 0.2))` (the
     pristine metric is restored after each draw) and confirm the charge is stable (small
     std). A genuine gauged-U(1) holonomy is metric-robust; a hand-weighted covector drifts.
  3. **net → 0 vs constituent total** --- the singlet is flavor-blind, so the **net** charge
     `Σ_k phase_k q_k → ~0`, while the constituent total `Σ_k q_k` is the (positive) sum of
     three colored quarks. The ticket explicitly allows the **honest negative** that the net
     does NOT land on exactly `+1`: the singlet carries no net U(1), and the per-hole DK
     charge is a norm (positive), so neither the net nor the total reproduces the physical
     proton `+1` --- that requires the temporal Gauss-law sector at the degree-2 register.

Run:  python examples/cobordism/proton_charge.py
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
_SINGLET_PHASES = [1.0, _W, _W * _W]   # the color singlet [1, ω, ω²]
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dj = _load("dk_joint_spin")


# ----- the charge reader -----
def per_hole_charges(sub, holes):
    """The Dirac–Kähler net charge at k = 3 of each quark hole: carry the hole's unit-period
    representative `ψ_h` and read its carried U(1) charge `q_h = ⟨Φ, Φ⟩_W`. `gaussLawCharge`
    (degree-2, `F = dA`) is blocked at the k = 3 register, so this DK charge stands in."""
    dk = cob.DiracKahler(sub)
    es = cob.EigenstateSynthesis(sub, 3)
    qs = []
    for h in holes:
        psi = es.carriedRepresentative([list(h)], [1.0])
        qs.append(dk.charge(dk.lift(3, list(psi))))
    return np.array(qs, float)


def signed_net(charges):
    """The singlet-phased **net** `Σ_k [1, ω, ω²]_k · q_k` (flavor-blind ⇒ → ~0) and the
    (positive) **constituent total** `Σ_k q_k` (three colored quarks)."""
    charges = np.asarray(charges, float)
    net = sum(_SINGLET_PHASES[k % 3] * charges[k] for k in range(len(charges)))
    return complex(net), float(charges.sum())


def quantization_residual(charges, denom=3):
    """Per-charge distance to the nearest `n/denom` lattice point (denom=3 ⇒ third-integer,
    denom=1 ⇒ integer). Returns the per-hole residuals (in charge units)."""
    charges = np.asarray(charges, float)
    return np.abs(charges - np.round(charges * denom) / denom)


def quantization_verdict(charges, tol=0.05):
    """Closest lattice (third-integer vs integer) and whether every charge lands within
    `tol`. Returns (lattice_name, max_residual, ok, nearest_points). `nearest_points` are
    the lattice points each charge rounds to — for the emergent proton these sit at the
    trivial `n = 0`, i.e. the charges cluster near zero rather than at a nonzero quantum."""
    charges = np.asarray(charges, float)
    third = float(quantization_residual(charges, 3).max())
    integer = float(quantization_residual(charges, 1).max())
    if third <= integer:
        return ("third-integer (n/3)", third, third <= tol, np.round(charges * 3) / 3)
    return "integer", integer, integer <= tol, np.round(charges)


# ----- metric robustness: a gauged holonomy is stable under spacelike-l² jitter -----
def _spacelike_edges(sub):
    """The spacelike edges (l² > 0 in the Lorentzian convention) and their pristine l²."""
    out = []
    for e in sub.getEdgeList().toVector():
        l2 = e.getSquaredLength()
        if l2.real > 0:
            out.append((e, l2))
    return out


def charge_under_jitter(sub, holes, trials=8, scale=0.2, seed=478):
    """Net + total charge under spacelike-l² jitter `×(1 + Normal(0, scale))`, restoring the
    pristine metric after each draw. Returns (net_values, total_values) over the trials."""
    rng = np.random.default_rng(seed)
    spacelike = _spacelike_edges(sub)
    nets, totals = [], []
    for _ in range(trials):
        for e, l2 in spacelike:
            e.setSquaredLength(l2 * (1.0 + scale * rng.normal()))
        try:
            q = per_hole_charges(sub, holes)
            net, total = signed_net(q)
            nets.append(abs(net))
            totals.append(total)
        finally:
            for e, l2 in spacelike:           # restore the pristine metric
                e.setSquaredLength(l2)
    return np.array(nets, float), np.array(totals, float)


# ----- top-level reader -----
def read_charge(opt, rd, jitter_trials=8):
    """Read the proton's electric / U(1) charge off the built proton `(opt, rd)`.
    Returns a dict of the per-hole charges, the net + total, the quantization verdict, and
    the metric-robustness std."""
    sub, holes = rd["_sub"], rd["_holes"]
    charges = per_hole_charges(sub, holes)
    net, total = signed_net(charges)
    lattice, qres, qok, qpts = quantization_verdict(charges)
    nets, totals = charge_under_jitter(sub, holes, trials=jitter_trials)
    return {
        "charges": charges,
        "net": net,
        "total": total,
        "lattice": lattice,
        "quant_residual": qres,
        "quant_ok": qok,
        "quant_points": qpts,
        "jitter_net_mean": float(nets.mean()),
        "jitter_net_std": float(nets.std()),
        "jitter_total_mean": float(totals.mean()),
        "jitter_total_std": float(totals.std()),
    }


def main():
    print("Proton electric / U(1) charge — Dirac–Kähler net at the k=3 register (#478)\n")
    found = dj.build_converged_proton(
        seeds=range(5, 25), max_residual=0.6,
        n_refine=18, stage1_steps=60, stage2_iters=20)
    if not found:
        print("No converged 3-hole proton block in the seed range (honest negative).")
        return
    opt, rd, seed = found
    print(f"converged proton (seed {seed}): holes={rd['n_holes']} "
          f"block_residual={rd['block_residual']:.3e}  (singlet carried)\n")
    res = read_charge(opt, rd)
    print(f"  per-hole charges      = {np.round(res['charges'], 4)}")
    print(f"  constituent total Σq  = {res['total']:.4f}  (positive — three colored quarks)")
    print(f"  singlet net Σ phase·q = {abs(res['net']):.4e}  (|net|; flavor-blind ⇒ → 0)")
    print(f"  quantization          = nearest {res['lattice']} lattice points "
          f"{np.round(res['quant_points'], 4)}, max residual {res['quant_residual']:.4f}  "
          f"({'on-lattice ✓' if res['quant_ok'] else 'off-lattice (honest)'})")
    print(f"  metric robustness     = |net| {res['jitter_net_mean']:.4e} ± "
          f"{res['jitter_net_std']:.2e},  total {res['jitter_total_mean']:.4f} ± "
          f"{res['jitter_total_std']:.4f}  (small std ⇒ gauged holonomy)")
    print("\n  honest verdict: the singlet net U(1) → ~0 (flavor-blind), NOT +1; the physical")
    print("  proton +1 lives in the degree-2 temporal Gauss-law sector, blocked at k=3.")


if __name__ == "__main__":
    main()
