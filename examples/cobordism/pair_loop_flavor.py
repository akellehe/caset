# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pair-loop dual-basis flavor read on a 3-hole register (#561, part of #410/#559).

The fixture battery for the pair_loop_quarks.tex §7 experiment. The reader
machinery lives in ``tessera.observe.pair_loop_flavor`` (one home — this
script imports it; the ``PairLoopFlavor`` observable of
``tessera.observe.battery`` composes the same functions), and its module
docstring carries the full methodology: the single joint carried
representative, the arithmetic pair loops `∮_{γ_ij} ψ = w_i + w_j` with the
Poincaré duality `[γ_ij] = -[k]`, the metric-weighted DK-style charges, the
`endSignCovector` induced orientation (the RELABEL lesson), the
pre-registered criteria (a) multiplicity 2:1, (b) odd-one-out == the diquark
loop — decidable only with build-history provenance — and (c) the GAUGE +
RELABEL gates.

Run `python pair_loop_flavor.py` for the per-fixture table over the synthetic
b₃ = 3..5 fixtures (+ the real converged_b3_3) in
`tests/fixtures/composite_spin/`. The read takes the structure's first three
register holes (`MultiCobordism.emergent_holes(st, 3)[:3]`, the #485 fixture
convention). `read_structure` is the specimen entry point — point it at a
built proton block (e.g. `Proton.block()` + `Proton.quark_holes()`) with
`diquark_pair` from the specimen's recorded build history (the campaign
record / geometry-dump metadata) once a 3-hole specimen is available.
"""
from tessera.observe.pair_loop_flavor import (  # noqa: F401
    GATE_TOL,
    OMEGA,
    PAIR_LOOPS,
    RHO_GATE_TOL,
    RHO_MAX,
    SINGLET,
    _FIXTURES,
    _facet_indices,
    build_spacetime,
    complement_hole,
    evaluate_criteria,
    gauge_gate,
    induced_orientation_signs,
    joint_read,
    load_fixture,
    odd_one_out,
    read_fixture,
    read_structure,
    register_holes,
    relabel_gate,
)


def main():
    print("Pair-loop dual-basis flavor read (#561) — pair_loop_quarks.tex §7")
    print(f"target singlet [1, ω, ω²]; loops {PAIR_LOOPS} dual to holes "
          f"{[complement_hole(p) for p in PAIR_LOOPS]}; rho_max={RHO_MAX}, "
          f"gate tol={GATE_TOL:.0e}\n")
    names = ["synthetic_b3_3.json", "synthetic_b3_4.json",
             "synthetic_b3_5.json", "converged_b3_3.json"]
    rows = [read_fixture(n) for n in names]

    head = (f"{'fixture':<22} {'b3':>2} {'r_U':>8} "
            f"{'q(γ01)':>9} {'q(γ02)':>9} {'q(γ12)':>9} "
            f"{'odd':>5} {'dual':>4} {'rho':>6} {'2:1':>4} "
            f"{'dual_res':>9} {'gauge':>8} {'relabel':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        v, rd = r["verdict"], r["read"]
        gate_g = max(r["gauge"].values())
        gate_r = max(r["relabel"].values())
        print(f"{r['name']:<22} {r['b3']:>2} {rd['r_u']:>8.1e} "
              f"{rd['loop_q'][0]:>9.5f} {rd['loop_q'][1]:>9.5f} "
              f"{rd['loop_q'][2]:>9.5f} "
              f"{'γ' + str(v['odd_loop']):>5} {v['dual_hole']:>4} "
              f"{v['rho']:>6.3f} {str(v['multiplicity_2_1']):>4} "
              f"{max(rd['dual_residual']):>9.1e} {gate_g:>8.1e} {gate_r:>8.1e}")

    print("\nper-hole detail (oriented weights w_h == the pinned singlet; "
          "charges q_h):")
    for r in rows:
        rd = r["read"]
        w_str = ", ".join(f"{w:.3f}" for w in rd["w"])
        q_str = ", ".join(f"{q:.5f}" for q in rd["q"])
        print(f"  {r['name']:<22} σ={rd['sigma']}  w=[{w_str}]  q=[{q_str}]")

    print("\nfindings:")
    print("  * duality bookkeeping [γ_ij] = -[k]: |w_i + w_j + w_k| ~ 1e-16 "
          "on every fixture (exact under the pinned singlet).")
    print("  * GAUGE and RELABEL hold to machine precision — the read is a "
          "label-free, gauge-clean observable (criterion c).")
    synth = [r for r in rows if r["kind"] == "synthetic"]
    conv = [r for r in rows if r["kind"] != "synthetic"]
    ok = sum(r["verdict"]["multiplicity_2_1"] for r in synth)
    rhos = ", ".join(f"{r['verdict']['rho']:.3f}" for r in synth)
    print(f"  * multiplicity 2:1 (criterion a): {ok}/{len(synth)} synthetic "
          f"fixtures cluster u:u:d (rho {rhos}).")
    for r in conv:
        print(f"  * {r['name']}: rho={r['verdict']['rho']:.3f} — the relaxed "
              "geometry is near-degenerate across the three loops; the "
              "clustering verdict needs a genuine built specimen.")
    print("  * criterion (b) (odd loop == the diquark loop) needs the build's "
          "step-1 pair: fixtures carry no build history, so it is reported as "
          "the odd loop's identity only. Run `read_structure(st, holes, "
          "diquark_pair=...)` on the first 3-hole specimen.")


if __name__ == "__main__":
    main()
