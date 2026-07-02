# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Mass and radius read the right way off a converged emergent 4D interior (#566).

The runnable battery for the #451 geometric-proton methodology ported to the
genuinely 4D emergent builds: the canonical ``tessera.cobordism.Proton().block()``
and the emergent arm ``tessera.cobordism.ProtonIngredients().block()`` (or any
other converged 4D ``Spacetime``). Everything is a *post-hoc observable
reader* — nothing shapes the lattice.

The reader machinery lives in ``tessera.observe.mass_radius`` (one home —
this script imports it; the ``MassRadius`` observable of
``tessera.observe.battery`` composes the same functions), and its module
docstring carries the load-bearing methodology rules: triangle hinges on
d = 4, closed-fan interior selection with the census always reported, the
C++-built skeleton (never a Python-driven materialization), signature-aware
readings with Im(deficit) always accounted, the dimension-correct fourth-root
radius, and the r·m table with its definitional spread stated first.

Importable (the tests reuse these): ``build_skeleton``, ``interior_hinges``,
``masses``, ``radii``, ``localization``, ``rm_table``, ``measure``, ``report``.

Run (bounded budgets — the fast-test pattern, one attempt per arm):

    python emergent_mass_radius.py                 # canonical Proton
    python emergent_mass_radius.py --arm both      # + the ProtonIngredients arm
    python emergent_mass_radius.py --seed 3
"""
import argparse

import tessera

from tessera.observe.mass_radius import (  # noqa: F401
    IM_TOL,
    PHYSICAL_RM,
    build_skeleton,
    interior_hinges,
    localization,
    masses,
    measure,
    radii,
    rm_table,
)

cob = tessera.cobordism


def report(o):
    """Print one measurement dict in the #451 layout: census first, then the r·m
    spread up front, then the per-definition table, then localization."""
    c, mass, rad, loc, rm = (o["census"], o["mass"], o["radius"],
                             o["localization"], o["rm"])
    print(f"=== {o['label'] or 'converged spacetime'} ===")
    print(f"-- census: {c['n_hinges_interior']} interior (closed-fan) hinges of "
          f"{c['n_hinges_total']} triangles "
          f"({c['n_hinges_boundary']} boundary artefacts excluded); "
          f"{c['n_tops']} top 4-cells, {c['n_boundary_tets']} boundary tetrahedra, "
          f"{o['n_holes']} register holes ({c['n_hole_vertices']} hole vertices)")
    if c["n_hinges_interior"] == 0:
        print("   no interior hinges — every triangle touches the boundary; "
              "the geometric readings below are NaN by construction\n")
    print(f"-- r·m (physical anchor ~ {rm['physical']:.1f}; ORDER-OF-MAGNITUDE "
          f"claim only):")
    print(f"   definitional spread: {rm['spread_min']:.3g} .. {rm['spread_max']:.3g} "
          f"across {len(rm['combos'])} mass x radius definitions")
    for name, value in rm["combos"].items():
        print(f"     {name:24s} = {value:10.4g}")
    print(f"-- mass: m_shell (intensive) = {mass['m_shell']:.4g}   "
          f"m_sum = {mass['m_sum']:.4g}   m_action = {mass['m_action']:.4g}")
    if mass["n_im_nonzero"] == 0:
        im_note = "ZERO (all-spacelike fans)"
    else:
        im_note = (f"NONZERO on {mass['n_im_nonzero']} hinges (boost content); "
                   "masses use Re ε only")
    print(f"   Im(deficit): max |Im ε| = {mass['max_abs_im']:.3g} over the interior "
          f"hinges — {im_note}")
    print(f"-- radius: r_dual = V_dual^(1/4) = {rad['r_dual']:.4g} "
          f"(V_dual = {rad['Vdual']:.4g} over {rad['n_interior_vertices']} interior "
          f"vertices)   cross-check r_primal = {rad['r_primal']:.4g} "
          f"(V_primal = {rad['Vprimal']:.4g})")
    print(f"-- localization: PR = {loc['PR']:.3f} (uniform reference 1.0) -> "
          f"{loc['concentration']:.2f}x more concentrated;   "
          f"mean Re ε = {loc['mean_re']:+.4g} "
          f"({'positive-curvature lump' if loc['mean_re'] > 0 else 'NET NEGATIVE — not the bound-state sign'});   "
          f"std/|mean| = {loc['std_over_mean']:.2f}")
    if loc["shell_profile"]:
        cells = "  ".join(
            f"shell {shell}: n={p['n']}, mean Re ε={p['mean_re']:+.3g}, "
            f"w={p['weight_share']:.0%}"
            for shell, p in loc["shell_profile"].items())
        print(f"   shell profile (BFS from the hole vertices): {cells}")
        print(f"   weighted RMS shell radius = {loc['rms_shell_radius']:.3f} shells; "
              f"|curvature| within shell<=1 = {loc['frac_within_shell1']:.0%}")
    else:
        print("   shell profile: skipped (no register holes to seed the BFS)")
    print()


def _measure_arm(builder, label, seed, max_restarts):
    """Build one arm with the bounded fast-test budget and measure its block()."""
    print(f"building {label} (seed={seed}, max_restarts={max_restarts}, "
          f"defaults otherwise) ...")
    arm = builder(seed=seed)
    arm.build(max_restarts=max_restarts)
    holes = (arm.quark_holes() if hasattr(arm, "quark_holes")
             else arm.emergent_holes())
    o = measure(arm.block(), holes, label)
    o["converged"] = arm.converged()
    return o


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("canonical", "ingredients", "both"),
                        default="canonical",
                        help="which build to read: the canonical Proton, the "
                             "emergent-arm ProtonIngredients, or both (the "
                             "geometry A/B comparison)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-restarts", type=int, default=1,
                        help="bounded validation budget (the fast-test pattern)")
    args = parser.parse_args()

    results = []
    if args.arm in ("canonical", "both"):
        results.append(_measure_arm(cob.Proton, "canonical Proton().block()",
                                    args.seed, args.max_restarts))
    if args.arm in ("ingredients", "both"):
        results.append(_measure_arm(cob.ProtonIngredients,
                                    "emergent-arm ProtonIngredients().block()",
                                    args.seed, args.max_restarts))
    print()
    for o in results:
        report(o)

    if len(results) == 2:
        a, b = results
        print("-- emergent-arm vs canonical (same seed, same budget) --")
        for key, path in (("interior hinges", ("census", "n_hinges_interior")),
                          ("r_dual", ("radius", "r_dual")),
                          ("m_shell", ("mass", "m_shell")),
                          ("PR", ("localization", "PR")),
                          ("mean Re ε", ("localization", "mean_re"))):
            va = a[path[0]][path[1]]
            vb = b[path[0]][path[1]]
            print(f"   {key:16s} canonical {va:10.4g}   emergent {vb:10.4g}")
        print()

    print("VERDICT (the #451 discipline): the robust evidence is the census-clean "
          "interior selection,\n  the localization (participation ratio vs the "
          "uniform reference) and the sign of the mean\n  deficit — r·m is "
          "definition-sensitive, so its table is an order-of-magnitude reading, "
          "never\n  a sharp validator.")


if __name__ == "__main__":
    main()
