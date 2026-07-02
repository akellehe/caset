# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ONE observable-battery construction over a ProtonIngredients specimen (#583).

Constructs the blessed read context (``tessera.observe.Register``) for a
specimen and runs the full gated battery whenever the specimen carries three
register holes — below three, every inapplicable observable reports its skip
reason instead of crashing. One uniform JSON record + a readable table; b₃ is
always recorded next to the hole census with the ``holes_vs_b3_divergent``
flag (the campaign taught us holes and b₃ can disagree).

## Input modes

* ``--geometry dump.json`` — **the faithful path**: a schema-1 campaign
  geometry dump (top cells in intrinsic vertex order, per-edge complex
  squared lengths, per-vertex times) rebuilt via ``Spacetime.fromCells``.
  The dump is an attempt's ONLY faithful record: the engine build is NOT
  process-deterministic (identical fresh processes diverge on the same seed —
  measured, not hypothetical), so a base seed labels an attempt, it does not
  reproduce it. The rebuilt state is verified against the dump (cells, edge
  lengths, and any recorded betti/holes metadata) before measuring.
* ``--seed N`` — a **fresh bounded attempt**: drives a new
  ``tessera.cobordism.ProtonIngredients(seed=N)`` build at the given budgets
  (two-step by default; ``--joint`` drives ``build_joint`` — the #560
  collapsed event graph). This is a NEW SAMPLE of the seed's attempt
  distribution, never a reproduction of any prior record. In the joint arm
  the kept node's output blocks (``output_blocks()``) flow into the battery
  as block provenance automatically.

Provenance (build history: the diquark pair, output blocks) travels via the
campaign record / geometry-dump metadata or ``--provenance file.json`` — it
is never guessed and never re-derived by re-running a build. Recognized keys:
``diquark_pair`` (step-1 hole indices, for the pair-loop criterion (b)) and
``blocks`` (label + vertices + target_re/target_im each, for the per-block
residuals).

## The campaign-analyzer import surface

The analyzer consumes the same package surface this script drives::

    from tessera.observe import Register, battery, load_geometry_dump, rebuild_spacetime
    dump = load_geometry_dump(path)
    st = rebuild_spacetime(dump)
    record = battery.measure_all(make_register(st), provenance=None)

(``make_register`` here is the one-liner hole-count policy: a 3-hole register
when the specimen has one, otherwise all the holes it does have, so the
battery can report per-observable skips.)

Run:

    python observe_proton_ingredients.py --geometry seed_7_geometry.json
    python observe_proton_ingredients.py --seed 1                  # fresh two-step attempt
    python observe_proton_ingredients.py --seed 1 --joint --out record.json
"""
import argparse
import json

import tessera

from tessera.observe import (
    Register,
    battery,
    load_geometry_dump,
    rebuild_spacetime,
    verify_rebuild,
)

cob = tessera.cobordism

#: jointNode's output-block seeding order (baryon at v3, antibaryon at v4).
JOINT_BLOCK_LABELS = ("baryon", "antibaryon")


def make_register(st, degree=3):
    """The construction's hole-count policy: ask for the full 3-hole register
    when the specimen carries one; otherwise take every hole it does have (so
    the battery reports per-observable skip reasons instead of this
    constructor raising). The surplus warning still fires when more than
    three holes exist — the read then covers a sub-register."""
    total = len(cob.MultiCobordism.emergent_holes(st, degree))
    return Register(st, count=min(3, total), degree=degree)


def blocks_provenance(output_blocks):
    """``ProtonIngredients.output_blocks()`` as battery block provenance
    (label + vertices + target), in jointNode's seeding order."""
    blocks = []
    for index, block in enumerate(output_blocks):
        label = (JOINT_BLOCK_LABELS[index]
                 if index < len(JOINT_BLOCK_LABELS) else f"block{index}")
        blocks.append({
            "label": label,
            "vertices": [int(v) for v in block.vertices],
            "target": list(block.target),
        })
    return blocks


def load_provenance(path=None, dump=None):
    """Provenance from the recognized geometry-dump metadata keys, overlaid
    by an explicit ``--provenance`` JSON file. Returns None when neither
    supplies anything — the battery then reports the provenance-gated
    channels as not evaluable, never guessed."""
    provenance = {}
    if dump is not None:
        for key in ("diquark_pair", "blocks"):
            if dump.get(key) is not None:
                provenance[key] = dump[key]
    if path is not None:
        with open(path) as fh:
            provenance.update(json.load(fh))
    return provenance or None


def observe_geometry_dump(path, provenance_path=None, gates=True):
    """The faithful path: load + verify a schema-1 geometry dump, rebuild the
    exact state, and run the battery. Returns the record."""
    dump = load_geometry_dump(path)
    st = rebuild_spacetime(dump)
    mismatches = verify_rebuild(st, dump)
    if mismatches:
        raise RuntimeError(
            f"rebuilt state does not match the geometry dump {path}: "
            f"{mismatches}")
    register = make_register(st)
    provenance = load_provenance(provenance_path, dump)
    record = battery.measure_all(register, provenance=provenance, gates=gates)
    record["input"] = {
        "mode": "geometry_dump",
        "path": str(path),
        "schema": dump["schema"],
        "dump_verified": True,
        "base_seed": dump.get("base_seed"),
        "note": ("the dump is the attempt's faithful record; the engine "
                 "build is not process-deterministic, so only the dump — "
                 "never a re-run — reproduces this state"),
    }
    return record


def observe_fresh_attempt(seed, joint=False, max_restarts=1, init_steps=40,
                          evolve_steps=20, stage2_max_iters=10,
                          provenance_path=None, gates=True):
    """A fresh bounded ProtonIngredients attempt (a NEW SAMPLE at this seed —
    never a reproduction of any prior attempt), measured post-hoc. The joint
    arm feeds its kept output blocks into the battery as block provenance."""
    ingredients = cob.ProtonIngredients(seed=seed)
    drive = ingredients.build_joint if joint else ingredients.build
    drive(max_restarts=max_restarts, init_steps=init_steps,
          evolve_steps=evolve_steps, stage2_max_iters=stage2_max_iters)

    register = make_register(ingredients.block())
    provenance = load_provenance(provenance_path) or {}
    if joint and "blocks" not in provenance:
        provenance["blocks"] = blocks_provenance(ingredients.output_blocks())
    record = battery.measure_all(register, provenance=provenance or None,
                                 gates=gates)
    record["input"] = {
        "mode": "fresh_attempt",
        "arm": "joint" if joint else "two_step",
        "seed": int(seed),
        "budgets": {"max_restarts": max_restarts, "init_steps": init_steps,
                    "evolve_steps": evolve_steps,
                    "stage2_max_iters": stage2_max_iters},
        "note": ("a fresh attempt at this seed — NOT a reproduction of any "
                 "prior record (the engine build is not "
                 "process-deterministic); the faithful record of an attempt "
                 "is its geometry dump"),
    }
    record["build"] = {
        "converged": ingredients.converged(),
        "stationary": ingredients.stationary(),
        "persistent": ingredients.persistent(),
        "kept_seed": int(ingredients.seed()),
        "final_objective": float(ingredients.final_objective()),
        "input_residual": float(ingredients.input_residual()),
        "singlet_residual": float(ingredients.singlet_residual()),
        "diquark_residual": float(ingredients.diquark_residual()),
        "baryon_residual": float(ingredients.baryon_residual()),
        "antibaryon_residual": float(ingredients.antibaryon_residual()),
    }
    return record


def render_table(record):
    """The readable side of the record: the register census, then one block
    per observable (status, gates, and the observable's headline channels)."""
    lines = []
    reg = record["register"]
    lines.append("=== observable battery ===")
    if "input" in record:
        source = record["input"]
        detail = source.get("path", source.get("seed"))
        lines.append(f"input: {source['mode']}"
                     + (f" ({source.get('arm')})" if "arm" in source else "")
                     + f" [{detail}]")
    lines.append(
        f"register: holes_used={reg['holes_used']} of "
        f"holes_total={reg['holes_total']}, b3={reg['b3']}, "
        f"divergent={reg['holes_vs_b3_divergent']}, "
        f"degree={reg['degree']}, cells={reg['n_top_cells']}, "
        f"causal_content={reg['causal_content']}")
    if reg["dropped_holes"]:
        lines.append(f"  dropped holes: {reg['dropped_holes']}")
    if "build" in record:
        b = record["build"]
        lines.append(
            f"build: converged={b['converged']} (stationary={b['stationary']}, "
            f"persistent={b['persistent']}), F={b['final_objective']:.4g}, "
            f"r_U={b['input_residual']:.3g}, "
            f"singlet diag={b['singlet_residual']:.3g}")

    for name, entry in record["observables"].items():
        if entry["status"] != "measured":
            lines.append(f"-- {name}: {entry['status']}")
            continue
        gates = entry.get("gates")
        gate_note = ""
        if gates:
            gate_note = (f"   [gauge {gates['gauge_delta']:.1e}"
                         f"{'' if gates['gauge_ok'] else ' FLAGGED'}, "
                         f"relabel {gates['relabel_delta']:.1e}"
                         f"{'' if gates['relabel_ok'] else ' FLAGGED'}]")
        lines.append(f"-- {name}: measured{gate_note}")
        r = entry["record"]
        if name == "singlet_diagnostic":
            lines.append(
                f"     singlet r_state={r['singlet_residual']:.3g}  "
                f"betti={r['betti']}  holes={r['holes_total']}  b3={r['b3']}  "
                f"divergent={r['holes_vs_b3_divergent']}")
        elif name == "block_residuals":
            for block in r["blocks"]:
                lines.append(
                    f"     {block['label']}: residual={block['residual']:.3g}"
                    f" (cells={block['n_cells_in_region']}, "
                    f"full_leak={block['full_leak']}, "
                    f"target_norm2={block['target_norm2']:.3g})")
        elif name == "mass_radius":
            lines.append(
                f"     m_shell={r['mass']['m_shell']:.4g}  "
                f"m_sum={r['mass']['m_sum']:.4g}  "
                f"m_action={r['mass']['m_action']:.4g}  "
                f"r_dual={r['radius']['r_dual']:.4g}  "
                f"r_primal={r['radius']['r_primal']:.4g}")
            lines.append(
                f"     PR={r['localization']['PR']:.3f}  "
                f"mean Re eps={r['localization']['mean_re']:+.4g}  "
                f"r*m spread {r['rm']['spread_min']:.3g}.."
                f"{r['rm']['spread_max']:.3g} "
                f"(anchor ~{r['rm']['physical']:.1f}; order-of-magnitude only)")
        elif name == "pair_loop_flavor":
            q = ", ".join(f"{x:.5f}" for x in r["loop_q"])
            lines.append(
                f"     loop_q=[{q}]  odd={r['odd_loop']} (dual hole "
                f"{r['dual_hole']})  rho={r['rho']:.3f}  "
                f"2:1={r['multiplicity_2_1']}  r_u={r['r_u']:.2g}")
            lines.append(
                f"     odd_is_diquark_loop={r['odd_is_diquark_loop']} "
                f"({r['odd_is_diquark_loop_status']})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--geometry", metavar="DUMP_JSON",
                        help="a schema-1 campaign geometry dump — the "
                             "attempt's faithful record (the ONLY rebuild "
                             "path; seeds label attempts, they do not "
                             "reproduce them)")
    source.add_argument("--seed", type=int,
                        help="drive a FRESH bounded ProtonIngredients attempt "
                             "at this seed (a new sample, never a "
                             "reproduction)")
    parser.add_argument("--joint", action="store_true",
                        help="fresh attempts drive build_joint (#560) instead "
                             "of the two-step build; the kept output blocks "
                             "feed the per-block residuals")
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--init-steps", type=int, default=40)
    parser.add_argument("--evolve-steps", type=int, default=20)
    parser.add_argument("--stage2-max-iters", type=int, default=10)
    parser.add_argument("--provenance", metavar="JSON",
                        help="build-history provenance (diquark_pair, blocks) "
                             "— from the campaign record; never guessed")
    parser.add_argument("--no-gates", action="store_true",
                        help="skip the GAUGE/RELABEL gates (faster; the "
                             "record then carries no gate residuals)")
    parser.add_argument("--out", metavar="JSON",
                        help="write the JSON record here (default: print it "
                             "after the table)")
    args = parser.parse_args()

    if args.geometry:
        record = observe_geometry_dump(args.geometry, args.provenance,
                                       gates=not args.no_gates)
    else:
        record = observe_fresh_attempt(
            args.seed, joint=args.joint, max_restarts=args.max_restarts,
            init_steps=args.init_steps, evolve_steps=args.evolve_steps,
            stage2_max_iters=args.stage2_max_iters,
            provenance_path=args.provenance, gates=not args.no_gates)

    print(render_table(record))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=1)
        print(f"\nrecord written to {args.out}")
    else:
        print("\n" + json.dumps(record))


if __name__ == "__main__":
    main()
