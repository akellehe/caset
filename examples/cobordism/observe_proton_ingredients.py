# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Observe an emergent proton — the battery driver (#593, part of #559).

The battery is C++ (``tessera.observables``): pure-reader Observables over a
``RegisterContext``, with C++ GAUGE/RELABEL gates. This driver is the ONE thin,
classless Python surface — it constructs no framework classes; every measurement
is a bound C++ read.

Two input modes:

* ``--geometry <dump>`` — a schema-1 geometry dump (the attempt's ONLY faithful
  record: the engine build is NOT process-deterministic, so a base seed labels
  an attempt, it never reproduces it). The dump JSON is rehydrated into a LIVE
  complex by the ``LiveComplex`` loader (``fromCells`` + the recorded metric +
  ``materializeFacets`` — outside the reader), then read. This is the faithful
  path.

* ``--seed N [--joint]`` — a FRESH ``ProtonIngredients`` attempt, LABELED as
  such and never a reproduction (the #578 finding). The built complex is read
  exactly like a rehydrated one — the observable never distinguishes them.

A register with >= 3 emergent holes runs the full battery (singlet, blocks,
mass, radius, pair-loop) with GAUGE/RELABEL gates; a sub-3-hole specimen reports
per-observable skip reasons. The output is a JSON-able record plus a readable
table.
"""
import argparse
import json
import sys

import tessera

_obs = tessera.observables
_cob = tessera.cobordism

#: The dump format this driver reads (schema-1: the #562 campaign worker's).
GEOMETRY_SCHEMA = 1
#: Provenance block labels for the joint arm's two output blocks, in order.
JOINT_BLOCK_LABELS = ("baryon", "antibaryon")


def load_geometry_dump(path):
    """Load + validate a schema-1 geometry dump (raises on a wrong/missing
    schema or missing geometry keys — a non-dump record is never half-read)."""
    with open(path) as fh:
        dump = json.load(fh)
    if dump.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError(
            f"{path}: geometry dump schema {dump.get('schema')!r} is not the "
            f"supported schema {GEOMETRY_SCHEMA}")
    missing = [k for k in ("dimensions", "cells", "edges", "vertex_times")
               if k not in dump]
    if missing:
        raise ValueError(f"{path}: geometry dump is missing {missing}")
    return dump


def rehydrate(dump):
    """A LIVE Spacetime carrying the dumped final state, via the ``LiveComplex``
    loader (the construction lives OUTSIDE the reader). Never re-runs a build."""
    edges = {}
    for u, v, re_l2, im_l2 in dump["edges"]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        edges[key] = complex(re_l2, im_l2)
    times = {int(vid): float(t) for vid, t in dump["vertex_times"]}
    return _obs.LiveComplex.load([[int(v) for v in c] for c in dump["cells"]],
                                 edges, times, int(dump["dimensions"]))


def verify_dump(st, dump):
    """Check the rehydrated complex matches the dump's recorded combinatorial
    reads (betti / holes), when present. Raises when they disagree."""
    checks = {
        "betti": lambda: [int(b) for b in _cob.MultiCobordism.betti(st)],
        "holes": lambda: len(_cob.MultiCobordism.emergent_holes(st, 3)),
    }
    for key, compute in checks.items():
        if key in dump and dump[key] != compute():
            raise RuntimeError(
                f"rehydrated {key} {compute()} does not match the dump's "
                f"recorded {dump[key]} — the dump is not faithful")


def make_register(st, target=None):
    """A ``RegisterContext`` over the live complex: a 3-hole register when the
    specimen hosts it, otherwise all the holes it has (so the battery reports
    per-observable skips instead of raising here)."""
    available = len(_cob.MultiCobordism.emergent_holes(st, 3))
    count = min(available, 3)
    if target is None:
        target = _cob.Proton.singlet()
    return _obs.RegisterContext(st, count, 3, target)


def blocks_from_provenance(provenance):
    """Build the ``BlockResiduals`` provenance blocks from a JSON-able list
    (``[{label, vertices, target | target_re/target_im}]``)."""
    blocks = []
    for index, block in enumerate(provenance.get("blocks", [])):
        if "target" in block:
            target = [complex(t) for t in block["target"]]
        else:
            target = [complex(re, im) for re, im in
                      zip(block["target_re"], block["target_im"])]
        label = str(block.get("label", f"block{index}"))
        blocks.append(_obs.Block(label, [int(v) for v in block["vertices"]],
                                 target))
    return blocks


def battery_observables(provenance):
    """The battery line-up, in measurement order. ``BlockResiduals`` carries the
    provenance blocks (empty ⇒ it skips no_provenance); ``PairLoopFlavor``
    carries the recorded diquark pair when the build history supplies it."""
    diquark = provenance.get("diquark_pair")
    pair_loop = (_obs.PairLoopFlavor((int(diquark[0]), int(diquark[1])))
                 if diquark is not None else _obs.PairLoopFlavor())
    return [
        _obs.SingletResidual(),
        _obs.BlockResiduals(blocks_from_provenance(provenance)),
        _obs.EmergentMass(),
        _obs.EmergentRadius(),
        pair_loop,
    ]


def measure_battery(ctx, provenance, gates=True):
    """Measure every battery observable against ``ctx`` — a flat JSON-able
    record: the register census, then per-observable measured (record + gate
    residuals) or skipped(reason) entries."""
    record = {
        "register": ctx.summary(),
        "provenance_keys": sorted(provenance) if provenance else [],
        "observables": {},
    }
    for observable in battery_observables(provenance):
        name = observable.record_key()
        reason = observable.skip_reason(ctx)
        if reason:
            record["observables"][name] = {
                "status": f"skipped({reason})", "reason": reason}
            continue
        entry = {"status": "measured", "record": observable.record(ctx)}
        if gates:
            result = _obs.ObservableGates.evaluate(observable, ctx)
            entry["gates"] = {
                "gauge_delta": result.gauge_delta,
                "relabel_delta": result.relabel_delta,
                "gauge_ok": result.gauge_ok,
                "relabel_ok": result.relabel_ok,
                "gate_tol": result.gate_tol,
            }
        record["observables"][name] = entry
    return record


def observe_geometry_dump(path, provenance_path=None):
    """The faithful path: schema-1 dump -> rehydrate -> verify -> battery. Dump
    metadata (``diquark_pair`` / ``blocks``) flows as provenance; a
    ``--provenance`` file overrides and extends it."""
    dump = load_geometry_dump(path)
    st = rehydrate(dump)
    verify_dump(st, dump)
    provenance = {}
    for key in ("diquark_pair", "blocks"):
        if key in dump:
            provenance[key] = dump[key]
    if provenance_path:
        with open(provenance_path) as fh:
            provenance.update(json.load(fh))
    record = measure_battery(make_register(st), provenance)
    record["input"] = {
        "mode": "geometry_dump", "path": path, "dump_verified": True,
        "base_seed": dump.get("base_seed"),
        "note": "the schema-1 dump is the attempt's faithful record; the engine "
                "build is NOT process-deterministic, so the base seed labels the "
                "attempt, never reproduces it",
    }
    return record


def observe_fresh_attempt(seed, joint=False, max_restarts=1, provenance_path=None):
    """The --seed path: a FRESH ``ProtonIngredients`` attempt (a NEW SAMPLE, never
    a reproduction of any prior attempt with this seed — the #578 finding). The
    built complex is read exactly like a rehydrated one."""
    ingredients = _cob.ProtonIngredients(seed)
    ingredients.build(max_restarts)
    st = ingredients.block()
    provenance = {}
    if provenance_path:
        with open(provenance_path) as fh:
            provenance.update(json.load(fh))
    record = measure_battery(make_register(st), provenance)
    record["input"] = {
        "mode": "fresh_attempt", "seed": int(seed), "joint": bool(joint),
        "note": "a FRESH attempt labeled by this seed — NEVER a reproduction "
                "(the engine build is not process-deterministic); dump the "
                "geometry for a faithful record",
    }
    return record


def render_table(record):
    """A readable text rendering of the battery record."""
    reg = record["register"]
    lines = ["observable battery",
             "=" * 60,
             f"input: {record.get('input', {}).get('mode', '?')}",
             f"register: dim={reg['dimensions']} n_top_cells={reg['n_top_cells']} "
             f"holes_used={reg['holes_used']} of holes_total={reg['holes_total']} "
             f"b3={reg['b3']} divergent={reg['holes_vs_b3_divergent']} "
             f"causal_content={reg['causal_content']}",
             f"betti: {reg['betti']}",
             "-" * 60]
    for name, entry in record["observables"].items():
        if entry["status"] != "measured":
            lines.append(f"{name}: {entry['status']}")
            continue
        gates = entry.get("gates", {})
        ok = (f"gauge_ok={gates.get('gauge_ok')} "
              f"relabel_ok={gates.get('relabel_ok')}") if gates else ""
        lines.append(f"{name}: measured  {ok}")
        rec = entry["record"]
        if name == _obs.SingletResidual().record_key():
            lines.append(f"    singlet_residual = {rec['singlet_residual']:.3e}")
        elif name == _obs.EmergentMass().record_key():
            lines.append(f"    m_shell={rec['mass']['m_shell']:.4f} "
                         f"m_sum={rec['mass']['m_sum']:.4f} "
                         f"m_action={rec['mass']['m_action']:.4f}")
            lines.append("    r.m table (order-of-magnitude only): "
                         f"spread [{rec['rm']['spread_min']:.3f}, "
                         f"{rec['rm']['spread_max']:.3f}], "
                         f"physical ~ {rec['rm']['physical']:.2f}")
        elif name == _obs.EmergentRadius().record_key():
            lines.append(f"    r_dual={rec['radius']['r_dual']:.4f} "
                         f"r_primal={rec['radius']['r_primal']:.4f}")
        elif name == _obs.PairLoopFlavor().record_key():
            lines.append(f"    rho={rec['rho']:.4f} "
                         f"multiplicity_2_1={rec['multiplicity_2_1']} "
                         f"odd_loop={tuple(rec['odd_loop'])} "
                         f"odd_is_diquark_loop={rec['odd_is_diquark_loop']} "
                         f"({rec['odd_is_diquark_loop_status']})")
        elif name == _obs.BlockResiduals([]).record_key():
            for row in rec["blocks"]:
                lines.append(f"    block {row['label']}: residual="
                             f"{row['residual']:.3e} "
                             f"(cells={row['n_cells_in_region']})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--geometry", metavar="DUMP",
                        help="a schema-1 geometry dump (the faithful path)")
    source.add_argument("--seed", type=int,
                        help="a FRESH ProtonIngredients attempt (never a "
                             "reproduction)")
    parser.add_argument("--joint", action="store_true",
                        help="(with --seed) label the fresh attempt as the joint "
                             "arm")
    parser.add_argument("--max-restarts", type=int, default=1,
                        help="(with --seed) build restart budget")
    parser.add_argument("--provenance", metavar="JSON",
                        help="a provenance JSON file (diquark_pair / blocks)")
    parser.add_argument("--json", action="store_true",
                        help="emit the JSON record instead of the table")
    args = parser.parse_args(argv)

    if args.geometry:
        record = observe_geometry_dump(args.geometry, args.provenance)
    else:
        record = observe_fresh_attempt(args.seed, joint=args.joint,
                                       max_restarts=args.max_restarts,
                                       provenance_path=args.provenance)
    if args.json:
        json.dump(record, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(render_table(record))
    return record


if __name__ == "__main__":
    main()
