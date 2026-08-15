# Out-of-band rebuild + characterization of a recorded campaign attempt (#578).
#
# THE rebuild path is the geometry dump — Spacetime.fromCells on the recorded final
# complex (seconds, exact by construction):
#
#   --from-dump P   rebuild from a geometry dump; verified against the dump's own
#                   recorded metadata (Betti/holes/singlet);
#   --seed N        uses the seed's dump when one exists. Passing --replay instead
#                   re-RUNS the attempt with the frozen drive — but the engine build
#                   is NOT process-deterministic (identical fresh processes diverge on
#                   the same seed), so a replay is a fresh sample from the seed's
#                   attempt distribution, not a reconstruction; the verdict comparison
#                   it reports is a divergence measurement, not a check that must pass.
#
# The rebuilt state is characterized with the READY observables — color (Betti,
# emergent holes, singlet residual, per-register unit carry), geometry (Regge
# stationarity, Lorentzian deficit-angle curvature in BOTH channels, dual volumes,
# edge-length ranges), spectral dimension on the 1-skeleton, and the deficit-angle
# Wilson-loop measurements — landing in one analysis JSON. The open physics readouts
# (#478 charge, #479 flavor, #480 mass/radius, #481 joint) plug in here later.
#
# Run in the campaign worktree's venv, single-threaded:
#
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
#   .venv-build/bin/python analyze_attempt.py --seed <base_seed> --run-dir .overnight
import argparse
import glob
import itertools
import json
import os
import sys

import tessera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker  # the frozen drive — replay MUST use the recording generation's bytes
import cmath

cob = tessera.cobordism

# Verdict fields compared when --replay re-runs an attempt. The engine build is not
# process-deterministic, so agreement is a divergence MEASUREMENT (how far a fresh
# sample of the same seed lands from the recorded one), not a check that must pass.
# Wall-time/host fields (elapsed_s, rss_mb) and artifact paths are excluded.
REPLAY_KEYS = ("converged", "stationary", "persistent", "stage2_iters_total",
               "holes", "max_holes", "max_b3", "betti", "singlet", "input_ru",
               "singlet_conj", "F", "cells", "edges", "re_min", "re_max", "im_max")


def find_verdict(run_dir, seed):
    """The recorded verdict line for a base seed, or None (in-flight/unknown)."""
    for path in sorted(glob.glob(os.path.join(run_dir, "worker_*.jsonl"))):
        if ".progress." in path:
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("base_seed") == seed:
                    return record
    return None


def dump_path_for(run_dir, seed):
    path = os.path.join(run_dir, "geometry", f"seed_{seed}_geometry.json")
    return path if os.path.exists(path) else None


def replay(seed):
    """Re-run the attempt the way the worker did (no recorder, no dumps): same node
    construction, same drive, same verdict fields — a fresh SAMPLE of this seed's
    attempt distribution, not a reconstruction (the build is not process-
    deterministic). Returns (result_record, final_formation_spacetime, state)."""
    ingredients = cob.ProtonIngredients(seed=seed)
    nodes = [(ingredients.joint_node(seed), worker.JOINT_LABEL)]
    state = worker.AttemptState()
    result = worker.run_attempt_on_nodes(seed, lambda record: None, None, state,
                                         nodes)
    return result, nodes[-1][0].st, state


def rebuild_from_dump(dump):
    """A Spacetime carrying the dumped final state: fromCells on the top cells,
    then the recorded per-vertex times and per-edge complex squared lengths."""
    st = tessera.spacetime.Spacetime.fromCells(dump["dimensions"], dump["cells"])
    vertices = st.getVertexList()
    for vid, t in dump["vertex_times"]:
        vertices.get(int(vid)).setTime(float(t))
    by_pair = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        by_pair[(min(a, b), max(a, b))] = e
    for u, v, re_l2, im_l2 in dump["edges"]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        by_pair[key].setLength(cmath.sqrt(complex(complex(re_l2, im_l2))))
    st.materializeFacets()
    return st


def compare(expected, actual, keys):
    """{key: (expected, actual)} for every key that differs (exact equality —
    the replay contract is bit-for-bit, so no tolerances)."""
    mismatches = {}
    for k in keys:
        if k in expected and expected[k] != actual.get(k):
            mismatches[k] = (expected[k], actual.get(k))
    return mismatches


def verify_state_against_dump(st, dump):
    """The rebuilt/replayed state carries exactly the dumped complex: same top
    cells (as vertex sets), same edge squared lengths, same register content."""
    mismatches = {}
    cells = sorted(sorted(int(v.getId()) for v in c.getVertices())
                   for c in st.getTopSimplices())
    dumped = sorted(sorted(int(v) for v in c) for c in dump["cells"])
    if cells != dumped:
        mismatches["cells"] = (len(dumped), len(cells))
    lengths = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        l2 = (e.getLength() * e.getLength())
        lengths[(min(a, b), max(a, b))] = (l2.real, l2.imag)
    dumped_lengths = {(min(int(u), int(v)), max(int(u), int(v))): (re, im)
                      for u, v, re, im in dump["edges"]}
    if lengths != dumped_lengths:
        wrong = sum(1 for k, val in dumped_lengths.items()
                    if lengths.get(k) != val)
        mismatches["edge_lengths"] = (len(dumped_lengths), wrong)
    for key, value in (("betti", list(cob.MultiCobordism.betti(st))),
                       ("holes", len(cob.MultiCobordism.emergent_holes(
                           st, worker.REGISTER_DEGREE))),
                       ("singlet", float(cob.MultiCobordism.r_state(
                           st, worker.REGISTER_DEGREE, cob.Proton.singlet())))):
        if key in dump and dump[key] != value:
            mismatches[key] = (dump[key], value)
    return mismatches


def characterize(st, degree=worker.REGISTER_DEGREE):
    """The ready observables on a rebuilt state. Each block is best-effort — a
    block that raises reports its error instead of killing the analysis."""
    out = {}

    def block(name, fn):
        try:
            out[name] = fn()
        except Exception as error:
            out[name] = {"error": repr(error)}

    def color():
        holes = [list(h) for h in cob.MultiCobordism.emergent_holes(st, degree)]
        es = cob.EigenstateSynthesis(st, degree)
        return {
            "betti": list(cob.MultiCobordism.betti(st)),
            "holes": holes,
            "singlet_residual": float(cob.MultiCobordism.r_state(
                st, degree, cob.Proton.singlet())),
            # a live register carries a unit period on its own cycle
            "per_register_unit_carry": [
                float(es.residualForPeriods([h], [1.0])) for h in holes],
        }

    def geometry():
        squared = [(e.getLength() * e.getLength()) for e in st.getEdgeList().toVector()]
        deficits, dual = [], []
        for s in st.getSimplices():
            if len(s.getVertices()) != 3:      # hinges = (d-2)=2-simplices
                continue
            try:
                deficits.append(complex(s.lorentzianDeficitAngle()))
                dual.append(abs(float(s.dualVolume())))
            except Exception:                  # boundary/degenerate hinge
                pass
        # BOTH channels of the complex Lorentzian deficit, symmetric statistics:
        # Re ε = the spatial (rotation) angle-defect, Im ε = the temporal (boost)
        # content — never Re-only.
        re_parts = [d.real for d in deficits]
        im_parts = [d.imag for d in deficits]

        def stats(parts):
            if not parts:
                return {"min": None, "mean": None, "max": None}
            return {"min": min(parts), "mean": sum(parts) / len(parts),
                    "max": max(parts)}

        return {
            "cells": len(st.getTopSimplices()),
            "edges": len(squared),
            "re_l2_min": min(l.real for l in squared),
            "re_l2_max": max(l.real for l in squared),
            "im_l2_max": max(abs(l.imag) for l in squared),
            "regge_grad_norm2": float(
                cob.MultiCobordism.regge_action_gradient(st)),
            "hinges": len(deficits),
            "deficit_spatial_re": stats(re_parts),
            "deficit_temporal_im": stats(im_parts),
            "dual_volume_total": sum(dual),
        }

    def spectral():
        sigmas = [0.5 * 2 ** i for i in range(8)]           # 0.5 .. 64
        top_k = len(next(iter(st.getTopSimplices())).getVertices()) - 1
        ds = st.getSpectralDimensionOnSkeleton(
            sigmas, 64, tessera.AllSimplexFilter(), top_k, 1)
        return {"sigmas": sigmas, "spectral_dimension": list(ds)}

    def wilson():
        # The deficit-angle (spin-connection) mode — the #477 register-spin readout.
        loop = tessera.WilsonLoop(st)
        loop.measureAllHinges(tessera.WilsonMode.DEFICIT_ANGLE)
        averages = loop.getAverageBySize()
        try:
            averages = {str(k): float(v) for k, v in dict(averages).items()}
        except (TypeError, ValueError):
            averages = [float(v) for v in averages]
        return {"deficit_angle_average_by_size": averages}

    block("color", color)
    block("geometry", geometry)
    block("spectral", spectral)
    block("wilson", wilson)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild a recorded campaign attempt and run the ready "
                    "observables on it.")
    ap.add_argument("--seed", type=int,
                    help="base seed of the recorded attempt (rebuilds from its dump)")
    ap.add_argument("--from-dump", dest="from_dump",
                    help="geometry dump to rebuild from")
    ap.add_argument("--replay", action="store_true",
                    help="with --seed: re-RUN the attempt with the frozen drive "
                         "instead of loading the dump — a fresh sample of the seed's "
                         "attempt distribution (the build is not process-"
                         "deterministic); the verdict comparison is a divergence "
                         "measurement")
    ap.add_argument("--run-dir", default=".overnight", dest="run_dir")
    ap.add_argument("--out", help="analysis JSON path (default: "
                                  "<run-dir>/analysis_seed_<seed>.json)")
    args = ap.parse_args()
    if args.seed is None and not args.from_dump:
        ap.error("one of --seed / --from-dump is required")

    analysis = {"seed": args.seed}
    dump = None
    dump_path = args.from_dump or (args.seed is not None
                                   and dump_path_for(args.run_dir, args.seed))
    if dump_path:
        with open(dump_path) as fh:
            dump = json.load(fh)
        analysis["dump"] = dump_path
        analysis.setdefault("seed", dump.get("base_seed"))

    verdict = (find_verdict(args.run_dir, analysis["seed"])
               if analysis["seed"] is not None else None)
    if verdict:
        analysis["verdict"] = {k: verdict[k] for k in
                               (*REPLAY_KEYS, "worker", "elapsed_s")
                               if k in verdict}

    if args.replay and args.seed is not None:
        print(f"re-running seed {args.seed} with the frozen drive — a fresh sample "
              f"of this seed's attempt distribution, NOT a reconstruction "
              f"(expect roughly the attempt's original wall time) ...", flush=True)
        result, st, _state = replay(args.seed)
        analysis["rebuild"] = "replay (fresh sample — not a reconstruction)"
        if verdict:
            divergence = compare(verdict, result, REPLAY_KEYS)
            analysis["replay_equals_verdict"] = not divergence
            if divergence:
                analysis["replay_divergence"] = {
                    k: {"recorded": a, "replayed": b}
                    for k, (a, b) in divergence.items()}
    else:
        if not dump:
            raise SystemExit(
                f"no geometry dump for seed {analysis['seed']} (dumps are written "
                f"per attempt going forward; older attempts kept them only under "
                f"the GIF keep-policy) — use --replay for a fresh sample instead")
        st = rebuild_from_dump(dump)
        analysis["rebuild"] = "dump"
        state_mismatches = verify_state_against_dump(st, dump)
        analysis["rebuild_matches_dump"] = not state_mismatches
        if state_mismatches:
            analysis["dump_mismatches"] = state_mismatches

    analysis["observables"] = characterize(st)

    out = args.out or os.path.join(
        args.run_dir, f"analysis_seed_{analysis['seed']}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(analysis, fh, indent=2)
    os.replace(tmp, out)
    print(f"analysis -> {out}")
    verified = analysis.get("rebuild_matches_dump",
                            analysis.get("replay_equals_verdict"))
    color = analysis["observables"].get("color", {})
    print(f"verified={verified}  betti={color.get('betti')}  "
          f"singlet_residual={color.get('singlet_residual')}")


if __name__ == "__main__":
    main()
