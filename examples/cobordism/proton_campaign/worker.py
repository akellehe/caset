# Throwaway sweep worker (#555): one emergent-arm attempt per base seed, driven
# stage-by-stage (mirroring ProtonIngredients.build with max_restarts=1) so the
# unbounded stage-2 descent streams a progress line every chunk, and — for the
# interesting attempts — records an animation in the style of the animation scripts
# (renderer.py reuses emergent_proton's panels; frames land at the real pass/chunk
# boundaries, so the GIF shows the exact recorded attempt).
#
# Machine-precision semantics: stage 2 is chunked, and a chunk that stops early on
# runStage2's built-in relTol=1e-9 stationarity test IS the stationarity verdict.
# Persistence: up to 3 continued evolve+relax passes; the last must leave holes, b_k,
# and F stable to 1e-9 relative. Nothing here steers toward any target — the register
# count is observed, never optimized for.
#
# Keep-policy for animations (disk-bounded): keep the GIF when the attempt ever showed
# b3 >= 2 or >= 3 holes, or ended with >= 2 holes; any b3 >= 3 sighting is kept
# unconditionally. Every attempt remains exactly reproducible from its recorded base
# seed regardless (single-threaded determinism), so a discarded 1-hole GIF loses
# nothing.
import argparse
import json
import os
import resource
import time

import tessera

cob = tessera.cobordism

INIT_STEPS = 180
EVOLVE_STEPS = 60
CANDIDATES = 8
PATIENCE = 15
STAGE2_BETA = 1.0
STAGE2_CHUNK = 25            # progress granularity; the stationarity test is the exit
PERSIST_PASSES = 3
PERSIST_REL_TOL = 1e-9
REGISTER_DEGREE = 3
PROGRESS_BYTE_CAP = 50 * 1024 * 1024        # per worker
GIF_BYTE_CAP = 5 * 1024 * 1024 * 1024       # global, across all workers

STEP_A_LABEL = "Step A — recombination (→ diquark {1, ω})"
STEP_B_LABEL = "Step B — formation (nothing pinned — final state emerges)"


def snapshot(node):
    st = node.st
    betti = cob.MultiCobordism.betti(st)
    squared = [e.getSquaredLength() for e in st.getEdgeList().toVector()]
    return {
        "F": float(node.objective()),
        "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
        "rU": float(node.r_u(st)),
        "holes": len(cob.MultiCobordism.emergent_holes(st, REGISTER_DEGREE)),
        "b3": int(betti[REGISTER_DEGREE]) if len(betti) > REGISTER_DEGREE else 0,
        "cells": len(st.getTopSimplices()),
        "edges": len(squared),
        "re_min": round(min(l.real for l in squared), 6),
        "re_max": round(max(l.real for l in squared), 6),
        "im_max": max(abs(l.imag) for l in squared),
    }


def gif_dir_full(gif_dir):
    total = sum(os.path.getsize(os.path.join(gif_dir, f))
                for f in os.listdir(gif_dir) if f.endswith(".gif"))
    return total > GIF_BYTE_CAP


class AttemptState:
    """Running extremes the keep-policy and the final record read."""

    def __init__(self):
        self.max_b3 = 0
        self.max_holes = 0

    def see(self, snap):
        self.max_b3 = max(self.max_b3, snap["b3"])
        self.max_holes = max(self.max_holes, snap["holes"])


def run_attempt_on_nodes(base, progress, recorder, state, nodes):
    """The attempt physics on already-built nodes: init → evolve → stage-2 to genuine
    stationarity per node, then the persistence loop on step B. Identical drive with
    and without a recorder — frames are taken between engine calls, never instead."""
    step_a, step_b = nodes[0][0], nodes[1][0]

    def frame(node_index, phase, subtitle=""):
        if recorder:
            recorder.frame(node_index, phase, subtitle)

    def stage2_to_stationarity(node, node_index, node_tag):
        iters_done = 0
        previous_f = float(node.objective())
        while True:
            t0 = time.time()
            node.run_stage2(beta=STAGE2_BETA, max_iters=STAGE2_CHUNK)
            iters_done += STAGE2_CHUNK
            stationary = bool(node.last_stage2_stationary)
            snap = snapshot(node)
            state.see(snap)
            progress(dict(base_seed=base, node=node_tag, phase="stage2",
                          iters=iters_done, dF=previous_f - snap["F"],
                          stationary=stationary,
                          chunk_s=round(time.time() - t0, 1), **snap))
            frame(node_index, "stage2", f" · iter {iters_done}")
            previous_f = snap["F"]
            if stationary:
                return iters_done

    def run_node(node, node_index, node_tag):
        t0 = time.time()
        node.run_stage1(max_steps=INIT_STEPS, n_candidate_moves=CANDIDATES,
                        patience=PATIENCE, grow_boundaries=True)
        snap = snapshot(node)
        state.see(snap)
        progress(dict(base_seed=base, node=node_tag, phase="init",
                      pass_s=round(time.time() - t0, 1), **snap))
        frame(node_index, "init")
        t0 = time.time()
        node.run_stage1(max_steps=EVOLVE_STEPS, n_candidate_moves=CANDIDATES,
                        patience=PATIENCE, grow_boundaries=False)
        snap = snapshot(node)
        state.see(snap)
        progress(dict(base_seed=base, node=node_tag, phase="evolve",
                      pass_s=round(time.time() - t0, 1), **snap))
        frame(node_index, "evolve")
        return stage2_to_stationarity(node, node_index, node_tag)

    progress({"base_seed": base, "phase": "attempt_start"})
    frame(0, "seed")
    frame(1, "seed")
    run_node(step_a, 0, "A")
    diquark_ru = float(step_a.r_u(step_a.st))
    stage2_iters = run_node(step_b, 1, "B")

    persistent = False
    for persist_pass in range(1, PERSIST_PASSES + 1):
        before = snapshot(step_b)
        step_b.run_stage1(max_steps=EVOLVE_STEPS, n_candidate_moves=CANDIDATES,
                          patience=PATIENCE, grow_boundaries=False)
        stage2_iters += stage2_to_stationarity(step_b, 1, "B")
        after = snapshot(step_b)
        state.see(after)
        stable_f = abs(after["F"] - before["F"]) <= PERSIST_REL_TOL * max(
            abs(before["F"]), 1.0)
        persistent = (after["holes"] == before["holes"]
                      and after["b3"] == before["b3"] and stable_f)
        progress(dict(base_seed=base, node="B", phase="persistence",
                      persist_pass=persist_pass, persistent=persistent,
                      dF=before["F"] - after["F"], **after))
        frame(1, "persistence", f" · pass {persist_pass}")
        if persistent:
            break

    stationary = bool(step_b.last_stage2_stationary)
    st = step_b.st
    squared = [e.getSquaredLength() for e in st.getEdgeList().toVector()]
    final = snapshot(step_b)
    return {
        "converged": stationary and persistent,
        "stationary": stationary,
        "persistent": persistent,
        "stage2_iters_total": stage2_iters,
        "holes": final["holes"],
        "max_holes": state.max_holes,
        "max_b3": state.max_b3,
        "betti": list(cob.MultiCobordism.betti(st)),
        "singlet": float(cob.MultiCobordism.r_state(st, REGISTER_DEGREE,
                                                    cob.Proton.singlet())),
        "input_ru": final["rU"],
        "F": final["F"],
        "cells": final["cells"],
        "edges": final["edges"],
        "re_min": min(l.real for l in squared),
        "re_max": max(l.real for l in squared),
        "im_max": max(abs(l.imag) for l in squared),
        "diquark_ru": diquark_ru,
    }


def run_one(base, progress, args, frame_dir, gif_dir):
    """Build the attempt's nodes, run it (recorded when --animate), apply the GIF
    keep-policy, return the result record fields."""
    ingredients = cob.ProtonIngredients(seed=base)
    nodes = [(ingredients.recombination_node(base), STEP_A_LABEL),
             (ingredients.formation_node(base + 1), STEP_B_LABEL)]
    state = AttemptState()
    recorder = None
    if args.animate:
        try:
            import renderer
            recorder = renderer.AttemptRecorder(nodes, frame_dir)
        except Exception:
            recorder = None      # rendering is best-effort; physics always runs
    try:
        result = run_attempt_on_nodes(base, progress, recorder, state, nodes)
    except Exception:
        if recorder:
            recorder.finish(None)
        raise
    if recorder:
        keep = (state.max_b3 >= 2 or result["holes"] >= 2 or state.max_holes >= 3)
        gif_path = None
        if keep and not gif_dir_full(gif_dir):
            tag = "CONVERGED" if result["converged"] else "unconverged"
            gif_path = os.path.join(
                gif_dir,
                f"seed_{base}_b3max{state.max_b3}_holes{result['holes']}_{tag}.gif")
        result["gif"] = recorder.finish(gif_path)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--deadline", type=float, required=True)   # epoch; gates NEW attempts
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--animate", action="store_true",
                    help="record per-attempt frames; GIFs kept by the keep-policy")
    args = ap.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.out))
    gif_dir = os.path.join(run_dir, "animations")
    frame_dir = os.path.join(run_dir, f"tmp_frames_w{args.worker}")
    os.makedirs(gif_dir, exist_ok=True)

    progress_path = args.out.replace(".jsonl", ".progress.jsonl")
    base = args.seed_base
    # Restart-safe: after a reboot/relaunch, skip every base seed already recorded so
    # the campaign never re-runs (and double-counts) an attempt.
    recorded = set()
    if os.path.exists(args.out):
        with open(args.out) as previous:
            for line in previous:
                try:
                    recorded.add(json.loads(line).get("base_seed"))
                except Exception:
                    pass
    with open(args.out, "a", buffering=1) as out, \
            open(progress_path, "a", buffering=1) as prog:
        capped = False

        def progress(record):
            nonlocal capped
            if capped:
                return
            if prog.tell() > PROGRESS_BYTE_CAP:
                prog.write(json.dumps({"worker": args.worker,
                                       "progress_capped": True}) + "\n")
                capped = True
                return
            record["worker"] = args.worker
            record["t"] = round(time.time())
            prog.write(json.dumps(record) + "\n")

        while time.time() < args.deadline:
            while base in recorded:
                base += 2
            started = time.time()
            record = {"worker": args.worker, "base_seed": base}
            try:
                record.update(run_one(base, progress, args, frame_dir, gif_dir))
            except Exception as error:   # record and move on — the sweep must survive
                record["error"] = repr(error)
            record["elapsed_s"] = round(time.time() - started, 1)
            record["rss_mb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            out.write(json.dumps(record) + "\n")
            base += 2
    with open(args.out + ".done", "w") as marker:
        marker.write("done\n")


if __name__ == "__main__":
    main()
