# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The #560 experiment: the joint 3-pair formation node WITH the final-state knob.

ONE co-optimized ``MultiCobordism`` (the #489 shape, through the canonical engine),
assembled here in the experiment layer purely from the engine's PUBLIC surface —
``Proton``, ``ProtonIngredients``, and ``MultiCobordism`` are not modified:

  * inputs  — the Z₃-orbit neutral triple ``{1,-1,0} ⊔ {0,1,-1} ⊔ {-1,0,1}``
    (each Σ = 0), seeded at v0/v1/v2 of the single Δ⁴ seed, held representable
    through their r_U terms for the whole build;
  * outputs — PINNED: the baryon ``{1,ω,ω²}`` seeded at v3 and the antibaryon
    ``{1,ω̄,ω̄²}`` at v4 (ω = exp(2πi/3)).

Exactly one variable differs from the canonical ``Proton.build()`` — the event
graph (joint vs two-step) — and exactly one from the campaign's joint
inputs-only node (``ProtonIngredients.joint_node``) — the final-state knob
(pinned vs empty outputs). That makes this node the control arm between the
two: it can never claim an emergent singlet (the answer is pinned), but it
measures joint-shape FEASIBILITY independent of basin rarity, and its A/B
against the canonical two-step isolates what the event graph alone changes.

Microcausality (the #559 decision log, recorded here as the experiment runs):
causal structure lives in the MOVE HISTORY — every accepted change is one gated
local move — not in the boundary-block count, so the joint node is physically
admissible.

The drive is the campaign worker's recipe verbatim (init pass with
``grow_boundaries=True`` → evolution pass with ∂W frozen → stage-2 chunks to
genuine stationarity → persistence passes), with an explicit stage-2 iteration
cap for battery-scale runs (a cap-stop is recorded as ``stationary=False`` —
never promoted). The engine build is NOT process-deterministic: a base seed
labels an attempt, it does not reproduce it; every attempt therefore writes a
schema-1 geometry dump (the frozen campaign writer) whose metadata carries the
output-block provenance, so ``observe_proton_ingredients.py --geometry`` can
run the C++ observable battery (``BlockResiduals`` included) on any attempt
after the fact.

Subcommands:
  selfcheck            node/host shape assertions + r_U input-weight linearity
  attempt              one attempt (one arm, one seed) → one JSON verdict line
  aggregate            criteria + rates over verdict JSONL (battery or calibration)

Parallel batteries are launched by the caller, one single-threaded process per
attempt, e.g.:
  printf '%s\\n' 101 103 105 | OMP_NUM_THREADS=1 xargs -P 12 -I{} \\
    python joint_pinned_proton.py attempt --arm joint-pinned --seed {} --out b.jsonl
Raw verdicts/dumps are issue artifacts (#560), not repo content.
"""
import argparse
import cmath
import json
import math
import os
import sys
import time

import tessera

cob = tessera.cobordism

# The frozen campaign scripts own the schema-1 geometry-dump writer; reuse it
# rather than fork the format (worker.py imports tessera and nothing heavier).
_CAMPAIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "proton_campaign")
if _CAMPAIGN_DIR not in sys.path:
    sys.path.insert(0, _CAMPAIGN_DIR)
import worker as campaign_worker  # noqa: E402  (path insert above)

REGISTER_DEGREE = 3
# The engine's own defaults (Proton/ProtonIngredients ctor): the A/B varies the
# event graph, never the knobs.
GAMMA = 50.0
INPUT_WEIGHT = 20.0
# The campaign worker's drive constants, verbatim.
INIT_STEPS = 180
EVOLVE_STEPS = 60
CANDIDATES = 8
PATIENCE = 15
STAGE2_BETA = 1.0
STAGE2_CHUNK = 25
PERSIST_PASSES = 3
PERSIST_REL_TOL = 1e-9
# Battery-scale stationarity budget: the campaign observed 325–6650 stage-2
# iterations to genuine stationarity on the inputs-only joint node; a cap-stop
# below is recorded honestly as stationary=False.
STAGE2_ITER_CAP = 2500

CRITERIA_RESIDUAL_TOL = 0.5      # the ticket's baryon-block singlet bar
CRITERIA_CLUSTER_HOLES = 3       # ≥3 emergent holes on the baryon block
CRITERIA_CHARGE_SPREAD = 0.1     # radians of per-hole Z₃ phase spread


def z3_pairs():
    """The Z₃-orbit neutral triple — each pair sums to zero (a q-q̄ pair), and
    cycling the three color slots permutes the pairs into each other."""
    return [[complex(1), complex(-1), complex(0)],
            [complex(0), complex(1), complex(-1)],
            [complex(-1), complex(0), complex(1)]]


def baryon_target():
    """The color singlet {1, ω, ω²} — the canonical proton target."""
    return [complex(z) for z in cob.Proton.singlet()]


def antibaryon_target():
    """The conjugate singlet {1, ω̄, ω̄²} — the CPT partner block's target."""
    return [complex(z).conjugate() for z in cob.Proton.singlet()]


def minimal_seed_host():
    """The engine's minimal seed, rebuilt from the public surface: one Δ⁴
    pentatope with the uniform all-spacelike metric (ℓ² = +1). Mirrors the
    (private) ``Proton::buildMinimalSeed`` — deliberately all-spacelike: at
    initialization no time has passed, so no causal structure is put in by
    hand; any causal content must emerge."""
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    host = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, tessera.SolidSimplex(4))
    host.build()
    for edge in host.getEdgeList().toVector():
        edge.setSquaredLength(complex(1.0, 0.0))
    return host


def joint_pinned_node(seed, register_degree=REGISTER_DEGREE, gamma=GAMMA,
                      input_weight=INPUT_WEIGHT, precone=0):
    """The #560 node, seeded and NOT yet run: three Z₃ pair inputs at v0/v1/v2,
    baryon ⊔ antibaryon output targets at v3/v4, on the single Δ⁴ seed. The
    caller drives it (run_stage1/run_stage2), exactly like the engine's own
    node factories."""
    host = minimal_seed_host()
    # Seed vertex ids in intrinsic list order (never sorted): v0..v4 of the Δ⁴.
    seed_ids = [v.getId() for v in host.getVertexList().toVector()]
    node = cob.MultiCobordism(host, z3_pairs(),
                              [baryon_target(), antibaryon_target()],
                              degrees=[register_degree], gamma=gamma,
                              seed=seed, precone=precone)
    node.set_input_residual_weight(input_weight)
    node.seed_inputs(seed_ids[:3])
    node.seed_outputs(seed_ids[3:5])
    return node


def two_step_nodes(base_seed, register_degree=REGISTER_DEGREE, gamma=GAMMA,
                   input_weight=INPUT_WEIGHT, precone=0):
    """The canonical arm's two nodes for one attempt, from ``Proton``'s own
    factories (A-seed = base, B-seed = base+1, the build's restart-0 pairing)."""
    proton = cob.Proton(seed=base_seed, register_degree=register_degree,
                        gamma=gamma, input_weight=input_weight, precone=precone)
    return proton.recombination_node(base_seed), proton.formation_node(base_seed + 1)


def snapshot(node, register_degree=REGISTER_DEGREE):
    """The campaign worker's answer-agnostic summary of a node's state."""
    st = node.st
    betti = cob.MultiCobordism.betti(st)
    squared = [e.getSquaredLength() for e in st.getEdgeList().toVector()]
    return {
        "F": float(node.objective()),
        "rU": float(node.r_u(st)),
        "holes": len(cob.MultiCobordism.emergent_holes(st, register_degree)),
        "b3": int(betti[register_degree]) if len(betti) > register_degree else 0,
        "cells": len(st.getTopSimplices()),
        "edges": len(squared),
        "re_min": min(l.real for l in squared),
        "re_max": max(l.real for l in squared),
        "im_max": max(abs(l.imag) for l in squared),
    }


class _Extremes:
    """Running maxima the keep-policy-style fields report."""

    def __init__(self):
        self.max_b3 = 0
        self.max_holes = 0

    def see(self, snap):
        self.max_b3 = max(self.max_b3, snap["b3"])
        self.max_holes = max(self.max_holes, snap["holes"])


def stage2_to_stationarity(node, extremes, iter_cap, chunk=STAGE2_CHUNK,
                           beta=STAGE2_BETA):
    """Chunked stage-2 descent until the engine's built-in relTol stationarity
    test fires, or the iteration cap (recorded as non-stationary — a budget
    stop is never promoted to a verdict)."""
    iters = 0
    while True:
        node.run_stage2(beta=beta, max_iters=chunk)
        iters += chunk
        extremes.see(snapshot(node))
        if node.last_stage2_stationary:
            return iters, True
        if iters >= iter_cap:
            return iters, False


def drive_node(node, extremes, init_steps, evolve_steps, iter_cap):
    """The engine drive, campaign-worker recipe: INITIALIZATION pass
    (grow_boundaries=True) → EVOLUTION pass (∂W frozen) → stage-2 to
    stationarity."""
    node.run_stage1(max_steps=init_steps, n_candidate_moves=CANDIDATES,
                    patience=PATIENCE, grow_boundaries=True)
    extremes.see(snapshot(node))
    node.run_stage1(max_steps=evolve_steps, n_candidate_moves=CANDIDATES,
                    patience=PATIENCE, grow_boundaries=False)
    extremes.see(snapshot(node))
    return stage2_to_stationarity(node, extremes, iter_cap)


def persistence_loop(node, extremes, evolve_steps, iter_cap,
                     passes=PERSIST_PASSES, rel_tol=PERSIST_REL_TOL):
    """Up to ``passes`` continued evolve+relax passes; the LAST must leave the
    answer-agnostic summary (holes, b₃, F) stable. Returns (persistent,
    extra stage-2 iters, final stationarity of the last relax)."""
    persistent, extra_iters, stationary = False, 0, False
    for _ in range(passes):
        before = snapshot(node)
        node.run_stage1(max_steps=evolve_steps, n_candidate_moves=CANDIDATES,
                        patience=PATIENCE, grow_boundaries=False)
        iters, stationary = stage2_to_stationarity(node, extremes, iter_cap)
        extra_iters += iters
        after = snapshot(node)
        stable_f = abs(after["F"] - before["F"]) <= rel_tol * max(
            abs(before["F"]), 1.0)
        persistent = (after["holes"] == before["holes"]
                      and after["b3"] == before["b3"] and stable_f)
        if persistent:
            break
    return persistent, extra_iters, stationary


def block_read(st, vertices, target, register_degree=REGISTER_DEGREE):
    """One provenance block's read, mirroring the landed ``BlockResiduals``
    scoring: the ambient top cells whose vertices ALL lie in the block's region
    form its own sub-complex (uniform metric, via the canonical ``fromCells``),
    scored with the relabeling-invariant r_state; an empty region reports the
    full leak ‖target‖². Also reports the sub-complex's own emergent holes —
    the ticket's \"holes clustered on the block's region\" count."""
    region = set(int(v) for v in vertices)
    inside = [[int(v.getId()) for v in c.getVertices()]
              for c in st.getTopSimplices()
              if all(int(v.getId()) in region for v in c.getVertices())]
    leak = sum(abs(z) ** 2 for z in target)
    if not inside:
        return {"n_cells_in_region": 0, "holes_in_region": 0,
                "residual": leak, "full_leak": leak}
    sub = tessera.spacetime.Spacetime.fromCells(len(inside[0]) - 1, inside)
    return {
        "n_cells_in_region": len(inside),
        "holes_in_region": len(cob.MultiCobordism.emergent_holes(
            sub, register_degree)),
        "residual": float(cob.MultiCobordism.r_state(
            sub, register_degree, list(target))),
        "full_leak": leak,
    }


def flavor_read(st, register_degree=REGISTER_DEGREE):
    """The landed C++ ``PairLoopFlavor`` observable (#576/#594) on the relaxed
    whole: ONE correlated multi-hole read whose record carries the per-hole DK
    charges ``q`` (q_h = Σ_{c∈∂h} W_c |ψ_c|², orientation-signed and
    gauge/relabel-fixed), the pair-loop charges, ρ, and the dual residuals.
    This is the ticket's per-hole-charge read, exactly as written — no
    substitution. Requires 3 register holes; below that the observable's own
    skip reason is recorded."""
    observables = tessera.observables
    available = len(cob.MultiCobordism.emergent_holes(st, register_degree))
    context = observables.RegisterContext(st, min(available, 3),
                                          register_degree, cob.Proton.singlet())
    flavor = observables.PairLoopFlavor()
    reason = flavor.skip_reason(context)
    if reason:
        return {"evaluable": False, "n_holes": available, "reason": str(reason)}
    record = dict(flavor.record(context))
    record["evaluable"] = True
    record["n_holes"] = available
    return record


def charge_probe_pairwise(st, register_degree=REGISTER_DEGREE):
    """DIAGNOSTIC ONLY (never a criterion): the pairwise period fit — hole 0 as
    reference, hole j's best-fitting relative phase over {1, ω, ω²}. When the
    carried period space is rich enough, a 2-hole fit is exactly solvable for
    EVERY phase (measured: winner→runner-up margins ~1e-29 on a b₃=2 campaign
    specimen), so the argmin is noise; ``degenerate`` flags that honestly and
    the criterion read lives in ``flavor_read`` instead."""
    holes = [list(h) for h in
             cob.MultiCobordism.emergent_holes(st, register_degree)]
    if len(holes) < 2:
        return {"evaluable": False, "n_holes": len(holes)}
    es = cob.EigenstateSynthesis(st, register_degree)
    omega = cob.Proton.omega()
    charges, margins = [0], []
    for hole in holes[1:]:
        residuals = [float(es.residualForPeriods([holes[0], hole],
                                                 [complex(1.0), omega ** m]))
                     for m in range(3)]
        ranked = sorted(range(3), key=residuals.__getitem__)
        charges.append(ranked[0])
        margins.append(residuals[ranked[1]] - residuals[ranked[0]])
    return {
        "evaluable": True,
        "n_holes": len(holes),
        "charges": charges,                    # units of 2π/3, hole 0 = reference
        "margins": margins,                    # residual gap winner → runner-up
        "degenerate": bool(margins and min(margins) < 1e-9),
        "unit_carry": [float(es.residualForPeriods([h], [1.0])) for h in holes],
    }


def whole_reads(node, register_degree=REGISTER_DEGREE):
    """The answer-agnostic verdict fields plus the singlet/conjugate
    diagnostics, exactly as the campaign records them — reads, never drives."""
    st = node.st
    final = snapshot(node, register_degree)
    return {
        "holes": final["holes"],
        "betti": [int(b) for b in cob.MultiCobordism.betti(st)],
        "singlet": float(cob.MultiCobordism.r_state(
            st, register_degree, baryon_target())),
        "singlet_conj": float(cob.MultiCobordism.r_state(
            st, register_degree, antibaryon_target())),
        "r_u": final["rU"],
        "F": final["F"],
        "cells": final["cells"],
        "edges": final["edges"],
        "re_min": final["re_min"],
        "re_max": final["re_max"],
        "im_max": final["im_max"],
    }


def _blocks_provenance(node):
    """The live node's output blocks as observe_proton_ingredients.py
    provenance (labels in JOINT_BLOCK_LABELS order: baryon, antibaryon)."""
    blocks = []
    for label, block in zip(("baryon", "antibaryon"), node.outputs):
        target = list(block.target)
        blocks.append({
            "label": label,
            "vertices": [int(v) for v in block.vertices],
            "target_re": [z.real for z in target],
            "target_im": [z.imag for z in target],
        })
    return blocks


def attempt_joint_pinned(seed, gamma, input_weight, budgets, geom_dir=None):
    """One joint-pinned attempt: build, drive, persist, read. Returns the
    verdict record (one JSON line)."""
    extremes = _Extremes()
    node = joint_pinned_node(seed, gamma=gamma, input_weight=input_weight)
    started = time.time()
    iters, _ = drive_node(node, extremes, budgets["init_steps"],
                          budgets["evolve_steps"], budgets["stage2_iter_cap"])
    persistent, extra, stationary = persistence_loop(
        node, extremes, budgets["evolve_steps"], budgets["stage2_iter_cap"],
        budgets["persist_passes"])
    record = {
        "arm": "joint-pinned",
        "base_seed": seed,
        "gamma": gamma,
        "input_weight": input_weight,
        "converged": bool(stationary and persistent),
        "stationary": bool(stationary),
        "persistent": bool(persistent),
        "stage2_iters_total": iters + extra,
        "max_holes": extremes.max_holes,
        "max_b3": extremes.max_b3,
        "elapsed_s": round(time.time() - started, 1),
    }
    record.update(whole_reads(node))
    st = node.st
    blocks = _blocks_provenance(node)
    for spec in blocks:
        target = [complex(re, im) for re, im in
                  zip(spec["target_re"], spec["target_im"])]
        try:
            record[f"{spec['label']}_block"] = block_read(
                st, spec["vertices"], target)
        except Exception as error:      # best-effort read, analyzer-style
            record[f"{spec['label']}_block"] = {"error": repr(error)}
    try:
        record["flavor"] = flavor_read(st)
    except Exception as error:
        record["flavor"] = {"evaluable": False, "error": repr(error)}
    try:
        record["charge_probe"] = charge_probe_pairwise(st)
    except Exception as error:
        record["charge_probe"] = {"error": repr(error)}
    if geom_dir:
        meta = {"arm": "joint-pinned", "base_seed": seed, "gamma": gamma,
                "input_weight": input_weight, "blocks": blocks}
        meta.update({k: record[k] for k in
                     ("converged", "stationary", "persistent", "holes",
                      "betti", "singlet", "singlet_conj", "F")})
        record["geometry"] = campaign_worker.dump_geometry(
            st, os.path.join(geom_dir, f"joint_pinned_seed_{seed}_geometry.json"),
            meta)
    return record


def attempt_two_step(seed, gamma, input_weight, budgets, geom_dir=None):
    """One canonical two-step attempt (A then B) under the IDENTICAL drive and
    budgets — the A/B partner. Reads are on step B's whole, as the canonical
    build reads them."""
    extremes = _Extremes()
    step_a, step_b = two_step_nodes(seed, gamma=gamma, input_weight=input_weight)
    started = time.time()
    iters_a, stationary_a = drive_node(step_a, _Extremes(),
                                       budgets["init_steps"],
                                       budgets["evolve_steps"],
                                       budgets["stage2_iter_cap"])
    diquark_ru = float(step_a.r_u(step_a.st))
    iters_b, _ = drive_node(step_b, extremes, budgets["init_steps"],
                            budgets["evolve_steps"], budgets["stage2_iter_cap"])
    persistent, extra, stationary = persistence_loop(
        step_b, extremes, budgets["evolve_steps"], budgets["stage2_iter_cap"],
        budgets["persist_passes"])
    record = {
        "arm": "two-step",
        "base_seed": seed,
        "gamma": gamma,
        "input_weight": input_weight,
        "converged": bool(stationary and persistent),
        "stationary": bool(stationary),
        "persistent": bool(persistent),
        "stationary_a": bool(stationary_a),
        "diquark_ru": diquark_ru,
        "stage2_iters_total": iters_a + iters_b + extra,
        "max_holes": extremes.max_holes,
        "max_b3": extremes.max_b3,
        "elapsed_s": round(time.time() - started, 1),
    }
    record.update(whole_reads(step_b))
    # The canonical arm's own answer-shaped gate, reported for reference (the
    # whole carries the singlet: colorResidual < tol with ≥ 3 holes).
    record["canonical_converged"] = bool(
        record["singlet"] < CRITERIA_RESIDUAL_TOL
        and record["holes"] >= CRITERIA_CLUSTER_HOLES)
    try:
        record["flavor"] = flavor_read(step_b.st)
    except Exception as error:
        record["flavor"] = {"evaluable": False, "error": repr(error)}
    try:
        record["charge_probe"] = charge_probe_pairwise(step_b.st)
    except Exception as error:
        record["charge_probe"] = {"error": repr(error)}
    if geom_dir:
        meta = {"arm": "two-step", "base_seed": seed, "gamma": gamma,
                "input_weight": input_weight}
        meta.update({k: record[k] for k in
                     ("converged", "stationary", "persistent", "holes",
                      "betti", "singlet", "singlet_conj", "F")})
        record["geometry"] = campaign_worker.dump_geometry(
            step_b.st,
            os.path.join(geom_dir, f"two_step_seed_{seed}_geometry.json"), meta)
    return record


def evaluate_criteria(record):
    """The ticket's pre-registered per-attempt criteria on a joint-pinned
    verdict (rate criteria are aggregate-level). Each is True/False/None
    (None = not evaluable on this attempt, with the reason recorded)."""
    baryon = record.get("baryon_block", {})
    antibaryon = record.get("antibaryon_block", {})
    flavor = record.get("flavor", {})
    q = flavor.get("q") if flavor.get("evaluable") else None
    criteria = {
        "holes_clustered_on_baryon": (
            baryon.get("holes_in_region", 0) >= CRITERIA_CLUSTER_HOLES
            if "holes_in_region" in baryon else None),
        "baryon_singlet_lt_tol": (
            baryon["residual"] < CRITERIA_RESIDUAL_TOL
            if "residual" in baryon else None),
        "antibaryon_conjugate_lt_tol": (
            antibaryon["residual"] < CRITERIA_RESIDUAL_TOL
            if "residual" in antibaryon else None),
        "charge_spread_ge_tol": (
            (max(q) - min(q)) >= CRITERIA_CHARGE_SPREAD
            if q else None),
    }
    return criteria


def selfcheck():
    """Shape + arithmetic assertions on the un-run node (deterministic: no
    stage has run, so no RNG draw has been consumed)."""
    host = minimal_seed_host()
    cells = host.getTopSimplices()
    assert len(cells) == 1, f"minimal seed has {len(cells)} top cells, want 1"
    vertex_ids = [v.getId() for v in host.getVertexList().toVector()]
    assert len(vertex_ids) == 5, f"minimal seed has {len(vertex_ids)} vertices"
    edges = host.getEdgeList().toVector()
    assert len(edges) == 10, f"minimal seed has {len(edges)} edges, want 10"
    assert all(e.getSquaredLength() == complex(1.0, 0.0) for e in edges), \
        "minimal seed metric is not the uniform all-spacelike l^2=+1"
    assert [int(b) for b in cob.MultiCobordism.betti(host)] == [1, 0, 0, 0, 0]

    node = joint_pinned_node(7)
    inputs, outputs = list(node.inputs), list(node.outputs)
    assert len(inputs) == 3 and len(outputs) == 2, \
        f"node has {len(inputs)} inputs / {len(outputs)} outputs, want 3/2"
    for block, want in zip(inputs, z3_pairs()):
        assert list(block.target) == want, "input target mismatch"
    assert list(outputs[0].target) == baryon_target(), "baryon target mismatch"
    assert list(outputs[1].target) == antibaryon_target(), \
        "antibaryon target mismatch"
    # seedBlocks starts each block's region as its seed vertex's CELL
    # neighbourhood; on the one-cell Δ⁴ that is the whole pentatope for every
    # block (block identity lives in the target pairing; regions differentiate
    # as runStage1 grows them) — identical to the engine's own node factories.
    node_vertex_ids = {v.getId() for v in node.st.getVertexList().toVector()}
    assert len(node_vertex_ids) == 5
    for block in inputs + outputs:
        assert set(block.vertices) == node_vertex_ids, \
            "un-run block region must be the Δ⁴ cell-neighbourhood"

    # r_U is affine in the input weight: r_u = r_out + w * r_in_sum, so equal
    # weight steps give equal r_u steps (machine precision on the un-run node).
    values = []
    for weight in (5.0, 10.0, 15.0):
        probe = joint_pinned_node(7, input_weight=weight)
        values.append(probe.r_u(probe.st))
    step1, step2 = values[1] - values[0], values[2] - values[1]
    assert abs(step1 - step2) <= 1e-12 * max(1.0, abs(step1)), \
        f"r_U not affine in input weight: steps {step1} vs {step2}"

    # A short drive runs end-to-end and the objective stays finite.
    node.run_stage1(max_steps=5, n_candidate_moves=4, patience=3,
                    grow_boundaries=True)
    node.run_stage2(beta=STAGE2_BETA, max_iters=5)
    objective = float(node.objective())
    assert math.isfinite(objective), "objective is not finite after a short drive"
    print("selfcheck OK "
          f"(un-run r_u at w=5/10/15: {values[0]:.6f}/{values[1]:.6f}/"
          f"{values[2]:.6f}; short-drive F={objective:.6f})")


def aggregate(paths, by):
    """Rates + criteria over verdict JSONL files. ``--by arm`` is the battery
    A/B; ``--by config`` is the calibration grid (grouped by Γ × input
    weight, joint-pinned only, choice rule pre-registered in #560)."""
    records = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    groups = {}
    for record in records:
        if "error" in record and "arm" not in record:
            continue
        if by == "config" and record.get("arm") != "joint-pinned":
            continue  # the calibration grid is the joint-pinned arm only
        key = (record["arm"] if by == "arm"
               else (record["gamma"], record["input_weight"]))
        groups.setdefault(key, []).append(record)

    def rate(rows, predicate):
        hits = [predicate(r) for r in rows]
        known = [h for h in hits if h is not None]
        return (sum(known), len(known))

    summary = {}
    for key, rows in sorted(groups.items(), key=str):
        entry = {
            "n": len(rows),
            "converged": rate(rows, lambda r: r["converged"]),
            "stationary": rate(rows, lambda r: r["stationary"]),
            "persistent": rate(rows, lambda r: r["persistent"]),
            "mean_F": sum(r["F"] for r in rows) / len(rows),
            "mean_holes": sum(r["holes"] for r in rows) / len(rows),
            "max_b3": max(r["max_b3"] for r in rows),
            "mean_singlet": sum(r["singlet"] for r in rows) / len(rows),
        }
        joint = [r for r in rows if r["arm"] == "joint-pinned"]
        if joint:
            block_sums = [r["baryon_block"]["residual"]
                          + r["antibaryon_block"]["residual"]
                          for r in joint
                          if "residual" in r.get("baryon_block", {})
                          and "residual" in r.get("antibaryon_block", {})]
            entry["mean_block_residual_sum"] = (
                sum(block_sums) / len(block_sums) if block_sums else None)
            criteria_rates = {}
            for name in ("holes_clustered_on_baryon", "baryon_singlet_lt_tol",
                         "antibaryon_conjugate_lt_tol", "charge_spread_ge_tol"):
                criteria_rates[name] = rate(
                    joint, lambda r, n=name: evaluate_criteria(r)[n])
            entry["criteria"] = criteria_rates
        two_step = [r for r in rows if r["arm"] == "two-step"]
        if two_step:
            entry["canonical_converged"] = rate(
                two_step, lambda r: r.get("canonical_converged"))
        summary[str(key)] = entry

    if by == "config":
        # Pre-registered choice rule: among configs with >=1 stationary
        # attempt, the lowest mean output-block residual sum; ties -> lower
        # gamma, then lower weight.
        viable = [(key, rows) for key, rows in groups.items()
                  if any(r["stationary"] for r in rows)]
        choice = None
        if viable:
            def score(item):
                key, rows = item
                sums = [r["baryon_block"]["residual"]
                        + r["antibaryon_block"]["residual"]
                        for r in rows if "residual" in r.get("baryon_block", {})
                        and "residual" in r.get("antibaryon_block", {})]
                mean = sum(sums) / len(sums) if sums else float("inf")
                return (mean, key[0], key[1])
            choice = sorted(viable, key=score)[0][0]
        summary["_chosen_config"] = list(choice) if choice else None

    print(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck")

    one = sub.add_parser("attempt")
    one.add_argument("--arm", choices=("joint-pinned", "two-step"),
                     required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--gamma", type=float, default=GAMMA)
    one.add_argument("--input-weight", type=float, default=INPUT_WEIGHT)
    one.add_argument("--init-steps", type=int, default=INIT_STEPS)
    one.add_argument("--evolve-steps", type=int, default=EVOLVE_STEPS)
    one.add_argument("--stage2-iter-cap", type=int, default=STAGE2_ITER_CAP)
    one.add_argument("--persist-passes", type=int, default=PERSIST_PASSES)
    one.add_argument("--out", help="append the verdict line here (else stdout)")
    one.add_argument("--geometry-dir",
                     help="write the schema-1 geometry dump here")

    agg = sub.add_parser("aggregate")
    agg.add_argument("paths", nargs="+")
    agg.add_argument("--by", choices=("arm", "config"), default="arm")

    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        selfcheck()
        return 0
    if args.command == "aggregate":
        aggregate(args.paths, args.by)
        return 0

    budgets = {
        "init_steps": args.init_steps,
        "evolve_steps": args.evolve_steps,
        "stage2_iter_cap": args.stage2_iter_cap,
        "persist_passes": args.persist_passes,
    }
    if args.geometry_dir:
        os.makedirs(args.geometry_dir, exist_ok=True)
    runner = (attempt_joint_pinned if args.arm == "joint-pinned"
              else attempt_two_step)
    try:
        record = runner(args.seed, args.gamma, args.input_weight, budgets,
                        geom_dir=args.geometry_dir)
        record["budgets"] = budgets
        if record["arm"] == "joint-pinned":
            record["criteria"] = evaluate_criteria(record)
    except Exception as error:   # a battery must survive one bad attempt
        record = {"arm": args.arm, "base_seed": args.seed,
                  "gamma": args.gamma, "input_weight": args.input_weight,
                  "error": repr(error)}
    line = json.dumps(record)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
