# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Two spin-3/2 systems under the XY flip-flop as a two-body cobordism (#942).

Levels |k⟩, k = 0..3, count lowering steps from m = 3/2. The dimensionless
ladders act as Σ⁻|k⟩ = λ_k|k+1⟩, Σ⁺|k⟩ = λ_{k−1}|k−1⟩ with
λ = (√3, 2, √3, 0); the vanishing endpoints keep the dynamics in the four
levels. The exchange coupling H_int = ħJ(Σ₁⁺Σ₂⁻ + Σ₁⁻Σ₂⁺) defines the
dimensionless bilinear map B = H_int/ħJ, whose image of a product input is
χ = (Dψ)(Uφ)ᵀ + (Uψ)(Dφ)ᵀ with D the matrix of Σ⁻ and U = Dᵀ. Rows are K,
columns M, and B(ψ, φ) = Σ χ_KM |K⟩⊗|M'⟩. The first-order amplitude to an
orthogonal final state is A_{I→KM}(t) = −iJt χ_KM; branching ratios
|χ_F|²/|χ_F'|² are exact at leading order; the selection rule K + M = k + m
and the corner zeros persist to all orders because [H_int, N] = 0; the exact
amplitudes are the N-block exponentials.

Geometry (epic #938). Each input is a node on a single Δ³ seed grown by
stage 1 and 2 until its whole complex carries the state as a band of its
covariant degree-0 pencil on the seed's four vertices. The interaction node
W is a third Δ³ seed with growth room; the two input fibers are attached to
two vertex-disjoint tetrahedra of W (the attachment order is the attachment
permutation), and W's bulk is synthesized against χ read as the transfer
between the full frames on the two attached tetrahedra: the coupling block of
the whole between the two frames, in the Choi-decomposed (state) or operator
reading. Time is the recursion: one cobordism is the first-order velocity,
which is what this script measures against the algebraic χ, the first-order
amplitudes, the branching ratios, the selection rules, and the exact block
exponentials at Jt.

Swap test. New inputs are grown as new nodes and attached to the SAME W
with every bulk edge pinned (only the two attached tetrahedra's own edges
relax); the re-read transfer is compared with χ of the new inputs.

Run:  python examples/cobordism/two_body_xy_flip_flop.py --output ~/cobordism-runs/two-body/run.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from tessera import cobordism as cob

MC = cob.MultiCobordism
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])
SEED_CELLS = [[0], [1], [2], [3]]


def _unit(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


def _payload(x):
    a = np.asarray(x, dtype=complex)
    if a.ndim == 0:
        return [float(a.real), float(a.imag)]
    if a.ndim == 1:
        return [[float(z.real), float(z.imag)] for z in a]
    return [[[float(z.real), float(z.imag)] for z in row] for row in a]


# ---------------------------------------------------------------- algebra ----

def lowering():
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return D


def flip_flop(psi, phi):
    D = lowering()
    U = D.T
    return np.outer(D @ psi, U @ phi) + np.outer(U @ psi, D @ phi)


def bilinear_generator():
    """B = D⊗U + U⊗D on |K⟩⊗|M'⟩ (index 4K + M)."""
    D = lowering()
    return np.kron(D, D.T) + np.kron(D.T, D)


def exact_amplitudes(psi, phi, Jt):
    """A_{I→KM}(t) = ⟨K,M'|exp(−iJt B)|ψ⊗φ⟩ by the N-block exponentials."""
    B = bilinear_generator()
    state = expm(-1j * Jt * B) @ np.kron(psi, phi)
    return state.reshape(4, 4)


def projective_fit(T, chi):
    """c minimizing ‖cT − χ‖_F and the leak ‖cT − χ‖²/‖χ‖²."""
    tt = np.vdot(T, T).real
    if tt == 0.0:
        return 0j, 1.0
    c = np.vdot(T, chi) / tt
    return c, float(np.linalg.norm(c * T - chi) ** 2 / np.linalg.norm(chi) ** 2)


# --------------------------------------------------------------- geometry ----

def state_fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = SEED_CELLS
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def grow_state(psi, seed, precone, rounds, stage1_steps, candidates, stage2_iters, tolerance):
    """An input node: grow a single Δ³ until its whole complex carries ψ."""
    node = MC(MC.seed_simplex(3), [], [], degrees=[0], seed=seed, precone=precone, einstein_hilbert=False)
    node.set_whole_complex_fiber_target(state_fiber(psi))
    node.use_fiber_residuals(True)
    trace = [node.whole_complex_fiber_residual()]
    t0 = time.time()
    for _ in range(rounds):
        if trace[-1] < tolerance:
            break
        node.run_stage1(max_steps=stage1_steps, n_candidate_moves=candidates)
        node.run_stage2(beta=1.0, max_iters=stage2_iters, tolerance=1e-15)
        trace.append(node.whole_complex_fiber_residual())
    fiber = node.read_whole_complex_fiber()
    images = np.asarray(fiber.images)
    # the carried state: the band column closest to ψ (a rank-one band has one)
    overlaps = [abs(np.vdot(_unit(images[:, j]), _unit(psi))) for j in range(images.shape[1])]
    best = int(np.argmax(overlaps))
    carried = cob.BoundaryFiber()
    carried.degree = 0
    carried.cells = SEED_CELLS
    carried.images = images[:, [best]]
    carried.dualImages = np.asarray(fiber.dualImages)[:, [best]] if np.asarray(fiber.dualImages).size else images[:, [best]]
    record = {"residual_trace": [float(r) for r in trace], "converged": bool(trace[-1] < tolerance),
              "vertices": int(node.spacetime().getVertexList().size()), "seconds": round(time.time() - t0, 1),
              "band_rank": int(images.shape[1]), "carried_overlap": float(overlaps[best]),
              "carried_state": _payload(_unit(images[:, best]))}
    return node, carried, record


def disjoint_tetrahedra(st):
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(3)]
    for a, b in itertools.combinations(tets, 2):
        if not set(a) & set(b):
            return a, b
    return None


def interaction_node(fiber_a, fiber_b, chi, choi, seed, precone):
    """W: a Δ³ seed grown by gated cone-ins until two vertex-disjoint tetrahedra
    exist, the two fibers attached to them, χ as the two-body target."""
    for grown in range(precone, precone + 17, 2):
        node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
                  seed=seed, precone=grown, einstein_hilbert=False)
        pair = disjoint_tetrahedra(node.spacetime())
        if pair is not None:
            break
    else:
        raise RuntimeError("no two vertex-disjoint tetrahedra after growth")
    a, b = pair
    node.seed_inputs([0, 1])
    node.attach_input_fiber(0, fiber_a, [[v] for v in a])
    node.attach_input_fiber(1, fiber_b, [[v] for v in b])
    node.set_two_body_target(chi, choi)
    node.use_fiber_residuals(True)
    return node, a, b, grown


def two_body_record(node, psi, phi, Jt):
    read = node.read_two_body()
    T = np.asarray(read.transfer)
    chi = flip_flop(psi, phi)
    c, leak = projective_fit(T, chi)
    geometric = c * T
    populated = [(K, M) for K in range(4) for M in range(4) if abs(chi[K, M]) > 1e-12]
    absent = [(K, M) for K in range(4) for M in range(4) if abs(chi[K, M]) <= 1e-12]
    ratios_alg, ratios_geo = [], []
    for (K, M), (K2, M2) in itertools.combinations(populated, 2):
        ratios_alg.append(abs(chi[K, M]) ** 2 / abs(chi[K2, M2]) ** 2)
        ratios_geo.append(abs(geometric[K, M]) ** 2 / max(abs(geometric[K2, M2]) ** 2, 1e-300))
    exact = exact_amplitudes(psi, phi, Jt)
    first_order = -1j * Jt * chi
    # amplitudes to final states orthogonal to the input: F ≠ I components
    return {
        "residual": float(read.residual),
        "leak_of_chi_in_transfer": leak,
        "scale": _payload(c),
        "schmidt_spectrum": [float(s) for s in read.singular_values],
        "schmidt_rank": int(read.schmidt_rank),
        "reversal_residual": float(read.reversal_residual),
        "input_fiber_residuals": [float(r) for r in read.input_fiber_residuals],
        "transfer": _payload(T),
        "chi_geometric": _payload(geometric),
        "chi_algebraic": _payload(chi),
        "entry_error_max": float(np.abs(geometric - chi).max()),
        "first_order_amplitudes_algebraic": _payload(first_order),
        "first_order_amplitudes_geometric": _payload(-1j * Jt * geometric),
        "exact_amplitudes_at_Jt": _payload(exact),
        "first_order_vs_exact_orthogonal_max": float(max(
            abs(first_order[K, M] - exact[K, M]) for K in range(4) for M in range(4)
            if abs(np.kron(psi, phi).reshape(4, 4)[K, M]) < 1e-12) if any(
            abs(np.kron(psi, phi).reshape(4, 4)[K, M]) < 1e-12 for K in range(4) for M in range(4)) else 0.0),
        "branching_ratio_error_max": float(max((abs(g - a) / a for g, a in zip(ratios_geo, ratios_alg)), default=0.0)),
        "selection_rule_absent_entries": [[K, M] for K, M in absent],
        "selection_rule_leak_max": float(max((abs(geometric[K, M]) for K, M in absent), default=0.0)),
        "cells_a": [list(map(int, c)) for c in read.cells_a],
        "cells_b": [list(map(int, c)) for c in read.cells_b],
    }


def drive(node, rounds, stage1_steps, candidates, stage2_iters, tolerance, stage1=True):
    trace = [node.two_body_residual()]
    t0 = time.time()
    for _ in range(rounds):
        if trace[-1] < tolerance:
            break
        if stage1:
            node.run_stage1(max_steps=stage1_steps, n_candidate_moves=candidates)
        node.run_stage2(beta=1.0, max_iters=stage2_iters, tolerance=1e-15)
        trace.append(node.two_body_residual())
    return {"residual_trace": [float(r) for r in trace], "seconds": round(time.time() - t0, 1),
            "vertices": int(node.spacetime().getVertexList().size())}


def freeze_bulk(node):
    """Pin every edge not inside one of the two input block regions (the
    attached tetrahedra and the cells the blocks grew to), so only the blocks'
    own edges relax when new inputs are attached."""
    regions = [set(int(v) for v in block.vertices) for block in node.inputs]
    count = 0
    for e in node.spacetime().getEdgeList().toVector():
        u, v = int(e.getSource().getId()), int(e.getTarget().getId())
        if any(u in r and v in r for r in regions):
            continue
        node.declare_pinned_region(f"bulk_edge_{u}_{v}", {u, v})
        count += 1
    return count


def run(args):
    rng = np.random.default_rng(args.seed)
    psi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    budget = dict(precone=args.precone, rounds=args.state_rounds, stage1_steps=args.stage1_steps,
                  candidates=args.candidates, stage2_iters=args.stage2_iters, tolerance=args.state_tolerance)
    record = {"method": {
        "inputs": "two Δ³ seeds grown by stage 1/2 until the whole carries the state as a degree-0 band on the seed's vertices",
        "interaction": "a Δ³ seed with gated cone-ins until two vertex-disjoint tetrahedra exist; fibers attached there",
        "target": "chi read as the frame transfer T_AB (coupling block of the whole between the two frames)",
        "reading": "choi_decomposed" if args.choi else "operator",
        "Jt": args.Jt, "seed": args.seed, **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in budget.items()},
    }, "psi": _payload(psi), "phi": _payload(phi)}
    node_a, fiber_a, rec_a = grow_state(psi, args.seed + 1, **budget)
    node_b, fiber_b, rec_b = grow_state(phi, args.seed + 2, **budget)
    record["input_psi"], record["input_phi"] = rec_a, rec_b
    chi = flip_flop(psi, phi)
    W, a, b, grown = interaction_node(fiber_a, fiber_b, chi, args.choi, args.seed + 3, args.precone)
    record["interaction"] = {"cone_ins": int(grown), "attached_a": list(a), "attached_b": list(b),
                             "block_regions": [sorted(int(v) for v in blk.vertices) for blk in W.inputs],
                             "before": two_body_record(W, psi, phi, args.Jt)}
    record["interaction"]["drive"] = drive(W, args.rounds, args.stage1_steps, args.candidates, args.stage2_iters,
                                           args.tolerance)
    record["interaction"]["after"] = two_body_record(W, psi, phi, args.Jt)

    # Swap test on the frozen bulk.
    psi2 = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi2 = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    _, fiber_a2, rec_a2 = grow_state(psi2, args.seed + 4, **budget)
    _, fiber_b2, rec_b2 = grow_state(phi2, args.seed + 5, **budget)
    pinned = freeze_bulk(W)
    W.attach_input_fiber(0, fiber_a2, [[v] for v in a])
    W.attach_input_fiber(1, fiber_b2, [[v] for v in b])
    W.set_two_body_target(flip_flop(psi2, phi2), args.choi)
    swap = {"psi": _payload(psi2), "phi": _payload(phi2), "input_psi": rec_a2, "input_phi": rec_b2,
            "pinned_bulk_edges": int(pinned), "before_relax": two_body_record(W, psi2, phi2, args.Jt)}
    swap["relax"] = drive(W, args.rounds, args.stage1_steps, args.candidates, args.stage2_iters, args.tolerance,
                          stage1=False)
    swap["after_relax"] = two_body_record(W, psi2, phi2, args.Jt)
    # the frozen bulk's own reading of the new inputs' chi WITHOUT relaxing the blocks
    record["swap"] = swap
    after = record["interaction"]["after"]
    record["checks"] = {k: bool(v) for k, v in {
        "inputs_carried": rec_a["converged"] and rec_b["converged"],
        "two_body_fit_converged": after["residual"] < args.tolerance,
        "schmidt_rank_two": after["schmidt_rank"] == 2,
        "reversal_identity": after["reversal_residual"] < 1e-8,
        "selection_rules_respected": after["selection_rule_leak_max"] < 1e-6 * float(np.abs(chi).max()),
        "branching_ratios_match": after["branching_ratio_error_max"] < 1e-4,
        "swap_reproduces_new_chi": swap["after_relax"]["leak_of_chi_in_transfer"] < args.tolerance,
    }.items()}
    return record


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precone", type=int, default=8)
    p.add_argument("--state-rounds", type=int, default=8)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--stage1-steps", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--stage2-iters", type=int, default=150)
    p.add_argument("--state-tolerance", type=float, default=1e-10)
    p.add_argument("--tolerance", type=float, default=1e-8)
    p.add_argument("--Jt", type=float, default=0.05)
    p.add_argument("--operator", dest="choi", action="store_false", help="report the operator reading")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Every read here is on the chain-level Whitney pencil (epic #905's one knob).
    previous = cob.HodgeLaplacian.defaultMetricSource()
    cob.HodgeLaplacian.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        record = run(args)
    finally:
        cob.HodgeLaplacian.setDefaultMetricSource(previous)
    text = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps(record["checks"], indent=2))
    for key in ("input_psi", "input_phi"):
        print(key, "residual trace", ["%.2e" % r for r in record[key]["residual_trace"]], record[key]["seconds"], "s")
    it = record["interaction"]
    print("interaction: cone-ins", it["cone_ins"], "trace", ["%.2e" % r for r in it["drive"]["residual_trace"]],
          "schmidt", ["%.3f" % s for s in it["after"]["schmidt_spectrum"]])
    print("swap: trace", ["%.2e" % r for r in record["swap"]["relax"]["residual_trace"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
