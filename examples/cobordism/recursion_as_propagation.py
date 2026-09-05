# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Recursive-fiber layers as time propagation of the two-body map (#943).

One interaction cobordism represents the first-order velocity χ = B Ψ of the
pair state (#941, #942). This script develops the identification of
successive layers with time evolution experimentally, in two chains selected
by ``--chain``:

``velocity`` (the owner's stated picture, a DAG of nodes): layer n is a fresh
interaction node W_n on its own Δ³ seed; its two input fibers are the Schmidt
factors of the pair state Ψ_{n−1} (the one-particle content each frame must
carry: images u_k on A, v_k on B, weighted by √σ_k), its target is
χ_n = B Ψ_{n−1}, and the pair state advances by the Euler step
Ψ_n = Ψ_{n−1} − iJΔt χ_n^geo with the GEOMETRIC read χ_n^geo = c·T_AB(W_n)
(the algebraic χ_n is carried beside it as the reference chain).

``extend`` (the recursion proper): the SAME cobordism is continued — its
attached fibers are replaced by the Schmidt factors of Ψ_{n−1}, its target by
χ_n, and it grows and relaxes again — so the geometry composes across layers
instead of being rebuilt.

Measurements per layer: the leak of χ_n in T_AB(W_n), the Schmidt spectrum of
the read, and the pair state's distance from the exact evolution
e^{−iJnΔt B}Ψ₀ (the N-block exponentials) in three bookkeepings: Euler with
geometric reads, Euler with algebraic χ (the discretization error alone), and
exact. The identification is supported to the extent the geometric chain
tracks the exact one beyond what the algebraic Euler chain does on its own.
Nothing here is stated as established: the outcome is the measurement.

Run:  python examples/cobordism/recursion_as_propagation.py --chain velocity --layers 4
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


def lowering():
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return D


def apply_b(pair):
    """χ = B Ψ on the pair frame: D Ψ Uᵀ + U Ψ Dᵀ with U = Dᵀ."""
    D = lowering()
    return D @ pair @ D + D.T @ pair @ D.T


def bilinear_generator():
    D = lowering()
    return np.kron(D, D.T) + np.kron(D.T, D)


def exact_state(pair0, Jt):
    return (expm(-1j * Jt * bilinear_generator()) @ pair0.reshape(-1)).reshape(4, 4)


def projective_fit(T, chi):
    tt = np.vdot(T, T).real
    if tt == 0.0:
        return 0j, 1.0
    c = np.vdot(T, chi) / tt
    return c, float(np.linalg.norm(c * T - chi) ** 2 / np.linalg.norm(chi) ** 2)


def schmidt_fibers(pair, rank=2):
    """The pair state's Schmidt factors as the two frames' fibers."""
    u, s, vh = np.linalg.svd(pair)
    r = int(min(rank, np.sum(s > 1e-12 * s[0])))
    fa, fb = cob.BoundaryFiber(), cob.BoundaryFiber()
    for f, images in ((fa, u[:, :r] * np.sqrt(s[:r])), (fb, vh[:r, :].T * np.sqrt(s[:r]))):
        f.degree = 0
        f.cells = SEED_CELLS
        f.images = np.asarray(images, dtype=complex)
    return fa, fb, s


def disjoint_tetrahedra(st):
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(3)]
    for a, b in itertools.combinations(tets, 2):
        if not set(a) & set(b):
            return a, b
    return None


def fresh_interaction_node(seed, precone):
    for grown in range(precone, precone + 17, 2):
        node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
                  seed=seed, precone=grown, einstein_hilbert=False)
        pair = disjoint_tetrahedra(node.spacetime())
        if pair is not None:
            node.seed_inputs([0, 1])
            node.use_fiber_residuals(True)
            return node, pair, grown
    raise RuntimeError("no two vertex-disjoint tetrahedra after growth")


def attach_and_target(node, a, b, fa, fb, chi, choi):
    node.attach_input_fiber(0, fa, [[v] for v in a])
    node.attach_input_fiber(1, fb, [[v] for v in b])
    node.set_two_body_target(chi, choi)


def drive(node, rounds, stage1_steps, candidates, stage2_iters, tolerance):
    trace = [node.two_body_residual()]
    t0 = time.time()
    for _ in range(rounds):
        if trace[-1] < tolerance:
            break
        node.run_stage1(max_steps=stage1_steps, n_candidate_moves=candidates)
        node.run_stage2(beta=1.0, max_iters=stage2_iters, tolerance=1e-15)
        trace.append(node.two_body_residual())
    return {"residual_trace": [float(r) for r in trace], "seconds": round(time.time() - t0, 1),
            "vertices": int(node.spacetime().getVertexList().size())}


def run(args):
    rng = np.random.default_rng(args.seed)
    psi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    pair0 = np.outer(psi, phi)
    dt = args.Jt / args.layers
    pair_geo, pair_alg = pair0.copy(), pair0.copy()
    record = {"method": {"chain": args.chain, "layers": args.layers, "Jt": args.Jt, "seed": args.seed,
                         "precone": args.precone, "rounds": args.rounds, "stage2_iters": args.stage2_iters,
                         "reading": "choi_decomposed" if args.choi else "operator"},
              "psi": _payload(psi), "phi": _payload(phi), "layers_record": []}
    node = a = b = None
    for n in range(args.layers):
        chi_geo_target = apply_b(pair_geo)   # the velocity of the geometric chain's state
        chi_alg = apply_b(pair_alg)
        fa, fb, sigma = schmidt_fibers(pair_geo)
        if args.chain == "velocity" or node is None:
            node, (a, b), grown = fresh_interaction_node(args.seed + 10 * n, args.precone)
        else:
            grown = None
        attach_and_target(node, a, b, fa, fb, chi_geo_target, args.choi)
        layer = {"layer": n, "cone_ins": grown, "input_schmidt": [float(s) for s in sigma],
                 "drive": drive(node, args.rounds, args.stage1_steps, args.candidates, args.stage2_iters, args.tolerance)}
        read = node.read_two_body()
        T = np.asarray(read.transfer)
        c, leak = projective_fit(T, chi_geo_target)
        chi_geo = c * T
        layer.update({"leak": leak, "read_schmidt": [float(s) for s in read.singular_values],
                      "block_residuals": [float(r) for r in read.input_fiber_residuals],
                      "reversal_residual": float(read.reversal_residual)})
        # advance the three bookkeepings
        pair_geo = pair_geo - 1j * dt * chi_geo
        pair_alg = pair_alg - 1j * dt * chi_alg
        exact = exact_state(pair0, dt * (n + 1))
        def dist(x, y):
            return float(np.linalg.norm(x - y) / np.linalg.norm(y))
        layer.update({"distance_geometric_vs_exact": dist(pair_geo, exact),
                      "distance_algebraic_euler_vs_exact": dist(pair_alg, exact),
                      "distance_geometric_vs_algebraic_euler": dist(pair_geo, pair_alg),
                      "pair_geometric": _payload(pair_geo), "pair_exact": _payload(exact)})
        record["layers_record"].append(layer)
    last = record["layers_record"][-1]
    record["checks"] = {k: bool(v) for k, v in {
        "every_layer_carried_its_velocity": all(l["leak"] < args.tolerance for l in record["layers_record"]),
        "geometric_chain_tracks_exact_as_well_as_euler": last["distance_geometric_vs_exact"] <= 1.5 * last["distance_algebraic_euler_vs_exact"] + 1e-12,
        "reversal_identity": all(l["reversal_residual"] < 1e-8 for l in record["layers_record"]),
    }.items()}
    return record


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--chain", choices=("velocity", "extend"), default="velocity")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--Jt", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precone", type=int, default=14)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--stage1-steps", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--stage2-iters", type=int, default=200)
    p.add_argument("--tolerance", type=float, default=1e-8)
    p.add_argument("--operator", dest="choi", action="store_false")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
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
    for l in record["layers_record"]:
        print(f"layer {l['layer']}: leak {l['leak']:.3e} read schmidt {['%.3f' % s for s in l['read_schmidt']]} "
              f"geo-vs-exact {l['distance_geometric_vs_exact']:.3e} euler-vs-exact {l['distance_algebraic_euler_vs_exact']:.3e} "
              f"{l['drive']['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
