# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The load-bearing test of the annihilation identification (#660): does state
CONJUGATION at the inputs — the antiparticle convention (Proton.cpp: "the
antidiquark {1, ω²} is exactly the conjugate of the diquark {1, ω}") — map to
OPPOSITE Krein signature (the sign of vᵀWv) in the carried spectrum?

The setup is one neutral q-q̄ pair, the configuration pair creation already
carries: a `MultiCobordism` with input targets `[[ω], [ω̄]]` and an EMPTY
output list (#555 — nothing pinned downstream, the whole's state emerges),
driven with the combined `run` (an init pass growing the input regions, then
an evolution pass with the boundary frozen), restarted a few times because the
engine is not process-deterministic.

Per attempt the script records the LOCUS TRAJECTORY — `r_U`, the interval leak
`max|Im ℓ²|`, and the broken-pair count where defined — because the Krein
classification only exists on the real-ℓ² locus and live stage-2 exploration
is free to leave it (measured elsewhere: leaks ~ 0.3 on a spacelike-preconed
host, exactly 0 through 12 frames on a timelike-preconed one).

The verdict, when the best attempt ends ON the locus: every real mode gets its
eigenvalue, W-norm, and its AFFINITY to each input block — the |ψ|²-weighted
overlap with the block's vertex set (`affinity = Σ_c w_c·|c ∩ block|/(k+1)`) —
and the q-affine vs q̄-affine near-kernel modes are compared: the
identification predicts opposite signature signs. Off the locus the honest
verdict is that the test needs the quantised/real-ℓ² regime, with the measured
leak printed.

An exploratory example, deliberately not a CI test: budgets are small, the
engine is stochastic, and a non-carrying attempt is reported, not hidden.

    python qqbar_krein_signature.py
    python qqbar_krein_signature.py --restarts 4 --init 60 --evolve 30
    python qqbar_krein_signature.py --precone-timelike   # start on timelike material
"""
import argparse
import cmath
import os
import sys

if "OMP_NUM_THREADS" not in os.environ or "--threads" in sys.argv:
    _n = "16"
    if "--threads" in sys.argv:
        try:
            _n = sys.argv[sys.argv.index("--threads") + 1]
        except IndexError:
            pass
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _n

import numpy as np

import tessera

cob = tessera.cobordism

from krein_modes import KreinModes

_OMEGA = complex(cmath.exp(2j * cmath.pi / 3))
_DEGREE = 3


def build_pair_node(seed, gamma, precone, precone_timelike):
    """One neutral q-q̄ pair on a single-Δ⁴ seed: inputs {ω} and its conjugate
    {ω̄}, no pinned output. Anchors on two seed vertices so the two blocks
    differentiate as the complex grows."""
    host = tessera.Spacetime.fromCells(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)
    node = cob.MultiCobordism(host, [[_OMEGA], [_OMEGA.conjugate()]], [],
                              [_DEGREE], gamma, seed, precone, True,
                              precone_timelike)
    node.seed_inputs([0, 2])
    return node


def drive(node, init_steps, evolve_steps, candidates, beta):
    """The combined drive, one iteration at a time, recording the locus
    trajectory: (phase, r_U, max|Im ℓ²|, pair count or None)."""
    trajectory = []
    for phase, steps in (("init", init_steps), ("evolve", evolve_steps)):
        for _ in range(steps):
            node.run(max_iters=1, n_candidate_moves=candidates,
                     grow_boundaries=(phase == "init"), beta=beta)
            krein = KreinModes(node.st, _DEGREE)
            trajectory.append((phase, float(node.r_u(node.st)),
                               float(krein.imag_interval_leak),
                               krein.pair_count))
    return trajectory


def block_affinity(krein, mode_index, block_vertices):
    """|ψ|²-weighted overlap of one mode with a block's vertex set: 1 = the
    mode lives entirely on cells inside the block, 0 = entirely outside."""
    weight = krein.cell_weight([mode_index])
    block = set(block_vertices)
    return float(sum(
        w * len(block.intersection(cell)) / len(cell)
        for w, cell in zip(weight, krein.cells)))


def analyze(node):
    """The signature table and the verdict, honest about the locus."""
    krein = KreinModes(node.st, _DEGREE)
    print(f"\nfinal state: r_U = {float(node.r_u(node.st)):.3e}   "
          f"max|Im l2| = {krein.imag_interval_leak:.3e}   "
          f"on locus: {krein.on_locus}")
    if not krein.on_locus:
        print(f"VERDICT: undefined off the real-l2 locus ({krein.reason}) — "
              "the signature test needs the quantised/real regime; rerun or "
              "tune toward it")
        return
    blocks = list(node.inputs)
    labels = ["q  (omega)", "qbar (conj)"]
    print(f"broken pairs: {krein.pair_count}   real modes: "
          f"{len(krein.real_indices)}")
    print(f"{'lambda':>12} {'W-norm':>12} {'aff(q)':>8} {'aff(qbar)':>9}  side")
    rows = []
    order = sorted(krein.real_indices,
                   key=lambda i: abs(krein.eigenvalues[i].real))
    for i in order:
        affinities = [block_affinity(krein, i, b.vertices) for b in blocks]
        side = 0 if affinities[0] >= affinities[1] else 1
        rows.append((float(krein.eigenvalues[i].real),
                     float(krein.w_norms[i]), affinities, side))
    for value, norm, affinities, side in rows[:12]:
        print(f"{value:>12.4f} {norm:>+12.3e} {affinities[0]:>8.3f} "
              f"{affinities[1]:>9.3f}  {labels[side]}")
    # The verdict reads the NEAR-KERNEL end (smallest |lambda|), where the
    # carried register content lives: the affinity-weighted mean signature of
    # each side — the identification predicts opposite signs.
    near = rows[:max(4, 2)]
    mean_signature = [0.0, 0.0]
    weight_total = [0.0, 0.0]
    for value, norm, affinities, side in near:
        pull = affinities[0] - affinities[1]
        side_index = 0 if pull >= 0 else 1
        mean_signature[side_index] += abs(pull) * np.sign(norm)
        weight_total[side_index] += abs(pull)
    for side_index in (0, 1):
        if weight_total[side_index] > 0:
            mean_signature[side_index] /= weight_total[side_index]
    print(f"\naffinity-weighted mean signature over the {len(near)} "
          f"nearest-kernel real modes:")
    print(f"  {labels[0]}: {mean_signature[0]:+.3f}    "
          f"{labels[1]}: {mean_signature[1]:+.3f}")
    if weight_total[0] > 0 and weight_total[1] > 0:
        opposite = mean_signature[0] * mean_signature[1] < 0
        print(f"VERDICT: q vs qbar signatures "
              f"{'OPPOSITE — supports' if opposite else 'NOT opposite — challenges'} "
              "the conjugation = signature-flip identification (single "
              "stochastic attempt; fiddle before believing)")
    else:
        print("VERDICT: inconclusive — the near-kernel modes did not "
              "differentiate between the blocks (affinities tied)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--init", type=int, default=40)
    ap.add_argument("--evolve", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=50.0)
    ap.add_argument("--precone", type=int, default=6)
    ap.add_argument("--precone-timelike", action="store_true",
                    dest="precone_timelike",
                    help="start on timelike precone material (measured to hold "
                         "the real-l2 locus longer than a spacelike start)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    best = None
    for attempt in range(args.restarts):
        seed = args.seed + attempt
        node = build_pair_node(seed, args.gamma, args.precone,
                               args.precone_timelike)
        trajectory = drive(node, args.init, args.evolve, args.candidates,
                           args.beta)
        residual = trajectory[-1][1]
        on_locus_frames = sum(1 for _p, _r, leak, _n in trajectory
                              if leak <= 1e-9)
        print(f"attempt {attempt} (seed {seed}): final r_U = {residual:.3e}   "
              f"on-locus {on_locus_frames}/{len(trajectory)} iterations   "
              f"final leak = {trajectory[-1][2]:.3e}")
        for i, (phase, r_u, leak, pairs) in enumerate(trajectory):
            if i % 10 == 0 or i == len(trajectory) - 1:
                print(f"    [{i:>3} {phase:>6}] r_U={r_u:.3e}  "
                      f"leak={leak:.3e}  pairs={pairs}")
        if best is None or residual < best[0]:
            best = (residual, node)
    analyze(best[1])


if __name__ == "__main__":
    main()
