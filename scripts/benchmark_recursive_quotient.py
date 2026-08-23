#!/usr/bin/env python3
"""Benchmark the recursive response reduction (#768).

Examples:
    OMP_NUM_THREADS=8 python scripts/benchmark_recursive_quotient.py
    OMP_NUM_THREADS=8 python scripts/benchmark_recursive_quotient.py \
        --components 24 --component-size 64

Four scenarios, each comparing the structured path against the dense/cold
reference ON THE SAME inputs and reporting the median timings together with
the measured agreement, so the numbers are before/after AND correctness in
one run (the benchmark_analytic_kernel.py convention):

  static_reduction — the exact supported static response over C chained
                     components (per-component sparse factor solves +
                     component-local contributions) vs the one-shot dense
                     NumPy Schur complement of the whole interior block.
  feshbach_window  — F_B(lambda) across a frequency window by memoized
                     per-component shifted solves vs the dense NumPy solve
                     of the whole shifted interior per lambda.
  craig_bampton    — reduced-pencil window eigenvalues from the retained-
                     mode surrogate vs the dense eigensolve of the full
                     operator, with the residual/gap certificates.
  cache            — accepted local metric moves on a chain-of-components
                     spacetime: touched-star invalidation (recompute ONE
                     component through the #764 AnalyticCache) vs cold
                     recomputation of every component per move.
"""

import argparse
import cmath
import json
import statistics
import sys
import time

import numpy as np

import tessera

cob = tessera.cobordism


def median_timing(function, repeats):
    durations = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), value


def chain_operator(components, size, rng):
    """A PSD block-chain: `components` blocks of `size` interior cells, each
    hung between two shared junction cells — the junctions are the kept
    interface, the block bulks the per-component interiors."""
    n = components * (size + 1) + 1
    L = np.zeros((n, n))

    def couple(i, j, w):
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w

    member_sets = []
    for c in range(components):
        left = c * (size + 1)
        right = (c + 1) * (size + 1)
        bulk = [left + 1 + k for k in range(size)]
        chain = [left] + bulk + [right]
        for a, b in zip(chain[:-1], chain[1:]):
            couple(a, b, 0.5 + rng.random())
        member_sets.append(chain)
    junctions = [c * (size + 1) for c in range(components + 1)]
    return L, member_sets, junctions


def dense_schur(L, kept, interior):
    L_BB = L[np.ix_(kept, kept)]
    L_BI = L[np.ix_(kept, interior)]
    L_IB = L[np.ix_(interior, kept)]
    L_II = L[np.ix_(interior, interior)]
    return L_BB - L_BI @ np.linalg.solve(L_II, L_IB)


def flat(matrix):
    return [complex(z) for z in np.asarray(matrix, dtype=complex).reshape(-1)]


def benchmark_static(components, size, repeats, rng):
    L, member_sets, junctions = chain_operator(components, size, rng)
    n = L.shape[0]
    # The end junctions belong to one component only and are eliminated with
    # its bulk; the kept interface is the shared interior junctions.
    kept = junctions[1:-1]
    interior = sorted(set(range(n)) - set(kept))

    fl = flat(L)  # marshalling outside the timings (both paths get their
    #               native input form; the kernels are what is compared)

    def cold():
        quotient = cob.RecursiveQuotient.overMatrix(fl, n, [], member_sets)
        read = quotient.staticReduction()
        return np.asarray(read.effectiveOperator, dtype=complex).reshape(
            len(kept), len(kept))

    warm_quotients = [cob.RecursiveQuotient.overMatrix(fl, n, [], member_sets)
                      for _ in range(repeats)]

    def warm():
        # Construction (classification + regime) amortized: the repeated-
        # reduction pattern of spec section 18 ("affected component/window
        # factorization").
        read = warm_quotients.pop().staticReduction()
        return np.asarray(read.effectiveOperator, dtype=complex).reshape(
            len(kept), len(kept))

    def dense():
        return dense_schur(L, kept, interior)

    cold_s, structured_value = median_timing(cold, repeats)
    warm_s, warm_value = median_timing(warm, repeats)
    dense_s, dense_value = median_timing(dense, repeats)
    return {
        "dim": n,
        "kept": len(kept),
        "cold_median_s": cold_s,
        "warm_reduce_median_s": warm_s,
        "dense_median_s": dense_s,
        "speedup_cold": dense_s / cold_s if cold_s > 0 else float("inf"),
        "speedup_warm": dense_s / warm_s if warm_s > 0 else float("inf"),
        "max_deviation": float(
            np.max(np.abs(structured_value - dense_value))),
        "max_deviation_warm": float(
            np.max(np.abs(warm_value - dense_value))),
    }


def benchmark_feshbach(components, size, repeats, nodes, rng):
    L, member_sets, junctions = chain_operator(components, size, rng)
    n = L.shape[0]
    kept = junctions[1:-1]
    interior = sorted(set(range(n)) - set(kept))
    lambdas = np.linspace(0.05, 0.45, nodes)  # below the bulk spectrum

    quotient = cob.RecursiveQuotient.overMatrix(flat(L), n, [], member_sets)

    def structured():
        out = []
        for lam in lambdas:
            read = quotient.feshbach(complex(lam), 0.05, 0.45)
            out.append(np.asarray(read.response, dtype=complex))
        return out

    def dense():
        out = []
        L_II = L[np.ix_(interior, interior)]
        L_BI = L[np.ix_(kept, interior)]
        L_IB = L[np.ix_(interior, kept)]
        L_BB = L[np.ix_(kept, kept)]
        for lam in lambdas:
            F = (L_BB - lam * np.eye(len(kept)) -
                 L_BI @ np.linalg.solve(L_II - lam * np.eye(len(interior)),
                                        L_IB))
            out.append(F.reshape(-1))
        return out

    structured_s, structured_value = median_timing(structured, repeats)
    dense_s, dense_value = median_timing(dense, repeats)
    deviation = max(float(np.max(np.abs(a - b)))
                    for a, b in zip(structured_value, dense_value))
    return {
        "dim": n,
        "window_nodes": nodes,
        "structured_median_s": structured_s,
        "dense_median_s": dense_s,
        "speedup": dense_s / structured_s if structured_s > 0 else float("inf"),
        "max_deviation": deviation,
    }


def benchmark_craig_bampton(components, size, repeats, rng):
    L, member_sets, junctions = chain_operator(components, size, rng)
    n = L.shape[0]
    window = (-1e-9, 0.3)
    cutoff = 0.5

    quotient = cob.RecursiveQuotient.overMatrix(flat(L), n, [], member_sets)

    def surrogate():
        read = quotient.craigBampton(window[0], window[1], cutoff,
                                     residual_tolerance=1e-1)
        return read

    def dense():
        values = np.linalg.eigvalsh(L)
        return values[(values >= window[0]) & (values <= window[1])]

    surrogate_s, read = median_timing(surrogate, repeats)
    dense_s, fine = median_timing(dense, repeats)
    deviation = 0.0
    for value in read.windowEigenvalues:
        deviation = max(deviation, float(np.min(np.abs(fine - value))))
    return {
        "dim": n,
        "window": list(window),
        "retained_modes": int(np.sum(read.retainedModes)),
        "discarded_mode_gap": read.discardedModeGap,
        "surrogate_median_s": surrogate_s,
        "dense_median_s": dense_s,
        "speedup": dense_s / surrogate_s if surrogate_s > 0 else float("inf"),
        "window_eigenvalues": len(read.windowEigenvalues),
        "max_window_deviation": deviation,
        "max_eigen_residual": max(read.eigenResiduals, default=0.0),
    }


def build_component_chain_spacetime(components, size, rng):
    """The chain-of-components fixture as a Spacetime (k = 0), so accepted
    moves publish TouchedStars through the real #764 cache."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    n = components * (size + 1) + 1
    verts = [st.createVertex(i) for i in range(n)]
    weights = {}
    for c in range(components):
        left = c * (size + 1)
        chain = [left] + [left + 1 + k for k in range(size)] + \
                [(c + 1) * (size + 1)]
        for a, b in zip(chain[:-1], chain[1:]):
            st.createSimplex([verts[a], verts[b]])
            weights[(a, b)] = 0.5 + rng.random()
    for e in st.getEdgeList().toVector():
        key = (e.getSource().getId(), e.getTarget().getId())
        w = weights.get(key) or weights.get((key[1], key[0]))
        e.setLength(cmath.sqrt(complex(w)))
        e.setPhase(0.0)
    member_cells = []
    for c in range(components):
        left = c * (size + 1)
        chain = [left] + [left + 1 + k for k in range(size)] + \
                [(c + 1) * (size + 1)]
        member_cells.append([[v] for v in chain])
    return st, member_cells


def benchmark_cache(components, size, moves, repeats, rng):
    st, member_cells = build_component_chain_spacetime(components, size, rng)
    cache = cob.AnalyticCache(st)
    quotient = cob.RecursiveQuotient.overCells(
        st, 0, member_cells, cob.RecursiveQuotient.Options(), cache)
    quotient.staticReduction()  # warm every per-component entry

    edges = st.getEdgeList().toVector()
    move_plan = [(edges[rng.integers(len(edges))], 0.5 + rng.random())
                 for _ in range(moves)]

    def one_pass(cached):
        results = []
        for edge, squared_length in move_plan:
            edge.setLength(cmath.sqrt(complex(squared_length)))
            star = cob.TouchedStar()
            star.addChangedEdge(edge.getSource().getId(),
                                edge.getTarget().getId())
            cache.publish(star)
            if cached:
                quotient.invalidate()
                read = quotient.staticReduction()
            else:
                cold = cob.RecursiveQuotient.overCells(st, 0, member_cells)
                read = cold.staticReduction()
            results.append(np.asarray(read.effectiveOperator, dtype=complex))
        return results

    cached_s, cached_values = median_timing(lambda: one_pass(True), repeats)
    cold_s, cold_values = median_timing(lambda: one_pass(False), repeats)
    deviation = max(float(np.max(np.abs(a - b)))
                    for a, b in zip(cached_values, cold_values))
    return {
        "components": components,
        "component_size": size,
        "moves": moves,
        "cached_median_s": cached_s,
        "cold_median_s": cold_s,
        "speedup": cold_s / cached_s if cached_s > 0 else float("inf"),
        "hits": cache.hits,
        "misses": cache.misses,
        "invalidations": cache.invalidations,
        "max_deviation": deviation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--component-size", type=int, default=48)
    parser.add_argument("--window-nodes", type=int, default=8)
    parser.add_argument("--moves", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=768)
    args = parser.parse_args()
    if min(args.components, args.component_size, args.window_nodes,
           args.moves, args.repeats) < 1 or args.seed < 0:
        parser.error("sizes and repeats must be positive; seed non-negative")

    rng = np.random.default_rng(args.seed)
    result = {
        "components": args.components,
        "component_size": args.component_size,
        "repeats": args.repeats,
        "seed": args.seed,
        "static_reduction": benchmark_static(args.components,
                                             args.component_size,
                                             args.repeats, rng),
        "feshbach_window": benchmark_feshbach(args.components,
                                              args.component_size,
                                              args.repeats,
                                              args.window_nodes, rng),
        "craig_bampton": benchmark_craig_bampton(args.components,
                                                 args.component_size,
                                                 args.repeats, rng),
        "cache": benchmark_cache(args.components, args.component_size,
                                 args.moves, args.repeats, rng),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
