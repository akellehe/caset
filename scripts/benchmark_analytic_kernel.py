#!/usr/bin/env python3
"""Benchmark the analytic-first kernel and cache contract (#764).

Examples:
    OMP_NUM_THREADS=8 python scripts/benchmark_analytic_kernel.py
    OMP_NUM_THREADS=8 python scripts/benchmark_analytic_kernel.py --grid 24

Five scenarios, each comparing the structured/incremental path against the
dense/cold reference ON THE SAME inputs and reporting both the median
timings and the measured agreement (the dense-reference error a Certificate
carries), so the numbers are before/after AND correctness in one run:

  local_metric    — one accepted edge-length change on a product-graph
                    spacetime: Woodbury solve from the old LU + touched-star
                    factors vs a cold refactor + solve.
  local_topology  — one accepted edge creation (combinatorial change, same
                    vertex set): the same comparison.
  product_complex — spectrum of an actual product complex: pairwise
                    one-particle sums from two factor eigensolves vs the
                    dense eigensolve of the assembled Kronecker sum.
  second_quantized— one N-particle occupation sector by subset sums vs the
                    eager full-Fock enumeration, plus the hopping-block
                    assembly + one-particle eigensolve for the coupled pair.
  cache           — random accepted local moves over disjoint components:
                    touched-star invalidation (recompute one component) vs
                    cold recomputation of every component per move.

Timings are medians over identical evaluations; mutation-driven scenarios
rebuild their fixture per repeat so every timed run does identical work.
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


def build_graph(num_vertices, edges):
    """Spacetime holding an explicit weighted graph; edges are
    (src, tgt, squared_length, phase)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for src, tgt, _, _ in edges:
        st.createSimplex([verts[src], verts[tgt]])
    by_pair = {(e.getSource().getId(), e.getTarget().getId()): e
               for e in st.getEdgeList().toVector()}
    for src, tgt, squared_length, phase in edges:
        edge = by_pair.get((src, tgt))
        sign = 1.0
        if edge is None:
            edge = by_pair[(tgt, src)]
            sign = -1.0
        edge.setLength(cmath.sqrt(complex(squared_length)))
        edge.setPhase(sign * phase)
    return st


def path_edges(offset, count, rng):
    return [(offset + i, offset + i + 1, 0.5 + 2.0 * rng.random(),
             rng.random() - 0.5) for i in range(count - 1)]


def grid_product(side, rng):
    """An actual product complex: the Cartesian product of two weighted
    paths (side vertices each), as factor spacetimes + the product
    spacetime + the vertex pairing."""
    a_edges = path_edges(0, side, rng)
    b_edges = path_edges(0, side, rng)
    factor_a = build_graph(side, a_edges)
    factor_b = build_graph(side, b_edges)
    product_edges = []
    for (u, up, w, phi) in a_edges:
        for v in range(side):
            product_edges.append((u * side + v, up * side + v, w, phi))
    for (v, vp, w, phi) in b_edges:
        for u in range(side):
            product_edges.append((u * side + v, u * side + vp, w, phi))
    product = build_graph(side * side, product_edges)
    pairing = [(u * side + v, u, v) for u in range(side) for v in range(side)]
    return factor_a, factor_b, product, pairing


def laplacian_of(st):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    n = len(ids)
    flat = cob.HodgeLaplacian(st).laplacian(0)
    return flat, n, {vid: i for i, vid in enumerate(ids)}


def shifted(flat, n, shift):
    """The static response operator L + shift * I (the design-spec 5.3
    resolvent pencil at -shift). The bare degree-zero Hodge operator
    L_0 = d_1 W_1^-1 d_1^T is exactly singular (its row sums vanish, so the
    constant is always in the kernel), which is why the meaningful repeated
    solve is the shifted one; the shift cancels in the update factors."""
    out = list(flat)
    for i in range(n):
        out[i * n + i] += shift
    return out


def benchmark_local_change(side, repeats, rng, structural, shift=1.0):
    """One accepted local move on the product-graph spacetime: Woodbury from
    the old factorization vs cold refactor + solve, on the shifted response
    operator. `structural` switches the move from a metric change
    (setLength) to a topology change (a new edge between existing
    vertices)."""
    factor_a, factor_b, product, _ = grid_product(side, rng)
    del factor_a, factor_b
    base_flat, n, index = laplacian_of(product)
    base_flat = shifted(base_flat, n, shift)
    solver = cob.LowRankUpdate(base_flat, n)  # factored once, outside moves

    edges = product.getEdgeList().toVector()
    if structural:
        # Accepted topology move: one new edge between two existing vertices
        # two rows apart (not previously adjacent).
        verts = {v.getId(): v for v in product.getVertexList().toVector()}
        u, w = 0, 2 * side  # (0,0) and (2,0) in the grid
        product.createSimplex([verts[u], verts[w]])
        created = [e for e in product.getEdgeList().toVector()
                   if {e.getSource().getId(), e.getTarget().getId()} == {u, w}]
        created[0].setLength(1.0 + 0j)
        created[0].setPhase(0.1)
        touched_ids = [u, w]
    else:
        edge = edges[len(edges) // 2]
        edge.setLength(cmath.sqrt(2.75 + 0j))
        edge.setPhase(0.3)
        touched_ids = [edge.getSource().getId(), edge.getTarget().getId()]

    updated_flat, n2, _ = laplacian_of(product)
    assert n2 == n
    updated_flat = shifted(updated_flat, n, shift)
    touched = [index[i] for i in touched_ids]
    rhs = [complex(a, b) for a, b in rng.normal(size=(n, 2))]

    def structured():
        # The full per-move incremental path: build the exact touched-star
        # factors, verify they span the change, then one Woodbury solve.
        factors = cob.LowRankUpdate.factorsFromTouched(base_flat, updated_flat,
                                                       n, touched)
        assert factors.spansChange
        solver.setUpdate(factors.left, factors.right, factors.rank)
        return solver.solve(rhs)

    def cold():
        cold_solver = cob.LowRankUpdate(updated_flat, n)
        return cold_solver.solve(rhs)

    structured_s, structured_result = median_timing(structured, repeats)
    cold_s, cold_result = median_timing(cold, repeats)
    # Marginal per-rhs cost once the factorizations exist (the repeated-
    # solve regime of projector/transport reads).
    cold_solver = cob.LowRankUpdate(updated_flat, n)
    structured_marginal_s, _ = median_timing(lambda: solver.solve(rhs),
                                             repeats)
    cold_marginal_s, _ = median_timing(lambda: cold_solver.solve(rhs), repeats)
    deviation = float(np.max(np.abs(np.array(structured_result.values) -
                                    np.array(cold_result.values))))
    return {
        "dimension": n,
        "shift": shift,
        "update_rank": solver.updateRank,
        "structured_median_s": structured_s,
        "cold_median_s": cold_s,
        "speedup": cold_s / structured_s if structured_s > 0 else float("inf"),
        "structured_marginal_solve_s": structured_marginal_s,
        "cold_marginal_solve_s": cold_marginal_s,
        "structured_residual": structured_result.certificate.residual,
        "cold_residual": cold_result.certificate.residual,
        "max_solution_deviation": deviation,
    }


def benchmark_product_complex(side, repeats, rng):
    factor_a, factor_b, product, pairing = grid_product(side, rng)
    cert = cob.KuennethProduct.productCertificate(product, factor_a, factor_b,
                                                  pairing)
    assert cert.holds(), cert.describe()
    flat_a, na, _ = laplacian_of(factor_a)
    flat_b, nb, _ = laplacian_of(factor_b)
    n = na * nb
    dense = cob.DenseReference(max(2 * n, 1024))

    def structured():
        spec_a = dense.spectrum(flat_a, na, True)
        spec_b = dense.spectrum(flat_b, nb, True)
        return cob.KuennethProduct.pairwiseSpectrum(spec_a.values,
                                                    spec_b.values)

    def dense_reference():
        kron = cob.KuennethProduct.kroneckerSum(flat_a, na, flat_b, nb)
        return dense.spectrum(kron, n, True)

    structured_s, structured_values = median_timing(structured, repeats)
    dense_s, dense_result = median_timing(dense_reference, repeats)
    deviation = float(np.max(np.abs(np.real(structured_values) -
                                    np.real(dense_result.values))))
    return {
        "factor_dimensions": [na, nb],
        "product_dimension": n,
        "certificate_residual": cert.residual,
        "structured_median_s": structured_s,
        "dense_median_s": dense_s,
        "speedup": dense_s / structured_s if structured_s > 0 else float("inf"),
        "max_spectrum_deviation": deviation,
    }


def benchmark_second_quantized(modes, particles, block, repeats, rng):
    spectrum = [complex(a, b) for a, b in rng.normal(size=(modes, 2))]

    def sector():
        return cob.OccupationSpectra.subsetSums(spectrum, particles)

    def eager_fock():
        return cob.OccupationSpectra.fockSums(spectrum, 1 << (modes + 1))

    sector_s, sector_values = median_timing(sector, repeats)
    fock_s, fock_values = median_timing(eager_fock, repeats)

    # Coupled-pair hopping block at the one-particle level + dense
    # verification on a small crossover fixture.
    h_a = rng.normal(size=(block, block)) + 1j * rng.normal(size=(block, block))
    h_a = h_a + h_a.conj().T
    h_b = rng.normal(size=(block, block)) + 1j * rng.normal(size=(block, block))
    h_b = h_b + h_b.conj().T
    coupling = 0.1 * (rng.normal(size=(block, block)) +
                      1j * rng.normal(size=(block, block)))

    def flat(matrix):
        return [complex(z) for z in np.asarray(matrix).reshape(-1)]

    dense = cob.DenseReference(max(4 * block, 512))

    def hopping():
        assembled = cob.OccupationSpectra.hoppingBlock(
            flat(h_a), block, flat(h_b), block, flat(coupling))
        return dense.spectrum(assembled, 2 * block, True)

    hopping_s, hopping_result = median_timing(hopping, repeats)
    numpy_spectrum = np.linalg.eigvalsh(
        np.block([[h_a, coupling], [coupling.conj().T, h_b]]))
    hopping_deviation = float(np.max(np.abs(
        np.real(hopping_result.values) - numpy_spectrum)))
    return {
        "modes": modes,
        "particles": particles,
        "sector_terms": len(sector_values),
        "fock_terms": len(fock_values),
        "sector_median_s": sector_s,
        "eager_fock_median_s": fock_s,
        "speedup": fock_s / sector_s if sector_s > 0 else float("inf"),
        "hopping_block_dimension": 2 * block,
        "hopping_median_s": hopping_s,
        "hopping_max_deviation": hopping_deviation,
    }


def benchmark_cache(components, component_size, moves, repeats, rng):
    """Random accepted local metric moves over disjoint ring components: the
    touched-star cache workflow (recompute ONE component's Hodge-block
    eigendecomposition per move, siblings served from cache) against cold
    recomputation of EVERY component per move."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    component_vertices = []
    for c in range(components):
        ids = [component_size * c + i for i in range(component_size)]
        component_vertices.append(ids)
        verts = [st.createVertex(i) for i in ids]
        for i in range(component_size):
            st.createSimplex([verts[i], verts[(i + 1) % component_size]])
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(0.5 + rng.random())))
        e.setPhase(rng.random() - 0.5)

    edges_by_component = [
        [e for e in st.getEdgeList().toVector()
         if e.getSource().getId() in set(ids)]
        for ids in component_vertices]

    def component_payload(component):
        """Cold recompute of one component: its k = 0 Hodge block (D - A,
        the HodgeLaplacian degree-0 convention) assembled from its own
        edges, plus the dense eigendecomposition — the 'component
        factorization / spectral projector' payload class."""
        ids = sorted(component_vertices[component])
        pos = {v: i for i, v in enumerate(ids)}
        m = len(ids)
        adjacency = np.zeros((m, m), dtype=complex)
        degree = np.zeros(m)
        for e in edges_by_component[component]:
            s = pos[e.getSource().getId()]
            t = pos[e.getTarget().getId()]
            squared = complex(e.getLength()) ** 2
            weight = squared * np.exp(1j * e.getPhase())
            adjacency[s, t] += weight
            adjacency[t, s] += np.conj(weight)
            degree[s] += abs(squared)
            degree[t] += abs(squared)
        block = np.diag(degree) - adjacency
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        return block, eigenvalues, eigenvectors

    cache = cob.AnalyticCache(st)
    cert = cob.Certificate.algebraicallyExact(
        cob.CertificateDomain.Static,
        cob.CertificateRegime.PositiveSemidefinite, 0.0, 1e-15)
    for c, ids in enumerate(component_vertices):
        cache.store(ids, "hodge-block-spectrum", 0, component_payload(c), cert)

    move_plan = [(int(rng.integers(components)),
                  int(rng.integers(component_size)),
                  0.5 + 2.0 * rng.random()) for _ in range(moves)]

    def apply_move(component, edge_index, squared):
        edge = edges_by_component[component][edge_index]
        edge.setLength(cmath.sqrt(complex(squared)))
        return edge

    def cached_workflow():
        payloads = None
        for component, edge_index, squared in move_plan:
            edge = apply_move(component, edge_index, squared)
            star = cob.TouchedStar()
            star.addChangedEdge(edge.getSource().getId(),
                                edge.getTarget().getId())
            cache.publish(star)
            payloads = []
            for c, ids in enumerate(component_vertices):
                cached = cache.fetch(ids, "hodge-block-spectrum", 0)
                if cached is None:
                    cached = component_payload(c)
                    cache.store(ids, "hodge-block-spectrum", 0, cached, cert)
                payloads.append(cached)
        return payloads

    def cold_workflow():
        payloads = None
        for component, edge_index, squared in move_plan:
            apply_move(component, edge_index, squared)
            payloads = [component_payload(c) for c in range(components)]
        return payloads

    cached_s, cached_payloads = median_timing(cached_workflow, repeats)
    cold_s, cold_payloads = median_timing(cold_workflow, repeats)
    max_deviation = 0.0
    for (cached_block, cached_eigenvalues, _), (cold_block, cold_eigenvalues,
                                                _) in zip(cached_payloads,
                                                          cold_payloads):
        max_deviation = max(
            max_deviation,
            float(np.max(np.abs(cached_block - cold_block))),
            float(np.max(np.abs(cached_eigenvalues - cold_eigenvalues))))
    return {
        "components": components,
        "component_size": component_size,
        "moves": moves,
        "cached_median_s": cached_s,
        "cold_median_s": cold_s,
        "speedup": cold_s / cached_s if cached_s > 0 else float("inf"),
        "hits": cache.hits,
        "misses": cache.misses,
        "invalidations": cache.invalidations,
        "max_payload_deviation": max_deviation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=20,
                        help="side of the product grid (product dim = side^2)")
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--block", type=int, default=96)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--component-size", type=int, default=48)
    parser.add_argument("--moves", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=764)
    args = parser.parse_args()
    if min(args.grid, args.modes, args.block, args.components,
           args.component_size, args.moves,
           args.repeats) < 1 or args.particles < 0 or args.seed < 0:
        parser.error("sizes and repeats must be positive; seed/particles "
                     "non-negative")

    rng = np.random.default_rng(args.seed)
    result = {
        "grid": args.grid,
        "repeats": args.repeats,
        "seed": args.seed,
        "local_metric": benchmark_local_change(args.grid, args.repeats, rng,
                                               structural=False),
        "local_topology": benchmark_local_change(args.grid, args.repeats, rng,
                                                 structural=True),
        "product_complex": benchmark_product_complex(args.grid, args.repeats,
                                                     rng),
        "second_quantized": benchmark_second_quantized(
            args.modes, args.particles, args.block, args.repeats, rng),
        "cache": benchmark_cache(args.components, args.component_size,
                                 args.moves, args.repeats, rng),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
