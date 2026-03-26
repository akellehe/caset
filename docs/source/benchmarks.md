# Benchmarks

```{note}
Results on this page were generated with **caset {{version}}**.
The benchmark JSON log records the exact version, platform, and Python
version for each run so results can be compared across releases.
```

Build-time benchmarks for constructing CDT simplicial complexes across
dimensions 1D through 4D and a range of target sizes.

All timings are wall-clock averages over 5 repetitions on a single core,
measured with `time.perf_counter()`.  Run the benchmark script yourself to
get numbers for your hardware:

```bash
python examples/benchmarks/build_benchmark.py --save docs/source/assets/benchmarks/
```

The script writes a structured JSON log alongside the plots
(`benchmark_results.json`) for programmatic comparison across versions.

---

## Build Time vs. Complex Size

Build time scales roughly linearly with the number of target simplices in
every dimension.  Higher-dimensional complexes are more expensive per
simplex because each $d$-simplex carries $\binom{d+1}{2}$ edges and
$d + 1$ vertices that must be linked into the triangulation.

```{image} assets/benchmarks/build_time.png
:alt: Log-log plot of build time vs. target simplices for 1D-4D
:width: 80%
:align: center
```

---

## Build Throughput

Throughput (simplices per second) is highest in 2D (~70,000--95,000 simpl/s)
and decreases with dimension as the per-simplex bookkeeping grows.  4D
sustains ~30,000 simpl/s, meaning a 100k-simplex triangulation builds in
about 3 seconds.

```{image} assets/benchmarks/build_throughput.png
:alt: Bar chart of build throughput by dimension and target size
:width: 80%
:align: center
```

---

## Full Dashboard

The four-panel view combines build time, throughput, actual-vs-requested
size, and complex density (vertices and edges per simplex) across all
dimensions.

```{note}
1D CDT produces far fewer simplices than requested because the
triangulation is topologically a circle and the builder converges once the
minimal cell structure is complete.  The 1D data points are included for
completeness but are not directly comparable to 2D--4D.
```

```{image} assets/benchmarks/build_benchmarks.png
:alt: Four-panel benchmark dashboard
:width: 100%
:align: center
```

---

## Results Table

```{list-table} Build times (5-run average, v0.1.0)
:header-rows: 1
:widths: 8 15 15 15 15 15

* - Dim
  - Target
  - Actual
  - Vertices
  - Edges
  - Time (s)
* - 2D
  - 500
  - 500
  - 514
  - 1,007
  - 0.003
* - 2D
  - 10,000
  - 10,000
  - 10,042
  - 20,021
  - 0.075
* - 2D
  - 100,000
  - 100,000
  - 100,092
  - 200,046
  - 1.11
* - 3D
  - 500
  - 500
  - 521
  - 1,521
  - 0.028
* - 3D
  - 10,000
  - 10,000
  - 10,063
  - 30,063
  - 0.136
* - 3D
  - 100,000
  - 100,000
  - 100,138
  - 300,138
  - 1.78
* - 4D
  - 500
  - 500
  - 528
  - 2,042
  - 0.040
* - 4D
  - 10,000
  - 10,000
  - 10,084
  - 40,126
  - 0.177
* - 4D
  - 100,000
  - 100,000
  - 100,184
  - 400,276
  - 2.38
```

---

## Optimization History

The chart below compares the original unoptimized build times against the
current v{{version}} code.  The cumulative optimizations include:

- **Fingerprint kMax 64 → 8** — eliminated ~448 bytes of dead array per
  Fingerprint, reclaiming ~214 MB on a 100k lattice and dramatically
  improving cache utilization.
- **Per-simplex edges and cofaces: `unordered_set` → `vector`** — replaced
  hash tables with flat vectors for collections of 3-10 (edges) and 0-2
  (cofaces) elements, eliminating per-object hash table overhead.
- **Inlined trivial accessors** — moved `isTimelike()`, `size()`, `getTi()`,
  `getTf()`, `getOrientation()` from Simplex.cpp to the header so the
  compiler can inline them across translation units.
- **Return by const reference** — `getVertices()`, `getEdges()`,
  `getCofaces()`, `getSimplices()`, `getSource()`, `getTarget()` no longer
  copy containers or bump refcounts on every call.
- **O(1) simplex unregistration** — replaced linear scan with an index map.
- **Direct hash in `createSimplex()`** — eliminated temporary `Fingerprint`
  allocation by computing the XOR hash inline and using heterogeneous lookup.
- **CDT move locals** — replaced `unordered_set` with `vector` for small
  per-move vertex/simplex collections.

Build time improvement is **20-40%** across the board, with the largest
gains at smaller sizes where per-object overhead dominates.

```{image} assets/benchmarks/benchmark_comparison.png
:alt: Before vs. after benchmark comparison
:width: 100%
:align: center
```

To regenerate this comparison after further changes:

```bash
# Save a baseline
python examples/benchmarks/build_benchmark.py --save /tmp/baseline/

# ... make changes, rebuild ...

# Save and compare
python examples/benchmarks/build_benchmark.py --save /tmp/new/
python examples/benchmarks/compare_benchmarks.py \
    --before /tmp/baseline/benchmark_results.json \
    --after /tmp/new/benchmark_results.json \
    --save docs/source/assets/benchmarks/
```

---

## Running the Benchmark

```bash
# Default: 5 sizes x 4 dimensions x 5 repeats (~1 minute)
python examples/benchmarks/build_benchmark.py --save docs/source/assets/benchmarks/

# Quick check (fewer repeats)
python examples/benchmarks/build_benchmark.py --repeats 2 --save /tmp/bench/
```

The `--save` directory receives:

| File | Contents |
|:-----|:---------|
| `benchmark_results.json` | Structured log with metadata, per-trial timings, and summary statistics |
| `build_benchmarks.png` | Four-panel dashboard |
| `build_time.png` | Build time vs. size (standalone) |
| `build_throughput.png` | Throughput bar chart (standalone) |
