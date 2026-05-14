# Wilson loops

This page documents tessera's Wilson-loop observable: a holonomy-like
quantity around closed paths in the **dual graph** of a triangulated
`Spacetime`, evaluable in three modes (combinatorial topology, deficit-
angle curvature, or CDT causal orientation) and built from three
canonical loop-shape generators (hinge loops, BFS-discovered dual-
lattice loops, geodesic cycles).

## Quick taxonomy

| Concept                       | Class / enum                  | Header                              |
|-------------------------------|-------------------------------|-------------------------------------|
| Evaluation mode               | `tessera.WilsonMode`          | `include/observables/WilsonLoop.h`  |
| Loop-shape generator          | `tessera.LoopType`            | `include/observables/WilsonLoop.h`  |
| Loop data                     | `tessera.LoopPath`            | `include/observables/WilsonLoop.h`  |
| Single-loop result            | `tessera.WilsonResult`        | `include/observables/WilsonLoop.h`  |
| Workflow object               | `tessera.WilsonLoop`          | `include/observables/WilsonLoop.h`  |

All names above are exposed in `tessera`'s top-level namespace.

## Why dual-graph holonomy

On a triangulated $d$-dimensional spacetime, the **dual graph** has the
top-simplices ($d$-simplices) as nodes and shared facets ($(d-1)$-
simplices) as edges. A closed walk on this dual graph corresponds to a
loop that crosses successive facets, returning to its starting
top-simplex. The hinges ($(d-2)$-simplices) enclosed by such a loop
carry the local curvature (via their deficit angles in Regge calculus),
and the cyclic time-orientation pattern of the loop's simplices
carries the local causal structure (in CDT).

A Wilson loop in lattice gauge theory is the trace of a product of
parallel-transport operators around a closed path; on a curved
triangulation without an explicit gauge field, the closest natural
analogue is the holonomy of the Levi-Civita connection, which in 2D
reduces to $W = \cos(\sum_h \varepsilon_h)$ with $\varepsilon_h$ the
deficit angle at each enclosed hinge, and in higher $d$ obeys
$W = \frac{(d-2) + 2\cos\varepsilon}{d}$ for an elementary hinge loop
in U(1) approximation {cite}`Regge1961,Williams1992`. tessera computes
exactly this in `DEFICIT_ANGLE` mode; the other two modes give
complementary topological and causal probes of the same loop.

## Evaluation modes

`tessera.WilsonMode` selects one of three modes:

### `COMBINATORIAL`

Pure topology. Computes the **loop size** (number of top-simplices in
the loop) and the count of **enclosed hinges** (hinges contained in
every simplex of the loop, i.e. shared by every step of the walk).

```python
from tessera import WilsonLoop, WilsonMode
wl = WilsonLoop(spacetime)
loop = wl.geodesicLoop(some_simplex)
r = wl.evaluate(loop, WilsonMode.COMBINATORIAL)
print(r.value)            # = loop size (number of simplices)
print(r.enclosedHinges)   # number of hinges enclosed
print(r.contractible)     # True iff enclosedHinges == 0
```

`contractible` is `True` when no hinge is enclosed — the loop bounds a
disc and is homotopically trivial on the dual graph. A non-contractible
loop with no enclosed hinge would indicate a genuinely topological
nontriviality of the spacetime (a non-simply-connected manifold).

The combinatorial mode is the fastest of the three and the right tool
for ensemble-level questions about loop topology distributions (how
often is a loop contractible? what's the mean enclosed-hinge count at
loop size $\ell$?).

### `DEFICIT_ANGLE`

Curvature via Regge calculus.

For a **hinge loop** (a loop enclosing exactly one hinge $h$), the
Wilson value is exact:

$$
W = \frac{(d-2) + 2 \cos\varepsilon_h}{d}
$$

where $\varepsilon_h$ is the deficit angle at $h$ (in radians) and $d$
is the spacetime dimension. For a flat hinge ($\varepsilon = 0$), $W =
1$; the magnitude of the deviation from 1 is a direct measure of local
curvature.

For a **general loop** enclosing multiple hinges, tessera applies the
U(1)-approximation:

$$
W = \prod_{h\,\in\,\text{enclosed}} \cos\varepsilon_h
$$

This is exact in 2D (where there's a single hinge per loop on a flat
manifold and the product collapses) and a good leading-order estimate
in higher dimensions for small deficit angles.

```python
r = wl.evaluate(loop, WilsonMode.DEFICIT_ANGLE)
print(r.value)            # in [-1, 1] for U(1) approx; near 1 = flat
print(r.enclosedHinges)
```

### `CAUSAL`

CDT-style causal-structure probe. Walks the loop's top-simplices and
counts the **net change in time orientation** as the loop crosses each
shared facet:

$$
W_{\rm causal}(\gamma) \;=\; \sum_{i \in \gamma} \mathrm{sgn}\bigl(t_{f}^{(i+1)} - t_{f}^{(i)}\bigr) \cdot \mathbb{1}\bigl[|\Delta t_f| > \tfrac12\bigr]
$$

where $t_f^{(i)}$ is the "final time" stamp of simplex $i$ in the CDT
foliation and the indicator gates spurious near-zero crossings. The
result `causalWindingNumber` is the cumulative signed count; a
non-zero value indicates a loop that crosses a causal-boundary
structure (e.g. a black-hole horizon's bifurcation surface or a
foliation jump in a CDT triangulation).

```python
r = wl.evaluate(loop, WilsonMode.CAUSAL)
print(r.causalWindingNumber)   # signed integer
print(r.value)                 # = causalWindingNumber as a double
```

## Loop-shape generators

`tessera.LoopType` enumerates the three canonical loop families.
`WilsonLoop` exposes one factory method per family.

### `HINGE` — `hingeLoop(h)`

The elementary loop of top-simplices around a given hinge $h$,
ordered cyclically. In 2D this is the standard "plaquette" loop around
a vertex; in 3D it's the loop around an edge; in 4D it's the loop
around a triangle. The hinge loop has the property of enclosing exactly
one hinge ($h$ itself), which makes the `DEFICIT_ANGLE` formula above
exact.

```python
for h in spacetime.getHinges():
    loop = wl.hingeLoop(h)
    r = wl.evaluate(loop, WilsonMode.DEFICIT_ANGLE)
    # r.value is the U(1) Wilson value for the curvature at h
```

### `DUAL_LATTICE` — `dualLatticeLoop(start, targetLength)`

Discovers a loop of approximately `targetLength` simplices by
breadth-first search from `start` on the dual graph. Useful when you
want a population of loops at a fixed scale (analogous to specifying
the size of Wilson loops in lattice gauge theory).

```python
loop = wl.dualLatticeLoop(some_simplex, targetLength=20)
# Not guaranteed to be exactly length 20 — BFS may overshoot or
# return a shorter loop if the local connectivity doesn't permit
# closing at the target size.
```

### `GEODESIC` — `geodesicLoop(start)`

Shortest cycle through `start` on the dual graph (Dijkstra over a
suitably defined edge metric). Captures the local "girth" of the dual
graph around `start`.

```python
loop = wl.geodesicLoop(some_simplex)
print(len(loop))   # girth at this simplex
```

## `LoopPath` and `WilsonResult`

### `LoopPath`

A `LoopPath` records the loop as both an ordered sequence of top-
simplices and the shared facets between consecutive entries.

```python
print(loop.simplices)   # list of SimplexPtr, ordered
print(loop.facets)      # list of SimplexPtr, facets[i] = simplices[i] ∩ simplices[i+1 mod n]
print(len(loop))        # = len(loop.simplices)
```

The facet list is redundant given the simplex list (one can recover
the shared facet between any two adjacent simplices), but it's
materialised once at construction so downstream evaluation modes can
read it without re-scanning vertices.

### `WilsonResult`

A `WilsonResult` carries the scalar value plus the diagnostics
relevant to each mode:

| Field                   | Type   | Populated in     | Meaning                          |
|-------------------------|--------|-------------------|----------------------------------|
| `value`                 | double | all modes         | primary scalar (mode-specific)   |
| `loopSize`              | int    | all modes         | number of simplices in the loop  |
| `enclosedHinges`        | int    | COMBIN, DEFICIT   | hinges shared by every simplex   |
| `contractible`          | bool   | COMBIN            | `True` iff no hinge is enclosed  |
| `causalWindingNumber`   | int    | CAUSAL            | net time-orientation winding     |

Fields not populated by a given mode hold their zero / default value;
don't read them in a mode where they aren't set.

## End-to-end examples

### Example 1: hinge-loop curvature scan

Measure the deficit-angle Wilson value at every hinge of a
triangulation and look at its distribution.

```python
import statistics
from tessera import WilsonLoop, WilsonMode

wl = WilsonLoop(spacetime)
wl.measureAllHinges(WilsonMode.DEFICIT_ANGLE)
values = [m.value for m in wl.getMeasurements()]

print(f"n hinges: {len(values)}")
print(f"mean    : {statistics.mean(values):.4f}")
print(f"min/max : {min(values):.4f} / {max(values):.4f}")
# A flat manifold gives all values ≈ 1; sharper curvature pulls values
# away from 1.
```

### Example 2: contractibility statistics at a fixed loop size

```python
from tessera import WilsonLoop, WilsonMode, LoopType

wl = WilsonLoop(spacetime)
target = 8
n_contractible = 0
n_total = 0
for sigma in spacetime.getTopSimplices():
    loop = wl.dualLatticeLoop(sigma, targetLength=target)
    if len(loop) >= 2:
        r = wl.evaluate(loop, WilsonMode.COMBINATORIAL)
        n_total += 1
        if r.contractible:
            n_contractible += 1
print(f"contractible fraction at L={target}: "
      f"{n_contractible / n_total:.2%}")
```

### Example 3: ensemble averages by loop size

`getAverageBySize()` aggregates accumulated measurements into a `{size:
mean_value}` map — the standard form for Creutz-ratio-style analyses.

```python
wl = WilsonLoop(spacetime)
for sigma in spacetime.getTopSimplices()[:1000]:
    loop = wl.geodesicLoop(sigma)
    wl.measure(loop, WilsonMode.DEFICIT_ANGLE)

avg_by_size = wl.getAverageBySize()
for size in sorted(avg_by_size):
    print(f"L = {size}: <W> = {avg_by_size[size]:.4f}")
```

### Example 4: causal-winding detection on a CDT triangulation

```python
from tessera import WilsonLoop, WilsonMode

wl = WilsonLoop(cdt_spacetime)
wl.measureAllHinges(WilsonMode.CAUSAL)
windings = [m.causalWindingNumber for m in wl.getMeasurements()]
nonzero  = sum(1 for w in windings if w != 0)
print(f"non-contractible-in-time loops: {nonzero} / {len(windings)}")
```

A non-zero causal winding around a hinge means the timelike order of
the simplices flips during the loop — a sign of a non-trivial causal
structure (e.g. a CDT slice boundary or a topologically non-trivial
foliation).

## Measurement bookkeeping

`WilsonLoop` accumulates a `List[WilsonResult]` internally as you call
`measure(loop, mode)` or `measureAllHinges(mode)`. The standard
accessors are:

```python
wl.measure(loop, WilsonMode.DEFICIT_ANGLE)
wl.measure(other_loop, WilsonMode.DEFICIT_ANGLE)

print(len(wl.getMeasurements()))      # 2
print(wl.getAverageBySize())          # {size: mean value}

wl.reset()                            # clear accumulated measurements
print(len(wl.getMeasurements()))      # 0
```

`measureAllHinges(mode)` is the bulk shortcut for a curvature scan: it
walks every $(d-2)$-simplex of the spacetime, generates its hinge loop,
and records the evaluation. Skips loops of size < 2 (degenerate cases
where the hinge isn't actually surrounded by ≥ 2 distinct top-
simplices, typically on boundary triangulations).

## Performance notes

* `hingeLoop` is $O(\#\sigma)$ for the ordering scan, where $\#\sigma$
  is the number of top-simplices sharing the hinge — typically small.
* `dualLatticeLoop` is a bounded-depth BFS, $O(\#\text{visited} \cdot
  \#\text{dual neighbours})$. Worst case $O(\#\sigma)$ if the loop
  wraps far.
* `geodesicLoop` is $O(\#\sigma \log \#\sigma)$ in the worst case
  (Dijkstra on the dual graph).
* `evaluateDeficitAngle` is $O(\#\text{loop simplices} \cdot
  \#\text{hinges})$ because it scans hinges for membership in every
  loop simplex; for small loops this is cheap.

For ensemble-level scans across all hinges of a large triangulation
(`measureAllHinges`), the dominant cost is the per-hinge `evaluate`
call. At $N_{\rm hinge} = 10^4$ and small loops the full sweep is
seconds.

## Runnable example

The script ``examples/wilson_loops_curvature_scan.py`` puts the
hinge-loop curvature scan and Creutz-ratio aggregation together in a
self-contained driver. It builds a small CDT triangulation, optionally
equilibrates it with a few Metropolis sweeps, and emits per-hinge
statistics plus an aggregated ``<W> by loop size`` table.

```bash
python examples/wilson_loops_curvature_scan.py \
    --n-simplices 200 --d 3 --equilibrate 200 \
    --out-json /tmp/wilson-scan.json
```

For a quick smoke test (no I/O):

```bash
python examples/wilson_loops_curvature_scan.py --n-simplices 80 --d 3
```

## On holonomy proper

This module computes the **scalar trace** of the Levi-Civita holonomy
(in the U(1) approximation for multi-hinge loops). It does not return
the full rotation-matrix holonomy — the SO($d{-}1{,}1$) element you'd
get from parallel-transporting a tangent frame around the loop — nor a
parallel-transport routine for arbitrary vectors. The per-hinge
deficit angle is available on each ``Simplex`` via
``Simplex.deficitAngle()`` and via ``ReggeSolver.deficitAngle(hinge)``;
in the Regge formalism the deficit angle IS the holonomy angle in the
plane normal to the hinge, so for hinge loops you have the full
holonomy up to choice of normal frame. Higher-rank Wilson loops
(SU($N$), spinor reps) are not implemented.

## Reading list

* {cite}`Regge1961` — Regge's discrete general relativity and deficit
  angles.
* {cite}`Williams1992` — Wilson loops on Regge triangulations.
* {cite}`AmbjornJurkiewiczLoll2005` — causal dynamical triangulations,
  causal foliation, time-orientation structure.

See also `docs/source/theory.md` for the broader Regge / CDT
background and `docs/source/simplices.md` for the mesh primitives that
underlie `LoopPath`.
