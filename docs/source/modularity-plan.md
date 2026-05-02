# Modularity sweep via Pachner moves — implementation plan

Living document tracking the multi-phase build of an in-C++ Newman-Girvan
modularity sweep driven by transactional Pachner moves on a CDT
spacetime.  Updated as we go; checkboxes flip from `[ ]` to `[x]` when a
piece lands.  See the **Progress log** at the bottom for date-stamped
notes.

## Goal

Reproduce the (Q, D_S) trajectory plot from
`examples/modularity.py` — but driving the trajectory via Pachner moves
on a real simplicial spacetime instead of via abstract edge swaps on a
synthetic graph.  Two derived graphs from the same evolving spacetime:

- **1-skeleton** (vertices + edges of the simplicial complex).  Used
  for Newman-Girvan modularity Q with implicit labels
  `label(v) = v.id % M`.
- **Dual graph** (top simplices, with edges where they share a
  (d-1)-face).  Used for the spectral dimension D_S via heat-kernel
  diffusion.

The Python driver is a thin CLI + plotting wrapper; **the heavy lifting
moves into C++**.  Coverage scales over simplex dimension d ∈ {2, 3, 4, 5}.

## Architecture overview

```
┌────────────────────────────────────────────────────────┐
│  examples/modularity-cpp.py  (Python: CLI + plotting)  │
└────────────────┬───────────────────────────────────────┘
                 │  pybind11
                 ▼
┌────────────────────────────────────────────────────────┐
│  ModularityOptimizer  (C++)                            │
│  - informed Pachner proposal (community-aware)         │
│  - Q-direction acceptance filter                       │
│  - threshold-driven D_S measurement                    │
└──┬───────────────┬────────────────┬────────────────────┘
   │               │                │
   ▼               ▼                ▼
┌────────────┐ ┌─────────────┐  ┌────────────────────┐
│ PachnerMove│ │ Modularity  │  │ SparseGraph +      │
│ (apply/    │ │ (Q on       │  │ HeatKernel         │
│  rollback) │ │  1-skeleton)│  │ (D_S on dual)      │
└──┬─────────┘ └──┬──────────┘  └──┬─────────────────┘
   │              │                │
   └──────────────┼────────────────┘
                  ▼
        ┌──────────────────┐
        │   Spacetime      │  ← shared mutable state
        │  (vertices,      │
        │   edges,         │
        │   simplices)     │
        └──────────────────┘
```

## Phase plan

### Phase 0 — Safety net (tasks 15, 16) ✅

Before touching any existing CDT code, lock in current behavior.

- [x] **Task 15**: Write this plan document.
- [x] **Task 16**: `tests/test_pachner_characterization.py` (15 tests,
  all passing on current main).  Coverage:
  - **State-unchanged-on-rejection**: across high-k4, low-k4, and
    high-epsilon coupling regimes, every `False` return leaves the
    full state (top-simplex fingerprints, edge fingerprints, vertex
    IDs, counts) byte-identical.  This is the linchpin guarantee
    for the refactor.
  - **Edge-inventory deltas**: `add`/`flip`/`iflip`/`shift` are
    edge-monotonic — they only insert into `EdgeList`, never remove.
    `remove` deletes exactly the dropped vertex's incident edges
    (CDT.cpp:321-337); the dropped and added edge sets are disjoint.
  - **Coface integrity (eventually consistent)**: tessera registers
    cofaces lazily in `Simplex::getFacets()`.  After a Pachner move
    that creates new top simplices, those facets aren't registered
    until something walks them — typically the next
    `getDualAdjacency` call.  The locked-in invariant is:
    *after a walk*, every facet has its parent in its coface list.
  - **Action-delta consistency**: `computeAction()` delta matches
    `(k0, k4, delta, epsilon) · (dN0, dN41, dN32)` exactly, for every
    accepted move type.

#### Discoveries from Phase 0 — informs the refactor

1. **`Edge::toHash` is *not* fingerprint-stable across vertex
   relabeling**.  `Spacetime::swapVertexLabels` rewrites edge
   fingerprints in place via `EdgeList::rekeyEdge`.  Tests that need
   stable edge fingerprints must `setRelabelVertices(False)`.  For
   the AddMove rollback design: capturing edges by `EdgePtr` (not
   by hash) is the correct discipline.
2. **`removeSimplex` does not cascade-delete edges** — but
   `CDT::remove()` *does* explicitly delete edges incident to the
   dropped vertex (CDT.cpp:321-337).  RemoveMove rollback must
   capture those edges' (source ID, target ID, squaredLength) so it
   can reinsert them.
3. **Coface registration is lazy**.  AddMove/FlipMove/etc. do not
   need to force `getFacets()` on newly created simplices — that
   matches existing CDT behavior and tests don't care until something
   walks the structure.  The PachnerMove rollback must therefore not
   *remove* coface registrations on rollback (since they may not have
   been added yet).  We can defer to the lazy mechanism: rollback
   removes simplices, and any registrations they had are torn down
   by `removeSimplex` calling `facet.removeCoface(simplex)`.

### Phase 1 — Transactional Pachner infrastructure (tasks 1, 2, 3) ✅

- [x] **Task 1**: ``EdgeList::tryAdd`` returning ``std::pair<EdgePtr, bool>``.
  The bool reports whether the edge was freshly inserted (vs. found
  existing).  Existing ``add`` overloads kept intact for compatibility.
- [x] **Task 2**: ``Spacetime::createSimplexTracked`` returning a
  ``CreateSimplexResult { simplex, created, newEdges }`` struct.  The
  existing ``createSimplex(VertexPtrs)`` now delegates to it (zero
  behavioral change for existing callers).  ``newEdges`` is the list
  of edges this call freshly inserted into the EdgeList — captured via
  ``tryAdd``'s insertion bit.
- [x] **Task 3**: ``include/simulations/PachnerMove.h`` — abstract
  interface with ``propose()``, ``apply()``, ``rollback()``,
  combinatorial deltas (``dN0``, ``dN41``, ``dN32``), Metropolis log
  prefactor, ``touchedVertexIds()`` for informed proposals, and a
  ``moveType()`` tag.  Deliberately does *not* compute ΔS — the move
  reports its combinatorial deltas, the caller (CDT) plugs them into
  the action.  Keeps the move purely about geometry.

### Phase 2 — Concrete move classes (tasks 4–8) ✅

Each move shipped with a round-trip apply/rollback test asserting
byte-identical state recovery.

- [x] **Task 4**: `ShiftMove` (3,3) — 20 tests.
- [x] **Task 5**: `FlipMove` (2,d) — 17 tests.
- [x] **Task 6**: `IFlipMove` (d,2) — 10 tests.
- [x] **Task 7**: `AddMove` (2,2d) — 20 tests; relabel rollback works.
- [x] **Task 8**: `RemoveMove` (2d,2) — 18 tests; deleted-edge
  reinsertion uses captured (sourceId, targetId, squaredLength)
  triples to avoid stale `EdgePtr`s.

Per user feedback, headers + sources moved from `simulations/` to
`spacetime/` since Pachner moves apply to any spacetime.

### Phase 3 — CDT integration (tasks 9, 10) ✅

- [x] **Task 9**: `CDT::add()`/`remove()`/`flip()`/`iflip()`/
  `shiftImpl()` each refactored to a ~6-line wrapper:
  construct move → propose → accept(action) → apply.  All 98
  existing CDT+Pachner tests still pass.
- [x] **Task 10**: pybind11 bindings — `tessera.PachnerMove`,
  `tessera.AddMove/RemoveMove/FlipMove/IFlipMove/ShiftMove`,
  `cdt.proposeAdd()` etc.; 13 factory tests cover RNG sharing,
  no-mutation-on-propose, and end-to-end propose/apply/rollback
  through the bindings.

### Phase 4 — Observables (tasks 11, 12) ✅

- [x] **Task 11**: `include/observables/SparseGraph.h` +
  `HeatKernel.h`.
  - `SparseGraph`: CSR-stored undirected graph; built from
    `Spacetime::getDualAdjacency()`; `nNodes`, `nEdges`,
    `isBipartite` (BFS 2-coloring).
  - `HeatKernel`: Krylov-Lanczos approximation of
    `e^{-tL_sym}` diagonal entries on the symmetric normalized
    Laplacian.  Standalone — no Eigen.  Includes a hand-rolled
    Padé-13 with scaling-and-squaring for the small (≤30×30)
    tridiagonal projection.  ~250 lines.
  - `SparseGraph::spectralDimension(...)`: log-spaced t grid,
    diffuse, centered-difference D_S, mean over small/large
    tail-fractions.  Mirrors `examples/spectral_dimension.py`'s
    extraction.
- [x] **Task 12**: `Spacetime::getDualGraph()` +
  `modularityOnSkeleton(int M)`.

### Phase 5 — Optimizer (task 13) ✅

- [x] **Task 13**: `include/observables/ModularityOptimizer.h`.
  - Config: targetDq, maxIterations, nDiffusionWalks, maxSigma,
    bipartiteCheckFrequency, negativeRetryMax, epsilonQMax,
    krylovDim, targetNModules.
  - `Measurement` struct: Q, dsSmall, dsLarge, nVertices, nEdges,
    nSimplices, iter, direction.
  - `sweep(CDT& cdt, std::string direction, ProgressCallback)`
    — drives Pachner moves with **informed proposals**:
      - Score candidate moves by community alignment of touched
        vertices (analog of modularity.py's Type 1 swap).
      - Propose move; if `propose()` fails, try another type.
      - `apply()`; compute Q; if direction matches, commit
        (record measurement at every target_dq crossing); else
        `rollback()`.
      - Bipartite-detection and negative-D_S retry safety nets
        on every measurement, mirrored from modularity.py.

### Phase 6 — Python driver (task 14) ✅

- [x] **Task 14**: `examples/modularity-cpp.py` — thin CLI driver:
  - Build CDT spacetime per dimension d ∈ {2, 3, 4, 5}.
  - Optional `cdt.tune()` + `cdt.sweep(N)` thermalization.
  - Run `ModularityOptimizer` per d, collect measurements.
  - Multi-d overlay plot (one curve per d, viridis colormap).
  - Same status messages as `modularity.py`.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pachner rollback bookkeeping bug — silent state corruption that doesn't surface for many moves | High | High | Round-trip apply/rollback test per move type; full-fingerprint equality assertion |
| Refactor breaks existing test_pachner_*.py expectations | Medium | High | Phase-0 characterization tests; refactor only after they're green |
| Krylov-Lanczos heat kernel disagrees with scipy expm_multiply | Medium | Medium | Cross-check on small (~50-node) graph with known spectrum before use; keep krylovDim configurable |
| `cdt.tune()` fails to find pseudo-critical k4 for d ∉ {4} | Medium | Low | `tune()` is an existing well-tested operation; if it fails for d=5 we make CDT thermalization optional and document |
| EdgeList::tryAdd signature change breaks callers | Low | Medium | Add new overload, leave existing intact; migrate callers in a separate commit |
| Vertex relabeling during AddMove makes rollback ambiguous | Medium | High | AddMove records the swap partner; rollback un-swaps before removing the vertex |

## Open questions

1. **Informed-proposal scoring function**.  How do we score Pachner
   moves by community alignment?  First pass: count how many of the
   move's touched vertices share a label.  For up sweeps, prefer moves
   where touched vertices are *not* all in the same community
   (re-arranging them creates intra-community structure).  For down
   sweeps, prefer moves where touched vertices *are* mostly in one
   community (re-arranging tends to spread them out).  Will revisit
   after implementation reveals what works.

2. **D_S measurement frequency**.  Every target_dq crossing matches
   modularity.py.  Pachner moves can produce ΔQ > target_dq in a single
   step (especially add/remove which change |E|), so the Q-axis sampling
   will be coarser/uneven than modularity.py — acceptable per
   discussion.

3. **CDT.sweep() vs. ModularityOptimizer.sweep()**.  Should the
   optimizer reuse `cdt.sweep()` mechanics?  Decision: no.  The
   optimizer needs custom acceptance (Q-direction, not action-Metropolis),
   so it bypasses `CDT::sweep()` and uses the new `cdt.proposeXxx()`
   factories directly.

## Progress log

- **2026-05-02**: Plan v1 drafted.  Task list of 16 items established.
- **2026-05-02**: Phase 0 complete.  `tests/test_pachner_characterization.py`
  has 15 tests, all passing on current main.  Existing 69-test Pachner
  suite still green.  Three discoveries from writing the tests are
  documented under Phase 0 above and inform the AddMove/RemoveMove
  rollback design.
- **2026-05-02**: Phase 1 complete.  ``EdgeList::tryAdd`` and
  ``Spacetime::createSimplexTracked`` shipped.  ``include/simulations/
  PachnerMove.h`` defines the transactional interface.  Build green;
  characterization tests + existing Pachner suite all pass.  Next:
  Phase 2, starting with ``ShiftMove`` (simplest move — no vertex/edge
  net change, single-step rollback validates the design before tackling
  the harder moves).
- **2026-05-02**: Phase 2 complete.  All five move classes
  (``ShiftMove``, ``FlipMove``, ``IFlipMove``, ``AddMove``,
  ``RemoveMove``) shipped with apply/rollback, bound to Python, each
  with its own test suite (15-20 tests).  Headers + sources moved
  from ``simulations/`` to ``spacetime/`` (more accurate home — moves
  apply to any spacetime, not just CDT).  AddMove rollback handles
  vertex relabeling; RemoveMove rollback recreates deleted edges
  (with original squared lengths) and the deleted vertex.  Build
  green; 109/109 Pachner-related tests pass.
- **2026-05-02**: Phase 3 complete.  ``CDT::add()`` /
  ``remove()`` / ``flip()`` / ``iflip()`` / ``shiftImpl()`` refactored
  to use the new transactional move classes — each is now a
  ~6-line wrapper around propose / Metropolis-accept / apply.
  ``CDT::proposeXxx()`` factories exposed for the modularity
  optimizer.  All 98 existing CDT/Pachner tests + 13 new factory
  tests pass.
- **2026-05-02**: Phase 4 complete.  ``include/observables/
  SparseGraph.h`` + ``HeatKernel.h`` shipped — Krylov-Lanczos
  approximation of ``e^{-tL_sym}`` diagonal, no Eigen dependency,
  hand-rolled Padé-13 with scaling-and-squaring.  Two real bugs
  caught and fixed by the test suite: (1) inner/outer Padé
  coefficients were swapped; (2) Lanczos was breaking one iteration
  early due to a premature ``V.size() >= krylovDim`` exit.  Both
  reflected in regression tests against analytic heat-kernel values
  on path/triangle graphs.  ``Spacetime::getDualGraph()`` and
  ``modularityOnSkeleton(M)`` shipped + tested.
- **2026-05-02**: Phase 5 complete.  ``ModularityOptimizer`` shipped
  with sweep("up" | "down") driving Pachner moves with Q-direction
  acceptance.  Threshold-driven D_S measurement at every
  ``targetDq`` crossing.  Up-sweep early-exit when Q is within
  ``epsilonQMax`` of ``1 - 1/M``.  Optional progress callback.
  10/10 optimizer tests pass.
- **2026-05-02**: Phase 6 complete.  ``examples/modularity-cpp.py``
  shipped — thin Python driver: per-dimension d in {2,3,4,5}, build
  CDT spacetime, optionally tune+sweep, run ``ModularityOptimizer``,
  overlay-plot D_S vs Q.  3/3 driver smoke tests pass.

  **Final tally**: 162 new tests + 83 existing CDT/Pachner tests, all
  passing.  Build green throughout.  Project complete.
