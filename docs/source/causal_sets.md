# Causal sets

This page documents tessera's causal-set machinery: the `Poset` primitive,
the `Spacetime → CausetChain` adapter, and the causal-comparison harness
that compares the entanglement-derived majorization order, the
Lieb–Robinson cone, and the causet order on the same label set.

The motivation, in one sentence: the underlying causal structure of a
Lorentzian manifold is captured by the **partial order** on its points,
and a spacetime can be reconstructed up to conformal factor from that
order alone {cite}`C-MalamentSorkin1977, C-BombelliLeeMeyerSorkin1987`. tessera
exposes this partial order as a concrete data structure
(`tessera.Poset`) and provides factories that derive it from a
triangulated `tessera.Spacetime` or from a Schwinger TDVP quench run.

## Quick taxonomy

| Concept                       | Class                                | Header                              |
|-------------------------------|--------------------------------------|-------------------------------------|
| Partial order (Hasse covers)  | `tessera.quantum.Poset`              | `include/Poset.h`                   |
| Spacetime → 1D causet adapter | `tessera.quantum.Causet`             | `include/quantum/CausetChain.hpp`  |
| Causet-as-data                | `tessera.quantum.CausetChain`        | `include/quantum/CausetChain.hpp`  |
| Pairwise order agreement      | `tessera.quantum.OrderAgreement`     | `include/Poset.h`                   |
| `compareOrders(a, b, nLabels)`| free fn in `tessera.quantum`         | `include/Poset.h`                   |
| (cut, time) label             | `tessera.quantum.LabelSpacetime`     | `include/quantum/CausalCompare.hpp`|
| Three orders bundle           | `tessera.quantum.CausalOrders`       | `include/quantum/CausalCompare.hpp`|
| End-to-end comparison report  | `tessera.quantum.CausalComparisonReport` | `include/quantum/CausalCompare.hpp` |

All names above are exposed at the documented Python paths.

## The `Poset` primitive

A `Poset` represents a strict partial order on integer-indexed nodes via
its **Hasse cover edges** — the transitive reduction of the strict
relation. `a → b` in the cover graph means "$a$ strictly precedes $b$
with no intermediate", and the full order is the transitive closure of
the covers.

All classes named `tessera.X` below live under `tessera.quantum.X` —
the bindings are co-located with the quantum subsystem because the
historical first user of `Poset` was the causal-comparison harness.
Examples below use `from tessera.quantum import Poset, ...`.

### Constructing a poset by hand

```python
from tessera.quantum import Poset

p = Poset(4)              # 4 nodes: 0, 1, 2, 3
p.addCover(0, 1)
p.addCover(1, 2)
p.addCover(1, 3)
# 0 → 1 → 2
#         ↘ 3

print(p.getNodeCount, p.getCoverCount)   # 4, 3
print(sorted(p.covers))                  # [(0, 1), (1, 2), (1, 3)]
```

`addCover(a, b)` adds the cover edge $a \to b$ without validation —
callers are responsible for transitivity and acyclicity. The standard
usage is to feed covers from a transitive-reduction algorithm where
duplicates can't arise. For replacing the full cover list at once, use
`setCovers([(a, b), ...])`.

### Exporting a Hasse diagram

```python
dot = p.toDot()           # Graphviz DOT
with open("/tmp/p.dot", "w") as f:
    f.write(dot)
# render with: dot -Tsvg /tmp/p.dot -o /tmp/p.svg
```

The DOT representation uses node IDs as labels and one directed edge per
cover. Suitable for visual sanity checks at $\lvert V \rvert \lesssim 100$.

### Comparing two posets on a shared label set

```python
from tessera.quantum import Poset, compareOrders

a = Poset(4); a.addCover(0, 1); a.addCover(1, 2); a.addCover(2, 3)
b = Poset(4); b.addCover(0, 2); b.addCover(2, 1); b.addCover(1, 3)

stats = compareOrders(a, b, 4)
print(stats.kendallTau)         # in [-1, 1]
print(stats.discordantFraction) # in [0, 1]
print(stats.hasseEditDistance)  # in [0, 1]
print(stats.nConcordant, stats.nDiscordant, stats.nOnlyA, stats.nOnlyB)
```

The five counts (`nConcordant`, `nDiscordant`, `nOnlyA`, `nOnlyB`,
`neither`) partition the $\binom{N}{2}$ unordered label pairs. They
underpin the causal-comparison harness described below.

`compareOrders` is $O(N^3)$ via Floyd–Warshall transitive closure, then
$O(N^2)$ counting. Practical up to a few thousand nodes.

## `Spacetime → CausetChain` adapter

A triangulated `Spacetime` carries a finer structure than a 1D chain:
its vertices are distributed across (potentially many) time slices, and
its directed timelike edges define the local causal structure. The
`Causet` adapter flattens this into a 1D **chain of antichains** — a
shape compatible with the existing Schwinger MPO machinery, which
expects a 1D lattice with explicit nearest-neighbour-style hopping.

```python
from tessera import Spacetime
from tessera.quantum import Causet

# Build or load a Spacetime (any of tessera's standard topologies):
st = Spacetime.fromGraphML("path/to/triangulation.graphml")

chain = Causet.chainFrom(st)
print(chain.nSites)              # total vertices across all slices
print(len(chain.antichains))     # number of time slices
print(chain.times[0],            # first slice's integer time
      chain.antichains[0])       # vertex IDs in the first slice
print(chain.vertexIds[:5])       # flat-lattice → Spacetime vertex ID
print(chain.hoppingPairs[:5])    # (i, j) timelike-edge couplings
```

### Algorithm

`Causet.chainFrom(st)` walks the Spacetime's vertices, groups them by
integer time slice (truncating `Vertex.getTime()`), and produces:

* **`antichains[s]`** — the sorted list of vertex IDs at `times[s]`.
  The slice order is ascending in `times`.
* **`vertexIds[flat_idx]`** — the inverse map: flat lattice site →
  Spacetime vertex ID. `flat_idx` enumerates the antichain list in
  order, so `flat_idx(s, p) = Σ_{r<s} |antichains[r]| + p` where `p` is
  the position within `antichains[s]`.
* **`hoppingPairs`** — the $(i, j)$ flat-site pairs coupled by
  adjacent-time-slice timelike edges. Stored once per pair with $i < j$;
  the MPO builder applies $\sigma^+\sigma^-_+ \sigma^-\sigma^+$
  symmetrically.
* **`partialOrder`** — the Hasse-cover Poset on flat-lattice IDs,
  inherited from `Poset.fromSpacetime(st)` (transitive reduction of the
  directed-edge graph).

Edges with spacelike or null squared length are ignored, as are any
timelike edges with `src.time == tgt.time` (no propagation across the
same slice). Edges spanning non-adjacent slices are skipped — they're
transitively reduced out by `Poset.fromSpacetime` and wouldn't
contribute a physical hopping term anyway.

### Reduced 1D chain

When every antichain has exactly one vertex, the chain-of-antichains
coincides with the standard 1D lattice and `hoppingPairs` reduces to
`[(0, 1), (1, 2), ..., (N-2, N-1)]`. In that case the existing
`SchwingerHamiltonian.mpoChain(...)` runs unchanged with `params.N =
chain.nSites` and `chain.hoppingPairs` as the hopping graph.

### Threading the chain into the Schwinger pipeline

`TDVPConfig.hoppingPairs` is the connection point. If empty (the
default), the Schwinger TDVP runs on the standard 1D chain with NN
hopping; non-empty selects the causet hopping graph instead.

```python
from tessera.quantum import TDVPConfig, SchwingerQuench

cfg = TDVPConfig()
cfg.N = chain.nSites
cfg.a = 1.0
cfg.g = 1.0
cfg.m = 0.5
cfg.L0 = 0.0
cfg.dmrgMaxBondDim = 64
cfg.dmrgNSweeps    = 12
cfg.i0 = 0
cfg.d  = 1
cfg.dt = 0.1
cfg.T  = 0.5
cfg.snapshotEvery = 1
cfg.hoppingPairs = chain.hoppingPairs    # ← the causet rewiring

r = SchwingerQuench(cfg).evolve()
```

The DMRG ground state and TDVP evolution now propagate excitations along
the causet's timelike-edge graph rather than the regular 1D chain.

## The causal-comparison harness

`SchwingerQuench.compareCausalOrders(vLr)` runs the full DMRG → quench →
TDVP pipeline and compares three partial orders on the (cut, time) label
set:

1. **$\preceq_{\rm maj}$** — strict-majorization on Schmidt spectra
   across cuts and time.
2. **$\preceq_{\rm LR}$** — Lieb–Robinson cone: $(A, s) \preceq_{\rm LR}
   (B, t)$ iff $s < t$ and $\mathrm{dist}(A, B) \leq v_{LR} \cdot (t-s)$.
3. **$\preceq_{\rm cs}$** — causet order: on a regular chain this is
   time-only; on a Spacetime-derived chain it reads off `partialOrder`.

Each pair returns an `OrderAgreement` with `kendallTau`,
`discordantFraction`, and `hasseEditDistance`.

```python
report = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
print(f"maj vs LR: tau = {report.majVsLr.kendallTau:.3f}")
print(f"maj vs cs: tau = {report.majVsCs.kendallTau:.3f}")
print(f"LR  vs cs: tau = {report.lrVsCs.kendallTau:.3f}")
```

The strongest invariant is $\preceq_{\rm LR} \subset \preceq_{\rm cs}$:
every LR pair is automatically a causet pair in the same direction on a
regular chain, so `report.lrVsCs.kendallTau == 1.0` is a sanity check
that the order extraction is consistent.

### Reading `OrderAgreement`

* `nConcordant` — pairs related the same way in both orders.
* `nDiscordant` — pairs related opposite ways in both orders.
* `nOnlyA` / `nOnlyB` — pairs related in only one of the two orders.
* `kendallTau = (nConcordant - nDiscordant) / nComparableBoth` ∈ [-1, 1].
* `discordantFraction = nDiscordant / nComparableBoth` ∈ [0, 1].
* `hasseEditDistance = |E_a △ E_b| / |E_a ∪ E_b|` ∈ [0, 1], the
  symmetric-difference fraction of cover edges.

For the methodology-level falsification test of the entanglement-causes-
spacetime claim, `nOnlyA` with $(A, B) = (\preceq_{\rm maj}, \preceq_{\rm
LR})$ is the count of majorization-related pairs that lie outside the
Lieb–Robinson cone — a non-zero number falsifies the strong claim.

## `LabelSpacetime` and `CausalOrders` directly

Power users who want to skip `SchwingerQuench` and stitch the orders
themselves can call `CausalOrders.fromSnapshots`:

```python
from tessera.quantum import (
    CausalOrders, StandardMajorization, TDVPConfig, SchwingerQuench,
)

cfg = TDVPConfig()
# ... fill in ...
cfg.recordSpectra = True   # required for CausalOrders.fromSnapshots

r = SchwingerQuench(cfg).evolve()

orders = CausalOrders.fromSnapshots(
    r.snapshots,
    vLr=1.0,
    predicate=StandardMajorization())

print(len(orders.labels))      # = (number of cuts) × (number of snapshots)
print(orders.labels[0])         # LabelSpacetime(cutIdx, tIdx, intervalI, intervalJ, time)
print(orders.maj.getCoverCount)
print(orders.lr.getCoverCount)
print(orders.cs.getCoverCount)
```

The three returned `Poset` objects share a node-id scheme: node `k`
corresponds to `orders.labels[k]`, which carries the original cut
position, snapshot index, and physical time.

## Reading list

* {cite}`C-MalamentSorkin1977` — the order-reconstructs-conformal-class
  theorem.
* {cite}`C-BombelliLeeMeyerSorkin1987` — the causet program; spacetime
  emerges from a locally finite partial order.
* {cite}`C-LiebRobinson1972` — the Lieb–Robinson bound that bounds the
  $\preceq_{\rm LR}$ order.
* {cite}`C-HastingsKoma2006` — refined LR bound for lattice systems
  relevant to the Schwinger TDVP setup.

See also [Emergent Causal Order from Majorization](quantum-experiments/earlier-work/emergent-causal-order-from-majorization.md) §1 and §4.4 for the
scientific motivation behind the three-order comparison, and
[Emergent Spectral Dimension from the Schwinger TDVP State](quantum-experiments/earlier-work/emergent-spectral-dimension-schwinger-tdvp.md) §5 for
the integration of `CausetChain` with the holography pipeline.

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: C-
:labelprefix: C
:style: unsrt
```
