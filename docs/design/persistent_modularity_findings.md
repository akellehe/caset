# Label-free persistent modular components — findings (#765)

Implementation record for ticket #765 (epic #763, design spec section 8):
`include/observables/PersistentModularity.h`,
`src/observables/PersistentModularity.cpp`, the `ModularityOptimizer`
extension, and `tests/observables/test_persistent_modularity_python.py`.

## Exact identity and domain

On a finite nonnegative weighted undirected similarity graph (the complex
one-skeleton under a documented monotone weight map: `Unit` `w = 1`, or
`ExpNegAbsLength` `w = exp(-|l|)`), the implementation evaluates exactly

- generalized modularity
  `Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j/(2m)) [c_i = c_j]`
  through the per-community sufficient statistics
  `Q = sum_c [Sigma_in(c)/(2m) - gamma (S_c/(2m))^2]`, with the aggregated
  self-loop convention `A_CC = Sigma_in(C)`; and
- the cached local move gain
  `dQ(v: a->b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a)/(2 m^2)`,
  `O(deg v)` per move from cached community totals, so one local-move sweep
  is near `O(|E|)` up to revisits.

Both are closed forms in double arithmetic. The incremental ledger
(`Q_0` + accepted `dQ`, compensated summation) is compared against the cold
recomputation `modularityGamma(labels, gamma)` on every fixture; the
measured agreement is ~1e-16, asserted at 1e-14.

Effect classification (spec section 23): readout only. No ontology or
dynamics change; nothing here enters the emergence objective, and a
modularity read may not veto an otherwise certified fiber.

## Heuristic status

Partition discovery is deterministic multilevel aggregation (Louvain-style
local moves + aggregation) from the fixed seed sequence
`splitmix64(baseSeed + t)`. Global modularity maximization is NP-hard; no
global optimum is claimed. The best exact restart score is retained
(equal-score ties broken by sorted canonical hash lists) and the restart
spread `max - min` is reported. Communities carry no connectivity
guarantee. The score is blind to signed/complex Hodge weights.

## Label-freedom mechanism

Visit order and tie-breaking come from a canonical structural ranking:
iterated weighted color refinement (capped 1-WL over exact weight bits,
initial color from the ascending-order strength sum), then
individualization-refinement with BFS hop distances as the global splitting
signal. Component identity hashes (`ComponentId`, 32-hex, two independent
64-bit lanes) mix the aggregation level, sorted child hashes (lineage), and
the internal incidence tokens — oriented `(childHash_src, childHash_tgt,
weight bits)` at level 1, unordered child-hash pairs at aggregated levels
(aggregated weight bits are summation-order sensitive and are excluded;
the children already carry the exact level-below structure). Raw vertex
numbers never enter a hash or a rank comparison; the only id use is the
arbitrary-representative choice inside a structurally indistinguishable
refinement class (minimum cell id), which makes discovery a pure function
of the labeled graph — edge input order provably does not change the
result (tested).

## Positive results

- Planted disconnected (2xK6; K5/K6/K7) and planted modular (2xK8 + bridge,
  weighted 6-block) fixtures are recovered exactly, with `Q` equal to the
  hand-derived analytic values (2xK6: `Q = 1/2` exactly).
- Fortunato-Barthelemy ring (40 x K5, single links, `c > sqrt(2m)`): the
  exact evaluator reproduces the analytic resolution limit
  (`Q(pairs) > Q(singles)` at `gamma = 1`, reversed at `gamma = 4`);
  discovery merges cliques at `gamma = 1` and recovers all 40 at
  `gamma = 4`; no support-stable track spans the scan, so modularity does
  not hand the recursion a fake persistent scale.
- Homogeneous ring C60: arc size tracks `gamma`, restart spread is nonzero
  and surfaced, no full-range stable track.
- Relabeling: on unit/dyadic-weight fixtures the relabeled scan has
  bitwise-identical `q` and `qIncremental`, identical per-level canonical
  hash and size multisets, and identical track lifetimes; forced-partition
  fixtures also map supports pointwise under the permutation.
- Invalidation (`invalidatedAncestry`): touching one cell flags exactly the
  components whose support contains it — across every hierarchy level of
  every slice — plus their track; siblings stay valid (tested exhaustively
  against the full report).
- Continuity: `modularityGamma(labels, 1)` on the `Unit` graph equals
  `SparseGraph::modularity` and `Spacetime::modularityOnSkeleton` to
  <= 1e-15 / exactly; `Q_gamma` affinity `Q_2 = 2 Q_1 - Q_0` holds to
  1e-15. The legacy sweep path is untouched and its tests stay green.

## Negative results and limitations (recorded deliberately)

- On symmetric graphs (rings, FB ring below the limit, the toroidal CDT
  skeleton) a relabeling returns the automorphic image: identical scores
  and hash multisets, but supports rotated by a graph automorphism.
  Pointwise support equality under relabeling is only guaranteed for
  forced (asymmetry-pinned) partitions; the tests assert exactly this
  split.
- Automorphic twin components (two identical K6) share a canonical hash by
  construction; bookkeeping that must distinguish them (invalidation) is
  positional. This is intrinsic to structural identity, not a defect.
- Canonical hashes read the oriented incidence, so a rebuild that does not
  preserve stored edge source/target roles (`LiveComplex.relabel` via
  `Spacetime::fromCells`) changes hashes while scores and supports remain
  invariant. Persistence matching is support-based and unaffected.
- The FB `gamma = 1` discovered partition (22 communities, `Q = 0.9025`)
  is slightly below the ideal pairs partition (`Q = 0.90455...`): the
  deterministic heuristic does not always reach the known optimum. It is
  still above the singles score and no optimum claim is made.
- Individualization-refinement is capped (64 steps); a pathological union
  of many identical symmetric parts can exhaust the cap, after which
  remaining ties fall back to cell-id order (deterministic, documented).
- Non-dyadic weights leave ~1e-16 summation-order freedom in reported
  per-community statistics under relabeling (scores are still asserted at
  1e-14; measured 1e-16).

## Benchmark (near-O(|E|) sweep)

Warm timings (canonical ranking cached; `discover`, 1 restart, gamma = 1;
single-threaded, Release build on the development workstation):

| fixture | n | edges | cold s | warm s | warm us/edge |
|---|--:|--:|--:|--:|--:|
| ring | 1 000 | 1 000 | 0.001 | 0.001 | 0.87 |
| ring | 16 000 | 16 000 | 0.020 | 0.016 | 1.01 |
| ring | 128 000 | 128 000 | 0.238 | 0.193 | 1.51 |
| planted deg~8 | 2 000 | 8 225 | 0.004 | 0.003 | 0.39 |
| planted deg~8 | 32 000 | 132 884 | 0.118 | 0.092 | 0.69 |
| planted deg~8 | 64 000 | 265 699 | 0.290 | 0.257 | 0.97 |

Per-edge cost stays within ~1.7x across a 128x size range (hierarchy depth
and cache effects), consistent with near-`O(|E|)` per sweep. Before this
ticket no label-free discovery existed; the fixed-partition
`modularityOnSkeleton` path is unchanged (verified by its unchanged tests).

## Deferred interfaces

- Spectral-projector overlap: `setProjectorOverlapHook` is plumbed and
  reported per match (`projectorOverlap`, absent = unknown); a later ticket
  supplies the projectors. Matching decisions stay support-based here.
- `PersistenceTrack::weightAwareStatus` is a `Record`, `Null` (Python
  `None`) until the weight-aware gap/localization/persistence certificate
  tickets populate it. Unknown is never encoded as zero.
