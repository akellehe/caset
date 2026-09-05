# Two-body cobordism map on degree-0 tetrahedral fibers: findings (#941, #942, #947)

Records live under `~/cobordism-runs/two-body/` (`run-seed*.json` first build,
`run2-*` block regions fixed, `run3-*` analytic gradient with phase descent,
`run4-*` analytic gradient single-threaded) and `~/cobordism-runs/choi-encoding/`
(`fiber-floor-*.log`, the input-node drives).

## What was built

- **Input nodes (#940).** A single Δ³ seed grown by stage 1 (gated cone-ins) and
  stage 2 (geometry) until its whole complex carries a four-term state as a band of its
  covariant degree-0 pencil on the seed's four vertices, scored by the least-squares
  leak of the target in that band (`useFiberResiduals`). Coefficients come from lengths
  and connection values only.
- **Interaction node (#941).** Two piped fibers attached to two vertex-disjoint
  tetrahedra of a third seed grown by cone-ins; the bulk read as the frame transfer
  T_AB (the coupling block of the whole between the two frames) in the Choi-decomposed
  or operator reading; χ scored as the projective Frobenius leak beside the two blocks'
  fiber residuals.
- **Analytic gradient (#947).** The Riesz-projector derivative of the band images and
  the pencil-operator derivative of the transfer, validated by the Euler identity of the
  projective residuals (Σ_e s_e dF_e = 0 to round-off) and reproducing the recorded
  finite-difference draws exactly, 10–200× faster.

## Input nodes

| draw | threads | trace (rounds of stage 1 + stage 2) | time |
|---|---|---|---|
| seed 0, finite differences | 1 | 0.63 → 0.31 → 0.25 → 2.3e-31 | 25 s |
| seed 0, analytic | 1 | 0.63 → 0.31 → 0.25 → 2.332459591314688e-31 (identical) | 3.9 s |
| seed 3, finite differences | 2 | 0.95 → 8.0e-13 (round 0) → 7.5e-15 | 15 min |
| seed 3, analytic | 1 | 0.95 → 8.5e-15 (round 0) | 75 s |
| seeds 1, 2, analytic | 2 | plateaus at 0.09–0.11 after six rounds | 4 s |
| XY runs, seeds 1–5 (inputs) | 1 | 0.02–0.06 after eight rounds; two draws to 1e-31 in four rounds | 3–10 s |

The bare tetrahedron cannot carry a state with Σψ ≠ 0 (its band above zero is
{z : Σz = 0} for every length), so growth is required; whether a draw reaches the
state depends on the stage-1 moves, which are not process-deterministic (OpenMP
candidate scoring), so a seed labels an attempt rather than reproducing it.

## Interaction node

χ for generic ψ, φ has singular values (30.4, 8.8, 0, 0): Schmidt rank two with a
second channel at 0.29 of the first.

| bulk | attachment pair | transfer singular values at start | after stage 2 (stationary) | two-body leak | block residuals |
|---|---|---|---|---|---|
| 8 cone-ins, 12 vertices | best σ₂/σ₁ = 0.217 | (0.387, 0.084, 0.050, 0) | 16 accepted steps | 0.925 | 0.67, 0.32 |
| 14 cone-ins, 18 vertices | best σ₂/σ₁ = 0.297 | (0.265, 0.079, 0, 0) | 24 accepted steps | 0.644 | 0.12, 0.15 |
| XY run4 seed 1, 8 cone-ins, six rounds with growth | first disjoint pair | | | 0.692 | 0.20, 0.08 |
| XY run4 seed 2, 8 cone-ins, six rounds with growth | first disjoint pair | | | 0.922 | 0.00, 0.50 |

Once the blocks carry their fibers, stage 2 goes stationary with the two-body leak
between 0.64 and 0.93; further rounds of growth do not move it. The frame transfer
between two vertex-disjoint tetrahedra couples them through the bulk's inverse metric
with one strong channel and a second at most 0.3 of it on these complexes, and χ
needs two comparable channels: the bulk's coupling rank between the two frames bounds
the map it can carry. The swap test on the frozen bulk stays at the same leak
(0.67 and 1.00), as it must while the bulk itself does not carry χ.

Three engine defects surfaced and are fixed along the way: the band read at half the
gap leaked a neighbouring cluster into the projector (`bandContour` at a quarter gap,
64 nodes); `deltaF` built a `ReggeSolver` on the shared live spacetime inside the
OpenMP candidate loop (facets materialized once before it); and block sub-complexes
were built at unit lengths, so block fiber residuals were geometry-free
(`blockSubcomplexWithGeometry`).

## Where this leaves the experiment

- Single-node states: reached to machine precision on the draws that grow well; the
  analytic gradient makes each round seconds.
- The two-body map: not yet carried by a bulk grown by random cone-ins between two
  disjoint tetrahedra. The measured obstruction is the coupling rank between the
  frames, so the next lever is the bulk's connectivity: cells adjacent to both frames,
  or frames the specification lets overlap on shared cells through the carried Gram,
  rather than more budget.
- Time as the recursion (#943) waits on a single layer that carries χ.
