# Charged Cartan v0.1 — annihilation as worldline termination + photon emission

Comparison experiment under the
[Charged Cartan Monte Carlo v0.1 design](charged_cartan_monte_carlo_v0.1.md).
Toggles the two annihilation-semantics feature flags introduced after
the first β-scan revealed the v0.1 baseline collapses all phase
structure into a single `D_S ≈ 250` regime:

- `featureDeactivateOnAnnihilate` — annihilation removes the matched
  worldlines from the frontier (they terminate) instead of just
  neutralising their charges. The vertices stay in the spacetime so
  cell references remain valid; they're "inactive" — can't be picked
  for future moves.
- `featurePhotonOnAnnihilate` — each annihilation event additionally
  spawns a new neutral "photon" vertex on the frontier, carrying the
  released information (state = `I/2`, charge = 0).

The motivation is twofold. First, structural: the v0.1 baseline's
charge-only neutralisation lets annihilated worldlines continue
accruing proper-time on subsequent interactions, which is unphysical
under the "worldline = information momentum" framing — terminated
particles should *stop* propagating. Second, semantic: in QED an
electron-positron annihilation produces photons that carry off the
released energy/momentum. The combined flags model both aspects.

## Configuration

| | Value |
|---|---|
| Vertices `N` | 8 |
| Initial-layer charges | `ALTERNATING` (`q = ±1` by index parity) |
| Cell target `T` | 3000 |
| β values | 22 log-spaced over [10⁻⁴, 5×10⁻³] |
| Seeds per β | 10 |
| Krylov | 15 |
| σ-grid | 20 log-spaced over [10⁻², 10¹⁰] |
| `cpBias` | 0 |
| Equilibration | 200 rounds of (annihilate, pairCreate, interact) after `tune()` |

The equilibration step exists because `tune()` only calls `interact()`
— the new flags never fire under tune-only measurement. Running a
mixed-move loop after tune lets the charge dynamics actually affect
the resulting geometry.

The two flag configurations are run on **identical seeds and Delaunay
edges** so each per-β cluster is a direct paired comparison.

## Reproduce

```bash
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \
python /tmp/v01_compare_BplusIII.py
python examples/quantum/plot_v01_BplusIII_comparison.py
```

Records land at
`/tmp/interaction-history/v01_compare_BplusIII.json`; the plot is
written to
`docs/source/quantum-experiments/figures/v01_BplusIII_comparison.png`.

## Results

![Baseline vs (B + iii) comparison: peak D_S, σ_peak, vertex / cell /
frontier counts, and charge composition](figures/v01_BplusIII_comparison.png)

<!-- TODO: paste in the per-β summary table from the plotter's console
     output once the scan finishes.  Format:

     |    β    | baseline mean D_S ± std | B+iii mean D_S ± std | ΔV |
     |---------|------------------------|----------------------|----|
     | 1.0e-4  | x.xx ± y.yy            | x.xx ± y.yy          | +N |
     | ...     |                        |                      |    |
-->

## Discussion

<!-- To be filled in after data lands. Expected structural points to
     check / confirm / refute:

  1. Does B+iii change the flat `D_S ≈ 250` baseline regime?
     Hypothesis: yes — by removing annihilated worldlines from the
     frontier, the eligible-pair pool shrinks; by spawning photons,
     a population of neutral vertices accumulates. Both effects
     change the per-β graph topology.
  2. Does the σ at which D_S peaks shift?
     Hypothesis: photons add isolated-ish neutral vertices that
     contribute slow eigenmodes; D_S(σ) might develop a longer
     tail (peak at larger σ).
  3. Vertex count: B+iii should grow the vertex count by
     ≈ N_annihilations across the run (one photon per event).
  4. Charge composition: B+iii should show fewer ± vertices and
     more neutrals than baseline at the same β.
  5. Does any β value approach D_S = 4?
     (a priori, no — the structural argument hasn't changed; only
     the charge dynamics differ.)
-->

## Falsification check (H_DS4)

| Criterion | Expectation | Observed | Status |
|---|---|---|---|
| peak `D_S` approaches 4 somewhere in β | yes for H_DS4 | <!-- TBD --> | <!-- TBD --> |
| B + iii alters the flat-D_S regime found in v0.1 baseline | working hypothesis | <!-- TBD --> | <!-- TBD --> |
| Photon emission grows vertex count by ≈ annihilation count | yes (mechanical) | <!-- TBD --> | <!-- TBD --> |

## Known limitations carried over from v0.1

- `featureDeactivateOnAnnihilate` does **not** fix the Q-drift under
  `unInteract` that follows annihilation. See the parent design note
  for the explanation; the fix requires either treating annihilation
  events as first-class objects in the un-interact cascade BFS or
  moving to v0.2's intrinsic qudit-charge representation.
- The equilibration loop is only 200 rounds — much shorter than what
  a thermalised sample would need. Long-equilibration scans are a
  follow-up.

## See also

- [charged_cartan_monte_carlo_v0.1.md](charged_cartan_monte_carlo_v0.1.md)
  — design note for the v0.1 model and its feature-flag convention.
- [interaction_history_monte_carlo_writeup.md](interaction_history_monte_carlo_writeup.md)
  — the chargeless v2 scan this whole line of work compares against.
