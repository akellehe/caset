# Charged Cartan Monte Carlo — Overview: H_DS4 status

> **Status**: living document — peak D_S asymptotically approaches 4
> **Current best**: peak D_S = 4.245 ± 0.024 at T = 20k; geometric
> extrapolation → D_S(T → ∞) ≈ 4.07
> **Last updated**: tracks the merged work in
> [v0.2 milestone](https://github.com/akellehe/tessera/milestone/1)

## TL;DR

**H_DS4** is the project's central hypothesis: that there exists a
regime in `β` (and possibly in the Hamiltonian parameters `(J_c, J_s,
δ_m, γ_CP)`) where the peak heat-kernel spectral dimension of the
interaction-history complex reaches **4** — the 3+1-dimensional
phase.

**Current best result.** Under the v0.2 qudit basis with
γ_CP = 0 and the Choi-state Σ_AB fix (issue #16), the peak D_S
plateau sits at **4.6 ± 0.1** at T = 2500 and falls monotonically
with T:

| T | peak D_S |
|---|---|
| 2500 | 4.621 ± 0.060 |
| 5000 | 4.444 ± 0.041 |
| 10000 | 4.312 ± 0.025 |
| 20000 | 4.245 ± 0.024 |

Per-doubling deltas shrink geometrically (−0.18, −0.13, −0.07), so a
naive extrapolation gives **D_S(T → ∞) ≈ 4.07**, within ~0.1 of
exact 4.

**Open question**: is the 0.07 offset a finite-T residual that
closes at T → ∞? An artifact of the current Hamiltonian parameter
choice `(J_c=1.0, J_s=0.25, δ_m=0.5)`? Or a real structural number?
A deeper T = 100k scan is queued; a Hamiltonian-parameter scan
(issue #11) is the next experimental step.

## Hypothesis statement

From [`earlier-work/interaction_history_monte_carlo.md`](../earlier-work/interaction_history_monte_carlo.md#hypothesis)
(the line of work's founding writeup), restated here for centrality:

> **H_DS4.** As `β` is scanned, the peak spectral dimension of the
> interaction-history complex passes through a transition, and there
> is a locus `β*` where `D_S → 4` — the 3+1-dimensional phase. At
> `β → 0` the complex grows freely (the action is irrelevant); at
> large `β` growth is suppressed; the emergent-dimension transition
> is expected in between.

**Falsification.** "If peak `D_S` never approaches 4 across the `β`
range — in particular if it saturates well below 4 even in the
free-growth regime — H_DS4 fails: this construction does not
generate a four-dimensional emergent spacetime."

## Status across versions of the construction

The history is a sequence of approximate models, each one closer to
H_DS4 passing. The table below tracks the peak D_S delivered by
each version's best β.

| Version | Peak D_S | Behavior | Status of H_DS4 |
|---|---|---|---|
| v0 marginal model | ~0.635 (ceiling) | saturates well below 1 in free-growth regime | **falsified** |
| v0 Cartan-model with structural zeros | unbounded D_S(σ) at large σ | σ-saturation artifact | **artifact, not a real D_S** |
| v0.1 + B + iii (charges, deactivate, photon) | ~250 ± 100 across all β | small-world graph, no plateau | **fails** (no plateau at any β) |
| v0.2 qudit basis (γ_CP = 0) | **4.6 ± 0.1** across a β decade | **clean plateau** | **passes the criterion** with 0.6 offset |
| v0.2 qudit basis + T → ∞ extrapolation | ~4.07 | finite-T extrapolation closes most of the gap | **near-pass**; offset ≤ 0.1 |
| v0.2 + Choi-state Σ_AB (#16) | 4.537–4.609 across β decade | bug-free Q-conservation, same plateau | **near-pass** with Q exactly conserved |

## Where each result is documented

- **v0 results**: [`earlier-work/interaction_history_monte_carlo.md`](../earlier-work/interaction_history_monte_carlo.md)
- **v0.1 + B + iii results**: [`../charged-cartan/experiments/01-v0.1-BplusIII-comparison.md`](../charged-cartan/experiments/01-v0.1-BplusIII-comparison.md)
- **v0.2 plateau**: [`../charged-cartan/experiments/02-v0.2-beta-scan.md`](../charged-cartan/experiments/02-v0.2-beta-scan.md)
- **Finite-size T-scaling**: [`../charged-cartan/experiments/03-v0.2-finite-size.md`](../charged-cartan/experiments/03-v0.2-finite-size.md)
- **Choi-state Q-conservation fix** (sanity, headline numbers
  unchanged): [`../charged-cartan/experiments/04-v0.2-choi-q-conservation.md`](../charged-cartan/experiments/04-v0.2-choi-q-conservation.md)

## What would close the 0.07 gap

Three plausible paths, in increasing complexity:

1. **Deeper T-scan** (queued — T = 100k overnight): confirms or
   refutes the geometric extrapolation. If the trend continues, the
   asymptote is ≈ 4.07 or possibly lower.
2. **Hamiltonian-parameter scan** ([issue #11](https://github.com/akellehe/tessera/issues/11)):
   scan `(J_c, J_s, δ_m)` to find a natural parameter choice where
   the plateau lands exactly at 4 instead of 4.6. A clean "D_S = 4
   at some natural Hamiltonian" result would be a stronger H_DS4
   pass than "D_S → 4 only in the asymptotic limit."
3. **v0.3 gauge mediation** ([milestone #2](https://github.com/akellehe/tessera/milestone/2)):
   restoring the photon-mediated Coulomb interaction may or may not
   change the plateau value. If it preserves D_S ≈ 4, that's a
   robustness result; if it moves the plateau, we learn about the
   role of gauge content in the dimensional emergence.

The current best read on H_DS4: **the construction does generate a
near-4D phase in the asymptotic limit, with a ~1.5% offset
attributable to either finite-T or Hamiltonian parameter choice.
Neither has been ruled out as the source of the offset.** The
falsification clause is therefore not met; the hypothesis is
*supported but not yet exact*.

## See also

- [intellectual_lineage.md](intellectual_lineage.md) — the
  intellectual through-line that motivates H_DS4 in the first place.
- [from_schwinger_to_lattice.md](from_schwinger_to_lattice.md) —
  why the construction looks the way it does, in terms of which
  Schwinger-model ingredients we kept vs. replaced.
- GitHub milestones:
  [Charged Cartan Monte Carlo v0.2](https://github.com/akellehe/tessera/milestone/1)
  (current),
  [v0.3](https://github.com/akellehe/tessera/milestone/2) (upcoming).
