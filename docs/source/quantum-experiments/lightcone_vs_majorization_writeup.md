# Lightcone vs. majorization — first hypothesis-test scan

First experimental run of `tessera.quantum.SchwingerQuench.compareCausalOrders`
across a small (m/g, T, N) × vLr grid on a regular 1+1D chain. This
addresses the **strong falsification** criterion of
`docs/source/quantum-methodology.md` §1.2:

> If $\preceq_{\mathrm{maj}}$ contains pairs whose spatial separation
> exceeds the Lieb–Robinson cone at the corresponding time difference,
> the hypothesis is wrong.

The **weak falsification** criterion (≼_maj vs non-trivial ≼_cs) is
out of scope for this run — it requires the causet-embedded MPO
rebuild on a tessera-derived chain. The **trivial-agreement** caveat applies to the
≼_cs column here: on a regular 1+1D chain, ≼_cs reduces to the
time-only order, so ≼_LR ⊆ ≼_cs by construction (verified — `τ(LR, cs)
= 1.0` exactly).

## Setup

- Schwinger model on N = 10 / 14 staggered spin sites, OBC, $a = g = 1$.
- DMRG ground state at the specified $m/g$, then $\sigma^- \sigma^+$
  q-q̄ quench at $i_0 = 3$, separation $d = 3$.
- 2-site TDVP integration with $\Delta t = 0.1$, total $T = 1.0$ or
  $2.0$. Snapshots recorded every 5 steps so the inter-snapshot gap is
  $0.5$ — large enough that the LR cone responds to the vLr sweep.
  (At `snapshotEvery=1`, all vLr ∈ [0.5, 4] give cone bounds < 1
  lattice site for the smallest time gap, so the cone admits the same
  set of pairs across the whole sweep.)
- Cut family: contiguous intervals $[i, j]$, excluding the trivial
  full-chain cut. 54 cuts at N = 10, 104 at N = 14.
- Label set: cut × snapshot. Sizes from 162 (Regime A) to 728 (Regime D).

## Raw scan output

```
Regime A — light quark (m/g=0.5), N=10, T=1.0
   vLr   τ(maj,LR)    discord    edit   τ(maj,cs)   n_comp(maj,LR)   n_comp(LR,cs)
   0.50      0.3938     0.3031   0.962      0.3622             3583            6766
   2.00      0.3896     0.3052   0.956      0.3622             3991            7428
   4.00      0.3779     0.3111   0.957      0.3622             4221            7908
   8.00      0.3686     0.3157   0.959      0.3622             4514            8468
  16.00      0.3621     0.3190   0.960      0.3622             4662            8744

Regime B — heavy quark (m/g=5.0), N=10, T=1.0
   0.50      0.5469     0.2265   0.971      0.5131             4189            6766
   2.00      0.5382     0.2309   0.969      0.5131             4647            7428
   4.00      0.5312     0.2344   0.970      0.5131             4923            7908
   8.00      0.5210     0.2395   0.971      0.5131             5273            8468
  16.00      0.5134     0.2433   0.972      0.5131             5434            8744

Regime C — light quark (m/g=0.5), N=10, T=2.0
   0.50      0.5390     0.2305   0.963      0.5020           13517           25194
   2.00      0.5289     0.2355   0.959      0.5020           14350           26520
   4.00      0.5181     0.2409   0.960      0.5020           14834           27480
   8.00      0.5079     0.2460   0.962      0.5020           15465           28600
  16.00      0.5021     0.2489   0.963      0.5020           15771           29152

Regime D — light quark (m/g=0.5), N=14, T=1.0
   0.50      0.2745     0.3628   0.972      0.2462           11820           25166
   2.00      0.2687     0.3657   0.971      0.2462           12878           26988
   4.00      0.2619     0.3690   0.973      0.2462           13587           28444
   8.00      0.2540     0.3730   0.973      0.2462           14710           30468
  16.00      0.2470     0.3765   0.975      0.2462           15666           32168
```

(Reproducible: `python examples/quantum/lightcone_vs_majorization.py`.
Total runtime ~80 s on a single node. The N-scaling and cone-overflow
follow-up scans live in the same directory under
`lightcone_vs_majorization_n_scaling.py` and
`lightcone_vs_majorization_cone_overflow.py`.)

## Sanity checks

- **`τ(LR, cs) = 1.0` everywhere.** Verified out of band — `≼_LR` is a
  strict subset of `≼_cs` on the regular chain because `≼_cs` is just
  time-only, and any LR-related pair is necessarily cross-time. This
  is the methodology page's criterion-3 caveat in observational form.
- **`n_comp(LR, cs)` grows monotonically in vLr.** As the cone widens
  more pairs become LR-comparable. By the time we hit vLr = 16 with
  effective Δt = 0.5, the cone reaches 8 sites — close to (but not
  yet at) the diameter of the chain, so LR is still a strict subset
  of cs. Saturation would require vLr · T > N.
- **Σλ = 1 invariant** verified separately in
  `test_schmidt_invariants_dmrg.cpp` across 330 spectra — the
  agreement statistics aren't hostage to a Schmidt normalisation drift.

## Findings

### 1. The hypothesis as stated is rejected on the regular chain

`τ(maj, LR)` ranges from 0.25 (Regime D) to 0.55 (Regime B) and never
gets near 1.0. `τ(maj, cs)` is similar. The discordant fraction —
pairs that **both** orders relate but in **opposite directions** — sits
at 0.23–0.38. That's a non-trivial fraction of label pairs where ≼_maj
is *anti-aligned* with the time order, not just unaligned.

This is informative: a result of `τ ≈ 1` would have meant "trivial
agreement" per criterion 3 (since ≼_cs is forced to be time-only here).
A result of `τ ≈ 0` would have meant ≼_maj is uncorrelated with time.
What we see — `0.25 ≤ τ ≤ 0.55` with substantial discordance —
indicates ≼_maj is **partially** time-correlated and partially driven
by something the time order doesn't see.

### 2. `τ(maj, LR)` *decreases* as vLr widens

A wider LR cone sees more pairs as LR-comparable. If the new pairs
were concordant with ≼_maj, τ would rise. They're net *discordant*
instead — τ drifts down by ~0.03–0.05 over the vLr ∈ [0.5, 16] range
in every regime. The interpretation: the within-light-cone subset of
≼_LR aligns *better* with ≼_maj than the wider cone does. ≼_maj has a
near-cone signal (concordance there) plus a counter-cone signal (the
new pairs added at large vLr push τ down).

This inverts a naive expectation. It hints that the entanglement order
sees something like a *stricter* cone than the vLr = 1 free-fermion
bound — the agreement at vLr = 0.5 is the highest in every regime.

### 3. Heavy quark agrees better with time order than light quark

Compare Regime A (m/g = 0.5) vs Regime B (m/g = 5.0) at fixed N, T:
`τ(maj, LR)` jumps from 0.39 to 0.55 at vLr = 0.5. The heavy-quark
flux tube is approximately stable (we tested this in
`test_tdvp_string.cpp`); little entanglement spreading means the
Schmidt spectra evolve slowly in time, so ≼_maj tracks time more
closely. Light quark has string spreading and pair production, which
mixes spectra across cuts and makes ≼_maj less time-respecting.

### 4. Longer evolution doesn't converge τ toward 1

Regime C (T = 2.0) gives a higher τ than Regime A (T = 1.0) — 0.54 vs
0.39 — but the gap is more about the additional snapshots increasing
the relative weight of long-time pairs (which are more LR-comparable)
than about a real agreement-improving trend. Both regimes show the
same vLr → ↓τ direction.

### 5. Larger N decreases τ

Regime D (N = 14) gives τ ≈ 0.25 vs Regime A (N = 10) at 0.39. More
cut labels = more pairs = more opportunity for ≼_maj's non-time
structure to surface. This is the direction that *suggests* a
thermodynamic-limit reading: τ may continue to drop with N. We
haven't run the scan large enough to extrapolate.

## Strong-falsification follow-up — explicit cone-overflow counts

Added `nOnlyA` / `nOnlyB` to `OrderAgreement` (2026-04-25 patch)
to count pairs related by exactly one of the two orders. With
`(a, b) = (≼_maj, ≼_LR)`, `nOnlyA` is the count of
**majorization-related pairs whose endpoints lie OUTSIDE the LR
cone** — the explicit strong-falsification metric.

Re-run with N=10, 14, 20 (m/g=0.5, T=1, snapshotEvery=5):

```
Regime A — N=10, m/g=0.5
   vLr    τ(maj,LR)    n_maj∉LR    n_maj∉LR/|maj|    n_LR∉maj   |≼_maj|
   0.50      0.3938       3263         47.66%           3183       6846
   1.00      0.3938       3263         47.66%           3183       6846
   2.00      0.3896       2855         41.70%           3437       6846
   4.00      0.3779       2625         38.34%           3687       6846
   8.00      0.3686       2332         34.06%           3954       6846
  16.00      0.3621       2184         31.90%           4082       6846

Regime D — N=14, m/g=0.5
   1.00      0.2745      11673         49.69%          13346      23493
  16.00      0.2470       7827         33.32%          16502      23493

Regime E — N=20, m/g=0.5
   1.00      0.2088      44906         51.05%          58728      87959
  16.00      0.1825      31470         35.78%          70550      87959

Regime B — N=10, m/g=5.0  (heavy quark)
   1.00      0.5469       3848         47.88%           2577       8037
  16.00      0.5134       2603         32.39%           3310       8037
```

### What the numbers say

**At `vLr = 1.0` (free-fermion bound, the physically-relevant value
for our hopping coefficient)**: roughly **half** of all majorization-
related label pairs lie outside the Lieb–Robinson cone:

| N  | m/g | n_maj∉LR | |≼_maj| | n_maj∉LR / \|≼_maj\| |
|----|-----|----------|---------|----------------------|
| 10 | 0.5 | 3,263    | 6,846   | 47.7%                |
| 10 | 5.0 | 3,848    | 8,037   | 47.9%                |
| 14 | 0.5 | 11,673   | 23,493  | 49.7%                |
| 20 | 0.5 | 44,906   | 87,959  | 51.1%                |

The fraction is *near 50% across all four regimes* and *grows weakly
with N*. By the strict reading of methodology §1.2 #1, the hypothesis
is decisively rejected on the regular chain.

**At `vLr = 16` (a 16× looser cone, well above any physical
estimate)**: 32–36% of ≼_maj pairs are still outside the cone. So
*even unboundedly fast information transport cannot account for all
of ≼_maj's relations* — about a third of them point sideways or
backward in time. (Recall ≼_maj is a partial order, so an edge
`(A, s) ≼_maj (B, t)` with `s > t` is anti-causal in the lattice
sense.)

`nOnlyB` (pairs in the LR cone but not in ≼_maj) is comparable in
size to `nOnlyA` — ≼_maj is *not* a refinement of ≼_LR; the two
orders are largely orthogonal. They share a substantial concordant
subset, but each has a piece the other doesn't see.

### Caveats on the strong-falsification reading

- **nOnlyA includes same-time pairs.** A label pair `(A, s)` and
  `(B, s)` with `s_a = s_b` is NOT LR-comparable by construction
  (≼_LR is strictly cross-time). If ≼_maj relates them — same-time
  Schmidt-spectrum strict majorization — they show up as `nOnlyA`
  without being "outside the cone" in any geometric sense. To
  separate true cone-overflow from same-time hits we'd need a
  finer-grained breakdown. (Easy to add — split `nOnlyA` by
  whether `t_a == t_b`.)
- **Spectra equivalence classes.** Two cuts with identical Schmidt
  spectra get no ≼_maj cover edge between them, but they CAN both
  be ≼_maj-related to a third less-concentrated spectrum. The
  agreement counts treat the equivalence class as a single node;
  this is consistent with majorization's intended semantics (LOCC
  convertibility is reflexive-symmetric on equal spectra) but it
  means `|≼_maj|` is the count of *strict* majorization relations,
  not all comparable pairs.
- **Single Trotter seed.** All values above are point estimates from
  one DMRG/TDVP trajectory per regime. Bootstrap-over-seeds is the
  proper next step before quoting these numbers in any external
  context.
- **Trotter / bond-dim sensitivity.** Single-seed run, fixed
  bond-dim 80, fixed dt 0.1. The methodology page calls for
  bootstrap-over-Trotter-seeds to put a confidence band on these τ
  values. Until that is done, the τ ranges above are point estimates
  only.
- **Non-trivial ≼_cs.** All ≼_cs results above are time-only. The
  weak-falsification test requires a tessera-embedded chain with
  multi-vertex antichains, where ≼_cs has within-time-slice structure
  that ≼_LR does not.

## Reading toward a causet-embedded re-run

The result here is consistent with two readings of the hypothesis:

**Reading 1 — hypothesis-as-stated is too strong.** ≼_maj genuinely
has more structure than time / LR cone capture, and no causet ≼_cs on
the regular chain will recover the agreement.

**Reading 2 — the regular chain hides the structure.** On a foliated
CDT, ≼_cs gains within-slice structure. If the Schwinger state were
re-evolved on a non-trivial causet and ≼_cs replaced its trivial form
here, the within-slice ≼_maj relations may now have matching ≼_cs
relations — recovering the τ ≈ 1 signal.

Reading 2 is the interesting one because it would *vindicate* the
hypothesis as a non-trivial claim about causet-embedded entanglement
flow. Reading 1 would refute it. Either way, **the causet-embedded
re-run is the test that distinguishes them.**

## Recommended next steps before the causet-embedded build-out

A — Quick (under a day):

- Add `nOnlyA`, `nOnlyB` to `OrderAgreement` so we can directly
  count maj-pairs outside the LR cone.
- Bind `CausalOrders.fromSnapshots` to Python so a follow-up scan can
  dump the actual maj-cover edges that are outside the cone.
- Re-run the scan with a Trotter-bootstrap (5-10 seeds per regime)
  to put error bars on τ.

B — Medium (1-3 days):

- Causet-chain MPO re-run: `SchwingerHamiltonian::mpoChain` already
  accepts a `CausetChain` hopping pattern; for chain causets
  (1 vertex per slice) it reduces to the default MPO. Run a second
  scan with the chain-causet ≼_cs replacing the time-only ≼_cs.
  Compare τ(maj, cs) before and after.

C — Long (1-2 weeks):

- Re-run with a non-trivial CDT spacetime (multi-vertex antichains).
  Either accept long-range hopping in the MPO bond dim or switch to
  ITensor tree TN. This is the proper weak-falsification test.

The most informative next move is A → B → C in that order; A produces
the strong-falsification numbers (the criterion 1 piece) on the data
we already have, B ports the existing apparatus to the chain causet
without a major rebuild, C is the actual hypothesis test.
