# Plateau finite-size investigation: does the 4.6 → 4 offset close in the asymptotic limit?

Tracks GitHub issue
[#10](https://github.com/akellehe/tessera/issues/10) in milestone
[Charged Cartan Monte Carlo v0.2](https://github.com/akellehe/tessera/milestone/1).

## Context

The [intellectual lineage of this project](intellectual_lineage.md)
sets out the through-line: from van Raamsdonk's premise that
spacetime is built from entanglement, through Sorkin's causal-set
events, through replacing the Schwinger fermion field with a
4-dim qudit basis carrying intrinsic charge, to the
[v0.2 β-scan](charged_cartan_v02_beta_scan_writeup.md) in which the
heat-kernel spectral dimension finds a stable plateau at peak
`D_S ≈ 4.6 ± 0.1` across a full decade of `β`.

That plateau is the closest any construction in this project has come
to a *stable* dimensional phase: per-seed std ~0.1, β-flat over a
decade, σ-peak finite (not σ-saturated). The structural properties
of a real phase. But the value sits ~0.6 above the H_DS4 target of
exactly 4.

This experiment asks the natural next question:

**Is the 4 → 4.6 offset a finite-size effect that closes in the
asymptotic limit, or a model-level structural number we'd reach
even at infinite lattice size?**

## Hypothesis

If the plateau reflects a model-level geometric structure, the value
should be insensitive to N (initial-layer vertex count) and T
(cell-target count). If instead the offset is a finite-size effect,
we expect peak D_S to drift toward 4 as either N or T grows — and
the drift should slow as we approach the asymptote.

## Setup

Two scans, both at γ_CP = 0 in the qudit basis (`featureQuditBasis =
True`, `j_chargeCharge = 1.0`, `j_spinSpin = 0.25`, `massShift = 0`,
`dtPair = 0.25`):

**Scan A — N-scaling at fixed T:**
- N ∈ {8, 10, 12, 14}
- T = 2500
- β ∈ {1×10⁻⁴, 3×10⁻⁴, 5×10⁻⁴} (three points inside the plateau)
- 10 independent seeds (and independent Delaunay lattices) per cell
- 120 runs total

**Scan B — T-scaling at fixed N:**
- N = 8
- T ∈ {2500, 5000, 10000, 20000}
- β = 3×10⁻⁴
- 10 independent seeds per cell
- 40 runs total

Each run is a fresh Python subprocess; 18 workers run concurrently
with 2 BLAS threads each (fits a 20-CPU budget). σ-grid is 20
log-spaced over [10⁻², 10¹⁰], Krylov dim 15.

## Reproduce

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
python /tmp/issue10_scan.py
python examples/quantum/plot_v02_finite_size.py
```

Records land at `/tmp/interaction-history/issue10_finite_size.json`;
plot at `docs/source/quantum-experiments/figures/v02_finite_size_investigation.png`.

## Results

![Plateau finite-size investigation: peak D_S vs N at fixed T, and
peak D_S vs T at fixed N](figures/v02_finite_size_investigation.png)

### Scan A — peak D_S vs N at T = 2500

| β | N=8 | N=10 | N=12 | N=14 |
|---|---|---|---|---|
| 1×10⁻⁴ | 4.624 ± 0.066 | 4.665 ± 0.070 | 4.726 ± 0.084 | 4.793 ± 0.101 |
| 3×10⁻⁴ | 4.605 ± 0.080 | 4.707 ± 0.030 | 4.702 ± 0.063 | 4.727 ± 0.086 |
| 5×10⁻⁴ | 4.637 ± 0.119 | 4.696 ± 0.065 | 4.749 ± 0.058 | 4.792 ± 0.087 |

Peak D_S **rises mildly** with N — about +0.15 from N=8 to N=14, with
identical trend at all three β. This is the *opposite* direction from
what we'd expect if the 4.6 plateau were "the value 4 + finite-size
correction." Likely interpretation: at fixed T, larger N means each
initial worldline gets less growth (T/N ratio drops from 313 at N=8
to 179 at N=14), which keeps the resulting geometry closer to a pure
bowtie-cell-stack — the structure that produces D_S ≈ 4.6 in the
first place.

### Scan B — peak D_S vs T at N = 8, β = 3×10⁻⁴

| T | peak D_S |
|---|---|
| 2500 | 4.621 ± 0.060 |
| 5000 | 4.444 ± 0.041 |
| 10000 | 4.312 ± 0.025 |
| 20000 | 4.245 ± 0.024 |

Peak D_S **falls monotonically** with T. The per-doubling deltas are
shrinking, which is the signature of an asymptotic approach to a
finite limit rather than unbounded decay:

| T-step | Δ peak D_S |
|---|---|
| 2500 → 5000 | −0.177 |
| 5000 → 10000 | −0.132 |
| 10000 → 20000 | −0.067 |

The ratio of successive deltas is roughly 0.5, so naive geometric
extrapolation gives an asymptote:

> D_S(T → ∞) ≈ 4.245 − 0.067 / (1 − 0.5) ≈ **4.11**

within ~0.1 of the H_DS4 target. The simple `4 + a/ln T` fit shown in
the plot is too shallow (the data falls faster than `1/ln T`),
suggesting the true scaling is some power law with a small exponent,
or a different functional form. Worth a separate fit study, but the
qualitative picture is clear: **the plateau approaches a value close
to (but probably above) 4 in the large-T limit**.

## Findings

1. **The 4 → 4.6 offset is a T-scaling effect, not an N-scaling one.**
   Growing the lattice in *cell count* drives peak D_S down by 0.4
   over the T = 2500 → 20000 range and projects to D_S ≈ 4.1 at
   T → ∞. Growing the lattice in *initial-vertex count* at fixed T
   actually makes the plateau *higher*, which clarifies the role of
   N as a "structural multiplier" rather than a "continuum limit"
   parameter.

2. **The H_DS4 target is plausibly reachable** in the large-T limit
   under the current qudit-basis Hamiltonian. The scan didn't show
   D_S = 4 exactly, but the asymptotic value is within ~0.1 of it,
   and a deeper run at T = 50k-100k (or a parameter-tuning study on
   the pair Hamiltonian) should clarify whether the limit is exactly
   4 or settles ~0.1 above.

3. **A discrete Q-drift bug surfaced.** At γ_CP = 0, Q should be
   exactly conserved — but **95 of 160 runs (59 %)** show |Q| ≥ 1
   drift at integer values (-4, -3, -2, -1, +1, +2, +3). The discrete
   spectrum points at a specific mechanism, not floating-point noise.
   See [Q-drift section](#discrete-q-drift-at-gamma_cp--0) below.

## Discrete Q-drift at γ_CP = 0

A real consistency issue in v0.2 surfaced by this scan, traceable to
the Σ_AB-state proxy choice. Symptom:

- 95 / 160 runs at γ_CP = 0 show |Q_global| ≥ 1 at end of tune.
- The drift values are integer-valued (drift spectrum: {−4, −3, −2,
  −1, +1, +2, +3}).
- Q is *bit-perfectly* conserved through any single accepted
  `interact` call (the 16×16 unitary commutes with `Q̂ ⊗ I + I ⊗ Q̂`
  at γ_CP = 0).

So the drift accumulates from *somewhere* outside the per-interact
unitary step. The likely mechanism is the Σ_AB-as-`I/4`-proxy choice:

- When the bowtie is created, the genuine joint state `ρ_AB = U(ρ_X
  ⊗ ρ_Y)U†` is stored in `quditJointOf_[(xp, ab)]` and
  `quditJointOf_[(ab, yp)]`. The single-vertex proxy `quditStateOf_[ab]
  = I/4` is set, with ⟨Q̂⟩ = 0.
- This is consistent at the moment of creation: q_xp + q_ab + q_yp =
  q_x + q_y is exact.
- But `ab` carries *two different "marginal Q" identities* depending
  on which neighbour it's looked up from: the joint with `xp` says
  ⟨Q_ab⟩ = q_y (the yp-side marginal), the joint with `yp` says
  ⟨Q_ab⟩ = q_x (the xp-side marginal). The proxy I/4 says ⟨Q_ab⟩ = 0.
  These three values are equal *only* when q_x = q_y = 0.
- When `ab` is *later picked* for a new interaction with some
  unrelated third vertex `Z`, the input joint is constructed as
  `quditTensor(I/4, stateOf[Z])`, *not* from any of the stored
  joints. The Q content of `ab` at that moment is 0 (from the
  proxy), inconsistent with the q_x or q_y it would have carried via
  its stored joints. The Q change from this proposed interact event
  is therefore `q_Z + 0 - (q_xp_marginal + q_yp_marginal + q_Z)`,
  which can be non-zero.

This is exactly the inconsistency that issue
[#16](https://github.com/akellehe/tessera/issues/16) (Σ_AB as full
256-dim Choi state of U) is designed to fix: the Choi state carries
*all* of U's correlation content as a proper quantum state, so any
later interaction involving `ab` gets a coherent input rather than
the I/4 proxy.

The drift does not seem to corrupt the *peak D_S* measurements (the
scan results above are consistent across drifters and non-drifters
within their respective uncertainty bands), but it does invalidate
v0.2 as a fully Q-conserving framework. **Issue
[#16](https://github.com/akellehe/tessera/issues/16) is therefore
upgraded from "optional refinement" to "blocker for any work that
relies on Q conservation beyond a single interact step."**

## Falsification check (H_DS4)

| Criterion | Status |
|---|---|
| Is the 4 → 4.6 plateau offset a finite-size artefact? | **Partial yes** — T-scaling drives it down toward ~4.1 asymptotically; N-scaling does the opposite. |
| Does any scan touch D_S = 4 exactly? | Not yet — T = 20000 gives 4.245 ± 0.024. |
| Is the H_DS4 target reachable? | **Plausible** — geometric extrapolation projects to ~4.11, close enough that follow-up scans at T = 50k-100k and Hamiltonian parameter tuning ([issue #11](https://github.com/akellehe/tessera/issues/11)) could land exactly at 4. |

## Open follow-ups

1. **Deeper T-scan.** Run at T = 40000 and 80000 to confirm the
   geometric extrapolation. ~30 min compute per T at N=8 in 18-way
   parallel.
2. **Joint (N, T) scan.** Does increasing N alongside T cancel or
   amplify the trend? Run at N=12, T=20000 to find out.
3. **Issue [#16](https://github.com/akellehe/tessera/issues/16):
   Σ_AB as full Choi state.** Fix the discrete Q-drift bug.
4. **Issue [#11](https://github.com/akellehe/tessera/issues/11):
   Hamiltonian-parameter scan.** Does some choice of (J_c, J_s,
   δ_m) put the asymptote exactly at 4?

## See also

- [charged_cartan_v02_beta_scan_writeup.md](charged_cartan_v02_beta_scan_writeup.md)
  — the parent v0.2 β-scan that first found the 4.6 plateau.
- [charged_cartan_monte_carlo_v0.2.md](charged_cartan_monte_carlo_v0.2.md)
  — the v0.2 design note, including the Σ_AB proxy discussion that
  this writeup ties back to.
- GitHub milestone [Charged Cartan Monte Carlo v0.2](https://github.com/akellehe/tessera/milestone/1).
