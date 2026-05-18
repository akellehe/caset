# Charged Cartan Monte Carlo — design note v0.3 (gauge mediation)

Living design note for the v0.3 addition to the construction described
in [charged_cartan_monte_carlo_v0.1.md](charged_cartan_monte_carlo_v0.1.md).
v0.3 gives the photon vertices spawned by
`featurePhotonOnAnnihilate` a *physical role*: they mediate a
Coulomb-like interaction between charged vertices, implemented as
**input-state conditioning at interaction time**.

## Motivating principle: "gravity is matter"

The construction's whole premise — edge lengths derived from mutual
information, Regge action integrated over MI-weighted hinges — is
that geometry *is* the matter content. There is no separate matter
field on top of the metric; the metric is built from the quantum
state.

Under this view, a separate `S_matter = ⟨H⟩·dt` action term would
double-count the matter content (it's already in the lengths and
therefore already in `S_Regge`). What's *actually* missing isn't
another action — it's that **forces between charges don't propagate**
in v0.1. Two distant charges sit there in the frontier with no
awareness of each other; the photons spawned by annihilation are
inert bookkeeping vertices.

The Coulomb force in the Schwinger model is exactly this missing
piece: charges at sites `n`, `m` feel each other through the gauge
field even when they're not directly interacting. v0.3 restores this
by letting the photons mediate a correlation amplitude that conditions
the next interaction's input state.

## The mechanism

When the Monte Carlo picks `(i, j)` for interaction, the *input*
joint state `ρ_ij` is no longer `stateOf_[i] ⊗ stateOf_[j]` (or the
`jointOf_` entry if one exists). It's first **conditioned on the
gauge environment** — specifically on the photons reachable from both
`i` and `j` along graph paths of bounded depth — and only then fed
to `U`.

Procedure for each interact attempt (when `featureGaugeMediation` is
on):

1. **Enumerate photon paths from `i` to `j`** via breadth-first
   search bounded by `cfg.gaugeDepth`. A "photon path" is a sequence
   of vertices `i = v_0, v_1, …, v_k = j` such that the chain alternates
   between charged vertices and photon vertices, or is composed
   entirely of photons in the middle. Edges are the usual graph edges
   of the MI-weighted complex.

2. **Compute the Coulomb-like correlation amplitude**:
   ```
   C_ij = α · q_i · q_j · Σ_{paths p of length ℓ ≤ gaugeDepth}
                                  exp(-d_path(p))
   ```
   - `α = cfg.gaugeCouplingAlpha` is the dimensionless coupling.
   - `q_i, q_j` are the charges (in [−1, +1] from v0.1's `chargeOf_`,
     intrinsic from `Q̂` once v0.2 lands).
   - `d_path(p) = Σ ℓ_edge` is the graph-distance along path `p` in
     MI-weighted units.
   - Sign convention: `q_i · q_j > 0` (same-sign) gives a repulsive
     amplitude that *suppresses* the input MI of the proposed cell;
     `q_i · q_j < 0` (opposite) gives an attractive amplitude that
     *enhances* it.

3. **Mix the input joint state with a Coulomb-correlated state**:
   ```
   ρ_ij_input = (1 − |C_ij|) · ρ_ij_local + |C_ij| · ρ_ij_gauge
   ```
   where:
   - `ρ_ij_local` is the v0.1 input — the stored `jointOf_` entry if
     one exists, else `stateOf_[i] ⊗ stateOf_[j]`.
   - `ρ_ij_gauge` is the canonical gauge-correlated state for the
     sign of `C_ij`:
     - **C_ij > 0 (attractive, opposite charges)**: a Bell-like state
       `|Φ+⟩⟨Φ+|` (or its qudit-basis analog in v0.2).
     - **C_ij < 0 (repulsive, same charges)**: the
       *charge-aligned-separable* state — a classical mixture
       `½(|00⟩⟨00| + |11⟩⟨11|)` (or the v0.2 qudit analog) that has
       low MI but strict charge correlation.

4. **Apply U to the conditioned input**, then proceed with the v0.1
   bowtie construction (compute 10 edge MIs, propose cell, Metropolis
   accept).

5. **Propagate the gauge content into the outputs**. The bowtie's
   products `U_A, V_B, Σ_AB` carry forward `ρ_ij_input` (not
   `ρ_ij_local`) as the seed for their `jointOf_` entries — so the
   gauge dressing persists into the next generation.

## Why this matches "gravity is matter"

There is *no* separate matter action term. The total action stays:

```
S_total = β · S_Regge[MI-weighted lengths]
```

But the lengths fed to Regge are now richer:

- Direct MI from local quantum content (as before),
- Plus photon-mediated correlation (Coulomb-like, sign-sensitive),
- Plus whatever new MI structure the interaction produces given the
  conditioned input.

Gravity does all the work. The matter content is just more honest
about the gauge environment it lives in.

## What gauge mediation actually does to the geometry

Two predicted structural effects:

**(a) Charge clustering on the lattice.** Opposite-charge pairs feel
an effective MI attraction through any photon path between them.
Their candidate-cell input states are pulled toward Bell-correlation,
which gives them shorter MI-derived edges in the resulting bowtie,
which gives lower Regge cost for the cell, which gives higher
acceptance probability. Same-sign pairs experience the reverse and
form cells less often. The net effect is that *cells form
preferentially between opposite-sign charges that share a photon
path*. Spatial clustering of matter and antimatter, mediated by
photon density.

**(b) Photons become information conduits.** A photon vertex in the
middle of a long chain of cells extends gauge mediation to every
opposite-charge pair that can reach both ends. Photon-dense regions
of the lattice support more gauge-mediated cell formation; photon-
sparse regions revert to the v0.1 behaviour. This is the lattice
analog of "gauge field strength determines effective coupling."

## Configuration

```python
cfg.featureGaugeMediation  = True   # turn on v0.3 dynamics
cfg.gaugeDepth             = 4      # BFS depth bound for photon-path
                                    # enumeration (0 = no mediation,
                                    # 1 = nearest-photon only, …)
cfg.gaugeCouplingAlpha     = 0.1    # dimensionless coupling α
```

Defaults must keep v0.1 behaviour intact: `featureGaugeMediation =
false`, alpha = 0.

## Performance and complexity

**Per-interact cost.** BFS from `i` to `j` bounded by `gaugeDepth`
is `O(avg_degree^gaugeDepth)`. At our scale (frontier ~thousands,
average degree ~10), this is dominated by the depth bound:

| gaugeDepth | nodes visited per interact | extra time per interact |
|------------|----------------------------|-------------------------|
| 0 (off) | 0 | 0 |
| 1 | ~10 | ~µs |
| 2 | ~100 | ~10 µs |
| 3 | ~1000 | ~100 µs |
| 4 | ~10⁴ | ~ms |
| 5 | ~10⁵ | ~10ms |

We'll start with `gaugeDepth = 2 or 3` for diagnostic runs;
higher only if results suggest it matters.

**Path enumeration policy.** Three implementations possible, in
order of fidelity / cost:

1. **Nearest-photon only.** Walk BFS from `i`; first photon hit is
   the "carrier." Coulomb amplitude weighted only by `i → photon`
   distance, ignoring `photon → j` arm. Cheapest. Captures qualitative
   Coulomb behaviour.
2. **Single-photon mediation.** BFS-find shortest `i → photon → j`
   path; sum amplitude over all such single-photon paths up to
   gaugeDepth. Mid-cost. Captures one-photon-exchange (tree-level
   QED) physics.
3. **Multi-photon paths.** All paths up to gaugeDepth that pass
   through any number of photon vertices. Most expensive. Captures
   higher-order gauge contributions.

We'll implement (2) — single-photon mediation — as the default. It's
the right approximation of Coulomb at the lattice scale and avoids
the combinatorial blowup of (3).

## Generalisation to non-abelian gauge

The same scheme generalises to SU(N) gauge groups: each charge
carries a vector in the gauge-group representation, the correlation
amplitude is a group-theoretic invariant (e.g., `Tr(R_i R_j^†)` for
SU(2)), and the mediating gauge vertex carries the analogous
representation index. Deferred; U(1)/Coulomb is the right thing to
implement first.

## Tests (to be written alongside implementation)

- `cfg.featureGaugeMediation = false` is bit-identical to v0.1 +
  B + iii — backward compat.
- `gaugeCouplingAlpha = 0` is bit-identical to v0.1 + B + iii —
  off-by-coupling check.
- `q_i · q_j > 0` gives `C_ij ≤ 0` (repulsive), `q_i · q_j < 0` gives
  `C_ij ≥ 0` (attractive) — sign convention check.
- A controlled scenario: pair (i, +) and (j, −) at known graph
  distance through one photon; verify the input ρ_ij_input is
  Bell-dressed with the predicted amplitude.
- Q is still conserved by `interact` under gauge mediation (the
  mediation only conditions inputs; it doesn't add/remove charge).

## Comparison experiments (to run after implementation)

1. **Toggle scan**: v0.1 + B + iii baseline vs same + gauge mediation
   on, same β-grid. Look for D_S changes and charge-clustering
   changes via `getChargeCorrelation`.
2. **Depth scan**: fix β to a few values, scan `gaugeDepth ∈ {0, 1,
   2, 3, 4}` to see how reachable photons matter.
3. **Coupling scan**: fix β, scan `gaugeCouplingAlpha ∈ {0, 0.01,
   0.1, 1.0}` to find the regime where mediation visibly changes
   geometry.

## See also

- [charged_cartan_monte_carlo_v0.1.md](charged_cartan_monte_carlo_v0.1.md)
  — the v0.1 design with the photon-emission flag this builds on.
- [from_schwinger_to_lattice.md](from_schwinger_to_lattice.md) — the
  justification for why we dropped the Schwinger framing in favour of
  abstract Cartan dynamics. v0.3 restores the gauge content the
  Schwinger model would have provided, but as a *geometric*
  mechanism rather than a separate action.
