# Analytic matter-energy gradient (Part B) — implementation plan

Status: **Part A landed; the objective evolved past this doc.** What lands here is
**Part A** — the exact dual-action gradient (`ReggeSolver::actionGradientExact`,
`Simplex::lorentzianDeficitAngleGradient`, `Simplex::dualVolumeGradient`), native C++,
FD-verified, with regression + complex-action guards. The objective sketched below
(`F_β = r_U + β|S|`) was a checkpoint and has since been superseded, in order:
`r_U + β|S|` collapses to the action's zero set (`|S|→0`), so we moved to **stationary
action** `‖∇S‖² + Γ·r_U` (full complex `δS=0`, *not* `Re`-only), and then found the
genuine saddle needs **complex edge lengths** (timelike ⟺ `|Im(l)| > ε`, `ℓ² = l²`),
which land in a follow-up PR. The Part-B/`∇G` material below is kept as the historical
design trail. Original framing:

This document plans **Part B**: the exact analytic
gradient of the matter energy `E`, the last piece of

```
∇G = ∂(Re S) + κ ∂E + λ sign(Im S) ∂(Im S)
```

needed for the per-edge backreaction relaxation (#313) to converge without the
1764×-per-step finite-difference factor.

## Resolved design (build this)

After review, the relaxation objective is **not** the #312 backreaction
`Re S + κE + λ|Im S|`. It is the **mediation** free energy

```
F_β(W) = r_U(W) + β · |S_Regge(W)|
```

— the eigenvector **realizability residual** `r_U` (`residualForPeriods`), mediated by
the **full** dual Regge action's **magnitude** `|S| = |Re S + i·Im S|`, weighted by `β`.
Decisions behind this:

- **No Dirichlet source.** Matter enters *only* as `r_U`: the geometry must carry the
  register (the charge). Curvature near the charge is **emergent** — it must fall out of
  minimizing `F_β`, not be injected by a `⟨ψ, w₁ ψ⟩` energy. **`E` is never computed.**
- **The action mediates the runaway.** `|S|` *grows* with the conformal runaway
  (`|S|: 1658 → 2062` over the level-2 scan), so `β|S|` opposes it — there is no
  `Re S → −∞` for the optimizer to fall into. The old `Re S` *and* `λ|Im S|` both
  *decreased* with the runaway; only `κE` opposed it, so the action was **not** the
  mediator. The magnitude is. This is what "mediated by the Regge action" should mean.
- **`r_U` is basis-invariant** — built on the self-consistent `carriedRepresentative`
  (metric harmonics throughout), so §4's basis question is moot.

### Gradient
`∇F = ∂r_U + β·∂|S|`, with
- `∂|S| = (Re S · ∂Re S + Im S · ∂Im S) / |S|` — from Part A (`actionGradientExact`,
  already native + FD-verified).
- `∂r_U` — the residual derivative. `r_U = ‖(M − λI)p‖²`, `p = ψ/‖ψ‖`,
  `λ = pᵀMp` (real, M=metric L₁). Because `ρ := (M−λI)p ⟂ p`, the `∂λ` term drops:
  ```
  ∂r_U = 2·Re[ ρ†(∂M)p ] + (2/n)·Re[ ρ†(M−λI)∂ψ ] − (2 r_U/n)·Re[ p†∂ψ ]
  ```
  This **reuses** the Part B machinery below — `∂M` and `∂ψ` (via the basis-invariant
  `∂Π`) — but the assembled scalar is `∂r_U`, *not* `∂E`.

### Guards — "no Dirichlet source" is a checked invariant, not a hope
1. **Fresh module, no `E` code.** `energy()`/`grad_E` do not exist in the relaxation;
   the `∂Π/∂M/∂ψ` sub-machinery is reused, the final quantity is `∂r_U`.
2. **Single-source matter term** — only `residualForPeriods`; never a `w₁`-weighted norm
   of `ψ`.
3. **Definitional assertion**, every objective eval:
   `assert |F − (residualForPeriods(circles,target) + β·|dualReggeAction()|)| < 1e-12`.
   Any stray term (a `κE`, a `λ|Im S|`, anything) trips it on the spot.
4. **FD-verify `∇F` against FD of that exact `F`.** A stray `∂E` in the gradient fails it.

§1–§8 below document the original `∂E` derivation. Its sub-machinery (`∂Π`, `∂M`, `∂ψ`)
is what `∂r_U` reuses; `E` itself is not part of the objective.

---

## 1. What the energy is, exactly

From `backreaction_emergent_dual.EmergentDual.energy` (the #312 energy the
relaxation uses):

```
w1     = HodgeLaplacian(st).weights(1)                    # per-edge |volume| = sqrt(|l^2|)
P      = es.cyclePeriods(circles).reshape(dim, NC)        # dim=2, NC=9
c, *_  = lstsq(P.T, target)                               # target = P_ref[0], fixed
h      = c @ H                                            # H = harmonicMatrix(1,1e-9, metric=False)
E      = Re <h, w1 h>  =  Re sum_j w1[j] |h[j]|^2
```

Reading the C++ (`EigenstateSynthesis::assembleRegisterReadout`,
`cyclePeriods`, `carriedRepresentative`) pins down the pieces:

- **`cyclePeriods` are the periods of the _metric_ harmonics.** Line 832:
  `out.H = harmonicMatrix(k, 1e-9, metric=true)`. The period matrix is a **fixed,
  topological linear functional** of those harmonics:
  `P[r,q] = Σ_facets ±H_metric[r, facet]  =  (H_metric · Qᵀ)[r,q]`,
  where `Q` (m×n) is the circle→edge incidence with induced-orientation signs —
  **metric-independent**. So `P = H_metric Qᵀ`, and the only metric-dependent
  object in `P` is `H_metric`.
- **`weights(1)[e] = sqrt(|l^2_e|)`** (the edge length). Diagonal; verified
  `∂w1[e]/∂l^2_e = sign(l^2_e)/(2 sqrt(|l^2_e|))` against FD (machine precision).
  This is the easy half of `∂E`.
- **`target = P_ref[0]`** — the first metric harmonic's periods at the reference
  geometry, held fixed (the matter state / the carried "charge").

## 2. The derivation chain

```
∂E/∂l^2_e = Re[ <h, (∂w1) h>  +  2 <w1 h, ∂h> ]
∂h        = (∂c) @ H                       # H fixed (whichever H — see §4)
∂c        = ∂( pinv(Pᵀ) target )           # target fixed -> via ∂P
∂P        = (∂H_metric) Qᵀ                 # Q fixed
∂H_metric = eigenvector perturbation of the metric L1 null space
```

So the whole thing reduces to one new object: **`∂H_metric/∂l^2_e`**, the
derivative of the metric-harmonic basis. One eigendecomposition of the metric
`L1` gives the full spectrum `(λ_a, u_a)`; then standard first-order perturbation
theory gives the derivative — **with one crucial subtlety (§3).**

## 3. The crux: the metric null space is exactly degenerate

`ker L1` is **exactly degenerate** (λ = 0 for both harmonics — it's topological,
`b1 = 2`). First-order eigenvector PT,

```
∂u_i = Σ_{a: λ_a ≠ 0} [ u_aᵀ (∂M) u_i / (0 − λ_a) ] u_a   +   (within-null rotation Ω u_i)
```

splits into (i) the **cross terms** to the non-null modes — well defined, the
subspace "tilt", divides only by nonzero `λ_a`; and (ii) a **within-null rotation**
`Ω` (antisymmetric, mixing `u_1, u_2`). Because the kernel is *preserved* (it does
not split), **first-order PT does NOT determine `Ω`** — it is whatever the
eigensolver happens to return for a degenerate eigenspace.

Whether `∂E` depends on `Ω` is exactly the §4 question.

## 4. **Decision for review:** which `H` carries the representative

`h = c @ H`. There are two choices of `H`, and they differ in their gradient
behaviour:

| | `h = c @ H` uses | period fit `c` uses | self-consistent? | `∂E` clean? |
|---|---|---|---|---|
| **(A) #312 Python energy** | `H_comb` (metric=False) | `P = H_metric Qᵀ` | **no** (mixes two bases) | **no** |
| **(B) self-consistent** (C++ `carriedRepresentative`, line 937) | `H_metric` | `P = H_metric Qᵀ` | **yes** | **yes** |

**Why (A) is basis-dependent.** Under a within-null rotation `H_metric → R H_metric`
(R orthogonal): `P → R P`, so `c → R c`. Then
`h = (Rc) @ H_comb ≠ c @ H_comb` because `H_comb` is *not* co-rotated. So the energy
value — and its gradient — depend on the eigensolver's arbitrary `Ω`. The energy
is reproducible only because Eigen is deterministic; it is not a basis-invariant
quantity, and `∂E` would require reproducing Eigen's degenerate-subspace basis
derivative (not given by PT).

**Why (B) is clean.** With `h = c @ H_metric`, under `H_metric → R H_metric`:
`c → R c` and `h = (Rc) @ (R H_metric) = c @ H_metric` (RᵀR = I) — **invariant**.
Equivalently, in basis-free form,

```
h = Π Qᵀ (Q Π Qᵀ)^{-1} target ,        Π = Σ_{i∈null} u_i u_iᵀ   (the null-space projector)
```

— the minimum-norm metric harmonic whose periods are `target`. `Π` and its
derivative `∂Π` are basis-invariant (projector PT divides only by nonzero `λ_a`,
no `Ω`), so `∂h`, `∂E` are clean and well-defined. This is also exactly what the
C++ `carriedRepresentative` computes (plus a "leak" term that forces the periods
to equal `target` exactly when `target` is not fully in the carried space; at the
reference `leak = 0`, and the two agree at `E = 1.000000`).

**Recommendation: adopt (B).** The #312 Python energy's use of the combinatorial
`H` for the representative while fitting `c` to the metric periods is, on this
reading, a **latent basis-dependence bug** — the energy off the reference depends
on an arbitrary eigensolver basis. (B) removes that dependence, matches the C++
`carriedRepresentative`/`residualForPeriods` path, and has an exact analytic
gradient. It changes the energy *values* slightly off-reference (they agree at
the reference); the physics — minimum-norm matter field carrying the fixed charge
— is the same, arguably more correctly stated.

> **If you prefer to keep (A) verbatim**, Part B is still possible but its gradient
> is eigensolver-specific (we'd have to reproduce Eigen's degenerate basis
> derivative, which may be non-smooth). I'd advise against it; flagging for your
> call.

The rest of this plan assumes **(B)**.

## 5. The gradient under (B), in closed form

```
Π        = Σ_{i∈null} u_i u_iᵀ                                  (n×n projector)
∂Π/∂l^2_e = Σ_{i∈null} Σ_{a: λ_a≠0} (1/(0−λ_a)) (u_a u_aᵀ (∂M) u_i u_iᵀ + h.c.)
A        = Q Π Qᵀ   (m×m) ,   h = Π Qᵀ A^{-1} target
∂h       = (∂Π) Qᵀ A^{-1} target  −  Π Qᵀ A^{-1} (Q (∂Π) Qᵀ) A^{-1} target
∂E       = Re[ <h, (∂w1) h>  +  2 <w1 h, ∂h> ]
```

with `∂w1` diagonal (§1). Cost per gradient: **one** eigendecomposition of `M`
(for `{λ_a, u_a}`), then per-edge `∂M` (local) and the closed-form contractions —
no per-edge eigendecomposition. The leak term (when `target ∉ carried space`) adds
a piecewise-smooth correction differentiated the same way; at/near the reference
it is zero.

## 6. `∂M/∂l^2_e` — the metric L1 derivative

`M = laplacian(1, metric=true)` is assembled from the boundary maps and the Hodge
weights `W0, W1, W2` (simplex volumes), which depend on `l^2`. The dependence is
through the weights only, and each weight is a closed-form volume
(`Simplex::volume`, Cayley-Menger) — so `∂M/∂l^2_e` is **local and closed-form**
(reuse the same volume-derivative machinery as Part B/Piece 2's `∂R²`). For the
prototype we may finite-difference the *assembly* (`laplacian(1,true)` is cheap —
no eigendecomposition) to validate the PT first, then make `∂M` analytic for C++.

## 7. Verification plan (same rhythm as Part A)

1. **Python prototype**, level-0 then level-1 merge:
   - replicate the (B) energy `E_B = Re<ψ, w1 ψ>`, `ψ = carriedRepresentative`;
   - `∂Π` via PT (cross-terms only), `∂h`, `∂E_B`;
   - **verify `∂E_B` vs central differences of `E_B`** to ~1e-6, real **and**
     complex/boost hinges.
   - Also report `E_A − E_B` off the reference (quantify the §4 change).
2. **C++ port** + a regression test pinning the analytic `∂E` to FD of `E_B`.

## 8. C++ implementation

- `∂E`/`∂P` belongs with the register: a method on **`EigenstateSynthesis`**
  (it owns `cyclePeriods`/`carriedRepresentative` and the spectrum) or
  **`Register`** — e.g. `energyGradient(holes, target, w1)` returning per-edge
  `∂E/∂l^2`, or `carriedRepresentativeGradient` returning `∂ψ`. Reuse
  `HodgeLaplacian::spectrum(1, metric=true)` for `{λ_a, u_a}` and the
  volume-derivative helper for `∂M`.
- Assemble `∇G` (a small relaxation driver): `∂Re S` and `∂Im S` from
  `ReggeSolver::actionGradientExact` (Part A), `∂E` from Part B,
  `∇G = ∂Re S + κ ∂E + λ sign(Im S) ∂Im S`.
- Wire into a gradient-based per-edge relaxation; run the κ sweep that converges.

## 9. Open questions for review

1. **§4 — adopt the basis-invariant self-consistent energy (B)?** (Recommended.)
   It changes `E` off the reference and gives a clean exact gradient; (A) keeps the
   #312 Python values but has an eigensolver-specific gradient.
2. **Where the C++ method lives** — `EigenstateSynthesis` vs `Register` vs a new
   relaxation class for `∇G`.
3. **The leak term** — include its (piecewise-smooth) derivative now, or assume
   `target` stays in the carried space near the emergent dual (leak ≈ 0)? The
   former is exact everywhere; the latter is simpler and exact at/near the
   reference.
