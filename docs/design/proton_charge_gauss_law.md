# Electric charge as the temporal-sector Gauss-law holonomy

This note reads the proton's **electric charge** off the relaxed `W_ABC` color
singlet as a discrete **Gauss-law holonomy** `Q = ∮_S E`, and gives a verdict on
whether fractional (multiples-of-⅓) quark charge is structurally tied to
**triality** (the color `Z₃` the construction already carries) via the
Gell-Mann–Nishijima relation `Q = I₃ + Y/2`.

## The Gauss-law charge `Q = ∮_S E`

The register read-out is **kernel-only**: the carried representative
`ψ = carriedRepresentative(result_windows, [1, ω, ω²])` is a closed (harmonic)
1-cochain — the discrete U(1) connection that carries the singlet target on the
three result windows (`EigenstateSynthesis(st, 1)`). Its **field strength** is the
exact 2-cochain

```
F = dψ = curvatureFromConnection(ψ)        (EigenstateSynthesis(st, 2))
```

`F` splits by the causal type of each plaquette (`fieldStrengthSplit`, #417): the
**electric** part `E` lives on plaquettes carrying a timelike leg (one temporal
index, the discrete `F_{0i}`); the **magnetic** part `B` on purely-spacelike
plaquettes (`F_{ij}`).

The **electric charge** is then **Gauss's law on the temporal sector**:

```
Q(S) = ∮_S E = Σ_{p ∈ S} (induced-orientation sign) · E(p)
```

over a **closed surface** `S = ∂V` bounding the worldtube region `V` enclosing the
three quark windows. `EigenstateSynthesis::gaussLawCharge(F, enclosedVertices,
electricOnly)` builds `V` as the **closed star** of the window vertices (every
tetrahedron touching one), accumulates `∂V` as the signed boundary 2-chain
(interior faces shared by two `V`-cells cancel under the `(-1)^j` induced
orientation), and sums `E` (or the full `F`) over the surviving surface plaquettes.

### Why `Q` is a genuine gauged-U(1) holonomy

Because `F = dψ` is **exact**, the full closed-surface flux is

```
∮_S F = ⟨dψ, ∂V⟩ = ⟨ψ, ∂²V⟩ = 0
```

to round-off — the discrete coboundary `d` is **metric-free**, so the closed-surface
flux is **topologically protected** for any metric. This is the discrete Gauss law:
the net charge enclosed by a closed surface vanishes because the field strength is
sourceless (the singlet is neutral). It is the *same* protection as the exact Stokes
identity `σ_R = -(σ_A + σ_B + σ_C)` the color read-out already relies on.

## Measured values (relaxed `W_ABC` singlet, uniform `l² = 1`)

| quantity | value | reading |
|---|---|---|
| `‖F = dψ‖` | `0.189` | a genuine, nonzero field strength (sourced at the holes) |
| timelike edges | `0` | the relaxed seed is Riemannian (2+1 D reduced) |
| electric plaquettes | `0` | no timelike leg ⇒ the electric sector is **unpopulated** |
| `Q = ∮_S E` | `0` (exact) | the **electric Gauss-law charge** |
| `∮_S F` (full flux) | `4.4×10⁻¹⁷` | machine-precision **0** by genuine cancellation of nonzero `F` |
| covector `(⅔,⅔,−⅓)·periods` | `0.48` | the **non-holonomy** baseline (a hand-weighted number) |
| Dirac–Kähler per-window `j⁰` | `[2.225, 2.225, 2.225]` | the three quark charges are **equal** (A₄-symmetric apex, #413) |
| Dirac–Kähler net `Σ phaseₖ·qₖ` | `2.4×10⁻⁶` | the **net** Noether charge ⇒ 0 (neutral) |
| Dirac–Kähler total `Σ qₖ` | `6.70` | the constituent norm `⟨Φ,Φ⟩_W` (three colored quarks) |

So `Q = ∮_S E = 0`: the color singlet is electrically neutral, and the reduced
color-only geometry has **no populated temporal sector** to carry an electric field
in the first place.

### Jitter-robustness: holonomy vs covector

Under `N` independent spacelike-`l²` jitter perturbations (relative magnitude
`δ`, `numpy.random.default_rng(411)`), the protected Gauss-law `Q` stays put while
the non-equivariant covector drifts:

| `δ` | `std(∮_S F)` | `std(covector)` | `max|Δcovector|` | `std(covector)/std(∮_S F)` |
|---|---|---|---|---|
| `0.01` | `4.0×10⁻¹⁷` | `2.4×10⁻⁴` | `1.2×10⁻³` | `~6×10¹²` |
| `0.10` | `2.8×10⁻¹⁷` | `9.4×10⁻³` | `2.0×10⁻²` | `~3×10¹⁴` |
| `0.20` | `3.6×10⁻¹⁷` | `2.0×10⁻²` | `4.0×10⁻²` | `~6×10¹⁴` |

`Q = ∮_S E` is **exactly 0** for every jitter (the electric sector stays empty under
spacelike perturbations), and `∮_S F` stays at machine precision (the closed-surface
flux of an exact form is protected). The covector — a fixed `(⅔,⅔,−⅓)` weighting of
the per-window field-strength periods — is **not** A₄/color-`Z₃` equivariant, hence
**not** a closed-surface flux, so it drifts by `≥ 2` orders of magnitude more. The
period-based covector is itself a quasi-protected harmonic holonomy, so its drift
only reaches the `~0.07` "jittered-seed" scale under a sizable perturbation
(`δ ≈ 0.2`); the protection of `Q`, by contrast, is exact at every magnitude.

### Cross-check against the Dirac–Kähler conserved current `j⁰` (#415)

The Dirac–Kähler operator ships the conserved current's time component as the charge
density `j⁰_c = W_c |Φ_c|²` (`DiracKahler::chargeDensity`), summing to the Noether
charge `carriedCharge = ⟨Φ,Φ⟩_W` (`DiracKahler::charge`). On the carried
representative this **equals the period read-out's weighted norm** to `9×10⁻¹⁶` —
`j⁰` is the bona-fide conserved density of the harmonic the periods ride on.

The Gauss-law flux and the Noether charge **agree on neutrality**:

- The three per-window Noether charges are **equal** (`2.225` each — the A₄-symmetric
  apex interior, #413), so the **net** charge `Σₖ phaseₖ·qₖ` weighted by the singlet
  phases `[1, ω, ω²]` (which sum to `0`) is `2.4×10⁻⁶ ≈ 0` — the **net** Noether
  charge of the neutral singlet, matching `Q = ∮_S E = 0`.
- The positive **total** `carriedCharge = 6.70 = 3 × 2.225` is the constituent norm
  (the probability/particle measure for three colored quarks), **not** the net U(1)
  charge — it counts constituents, the neutral net is carried by the signed periods.

The Gauss-law net (`0`) and the Noether net (`0`) agree; the positive total counts
the three quarks. This is the discrete charge-conservation statement made concrete.

## Verdict: triality and the fractional-(⅓)-charge link

**The color-only `S²×I` geometry, as it stands, produces only the neutral total
(`Q = 0`); it cannot produce a flavor-dependent electric charge.** Three independent
facts pin this:

1. **The electric sector is unpopulated.** The relaxed singlet is Riemannian (zero
   timelike edges), so `fieldStrengthSplit` finds **no** electric plaquettes: the
   field strength `F = dψ` is entirely **magnetic** (the color holonomy lives in the
   spacelike plaquettes). The electric Gauss-law charge is therefore identically `0`
   — there is no temporal sector for an electric field to live in. Populating `E`
   requires genuine **Lorentzian worldlines** (timelike legs), i.e. relaxing the
   junction on a Lorentzian metric over the symmetric apex interior (#413), and —
   for the full `E⃗`-vector — a **3+1 D** cobordism over a triangulated `S³` (#418);
   the present `S²×I` is a 2+1 D reduced sector (out of scope here, per #410).

2. **The geometry is flavor-blind.** The construction carries **color** (the
   `Z₃ ⊂ S₃` window-cycling `g`, `P_out` eigenvalues `{1, ω, ω²}`) and nothing else.
   The three quark windows are **equivalent under A₄** — their Noether charges are
   equal to `12` digits. There is no isospin / `u`-vs-`d` index to break that
   symmetry, so the geometry assigns the windows their **color** values, never
   distinct **flavor-electric** values (`+⅔` for `u`, `−⅓` for `d`). The
   `(⅔,⅔,−⅓)` covector imposes exactly such a splitting **by hand**; it is not
   A₄-equivariant, not a closed-surface flux, and drifts under jitter — the
   numerical signature that it is **bookkeeping, not a holonomy**.

3. **The triality bridge is present but not yet sourced.** Gell-Mann–Nishijima
   `Q = I₃ + Y/2` with `Y/2 ~ baryon-number/triality` is structurally compatible
   with this geometry: the color `Z₃` center **is** the triality the construction
   carries, and the singlet is the definite-triality state `[1, ω, ω²]`. The
   ⅓-quanta of quark charge are exactly the values tied to a `Z₃`/`N`-ality center.
   But the geometry supplies **only** the `Y/2`-like (triality/baryon) factor; it has
   **no `I₃` (weak-isospin) factor**, because there is no flavor register. With both
   per-window charges equal and the triality phases summing to zero, the net is the
   neutral total — the proton's `+1` (and the neutron's `0`) require the **isospin
   splitting** `I₃ = +½` (`u`) vs `−½` (`d`) that a flavor index would provide.

**What the fractional/per-flavor split requires (named precisely).** A
flavor-dependent electric charge — `Q(uud) = +1` vs `Q(udd) = 0` — needs, in
addition to the triality/`Z₃` already carried, **either** (a) a **flavor (isospin)
register** parallel to color that breaks the A₄ window-equivalence and assigns
`I₃ = ±½` per window (the `u`/`d` label, #414), **or** (b) the **Dirac–Kähler
multiplicity** as the flavor/taste index (#414/#415) supplying the same `I₃`; and,
to carry a *nonzero* electric field at all, **(c) Lorentzian worldlines** populating
the electric sector, with the **full `E⃗`-vector** living on a **3+1 D `S³`** slice
(#418). Until then `∮_S E` correctly reports the only electric quantity the geometry
defines: the **neutral total, `Q = 0`** — robustly, as a genuine gauged-U(1)
holonomy, in contrast to the drifting hand-weighted covector.

## References

- `examples/cobordism/proton_observables.py` — `gauss_law_charge`, `covector_charge`,
  `dirac_kahler_net_charge`, and the `"charge_Q"` keys in `measure()`.
- `include/cobordism/EigenstateSynthesis.h`, `src/cobordism/EigenstateSynthesis.cpp`
  — `gaussLawCharge` (the `Q = ∮_S E` holonomy), `fieldStrengthSplit`,
  `curvatureFromConnection` (#417).
- `include/cobordism/DiracKahler.h` — `chargeDensity` / `charge` (the conserved
  current `j⁰`, #415).
- `docs/design/proton_electroweak_finding.md` — net-orientation = color flux; charge
  needs flavor (the prior conclusion this note refines).
- `docs/design/proton_bipartite_obstruction.tex` — triality, `ψ = (1, ω, ω²)`,
  `Z₃ ⊂ S₃`.
- `tests/cobordism/test_proton_charge_gauss_law.py` — the robustness, quantization,
  determinism, cross-check, and regression-guard tests.
