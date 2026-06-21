<!-- Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved. -->

# Flavor as a split of the existing register — the negative result

*Investigation (#414, epic #410).* Can the `u`/`d` flavor label — the isospin index
that separates the proton (`uud`, `Q = +1`) from the neutron (`udd`, `Q = 0`) — be
read off the **existing** `W_ABC` geometry as a **split of the register it already
carries**, rather than a parallel flavor *hole register* (explicitly rejected)? Two
candidate splits are tested, both read **off the relaxed geometry**, per input window
(A, B, C):

- **(i) the spatio-temporal split** — separate each window's transport by the causal
  type (timelike / cross-layer vs spacelike / within-layer) of the cells it rides on;
- **(ii) the Dirac–Kähler taste multiplicity** — the `4`-fold taste degeneracy of the
  discrete Dirac operator `D = d + δ` (#415) as the flavor index.

**Verdict: NO.** Both candidates collapse to a **window-independent** value on the
A₄-symmetric junction, so neither distinguishes `uud` from `udd`. The cause is
**structural, not dimensional** (the dimensional reach is #429's): flavor is a
**symmetry-odd** label and the singlet/confinement is a **symmetry-even** fact, and the
two cannot share the same A₄-symmetric register. This sharpens the prior conclusion
(`proton_electroweak_finding.md`, `proton_charge_gauss_law.md`): it is not merely that
the color-only geometry "happens to be flavor-blind", but that **no symmetry-respecting
split of the register can carry flavor** — a theorem, proved below and pinned in
`tests/cobordism/test_proton_flavor.py`.

## The construction and what is symmetric about it

The relaxed `W_ABC` junction (`examples/cobordism/proton_observables.py`) carries the
color singlet on the symmetric apex interior (#413, `Spacetime::symmetricStackCells`).
Its four windows — the three quark windows A, B, C and the result window R — are **one
A₄ orbit** at the icosahedron's tetrahedral vertex-orbits (#398). The window-cycling
`g` (the A₄ 3-cycle fixing R) is **transitive on {A, B, C}**: it permutes
A → B → C → A, and `P_out` carries the cyclic Z₃ spectrum `{1, ω, ω²}`. The transport
**intertwines color Z₃** on the symmetric interior:

```
‖ M P_in − P_out M ‖ / ‖M‖  =  4.5×10⁻¹⁴   on the exact uniform metric (#413),
                            ≈  9×10⁻⁶       on the lightly-relaxed (RELAX=25) build,
```

both **orders below** the prism's `4.26×10⁻²` discretization residual. This equivariance
is **load-bearing**: it is precisely what forces the singlet overlap to `1.000000` and
the color charge `σ → 0` (confinement). The flavor audit the
ticket mandates (F4 = the #412 G6 base-vertex-relabeling invariance) is **the same
equivariance**: a measurable that is invariant under the base-vertex relabelings that
realize `g` necessarily commutes with `g`.

## The obstruction (the theorem)

> **Claim.** Let `f = (f_A, f_B, f_C)` be any per-window measurable that is invariant
> under base-vertex relabeling (the F4 / #412 G6 property). Because the window-cycling
> `g` acts as a **transitive** permutation on `{A, B, C}` and `f` commutes with `g`,
> `f` is **constant on the orbit**: `f_A = f_B = f_C`. Hence its `u`-vs-`d` discriminator
> margin `max f − min f` is **identically 0** (exactly so on the uniform metric; bounded
> by the intertwining residual ~`9×10⁻⁶` on the relaxed build — the measured Dirac–Kähler
> margin `~3×10⁻⁶` sits right at that bound, confirming the collapse is *forced* by the
> equivariance, not a numerical accident).

A flavor label `uud` singles out **one** window as different from the other two — it is
**symmetry-odd** (it breaks the transitive `g`). So **F4 (relabeling-invariance) and F3
(a real discriminator, margin ≥ 0.1) are mutually exclusive** on the symmetric register.
The only way to make the per-window measurable differ is to **break the geometric A₄
symmetry**, which abandons the exact intertwining `M P_in = P_out M` — and with it the
singlet (the relaxation on a non-symmetric metric does not produce the singlet; #398's
greedy/jittered build reaches only `~0.74`). The proton and neutron must **both** be
color singlets (G1/G2), so the symmetry cannot be given up. The obstruction is
dimension-**independent**: it holds in the full emergent dimension too, because the three
windows remain a single A₄ orbit however far the apex interior is coned (#429).

## What each candidate measures (and why it collapses)

Measured on the relaxed junction (`build(max_iters=25)`, seed `410414`), per window:

### Candidate (i): the spatio-temporal split

| reading | A | B | C | margin |
|---|---|---|---|---|
| causal census (edges) | timelike `0`, spacelike `774`, null `0` (whole complex) | | | — |
| period timelike fraction | `0` | `0` | `0` | `0` |
| field `‖E‖` (timelike-leg plaquettes) | `0` | `0` | `0` | `0` |
| field `‖B‖` (spacelike plaquettes) | `0.19049` | `0.19046` | `0.19047` | `3×10⁻⁵` |
| electric plaquette count `#E` | `0` | `0` | `0` | — |
| `Q_electric = ∮_S E` per window | `0` | `0` | `0` | `0` |

The relaxed junction is **Riemannian — `0` timelike edges**, so the **electric
(timelike-leg) sector is empty**: `fieldStrengthSplit` finds no electric plaquettes and
every window's field strength `F = dψ` is entirely **magnetic**. The signed-edge period
rides only the three spacelike hole-boundary edges of its window, so its timelike
fraction is identically `0`. There is **nothing to split** — and the magnetic content
is **equal** across the three windows (A₄-symmetric), confirming the theorem. (Populating
`E` needs genuine Lorentzian worldlines, a *metric* choice — but even a populated `E`
sector would be A₄-equal per window, so it would still not label `u` vs `d`.)

### Candidate (ii): the Dirac–Kähler taste multiplicity

| reading | A | B | C | margin |
|---|---|---|---|---|
| per-window charge `q_k = ⟨Φ_k, Φ_k⟩_W` | `6.61798` | `6.61799` | `6.61799` | `3×10⁻⁶` |
| charge-density minimum `min_c j⁰_c` | `0` | `0` | `0` | (≥ 0) |
| `multiplicity()` | `4` | `4` | `4` | (fixed) |

The Dirac–Kähler charge collapses **two ways**:

1. **It is A₄-equal across windows** (`margin ~3×10⁻⁶`, machine precision) — the theorem
   again: `q_k` is a relabeling-invariant per-window measurable, so it is constant on the
   orbit. It cannot separate one window from the other two.
2. **It is positive-definite.** `j⁰_c = W_c |Φ_c|²` is a **norm / constituent density**
   (`dens_min = 0`), and `charge()` returns `⟨Φ, Φ⟩_W > 0`. It can never carry the
   **opposite signs** the flavor-electric charges need (`u: +⅔` vs `d: −⅓`); it counts
   constituents, it does not sign them.

Separately, the taste multiplicity is a **fixed framework constant** `multiplicity() = 4`
(four lattice **tastes**, the staggered/Dirac–Kähler doubling), **independent of the
window** and **not a 2-valued isospin doublet** (`4 ≠ 2`). There is moreover **no
per-window taste projector** — the `DiracKahler` API exposes the conserved current and
the `16 = 4 × 4` Clifford action, but the taste-block projector is explicitly deferred
(`include/cobordism/DiracKahler.h`, the `diracMatrices`/`tasteProjector` note). Even if it
existed, the A₄ symmetry would force all three windows into identical taste content.

### The discriminator

The would-be per-window `u`/`d` label vector `sign(q_k − mean q)·[|q_k − mean q| > 0.1]`
is the **constant zero vector** `[0, 0, 0]`: it is **neither `uud` nor `udd`**, and the
two assignments are indistinguishable. A `uud` labeling and a `udd` labeling produce the
**same** (neutral) geometry and the **same** charge — the existing `Q = ∮_S E = 0`,
`net Dirac–Kähler = 0` neutral total. The ticket's success target `Q(uud) = +1` vs
`Q(udd) = 0` is therefore **unreached** on the symmetric register.

## Why a parallel register is *also* not the answer (and what is)

The rejected route — a flavor *hole register* parallel to color — would add holes,
inflating `b₁` past `11` and double-counting holonomy (the G4 guard fails). It is rejected
not only as "arbitrary" but because it is the **wrong shape of fix**: flavor is not extra
**capacity**, it is a **symmetry-breaking** between otherwise-equivalent windows. The
faithful resolution is an explicit **isospin structure** that distinguishes the windows —
a (necessarily new) construction in which the three windows are **not** a single
symmetric orbit, e.g. an SU(2) isospin doublet attached to the windows so that one is
`I₃ = −½` (`d`) and two are `I₃ = +½` (`u`), giving `Q = I₃ + Y/2` with the triality
`Y/2` the construction already carries (`proton_charge_gauss_law.md`). That is
symmetry-**odd** new structure, by the theorem above; it cannot be recovered as a split
of the symmetric color register. The **dimensional** reach needed to host the full `E⃗`
and `4×4` Dirac structure is a separate question, owned by #429 (iterated apex-reflection
cobordism); it is **not** the obstruction here.

## Faithfulness

The conclusion is pinned in `tests/cobordism/test_proton_flavor.py` (N1–N7): the electric
sector is empty (N1), both candidate splits collapse to margin `0` (N2, N3), the
Dirac–Kähler charge is A₄-equal and positive-definite (N4), the discriminator is the
constant zero vector (N5), the structural cause — transitive 3-cycle + exact intertwining
⟹ constant on the orbit (N6) — and an `xfail` pinning the unreached `Q(uud)=+1` /
`Q(udd)=0` target with its precise obstruction (N7). The shared epic invariants
(`tests/cobordism/test_epic410_invariants.py`, G1–G5) are unchanged: this investigation
adds **no geometry** (`b₁ == 11`), only a read-out, and reads the same neutral singlet.

## References

- `docs/design/proton_electroweak_finding.md` — the #405 deferral; "charge needs flavor".
- `docs/design/proton_charge_gauss_law.md` — `Q = ∮_S E`; the A₄-equal per-window
  Noether charges; the `I₃` (isospin) factor named as the missing piece.
- `docs/design/proton_symmetric_windows.tex` — the exact intertwining / Stokes constraint
  and the g-invariant metric (the symmetry this note shows is incompatible with flavor).
- `examples/cobordism/proton_flavor_split.py` — the worked example reading the two
  candidate measurables and the discriminator collapse off the relaxed geometry.
- `include/cobordism/DiracKahler.h` — `charge` / `chargeDensity` (the positive-definite
  `j⁰`), `multiplicity` (the fixed `4`-taste constant), and the deferred taste projector.
- `include/cobordism/EigenstateSynthesis.h` — `curvatureFromConnection`,
  `fieldStrengthSplit`, `gaussLawCharge` (the E/B split and the temporal Gauss law).
