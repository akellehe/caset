# Total-space spin operator for the composite proton J² — design (#495, part of #410)

## Goal
Read the **composite** total-spin Casimir `J²` of the emergent proton (¾ for J=½, 15/4 for the
Δ) from the **whole carried representative at once** — the entangled three-quark state as one
Dirac–Kähler field — rather than from a product of per-hole spinors. #489 proved this is the
only way: the proton's ¾ is an entangled mixed-symmetry eigenstate, and any per-hole readout
reduces to a product, which provably sits at the mixture (~9/4).

## What #489 established (merged)
A converged proton (single co-optimized pair-creation build) whose output block **carries the
color singlet** with **independent per-hole flavor**, and a **validated** J² operator on clean
spin-½ qubits (proton → ¾, Δ → 15/4, product → 7/4). The gap is purely the **readout**: the
build creates entanglement, the per-hole-product read discards it.

## Characterization of the DK field (seed 5, residual ~4e-31)
- `DiracKahler(sub)`: `meshDimension=4`, `totalDimension=513`, `gammaDimension=16`,
  `multiplicity=4`; `blockOffsets=[0,20,109,284,453,513]` → `|C₀..C₄| = 20,89,175,169,60`
  (the 16-dim Kähler–Atiyah form fiber `Λ(ℝ⁴)` realized across the simplicial degrees).
- The carried joint representative `carriedRepresentative([h₀,h₁,h₂],[1,ω,ω²])` is a **single
  k=3 cochain** (169 values), lifted into the degree-3 block of the 513-length total field.
- The single-fiber spin Casimir `Σ² = ¼²·Σ_a [γ_i,γ_j]²` is **¾ uniformly** over the fiber —
  i.e. the *structural* spin-½ (#483). It is trivially ¾ for any field and is **not** the
  composite three-quark spin.

## The composite operator
The three quarks are the three register **holes**. The composite spin is

    J² = Σ_a ( Σ_{h∈holes} S_a^{(h)} )²,    S_a^{(h)} = P_h · Σ_a · P_h

acting on the **one** joint field `Φ` (so the cross-hole terms `S_a^{(h₁)}·S_a^{(h₂)}` carry the
correlations — the entanglement the per-hole product read threw away). `P_h` localizes to hole
`h`'s cells; `Σ_a` is the form-fiber spin generator. `⟨Φ|J²|Φ⟩/⟨Φ|Φ⟩` is the composite J².

## The core difficulty (why this is a new `DiracKähler` method)
`Σ_a` (the 16×16, gamma-built generator) acts on the form fiber `Λ(ℝ⁴)` **at a point**. But the
cochain stores components on **cells of different degrees that are not co-located** — the form
components at a "point" are spread over incident 0,1,2,3,4-cells. Promoting `Σ_a` to act on the
cochain therefore needs the **Whitney / Kähler–Atiyah identification** that assembles a
point-wise fiber from the incident cells (the "form degrees don't align point-by-point" snag
flagged when this was first scoped). This assembly is the substance of the operator and is why
it belongs in C++ on `DiracKähler` (alongside `lift`/`gammas`), not as a quick Python contraction.

## Plan
1. **Prototype (Python).** Build the #489 proton; attempt the localized-per-hole construction
   with the available `lift`/`gammas`/`blockOffsets`, using a Whitney-style assembly of the
   fiber at each hole. Validate `J² → ¾` on the built proton where the per-hole product gives the
   mixture; check GAUGE + RELABEL.
2. **C++ method.** If the prototype validates, add `DiracKähler::totalSpaceSpinJ2(holes)` (the
   assembled operator), bind it, doc it (`cpp_api.md`), and test it (¾/15/4 on synthetic +
   built-proton fixtures).
3. **Honest negative allowed.** If the carried representative does not retain the entanglement in
   a recoverable form (e.g. the Whitney assembly washes out the cross-hole correlation), report
   it — that would itself sharpen where the entanglement must instead be read (the build's
   intermediate diquark, or a different carried object).

See `docs/design/joint_proton_spin_findings.md` (#489) for the upstream finding.
