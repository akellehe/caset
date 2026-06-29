# Connected correlator C_ij, two ways — the escape-hatch test (#512, part of #410)

Falsifiable test of `docs/theory/cobordism/proton-spin/cartan_weyl_gluon.tex` §7: **can the
composite proton spin be read from inter-hole holonomy alone**, sidestepping the fiber↔cell
(Whitney/Kähler–Atiyah) point-fiber assembly? Operationally: is the connected two-hole spin
correlator `C_ij` a *holonomy invariant*?

Module: `examples/cobordism/dk_connected_correlator.py`. Test:
`tests/cobordism/test_dk_connected_correlator.py`. Transport/extraction reused from
`dk_composite_spin.py` (#485); instrument cross-checked against `dk_joint_spin.j2_three_qubit`
(#489).

## 0. The identity this rests on

For three spin-½ holes,

    J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩,   with   ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij,

where `C_ij ≡ ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩` is the **connected** correlator. `C_ij = 0` for *any*
product state. So `C_ij` is exactly the entangling content, and J² in correlator coordinates is
identical to the validated `j2_three_qubit` operator.

## 1. Instrument — C_ij is the entangling content (exact, hand-fed clean states)

| state | J² (=9/4 + 2Σ⟨S_i·S_j⟩) | Σ C_ij | reading |
|---|---|---|---|
| proton `2\|uud⟩−\|udu⟩−\|duu⟩` | **0.75** | **−0.75** | entangled — C_ij carries *all* of the shift |
| product `\|uud⟩` | 1.75 | 0.00 | product — C_ij = 0 |
| Δ `\|uuu⟩` | 3.75 | 0.00 | product — C_ij = 0 |

The proton is the **only** one of the three with non-zero `C_ij`; both the product `|uud⟩` (7/4)
and the Δ (15/4) reach their J² from the *marginals alone*. This both reproduces the #489
measuring stick and isolates `C_ij` as the quantity a holonomy read would have to supply to
reach ¾.

## 2. The two routes (on the same emergent geometry)

- **(a) vertical** — reconstruct per-hole spin from the single color-correlated carried
  representative `carriedRepresentative([h0,h1,h2],[1,ω,ω²])` (`joint_spinors`), transport to a
  common frame via the `Spin(4)` `wilson_line`, build the 3-qubit state, read `C_ij`. A
  reconstruction from per-hole spinors is **separable**, so `C_ij = 0` by construction.
- **(b) horizontal** — predict `⟨S_i·S_j⟩_holo = ¼ cos θ_ij` from the inter-hole Wilson-line
  SO(3) angle `θ_ij` *alone* (`R_ba = Tr[S_b W S_a W†]`, `cos θ = (Tr R − 1)/2`), with **no**
  contraction against endpoint spinors. Also computes a color-period estimate `¼ cos Δφ_ij` from
  the emergent Z₃ phases when available. NB it assigns each pair its own angle, so it is *not*
  bound by the n=3 frustration floor — the most generous holonomy-only read.

## 3. Result — the escape hatch is CLOSED

| fixture | vertical (joint) J² | vertical (product) J² | horizontal J² | θ_ij | reaches ¾? |
|---|---|---|---|---|---|
| `synthetic_b3_3` | 2.79 | 3.15 | 2.26 | 0°, 120°, 120° | **no** |
| `converged_b3_3` | 2.38 | 2.67 | 1.44 | 101°, 117°, 166° | **no** |

- **Vertical:** `Σ C_ij ≈ 0` on both fixtures (separable, as predicted); J² sits in the
  three-spin-½ mixture band `[3/2, 15/4]`, never ¾.
- **Horizontal:** the holonomy reproduces only the **separable, transport-angle** correlation
  `¼ cos θ_ij` — exactly the `C_ij = 0` content. On `converged_b3_3` it even dips to 1.44, *below*
  the n=3 separable floor of 3/2, because the pair-independent formula is not a realizable state
  correlator; it still does not reach ¾. The color-period estimate is inconsistent across
  fixtures (1.75 vs 3.75) — color phases alone do not supply a stable ¾ either.
- **Frame-free:** the horizontal J² is GAUGE- and RELABEL-invariant to ~1e-15
  (`|ΔGAUGE|=4e-16`, `|ΔRELABEL|=2e-15` on `synthetic_b3_3`), so it is a genuine observable, not
  a frame artifact.

**Neither route reaches the proton ¾.** The holonomy invariants carry the separable
(transport-angle) part of `⟨S_i·S_j⟩` and nothing of the entangling `C_ij`. This is exactly what
Claim 1 of `cartan_weyl_gluon.tex` predicts: the Wilson-line transport is `K`-type (a local
frame rotation), and a `K`-type operation cannot create entanglement — so a correlation built
from it is separable and floors above ¾.

## 4. Conclusion

`C_ij` is **not** a holonomy invariant. The escape hatch is **closed**: the composite proton
spin cannot be read from inter-hole holonomy alone, and the fiber↔cell (Whitney/Kähler–Atiyah)
quantum two-hole lift — a genuine joint `ρ_ij`, not a separable per-hole reconstruction — remains
necessary to recover ¾. This is a clean negative that *narrows* the open work: it rules out the
holonomy shortcut and points the total-space operator effort (#477/#495) at the joint-state lift
specifically, since that is the only place the entangling `C_ij` can come from.

The honest positives this also establishes: (i) `J²` in correlator coordinates reproduces the
validated instrument, and `C_ij` is demonstrably the entangling content (§1); (ii) the
holonomy-only `J²` is a well-defined, frame-free observable (§3) — it just measures the wrong
(separable) thing.
