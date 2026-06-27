# Joint 3-fermion proton spin — findings (#489, part of #410)

Resolving the **composite** total spin that distinguishes a proton (J=½, J²=¾) from a Δ
(J=3/2, J²=15/4), for the emergent bound state — the question #485 left open (a *product* of
per-hole spinors floors at the mixture; the proton's ¾ is an entangled, mixed-symmetry state).

Module: `examples/cobordism/dk_joint_spin.py`. Read machinery reused from
`dk_composite_spin.py` (#485): dual-edge `cell_frame`, spin `wilson_line`, the per-hole spinor
extraction.

## 1. The corrected premise

The earlier #488 attempt read the joint state off a *static* color singlet and found the per-hole
**flavor** index not independent. That was an artifact of feeding **two** quarks into a state
that intrinsically needs **three**: the three-quark entanglement Fermi statistics ties to spin
only appears when the quarks are genuinely *interacted into being*. So the proton is **built**,
not read off a hand-made singlet.

## 2. The build — simultaneous pair creation

One **co-optimized** `MultiCobordism` (the #491/#492 engine):

    3 neutral q-q̄ pairs (Σ = 0)  ⟶  [ proton , antiproton ]

The proton and antiproton legs are co-optimized in **one** system — interacting them separately
would not produce the correct operator structure. The diquark/antidiquark form as emergent
interior structure; charge is conserved end to end. The proton's three quarks are the **three
emergent holes of the proton output block**, carved out via the new
`MultiCobordism.outputs[0].verts` accessor (a read-only binding; the engine is unchanged), with
the relaxed metric copied onto the sub-complex.

## 3. What the build delivers (validated)

* **Carries the color singlet** — the proton output block's `r_state → 0` on converged seeds.
* **Independent per-hole flavor** — the Dirac–Kähler charge is distinguishable across the three
  quark holes (relative spread ~0.2–0.5). This is exactly the structure #488's two-quark read
  lacked; the corrected three-quark build supplies it.

## 4. The composite J² — honest negative, and why

The J² operator is **validated exact** on clean spin-½ states (a regression guard in
`test_dk_joint_spin.py`):

| state | J² | expected |
|---|---|---|
| proton eigenstate `2\|uud⟩−\|udu⟩−\|duu⟩` | 0.75 | ¾ |
| Δ `\|uuu⟩` | 3.75 | 15/4 |
| product `\|uud⟩` | 1.75 | 7/4 |

(The earlier 4-component Dirac read was a basis artifact — its spin sector is two degenerate
doublets; reducing each spinor to a clean spin-½ qubit via its Bloch vector `⟨S_a⟩` fixes it.)

On the built proton (carried singlet, independent flavor), the composite J² nonetheless sits at
the **indefinite mixture (~9/4)**, not ¾:

    product  joint J² ≈ 1.5    per-hole J² ≈ 1.8
    2+1      joint J² ≈ 2.2    per-hole J² ≈ 2.8

This is **not** a convergence or pairing problem. Every available readout reduces each hole to a
single-qubit Bloch vector, so the three-quark spin state is a **product** — and a product
provably cannot represent the proton's **entangled** mixed-symmetry ¾ (the table above: the
product `|uud⟩` is 7/4, only the *entangled* combination is ¾). The build creates the
entanglement; the per-hole readout discards it. Charge-clustering the flavors into a 2+1 (u/d)
pairing and symmetrizing the "uu" spins does not help — `|1,1⟩|½,−½⟩` is still 7/4, not the pure
J=½ eigenstate.

## 5. Conclusion — the path to ¾ is #477

Resolving ¾ needs a **total-space spin operator** acting on the whole carried representative at
once (the entangled three-quark state as one object), rather than a product of per-hole spinors.
That is exactly the open item already scoped as **#477** ("the total-space spin operator — the
16×16 `Σ_ij` promoted to the cells … likely a new `DiracKähler` C++ method"). #489 establishes
the build and the validated measuring stick; #477 is the readout that the definite ¾ requires.
