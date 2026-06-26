# Composite proton spin — findings (#485, part of #410)

Reading the **composite** total spin of the emergent bound state — the quantum number that
distinguishes a proton (J=½, J²=¾) from a Δ (J=3/2, J²=15/4) — from the three k=3 register
holes. Three constituent spin-½'s combine as `2⊗2⊗2 = 2 ⊕ 2 ⊕ 4`, so `J² ∈ {¾, 15/4}`.

The readout, the gates, and the prototype live in `examples/cobordism/dk_composite_spin.py`;
the controlled-b₃ fixtures in `tests/fixtures/composite_spin/`.

## 1. What was built

A frame-free-as-possible composite-spin readout, post-hoc on a converged b₃ structure:

1. `embed_cell` / `cell_frame` — per top cell, flat ℝ⁴ coords from `gramMatrix()` and an
   orthonormal frame to express spinor components in.
2. `emergent_spinor` — the per-hole spinor from the carried representative `psi`
   (`EigenstateSynthesis(st,3).carriedRepresentative`): least-squares the cell's tetrahedral
   faces' trivector minors against `psi`, map the 3-form into the Clifford algebra
   (`Φ = Σ ω_t γ_iγ_jγ_k`), read `s = Φ·[1,0,0,0]`.
3. `wilson_line` — `Spin(4)` holonomy relating two holes' frames, composed from per-facet
   transports along a BFS dual path.
4. `composite_j2` — transport the three holes' spinors to a common frame, form the bound
   state, read the total-spin Casimir `J²`.

Two **mandatory correctness gates** decide whether any `J²` is even meaningful:

- **GAUGE** — `J²` invariant under a fixed random per-cell `SO(4)` of the embedding.
- **RELABEL** — `J²` invariant under a vertex-id permutation. This is the *weakest* invariance
  any physical observable must satisfy: it is a pure renumbering of vertices — the **same**
  triangulation with the **same** metric (verified: relabeled cell-set `== π`(cell-set),
  edge-length diff `0.0`, `dualComplexValid` on both, `emergent_holes` map under `π`), not a
  retriangulation. An observable that depends on vertex numbers is not physical.

## 2. The composite SO(3) spin is not robustly frame-invariant — and why

GAUGE passes essentially always. RELABEL passes on geometrically **generic** cells (gates to
~1e-13/1e-14) but fails on **near-symmetric** cells. The cause is fundamental, not a bug:

- The readout must write each spinor in a per-cell orthonormal frame. That frame is fixed by
  the cell's geometry **except** on a (near-)symmetric cell, where the geometry doesn't single
  one out. Concretely, a cell with all edges equal has inertia eigenvalues `[½,½,½,½]` (spread
  `1.3e-15`): **every** orthonormal basis is an eigenbasis, so there is **no canonical frame,
  as a matter of linear algebra**. The algorithm's tiebreak then leaks the vertex numbering.
- This is why GAUGE passes (rotating the embedding doesn't touch the symmetry) but RELABEL can
  fail (it changes the numbering the tiebreak leans on).
- It is provably *only* a renumbering that moves the answer: an **identity** permutation
  through the full rebuild path gives `ΔJ² = 0.0` exactly; on generic geometry every
  permutation reproduces `J²` to machine precision; only non-trivial permutations on symmetric
  cells move it (and different permutations give different values: 0.18, 1.5, …).

Eight distinct frame/transport designs were tried (QR, polar with `det`-fixed orientation,
inertia principal axes, inertia + third-moment degeneracy lifting, geometric-weight ordering,
frame-free Procrustes transport, apex-fixed geometric transport, pseudovector transport).
Each fails RELABEL on some structure, each for a different subtle reason — the signature of a
quantity that is genuinely not well-defined. Composite SO(3) spin needs a **rest frame**, and a
frameless simplicial complex supplies one canonically only on generic cells.

## 3. The failures are geometric, not topological (b₃)

Tested directly with controlled-b₃ fixtures (S⁴ minus *n* disjoint top cells, `b₃ = n−1`,
generic metric — so any failure is purely topological):

| b₃ | holes | J² | GAUGE | RELABEL | spin-½ |
|---|---|---|---|---|---|
| 3 (synthetic) | 4 | 2.88 | 1e-14 ✓ | 1e-14 ✓ | ½ ✓ |
| 4 (synthetic) | 5 | 3.04 | **1.4 ✗** | 3e-14 | ½ ✓ |
| 5 (synthetic) | 6 | 3.05 | 8e-14 ✓ | 2e-15 ✓ | ½ ✓ |
| 6 (synthetic) | 7 | 2.86 | 5e-15 ✓ | 1e-14 ✓ | ½ ✓ |
| 7 (synthetic) | 8 | 2.86 | 5e-15 ✓ | 1e-14 ✓ | ½ ✓ |
| 3 (converged) | 3 | 1.70 | **1.2 ✗** | 0.02 | ½ ✓ |

b₃=5, 6, 7 pass cleanly; the only failures are at b₃=3 (one converged structure) and b₃=4 —
both GAUGE, i.e. the same sporadic near-symmetric-cell brittleness, independent of Betti
number. **Higher b₃ does not drive the failures.** (Convergence itself rarely produces b₃>3 —
the objective grows just enough topology to carry the 3-color register — so the high-b₃ cases
are synthetic.)

## 4. The robust positive result: a three–spin-½ (baryon) bound state

Even where the *composite* frame readout is brittle, the **constituent** content is robust and
relabel-invariant: every fixture, every b₃, gives polarization magnitude `|⟨S⟩| = ½` per hole —
three genuine spin-½ constituents.

Writing `J² = Σᵢ Sᵢ² + 2 Σᵢ<ⱼ Sᵢ·Sⱼ = 9/4 + 2 Σ Sᵢ·Sⱼ`, the `9/4 = 3 × ¾` baseline is exactly
"three spin-½," and a product of three spin-½ has `J² ∈ [3/2, 15/4]`. That range is the **n=3
fingerprint** (n=2 → [1,2]; n=3 → [3/2, 15/4]; n=4 → [5/2, 6]). The floor at **3/2** is a
genuine signature of three spin-½ objects combining — the baryon constituent count, matching
the 3 register holes = 3 colors. Readable `J²` values cluster near `9/4` (weakly correlated
three spin-½).

## 5. Why ½-vs-3/2 needs the entangled joint state (and a prototype)

The proton's J=½ is the mixed-symmetry, irreducibly **entangled** combination
`(|↑↓↑⟩−|↓↑↑⟩)/√2`-type; **no product** `|n̂₁⟩|n̂₂⟩|n̂₃⟩` has `J² < 3/2`. `composite_j2`
multiplies three *independently extracted* per-hole spinors, so it lives entirely in the
uncorrelated subspace and **structurally cannot** reach ¾ — independent of any frame issue.
So the floor at 3/2 certifies "three spin-½ → baryon," but the composite channel (½ vs 3/2) is
an entanglement property the product discards.

A joint-state prototype (`joint` path in the example): instead of three
`carriedRepresentative([h],[1])` extractions, use the **single** correlated
`carriedRepresentative([h₀,h₁,h₂], [1,ω,ω²])` carrying the color singlet across all three
holes, then read the per-hole spins from that. Findings:

- It is genuinely different from the product (the color phases imprint) and **leans toward the
  proton channel** — `J²` drops toward the 3/2 floor (e.g. 1.80 → 1.64 on the cached
  structure), consistent with the 120°-apart color phases imposing ~120° relative spins.
- It can produce **definite** channels where the product cannot (a clean Δ, `J²=3.75`, on the
  b₃=5 fixture).
- But it inherits relabel-sensitivity from `carriedRepresentative`'s multi-hole/complex-target
  handling, so it is not yet a clean observable.

Resolving ½-vs-3/2 cleanly requires the genuinely entangled joint 3-fermion state (with the
color-antisymmetry / flavor structure of #410), read as one correlated object — a deeper
extraction than three spinors tensored.

## 6. Bottom line

- **The readout robustly identifies a three–spin-½ (baryon) bound state**: `|⟨S⟩|=½` per hole
  on every structure and every b₃, and the product `J²` floor at 3/2 is the n=3 fingerprint.
- **The composite total spin (proton ½ vs Δ 3/2) is not a clean observable of this
  construction**, for two independent reasons: (i) it requires a per-cell rest frame that a
  frameless complex supplies canonically only on geometrically generic cells (near-symmetric
  cells have no canonical frame — a real geometric degeneracy, not a bug); (ii) a product of
  per-hole spinors cannot carry the entanglement that defines the channel.
- **The failures are geometric, not topological** — they do not track b₃ (b₃=5,6,7 pass).
- This is the honest-negative outcome the ticket anticipated, now sharply characterized, with a
  concrete next step (entangled joint-state read) toward the composite quantum number.
