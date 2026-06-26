# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Entangled joint 3-fermion spin read — proton ½ vs Δ 3/2 (#488, part of #410).

#485 (`dk_composite_spin.py`) showed the **composite** total spin can't be read from a product
of independently-extracted per-hole spinors: a product floors at `J²=3/2`, but the proton's
`J²=¾` lives in an **entangled, mixed-symmetry** state. The channel is fixed by the **exchange
symmetry** of the 3-quark wavefunction:

* `color ⊗ flavor ⊗ spin ⊗ spatial` is antisymmetric; ground-state spatial is symmetric.
* the **color singlet `[1,ω,ω²]` is antisymmetric** (already carried — confinement) ⇒
  **`flavor ⊗ spin` is symmetric**.
* that symmetric `flavor⊗spin` splits into **Δ** (sym spin × sym flavor) and **proton**
  (mixed spin × mixed flavor, Clebsch-combined) — the proton's mixed-symmetry spin is entangled.

So the proton ¾ is only reachable from the **joint** state, built with color + flavor. This
module composes #485's machinery (dual-edge `cell_frame`, spin `wilson_line`, the `joint_*`
read) with #479's per-hole DK **taste/flavor** index, projects onto the color singlet
(antisymmetric `[1,ω,ω²]`, summed over hole permutations with the Z₃ signs), and decomposes the
resulting symmetric `flavor⊗spin` into the Δ (sym-sym) and proton (mixed-mixed) channels.

Validation: `J²` must resolve to a **definite** channel (¾ / 15/4), pass GAUGE + RELABEL (the
(anti)symmetrization should restore the RELABEL-invariance the ordered-`[1,ω,ω²]` joint read
lacked), and be robust across converged structures. **Honest negative is valid.** Post-hoc
only, never a loop condition.
"""
# Build plan (this PR fills these in, reusing dk_composite_spin.py + the DK taste sector):
# 1. per-hole (spin, flavor) extraction from the joint carriedRepresentative.
# 2. spin transported to a common frame (dual-edge frame + wilson_line, from #485); flavor internal.
# 3. color-singlet projection: antisymmetrize over the 3 holes with the [1,ω,ω²] Z₃ signs.
# 4. exchange-symmetry decomposition of flavor⊗spin → Δ (sym-sym) / proton (mixed-mixed); read J².
