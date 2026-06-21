<!-- Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved. -->

# The electroweak quantum numbers need a richer construction (#405)

**Question (#405).** With the color singlet in hand (#398), can the proton's electroweak
quantum numbers — electric charge **+1**, spin **½**, isospin, flavor **uud**, baryon
number 1 — be read **emergent-first** off the existing W_ABC junction?

**Answer: not in the color-only construction.** The three input windows are *color*
registers (3 holes = 3 colors), related by the A₄/color-Z₃ symmetry and flavor-blind.
Every electroweak number needs structure the geometry does not carry.

## Per quantum number

| number | status | why | requirement |
|---|---|---|---|
| color singlet | **achieved** (#398) | the A₄→Z₃ intertwining forces it | — |
| electric charge +1 | not readable | charge is **flavor**-dependent (u:+⅔, u:+⅔, d:−⅓); the three windows are identical color registers, so nothing distinguishes the proton (uud, +1) from the neutron (udd, 0) | a flavor register |
| isospin / flavor (uud) | not readable | the windows do not label u vs d | a flavor register |
| spin ½ | not addressable | the construction is a spatial geometry with scalar/complex cochains — no spinor content | fermionic/spinor fields |
| baryon number 1 | structural, not a read | the 3→1 junction binds three inputs to one result by topology; it is the *shape* of W_ABC, not a computed observable | a baryon-number operator (post-flavor) |

## On "net orientation as charge" (a considered idea)

A natural emergent-first idea is to read charge from the **net orientation of the
simplices**. Three readings, none of which gives the electric charge:

- **As a flux (discrete Gauss law):** the signed sum of simplex orientations *is* a
  flux/charge in discrete exterior calculus — and in this construction that flux is the
  **color charge σ** (the +1-eigenmode period) we already read. It is →0 (the singlet is
  neutral). So net orientation = the color charge, correctly zero, not a new electric one.
- **As causal orientation:** the net *time*-orientation of the cross-layer (worldline)
  simplices carries the particle/antiparticle distinction (Feynman/CPT) — the **sign** of
  charge; flipping it gives the antiproton. But it is a ±1 sign fixed by the construction's
  orientation, not a magnitude.
- Either way it yields a sign or the (zero) color flux. It **cannot distinguish the proton
  (+1) from the neutron (0)**, because that distinction is flavor — which the color-only
  construction does not encode.

The idea is sound and connects to real structure (the color flux; the particle/antiparticle
sign), but it does not produce the flavor-dependent electric charge.

## Conclusion

Both proton follow-ons hit the same architectural boundary: #403 (size) needs a confinement
scale, #405 (electroweak) needs flavor. Neither is an emergent read in the color-only
geometry; both require a **richer construction** (a flavor register parallel to color;
spinor content for spin) — new structure, not a seed or read-out change.

**Status:** #405 is **deferred**. The proton program in this construction has reached its
natural scope — the color-confinement skeleton (#398 singlet, #400 observables, #404
convergence). The minimal extension is a flavor register; that is future work.
