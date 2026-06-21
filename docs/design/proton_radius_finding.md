<!-- Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved. -->

# The proton radius is not dynamically emergent in the pure-geometry junction (#403)

**Question (#403).** The emergent-observables read (#400) found the proton radius pinned
to the lattice scale (`r ≈ 1.06 ≈ uniform l² = 1`): the relaxed geometry stayed
~uniform, so `r·m` was dominated by the mass and not a clean dimensionless probe. Can a
genuine bound-state size be made to **emerge** off the relaxed geometry, without imposing
one — un-pinning the radius?

**Answer: not within the pure-geometry construction.** Two emergent-first routes were
tried; both confirm there is no emergent length scale to find. Un-pinning the radius
requires a localizing force (a confinement scale / mass term) that the construction does
not contain by design.

## What was tried

### 1. The van Raamsdonk / entanglement seed (`set_entangled_metric`)
The singlet's mutual information, computed from its reduced density matrices (not
hand-picked), is **uniform**: `intra-window = cross-window = log 3` — the color singlet
`ε` is maximally and *symmetrically* entangled. A uniform MI seeds a uniformly rescaled
metric, not a localized one:

| seed | bound-region rms | bulk rms | r·m |
|---|---|---|---|
| uniform (the #400 baseline) | 1.06 | 1.02 | **6.94** |
| VR / entanglement (singlet MI) | 11.9 | 9.0 | **77** |

There is no interior/exterior contrast (bulk/bound ≈ 0.8) and `r·m` gets **worse** (77 vs
6.9, target ~4.0). At the seed the quark windows are momentarily compact, but the
stationary-point relaxation expands them away. (Forcing localization with *asymmetric*
intra ≠ cross would be fabricating a size — not emergent.)

### 2. Better radius observables on the existing relaxed geometry
Read off the #400 uniform-relaxed singlet (no new seed/physics):

| radius observable | value | r·m |
|---|---|---|
| RMS edge (baseline) | 1.06 | 6.94 |
| inter-quark geodesic distance | **1.00** | 6.53 |
| deficit-half radius (geodesic r enclosing 50% curvature) | **1.00** | 6.53 |
| min-MI cut (the entanglement boundary, via `I ~ exp(−ℓ)`) | 2.12 | 13.85 |

Every metric-based radius is **~1 lattice unit**. The reasons:
- the relaxed geometry is essentially uniform — the quarks sit on *adjacent* windows, so
  the inter-quark geodesic distance is exactly one edge;
- the curvature is **spread, not localized**: total deficit ≈ 1524 over 684 edges ≈
  2.2/edge, roughly uniform — that is the coarse lattice's *bare* curvature, not a
  concentrated matter mass;
- the **minimal-MI cut** (a cut enclosing the particle has weight `Σ exp(−ℓ)` over its
  edges, a van Raamsdonk MI proxy) therefore lands on the sphere's geometric
  bottleneck (equator/antipode), not a particle boundary — because there is no localized
  particle to cut tangent to. A full graph min-cut would behave the same.

## Conclusion

The proton's "size" in this construction is the **fixed angular separation of the quark
windows on the `S²`** — a construction choice (the icosahedral `A₄` window positions),
not a dynamically emergent scale. The relaxation does not change it (uniform geometry),
and a finer lattice (#404) does not help (it is more resolution of the same uniform
geometry, with the window separation fixed in lattice units).

To make the radius **dynamically emergent** one needs a localizing force that sets a
length — a confinement potential or a mass term that concentrates the matter curvature.
That is new physics, and it departs from the pure-geometry, no-Dirichlet-source design
(matter is observed *as* curvature, never imposed). It is out of scope for a seed or
read-out change.

**Status:** #403 is **deferred** pending such a mechanism. The lattice-pinned radius is
accepted for now: the construction is the color-confinement *skeleton* of the proton
(emergent color singlet, confinement, the right dimensionless ballpark), not a
quantitatively resolved particle — consistent with the assessment in
`proton_so_far` (#398/#400/#404). The next tractable step is the electroweak quantum
numbers (#405).
