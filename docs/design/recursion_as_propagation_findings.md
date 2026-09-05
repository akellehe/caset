# Recursive-fiber layers as time propagation: first records (#943)

Records: `~/cobordism-runs/recursion/{velocity-l4,extend-l4,velocity-l8-seed1}.json`.
Script: `examples/cobordism/recursion_as_propagation.py`. All reads on the Whitney
pencil, analytic gradient (#951), single-threaded, phase descent off, 14 gated cone-ins
per fresh node, four rounds of stage 1 and stage 2 per layer, Jt = 0.2 in total.

## What a layer is, in both chains

The pair state Ψ_n lives on the pair frame (a 4×4 on |K⟩⊗|M'⟩). Layer n carries the
velocity χ_n = B Ψ_{n−1} as its two-body target; its two input fibers are the Schmidt
factors of Ψ_{n−1} (rank two: the one-particle content each frame must carry); the
state advances by the Euler step Ψ_n = Ψ_{n−1} − iJΔt χ_n^geo with the geometric read
χ_n^geo = c·T_AB(W_n). Beside it run the algebraic Euler chain (the discretization
error alone) and the exact N-block exponentials. In the `velocity` chain every layer
is a fresh interaction node; in the `extend` chain the same cobordism is continued.

## Records

Distances are ‖Ψ − Ψ_exact‖/‖Ψ_exact‖ after the layer.

| chain | layer | leak of χ_n in T_AB | read Schmidt spectrum | geometric vs exact | algebraic Euler vs exact |
|---|---|---|---|---|---|
| velocity, 4 layers | 0 | 0.797 | (0.644, 0.216, 0.020, 0) | 0.157 | 0.022 |
| | 1 | 0.875 | (0.581, 0.099, 0.057, 0) | 0.316 | 0.045 |
| | 2 | 0.725 | (0.646, 0.149, 0.060, 0) | 0.457 | 0.069 |
| | 3 | 0.704 | (0.671, 0.404, 0.052, 0) | 0.577 | 0.094 |
| extend, 4 layers | 0 | 0.879 | (0.592, 0.152, 0.058, 0) | 0.166 | 0.022 |
| | 1 | 0.790 | (0.670, 0.202, 0.054, 0) | 0.317 | 0.045 |
| | 2 | 0.707 | (0.850, 0.172, 0.057, 0) | 0.453 | 0.069 |
| | 3 | 0.653 | (0.984, 0.180, 0.091, 0) | 0.580 | 0.094 |
| velocity, 8 layers, seed 1 | 0 | 0.887 | (0.421, 0.165, 0.029, 0) | 0.060 | 0.0034 |
| | 3 | 0.701 | (0.841, 0.169, 0.121, 0) | 0.220 | 0.014 |
| | 5 | 0.672 | (1.2e7, 0.501, 0.099, 0) | 0.314 | 0.021 |
| | 7 | 0.877 | (0.507, 0.122, 0.030, 0) | 0.419 | 0.028 |

## Reading

- No layer carries its velocity: the leak stays at 0.65–0.90 in both chains, the
  single-layer bound of #941 (the coupling rank between two disjoint tetrahedra) met
  again at every layer. The geometric chain therefore drifts from the exact evolution
  linearly in the number of layers, about six times the algebraic Euler chain's
  discretization error, and halving Δt (eight layers) halves both but leaves the ratio.
  This is the leak compounding, as predicted, not a statement about the identification.
- The `extend` chain's continued cobordism strengthens its first coupling channel layer
  by layer (0.59 → 0.98) without opening the second to χ's ratio (0.29): continuing the
  same bulk does not add channels between the frames.
- One read in the eight-layer chain returned a first singular value of 1.2e7: a
  near-singular geometry that the admissibility gate admitted and the read did not
  refuse. The reversal identity still held. A transfer read should carry a conditioning
  certificate; recorded here, not yet added.

## Where this leaves #943

The chains, the bookkeepings, and the measurements are in place and cheap (seconds per
layer). The identification of levels with time cannot be supported or refuted until a
single layer carries its velocity; that waits on the coupling-rank lever recorded on
#941 (bulk connectivity between the frames, or overlapping frames through the carried
Gram). The `extend` chain is the one in which geometry composes and should be the
test once a layer carries χ.
