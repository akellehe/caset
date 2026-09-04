# Choi-encoding experiment: a synthesized bulk as the encoding of a chosen operator (#936)

Records of `examples/cobordism/choi_encoding.py` live under
`~/cobordism-runs/choi-encoding/` (`onemode-diagonal.json`, `onemode-whitney.json`,
`run-diagonal.json`, `full-L6-g12.json`, `probe-*.log`).

## The experiment

Two prepared unit 3-circles A and B are the inputs and the whole complex (bulk plus
boundary) is the output. An operator `U` on the direct sum of the two boundary
registers is chosen; its outputs `Θ_j = U(ψ_j ⊕ φ_j)` are computed algebraically for a
spanning set of input pairs; the bulk is synthesized with the boundary geometry and
amplitudes fixed so that the whole complex carries every witness as an eigenstate at
one common eigenvalue whose whole-complex readout is `Θ_j`. The bulk is then frozen and
held-out inputs, including attachment permutations of the boundary cells, are read by
the Poincaré–Steklov extension `Ψ_I = −(L_II − λ)^{-1} L_I∂ Ψ_∂` and compared with their
algebraic outputs.

**Representation.** The degree-0 operator of `EigenstateSynthesis` is the U(1)
connection Laplacian `L = D − A` on vertices (edge weights and phases). It is not a
Hodge operator, so the process-wide metric source does not enter it; the record reads
bit-identically under diagonal weights and the Whitney pencil (`onemode-diagonal.json`
and `onemode-whitney.json` agree to the last digit). The isolated unit 3-circle has
eigenvalues `{0, 3, 3}`; the two ω-modes at 3 are each circle's qubit. The
whole-complex readout is the coordinate vector of the witness on the seed bulk's
interior vertices in a fixed orthonormal frame (rows of the unitary discrete Fourier
transform), one row per output dimension.

**The primitive.** `MultiCobordism::relaxWholeComplexReadoutTargets` fixes the boundary
geometry and both components' amplitudes for every witness, imposes the readout
constraints exactly by parametrizing each witness's free amplitudes on the affine
solution set of its readout system (a particular solution plus the readout null space;
no penalty weight), and minimizes only the common-eigenvalue Rayleigh residual of
`relaxBoundaryStatePairs`. It is an additive affine-auxiliary path of the shared
fixed-cochain optimizer: with no readout system the parameter vector and arithmetic are
unchanged, and the same `relax_boundary_state_pairs` call on this branch and on `main`
(`1232c8e`, built into a throwaway venv) returns bit-identical residual, eigenvalue, and
trace (`0.027163236386005214`, `3.10327005689614`).

## What the whitepaper fixes in advance

Every generator in the construction is quadratic, so the dynamics is exactly
quasi-free; the full exterior space can represent non-Gaussian sectors, but no
generator currently present produces them from Gaussian boundary data. A synthesized
bulk is therefore a linear map from boundary data `ψ ⊕ φ` to the whole-complex state,
and the operator classes are:

1. one-particle operators `Θ = U(ψ ⊕ φ)` — encodable by a linear bulk;
2. the fermionic Fock lift `Γ(U)(ψ ∧ φ)` — encodable by the same bulk, read as
   determinants of one-particle reads (Cauchy–Binet);
3. tensor-product operators on `ψ ⊗ φ` — not encodable in the direct one-particle
   reading: the product spanning set `{e_a ⊕ e_b}` is linearly dependent in the direct
   sum (`(e_0⊕e_0) − (e_0⊕e_1) − (e_1⊕e_0) + (e_1⊕e_1) = 0`), so a linear extension with
   `L_II − λ` invertible forces `Σ_j c_j Ψ_j = 0` and hence `Σ_j c_j Θ_j = 0`, which every
   tensor-product operator violates (norm 2 for CNOT and for the identity on `⊗`),
   while every one-particle operator respects it (`2.6e-16` measured).

## Results

### Class 1 and class 2, one mode per side (a beam splitter between A and B)

`onemode-diagonal.json` / `onemode-whitney.json`: 3-layer annulus, two witnesses,
8 restarts, growth ≤ 12, 1000 iterations, `epsilon = 1e-16`.

| quantity | value |
|---|---|
| fit residual (sum of squared eigen-residuals) | 7.18e-17, converged after 4 growths, 24 s |
| common eigenvalue | 6.7321 |
| readout deviation on the returned witnesses | 2.5e-16 |
| boundary drift | 0 (bit-identical) |
| witnesses equal the extension of their own boundary values | 1.3e-8 |
| recovered operator vs `U ∈ U(2)` (Frobenius) | 1.6e-8 |
| held-out reads vs `U x` (16 inputs, max) | 1.5e-8 |
| attachment rotations `C₃ × C₃` (9) vs `U` of the rotated inputs (max) | 1.5e-8 |
| Dirichlet gap `σ_min(L_II − λ)` on reads | 0.10 |
| class 2: two-particle amplitudes (determinants of reads) vs `Γ(U)(ψ ∧ φ)` (16 trials, max) | 6.9e-9 |

The read errors are the square root of the fit residual, as they must be. Every check
of the protocol passes: the bulk represents the pairs it was relaxed on, unseen inputs
in the span read as their algebraic outputs, and permuted attachments read as the
outputs of the permuted inputs. With one mode per side a reflection of a circle swaps
its ω-modes and leaves the channel, so the attachment permutations range over the
rotations.

### Class 1, both modes per side (qubits, `U ∈ U(4)`, four witnesses)

The four-witness common-eigenvalue fit did not converge in any probed configuration.
Parameter counts (real unknowns vs real equations) are not the limit: at 5 and 6 layers
the slack is positive before any growth.

Budgets are written restarts × growth × iterations; "cold" passes redraw every
restart at random, "warm" passes descend first from the previous pass's witnesses
(the warm start landed after the cold rows were measured).

| bulk | witnesses | budget | best residual | eigenvalue | growth | free edges | source |
|---|---|---|---|---|---|---|---|
| 3 layers | 4 | 4 × 8 × 400, cold | 2.45e-4 | 3.362 | 8 | 48 | `run-diagonal.json` |
| 3 layers | 4 | 8 × 24 × 1000, cold | 5.75e-5 (best pass 1.3e-5) | 3.536 | 24 | 96 | `probe-readout.log` |
| 3 layers, **pairs only (no readout)** | 4 | 8 × 24 × 1000, cold | **9.64e-17, converged** | 5.160 | 23 | 93 | `probe-pairs.log` |
| 5 layers | 4 | 8 × 8 × 1000, cold | 7.78e-4 | 3.548 | 8 | 66 | `probe-readout-L5.log` |
| 5 layers, pairs only (no readout) | 4 | 8 × 8 × 1000, cold | 2.75e-5 (best pass 9.4e-6) | 3.957 | 8 | 66 | `probe-pairs-L5.log` |
| 6 layers | 4 | 8 × 8 × 1000, cold | 1.85e-4 (best pass 6.5e-5) | 3.604 | 8 | 75 | `probe-readout-L6.log` |
| 3 layers, each witness its own eigenvalue | 4 | 8 × 12 × 1000, cold | 7.33e-4 | 3.03–3.04 | 12 | | `probe2-indep.log` |
| 3 layers | 3 | 8 × 12 × 1000, cold | 8.06e-5 | 3.212 | 12 | | `probe2-three.log` |
| 3 layers | 3 | 8 × 12 × 1000, warm | 8.06e-5 (monotone trace, same floor) | 3.212 | 12 | | `warm-three.log` |
| 3 layers, two witnesses on both modes | 2 | 8 × 12 × 1000, cold | **9.73e-17, converged** | 4.579 | 5 | | `probe2-twomixed.log` |

Three readings. (i) Four independent common-eigenvalue eigenstates with prescribed
boundary traces ARE realizable on the 3-layer annulus: the pairs-only fit reaches
9.6e-17 after 23 growths, so the eigen-realizability alone is not the obstruction.
(ii) The difficulty scales with the witness count, not with the mode content: two
witnesses on both modes converge in 5 growths; three do not converge in 12; dropping
the common eigenvalue does not help. (iii) The readout constraint is what the
four-witness fit has not yet met: it pins the operator the bulk implements to the
chosen `U ∈ U(4)`, whereas the pairs-only fit lets the output emerge. The
growth-40 warm-start run of the readout fit is the pending measurement.

| configuration | witnesses | growth | free edges | real parameters | real equations | slack |
|---|---|---|---|---|---|---|
| 3 layers | 4 | 0 | 24 | 64 | 96 | −32 |
| 3 layers | 4 | 8 | 48 | 176 | 160 | 16 |
| 5 layers | 4 | 0 | 42 | 148 | 144 | 4 |
| 6 layers | 4 | 0 | 51 | 190 | 168 | 22 |
| 3 layers | 2 | 0 | 24 | 64 | 48 | 16 |

This is the realizability question #901 and #903 characterize, met here for a
prescribed four-dimensional operator; the two-witness coupled fit has converged before
(the merged operator experiment) and converges here in every form tried.

### Class 3 (CNOT over the product spanning set)

`run-diagonal.json` (3 layers, 4 × 8 × 400): the CNOT fit floors at 3.7e-4 and the
identity-on-`⊗` control at 4.2e-4, but the one-particle control on the same product
spanning set also floors (1.1e-4), so at this budget the numerical floor is not yet
attributable to the tensor structure; the algebraic obstruction (norm 2 versus 2.6e-16)
is. Two further readings from the record: the dependency witness `Σ_j c_j Ψ_j` has norm
30 (CNOT) and 17 (identity), and the Dirichlet gap is 1e-4 (CNOT) and 6e-4 (identity)
versus 0.10 on the converged one-mode bulk — the fit lowers its residual by driving
`L_II − λ` toward singularity, which is the one way a linear bulk can hold a nonzero
combination with zero boundary trace. That is the predicted evasion, and it is why the
held-out read (which needs `L_II − λ` invertible) is the operative test.

### Class 2, the decomposability witness

Every output the bulk reads for a product of one-particle inputs is a decomposable
bivector (Pfaffian 9e-17 measured over 16 trials), while the Bell-type target
`(e_0∧e_1 + e_2∧e_3)/√2` has Pfaffian 1/2 and best overlap `1/√2` with any decomposable
state. No quasi-free lift reaches it; this is the whitepaper's non-Gaussian boundary,
stated for the two-particle sector.

## Where this leaves the experiment

- The primitive is in place and exact: readouts hold to round-off, the boundary is
  bit-identical, and the no-readout path is unchanged.
- The Choi-encoding claim holds as stated for the two-dimensional direct sum (one mode
  per side): the bulk encodes a generic `U(2)` between the A and B modes, and its Fock
  lift, with reads at the square root of the fit residual.
- For qubits on both sides the four-witness common-eigenvalue fit is the open
  realizability item; every larger-budget probe lowers the residual under growth but
  none reaches the tolerance. The CNOT falsification is established algebraically and
  its numerical form waits on that fit.
