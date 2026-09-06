# Simplicial qubit — implementation map and measurements (#955)

The construction is `simplicial_qubit_spec.md` (this directory), implemented
section by section in `observables::SimplicialQubit`. This note records where
each section lives, what the tests check, and the numbers behind the
section-12 assertions.

## Section by section

| Spec | Implementation |
|---|---|
| §2 input data structures | the constructor `SimplicialQubit(vertices, edges, faces, lengths, cycle_A, cycle_B)`; a `Spacetime` of dimension 2 holds vertices, edges and lengths underneath (`spacetime()`), and can be read directly by the second constructor |
| §2 validation on load | every edge in exactly 2 faces, χ = 0, consistent face orientations (opposite traversal of every edge), strict triangle inequality, closed cycles, independent homology classes (rank of [d1ᵀ \| c_A \| c_B]); each failure is a `ValueError` naming the offending element |
| §3 incidence matrices | `d0()`, `d1()`, and the exact check d1·d0 = 0 |
| §4 per-triangle geometry | `angles()` (law of cosines), `areas()` (Heron), `layout()` (p_i = 0, p_j = (c, 0), p_k = (b cos α_i, b sin α_i)); every per-face vector lives in that frame |
| §5 cotangent weights | `weights()` = ½(cot α_e + cot β_e); `negative_weight_edges()`, `non_delaunay_edges()` flagged with a warning; the optional pass `intrinsic_delaunay()` flips α_e + β_e > π edges until none remain, rerouting the marked cycles through the quadrilateral |
| §6 harmonic space | `harmonic_basis()`: the SVD null space of [d1; d0ᵀ M1] at scipy's tolerance, asserted two-dimensional (`RuntimeError` otherwise) |
| §7 L² inner product | `barycentric_gradients()` (rot90 of the opposite edge over 2A_t, sum verified zero), the Whitney interpolant at the barycenter, ⟨ω, η⟩ = Σ_t A_t W_t(ω)·W_t(η) |
| §8 complex structure | `gram()`, `rotation_pairing()`, `complex_structure()` = G⁻¹Rᵀ, `j_residual()` = ‖J² + I‖_F reported, never symmetrized |
| §9 holomorphic line | eigenvector nearest −i, `holomorphic_form()`, `periods()`, `tau()`; conjugate eigenvector when Im τ < 0 (a warning names it); marking (B, −A) and −1/τ when \|P_A\| vanishes (`marking_swapped()`) |
| §10 qubit state | `state()`, `bloch()` (unit norm asserted), `density_matrix()` |
| §11 metrics | module functions `fubini_study_distance(q1, q2)` and `weil_petersson_distance(q1, q2)` |
| §12 reference cases | `SimplicialQubit.flat_torus(tau, nx, ny)` and the tests below |
| §13 degeneration | `condition_m1()`, `condition_g()`, `near_degenerate()` against `degeneracy_threshold` (a Python `UserWarning`, never a failure) |
| §14 public API | the names above, snake_case, in `tessera.observables` |
| §15 scope | the other hemisphere is the opposite face orientation (or `reversed=True` when reading a `Spacetime`); gates act on `state()` |

The `Spacetime` container sorts the vertex order of every face, so the
consistently oriented faces of §2 are kept on the qubit; reading a `Spacetime`
derives an orientation from the fundamental class (`ChainComplex`).

## Measurements

Reference table (§12), 4 × 4 grid:

| torus | \|τ̂ − τ\| | ‖J² + I‖_F | \|r\| − 1 | non-Delaunay edges | cond G |
|---|---|---|---|---|---|
| square, τ = i | 5.4e-16 | 3.2e-16 | 0 | 0 | 3.0 |
| rectangle, τ = 2i | 1.3e-15 | 6.3e-16 | 0 | 0 | 6.2 |
| shear, τ = 0.3 + i | 6.3e-16 | 8.1e-16 | 1e-16 | 16 (negative weights) | 5.4 |
| hexagonal, τ = e^{iπ/3} | 2.5e-15 | 1.2e-15 | 1e-16 | 16 (negative weights) | 9.0 |

The construction is exact on flat tori whether or not the triangulation is
Delaunay: negative cotangent weights are flagged, as §5 says, and change
nothing on a flat metric. `intrinsic_delaunay()` on the hexagonal torus makes
16 flips, leaves every edge at length 1/4 (equilateral triangles), no
violations, and τ̂ − τ at 4e-15.

Refinement (§12) on a non-flat metric — the square torus with lengths scaled by
exp(φ) at edge midpoints, φ = 0.3 sin(2πx) cos(2πy) (conformally flat, so the
conformal structure is still τ = i):

| N | n_E | ‖J² + I‖_F | \|τ̂ − i\| | rates |
|---|---|---|---|---|
| 4 | 48 | 3.87e-2 | 1.45e-2 | |
| 8 | 192 | 1.60e-2 | 1.21e-3 | 1.28 / 3.57 |
| 16 | 768 | 4.66e-3 | 8.24e-5 | 1.78 / 3.88 |
| 32 | 3072 | 1.21e-3 | 5.26e-6 | 1.94 / 3.97 |

Both decrease monotonically; the residual tends to second order in the mesh
size and τ to fourth. The spec promises first order.

Pinching (§13), τ = ir on the 4 × 4 grid:

| r | τ̂ | r_z | cond G | d_FS to the square | d_WP to the square |
|---|---|---|---|---|---|
| 0.5 | 0.5i | 0.600 | 6.2 | 0.322 | 0.693 |
| 0.1 | 0.1i | 0.980 | 134 | 0.686 | 2.303 |
| 0.02 | 0.02i | 0.9992 | 3333 | 0.765 | 3.912 |

The state converges to |0⟩, d_FS stays finite and d_WP grows like −log r.

One property of the §13 detector on the §12 construction is worth knowing:
every diagonal of an axis-aligned rectangular grid is opposite a right angle in
both of its triangles, so its cotangent weight is exactly zero and
cond(M1) is ~1e16 for every rectangular torus, pinched or not. The warning
fires (as the spec prescribes; the message says which weights vanish), the
state is exact, and cond(G) is the number that tracks the pinching.
