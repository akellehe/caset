# Simplicial qubit representation — implementation spec

Build a library that encodes a single qubit state as the holomorphic line in the harmonic space of the metric Hodge Laplacian on a triangulated torus. Input is intrinsic geometry (edge lengths). Output is a point of CP^1.

## 1. Objective

Given a simplicial complex `K` homeomorphic to `T^2` with a discrete metric (edge lengths), compute:

- the harmonic 1-cochain space `H = ker L_1`, of real dimension 2
- the complex structure `J` induced by the discrete Hodge star
- the holomorphic line `H^{1,0} = ker(J + i)`
- the period ratio `tau` in the upper half plane
- the qubit state `|psi> ∝ |0> + tau |1>`, its Bloch vector, and its density matrix

The dimension of `H` is topological (`beta_1 = 2`). The choice of line inside it is metric. Both must be computed, not assumed.

## 2. Input data structures

```
Vertices:  V = [0 .. nV-1]
Edges:     E = [(i, j)]        i < j, oriented i -> j
Faces:     F = [(i, j, k)]     consistently oriented (all counterclockwise w.r.t. surface orientation)
Lengths:   ell: E -> R_{>0}
Cycles:    A, B: lists of (edge_index, sign) forming closed loops, with intersection number A . B = +1
```

Requirements to validate on load:

- every edge belongs to exactly 2 faces (closed surface)
- `nV - nE + nF == 0` (Euler characteristic zero)
- face orientations are consistent (each edge appears with opposite sign in its two faces)
- each triangle satisfies the strict triangle inequality
- `A` and `B` are closed and their homology classes are independent

## 3. Incidence matrices

`d0` is `nE x nV`: row for edge `(i,j)` has `-1` at `i` and `+1` at `j`.

`d1` is `nF x nE`: row for face `(i,j,k)` has `+1` or `-1` for each of its three edges according to whether the edge orientation agrees with the face's boundary traversal.

Check: `d1 @ d0 == 0` exactly.

## 4. Per-triangle local geometry

For each face `t = (i,j,k)` with edge lengths `a = ell(jk)`, `b = ell(ki)`, `c = ell(ij)`:

Angles by the law of cosines,

    cos(alpha_i) = (b^2 + c^2 - a^2) / (2 b c)

and cyclically for `alpha_j`, `alpha_k`.

Area by Heron's formula with `s = (a+b+c)/2`,

    A_t = sqrt(s (s-a) (s-b) (s-c))

Local planar layout (intrinsic, no global embedding needed): place

    p_i = (0, 0)
    p_j = (c, 0)
    p_k = (b cos(alpha_i), b sin(alpha_i))

Use this frame for all per-face vector computations. Frames of different faces are never compared directly.

## 5. Cotangent weights

For interior edge `e` with opposite angles `alpha_e`, `beta_e` in its two adjacent faces:

    w_e = 0.5 * (cot(alpha_e) + cot(beta_e))

Set `M1 = diag(w_e)`.

Weights may be negative for non-Delaunay triangulations. This is permitted, but flag it: the construction is numerically stable only when the intrinsic Delaunay condition `alpha_e + beta_e <= pi` holds on every edge. Provide an optional intrinsic Delaunay edge-flip preprocessing pass.

## 6. Harmonic space

A 1-cochain `omega` in `R^nE` is harmonic iff it is closed and co-closed:

    d1 @ omega = 0                  (closed)
    d0.T @ M1 @ omega = 0           (co-closed)

Stack and take the nullspace:

    S = vstack([d1, d0.T @ M1])
    H = null_space(S)               # nE x 2

Assert `H.shape[1] == 2`. If not, the input is not a torus or the weights are degenerate.

Note `M0` is not needed: `delta_1 = M0^{-1} d0.T M1`, and `M0^{-1}` is invertible, so it does not change the kernel.

## 7. L2 inner product on 1-cochains

Use Whitney 1-forms. On face `t` with vertices `i,j,k`, the barycentric gradients in the local frame are

    grad_lambda_i = rot90(p_k - p_j) / (2 A_t)
    grad_lambda_j = rot90(p_i - p_k) / (2 A_t)
    grad_lambda_k = rot90(p_j - p_i) / (2 A_t)

where `rot90(x, y) = (-y, x)`. Verify `grad_lambda_i + grad_lambda_j + grad_lambda_k == 0`.

The Whitney interpolant of a 1-cochain, evaluated at the barycenter of `t`, is the constant vector

    W_t(omega) = (1/3) * sum over the three oriented edges (u,v) of t of
                 omega_{uv} * (grad_lambda_v - grad_lambda_u)

with each `omega_{uv}` taken with the sign relating the stored edge orientation to `(u,v)`.

The inner product is

    <omega, eta> = sum over faces t of  A_t * dot(W_t(omega), W_t(eta))

## 8. Complex structure J

Define `J` on the harmonic space by rotate-then-project:

1. `rho`: rotate each per-face vector by +90 degrees in its local frame — `W_t -> rot90(W_t)`.
2. Project the rotated field back onto `H`. Since the rotated field is generally not a harmonic cochain, do this as an L2-orthogonal projection.

Concretely, with `H = [h1, h2]` a basis of the harmonic space, build the 2x2 Gram matrix

    G[a][b] = <h_a, h_b>

and the 2x2 rotation-pairing matrix

    R[a][b] = sum over faces t of A_t * dot(rot90(W_t(h_a)), W_t(h_b))

Then the matrix of `J` in the basis `{h1, h2}` is

    J = G^{-1} @ R.T

Validate `J @ J ≈ -I`. Report the residual `||J@J + I||_F` as a discretization-error diagnostic; it goes to zero under refinement. Do not silently symmetrize or renormalize `J` — expose the residual.

## 9. Holomorphic line and period ratio

Complexify. The holomorphic line is the eigenspace of `J` with eigenvalue `-i` (convention: `star(dz) = -i dz`).

    eigenvalues, eigenvectors = eig(J)
    c = eigenvector for eigenvalue closest to -1j          # complex 2-vector
    omega = c[0] * h1 + c[1] * h2                          # complex 1-cochain

Periods are signed sums along the marked cycles:

    P_A = sum over (e, s) in A of s * omega[e]
    P_B = sum over (e, s) in B of s * omega[e]

    tau = P_B / P_A

Require `Im(tau) > 0`. If `Im(tau) < 0`, the surface orientation or the eigenvalue branch is flipped; take the conjugate eigenvector and recompute. If `|P_A|` is near zero, the marking is degenerate for this metric — swap the roles of `A` and `B` and report `-1/tau`.

## 10. Qubit state

    |psi> = (|0> + tau |1>) / sqrt(1 + |tau|^2)

Bloch vector, with `N = 1 + |tau|^2`:

    r_x = 2 * Re(tau) / N
    r_y = 2 * Im(tau) / N
    r_z = (1 - |tau|^2) / N

Density matrix `rho = 0.5 * (I + r . sigma)`. Assert `|r| == 1` to machine precision.

## 11. Metrics on the resulting state space

Two distinct metrics. Implement both as separate functions; do not conflate them.

Fubini-Study (distinguishability):

    ds^2_FS = |dtau|^2 / (1 + |tau|^2)^2

    d_FS(psi_1, psi_2) = arccos( |1 + conj(tau_1) tau_2| / sqrt((1+|tau_1|^2)(1+|tau_2|^2)) )

Weil-Petersson / Poincare (moduli distance between shapes):

    ds^2_WP = |dtau|^2 / (Im tau)^2

    d_WP(tau_1, tau_2) = arccosh( 1 + |tau_1 - tau_2|^2 / (2 Im(tau_1) Im(tau_2)) )

These are conformally equivalent but not isometric; curvatures are `+4` and `-1`.

## 12. Reference test cases

Flat torus `C / (Z + tau Z)`, unit square fundamental domain split by its diagonal, sides identified. The construction is exact on flat tori, so these are equality tests to numerical tolerance.

| Geometry | Expected tau | Expected Bloch vector |
|---|---|---|
| Square (unit square, no shear) | `i` | `(0, 1, 0)` |
| Rectangle, aspect ratio r | `i r` | `(0, 2r, 1-r^2) / (1+r^2)` |
| Sheared unit cell, shear s | `s + i` | `(2s, 2, -s^2) / (2 + s^2)` |
| Hexagonal | `exp(i pi / 3)` | `(1/2, sqrt(3)/2, 0)` |

Additional assertions:

- `dim(H) == 2` for every valid torus input
- `||J@J + I||_F` decreases monotonically under uniform refinement
- `tau` is invariant under uniform scaling of all edge lengths
- `tau` is invariant under refinement of the triangulation at fixed geometry (to tolerance)
- Modular transformations: relabeling cycles as `A' = B`, `B' = -A` maps `tau -> -1/tau`; `B' = A + B` maps `tau -> tau + 1`
- `|r| == 1` for all outputs

## 13. Degeneration behavior

As a cycle pinches (some `ell -> 0`, or aspect ratio `r -> 0` or `inf`):

- `Im(tau) -> 0` or `inf`
- `M1` becomes singular; cotangent weights diverge
- the state converges to `|0>` or `|1>` smoothly, with `d_FS` finite
- `d_WP` diverges logarithmically

Detect near-degeneracy via `cond(M1)` and the condition number of the 2x2 Gram matrix `G`. Emit a warning above a configurable threshold rather than failing; the state output remains valid.

## 14. Public API

```python
class SimplicialQubit:
    def __init__(self, vertices, edges, faces, lengths, cycle_A, cycle_B): ...

    def harmonic_basis(self) -> np.ndarray:      # nE x 2 real
    def complex_structure(self) -> np.ndarray:   # 2 x 2 real, J
    def j_residual(self) -> float:               # ||J@J + I||_F
    def holomorphic_form(self) -> np.ndarray:    # nE complex
    def periods(self) -> tuple[complex, complex] # (P_A, P_B)
    def tau(self) -> complex
    def state(self) -> np.ndarray                # 2-vector complex, normalized
    def bloch(self) -> np.ndarray                # 3-vector real
    def density_matrix(self) -> np.ndarray       # 2 x 2 complex

    @classmethod
    def flat_torus(cls, tau: complex, nx: int, ny: int) -> "SimplicialQubit": ...

def fubini_study_distance(q1, q2) -> float
def weil_petersson_distance(q1, q2) -> float
```

Dependencies: `numpy`, `scipy` (`scipy.linalg.null_space`, `scipy.linalg.eig`). No mesh library required.

## 15. Scope constraints

These are properties of the construction, not bugs to work around.

- Coverage is one open hemisphere of the Bloch sphere. `tau` ranges over the upper half plane, whose image under the Cayley map `tau -> (tau - i)/(tau + i)` is the open unit disk. The other hemisphere requires the opposite surface orientation, carried as a separate boolean.
- `tau` depends on the marking `(A, B)`. Changing the marking acts by `SL(2, Z)`. A canonical state requires the marking to be fixed and stored with the complex.
- The true moduli space is `H / SL(2, Z)`, an orbifold with a cusp, not `CP^1`. The map from geometry to state is onto a hemisphere only after fixing the marking.
- There is no action of `SU(2)` on edge lengths. Unitary gates have no realization as metric deformations. If gate simulation is needed, apply gates to the `state()` output as ordinary 2x2 matrices; do not attempt to pull them back to the complex.
- The construction is exact for flat tori and first-order accurate in mesh size otherwise.

## 16. Complex geometry

Lengths may be complex. Every formula of §4–§9 is taken over C: the principal
branch of acos in §4, the continuation branch (from the real reference) of the
square roots in §4 (Heron) and §9, the transpose (bilinear, not Hermitian)
pairing in §7 and §8, a complex null space in §6. Link phases are a pure gauge.
They enter through the twisted incidences of §3 and §6 (each edge value
carried to its cell's base vertex by the link), through the pairings of §7 and
§8 taken between the kernel and the dual kernel (the same construction under
the inverse links), and through the periods of §9, taken with parallel
transport from one common base point of the two cycles; together these leave
`tau`, the state and the coefficients in the period frame invariant. A
connection that is not a pure gauge (flux through a face, or holonomy around a
cycle) is refused by name. The eigenline of §9 is chosen by continuity from the real
reference, since `Im(tau) > 0` is not a criterion off the real locus.
