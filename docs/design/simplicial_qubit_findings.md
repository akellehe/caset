# Simplicial qubit — implementation notes and measurements (#955)

The theory is `simplicial_qubit_spec.md` (this directory). This note records
where the construction lives in the code, the two places the implementation
departs from the spec's letter and why, and the numbers the tests rest on.

## Where it lives

| Spec object | Code |
|---|---|
| triangulated torus, discrete metric | a `Spacetime` of dimension 2: complex edge lengths and link phases on `Edge` |
| `flat_torus(tau, nx, ny)` | `observables::SimplicialQubit::flatTorus` over `SimplicialProduct(PolygonCircle(nx), PolygonCircle(ny))` |
| validation on load (§2) | `Spacetime::getBoundary`, `ChainComplex::{eulerCharacteristic, bettiNumbers, dualComplexIsValid, fundamentalClass}`, `WhitneyMass` certificate |
| incidence matrices (§3) | `ChainComplex::boundaryMatrix`, canonical ascending-id orientation |
| harmonic space (§6) | `ChainHodge::harmonicChains(1)`: the exact zero mode, images = edge integrals, chains = `M_1` × images |
| L² inner product on 1-cochains (§7) | `WhitneyMass::assemble(K, s, 1)`; on the basis `ChainHodge::harmonicGram` |
| rotation pairing and `J` (§8) | `ChainComplex::cupProductForm(1)` (the pairing) and `J = G^{-1} R^T` in `SimplicialQubit::read` |
| periods, `tau` (§9) | signed walk sums over the marking (the `EdgeLoop` sign rule) in `SimplicialQubit::read` |
| state, Bloch, density (§10) | `SimplicialQubit::{stateOf, blochOf, densityOf}` |
| the two metrics (§11) | `SimplicialQubit::{fubiniStudyDistance, weilPeterssonDistance}` |
| degeneration (§13) | `SimplicialQubitRead::{metricCondition, gramCondition, nearDegenerate, warning}` |
| phases on | `CovariantChainHodge::harmonicChains(1)`: the twisted zero mode; rank 2 certifies a pure gauge |

## Two departures from the spec's letter

**Whitney mass for harmonicity, not diagonal cotangent weights (§5).** The
spec takes the harmonic space with the DEC weights `diag(w_e)` and the inner
product with Whitney forms. The pencil already carries one consistent metric,
the Whitney mass `M_1`, positive definite for every nondegenerate real metric
and complex symmetric for complex lengths; using it for both keeps the zero
mode and the Gram in one convention and removes the stability caveat the
optional Delaunay flip pass addresses (negative cotangent weights). The DEC
route exists as `HodgeLaplacian::MetricSource::DiagonalWeights` and is not
wired here.

**The rotation pairing is integrated exactly (§8).** "Rotate each per-face
vector by 90° and project back" is, with the integrals done exactly,
`R_ab = ∫ W(z_a) ∧ W(z_b)`, and on closed cochains that equals the cup-product
pairing of the classes ⟨z_a ∪ z_b, [K]⟩ — a metric-free integer-combinatorial
form. So `R` carries no discretization error at all; the geometry enters only
through the Gram. A scratch check of the per-triangle Whitney wedge gave the
coefficient matrix (1/6)[[0,1,1],[−1,0,−1],[−1,1,0]] on edges (01,12,02) and
reproduced the intersection number +1.000 on the four reference tori, the same
number the cup product gives. For a 2×2 antisymmetric `R` the algebra closes:
`J² = −(r²/det G)·I` with `r = R_01`, so the spec's residual `‖J² + I‖_F =
√2·|1 − r²/det G|` is the Riemann-bilinear defect of the discrete harmonic
forms, and in the period-dual basis `τ = (−g_12 + i√det G)/g_22`.

## Measurements (4 × 4 grid unless noted)

Reference table (spec §12), exact to rounding:

| torus | \|τ̂ − τ\| | ‖J² + I‖_F | A·B | cond M₁ | cond G |
|---|---|---|---|---|---|
| square, τ = i | 5.5e-17 | 1.5e-15 | +1.000000000000 | 3.00 | 1.29 |
| rectangle, τ = 2i | 2.0e-15 | 6.8e-15 | +1.000000000000 | 6.17 | 4.11 |
| shear, τ = 0.3 + i | 3.7e-16 | 1.2e-15 | +1.000000000000 | 5.03 | 2.33 |
| hexagonal, τ = e^{iπ/3} | 4.5e-16 | 1.2e-15 | +1.000000000000 | 8.00 | 3.86 |

Conformally flat square torus, lengths scaled by exp(φ) at edge midpoints with
φ = 0.3 sin(2πx) cos(2πy) (conformal structure still τ = i):

| N | n₁ | ‖J² + I‖_F | \|τ̂ − i\| | rate (residual / τ) |
|---|---|---|---|---|
| 4 | 48 | 2.92e-2 | 1.94e-2 | — |
| 8 | 192 | 1.59e-2 | 3.76e-4 | 0.87 / 5.69 |
| 16 | 768 | 4.66e-3 | 2.81e-5 | 1.78 / 3.75 |
| 32 | 3072 | 1.21e-3 | 3.52e-6 | 1.94 / 3.00 |

The residual settles to second order in the mesh size and τ to third; the spec
promises first order. Above the dense crossover (n₁ ≥ 512) the kernel comes
from the sparse rank-revealing QR (gap unmeasured) and the metric condition
number is not reported.

Phases on: a phase of 0.7 on one edge of the square torus drops the twisted
harmonic rank to 0 and the read refuses by name; a pure-gauge assignment
φ_e = g(target) − g(source) leaves τ unchanged to 1e-10 with rank 2; a flat
holonomy across the seam refuses.

Pinching, τ = 0.05 i: τ̂ = 0.05 i to 1e-9, Bloch (0, 0.0998, 0.995) (towards
|0⟩), cond M₁ = 5.3e2, cond G = 4.1e2, d_FS(τ̂, i) = 0.735 finite while
d_WP(τ̂, i) = 3.00 and growing like −log Im τ.

## Out of scope here

Relaxing a torus toward a target τ (state → geometry by descent) is the
cobordism engine's job; the representation is what it will be read against.
Gates act on `state()`; they have no realization as metric deformations.
