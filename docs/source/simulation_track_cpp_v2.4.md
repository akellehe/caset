# Simulation Track: Building a Hydrogen Atom from Simplicial Geometry (C++ v2.4)

## Companion to the Main Curriculum

This document details the software architecture, data structures, numerical methods, and integration milestones for the simulation codebase that culminates in constructing a hydrogen atom as a simplicial complex with torsion. All code is C++20, built with CMake, using Eigen for linear algebra and CGAL for mesh generation.

---

## Architecture Overview

The codebase consists of 8 modules. Each module's output is a required input to later modules. The dependency graph is:

```
S1 geometry ──→ S2 weyl_chamber ──→ S3 composition ──→ S4 regge
                                          │                  │
                                          ▼                  ▼
                                    S5 causal ──────→ S6 gauge
                                                         │
                                                         ▼
                                                    S7 matter
                                                         │
                                                         ▼
                                                    S8 hydrogen
```

**Language choice: C++ throughout.** The computational bottleneck is the variational solver for the Einstein–Cartan action ($\sim 9N$ DOF, many L-BFGS iterations each requiring a full gradient evaluation). C++ gives you direct control over memory layout (SoA vs AoS for cache performance on the simplex arrays), easy integration with CGAL for mesh generation, Eigen's expression templates for zero-overhead linear algebra, and CppAD or Enzyme for automatic differentiation without the overhead of a tracing JIT. The early algebraic modules (S1–S3) are equally fast in any language, but writing them in C++ from the start avoids a rewrite when the solver arrives.

### Project Layout

```
simplicial-hydrogen/
├── CMakeLists.txt
├── conanfile.txt              # or vcpkg.json
├── include/
│   └── simhydro/
│       ├── geometry.hpp       # S1: eigenphases, edge lengths, volumes
│       ├── weyl_chamber.hpp   # S2: KAK decomposition
│       ├── composition.hpp    # S3: gate composition, torsion
│       ├── regge.hpp          # S4: Regge action, deficit angles
│       ├── causal.hpp         # S5: causal sets, 4-simplex
│       ├── gauge.hpp          # S6: holonomies, chiral split
│       ├── matter.hpp         # S7: fermion sources, EC solver
│       ├── hydrogen.hpp       # S8: Coulomb solver, ground state
│       ├── types.hpp          # core data structures
│       └── linalg.hpp         # SU(4) utilities, matrix exponential
├── src/
│   ├── geometry.cpp
│   ├── weyl_chamber.cpp
│   ├── ...
│   └── main.cpp               # driver / integration tests
├── tests/
│   ├── test_geometry.cpp
│   ├── test_composition.cpp
│   ├── ...
│   └── CMakeLists.txt          # Google Test
├── bench/
│   └── bench_solver.cpp        # Google Benchmark
└── scripts/
    └── plot.py                 # matplotlib for post-processing VTK output
```

---

## Build System

```cmake
# CMakeLists.txt (top-level)
cmake_minimum_required(VERSION 3.22)
project(simplicial_hydrogen LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED)
find_package(CGAL REQUIRED)
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)            # for readable output
# Phase 7+:
find_package(CppAD REQUIRED)          # automatic differentiation
# or: use Enzyme (LLVM plugin, no find_package — set via compiler flags)

add_library(simhydro
    src/geometry.cpp
    src/weyl_chamber.cpp
    src/composition.cpp
    src/regge.cpp
    src/causal.cpp
    src/gauge.cpp
    src/matter.cpp
    src/hydrogen.cpp
)
target_include_directories(simhydro PUBLIC include)
target_link_libraries(simhydro PUBLIC Eigen3::Eigen CGAL::CGAL fmt::fmt)

add_executable(run_sim src/main.cpp)
target_link_libraries(run_sim PRIVATE simhydro)

enable_testing()
add_subdirectory(tests)
```

---

## Typedefs and Constants (`types.hpp`)

```cpp
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <array>
#include <vector>
#include <optional>
#include <cstdint>

namespace simhydro {

// ── Scalar and matrix types ──────────────────────────────────────────────
using Real    = double;
using Complex = std::complex<double>;

using Vec3  = Eigen::Vector3d;                       // spatial vectors
using Vec4  = Eigen::Vector4d;                       // spacetime vectors
using Mat3  = Eigen::Matrix3d;
using Mat4  = Eigen::Matrix4d;
using Mat4c = Eigen::Matrix4cd;                      // 4×4 complex (SU(4) elements)
using Vec4c = Eigen::Vector4cd;

// Fixed-size arrays matching the combinatorics of a tetrahedron
using Edges6   = Eigen::Matrix<Real, 6, 1>;          // 6 squared edge lengths
using Areas4   = Eigen::Matrix<Real, 4, 1>;          // 4 face areas
using Verts4x3 = Eigen::Matrix<Real, 4, 3, Eigen::RowMajor>;  // 4 vertices in R^3

// ── Physical constants (natural units: ℏ = c = 1 unless otherwise noted) ─
inline constexpr Real PI       = 3.14159265358979323846;
inline constexpr Real SQRT3    = 1.73205080756887729353;
inline constexpr Real VOL_COEFF = 16.0 * SQRT3 / 27.0;   // 16√3/27 ≈ 1.0264
inline constexpr Real ALPHA_EM = 1.0 / 137.035999;        // fine-structure constant
inline constexpr Real M_ELECTRON_EV = 0.51099895e6;       // electron mass in eV

// ── Index type ───────────────────────────────────────────────────────────
using Id = std::int32_t;
inline constexpr Id NULL_ID = -1;

} // namespace simhydro
```

---

## Core Data Structures (`types.hpp`, continued)

These structs persist across all modules. Define them in Week 4 and extend as fields are populated by later modules.

### `Tetrahedron`

A single 3-simplex, the atomic unit of a spatial slice.

```cpp
struct Tetrahedron {
    Id       id;
    Vec3     c;             // Weyl parameters (c1, c2, c3), 0 ≤ c3 ≤ c2 ≤ c1 ≤ π/2
    Verts4x3 vertices;      // eigenphase-embedded 3D positions p_k = (λ_k/√3) n_k
    Edges6   edges_sq;      // squared edge lengths D_jk from Table 1
    Areas4   face_areas;    // areas of the four triangular faces
    Real     volume;        // V = (16√3/27) c1 c2 c3

    // Construct from Weyl parameters; compute all derived quantities.
    static Tetrahedron from_weyl(Id id, Real c1, Real c2, Real c3);
};
```

**Key relations implemented here (from paper §3–5):**

Eigenphases: $\lambda_k = \vec{n}_k \cdot \vec{c}$, where $\vec{n}_k$ are the four normals of the reference tetrahedron.

Squared edge lengths (Table 1 of the paper):
$$D_{jk} = \frac{2}{3}\sum_{i=1}^{3}\left(\lambda_j^{(i)} - \lambda_k^{(i)}\right)^2$$
where $\lambda_k^{(i)}$ is the $i$-th component of the $k$-th eigenphase vector.

Volume: $V = \frac{16\sqrt{3}}{27}c_1 c_2 c_3$.

Face areas: four values from Thm 5.1, each a function of $c_i$ and $\lambda_k$.

### `Face`

A triangular face shared between two tetrahedra (or on the boundary).

```cpp
struct Face {
    Id       id;
    Id       tet_A;                    // id of tetrahedron A
    Id       tet_B;                    // id of tetrahedron B (NULL_ID if boundary)
    std::array<Id, 3> vertex_ids;      // three shared vertex indices
    std::array<int, 3> sigma;          // permutation induced by W on face vertices
    Mat4c    W;                        // connection W ∈ SU(2)×SU(2) ⊂ SU(4)
    Vec3     torsion;                  // T_jk = D_jk(c^A) − D_{σ(j)σ(k)}(c^B)

    // Populated in S6:
    Eigen::Matrix<Real, 8, 1> torsion_su3;  // projection onto su(3) ⊂ p
    Real     torsion_u1 = 0.0;               // projection onto u(1)_{B−L} ⊂ p

    bool is_boundary() const { return tet_B == NULL_ID; }
};
```

**Key equation (paper Eq. 18):**
$$\mathcal{T}_{jk} = D_{jk}(c^A) - D_{\sigma(j)\sigma(k)}(c^B)$$

The face-matching condition $\mathcal{T}_{jk} = 0$ for all three shared edges $(j,k)$ is 3 equations in the 6 continuous parameters of $W$, leaving 3 residual DOF (torsion degrees of freedom).

### `Hinge`

An edge (in 3D) or triangle (in 4D) where curvature concentrates.

```cpp
struct Hinge {
    Id       id;
    std::vector<Id> incident_tets;     // tetrahedra sharing this hinge
    Real     area;                     // A_h (edge length in 3D, triangle area in 4D)
    Real     deficit_angle;            // ε_h = 2π − Σ dihedral angles
};
```

### `SimplicialComplex`

The central data structure. A collection of tetrahedra glued along shared faces.

```cpp
class SimplicialComplex {
public:
    // ── Topology ──────────────────────────────────────────────────────
    std::vector<Tetrahedron>   tets;
    std::vector<Face>          faces;
    std::vector<Vec3>          vertices;       // global vertex positions (3D)
    std::vector<Hinge>         hinges;

    // Adjacency: tet_id → list of neighbouring tet_ids
    std::vector<std::vector<Id>> adjacency;

    // ── Gauge fields (populated by S6+) ──────────────────────────────
    std::vector<Mat4c>         holonomies;     // one per hinge

    // ── Queries ──────────────────────────────────────────────────────
    const Tetrahedron& tet(Id i) const { return tets[i]; }
    const Face&        face(Id i) const { return faces[i]; }
    std::span<const Id> neighbours(Id tet_id) const;
    std::vector<Id>     faces_of_tet(Id tet_id) const;

    // ── Physics ──────────────────────────────────────────────────────
    Real regge_action(Real G) const;
    Real ec_action(Real G) const;

    // ── Gradients (see Autodiff section) ─────────────────────────────
    // Returns ∂S/∂c_i for each interior tet, ∂S/∂W for each interior face
    struct Gradient {
        std::vector<Vec3>  dc;       // one per interior tet
        std::vector<Mat4c> dW;       // one per interior face
    };
    Gradient gradient(Real G) const;

    // ── Construction ─────────────────────────────────────────────────
    void build_hinges();              // compute hinge topology from face/tet data
    void compute_deficit_angles();    // populate hinge deficit angles
    void compute_holonomies();        // populate holonomies (S6+)
};
```

### `FourSimplex` (added in S5)

A 4-simplex built from a causal link.

```cpp
struct FourSimplex {
    Id       id;
    Vec4     past_apex;                // (x, y, z, t) of past vertex
    Vec4     future_apex;              // (x, y, z, t) of future vertex
    Id       shared_face_id;           // the triangular face connecting the two 3-simplices
    Eigen::Matrix<Real, 10, 1> edge_lengths_sq;  // all 10 squared edge lengths
    std::array<int, 4> signature;      // signs of Gram matrix eigenvalues (should be −,+,+,+)
};
```

### `FermionSource` (added in S7)

A localised torsion excitation with definite quantum numbers.

```cpp
enum class Particle { u_L, d_L, u_R, d_R, e_L, nu_L, e_R, nu_R };
enum class Chirality { L, R };

struct FermionSource {
    Particle particle;
    int      generation;       // 1, 2, or 3

    // Derived quantum numbers (from Pati–Salam branching)
    int      su3_dim;          // 3, -3 (antifundamental), or 1
    Real     B_minus_L;        // +1/3 (quarks) or −1 (leptons)
    Real     T3L;              // weak isospin third component
    Real     Q_em;             // electric charge = T3 + Y/2

    // Torsion prescription
    Eigen::Matrix<Real, 9, 1> torsion_direction;  // unit vector in p ≅ su(3) ⊕ u(1)
    Real     torsion_magnitude;                    // |T| at the source faces

    // Convenience constructors
    static FermionSource electron(int gen = 1);
    static FermionSource up_quark(int color, int gen = 1);
    static FermionSource down_quark(int color, int gen = 1);
    static FermionSource neutrino(int gen = 1);

    Eigen::Matrix<Real, 9, 1> torsion_tensor() const;
};
```

---

## SU(4) Linear Algebra Utilities (`linalg.hpp`)

The entire project rests on $4\times 4$ complex unitary matrices. Eigen handles the storage; you supply the physics.

```cpp
#pragma once
#include "types.hpp"

namespace simhydro::linalg {

// ── Pauli matrices ───────────────────────────────────────────────────
inline const Eigen::Matrix2cd sigma_x = (Eigen::Matrix2cd() <<
    0, 1, 1, 0).finished();
inline const Eigen::Matrix2cd sigma_y = (Eigen::Matrix2cd() <<
    0, Complex(0,-1), Complex(0,1), 0).finished();
inline const Eigen::Matrix2cd sigma_z = (Eigen::Matrix2cd() <<
    1, 0, 0, -1).finished();
inline const Eigen::Matrix2cd I2 = Eigen::Matrix2cd::Identity();

// ── Tensor product (Kronecker product) ───────────────────────────────
// Eigen doesn't have a built-in kron, so provide one for 2×2 → 4×4.
Mat4c kron(const Eigen::Matrix2cd& A, const Eigen::Matrix2cd& B);

// ── su(4) basis ──────────────────────────────────────────────────────
// Returns 15 traceless anti-Hermitian generators {iλ_a} normalised as
// tr(T_a T_b) = −δ_{ab}/2.
std::array<Mat4c, 15> su4_basis();

// Split into k (6-dim) and p (9-dim) under the Cartan decomposition
// SU(4) / (SU(2) × SU(2)).
struct CartanDecomp {
    std::array<Mat4c, 6> k_basis;    // su(2)_L ⊕ su(2)_R
    std::array<Mat4c, 3> a_basis;    // maximal abelian in p
    std::array<Mat4c, 6> m_basis;    // remaining p directions
};
CartanDecomp cartan_decomposition();

// ── Matrix exponential ───────────────────────────────────────────────
// For 4×4 Hermitian or anti-Hermitian matrices. Uses Eigen's
// SelfAdjointEigenSolver when the input is Hermitian; otherwise falls
// back to Schur decomposition.
Mat4c matrix_exp(const Mat4c& A);

// ── SU(4) random Haar-distributed matrix ─────────────────────────────
// QR of a complex Gaussian matrix, phase-corrected.
Mat4c random_su4(std::mt19937_64& rng);

// ── Magic basis matrix Q ─────────────────────────────────────────────
// Used in the Makhlin invariants and KAK decomposition.
const Mat4c& magic_basis();

// ── Projections ──────────────────────────────────────────────────────
// Project X ∈ su(4) onto k or p.
Mat4c project_k(const Mat4c& X);
Mat4c project_p(const Mat4c& X);

// Project T ∈ p onto su(3) (8 components) and u(1)_{B−L} (1 component).
Eigen::Matrix<Real, 8, 1> project_su3(const Eigen::Matrix<Real, 9, 1>& T_in_p);
Real project_u1(const Eigen::Matrix<Real, 9, 1>& T_in_p);

} // namespace simhydro::linalg
```

**Implementation notes:**
- `kron`: for $2\times 2$ inputs this is a $4\times 4$ output. Write it explicitly (16 entries) — no loops, no allocations. The compiler will vectorise.
- `matrix_exp`: for anti-Hermitian $A = iH$ with $H$ Hermitian, diagonalise $H$ via `SelfAdjointEigenSolver`, exponentiate the (real) eigenvalues, reconstruct. This is $O(n^3)$ with $n = 4$, so ~nanoseconds per call.
- `random_su4`: draw a $4\times 4$ complex Gaussian, QR-factorise, fix phases so $\det = 1$. Use `std::normal_distribution<double>` for the entries.

---

## Module S1: `geometry` (Weeks 1–4)

### What You Build
Eigenphase embedding, edge lengths, volumes, face areas, and the Cayley–Menger determinant.

### Key Functions

```cpp
namespace simhydro::geometry {

// Eigenphases λ_k = n_k · c, with Σ λ_k = 0.
Eigen::Vector4d eigenphases(const Vec3& c);

// Embed in R^3: p_k = (λ_k / √3) n_k.
Verts4x3 embed(const Vec3& c);

// Six squared edge lengths D_jk from Table 1 of the paper.
// Ordering: (01, 02, 03, 12, 13, 23).
Edges6 edge_lengths_sq(const Vec3& c);

// Volume via closed form.
Real volume(const Vec3& c);

// Cayley–Menger determinant (5×5). Should equal 288 V².
Real cayley_menger(const Vec3& c);

// Four face areas from Thm 5.1.
Areas4 face_areas(const Vec3& c);

// Six dihedral angles. Sum should equal 2π (Descartes).
Edges6 dihedral_angles(const Vec3& c);

// Construct a Tetrahedron with all fields populated.
Tetrahedron make_tet(Id id, Real c1, Real c2, Real c3);

} // namespace simhydro::geometry
```

### Test Battery (Google Test)

```cpp
TEST(Geometry, VolumeClosedFormMatchesTripleProduct) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.01, PI / 2.0);
    for (int i = 0; i < 10'000; ++i) {
        Vec3 c = random_weyl_triple(rng, dist);   // sample with c3 ≤ c2 ≤ c1
        Real v_closed = geometry::volume(c);
        Verts4x3 verts = geometry::embed(c);
        Real v_triple = std::abs((verts.row(1) - verts.row(0))
            .cross(verts.row(2) - verts.row(0))
            .dot(verts.row(3) - verts.row(0))) / 6.0;
        EXPECT_NEAR(v_closed, v_triple, 1e-12 * v_closed);
    }
}

TEST(Geometry, CayleyMengerEquals288Vsq) { /* ... */ }
TEST(Geometry, AreaSumRule)              { /* ... */ }
TEST(Geometry, DegenerateCases)          { /* c3=0 → V=0; CNOT → flat */ }
```

### Milestone S1
You can take any point in the Weyl chamber and produce a fully characterised tetrahedron with all metric data. This is the geometric atom of the entire simulation.

---

## Module S2: `weyl_chamber` (Weeks 13–16)

### What You Build
KAK decomposition of arbitrary $U \in SU(4)$ and the Weyl chamber characterisation.

### KAK Decomposition Algorithm

Given $U \in SU(4)$:
1. Compute the Makhlin matrix: $M = Q^\dagger U^T Q \cdot U$, where $Q$ is the magic basis matrix
2. Diagonalise $M$: eigenvalues are $\{e^{-2i\lambda_k}\}$
3. Extract eigenphases: $\lambda_k = \vec{n}_k \cdot \vec{c}$
4. Solve the $4\times 3$ linear system $N \vec{c} = \vec{\lambda}$ (least-squares via `Eigen::JacobiSVD`)
5. Fold into the Weyl chamber: apply Weyl group elements until $0 \leq c_3 \leq c_2 \leq c_1 \leq \pi/2$ and $c_1 + c_2 \leq \pi/2$
6. Extract $K_1, K_2 \in SU(2) \otimes SU(2)$

```cpp
namespace simhydro::weyl {

struct KAKResult {
    Mat4c K1;           // left local unitary (SU(2) ⊗ SU(2))
    Vec3  c;            // Weyl parameters
    Mat4c K2;           // right local unitary
};

KAKResult kak_decompose(const Mat4c& U);
Mat4c     kak_reconstruct(const KAKResult& kak);

// Entangling power: e_p ∈ [0, 2/9].
Real entangling_power(const Vec3& c);

// Check if a point is inside the Weyl chamber.
bool in_weyl_chamber(const Vec3& c);

// Fold an arbitrary (c1, c2, c3) into the Weyl chamber via Weyl group action.
Vec3 fold_to_chamber(Vec3 c);

} // namespace simhydro::weyl
```

### Test Battery
1. Round-trip $10^3$ random $SU(4)$ matrices: $\|U - U'\|_F < 10^{-10}$
2. Known gates: CNOT→$(\pi/4, 0, 0)$, SWAP→$(\pi/4, \pi/4, \pi/4)$, identity→$(0, 0, 0)$
3. All decomposed $\vec{c}$ satisfy the Weyl chamber inequalities
4. Entangling power matches literature for standard gates

### Milestone S2
You can decompose any 2-qubit gate into its geometric content (a tetrahedron) plus local frames.

---

## Module S3: `composition` (Weeks 19–22)

### What You Build
The gate composition map, torsion computation, and multi-simplex assembly.

### Master Spectral Formula (Thm 9.1)

Given two non-local cores $N_1 = \exp(-i\vec{c}^{(1)}\cdot\vec{H})$, $N_2 = \exp(-i\vec{c}^{(2)}\cdot\vec{H})$, and connection $W \in SU(2)^2$:

1. Compute diagonal matrices $\Phi_i = \text{diag}(e^{-2i\lambda_k^{(i)}})$ in the Bell basis
2. Compute $R = Q^\dagger W Q \in SO(4)$ (orthogonal part in the Bell basis)
3. Form $B = \Phi_1 R^T \Phi_2 R$
4. Diagonalise $B$: eigenvalues $\{e^{-2i\lambda'_k}\}$ give the composite eigenphases
5. Solve for $\vec{c}'$, fold into $\mathcal{W}$

### Key Functions

```cpp
namespace simhydro::composition {

// Composite Weyl parameters from the spectral formula.
Vec3 compose(const Vec3& c1, const Vec3& c2, const Mat4c& W);

// 4×4 doubly stochastic mixing matrix P_{jl} = R_{lj}^2.
Mat4 mixing_matrix(const Mat4& R);

// Weyl group action matrix for permutation σ ∈ S_4.
Mat3 weyl_group_action(const std::array<int, 4>& sigma);

// Shape mismatch tensor (torsion) at a shared face.
Vec3 torsion(const Tetrahedron& A, const Tetrahedron& B, const Face& f);

// Find W such that T_{jk} = 0 (3 eqns, 6 unknowns in W ∈ SU(2)²).
// Uses Newton's method. Returns the connection and convergence info.
struct TorsionFreeSolution {
    Mat4c W;
    Real  residual;
    int   iterations;
};
TorsionFreeSolution solve_torsion_free(const Vec3& cA, const Vec3& cB,
                                        Real tol = 1e-12, int max_iter = 50);

// Build a SimplicialComplex from a list of tetrahedra and face connectivity.
SimplicialComplex build_complex(std::vector<Tetrahedron> tets,
                                 std::vector<Face> faces,
                                 std::vector<Vec3> vertices);

} // namespace simhydro::composition
```

### Torsion-Free Solver Detail

Parameterise $W = \exp(i\vec\alpha\cdot\vec\sigma) \otimes \exp(i\vec\beta\cdot\vec\sigma)$ with $\vec\alpha, \vec\beta \in \mathbb{R}^3$. Define $F : \mathbb{R}^6 \to \mathbb{R}^3$ by $F(\vec\alpha, \vec\beta) = \vec{\mathcal{T}}(c^A, c^B, W(\vec\alpha, \vec\beta))$. Newton's method on the underdetermined system: compute the $3\times 6$ Jacobian $J = \partial F/\partial(\vec\alpha, \vec\beta)$ by finite differences (6 evaluations per step), then update $(\vec\alpha, \vec\beta) \mathrel{-}= J^T(JJ^T)^{-1}F$ (minimum-norm step). Converges in 3–5 iterations.

### Test Battery
1. Permutation composition for all 24 elements of $S_4$
2. Round-trip: compose, then KAK-decompose; parameters match $\vec{c}'$
3. Torsion-free: $10^2$ random $(c^A, c^B)$ pairs → $\|\vec{\mathcal{T}}\| < 10^{-12}$
4. Glue 5 tetrahedra; verify all face adjacencies are consistent

### Milestone S3
You can build multi-simplex spatial complexes, compute torsion at every face, and solve for torsion-free connections. You now have a discretised 3-manifold.

---

## Module S4: `regge` (Weeks 23–28)

### What You Build
The Regge action, deficit angles, the Einstein–Cartan extension, and the variational solver.

### Deficit Angles

At a hinge $h$ (an edge in 3D, a triangle in 4D), the deficit angle is:
$$\epsilon_h = 2\pi - \sum_{\text{tets sharing } h} \theta_h^{(\text{tet})}$$

### Regge and Einstein–Cartan Actions

$$S_R = \frac{1}{8\pi G}\sum_h A_h \epsilon_h$$

$$S_{\text{EC}} = \frac{1}{8\pi G}\sum_h A_h(c)\,\epsilon_h(c,W) + \frac{1}{2}\sum_f \mathcal{T}_{jk}(c,W)\,\Sigma_f^{jk}$$

Variation w.r.t. $W$ at fixed $c$ → torsion-free condition. Variation w.r.t. $c$ at fixed $W$ → discrete Einstein equation.

### The Variational Solver (Critical Infrastructure for S7–S8)

**This is the single most important piece of code in the entire project.** Every later module depends on it.

```cpp
namespace simhydro::regge {

struct SolverOptions {
    Real  G         = 1.0;         // Newton's constant (set to 1 in Planck units)
    Real  tol       = 1e-8;        // gradient norm convergence criterion
    int   max_iter  = 1000;
    enum class Method { LBFGS, CG, NEWTON } method = Method::LBFGS;
    int   lbfgs_m   = 20;         // L-BFGS memory depth
};

struct SolverResult {
    SimplicialComplex solution;
    Real   final_action;
    Real   final_grad_norm;
    int    iterations;
    bool   converged;
};

// Minimise S_EC over interior Weyl params and connections.
//
// DOF count per interior simplex:
//   3 (Weyl params) + 6 per interior face (connection params)
// Total DOF ≈ 9 × N_interior_simplices
//
// boundary_tets: indices of tetrahedra with fixed Weyl params
// boundary_faces: indices of faces with fixed connections
// sources: torsion sources (empty for vacuum)
SolverResult solve_field_equations(
    SimplicialComplex complex,
    const std::vector<Id>& boundary_tets,
    const std::vector<Id>& boundary_faces,
    const std::vector<std::pair<Id, FermionSource>>& sources,
    const SolverOptions& opts = {}
);

} // namespace simhydro::regge
```

### Gradient Computation — Three Strategies

**Strategy 1: Finite differences (Phase 4, prototyping).**

$\partial S/\partial c_i \approx [S(c_i + h) - S(c_i - h)]/(2h)$ with $h \sim 10^{-7}$. Each gradient requires $2 \times N_{\text{DOF}}$ action evaluations. Fine for $N < 100$.

```cpp
// Finite-difference gradient (fallback)
SimplicialComplex::Gradient gradient_fd(const SimplicialComplex& K, Real G, Real h = 1e-7);
```

**Strategy 2: CppAD automatic differentiation (Phase 7+, recommended).**

CppAD (or its younger sibling `CppADCodeGen`) provides forward- and reverse-mode AD for C++. The workflow:

1. Rewrite the action evaluation to operate on `CppAD::AD<double>` scalars instead of `double`
2. Trace the computation graph once (tape recording)
3. Evaluate gradients via reverse mode in $O(1)\times$ the cost of a single action evaluation

```cpp
#include <cppad/cppad.hpp>

using ADReal = CppAD::AD<double>;

// Templated action evaluation (works for both double and AD<double>)
template<typename Scalar>
Scalar ec_action_impl(const std::vector<Scalar>& dof_vector,
                       const SimplicialComplex& topology,
                       Scalar G);

// Build the tape and return a function object for gradient evaluation
class ECActionAD {
public:
    ECActionAD(const SimplicialComplex& topology, Real G);

    // Evaluate action and gradient at the given DOF vector.
    // Returns (action, gradient) where gradient.size() == dof_vector.size().
    std::pair<Real, std::vector<Real>> evaluate(const std::vector<Real>& dof_vector) const;

private:
    CppAD::ADFun<Real> tape_;
    size_t n_dof_;
};
```

**Alternative: Enzyme (LLVM compiler plugin).**

Enzyme differentiates native C++ at the LLVM IR level. No code changes needed — annotate the function and compile with the Enzyme pass. Potentially faster than CppAD (no tape overhead), but requires LLVM toolchain setup.

```cpp
// With Enzyme, you just write the plain action function:
double ec_action(const double* dof, int n_dof, /* topology */ ...);

// And request its gradient at compile time:
// __enzyme_autodiff(ec_action, enzyme_dup, dof, grad, enzyme_const, n_dof, ...);
```

**Strategy 3: Analytic Jacobian (optional, for maximum speed).**

Hand-derive $\partial A_h/\partial c_i$, $\partial\epsilon_h/\partial c_i$, $\partial\epsilon_h/\partial W$. The eigenphase formulae make this tractable but tedious. Only pursue if profiling shows AD is the bottleneck.

### L-BFGS Implementation

Use an existing C++ L-BFGS library rather than writing your own:

- **LBFGSpp** (header-only, Eigen-based): `#include <LBFGSpp/LBFGS.h>`. Directly compatible with the Eigen vectors used throughout the project.
- **libLBFGS** (C library by Naoaki Okazaki): well-tested, slightly more setup.

```cpp
#include <LBFGSpp/LBFGS.h>

// Wrap the EC action + CppAD gradient into LBFGSpp's interface
class ECObjective {
    ECActionAD ad_;
public:
    ECObjective(const SimplicialComplex& K, Real G) : ad_(K, G) {}

    Real operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        std::vector<Real> xv(x.data(), x.data() + x.size());
        auto [action, gv] = ad_.evaluate(xv);
        grad = Eigen::Map<const Eigen::VectorXd>(gv.data(), gv.size());
        return action;
    }
};
```

### Convergence Verification (CMS Theorem)

**2D test (Week 24):** Triangulate $S^2$ with $N = 20, 80, 320, 1280$ triangles (icosahedral refinement). Verify $\sum_v A_v \epsilon_v \to 4\pi$ at rate $O(N^{-1}) \sim O(h^2)$.

**3D test (Week 26):** Triangulate the round $S^3$ using CGAL's `Mesh_3` module. Fill tetrahedra with eigenphase-embedded data. Verify Regge action convergence at $O(h^2)$.

### Fatness Bound

CMS convergence requires the fatness $\eta = V/\ell_{\max}^n \geq \kappa > 0$. For eigenphase-embedded tetrahedra:
$$\eta(c_1, c_2, c_3) = \frac{(16\sqrt{3}/27)\,c_1 c_2 c_3}{\max_{jk}(D_{jk})^{3/2}}$$

The degeneration boundary ($\eta = 0$) is at $c_3 = 0$. For the hydrogen atom, fatness must hold everywhere in the mesh — this constrains the Weyl parameters away from degenerate configurations.

### Milestone S4
You can compute the Regge action on any simplicial complex, verify CMS convergence, and solve the vacuum field equations by minimising $S_{\text{EC}}$. The variational solver is operational.

**Berry curvature interpretation.** The deficit angle $\epsilon_h$ is the discrete Berry curvature of the Levi-Civita connection: parallel-transporting a vector around a loop encircling hinge $h$ rotates it by $\epsilon_h$. The Regge action $\sum_h A_h\epsilon_h$ is the gravitational Wilson action. Verify this explicitly by implementing:

```cpp
namespace simhydro::regge {

// Parallel-transport a vector around a hinge using face-to-face rotation matrices.
// Returns the total rotation angle (should equal deficit_angle to machine precision).
Real berry_phase_around_hinge(const SimplicialComplex& K, Id hinge_id, const Vec3& initial_vec);

} // namespace simhydro::regge
```

Test on $S^2$: the Berry phase around each vertex should equal the deficit angle, and $\sum_v \epsilon_v = 4\pi$ is the discrete Chern number (= Euler characteristic) of the bundle.

---

## Module S5: `causal` (Weeks 29–32)

### What You Build
Causal structure from quantum circuits, 4-simplex construction, Lorentzian edge lengths.

```cpp
namespace simhydro::causal {

struct Gate {
    Id   id;
    std::array<int, 2> qubits;    // the two qubits this gate acts on
    int  time_step;                // discrete time ordering
    Vec3 c;                        // Weyl parameters
};

struct CausalSet {
    std::vector<Gate> gates;
    std::vector<std::pair<Id, Id>> links;  // causal links (immediate successors)

    static CausalSet from_circuit(const std::vector<Gate>& circuit);
    int  myrheim_meyer_dimension() const;  // should give ~4
    Real proper_time(Id from, Id to) const; // τ = ℓ_P · n^{1/4}
};

// Construct a 4-simplex from a causal link.
FourSimplex build_4simplex(const CausalSet& cs, Id link_idx,
                            const SimplicialComplex& spatial);

} // namespace simhydro::causal
```

### 4-Simplex Construction (Thm 7.2)

Given causal link $(e_\alpha \prec e_\beta)$ sharing qubit $q$:
- 5 vertices: past apex $P_{m_\alpha} = (\vec{p}_{m_\alpha}^\alpha, t_\alpha)$, three face vertices at $\bar{t}$, future apex $P_{m_\beta} = (\vec{p}_{m_\beta}^\beta, t_\beta)$
- Spatial edges from Table 1 (positive $D_{jk}$)
- Timelike edges: $s^2 = |\Delta\vec{p}|^2 - (\Delta t)^2 < 0$
- Proper time: $\tau = \ell_P \cdot n^{1/d}$ with $d = 4$

### Milestone S5
You can input a quantum circuit and output a 4D Lorentzian simplicial complex with correct signature $(-,+,+,+)$.

---

## Module S6: `gauge` (Weeks 33–38)

### What You Build
The $\mathfrak{su}(4)$ decomposition on the complex, holonomies, the chiral split, and the Wilson action.

```cpp
namespace simhydro::gauge {

// Chiral decomposition of a torsion bivector.
struct ChiralPair {
    Mat4c T_plus;    // self-dual (left-handed)
    Mat4c T_minus;   // anti-self-dual (right-handed)
};
ChiralPair chiral_split(const Mat4c& torsion_bivector, const Vec4& time_direction);

// Gauge content from the curvature–torsion commutator.
// Returns [K⁺, T⁺] ∈ m_L.
Mat4c gauge_content_L(const Mat4c& K_plus, const Mat4c& T_plus);

// Holonomy around a hinge, ordered by causal direction.
Mat4c ordered_holonomy(const SimplicialComplex& K, Id hinge_id);

// Wilson action restricted to the SU(3) sector.
Real wilson_action(const SimplicialComplex& K, Real beta);

// Verify iterated commutator generation chain: a(3) → p(9) → su(4)(15).
struct GenerationResult {
    int dim_step0;   // should be 3 (dim a)
    int dim_step1;   // should be 9 (dim p = a + [k, a])
    int dim_step2;   // should be 15 (dim su(4) = p + [m, m])
};
GenerationResult verify_gauge_generation();

} // namespace simhydro::gauge
```

### The Three Decompositions of $\mathfrak{su}(4)$

**Decomposition I — Spacetime (Cartan):**
$\mathfrak{su}(4) = \mathfrak{k}\,(6) \oplus \mathfrak{p}\,(9) = \mathfrak{su}(2)_L \oplus \mathfrak{su}(2)_R \oplus \mathfrak{a}\,(3) \oplus \mathfrak{m}\,(6)$

**Decomposition II — Internal gauge (Pati–Salam):**
$\mathfrak{p} \cong \mathfrak{su}(3)\,(8) \oplus \mathfrak{u}(1)_{B-L}\,(1)$

**Decomposition III — Chiral:**
$\mathfrak{m} = \mathfrak{m}_L \oplus \mathfrak{m}_R$; gauge generation via $[K^+, \mathcal{T}^+] \in \mathfrak{m}_L$ (self-dual only).

### Milestone S6
You can compute holonomies, their chiral decomposition, and the Wilson action on any simplicial complex. You can verify the gauge algebra generation chain numerically. You can quantify gauge/gravity separation and have surveyed $n$-qubit alternatives.

### Step 6.3b: The $n$-Qubit Cartan Survey (Validating the $n=2$ Assumption)

The paper's foundational assumption is $\mathbb{C}^2 \otimes \mathbb{C}^2$ (two qubits). This is a choice, analogous to NCG choosing $\mathcal{A}_F = \mathbb{C} \oplus \mathbb{H} \oplus M_3(\mathbb{C})$ or the SM postulating $SU(3) \times SU(2) \times U(1)$. The $n$-qubit survey tests whether this choice is uniquely viable.

```cpp
namespace simhydro::gauge {

struct NQubitSurveyResult {
    int n;                          // number of qubits
    int dim_G;                      // dim SU(2^n) = 2^(2n) - 1
    int dim_k;                      // dim of maximal compact subalgebra K
    int dim_p;                      // dim of complement p
    int dim_a;                      // dim of maximal abelian in p (= candidate spacetime dim)
    std::string gauge_content;      // maximal subalgebra of p
    bool has_su3;                   // does p contain su(3)?
    bool has_u1;                    // does p contain u(1)?
    bool pati_salam_compatible;     // correct embedding for PS branching?
};

// Survey n = 1, ..., n_max qubits.
// For each, compute Cartan decomposition of SU(2^n) w.r.t. K = SU(2)^{2^{n-1}},
// identify dim(k), dim(p), dim(a), and gauge content of p.
std::vector<NQubitSurveyResult> nqubit_cartan_survey(int n_max = 4);

} // namespace simhydro::gauge
```

**Expected results:**

| $n$ | $G$ | $\dim\mathfrak{k}$ | $\dim\mathfrak{p}$ | $\dim\mathfrak{a}$ | Gauge content of $\mathfrak{p}$ | Viable? |
|-----|-----|-------|-------|-------|------|---------|
| 1 | $SU(2)$ | 1 | 2 | 1 | $U(1)$ only | No — no entanglement, no geometry |
| 2 | $SU(4)$ | 6 | 9 | 3 | $SU(3) \times U(1)$ | **Yes** — SM gauge content, $d = 3+1$ |
| 3 | $SU(8)$ | 28 | 35 | 7 | $SU(7) \times U(1)$? | Probably not — wrong gauge group, $d = 7+1$ |
| 4 | $SU(16)$ | 120 | 135 | 15 | $SU(15) \times U(1)$? | Probably not — far too large |

If $n = 2$ is the unique case matching SM physics, the minimality argument ("two qubits = smallest entangling system") is justified by uniqueness. This is the paper's analog of NCG's classification theorems for finite spectral triples.

**Test:** Run `nqubit_cartan_survey(4)`. Verify the table. If $n = 3$ gives a viable theory, document it — this would mean the paper needs a stronger selection principle than minimality.

### Step 6.4: Berry Curvature from Holonomies

The holonomy $H_h$ around a hinge is a discrete Wilson loop of the $SU(4)$ connection — a discrete Berry phase. The field strength (Berry curvature) is extracted via $U_\square \approx \exp(ia^2 F_{\mu\nu})$.

```cpp
namespace simhydro::gauge {

struct BerryCurvature {
    Mat4c  F_total;                              // full su(4)-valued field strength
    Eigen::Matrix<Real, 8, 1> F_su3;            // SU(3) component
    Eigen::Matrix<Real, 3, 1> F_su2L;           // SU(2)_L component
    Eigen::Matrix<Real, 3, 1> F_su2R;           // SU(2)_R component
    Real   F_u1;                                 // U(1)_{B-L} component
    Real   F_em;                                 // U(1)_EM component
};

// Field strength from holonomy: F_h = Im(log H_h) / A_h.
BerryCurvature berry_curvature_from_holonomy(const Mat4c& H_h, Real area);

// BCH expansion of log H_h into commutator orders.
// Order-1 commutator [kappa+tau, kappa'+tau'] is the discrete [A_a, A_b].
struct BCHDecomposition {
    Mat4c order_0;   // linear: sum of generators
    Mat4c order_1;   // quadratic: (1/2) sum of commutators
    Mat4c order_2;   // cubic: nested commutators
    Mat4c total;     // full log H_h
};
BCHDecomposition bch_decompose(const SimplicialComplex& K, Id hinge_id,
                                int max_order = 3);

// Berry curvature at every hinge, projected onto gauge components.
std::vector<BerryCurvature> berry_curvature_field(const SimplicialComplex& K);

// Discrete Chern number: (1/8pi^2) sum_h A_h tr(F_h wedge F_h).
Real chern_number(const SimplicialComplex& K);

// ---- Momentum-space Berry curvature (periodic complex) ----

struct BandBerry {
    Vec3  k;                   // crystal momentum
    Mat4c Omega;               // non-abelian Berry curvature at k
    Real  chern_2d;            // Chern number of 2D BZ slice
};

// Band Berry curvature for a periodic complex. Requires periodic BCs.
// Lattice method: link variables U_mn(k) = <u_m(k)|u_n(k+dk)>,
// plaquette P = U_12 U_23 U_34 U_41, Omega = Im(log det P) / dk^2.
std::vector<BandBerry> band_berry_curvature(
    const SimplicialComplex& unit_cell,
    const std::array<int, 3>& k_grid,
    int n_bands = 4);

} // namespace simhydro::gauge
```

**Implementation notes:**
- `berry_curvature_from_holonomy`: compute $\log H_h$ via Schur decomposition, divide by hinge area, project onto gauge subalgebras.
- `bch_decompose`: extract $\kappa_k = \log K_k$, $\tau_k = \log T_k$ from the ordered face sequence around the hinge. The order-1 commutator $[\kappa + \tau, \kappa' + \tau']$ is the non-abelian contribution that distinguishes $SU(3)$ from $U(1)$.
- `band_berry_curvature`: Phase 8+ extension. Fourier-transform the torsion field equation over the periodic complex to get $\mathbf{k}$-dependent eigenvalue problem. Berry curvature of the resulting bands computed by the standard lattice plaquette method.

**Tests:**
- Flat complex: all Berry curvatures zero ($< 10^{-12}$)
- Single source: Berry curvature localised near source
- $S^2$ with $SU(2)$ connection: Chern number is integer
- BCH orders sum to full $\log H_h$ within truncation error

### Step 6.5: Gauge/Gravity Separation Diagnostic (Critical Validation)

The central structural claim of the framework is that the $\mathfrak{k}$ (spacetime) and $\mathfrak{p}$ (gauge) sectors, while originating from the same simple group $SU(4)$, separate cleanly at low energies. Unlike NCG (Chamseddine–Connes), where the separation is structural (the algebra $C^\infty(M) \otimes \mathcal{A}_F$ is a tensor product), here the separation is dynamical — controlled by the gravitational coupling $G$.

The concern (from CM theorem considerations): the $[\mathfrak{p}, \mathfrak{p}] \subseteq \mathfrak{k}$ Cartan bracket means gauge field self-interaction feeds directly into spacetime curvature. If this backreaction is $O(1)$ rather than $O(G)$, the sectors don't separate and the framework fails.

The comparison to standard physics: in GR + Yang–Mills, the gauge field stress-energy $T_{\mu\nu}^{\text{YM}} \propto \text{tr}(F_{\mu\alpha}F_\nu{}^\alpha)$ curves spacetime via $G_{\mu\nu} = 8\pi G\, T_{\mu\nu}$. The $[\mathfrak{p}, \mathfrak{p}] \to \mathfrak{k}$ term should reproduce exactly this coupling. The analogous term in MacDowell–Mansouri gravity ($[e,e] \to \mathfrak{so}(3,1)$) gives the cosmological constant — see Wise (2010), arXiv:gr-qc/0611154.

```cpp
namespace simhydro::gauge {

struct SeparationDiagnostic {
    Real R_kk;        // ||[A^k, A^k]||  — geometric curvature (from W connections)
    Real R_pp;        // ||[A^p, A^p]||_k — gauge backreaction on geometry (from Weyl params)
    Real F_kp;        // ||[A^k, A^p]||   — minimal coupling / covariant derivative
    Real F_pp_in_p;   // ||gauge self-interaction projected onto p||
    Real ratio;       // R_pp / (R_kk + ||sum kappa_k||) — THE critical number
};

// Compute at each hinge by decomposing BCH expansion of log(H_h).
// Extracts kappa_k (from connections, in k) and alpha_k (from Weyl params, in a ⊂ p),
// then computes each commutator order's projection onto k and p.
SeparationDiagnostic separation_diagnostic(
    const SimplicialComplex& K, Id hinge_id);

// Full scan: compute ratio at every hinge, report statistics.
struct SeparationReport {
    Real r_max;       // worst-case ratio
    Real r_mean;      // average ratio
    Real r_median;
    std::vector<std::pair<Id, Real>> r_by_hinge;   // for spatial distribution plot
};
SeparationReport separation_scan(const SimplicialComplex& K);

// G-scaling test: vary effective G (rescaling k-p coupling in the action),
// solve the field equations at each G, measure r. Verify r ∝ G.
struct ScalingResult {
    std::vector<Real> G_values;
    std::vector<Real> r_values;
    Real fitted_exponent;   // should be 1.0 ± 0.1
    Real R_squared;         // should be > 0.99
};
ScalingResult separation_scaling_test(
    std::function<SimplicialComplex(Real)> complex_factory,
    const std::vector<Real>& G_values);

// Gravitational response to colour sources (acid test, Phase 7+).
// Place a q-qbar meson, measure simultaneously:
// (a) p-component field: flux tube with string tension sigma
// (b) k-component field: tiny perturbation from flux tube energy density
struct GravitationalResponse {
    Real sigma;                 // string tension from p-component
    Real delta_g_max;           // max metric perturbation from k-component
    Real expected_delta_g;      // G * sigma / c^4 (in natural units)
    Real ratio;                 // delta_g_max / expected_delta_g (should be ~1)
};
GravitationalResponse gravitational_response_to_colour(
    const SimplicialComplex& K, const MesonConfig& meson);

} // namespace simhydro::gauge
```

**Implementation notes:**
- `separation_diagnostic`: Extract $\kappa_k = \log K_k$ and $\alpha_k = \log(\text{Weyl part})$ from the ordered face sequence around each hinge. Compute BCH terms to order 2. Project $[\alpha, \alpha]$ onto $\mathfrak{k}$ using the Killing form inner product with $\mathfrak{k}$-basis generators.
- `separation_scaling_test`: Introduce a parameter $G_{\text{eff}}$ in the action that scales the $\mathfrak{k}$-$\mathfrak{p}$ cross-coupling. At $G_{\text{eff}} = 0$, the sectors fully decouple. Verify $r(G_{\text{eff}})$ is linear. No free parameter other than $G_{\text{eff}}$ should control the ratio.
- `gravitational_response_to_colour`: Phase 7+ extension. Requires the meson flux tube from S7c. In practice, verify the *scaling* $\delta g \propto G \cdot \sigma$ rather than the absolute number (simulations won't have 38 digits of precision).

**Pass/fail criteria:**

| Regime | Expected $r$ | Fail if |
|--------|-------------|---------|
| Vacuum (no gauge sources) | $0$ exactly | $r > 10^{-12}$ |
| Far from sources | $\ll 1$ | $r > 10^{-2}$ |
| Near meson flux tube (S7) | small, nonzero | $r \sim O(1)$ |
| Inside proton (S7) | still $\ll 1$ | $r \sim O(1)$ |
| $G$-scaling | $r \propto G$, exponent $= 1.0$ | exponent $\neq 1.0 \pm 0.1$ |

**Critical:** If $r \sim O(1)$ anywhere in the physical regime, the framework fails. The gauge and gravity sectors are inseparably entangled, and a structural product (as in NCG) would be required instead of a simple group.

If $r \propto G \ll 1$, the dynamical separation works — the Cartan decomposition gives the same physics as a product structure, but with higher predictive power (gauge group predicted, not input).

---

## Module S7: `matter` (Weeks 39–46)

### What You Build
Torsion sources with definite quantum numbers, the full EC solver with sources, and the proton.

### Step 7.1: Source Specification

| Particle | $SU(3)_c$ | $B-L$ | $T_{3L}$ | $Q_{\text{em}}$ | Torsion direction in $\mathfrak{p}$ |
|----------|-----------|-------|---------|---------|-------------------------------------|
| $u_L$ | $\mathbf{3}$ | $+1/3$ | $+1/2$ | $+2/3$ | $\mathfrak{su}(3)$ fund + $\mathfrak{u}(1)$ |
| $d_L$ | $\mathbf{3}$ | $+1/3$ | $-1/2$ | $-1/3$ | $\mathfrak{su}(3)$ fund + $\mathfrak{u}(1)$ |
| $e_L$ | $\mathbf{1}$ | $-1$ | $-1/2$ | $-1$ | $\mathfrak{u}(1)$ only |
| $\nu_L$ | $\mathbf{1}$ | $-1$ | $+1/2$ | $0$ | $\mathfrak{u}(1)$ only |

### Step 7.2: Source Placement and Solve

```cpp
namespace simhydro::matter {

// Assign torsion at faces near a vertex to match a FermionSource.
void place_source(SimplicialComplex& K, const FermionSource& src,
                   Id center_vertex, Real radius);

// Extract a specific gauge field component across the entire complex.
enum class GaugeComponent { SU3, SU2L, U1_EM, U1_BL };
std::vector<Real> extract_field(const SimplicialComplex& K, GaugeComponent comp);

} // namespace simhydro::matter
```

### Step 7.3: Meson ($q\bar{q}$) Test

Place $\mathbf{3}$ and $\bar{\mathbf{3}}$ sources separated by $N$ simplices. Solve. Measure:

1. Total EC action $S(r)$ at separations $r = 3, 5, 8, 12, 16, 20$ (lattice units)
2. Fit $S(r) = \sigma r + c_0$. Extract string tension $\sigma$.
3. Flux tube cross-section at midpoint: count faces with $> 10\%$ peak torsion. Should be $r$-independent.

### Step 7.4: Proton (Baryon, Y-Junction)

```cpp
namespace simhydro::matter {

struct ProtonResult {
    SimplicialComplex complex;         // solved complex with flux tubes
    Vec3              junction_pos;    // Steiner point of the Y-junction
    std::array<Real, 3> tube_lengths;  // L_1, L_2, L_3
    Real              energy;          // E = Σ σ L_i + E_junction
    bool              is_color_singlet;
};

// Build and solve for the proton.
// quark_positions: three vertices forming a triangle (~1 fm spacing).
ProtonResult build_proton(SimplicialComplex& K,
                           const std::array<Vec3, 3>& quark_positions,
                           const regge::SolverOptions& opts = {});

} // namespace simhydro::matter
```

Three quark sources at the vertices of a triangle. Y-junction at the Fermat/Steiner point. Solve the EC equations. Verify: $120°$ junction angles, colour singlet ($\|\sum \mathcal{T}^a\| < 10^{-6}$), finite action.

### Milestone S7
You have a working QCD-like simulation. You can construct mesons and baryons, observe confinement, and measure the string tension.

---

## Module S8: `hydrogen` (Weeks 47–54)

### What You Build
The electron, the Coulomb field, the hydrogen atom ground state, and the continuum limit check.

### Step 8.1: The Discrete Poisson Solver (U(1) Torsion Field)

The EM field is abelian — the discrete field equation reduces to the Laplace/Poisson equation on the dual graph. Define a scalar potential $\phi$ (one value per tetrahedron). At each interior face $f$ between tets $\alpha, \beta$:

$$\sum_{f \in \partial\alpha} \frac{A_f}{h_{\alpha\beta}}(\phi_\alpha - \phi_\beta) = 4\pi\alpha\,\rho_\alpha$$

This is a sparse symmetric positive-definite linear system $L\phi = \rho$.

```cpp
namespace simhydro::hydrogen {

// Build the cotangent-weighted graph Laplacian of the dual mesh.
// Returns a sparse matrix L such that L φ = ρ gives the discrete Poisson equation.
Eigen::SparseMatrix<Real> dual_laplacian(const SimplicialComplex& K);

// Solve L φ = ρ for the electrostatic potential.
// charges: map from tet_id → charge density at that tet.
Eigen::VectorXd solve_poisson(const Eigen::SparseMatrix<Real>& L,
                                const std::vector<std::pair<Id, Real>>& charges);

// Extract V(r) from the solved potential.
// Returns a vector of (r, φ) pairs sampled along a radial line.
std::vector<std::pair<Real, Real>> radial_potential(
    const SimplicialComplex& K, const Eigen::VectorXd& phi,
    const Vec3& origin, const Vec3& direction, int n_samples = 100);

} // namespace simhydro::hydrogen
```

**Implementation:** Use `Eigen::SimplicialLDLT` (sparse Cholesky) for the Poisson solve. For $N = 5000$ tets, $L$ is $5000 \times 5000$ with $\sim 25000$ nonzeros. Factorisation takes $< 0.1$ s; back-substitution is negligible.

### Step 8.2: Coulomb Potential Verification

Two point charges $Q = \pm 1$ at separation $r$. Solve $L\phi = \rho$. Verify $\phi(r') \approx -\alpha/r'$.

### Step 8.3: The Hydrogen Ground State

```cpp
namespace simhydro::hydrogen {

struct HydrogenResult {
    SimplicialComplex complex;
    Real              bohr_radius;      // fitted a_0 (should be ~0.529 Å)
    Real              ground_energy;    // E_0 (should be ~−13.6 eV)
    Eigen::VectorXd   torsion_profile;  // T(r) at each tet
    Eigen::VectorXd   potential;        // φ at each tet
    bool              converged;
};

struct HydrogenOptions {
    int    n_tets          = 5000;        // mesh size
    Real   mesh_radius     = 5.0;         // in units of a_0
    Real   proton_charge   = 1.0;         // Q_p
    Real   electron_charge = -1.0;        // Q_e
    Real   m_e             = M_ELECTRON_EV;
    int    n_variational   = 1;           // 1 = single exponential, 2+ = Ritz basis
};

// Full hydrogen ground-state solve.
HydrogenResult solve_ground_state(const HydrogenOptions& opts = {});

// Convergence study: solve at multiple mesh sizes and return scaling data.
struct ConvergencePoint {
    int   n_tets;
    Real  mesh_spacing;       // effective a
    Real  bohr_radius;
    Real  ground_energy;
};
std::vector<ConvergencePoint> convergence_study(
    const std::vector<int>& mesh_sizes = {500, 1000, 2000, 5000, 10000});

} // namespace simhydro::hydrogen
```

**Physical decomposition of the hydrogen atom:**

1. **Proton (from S7):** three quarks + SU(3) flux tubes + Y-junction. Treat as point charge $Q = +1$ (justified: $r_p \ll a_0$).
2. **Electron:** colour singlet, $Q = -1$. Torsion only in $\mathfrak{u}(1)_{B-L}$.
3. **EM field:** $U(1)_{\text{EM}}$ torsion connecting proton and electron. Produces $V(r) = -\alpha/r$.

**Energy functional:**

$$E(R) = \underbrace{\frac{\hbar^2}{2m_e R^2}}_{\text{kinetic (torsion gradient)}} - \underbrace{\frac{\alpha}{R}}_{\text{Coulomb (U(1) torsion)}}$$

Both terms are computed on the simplicial complex:
- Kinetic: torsion gradient energy $T = \frac{1}{2}\sum_f A_f |\nabla\mathcal{T}_f|^2$
- Coulomb: from the Poisson solver

Minimise $E(R)$ → $R^* = a_0 = \hbar^2/(m_e\alpha)$.

### Step 8.4: Self-Consistent Solution on the Complex

1. Generate a spherical tetrahedral mesh using CGAL's `Mesh_3` (radius $\sim 5a_0$, $\sim 5000$ tets, graded refinement near origin)
2. Place proton ($Q = +1$) at origin
3. Variational ansatz for electron torsion profile $\mathcal{T}(r) = \mathcal{T}_0 f(r)$:
   - Trial 1: $f(r) = e^{-r/R}$ (one parameter $R$) — minimise over $R$ via Brent's method
   - Trial 2: $f(r) = (1 + \beta r)e^{-r/R}$ (two parameters)
   - Trial 3: expand in $n_{\text{basis}}$ radial basis functions on the mesh (full Ritz variational problem → generalised eigenvalue problem)
4. For each trial, compute $T[f]$ and $V[f]$ on the mesh, minimise $E[f] = T[f] + V[f]$

### Step 8.5: Excited States

Use the Ritz method (Trial 3 above) with $n_{\text{basis}} \geq 10$. The eigenvalue problem:
$$H_{\text{eff}}\,\vec{c} = E\,S\,\vec{c}$$
where $H_{\text{eff}}$ is the kinetic + Coulomb matrix in the basis, and $S$ is the overlap matrix. Solve via `Eigen::GeneralizedSelfAdjointEigenSolver`.

First three eigenvalues should approach $-13.6/n^2$ eV for $n = 1, 2, 3$.

### Step 8.6: Berry Curvature of Hydrogen Eigenstates

The Ritz eigenstates $|\psi_n(\mathbf{R})\rangle$ depend on the proton position $\mathbf{R}$, defining a Berry connection on parameter space.

```cpp
namespace simhydro::hydrogen {

struct HydrogenBerry {
    std::vector<Eigen::MatrixXcd> A;     // A[a] = Berry connection, n_deg x n_deg
    std::vector<Eigen::MatrixXcd> Omega; // Omega[{ab}] = Berry curvature
    std::vector<Eigen::MatrixXcd> comm;  // [A_a, A_b] commutator contribution
};

// Berry curvature by finite-difference displacement of the proton position.
HydrogenBerry compute_berry_curvature(
    const SimplicialComplex& K,
    const HydrogenOptions& opts,
    const std::vector<int>& subspace_indices,  // degenerate states
    Real delta = 0.01);                         // displacement in a_0

// Spin-orbit coupling from chiral torsion decomposition T^+/-.
struct SOCResult {
    int n, l;
    Real j;           // j = l +/- 1/2
    Real delta_E;     // SOC splitting (computed)
    Real expected;    // alpha^2 * E_n/n * (1/(j+1/2) - 3/(4n))
};
std::vector<SOCResult> compute_fine_structure(
    const SimplicialComplex& K, const HydrogenOptions& opts, int n_max = 3);

} // namespace simhydro::hydrogen
```

**Physical content:**

1. **Ground state ($n = 1$):** non-degenerate, abelian Berry curvature. $\Omega_{ab} = 0$ by spherical symmetry.

2. **$n = 2$ subspace:** four-fold degenerate ($2s + 2p_{x,y,z}$). Berry connection is $U(4)$-valued (Wilczek-Zee). Commutator $[\mathcal{A}_a, \mathcal{A}_b] \neq 0$ — non-abelian Berry curvature with monopole structure at the degeneracy point.

3. **Spin-orbit coupling:** $\mathcal{T}^\pm = \frac{1}{2}(\mathcal{T} \pm i\star\mathcal{T})$ couples orbital torsion to chirality. Including $\mathcal{T}^\pm$ in the variational problem should split $n = 2$ by $\Delta E_{n,l,j} = \frac{\alpha^2 E_n}{n}(1/(j+1/2) - 3/(4n))$ without imposing $H_{\text{SOC}}$ by hand.

**Tests:**

| Test | Expected | Tolerance |
|------|----------|-----------|
| $\Omega_{ab}(n=1)$ | $0$ | $< 10^{-10}$ |
| $[\mathcal{A}_a, \mathcal{A}_b]$ for $n=2$ | non-trivial | matrix norm $> 10^{-6}$ |
| Fine structure splitting | $\sim 4.5 \times 10^{-5}$ eV | $\sim 50\%$ (stretch goal) |
| Chern number of $n=2$ bundle | integer | exact |

The fine structure test demands $\alpha^2 \sim 5\times 10^{-5}$ relative precision — far beyond the ground-state mesh requirements. Treat as a stretch goal.

### Milestone S8
The hydrogen ground state has $a_0 \approx 0.53$ Å and $E_0 \approx -13.6$ eV, computed entirely on the eigenphase-embedded simplicial complex.

---

## Mesh Generation with CGAL

CGAL's `Mesh_3` module generates high-quality tetrahedral meshes with graded refinement — essential for the multi-scale problems in Phases 4–8.

```cpp
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Implicit_mesh_domain_3.h>

using K       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Domain  = CGAL::Labeled_mesh_domain_3<K>;
using Tr      = CGAL::Mesh_triangulation_3<Domain>::type;
using C3T3    = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
using Criteria = CGAL::Mesh_criteria_3<Tr>;

// Generate a graded spherical mesh for the hydrogen atom.
// finer_radius: region with fine resolution (e.g., 2 * a_0)
// outer_radius: total mesh radius (e.g., 5 * a_0)
C3T3 generate_hydrogen_mesh(Real finer_radius, Real outer_radius,
                             Real fine_cell_size, Real coarse_cell_size);

// Convert CGAL mesh to SimplicialComplex.
SimplicialComplex cgal_to_complex(const C3T3& mesh);
```

This gives you adaptive refinement: fine mesh near the proton (to resolve the $1/r$ singularity), coarser mesh in the outer region (to keep the DOF count manageable).

---

## Visualisation and I/O

Since the core code is C++, visualisation is best done by writing mesh data to standard formats and post-processing externally.

```cpp
namespace simhydro::io {

// Write a SimplicialComplex to VTK format for ParaView / VisIt.
// Scalar fields (volume, torsion magnitude, potential, etc.) are
// written as cell data.
void write_vtk(const SimplicialComplex& K,
               const std::string& filename,
               const std::unordered_map<std::string, std::vector<Real>>& cell_data = {});

// Write the Weyl chamber and marked points to a JSON file
// for plotting with a Python / matplotlib script.
void write_weyl_chamber_json(const std::string& filename,
                              const std::vector<Vec3>& points,
                              const std::vector<std::string>& labels = {});

// Write convergence data to CSV.
void write_csv(const std::string& filename,
               const std::vector<std::vector<Real>>& columns,
               const std::vector<std::string>& headers);

} // namespace simhydro::io
```

**Recommended visualisation pipeline:**
- **VTK output → ParaView** for 3D mesh inspection, torsion field visualisation, flux tube cross-sections
- **JSON/CSV → Python matplotlib** for convergence plots, $V(r)$, energy vs $R$, Weyl chamber scatter plots

This avoids pulling a GUI library into the C++ build while giving you publication-quality figures.

---

## Numerical Checklist by Phase

### Phase 0–2 Tests (Algebraic)
| Test | Expected | Tolerance |
|------|----------|-----------|
| KAK round-trip | $\|U - U'\|_F = 0$ | $< 10^{-10}$ |
| Volume formula | $V = (16\sqrt{3}/27)c_1c_2c_3$ | exact to `double` |
| CM determinant | $\text{CM} = 288V^2$ | exact to `double` |
| Commutator table | $[\mathfrak{k},\mathfrak{k}] \subseteq \mathfrak{k}$ etc. | $< 10^{-14}$ |
| $S_4 \cong W(A_3)$ | 24 group elements match | exact |

### Phase 3 Tests (Composition)
| Test | Expected | Tolerance |
|------|----------|-----------|
| Permutation composition | $\vec{c}' = \vec{c}^{(1)} + \frac{1}{4}\mathcal{W}(\sigma)\vec{c}^{(2)}$ | $< 10^{-10}$ |
| Torsion-free solver | $\|\mathcal{T}\| = 0$ | $< 10^{-12}$ |
| Universality | 3 generic gates reach all of $\mathcal{W}$ | $> 99\%$ coverage |

### Phase 4 Tests (Regge)
| Test | Expected | Tolerance |
|------|----------|-----------|
| Gauss–Bonnet ($S^2$, $N = 1280$) | $\sum A_v\epsilon_v = 4\pi$ | $< 0.1\%$ |
| 3D Regge convergence rate | $O(h^2)$ | fit exponent $2.0 \pm 0.2$ |
| Flat-space extremum | $\epsilon_h = 0$ everywhere | $< 10^{-8}$ |

### Phase 5–6 Tests (Causal + Gauge + Berry Curvature + Separation)
| Test | Expected | Tolerance |
|------|----------|-----------|
| Spacetime dimension (Myrheim–Meyer) | $d \approx 4$ | $3.5 < d < 4.5$ |
| Gram matrix signature | $(-,+,+,+)$ | exact signs |
| $[\mathfrak{a} \to \mathfrak{p}]$ generation | dim $3 \to 9$ | exact |
| $[\mathfrak{p} \to \mathfrak{su}(4)]$ generation | dim $9 \to 15$ | exact |
| Slansky branching $\mathbf{15} \to (\mathbf{8},0)+(\mathbf{1},0)+(\mathbf{3},1)+(\bar{\mathbf{3}},-1)$ | $9 + 6 = 15$ | exact |
| $n$-qubit survey: $n=2$ gives $SU(3) \times U(1)$ in $\mathfrak{p}$ | unique viability | report for $n = 1\ldots 4$ |
| Chiral selection rule | $[K^+, \mathcal{T}^-] = 0$ | $< 10^{-12}$ |
| Berry curvature (flat complex) | $F_h = 0$ for all hinges | $< 10^{-12}$ |
| BCH order-1 commutator vs full $\log H_h$ | agrees to $O(A_h^2)$ | $< 10\%$ relative |
| $SU(3)$ Berry curvature near source | localised, non-zero | qualitative |
| $U(1)$ Berry curvature near source | $1/r^2$ fall-off (Coulomb) | qualitative |
| Chern number ($S^2$ with $SU(2)$ connection) | integer | exact |
| Separation ratio $r$ (vacuum) | $0$ | $< 10^{-12}$ |
| Separation ratio $r$ (with sources) | $\ll 1$ | $< 10^{-2}$ |
| Separation $G$-scaling exponent | $r \propto G^1$ | exponent $1.0 \pm 0.1$ |

### Phase 7 Tests (Matter + Separation with Sources)
| Test | Expected | Tolerance |
|------|----------|-----------|
| Single quark action | divergent | grows with boundary distance |
| Meson $V(r)$: linearity | $R^2 > 0.99$ for linear fit | |
| Meson flux tube $A_\perp$ | constant vs $r$ | $< 20\%$ variation |
| Proton junction angles | $\sim 120°$ each | $< 10°$ deviation |
| Proton colour charge | singlet | $\|\sum \mathcal{T}^a\| < 10^{-6}$ |
| Separation $r$ near meson flux tube | $\ll 1$ | $< 10^{-2}$ |
| Separation $r$ inside proton | $\ll 1$ | $< 10^{-2}$ |
| Gravitational response $\delta g \propto G \cdot \sigma$ | linear in $G$ | exponent $1.0 \pm 0.1$ |

### Phase 8 Tests (Hydrogen + Berry Curvature)
| Test | Expected | Tolerance |
|------|----------|-----------|
| Coulomb $V(r)$ on mesh | $-\alpha/r$ | $< 5\%$ at $r > 2a$ |
| Kinetic energy scaling | $T \propto 1/R^2$ | exponent $-2.0 \pm 0.1$ |
| $a_0$ (fine mesh) | 0.529 Å | $< 5\%$ |
| $E_0$ (fine mesh) | −13.6 eV | $< 5\%$ |
| Torsion profile | $\propto e^{-2r/a_0}$ | qualitative |
| $E_2 - E_1$ | 10.2 eV | $< 10\%$ |
| Berry curvature $\Omega(n=1)$ | $0$ (non-degenerate) | $< 10^{-10}$ |
| $[\mathcal{A}_a, \mathcal{A}_b]$ for $n=2$ | non-zero matrix | norm $> 10^{-6}$ |
| Chern number of $n=2$ bundle | integer | exact |
| Fine structure (SOC, stretch) | $\Delta E \sim \alpha^2 E_n / n$ | $\sim 50\%$ |

---

## Estimated Compute Times (C++)

All times assume a single core of a modern x86-64 CPU (e.g., Ryzen 7 / i9, ~4 GHz). C++ with Eigen and -O2 is roughly 10–50× faster than Python/NumPy for the inner loops. The dominant cost is the L-BFGS solver.

| Task | DOF | Method | Wall time |
|------|-----|--------|-----------|
| KAK decomposition ($10^3$ matrices) | — | eigensolver | ~0.05 s |
| Composition map ($10^4$ sweeps) | — | spectral | ~0.5 s |
| 2D Gauss–Bonnet ($N = 1280$) | ~1300 | direct | ~0.005 s |
| 3D Regge convergence ($N = 5000$) | ~15000 | L-BFGS (CppAD) | ~1–3 min |
| Meson field solve ($N = 500$) | ~4500 | L-BFGS (CppAD) | ~15 s |
| Proton Y-junction ($N = 1000$) | ~9000 | L-BFGS + junction opt | ~2 min |
| Coulomb solver ($N = 5000$) | ~5000 | `SimplicialLDLT` | ~0.05 s |
| Hydrogen variational ($N = 5000$, Trial 1) | ~5010 | Cholesky + 1D Brent | ~0.5 s |
| Hydrogen variational ($N = 5000$, Trial 3) | ~5100 | generalised eigensolver | ~5 s |
| Hydrogen continuum limit (4 mesh sizes) | — | above × 4 | ~30 s |

With finite-difference gradients instead of CppAD: multiply L-BFGS times by $\sim N_{\text{DOF}}$. This makes the Regge convergence test take ~hours and the proton ~days. **Use autodiff.**

---

## Software Dependencies

### Required (all phases)

| Library | Version | Purpose | Install |
|---------|---------|---------|---------|
| **Eigen** | ≥ 3.4 | Dense & sparse linear algebra | `apt install libeigen3-dev` or vcpkg |
| **CGAL** | ≥ 5.5 | Mesh generation (`Mesh_3`) | `apt install libcgal-dev` or vcpkg |
| **fmt** | ≥ 9.0 | String formatting | vcpkg / conan |
| **Google Test** | ≥ 1.12 | Unit tests | vcpkg / conan |
| **CMake** | ≥ 3.22 | Build system | system package |

### Required (Phase 7+)

| Library | Version | Purpose | Install |
|---------|---------|---------|---------|
| **CppAD** | ≥ 2024 | Automatic differentiation | `apt install cppad` or from source |
| **LBFGSpp** | ≥ 0.3 | L-BFGS optimiser (header-only, Eigen-based) | `git clone` / vcpkg |

### Optional

| Library | Purpose | When |
|---------|---------|------|
| **Enzyme** | LLVM-based AD (alternative to CppAD) | If CppAD tape overhead is a bottleneck |
| **Google Benchmark** | Microbenchmarking | Profiling inner loops |
| **VTK** | Direct VTK writing from C++ | If you want C++-side rendering (usually overkill) |
| **pybind11** | Python bindings | If you want to call C++ from Python for plotting |

### Minimal `conanfile.txt`

```
[requires]
eigen/3.4.0
cgal/5.6
fmt/10.1.1
gtest/1.14.0
cppad/20240000.4
lbfgspp/0.3.0

[generators]
CMakeDeps
CMakeToolchain
```

---

## Performance Notes

**Memory layout.** The `SimplicialComplex` stores `tets` and `faces` as `std::vector` (contiguous). For the L-BFGS inner loop, pack the DOF into a single `Eigen::VectorXd` of length $N_{\text{DOF}} = 3N_{\text{interior\_tets}} + 6N_{\text{interior\_faces}}$. The unpacking/repacking adds negligible overhead compared to the action evaluation.

**Cache behaviour.** The action evaluation visits every tet and every face once per iteration. For $N = 5000$ tets, the working set is $\sim 5000 \times (3 \text{ doubles} + 24 \text{ doubles for vertices} + \ldots) \sim 1$ MB — fits in L2 cache. At $N = 50000$, it spills to L3 but is still dominated by compute (eigensolvers, matrix exponentials), not memory bandwidth.

**Parallelism.** The action $S_{\text{EC}} = \sum_h A_h\epsilon_h + \sum_f \mathcal{T}\Sigma$ is a reduction over independent hinges and faces. Parallelise with `#pragma omp parallel for reduction(+:action)` over the hinge/face loops. The L-BFGS iterations are sequential, but each action+gradient evaluation is parallelisable. For $N > 10000$, expect near-linear speedup to $\sim 4$ cores.

**CppAD tape size.** For $N = 5000$ DOF, the CppAD tape (forward sweep recording) uses $\sim 100$ MB of memory. This is fine for a desktop. For $N > 50000$, consider Enzyme (no tape) or checkpoint-based CppAD.

---

## Where It Could Break

The simulation has four potential failure points where the framework's physics content is genuinely tested, not just its algebra.

**Failure point 0: Gauge/gravity separation (Step 6.5).** The most fundamental test. If the separation diagnostic ratio $r = \|[\mathfrak{p}, \mathfrak{p}]\|_\mathfrak{k} / \|F^\mathfrak{k}_{\text{geometric}}\|$ is $O(1)$ rather than $O(G)$, the gauge and gravity sectors are inseparably entangled and the framework cannot reproduce the observed decoupling of QCD from gravity at hadronic scales. This would validate the critique that a simple group ($SU(4)$) cannot replace NCG's product structure ($C^\infty(M) \otimes \mathcal{A}_F$) for generating separate gauge and spacetime sectors. The Coleman–Mandula theorem would then apply to the pre-geometric structure, not just the emergent S-matrix, and the paper's approach would be fundamentally blocked. The G-scaling test ($r \propto G$) distinguishes dynamical separation (paper's claim) from algebraic accident.

**Failure point 1: Torsion gradient energy scaling (Step 8.4).** If the gradient energy scales as $T \propto 1/R^p$ with $p \neq 2$, the hydrogen atom doesn't work. $p = 2$ is required by dimensional analysis and the continuum limit, but dimensional analysis only holds if the discrete-to-continuum limit is uniform — the fatness bound must not be violated by the electron's torsion profile.

**Failure point 2: Mass identification (Step 8.5).** The electron mass $m_e$ must appear as the coefficient of $1/R^2$ in the gradient energy. In the framework, $m_e$ is set by the Casimir energy of the $\mathbf{4}$ representation and the Weyl chamber ordering. Whether the resulting number is $0.511$ MeV/$c^2$ is an extremely non-trivial prediction. For the simulation, you can treat $m_e$ as an input. But the framework claims to predict it.

**Failure point 3: Multi-scale consistency (Step 8.8).** The proton and atom live at scales separated by a factor of $\sim 10^5$. The eigenphase-embedded tetrahedra must maintain fatness across this range. If the Weyl parameters required to reproduce proton-scale physics push the tetrahedra into degenerate configurations, the CMS convergence theorem fails and the whole construction collapses.

Document these outcomes carefully. A failure at any of these points is as scientifically valuable as a success — it identifies exactly where the "spacetime from entanglement" framework needs modification.
