// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H
#define TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "cobordism/HodgeLaplacian.h"
#include "mesh/Edge.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh { class Edge; class Vertex; class Simplex; }
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;
using ::tessera::mesh::Edge;

/// # EigenstateSynthesis
///
/// The §4b inverse eigenvector problem on a **fixed** complex, degree-parameterized
/// by `int k`: given a target state \f$ \psi \f$, score how close the complex's
/// current Hermitian edge weights make \f$ \psi \f$ to being an eigenvector of the
/// degree-\f$ k \f$ Hodge Laplacian \f$ L_k \f$ (via `HodgeLaplacian`), and
/// read/write those weights so a search can perturb them.
///
/// At \f$ k = 0 \f$ \f$ L_0 = D - A \f$ is the U(1)-weighted graph Laplacian (the
/// magnitude convention) and \f$ \psi \f$ is a vertex vector (length \f$ |V| \f$,
/// sorted-id order). At \f$ k \geq 1 \f$ \f$ L_k \f$ is the **metric Hodge
/// Laplacian** on \f$ k \f$-cochains and \f$ \psi \f$ is a \f$ k \f$-form (length
/// \f$ |C_k| \f$, the canonical `ChainComplex` \f$ k \f$-cell order); the tunable
/// parameters stay the **edge** squared-lengths (`Edge::setSquaredLength`), which
/// feed the per-simplex volume weights \f$ W_k \f$ of \f$ L_k \f$ through
/// `Simplex::volume` (the U(1) phases enter only the \f$ k = 0 \f$ operator). This
/// lifts the §5.0 fixed-boundary interior fill from the \f$ k = 0 \f$ 2-complex
/// setting to the \f$ k = 1 \f$ boundary harmonic of a 3-manifold-with-boundary
/// (#176): a 3-manifold \f$ W \f$ whose \f$ \ker L_1(\partial W) \f$ is matched by
/// pinning \f$ \partial W \f$ and filling the interior. `cellSimplices()` gives the
/// sorted vertex-id tuple of each \f$ \psi \f$ component, so a caller can pin the
/// boundary \f$ k \f$-cells to a target form and leave the interior free.
///
/// The search itself (non-convex, multi-restart, e.g. `scipy.optimize.minimize`
/// L-BFGS-B over the flat \f$ \{w_{ij}\}\cup\{\theta_{ij}\} \f$ vector) drives
/// the cone-and-retry growth loop in a separate stage; this class is the
/// **residual + parameter access** core it calls. Growing the complex
/// (coning-in auxiliary vertices) is out of scope here.
///
/// ## Residual
///
/// For a unit target \f$ \psi \f$ the eigenvalue-agnostic residual is the
/// squared norm of the component of \f$ L\psi \f$ orthogonal to \f$ \psi \f$,
/// \f$ r(\psi) = \big\|\,(I-\psi\psi^\dagger)\,L\,\psi\,\big\|^2
///            = \|L\psi - \lambda\psi\|^2,\quad \lambda = \psi^\dagger L\psi, \f$
/// so \f$ r = 0 \iff L\psi \parallel \psi \iff \psi \f$ is an eigenvector, and
/// the realized eigenvalue is the Rayleigh quotient \f$ \lambda \f$. A non-unit
/// \f$ \psi \f$ is normalized internally (the eigenvector condition is
/// scale-invariant). The Laplacian is reassembled from the **current** edge
/// weights/phases on every call, so the residual tracks in-place perturbations
/// of `setWeights` / `setPhases`.
///
/// ## Parameters
///
/// The tunable parameters are the per-edge squared-length magnitudes
/// \f$ \{w_{ij}\} \f$ (`Edge::setSquaredLength`) and U(1) connection phases
/// \f$ \{\theta_{ij}\} \f$ (`Edge::setPhase`), in a stable edge order fixed at
/// construction (the `EdgeList` order, restricted to the edges that carry weight
/// in \f$ L \f$: both endpoints present, no self-loops). `weights()` / `phases()`
/// read them; `setWeights()` / `setPhases()` write them back in place — no mesh
/// rebuild. `psi` components are indexed in the same sorted-vertex-id order as
/// `HodgeLaplacian` (\f$ k=0 \f$), so they align with the operator.
///
/// ## Fixed-boundary interior fill (§5.0)
///
/// The realizability oracle (#138) fills the **interior** of a bulk \f$ W_{AB} \f$
/// whose boundary is *pinned* — the §4b cone-and-retry restricted to the interior.
/// The tunable edges split into the **boundary** set \f$ \partial W \f$ (those on a
/// codimension-one face belonging to exactly one top cell — held fixed) and the
/// **interior** set (everything else — free). `interiorWeights()` / `interiorPhases()`
/// and `setInteriorWeights()` / `setInteriorPhases()` read/write *only* the interior
/// edges, so a search drives \f$ r\to 0 \f$ for a target output eigenvector while
/// \f$ \partial W \f$ stays byte-identical; `boundaryEdges()` exposes that fixed set
/// for verification. `growInterior()` cones a fresh interior vertex into a top cell
/// via the boundary-fixed pre-geometric Pachner add (#112) — a topology-preserving
/// \f$ 1\!\to\!(d+1) \f$ stellar subdivision that enriches the interior with
/// \f$ \partial W \f$ untouched — and re-captures the partition, so the loop can
/// optimize, then grow, then re-optimize. `interiorVertexCount()` /
/// `numInteriorEdges()` report the interior complexity reached: a reachable target
/// drives \f$ r\to 0 \f$ at some minimal complexity, an unreachable one floors at a
/// positive residual (the spectral obstruction the oracle consumes). The boundary
/// edge classification needs genuine codim-one structure (top cells of \f$ \ge 3 \f$
/// vertices); on a pure 1-complex every edge is interior (the free §4b regime).
class EigenstateSynthesis {
  public:
    /// Construct over a fixed triangulation at degree `k` (default \f$ 0 \f$, the
    /// vertex graph Laplacian). The \f$ k \f$-cell order (`cellSimplices()`) and the
    /// tunable edge order are captured now; edge weights/phases are read live on
    /// each residual query. The held `shared_ptr` keeps the spacetime alive.
    /// @throws std::runtime_error if `k < 0`.
    explicit EigenstateSynthesis(std::shared_ptr<Spacetime> st, int k = 0);

    /// The cochain degree \f$ k \f$ this synthesizer scores against (the
    /// `HodgeLaplacian` degree of \f$ L_k \f$).
    [[nodiscard]] int degree() const noexcept { return k_; }

    /// The operator dimension \f$ N \f$ — the required length of any `psi`:
    /// \f$ |V| \f$ at \f$ k = 0 \f$, else \f$ |C_k| \f$ (the number of
    /// \f$ k \f$-cells).
    [[nodiscard]] std::size_t order() const noexcept { return order_; }

    /// The sorted vertex-id tuple of each \f$ \psi \f$ component, in operator
    /// order (length `order()`): a single-vertex tuple per component at
    /// \f$ k = 0 \f$, else the \f$ k \f$-cell vertex tuples in the canonical
    /// `ChainComplex` column order. Lets a caller identify which components are the
    /// boundary \f$ k \f$-cells carrying a target form vs. the interior ones.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cellSimplices()
        const noexcept {
      return cellOrdering_;
    }

    /// Number of tunable edges — the length of `weights()` / `phases()`.
    [[nodiscard]] std::size_t numEdges() const noexcept { return edges_.size(); }

    /// The eigenvalue-agnostic residual \f$ r(\psi) = \|(I-\psi\psi^\dagger)L\psi\|^2 \f$
    /// against the current edge weights/phases. \f$ \psi \f$ is normalized
    /// internally; \f$ r = 0 \iff L\psi \parallel \psi \f$.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] double residual(const std::vector<std::complex<double>> &psi) const;

    /// The Rayleigh quotient \f$ \lambda = \psi^\dagger L\psi / \psi^\dagger\psi \f$
    /// (real; \f$ L \f$ Hermitian) — the realized eigenvalue when \f$ r = 0 \f$.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] double rayleigh(const std::vector<std::complex<double>> &psi) const;

    /// \f$ L\psi \f$ against the current edge weights/phases (no normalization),
    /// for direct \f$ L\psi \parallel \psi \f$ cross-checks. Length `order()`.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] std::vector<std::complex<double>> apply(
        const std::vector<std::complex<double>> &psi) const;

    /// The edge magnitudes \f$ \{w_{ij}\} \f$ (`Edge::getSquaredLength`) in the
    /// stable edge order, length `numEdges()`.
    [[nodiscard]] std::vector<std::complex<double>> weights() const;

    /// The edge phases \f$ \{\theta_{ij}\} \f$ (`Edge::getPhase`) in the stable
    /// edge order, length `numEdges()`.
    [[nodiscard]] std::vector<double> phases() const;

    /// Write the edge magnitudes in place (`Edge::setSquaredLength`).
    /// @throws std::runtime_error if `w.size() != numEdges()`.
    void setWeights(const std::vector<double> &w);

    /// Write the edge phases in place (`Edge::setPhase`).
    /// @throws std::runtime_error if `theta.size() != numEdges()`.
    void setPhases(const std::vector<double> &theta);

    // === Fixed-boundary interior fill (§5.0) ===

    /// Number of interior tunable edges (not on \f$ \partial W \f$) — the length
    /// of `interiorWeights()` / `interiorPhases()` and the free parameters a
    /// fixed-boundary search varies.
    [[nodiscard]] std::size_t numInteriorEdges() const noexcept {
      return interiorEdgeIdx_.size();
    }

    /// Number of boundary tunable edges (on \f$ \partial W \f$, held fixed).
    [[nodiscard]] std::size_t numBoundaryEdges() const noexcept {
      return boundaryEdgeIdx_.size();
    }

    /// Number of interior vertices (on no boundary face) — the coned-in apexes;
    /// the interior complexity the synthesis grows / reports.
    [[nodiscard]] std::size_t interiorVertexCount() const noexcept {
      return interiorVertexCount_;
    }

    /// The interior edge magnitudes \f$ \{w_{ij}\} \f$ in interior-edge order,
    /// length `numInteriorEdges()`.
    [[nodiscard]] std::vector<std::complex<double>> interiorWeights() const;

    /// The interior edge phases \f$ \{\theta_{ij}\} \f$ in interior-edge order,
    /// length `numInteriorEdges()`.
    [[nodiscard]] std::vector<double> interiorPhases() const;

    /// Write the interior edge magnitudes in place; the boundary edges are left
    /// untouched. @throws std::runtime_error if `w.size() != numInteriorEdges()`.
    void setInteriorWeights(const std::vector<double> &w);

    /// Write the interior edge phases in place; the boundary edges are left
    /// untouched. @throws std::runtime_error if `theta.size() != numInteriorEdges()`.
    void setInteriorPhases(const std::vector<double> &theta);

    /// The boundary tunable edges as sorted \f$ (\min\text{id},\max\text{id}) \f$
    /// endpoint pairs — the fixed \f$ \partial W \f$ edge set, for asserting it is
    /// untouched through an interior fill / growth sweep.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>>
    boundaryEdges() const;

    /// The interior tunable edges as sorted \f$ (\min\text{id},\max\text{id}) \f$
    /// endpoint pairs (the complement of `boundaryEdges()`).
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>>
    interiorEdges() const;

    /// Cone a fresh interior vertex into a top cell via the boundary-fixed
    /// pre-geometric Pachner add (#112): a \f$ 1\!\to\!(d+1) \f$ stellar
    /// subdivision (valid in any dimension — \f$ 1\!\to\!4 \f$ on a tetrahedron in
    /// 3D) that leaves \f$ \partial W \f$ exactly fixed while enriching the
    /// interior. Re-captures the \f$ k \f$-cell order (`cellSimplices()`), tunable
    /// edges, and interior/boundary partition, so `order()` and
    /// `numInteriorEdges()` grow. At \f$ k = 0 \f$ the new apex is the largest id,
    /// appended last, so existing `psi` indices are preserved; at \f$ k \geq 1 \f$
    /// the new \f$ k \f$-cells interleave in the canonical order, so re-identify
    /// the boundary/interior components via `cellSimplices()` (by sorted vertex-id
    /// tuple) after each grow. The RNG `seed` makes the target-cell choice
    /// reproducible. Returns `false` if no top cell can be subdivided (e.g. a
    /// 1-complex, top cells of \f$ <3 \f$ vertices), leaving the complex unchanged.
    bool growInterior(std::uint64_t seed);

    // === Free interior connectivity (general growth primitive, #200) ===

    /// Add a fresh interior vertex with an **arbitrary** specified set of incident
    /// simplices — the cone-free generalization of `growInterior`. Each entry of
    /// `incidentSimplices` is a set of **existing** vertex ids; the new vertex
    /// together with that set forms one new simplex, whose 1-skeleton (every
    /// pairwise edge) is materialized. So a singleton entry \f$ \{u\} \f$ wires the
    /// new vertex to \f$ u \f$ by an edge, and the \f$ d \f$ facets of a top cell
    /// reproduce that cell's cone connectivity — coning is one special case.
    ///
    /// The new vertex takes the largest id (it appends last in sorted-id order, so
    /// the boundary-support \f$ \psi \f$ prefix is preserved). The move is **purely
    /// additive** (nothing is removed) and validates **only** the two invariants
    /// the experiment allows: (a) the result is a valid downward-closed abstract
    /// simplicial complex (every pair within each new simplex carries an edge), and
    /// (b) the pinned boundary \f$ \partial W \f$ is **bit-exact** untouched (same
    /// edge set, same weights/phases). **No** manifold / pseudomanifold /
    /// orientability / purity / topology constraint is imposed. Re-captures the
    /// operator and the interior/boundary partition.
    ///
    /// Returns `false`, leaving the complex **unchanged** (rolled back), if a spec
    /// is empty, references a missing vertex, repeats a vertex, or the attach would
    /// perturb \f$ \partial W \f$. Because only the k=0 graph Laplacian's edges feed
    /// `residual()`, wiring the interior vertex by edges (singleton specs) is always
    /// boundary-safe: a new edge to a brand-new vertex creates no new top cell and
    /// changes no facet count among the pinned boundary.
    bool attachInteriorVertex(
        const std::vector<std::vector<std::uint64_t>> &incidentSimplices);

    /// Undo the most recent `attachInteriorVertex` (LIFO): remove the simplices
    /// and edges it created and the interior vertex it added, restoring the
    /// complex bit-exactly, and re-capture. Returns `false` if there is no attach
    /// to undo. Lets a connectivity search try a candidate, score it, and roll
    /// back to try the next.
    bool detachLastInteriorVertex();

    /// All vertex ids in the complex, sorted ascending — the candidate pool a
    /// connectivity search wires a fresh interior vertex into.
    [[nodiscard]] std::vector<std::uint64_t> vertexIds() const;

    /// The boundary (\f$ \partial W \f$) vertex ids, sorted ascending — the
    /// vertices on a codim-one face of exactly one top cell. A "boundary-star"
    /// connectivity candidate wires the new vertex to these.
    [[nodiscard]] std::vector<std::uint64_t> boundaryVertexIds() const;

    /// The top cells as sorted vertex-id tuples (the \f$ d+1 \f$-vertex simplices).
    /// The "cone-equivalent" connectivity candidate wires the new vertex to one of
    /// these cells' vertices, reproducing `growInterior`'s 1-skeleton.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> topCells() const;

    /// The dual-complex validity verdict (`ChainComplex::dualComplexIsValid`)
    /// for the synthesizer's **current** complex: top cells from the surgery
    /// state and, when the degree sits at \f$ k = n - 1 \f$ (the register
    /// layers), the \f$ k \f$-cell universe (`cellSimplices()`) checked for
    /// dangling facets. Topology-changing moves should be accepted only while
    /// this stays true — validity in the dual space, not merely scoreability
    /// on the primal lattice.
    [[nodiscard]] std::pair<bool, std::string> dualComplexValid() const;

    // === The carried register read-outs (#286) ===

    /// The **period matrix** of the current harmonics over the boundary cycles
    /// of the given (removed) cells: a flat row-major
    /// \f$ \dim\ker L_k \times |\text{holes}| \f$ complex array whose entry
    /// \f$ [\,r\,|\text{holes}| + q\,] \f$ is harmonic \f$ r \f$ summed over
    /// hole \f$ q \f$'s facets with the induced-orientation signs of the
    /// boundary operator — facet \f$ j \f$ of the sorted hole drops vertex
    /// \f$ v_j \f$ and carries \f$ (-1)^j \f$, so a circle (a triangle hole at
    /// \f$ k = 1 \f$) contributes \f$ +(a,b) + (b,c) - (a,c) \f$ and a sphere
    /// (a tetrahedron hole at \f$ k = 2 \f$) its four \f$ (-1)^j \f$-signed
    /// triangles: degree-general. Each hole is a \f$ (k\!+\!2) \f$-vertex
    /// tuple (sorted internally) whose facets must all be \f$ k \f$-cells of
    /// the **current** complex — the cycles surgery (`removeInteriorCell`)
    /// leaves behind. Harmonics are read fresh from the live complex
    /// (`HodgeLaplacian::harmonicMatrix`, \f$ |\lambda| < 10^{-9} \f$, metric
    /// weights), rows in ascending-eigenvalue order. Empty when
    /// \f$ \ker L_k = 0 \f$.
    /// @throws std::runtime_error if a hole has the wrong vertex count or one
    ///   of its facets is not a \f$ k \f$-cell of the complex.
    [[nodiscard]] std::vector<std::complex<double>> cyclePeriods(
        const std::vector<std::vector<std::uint64_t>> &holes) const;

    /// The **verdict primitive** every realizability experiment flows through,
    /// in one call: the genuine residual of the carried representative of
    /// `targetPeriods` over the `holes` cycles. Builds the period matrix
    /// \f$ P \f$ (`cyclePeriods`), solves the least-squares projection
    /// \f$ \min_c \|P^{\top} c - \text{target}\| \f$ (minimum-norm, so a
    /// rank-deficient carried space matches `numpy.linalg.lstsq`), forms the
    /// harmonic combination \f$ \psi = \sum_r c_r h_r \f$, attaches the
    /// uncarried remainder (the **minimal leak**) to one facet per hole so the
    /// cochain's periods are exactly `targetPeriods` — the hole's first facet
    /// in the degree's established walk order, both of boundary sign
    /// \f$ +1 \f$: the \f$ (a,b) \f$ edge of a circle at \f$ k = 1 \f$, the
    /// drop-\f$ v_0 \f$ facet at every other degree — and returns
    /// `residual(psi)`: \f$ \to 0 \f$ iff the targets lie in the carried
    /// period space, floored otherwise, with the leak the certificate.
    /// @throws std::runtime_error if `targetPeriods.size() != holes.size()`
    ///   or a hole is malformed (see `cyclePeriods`).
    [[nodiscard]] double residualForPeriods(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// **Arbitrary-degree** exact analytic gradient \f$ \partial r_U / \partial l^2_e \f$
    /// of `residualForPeriods` w.r.t. each edge's squared length, in `ChainComplex`
    /// 1-cell (edge) order, certified by the exact Euler identity
    /// \f$ \sum_e l^2_e\,\partial r_U/\partial l^2_e = -r_U \f$ at every register degree
    /// \f$ k \ge 1 \f$ (finite difference is roundoff-limited and does not converge).
    /// At \f$ k = 1 \f$ it routes through the **fast** low-rank edge-loop core
    /// (`periodGradientOverLoops`, which builds the chain complex once) so a relaxation
    /// loop stays affordable; at \f$ k \ge 2 \f$ it uses the degree-generic
    /// `periodGradientGeneral`. The two are value-identical at \f$ k = 1 \f$ (verified
    /// to 1.7e-15). At \f$ k = 0 \f$ the gradient runs against the genuinely
    /// **complex Hermitian** vertex operator \f$ L_0 = D - A \f$ (full \f$ l^2 \f$ and
    /// U(1) phases — `periodGradientDegreeZero`, #589); \f$ L_0 \f$ is homogeneous of
    /// degree **+1** in \f$ l^2 \f$, so the \f$ k = 0 \f$ Euler identity is
    /// \f$ \sum_e l^2_e\,\partial r_U/\partial l^2_e = +2\,r_U \f$.
    /// @throws std::runtime_error if `targetPeriods.size() != holes.size()`.
    [[nodiscard]] std::vector<double> residualForPeriodsGradient(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;


    /// The **carried representative** \f$ \psi \f$ that `residualForPeriods`
    /// scores — exposed as a cochain in its own right (the read-out
    /// `residualForPeriods` builds internally but does not return). Builds the
    /// period matrix \f$ P \f$ (`cyclePeriods`), least-squares-projects
    /// `targetPeriods` onto the carried period rows (minimum-norm, as
    /// `numpy.linalg.lstsq`), forms the harmonic combination
    /// \f$ \psi = \sum_r c_r h_r \f$, and attaches each hole's uncarried
    /// remainder (the minimal leak) to the hole's first walk-order facet, so the
    /// returned cochain's periods are exactly `targetPeriods`. A full
    /// `order()`-length cell vector; `residual` of it is `residualForPeriods`.
    /// @throws std::runtime_error if `targetPeriods.size() != holes.size()` or a
    ///   hole is malformed (see `cyclePeriods`).
    [[nodiscard]] std::vector<std::complex<double>> carriedRepresentative(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // === Periods over arbitrary signed edge-loops (#363) ===
    // A removed triangle reads only its own boundary, but a register cycle (e.g.
    // a torus S^1) is no triangle's boundary. These read the period of the live
    // harmonics over ANY closed walk of oriented edges, so both cycles of a T^2
    // qubit register are pinnable.

    /// A 1-cycle as a closed walk of directed `Edge`s: each edge's
    /// `getSource() -> getTarget()` is the traversal step, contributing
    /// \f$ +h(u,v) \f$ when \f$ u < v \f$ (source id < target id) else
    /// \f$ -h(u,v) \f$. A removed triangle \f$ h_0 < h_1 < h_2 \f$ is the loop
    /// \f$ h_0 \to h_1 \to h_2 \to h_0 \f$ (the identical signed covector and
    /// leak edge). Only the endpoints are read; the edges' lengths are unused.
    using EdgeLoop = std::vector<Edge>;

    /// `cyclePeriods` over signed edge-loops: a flat row-major
    /// \f$ \dim\ker L_k \times |\text{loops}| \f$ matrix whose
    /// \f$ [r\,|\text{loops}|+q] \f$ entry is harmonic \f$ r \f$ summed along
    /// loop \f$ q \f$. Reads the live harmonics.
    [[nodiscard]] std::vector<std::complex<double>> cyclePeriodsOverLoops(
        const std::vector<EdgeLoop> &loops) const;

    /// `residualForPeriods` over signed edge-loops: \f$ \to 0 \f$ iff
    /// `targetPeriods` lie in the span the live harmonics carry over `loops`.
    /// @throws std::runtime_error if `targetPeriods.size() != loops.size()`.
    [[nodiscard]] double residualForLoops(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// Exact analytic gradient of `residualForLoops` w.r.t. each edge's squared
    /// length, in `cellSimplices()` order — the shared core of
    /// `residualForPeriodsGradient`.
    /// @throws std::runtime_error if `targetPeriods.size() != loops.size()`.
    [[nodiscard]] std::vector<double> residualForLoopsGradient(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// The carried representative over signed edge-loops (the loop analogue of
    /// `carriedRepresentative`): the metric harmonic 1-cochain matching
    /// `targetPeriods` over `loops` (minimum-norm), a full `order()`-length cell
    /// vector. The whole-W metric harmonic the L_1(W) read-out rides on.
    [[nodiscard]] std::vector<std::complex<double>> carriedRepresentativeOverLoops(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// The periods of a GIVEN 1-cochain over signed edge-loops:
    /// \f$ \sum_{(u,v)\in\text{loop}} \pm\,\text{cochain}[uv] \f$ — the discrete
    /// \f$ \oint \f$ of a supplied cochain (vs the live harmonics in
    /// `cyclePeriodsOverLoops`). `cochain` is an `order()`-length cell vector.
    [[nodiscard]] std::vector<std::complex<double>> periodsOfCochainOverLoops(
        const std::vector<std::complex<double>> &cochain,
        const std::vector<EdgeLoop> &loops) const;

    // === The hard period-pin r_ψ (the realizability alternative, #377) ===
    // `residualForLoops`/`residualForPeriods` (r_U) build the EXACT-period state
    // (a minimal leak) and score its NON-harmonicity \f$ \|L\psi-\lambda\psi\|^2 \f$.
    // These instead keep the carried object a PURE harmonic and score the period
    // GAP it cannot match, \f$ r_\psi = \|P^{\top}c - \text{target}\|^2 \f$ with
    // \f$ c \f$ the least-squares fit — the §5.0 hard period-pin (#353's `r_ψ`).
    // Same zero set as r_U (\f$ \to 0 \f$ iff the target lies in the carried
    // period span), different off-zero shape and gradient. Selectable in
    // `MergeCobordism` for comparison; r_U is the default.

    /// The hard period-pin residual over signed edge-loops:
    /// \f$ r_\psi = \|\,P^{\top} c - \text{target}\,\|^2 \f$, where the columns of
    /// \f$ P^{\top} \f$ are the live harmonics' periods over `loops` and
    /// \f$ c = (P^{\top})^{+}\,\text{target} \f$ is their least-squares fit — the
    /// squared norm of the part of `targetPeriods` no pure harmonic can carry.
    /// \f$ \to 0 \f$ iff the target lies in the carried period span (the same
    /// realizable set as `residualForLoops`), floored otherwise; **no leak**, so
    /// the carried object stays a pure harmonic (vs r_U's exact-period state).
    /// @throws std::runtime_error if `targetPeriods.size() != loops.size()`.
    [[nodiscard]] double periodGapForLoops(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// Exact analytic gradient \f$ \partial r_\psi / \partial l^2_e \f$ of
    /// `periodGapForLoops` w.r.t. each edge's squared length, in `cellSimplices()`
    /// order. The least-squares optimality \f$ P^{\top\!}\,r = 0 \f$ (envelope
    /// theorem) kills the \f$ \partial c \f$ term, so
    /// \f$ \partial r_\psi = 2\,\Re\big(r^{\dagger}\,(Q\,\partial U_n)\,c\big) \f$
    /// — only the harmonic-subspace perturbation \f$ \partial U_n \f$ enters
    /// (no leak, no \f$ \partial\psi \f$ chain). Same first-order eigenvector
    /// perturbation machinery as `residualForLoopsGradient`. \f$ O(n_1^3) \f$.
    /// @throws std::runtime_error if `targetPeriods.size() != loops.size()`.
    [[nodiscard]] std::vector<double> periodGapForLoopsGradient(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// `periodGapForLoops` over removed-triangle holes (the r_ψ analogue of
    /// `residualForPeriods`): each hole is a 3-vertex tuple whose oriented
    /// boundary \f$ h_0\to h_1\to h_2\to h_0 \f$ is the loop. Convenience +
    /// Python entry point.
    /// @throws std::runtime_error if `targetPeriods.size() != holes.size()` or a
    ///   hole has \f$ \ne 3 \f$ vertices.
    [[nodiscard]] double periodGapForPeriods(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    /// The analytic gradient of `periodGapForPeriods` (holes routed to
    /// `periodGapForLoopsGradient`), in `cellSimplices()` order.
    /// @throws std::runtime_error if `targetPeriods.size() != holes.size()` or a
    ///   hole has \f$ \ne 3 \f$ vertices.
    [[nodiscard]] std::vector<double> periodGapForPeriodsGradient(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // === The discovered operator: ker L₁(W − ∂W) (#363) ===

    /// The interior 1-cells of \f$ W - \partial W \f$ — the edges both of whose
    /// endpoints are **interior** vertices (on no \f$ \partial W \f$ face) — as
    /// sorted \f$ (u, v) \f$ tuples in canonical `ChainComplex` \f$ C_1 \f$ order:
    /// the column ordering of `bulkMinusBoundaryHarmonicMatrix`. Empty when there
    /// is no interior bulk (a bare, un-grown cobordism is all boundary).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> bulkMinusBoundaryCells()
        const;

    /// \f$ \ker L_1(W - \partial W) \f$ — the harmonic 1-forms of the
    /// **combinatorial** (unit-weight, signature-blind) Hodge Laplacian
    /// \f$ L_1 \f$ of the bulk with the **full \f$ \partial W \f$ subcomplex
    /// deleted**: the subcomplex induced on the interior vertices (the bulk
    /// "with the boundary removed"). Restricts the integer boundary maps
    /// \f$ \partial_1, \partial_2 \f$ (`ChainComplex`) to the interior cells and
    /// eigendecomposes \f$ L_1 = \partial_1^{\top}\partial_1 +
    /// \partial_2\partial_2^{\top} \f$, returning the \f$ |\lambda| < \text{tol} \f$
    /// eigenvectors stacked as the **rows** of a flat row-major
    /// \f$ \dim\ker L_1 \times |\text{interior } C_1| \f$ complex array
    /// (ascending-eigenvalue order), columns in `bulkMinusBoundaryCells()` order.
    /// This is the geometry the **discovered operator** is read from: surgery
    /// must first grow the interior so this is nonzero (a bare cobordism with
    /// only a handful of interior vertices carries 0). Read fresh from the live
    /// complex — surgery between calls moves it.
    [[nodiscard]] std::vector<std::complex<double>> bulkMinusBoundaryHarmonicMatrix(
        double tol = 1e-9) const;

    // === Surgery: the topology-changing interior remove move (#196) ===

    /// The interior top cells eligible for surgery removal: top cells whose
    /// vertices are **all interior** (on no \f$ \partial W \f$ face), as sorted
    /// vertex-id tuples. Removing such a cell (`removeInteriorCell`) cannot touch
    /// \f$ \partial W \f$ — none of its faces is a boundary face — so it is the
    /// boundary-fixed **topology-CHANGING** move: unlike `growInterior`'s
    /// topology-preserving stellar subdivision and the purely additive
    /// `attachInteriorVertex`, removing a cell can open a hole / handle, so
    /// \f$ b_k \f$ of the complex **moves** (the emergent topology the experiment
    /// reads off the witness). The candidate pool a surgery search enumerates.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> interiorTopCells() const;

    /// Surgery (#196): remove the interior top cell `cell` (a sorted vertex-id
    /// tuple from `interiorTopCells()`) together with any of its edges left
    /// **orphaned** — belonging to no remaining top cell — so the result stays a
    /// valid downward-closed complex. This is **topology-changing**: removing the
    /// cell opens a hole/handle, so \f$ b_k \f$ **moves** (e.g. a filled disk
    /// \f$ b_1\!=\!0 \f$ becomes an annulus \f$ b_1\!=\!1 \f$). The pinned boundary
    /// \f$ \partial W \f$ is held bit-exact: the cell has no boundary vertex
    /// (so no \f$ \partial W \f$ face is removed) and the move is **rejected**
    /// (complex unchanged) if any \f$ \partial W \f$ edge would vanish or change;
    /// the newly EXPOSED interior boundary (the opened hole) is allowed — that is
    /// the emergent surgery. Records the removal for `restoreLastRemoval`
    /// (try / score / roll back). Returns `false`, complex unchanged, if `cell`
    /// is not an interior top cell or the removal would touch \f$ \partial W \f$.
    bool removeInteriorCell(const std::vector<std::uint64_t> &cell);

    /// Undo the most recent `removeInteriorCell` (LIFO): re-create the removed top
    /// cell and the edges it orphaned, restoring their squared-lengths and phases
    /// bit-exactly, and re-capture. Returns `false` if there is no removal to undo.
    /// The surgery analogue of `detachLastInteriorVertex`.
    bool restoreLastRemoval();

    // === Gated topology moves: the checked cut and the composed stellar move ===

    /// The gated surgery cut: `removeInteriorCell(cell)`, then the dual-validity
    /// gate (`dualComplexValid`), rolled back via `restoreLastRemoval` when the
    /// cut violates the dual — the accept-a-move-only-while-the-dual-stays-valid
    /// composition the register / fill layers apply, as one move. Returns
    /// `{true, "ok"}` with the cut applied, or `{false, reason}` with the
    /// complex unchanged: either the cell is not a removable interior top cell
    /// (`removeInteriorCell` rejected it) or the gate verdict names the dual
    /// violation. The gate is rigorous for \f$ n \leq 3 \f$
    /// (`ChainComplex::dualComplexIsValid`); dimension-4 callers use explicit
    /// constructions, not gated moves.
    [[nodiscard]] std::pair<bool, std::string> removeInteriorCellChecked(
        const std::vector<std::uint64_t> &cell);

    /// The composed gated stellar move — the boundary-fixed interior
    /// \f$ 1 \to (d+1) \f$ subdivision built from the two surgery primitives:
    /// attach a fresh interior vertex onto `cell`'s facet fan
    /// (`attachInteriorVertex` with the \f$ d+1 \f$ codim-one facets —
    /// \f$ \partial W \f$ untouched, the new edges interior), remove the
    /// subdivided parent (`removeInteriorCell` — its facets keep two cofaces, so
    /// \f$ \partial W \f$ stays bit-exact), then gate on `dualComplexValid`,
    /// rolling back **both** in LIFO order (`restoreLastRemoval`, then
    /// `detachLastInteriorVertex`) on violation. Each accepted move adds exactly
    /// one interior vertex and preserves \f$ \ker L_k \f$ (the fan is homotopic
    /// to the cell it replaces).
    ///
    /// On acceptance the bulk's edges are re-pinned uniform — squared length 1,
    /// phase 0, the unit cochain metric the register / fill seeds are built
    /// with: the attach wires the fan edges through `createSimplexTracked`,
    /// whose tracked metric follows the endpoints' TIME rule (timelike
    /// \f$ l^2 \f$ on a time difference) rather than a causal cone placement.
    /// On all-same-time seeds that already yields spacelike unit edges, but the
    /// documented unit metric holds by construction, not by the default-time
    /// coincidence. Returns `{true, "ok"}` on acceptance, or `{false, reason}`
    /// with the complex unchanged (and no re-pin).
    [[nodiscard]] std::pair<bool, std::string> stellarSubdivideInterior(
        const std::vector<std::uint64_t> &cell);

    // === Charge sector: the E/B split of the field strength F ∈ Ω² (#417) ===
    // The field-strength is a 2-cochain F ∈ Ω². On a Lorentzian complex each
    // plaquette has a causal type, so F splits by it: the ELECTRIC part lives on
    // plaquettes carrying a timelike edge (one temporal leg, the discrete F_{0i});
    // the MAGNETIC part on purely-spacelike plaquettes (all-spatial, F_{ij}). This
    // is a read-out on the live degree-2 complex — it only classifies cells by
    // `Edge::isTimelike()` and partitions a supplied F, never mutating geometry.

    /// The electric/magnetic split of a field-strength 2-cochain `F` by the causal
    /// type of each plaquette. `electric` is `F` on plaquettes with a timelike edge
    /// (zero elsewhere); `magnetic` is `F` on purely-spacelike plaquettes (zero
    /// elsewhere); so `electric + magnetic == F` componentwise. `electricCells` /
    /// `magneticCells` are the disjoint, complete index lists into
    /// `cellSimplices()` (degree-2). A plaquette is electric iff any of its three
    /// edges in the live complex is `Edge::isTimelike()`, else magnetic.
    struct FieldStrengthSplit {
      std::vector<std::complex<double>> electric;  // F on timelike-leg plaquettes
      std::vector<std::complex<double>> magnetic;  // F on purely-spacelike ones
      std::vector<std::size_t> electricCells;      // E indices into cellSimplices()
      std::vector<std::size_t> magneticCells;      // B indices into cellSimplices()
    };

    /// Split a field-strength 2-cochain `F` (an `order()`-length cochain in the
    /// degree-2 `cellSimplices()` order) into its electric (timelike-leg) and
    /// magnetic (purely-spacelike) parts by the causal type of each plaquette's
    /// edges (`Edge::isTimelike()` on the live complex). The returned `electric`
    /// and `magnetic` are `order()`-length cochains agreeing with `F` on their own
    /// support and zero elsewhere (`electric + magnetic == F`), plus the disjoint
    /// index lists `electricCells` / `magneticCells` (their union is all cells).
    /// @throws std::runtime_error if `degree() != 2`, if `F.size() != order()`, or
    ///   if a plaquette's edge is missing from the complex.
    [[nodiscard]] FieldStrengthSplit fieldStrengthSplit(
        const std::vector<std::complex<double>> &F) const;

    /// The curvature 2-cochain \f$ F = dA \f$ built from a carried U(1) connection
    /// 1-cochain `A` by discrete coboundary: on each degree-2 cell \f$ (a,b,c) \f$
    /// (sorted) \f$ F = A(a,b) + A(b,c) - A(a,c) \f$, the induced-orientation signed
    /// edge sum the period read-out uses (`cyclePeriods`,
    /// `examples/cobordism/proton_observables.py:55-56`). `A` is a degree-1 cochain
    /// indexed in the canonical `ChainComplex` 1-cell order (length = the number of
    /// 1-cells/edges, i.e. `EigenstateSynthesis(st, 1).order()`); this instance is
    /// degree 2 and returns an `order()`-length 2-cochain. Because \f$ d\circ d = 0 \f$
    /// the result is gauge-invariant: a pure gauge \f$ A \to A + d\chi \f$ leaves
    /// `F` (hence its E/B split) unchanged.
    /// @throws std::runtime_error if `degree() != 2`, if `A.size()` is not the
    ///   number of 1-cells, or if a 2-cell's edge is not a 1-cell of the complex.
    [[nodiscard]] std::vector<std::complex<double>> curvatureFromConnection(
        const std::vector<std::complex<double>> &A) const;

    /// The discrete Gauss-law charge \f$ Q = \oint_S E \f$: the temporal-sector
    /// flux of the field-strength 2-cochain `F` through the closed surface
    /// \f$ S = \partial V \f$ bounding the worldtube region `V` of the
    /// `enclosedVertices` (the quark windows). `V` is the closed star — every
    /// degree-3 cell (tetrahedron) touching an enclosed vertex — and `S` its
    /// boundary 2-chain (interior faces shared by two `V`-cells cancel under the
    /// induced \f$ (-1)^j \f$ orientation, leaving the enclosing surface). `Q` is
    /// the orientation-signed sum of `F` over `S`, restricted to the ELECTRIC
    /// plaquettes (a timelike leg, the discrete \f$ F_{0i} \f$; `fieldStrengthSplit`'s
    /// `electricCells`) when `electricOnly`, else the full flux. Because
    /// \f$ F = d\psi \f$ is exact, the full flux is \f$ \langle d\psi, \partial V\rangle
    /// = \langle \psi, \partial^2 V\rangle = 0 \f$ to round-off — the topological
    /// protection (metric-free \f$ d \f$): a genuine gauged-U(1) holonomy that does
    /// not drift under metric jitter, unlike a hand-weighted flavor covector. On an
    /// all-spacelike (Riemannian) complex no plaquette is electric, so the electric
    /// `Q` is exactly `0` (the neutral total of the reduced color-only sector). A
    /// degree-2 read-out: it classifies cells by `Edge::isTimelike()` and sums a
    /// supplied `F`, never mutating geometry.
    /// @throws std::runtime_error if `degree() != 2` or `F.size() != order()`.
    [[nodiscard]] std::complex<double> gaussLawCharge(
        const std::vector<std::complex<double>> &F,
        const std::vector<std::uint64_t> &enclosedVertices,
        bool electricOnly = true) const;

  private:
    std::shared_ptr<Spacetime> st_;
    int k_{0};  // the Hodge degree of L_k that apply()/residual() score against
    // The Hodge Laplacian operator over the same complex. laplacian(k_)
    // reassembles L_k from the live edges/volumes on each call, so perturbing the
    // edge squared-lengths and re-querying is honest (the eigendecomposition
    // cache is untouched by the matrix path).
    HodgeLaplacian laplacian_;
    std::size_t order_{0};  // N = operator dimension (|V| at k=0, else |C_k|)
    // The sorted vertex-id tuple of each psi component, in operator order: the
    // sorted-id vertices at k=0, else the ChainComplex k-cell column order.
    // Re-captured after growInterior() mutates the complex.
    std::vector<std::vector<std::uint64_t>> cellOrdering_{};
    // The tunable edges, in EdgeList order, restricted to those carrying weight
    // in L (both endpoints present, no self-loops). Raw pointers owned by the
    // EdgeList; valid for the complex's lifetime (kept alive via st_). Re-captured
    // after growInterior() adds a vertex.
    std::vector<::tessera::mesh::Edge *> edges_{};

    // Indices into edges_ partitioning the tunable edges into interior (free) and
    // boundary (on ∂W, held fixed) by classifyBoundary(). Interior-order is the
    // stable parameter order for setInteriorWeights / setInteriorPhases.
    std::vector<std::size_t> interiorEdgeIdx_{};
    std::vector<std::size_t> boundaryEdgeIdx_{};
    std::size_t interiorVertexCount_{0};  // vertices on no boundary face
    // Boundary (∂W) vertex ids, sorted ascending — persisted from
    // classifyBoundary() for boundaryVertexIds().
    std::vector<std::uint64_t> boundaryVertexIdsSorted_{};

    // One attachInteriorVertex() record, for exact rollback (detach) and for the
    // boundary-fixed connectivity search. Raw pointers owned by the Spacetime
    // (stable-address storage), valid for its lifetime (kept alive via st_).
    struct Attachment {
      ::tessera::mesh::Vertex *vertex{nullptr};
      std::vector<::tessera::mesh::Edge *> createdEdges{};
      std::vector<::tessera::mesh::Simplex *> createdSimplices{};
    };
    std::vector<Attachment> attachments_{};

    // One removeInteriorCell() record, for exact restore. The removed top cell's
    // sorted vertex tuple (its vertices are kept — only the top simplex and its
    // orphaned edges are deleted), plus each orphaned edge as (u, v, complex
    // squaredLength, phase) so restoreLastRemoval re-creates them bit-exactly —
    // the FULL complex ℓ², never its real part alone (#581).
    struct Removal {
      std::vector<std::uint64_t> cell{};
      std::vector<std::tuple<std::uint64_t, std::uint64_t,
                             std::complex<double>, double>>
          removedEdges{};
    };
    std::vector<Removal> removals_{};

    // The shared register read-out assembly (#286): the fresh harmonic matrix
    // H (flat row-major dim × order(), HodgeLaplacian::harmonicMatrix on the
    // live complex), the period matrix P (flat row-major dim × |holes|, the
    // boundary-operator-signed facet sums), and each hole's leak column (the
    // cochain index its uncarried remainder attaches to). Backs cyclePeriods
    // and residualForPeriods.
    struct RegisterReadout {
      std::vector<std::complex<double>> H{};
      std::vector<std::complex<double>> P{};
      std::vector<std::size_t> leakColumns{};
      std::size_t dim{0};
    };
    [[nodiscard]] RegisterReadout assembleRegisterReadout(
        const std::vector<std::vector<std::uint64_t>> &holes) const;

    // The loop analogue of assembleRegisterReadout: P[r*m+q] = sum over loop q's
    // oriented edges of (+/-1) * H[r, edge]; leak = the loop's first edge.
    [[nodiscard]] RegisterReadout assembleReadoutOverLoops(
        const std::vector<EdgeLoop> &loops) const;

    // The minimum-norm least-squares fit of targetPeriods onto the carried
    // period rows P^T (c = (P^T)^+ target via the SVD) — the single projection
    // both the carried representative (r_U) and the period gap (r_psi) ride on, so
    // their realizable zero sets coincide. Length ro.dim (empty when ro.dim == 0).
    [[nodiscard]] std::vector<std::complex<double>> lstsqOverReadout(
        const RegisterReadout &ro,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // The carried representative from a finished read-out — shared by the hole
    // and loop period paths (the lstsq projection plus each cycle's leak).
    [[nodiscard]] std::vector<std::complex<double>> carriedFromReadout(
        const RegisterReadout &ro,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // The cycle-agnostic core of the period-residual gradient: exact
    // d r_U / d l^2 with the cycles given as signed edge-loops.
    [[nodiscard]] std::vector<double> periodGradientOverLoops(
        const std::vector<EdgeLoop> &loops,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // Degree-generic d r_U / d l^2 over removed-(k+1)-cell holes (M = L_k, the
    // per-edge analytic dL_k/dl^2). The k = 0 and k >= 2 path of
    // residualForPeriodsGradient (k = 0 dispatches to periodGradientDegreeZero);
    // value-identical to periodGradientOverLoops at k = 1.
    /// Degree-generic exact `∂r_ψ/∂ℓ²` for the period GAP
    /// `r_ψ = ‖A c − t‖²` (`A = Q·U_n`, `c` the least-squares fit), in
    /// ChainComplex 1-cell order — the `k ≥ 2` sibling of the edge-loop core
    /// `periodGapForLoopsGradient`, which `periodGapForPeriodsGradient` routes
    /// to at `k = 1`.
    ///
    /// Least-squares optimality (`Aᴴr = 0`) drops the `∂c` term by the envelope
    /// theorem, leaving `2 Re(rᴴ (Q ∂U_n) c)`. The fit is the SVD pseudo-inverse,
    /// matching the value's `lstsqOverReadout` and staying defined when `A` is
    /// rank-deficient. The null tolerance is `harmonicMatrix`'s `1e-9`, so the
    /// harmonic set differentiated is the one the value reads.
    ///
    /// Certified by the degree-0 Euler identity `Σ ℓ²·∂r_ψ/∂ℓ² = 0`: `L_k` is
    /// homogeneous of degree −1, so a uniform rescale leaves the kernel — and
    /// therefore `A`, `c` and the gap — unchanged. Throws at `k = 0`.
    [[nodiscard]] std::vector<double> periodGapGradientOverHoles(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    [[nodiscard]] std::vector<double> periodGradientGeneral(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // Exact d r_U / d l^2 at k = 0 against the genuinely COMPLEX Hermitian
    // vertex operator L_0 = D - A (D_ii = sum |l^2|, A_ij = l^2 e^{i phase};
    // the k >= 1 cores' laplacian(k).real() projection would be the WRONG
    // operator here, #580/#589). A hole is a removed 1-cell: a vertex pair
    // whose drop-v_j facets carry (-1)^j, the assembleRegisterReadout
    // convention. The least-squares fit uses the SVD pseudo-inverse (matching
    // lstsqOverReadout) and its constant-rank derivative — at k = 0 the
    // period matrix is generically rank-deficient (a globally gauge-flat
    // harmonic has zero period on every hole), so the k >= 1 cores'
    // normal-equations inverse would be singular. Euler identity at k = 0:
    // Sum_e l^2_e d r_U/d l^2_e = +2 r_U (L_0 is homogeneous of degree +1 in
    // l^2, vs the degree -1 metric L_k at k >= 1 whose identity is -r_U).
    [[nodiscard]] std::vector<double> periodGradientDegreeZero(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const std::vector<std::complex<double>> &targetPeriods) const;

    // Each removed-triangle hole (a 3-vertex tuple) as the oriented boundary loop
    // h0 -> h1 -> h2 -> h0, so the holes period-gap methods reuse the loop core.
    [[nodiscard]] std::vector<EdgeLoop> holeLoops(
        const std::vector<std::vector<std::uint64_t>> &holes,
        const char *who) const;

    // Re-create the top cell of a Removal (createSimplexTracked rebuilds its
    // missing edges = the orphaned ones) and restore those edges' weights/phases.
    // Does not re-capture (callers do). Returns false if a cell vertex is gone.
    bool applyRestore(const Removal &rem);

    // Remove everything an attachment created (its simplices, then its freshly
    // inserted edges, then its vertex) — the inverse of attachInteriorVertex's
    // mesh mutation. Does not re-capture (callers do).
    void rollbackAttachment(const Attachment &att);

    // (Re)build order_, cellOrdering_ and edges_ from the live complex (the
    // k-cell order at k_, plus the tunable edges). Called at construction and
    // after growInterior() mutates the complex.
    void capture();

    // (Re)build the interior/boundary edge partition (∂W = codim-1 faces in
    // exactly one top cell) and interiorVertexCount_ from the live complex.
    void classifyBoundary();
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H
