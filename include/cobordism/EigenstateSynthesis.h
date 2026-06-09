// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh { class Edge; class Vertex; class Simplex; }
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

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
    [[nodiscard]] std::vector<double> weights() const;

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
    [[nodiscard]] std::vector<double> interiorWeights() const;

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
    // orphaned edges are deleted), plus each orphaned edge as (u, v, squaredLength,
    // phase) so restoreLastRemoval re-creates them bit-exactly.
    struct Removal {
      std::vector<std::uint64_t> cell{};
      std::vector<std::tuple<std::uint64_t, std::uint64_t, double, double>>
          removedEdges{};
    };
    std::vector<Removal> removals_{};

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
