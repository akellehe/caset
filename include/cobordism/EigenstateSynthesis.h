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
#include <set>
#include <utility>
#include <vector>

#include "cobordism/HodgeLaplacian.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh { class Edge; }
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # EigenstateSynthesis
///
/// The §4b inverse eigenvector problem on a **fixed** complex: given a target
/// state \f$ \psi \f$, score how close the complex's current Hermitian edge
/// weights make \f$ \psi \f$ to being an eigenvector of the \f$ k=0 \f$ Hodge
/// Laplacian \f$ L = D - A \f$ (the magnitude convention, via
/// `HodgeLaplacian`), and read/write those weights so a search can perturb them.
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
    /// Construct over a fixed triangulation at cochain `degree` (\f$ k \f$). The
    /// vertex order (sorted id) and the tunable edge order are captured now; edge
    /// weights/phases are read live on each residual query. The held `shared_ptr`
    /// keeps the spacetime alive.
    ///
    /// `degree = 0` (default) is the §4b graph-Laplacian regime: a `psi` is a
    /// 0-cochain (vertex amplitudes, length \f$ |V| \f$) and `residual` scores it
    /// as an **eigenvector** of \f$ L_0 = D - A \f$. `degree = 1` is the §5.2
    /// regime for the DW bridge: a `psi` is a 1-cochain (edge amplitudes, length
    /// \f$ |C_1| \f$ in the `HodgeLaplacian`/`ChainComplex` column order) and
    /// `residual` scores it as a **harmonic** form (\f$ \ker L_1 \f$, the metric
    /// Hodge Laplacian) — \f$ r = \lVert L_1\psi\rVert^2 \f$. At \f$ k\ge 1 \f$ the
    /// real metric Laplacian is built from the simplex volumes
    /// (`Simplex::volume`, live in the edge lengths), so the tunable **weights**
    /// shape \f$ L_k \f$ while the U(1) **phases** do not enter it.
    /// @throws std::runtime_error if `degree < 0`.
    explicit EigenstateSynthesis(std::shared_ptr<Spacetime> st, int degree = 0);

    /// The cochain degree \f$ k \f$ this synthesizer scores at (0 = vertices /
    /// eigenvector; 1 = edges / harmonic).
    [[nodiscard]] int degree() const noexcept { return degree_; }

    /// The required length of any `psi`: the number of \f$ k \f$-cells —
    /// \f$ |V| \f$ at \f$ k = 0 \f$, \f$ |C_k| \f$ at \f$ k \ge 1 \f$ (the
    /// `HodgeLaplacian` operator dimension at this degree).
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

    /// Number of vertices \f$ N = |V| \f$ — the \f$ k = 0 \f$ `psi` length and the
    /// vertex budget for the output-boundary support (`= dimension()` at
    /// \f$ k = 0 \f$).
    [[nodiscard]] std::size_t order() const noexcept { return order_; }

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

    /// The `psi`-component indices (in the operator / `dimension()` order) whose
    /// \f$ k \f$-cell lies on \f$ \partial W \f$ — the **boundary support** a
    /// target boundary state occupies. At \f$ k = 1 \f$ these are the boundary
    /// edges of \f$ \partial W \f$ (a target harmonic 1-form is carried here); at
    /// \f$ k = 0 \f$ the boundary vertices. The complement
    /// (`interiorStateIndices()`) carries the free auxiliary amplitudes a
    /// fixed-boundary fill solves for. Returned ascending.
    [[nodiscard]] std::vector<std::size_t> boundaryStateIndices() const;

    /// The `psi`-component indices (operator order) **not** on \f$ \partial W \f$
    /// — the interior support (free auxiliary amplitudes); the complement of
    /// `boundaryStateIndices()` in \f$ [0, \text{dimension}()) \f$. Returned
    /// ascending.
    [[nodiscard]] std::vector<std::size_t> interiorStateIndices() const;

    /// Cone a fresh interior vertex into a top cell via the boundary-fixed
    /// pre-geometric Pachner add (#112): a \f$ 1\!\to\!(d+1) \f$ stellar
    /// subdivision that leaves \f$ \partial W \f$ exactly fixed while enriching the
    /// interior. Re-captures the vertex order, tunable edges, and interior/boundary
    /// partition, so `order()` grows by one (a `psi` must be extended on the new
    /// apex, appended last in sorted-id order) and `numInteriorEdges()` grows. The
    /// RNG `seed` makes the target-cell choice reproducible. Returns `false` if no
    /// top cell can be subdivided (e.g. a 1-complex, top cells of \f$ <3 \f$
    /// vertices), leaving the complex unchanged.
    bool growInterior(std::uint64_t seed);

  private:
    std::shared_ptr<Spacetime> st_;
    int degree_{0};  // cochain degree k the residual scores at
    // The Hodge Laplacian operator over the same complex. laplacian(degree_)
    // reassembles L_k from the live edges (k=0: L = D - A; k>=1: the metric
    // Hodge Laplacian from the live simplex volumes) on each call, so perturbing
    // the edges and re-querying is honest (the eigendecomposition cache is
    // untouched by the matrix path).
    HodgeLaplacian laplacian_;
    std::size_t order_{0};       // N = |V|, the sorted-id vertex order
    std::size_t dimension_{0};   // psi length = |k-cells| at degree_
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
    // The ∂W edge key set ((min,max) endpoint ids), shared by classifyBoundary()
    // (edge-parameter partition) and captureDegree() (psi-component partition).
    std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdgeKeys_{};

    // The operator-order k-cells (degree_), the index space a psi lives over
    // (ChainComplex column order); used to partition psi components into the
    // boundary support and the free interior support.
    std::vector<std::vector<std::uint64_t>> stateSimplices_{};
    std::vector<std::size_t> boundaryStateIdx_{};  // psi components on ∂W
    std::vector<std::size_t> interiorStateIdx_{};  // psi components off ∂W

    // (Re)build order_ and edges_ from the live vertex/edge lists. Called at
    // construction and after growInterior() mutates the complex.
    void capture();

    // (Re)build the interior/boundary edge partition (∂W = codim-1 faces in
    // exactly one top cell), boundaryEdgeKeys_, and interiorVertexCount_ from the
    // live complex.
    void classifyBoundary();

    // (Re)build dimension_, stateSimplices_, and the psi-component boundary /
    // interior partition for degree_ from the live complex (after classifyBoundary
    // has populated boundaryEdgeKeys_). Called at construction and after growth.
    void captureDegree();
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H
