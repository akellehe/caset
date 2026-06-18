// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef TESSERA_CUDA
#include "cuda/regge_cuda.h"
#endif

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
  class PachnerMove;
}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;


/// Regge equation solver.
///
/// Given a triangulated spacetime and a matter configuration, adjusts the
/// edge lengths so that the deficit angles at every hinge satisfy the
/// discretized Einstein equations:
///
/// \f[
///   \frac{\partial S}{\partial \ell^2_e} = 0
/// \f]
///
/// The total action is:
/// \f[
///   S = \underbrace{\sum_h A_h\,\varepsilon_h}_{S_{\text{grav}}}
///     \underbrace{- M \sum_{e \in W} \sqrt{-\ell^2_e}}_{S_{\text{matter}}}
/// \f]
/// (Timelike edges have \f$\ell^2 < 0\f$; spacelike edges have \f$\ell^2 > 0\f$.)
///
/// ## Algorithm
///
/// 1. Compute the Gram matrix \f$G\f$ of each top-simplex from its edge
///    lengths.
/// 2. Derive dihedral angles from the cofactors of \f$G\f$.
/// 3. Sum dihedral angles at each hinge → actual deficit angles.
/// 4. Minimize \f$F = \|\nabla S\|^2 = \sum_e (\partial S/\partial \ell^2_e)^2\f$
///    by gradient descent.  \f$F = 0\f$ exactly at a stationary point of
///    the action, i.e. when the Regge equations are satisfied.
///    (Minimizing \f$S\f$ directly diverges because it is unbounded below.)
///
class ReggeSolver {
  public:
    ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                MatterConfiguration matter);

    // ==================== Geometry queries ====================

    /// Dihedral angle at hinge \a h within top-simplex \a sigma.
    [[nodiscard]] double dihedralAngle(SimplexPtr sigma,
                                        SimplexPtr hinge) const;

    /// Deficit angle at a hinge: \f$\varepsilon_h = 2\pi - \sum_\sigma \theta_h^{(\sigma)}\f$.
    [[nodiscard]] double deficitAngle(SimplexPtr hinge) const;

    /// Area of a triangular hinge (for the Regge action weighting).
    [[nodiscard]] static double hingeArea(SimplexPtr hinge);

    /// Gravitational Regge action: \f$S_{\text{grav}} = \sum_h A_h\,\varepsilon_h\f$.
    [[nodiscard]] double reggeAction() const;

    /// **Dual** Lorentzian Regge action on \f$W^*\f$:
    /// \f$S_{\text{Regge}}(W^*) = \sum_h |\!\star\! h|\,\varepsilon_h\f$, the
    /// circumcentric **dual** content of each (d-2)-hinge
    /// (``Simplex::dualVolume``) weighted by its **complex Lorentzian** deficit
    /// (``Simplex::lorentzianDeficitAngle``). Returns ``std::complex``: the real
    /// part is the angle-defect curvature, the imaginary part the boost
    /// (rapidity / light-cone) content from timelike-normal-plane hinges. Pure
    /// gravity (matter-independent); the gravitational prior for the
    /// Regge-mediated synthesis objective. Refs: Regge (1961);
    /// Ambjorn-Jurkiewicz-Loll (Lorentzian CDT); Sorkin (Lorentzian angles);
    /// Asante-Dittrich arXiv:2104.00485.
    [[nodiscard]] std::complex<double> dualReggeAction() const;

    /// Point-particle matter action: \f$S_{\text{matter}} = -M \sum_{e \in W} \sqrt{-\ell^2_e}\f$.
    [[nodiscard]] double matterAction() const;

    /// Total action: \f$S = S_{\text{grav}} + S_{\text{matter}}\f$.
    /// The Regge equations are \f$\partial S/\partial \ell^2_e = 0\f$.
    [[nodiscard]] double totalAction() const;

    /// Squared gradient norm: \f$F = \sum_e (\partial S/\partial \ell^2_e)^2\f$.
    /// Non-negative; zero exactly when the Regge equations are satisfied.
    [[nodiscard]] double actionGradientNorm() const;

    /// **Exact analytic** gradient of the complex dual (Sorkin) Regge action
    /// ``dualReggeAction`` = Σ_h |★h|·ε_h: ∂S/∂ℓ²_e for each edge, in the
    /// ``getEdgeList()`` order (matching ``actionGradient``). Assembled by the
    /// product rule Σ_h [∂|★h|·ε_h + |★h|·∂ε_h] from the per-hinge analytic
    /// gradients ``Simplex::dualVolumeGradient`` and
    /// ``lorentzianDeficitAngleGradient`` — no finite differences. Complex (Re S
    /// and Im S together). Matches a central-difference of ``dualReggeAction`` to
    /// machine precision but at one pass, not 2·|E| action evaluations.
    [[nodiscard]] std::vector<std::complex<double>> actionGradientExact() const;

    /// Exact analytic Hessian ∂²S/∂ℓ²_e∂ℓ²_f of the dual Lorentzian Regge action,
    /// as a dense |E|×|E| complex matrix in the ``getEdgeList`` order. The next
    /// product-rule layer beyond ``actionGradientExact``:
    /// Σ_h [∂²|★h|·ε_h + ∂|★h|_e·∂ε_h_f + ∂|★h|_f·∂ε_h_e + |★h|·∂²ε_h], assembled
    /// from the per-hinge ``dualVolumeHessian`` / ``lorentzianDeficitAngleHessian``
    /// (and their gradients) — no finite differences. Removes the FD-Hessian floor
    /// in the stationary-action relaxation (exact Newton / Gauss-Newton).
    [[nodiscard]] std::vector<std::vector<std::complex<double>>>
    actionHessianExact() const;

    // ============ Incremental gradient (Pachner-local) ============
    //
    // ``actionGradientExact`` recomputes the whole gradient over every hinge
    // (``O(H)``) on each call. The methods below keep a **resident** gradient
    // ``∂S/∂ℓ²_e`` and the resident dual action ``S`` and update them
    // **incrementally** under a transactional Pachner move: only the hinges in
    // the move's touched neighborhood are re-evaluated, so a boundary-fixed move
    // costs ``O(#changed hinges)`` rather than ``O(H)``. This is the per-move
    // cost that gates a combinatorial search over triangulations (greedy /
    // annealing / RL).

    /// (Re)build the resident gradient and resident dual action from scratch
    /// over all hinges. Must be called once before the first
    /// ``applyMoveIncremental`` / ``rollbackMoveIncremental`` /
    /// ``applyLengthChangeIncremental`` (it establishes the baseline the
    /// per-update deltas maintain). Idempotent.
    ///
    /// After any single update the resident equals a from-scratch
    /// ``actionGradientExact`` on the *current* complex to machine precision.
    /// The deltas do carry state, though, so a long trajectory that passes
    /// through a near-degenerate geometry (a dual-volume pole, where the action
    /// itself is genuinely large) can lose low-order bits to catastrophic
    /// cancellation — the same numbers a from-scratch pass would also blow up
    /// on, but reconstructed exactly each time. Call this again to re-baseline
    /// (``O(H)``) if a search runs long enough for that to matter.
    void resetIncrementalGradient();

    /// The resident gradient ``∂S/∂ℓ²_e`` in ``getEdgeList()`` order — the same
    /// vector ``actionGradientExact`` returns, but maintained incrementally.
    /// Edges with no resident entry (none of their hinges contribute) read 0.
    [[nodiscard]] std::vector<std::complex<double>> incrementalGradient() const;

    /// The resident dual Lorentzian Regge action ``S = Σ_h |★h|·ε_h`` maintained
    /// alongside the gradient — tracks ``dualReggeAction`` after each move.
    [[nodiscard]] std::complex<double> incrementalAction() const noexcept {
        return residentAction_;
    }

    /// Subtract the touched region's old hinge contributions, commit
    /// ``move.apply()``, then add the region's new hinge contributions (the new
    /// cells' facets are materialized locally). After this returns, the resident
    /// gradient/action match a from-scratch ``actionGradientExact`` /
    /// ``dualReggeAction`` on the mutated complex to machine precision.
    /// ``move`` must have been successfully ``propose()``d. Requires a prior
    /// ``resetIncrementalGradient`` (throws ``std::logic_error`` otherwise).
    ///
    /// **Precondition — stable vertex IDs.** The resident gradient is keyed by
    /// vertex ID, so the move must not cosmetically *relabel* vertices: a
    /// ``swapVertexLabels`` re-keys edges across an arbitrary partner vertex's
    /// whole neighborhood (outside the touched region) and is neither local nor
    /// tracked here. Construct relabeling moves with their relabel toggle off
    /// (e.g. ``AddMove(..., relabel=false)``) — the same stable-ID regime the
    /// move tests use. Relabeling is a Markov-chain id randomization orthogonal
    /// to the geometry the gradient depends on.
    void applyMoveIncremental(::tessera::spacetime::PachnerMove &move);

    /// The ``rollback`` counterpart: subtract the touched region's current hinge
    /// contributions, replay ``move.rollback()``, then add the restored region's
    /// contributions. Restores the resident gradient/action to their pre-apply
    /// values to machine precision.
    void rollbackMoveIncremental(::tessera::spacetime::PachnerMove &move);

    /// Set edge \a edge's squared length to \a newSquaredLength, updating the
    /// resident gradient/action over only the edge's **coboundary** — the top
    /// cells containing it — in ``O(local)``. The geometric counterpart of
    /// ``applyMoveIncremental``: topology is unchanged (no edge keys added or
    /// removed), only the hinges sharing a top cell with this edge change.
    ///
    /// The dirty region is the union of those top cells' vertices, NOT just the
    /// two endpoints: a coface's *opposite* hinge (the (d-2)-face on the cell's
    /// other vertices) depends on the edge's length yet touches neither endpoint.
    ///
    /// After this returns the resident gradient/action match a from-scratch
    /// ``actionGradientExact`` / ``dualReggeAction`` to machine precision.
    /// Requires a prior ``resetIncrementalGradient`` (throws ``std::logic_error``
    /// otherwise). Changing an edge length does not touch vertex IDs, so there is
    /// no relabel caveat. To re-relax a local patch, call this per changed edge;
    /// each call is bounded by that edge's coface star.
    void applyLengthChangeIncremental(EdgePtr edge,
                                      std::complex<double> newSquaredLength);

    // ==================== Solver ====================

    /// One gradient-descent step minimizing \f$F = \|\nabla S\|^2\f$.
    ///
    /// @return F = \f$\|\nabla S\|^2\f$ before the update
    double step(double learningRate = 0.001);

    /// Iterate step() until convergence or max iterations.
    ///
    /// @param progress Optional callback invoked after each iteration with
    ///   (iteration, F).  Useful for progress bars.
    /// @return (converged, final_F, iterations)
    using ProgressCallback = std::function<void(int iter, double F)>;
    std::tuple<bool, double, int> solve(double tol = 1e-8,
                                         int maxIters = 5000,
                                         double learningRate = 0.001,
                                         ProgressCallback progress = nullptr);

    // ==================== Accessors ====================

    [[nodiscard]] const std::shared_ptr<Spacetime> &getSpacetime() const noexcept {
        return spacetime_;
    }

    [[nodiscard]] const MatterConfiguration& getMatter() const noexcept {
        return matter_;
    }

  private:
    std::shared_ptr<Spacetime> spacetime_;
    MatterConfiguration matter_;

    /// Collect all (d-2)-simplices (hinges) in the complex.
    [[nodiscard]] std::vector<SimplexPtr> collectHinges() const;

    /// Compute the gradient of the total action: ∂S/∂ℓ²_e for each edge.
    [[nodiscard]] std::vector<double> actionGradient() const;

    // ---- Incremental-gradient state + helpers ----

    using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

    /// Resident gradient ∂S/∂ℓ²_e keyed by the sorted (minId, maxId) edge — the
    /// same keying the per-hinge ``dualVolumeGradient`` /
    /// ``lorentzianDeficitAngleGradient`` maps use.
    std::map<EdgeKey, std::complex<double>> residentGradient_;
    /// Resident dual Regge action Σ_h |★h|·ε_h.
    std::complex<double> residentAction_{0.0, 0.0};
    /// True once ``resetIncrementalGradient`` has established the baseline.
    bool gradientResident_ = false;

    /// Add ``sign``·(this hinge's contribution) into the resident gradient and
    /// action: ε_h = ``lorentzianDeficitAngle``, |★h| = ``dualVolume``, and
    /// ∂(|★h|·ε_h)/∂ℓ²_e by the product rule — exactly the inner loop of
    /// ``actionGradientExact``.
    void accumulateHinge(SimplexPtr hinge, int sign);

    /// The hinges (deduplicated) that are a (d-2)-face of any top cell incident
    /// to one of \a vertexIds, materializing the facet/coface lattice of those
    /// tops as it walks. Superset of the hinges a local move can change; the
    /// subtract/add delta makes the unchanged ones cancel exactly.
    [[nodiscard]] std::vector<SimplexPtr>
    regionHinges(const std::vector<std::uint64_t> &vertexIds);

    /// The (sorted) edge keys incident to any of \a vertexIds in the current
    /// complex — used to prune entries for edges a move deletes.
    [[nodiscard]] std::set<EdgeKey>
    regionEdgeKeys(const std::vector<std::uint64_t> &vertexIds) const;

    /// Vertices of every top cell that contains the edge \a (u, v) — the dirty
    /// set whose incident hinges cover every hinge whose contribution depends on
    /// that edge's length (the endpoints alone miss each coface's opposite
    /// hinge).
    [[nodiscard]] std::vector<std::uint64_t>
    edgeCoboundaryVertexIds(VertexPtr u, VertexPtr v) const;

    /// Shared body of ``applyMoveIncremental`` / ``rollbackMoveIncremental``:
    /// subtract the region's contributions, run \a mutate (apply or rollback),
    /// add the region's contributions, then drop deleted edges' entries.
    void updateAround(const std::vector<std::uint64_t> &vertexIds,
                      const std::function<void()> &mutate);

#ifdef TESSERA_CUDA
    /// Flatten mesh topology into GPU-friendly arrays.
    [[nodiscard]] cuda::GpuMeshData flattenMeshForGpu() const;
#endif

};

} // namespace tessera::simulations
