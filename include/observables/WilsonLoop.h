// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include <map>
#include <complex>
#include <limits>
#include <memory>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;


/// Evaluation mode for Wilson loops, at increasing levels of geometric commitment.
enum class WilsonMode : uint8_t {
    COMBINATORIAL,   ///< Dual-graph topology only (loop length, enclosed hinges)
    DEFICIT_ANGLE,   ///< Uses deficit angles: W = ((d-2)+2cos(ε))/d
    CAUSAL,          ///< CDT causal orientation changes around the loop
    U1_CONNECTION    ///< U(1) connection holonomy: oriented Σ Edge::phase around a 1-skeleton cycle (mod 2π)
};

/// Which loop-shape generator to use.
enum class LoopType : uint8_t {
    HINGE,           ///< Elementary loop around a (d-2)-simplex
    DUAL_LATTICE,    ///< BFS-discovered loop of a target size
    GEODESIC         ///< Shortest cycle through a start simplex
};

/// A closed path through the dual graph: an ordered sequence of top-simplices
/// where consecutive simplices share a facet.
struct LoopPath {
    std::vector<SimplexPtr> simplices;  ///< ordered top-simplices
    std::vector<SimplexPtr> facets;     ///< facets[i] shared between simplices[i] and [i+1 mod n]
};

/// Result of evaluating a Wilson loop in any mode.
struct WilsonResult {
    /// Primary scalar. COMPLEX: in the deficit-angle mode the holonomy around a
    /// hinge is cos of the COMPLEX Lorentzian deficit — the boost part
    /// contributes a cosh, so |value| may exceed 1 and a mixed hinge yields a
    /// genuinely complex character. Real-valued modes fill the real part.
    ///
    /// In ``U1_CONNECTION`` mode this is a DERIVED VIEW of
    /// ``connectionAccumulation`` — its ``residualPhase()`` — kept because
    /// consumers read it as the mod-2π holonomy angle. The datum is the
    /// accumulation; this is one reading of it.
    std::complex<double> value{0.0, 0.0};

    /// The complete gauge-invariant datum of a ``U1_CONNECTION`` read: the
    /// UNREDUCED complex accumulation \f$ \Sigma_\gamma\varphi \f$ of the
    /// oriented ``Edge::phase`` around the cycle.
    ///
    /// Both components are carried. Around a CLOSED loop a gauge
    /// transformation \f$ \varphi\mapsto\varphi+d\chi \f$ telescopes to zero,
    /// so the whole complex sum is gauge-invariant — the imaginary part no
    /// less than the real one. Of the structure group
    /// \f$ \mathbb{C}^{*}=U(1)\times\mathbb{R}^{+} \f$ only the compact factor
    /// has winding, so only \f$ \mathrm{Re} \f$ quantizes; that makes
    /// \f$ e^{-\mathrm{Im}\Sigma} \f$ a gauge-invariant real rather than a
    /// quantum number, which is not a reason to discard it. If the
    /// non-compact direction is inert then \f$ \mathrm{Im}\Sigma\to 0 \f$ and
    /// the modulus tends to 1 — a cancellation to be OBSERVED, never imposed.
    ///
    /// Deliberately NOT reduced modulo 2π at accumulation time: reducing
    /// destroys the winding irrecoverably. ``holonomy()``,
    /// ``holonomyModulus()``, ``residualPhase()`` and ``windingNumber()`` are
    /// derived from this and must never replace it.
    ///
    /// NaN outside ``U1_CONNECTION`` mode — unmeasured, never zero.
    std::complex<double> connectionAccumulation{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};

    int    loopSize = 0;           ///< number of simplices in the loop
    int    enclosedHinges = 0;     ///< hinges enclosed (combinatorial)
    bool   contractible = true;    ///< is the loop contractible? (combinatorial)
    int    causalWindingNumber = 0;///< net orientation changes (causal)

    /// Fold an angle into the principal holonomy interval (−π, π].
    [[nodiscard]] static double principalAngle(double theta);

    /// The holonomy \f$ H(\gamma)=e^{i\Sigma} \f$, derived.
    [[nodiscard]] std::complex<double> holonomy() const;

    /// \f$ |H(\gamma)| = e^{-\mathrm{Im}\Sigma} \f$, derived. Exactly 1 when
    /// the connection is purely compact.
    [[nodiscard]] double holonomyModulus() const;

    /// \f$ \mathrm{Re}\Sigma \bmod 2\pi \f$ in (−π, π], derived.
    [[nodiscard]] double residualPhase() const;

    /// The whole \f$ 2\pi \f$ turns in \f$ \mathrm{Re}\Sigma \f$, derived.
    /// Recoverable only because the accumulation is stored unreduced.
    [[nodiscard]] long windingNumber() const;
};

/// Wilson loop observable on a triangulated spacetime.
///
/// Computes holonomy-like quantities around closed paths.  The dual-graph
/// modes (top-simplices as nodes, shared facets as edges) let users choose
/// between purely combinatorial, curvature-based, and causal-structure
/// analyses.  The ``U1_CONNECTION`` mode instead accumulates the U(1)
/// connection (``Edge::phase``) around a cycle on the primal 1-skeleton,
/// returning the gauge-invariant holonomy (mod 2π) — the Wilson-loop view of
/// the Stage-1 ``cobordism::HodgeLaplacian`` cycle flux.
///
/// Usage:
/// @code
///   auto wl = WilsonLoop(spacetime);
///   auto loop = wl.hingeLoop(some_hinge);
///   auto result = wl.evaluate(loop, WilsonMode::DEFICIT_ANGLE);
/// @endcode
class WilsonLoop {
  public:
    explicit WilsonLoop(std::shared_ptr<Spacetime> spacetime);

    // ==================== Unified interface ====================

    /// Evaluate the Wilson loop in the given mode.
    [[nodiscard]] WilsonResult evaluate(const LoopPath &loop,
                                         WilsonMode mode) const;

    // ==================== Per-mode methods ====================

    [[nodiscard]] WilsonResult evaluateCombinatorial(const LoopPath &loop) const;
    [[nodiscard]] WilsonResult evaluateDeficitAngle(const LoopPath &loop) const;
    [[nodiscard]] WilsonResult evaluateCausal(const LoopPath &loop) const;

    /// Connection holonomy around a closed cycle of vertices on the primal
    /// 1-skeleton.  Accumulates the COMPLEX ``Edge::phase`` oriented along the
    /// stored source→target direction (``+phase`` forward, ``−phase`` on
    /// reversal) and carries the total UNREDUCED in
    /// ``connectionAccumulation`` — the datum. ``value`` is its
    /// ``residualPhase()``, kept for consumers that read the mod-2π angle;
    /// ``loopSize`` is the number of edges.
    /// Returns an empty result if the cycle has fewer than two vertices or any
    /// consecutive pair is not joined by an edge (an open path).
    ///
    /// This is the Wilson-loop counterpart of the Stage-1 cycle flux carried
    /// by the Hermitian-weighted ``cobordism::HodgeLaplacian`` (the same
    /// oriented phase sum); restricted to phases in {0, π} it reproduces the
    /// ℤ₂ flux.
    [[nodiscard]] WilsonResult evaluateU1Connection(
        const std::vector<VertexPtr> &cycle) const;

    // ==================== Loop generators ====================

    /// Loop of top-simplices around a hinge, ordered cyclically.
    [[nodiscard]] LoopPath hingeLoop(SimplexPtr hinge) const;

    /// BFS-discovered loop of approximately \a targetLength simplices.
    [[nodiscard]] LoopPath dualLatticeLoop(SimplexPtr start,
                                            int targetLength) const;

    /// Shortest cycle through \a start in the dual graph.
    [[nodiscard]] LoopPath geodesicLoop(SimplexPtr start) const;

    // ==================== Measurement ====================

    void measure(const LoopPath &loop, WilsonMode mode);
    void measureAllHinges(WilsonMode mode);
    void reset();
    [[nodiscard]] const std::vector<WilsonResult> &getMeasurements() const;
    [[nodiscard]] std::map<int, std::complex<double>> getAverageBySize() const;

  private:
    std::shared_ptr<Spacetime> spacetime_;
    int d_;  // spacetime dimension
    std::vector<WilsonResult> measurements_;

    /// Top-simplices sharing a facet with \a sigma.
    [[nodiscard]] std::vector<SimplexPtr> dualNeighbors(SimplexPtr sigma) const;

    /// Shared facet between two adjacent top-simplices (or nullptr).
    [[nodiscard]] SimplexPtr findSharedFacet(SimplexPtr a, SimplexPtr b) const;

    /// Edge joining two vertices on the primal 1-skeleton (nullptr if none).
    [[nodiscard]] EdgePtr edgeBetween(VertexPtr a, VertexPtr b) const;

    /// Build a LoopPath from an ordered vector of simplices.
    [[nodiscard]] LoopPath buildLoopPath(
        const std::vector<SimplexPtr> &simplices) const;

    /// BFS over the dual graph starting at \a start, yielding each
    /// cycle to ``onCycle(path)``. The path runs start → ... → start.
    /// If \a onCycle returns true, the walk terminates; otherwise it
    /// continues searching.
    ///
    /// ``maxDepth < 0`` disables the depth cap. ``minCurDepth`` filters
    /// out cycles whose far endpoint is too close to the start
    /// (used by the target-length search to skip trivial back-and-forth
    /// cycles of length 2).
    ///
    /// Shared implementation backing both ``geodesicLoop`` (first
    /// cycle, no depth cap) and ``dualLatticeLoop`` (best target-length
    /// cycle within a depth budget).
    template <typename OnCycleFn>
    void bfsFindCycles(SimplexPtr start, int maxDepth, int minCurDepth,
                          OnCycleFn onCycle) const;

};

} // namespace tessera::observables
