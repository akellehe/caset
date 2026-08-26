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
    U1_CONNECTION    ///< Legacy enum name: direct C* link product on a 1-cycle
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
    /// In ``U1_CONNECTION`` mode this is the direct multiplicative C* holonomy.
    std::complex<double> value{0.0, 0.0};

    /// Retired additive-phase payload. It remains for record compatibility but
    /// is NaN for direct-link reads: no logarithm lift is selected.
    std::complex<double> connectionAccumulation{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};

    /// Branch-free ordered product of the oriented edge links. This is the
    /// primary connection datum. NaN outside connection mode.
    std::complex<double> connectionHolonomy{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};

    int    loopSize = 0;           ///< number of simplices in the loop
    int    enclosedHinges = 0;     ///< hinges enclosed (combinatorial)
    bool   contractible = true;    ///< is the loop contractible? (combinatorial)
    int    causalWindingNumber = 0;///< net orientation changes (causal)

    /// Fold an angle into the principal holonomy interval (−π, π].
    [[nodiscard]] static double principalAngle(double theta);

    /// The direct multiplicative holonomy \f$H(\gamma)\in\mathbb C^*\f$.
    [[nodiscard]] std::complex<double> holonomy() const;

    /// Presentation certificate \f$|H(\gamma)|\f$; never a normalized field.
    [[nodiscard]] double holonomyModulus() const;

    /// Presentation-only principal argument of the direct holonomy.
    [[nodiscard]] double residualPhase() const;

    /// Always zero for a single product. Winding requires an explicitly
    /// matched relative path plus a continuously tracked lift.
    [[nodiscard]] long windingNumber() const;
};

/// Wilson loop observable on a triangulated spacetime.
///
/// Computes holonomy-like quantities around closed paths.  The dual-graph
/// modes (top-simplices as nodes, shared facets as edges) let users choose
/// between purely combinatorial, curvature-based, and causal-structure
/// analyses. The legacy-named ``U1_CONNECTION`` mode multiplies the full
/// \f$\mathbb C^*\f$ links around a primal 1-cycle without compactifying them.
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
    /// 1-skeleton. Multiplies direct oriented links in C*. No logarithm,
    /// argument, or compact projection is used. ``value`` and
    /// ``connectionHolonomy`` carry the exact product.
    /// Returns an empty result if the cycle has fewer than two vertices or any
    /// consecutive pair is not joined by an edge (an open path).
    ///
    /// This is the Wilson-loop counterpart of the same direct links consumed
    /// by the current one-particle operator.
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
