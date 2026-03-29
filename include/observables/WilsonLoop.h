// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include "mesh/ForwardDeclarations.h"
#include <map>
#include <memory>
#include <vector>

namespace caset {

class Spacetime;

/// Evaluation mode for Wilson loops, at increasing levels of geometric commitment.
enum class WilsonMode : uint8_t {
    COMBINATORIAL,   ///< Dual-graph topology only (loop length, enclosed hinges)
    DEFICIT_ANGLE,   ///< Uses deficit angles: W = ((d-2)+2cos(ε))/d
    CAUSAL           ///< CDT causal orientation changes around the loop
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
    double value = 0.0;            ///< primary scalar
    int    loopSize = 0;           ///< number of simplices in the loop
    int    enclosedHinges = 0;     ///< hinges enclosed (combinatorial)
    bool   contractible = true;    ///< is the loop contractible? (combinatorial)
    int    causalWindingNumber = 0;///< net orientation changes (causal)
};

/// Wilson loop observable on a triangulated spacetime.
///
/// Computes holonomy-like quantities around closed paths in the dual graph
/// (top-simplices as nodes, shared facets as edges).  Three evaluation modes
/// let users choose between purely combinatorial, curvature-based, and
/// causal-structure analyses.
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
    [[nodiscard]] std::vector<WilsonResult> getMeasurements() const;
    [[nodiscard]] std::map<int, double> getAverageBySize() const;

  private:
    std::shared_ptr<Spacetime> spacetime_;
    int d_;  // spacetime dimension
    std::vector<WilsonResult> measurements_;

    /// Top-simplices sharing a facet with \a sigma.
    [[nodiscard]] std::vector<SimplexPtr> dualNeighbors(SimplexPtr sigma) const;

    /// Shared facet between two adjacent top-simplices (or nullptr).
    [[nodiscard]] SimplexPtr findSharedFacet(SimplexPtr a, SimplexPtr b) const;

    /// Build a LoopPath from an ordered vector of simplices.
    [[nodiscard]] LoopPath buildLoopPath(
        const std::vector<SimplexPtr> &simplices) const;

};

} // namespace caset
