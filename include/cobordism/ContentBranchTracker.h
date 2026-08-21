// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CONTENTBRANCHTRACKER_H
#define TESSERA_COBORDISM_CONTENTBRANCHTRACKER_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cobordism/ChainComplex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// One accepted-step snapshot of the lifted top-simplex contents.
///
/// `contents[i]` is a continuously chosen square root of the squared content of
/// `cells[i]`, including its local orientation sign. `orientation` is the same
/// orientation local system in the matching branch gauge: when continuation
/// flips one root, all incident transition functions are gauge-transformed too,
/// so its covariant Laplacian and loop holonomies do not change.
struct ContentBranchSnapshot {
    std::vector<std::vector<std::uint64_t>> cells{};
    std::vector<std::complex<double>> contents{};
    OrientationLocalSystem orientation{};
    std::size_t continuedCells{0};
    std::size_t seededCells{0};
    /// Cells whose continued lift is the negative of the current canonical
    /// orientation-times-principal-root seed (a sheet occupancy, not an event
    /// counter; it may remain nonzero over many accepted steps).
    std::size_t principalBranchFlips{0};
    /// Continued cells essentially equidistant from both sheets. A nonzero
    /// value means the accepted step was too large, or passed through zero, for
    /// nearest-sheet continuation to identify the analytic lift reliably.
    std::size_t ambiguousCells{0};
};

/// Stateful path lifting for the two-sheeted map \f$V\mapsto V^2\f$.
///
/// `update()` computes each current principal `Simplex::volume()` and, for a
/// cell present in the previous accepted snapshot, chooses \f$V\f$ or \f$-V\f$
/// by proximity to that previous lift. A new cell is seeded from the canonical
/// spanning-forest trivialization of its orientation local system. Thus a path
/// of \f$V^2\f$ winding once around zero returns on the opposite sheet instead
/// of jumping at the principal square-root cut.
///
/// This class is deliberately separate from `HodgeLaplacian`: an optimizer
/// probes rejected line-search points and may evaluate them out of order, so a
/// mutable branch history inside the objective would make it path-dependent and
/// invalidate its gradient. Call `update()` only after a geometry step has been
/// accepted, then pass/read the immutable snapshot for diagnostics or an
/// explicitly branch-aware experimental operator.
///
/// Continuation is faithful only when accepted steps are fine enough that a
/// root moves by less than a quarter turn (phase \f$<\pi/2\f$), equivalently
/// its square moves by less than half a turn, without passing through zero.
class ContentBranchTracker {
  public:
    /// Forget all path history. The next update seeds every cell from the
    /// orientation local system.
    void reset() noexcept { snapshot_ = {}; }

    [[nodiscard]] bool initialized() const noexcept {
      return !snapshot_.cells.empty();
    }

    /// Lift the current top-cell contents relative to the previous accepted
    /// snapshot, replace the stored snapshot, and return it by value.
    [[nodiscard]] ContentBranchSnapshot update(const Spacetime &spacetime);

    [[nodiscard]] const ContentBranchSnapshot &snapshot() const noexcept {
      return snapshot_;
    }

  private:
    ContentBranchSnapshot snapshot_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CONTENTBRANCHTRACKER_H
