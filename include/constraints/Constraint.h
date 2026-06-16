// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CONSTRAINT_H
#define TESSERA_CONSTRAINT_H

#include <memory>

namespace tessera::spacetime { class Spacetime; }

namespace tessera {
using namespace ::tessera::spacetime;

/// Types of constraints that can be applied to a spacetime.
///
///   - **PachnerMove**: Constraints specific to a proposed Pachner move (e.g.,
///     checking that a bistellar flip preserves the manifold condition).
///   - **All**: Global constraints that must hold at all times (e.g.,
///     the simplicial complex must remain a manifold, the causal structure
///     must be preserved).
enum class ConstraintType : uint8_t {
  PachnerMove = 0,
  All = 1
};

/// # Constraint Base Class
///
/// Encodes conditions that a spacetime triangulation must satisfy. In CDT, the
/// central constraint is **causality**: every \f$ d \f$-simplex must have its
/// vertices distributed across exactly two adjacent time slices, preserving the
/// global time foliation. This ensures that each spatial slice is a closed
/// \f$(d\!-\!1)\f$-manifold and that the causal structure is well-defined.
///
/// Additional constraints may include:
///   - **Manifoldness**: every \f$(d\!-\!1)\f$-face is shared by at most 2
///     \f$ d \f$-simplices (the link of every simplex is a sphere).
///   - **Topology preservation**: Pachner moves must not change the topology of
///     the spatial slices or the overall manifold.
///   - **Volume bounds**: the total four-volume \f$ N_4 \f$ must remain within
///     a specified range during the simulation.
///
class Constraint {
  public:
    virtual ~Constraint() = default;

    /// Check whether a constraint is satisfied for the current spacetime state.
    ///
    /// @param spacetime The spacetime to check
    /// @param type_ The type of constraint check to perform
    /// @return true if the constraint is satisfied
    virtual bool applies(const std::shared_ptr<Spacetime> &spacetime, const ConstraintType &type_) = 0;
};

} // tessera

#endif //TESSERA_CONSTRAINT_H
