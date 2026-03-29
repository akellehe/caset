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

#ifndef CASET_CONSTRAINT_H
#define CASET_CONSTRAINT_H

#include <memory>

namespace caset {

class Spacetime;

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

} // caset

#endif //CASET_CONSTRAINT_H
