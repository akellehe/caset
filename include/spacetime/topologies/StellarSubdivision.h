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

#ifndef TESSERA_STELLARSUBDIVISION_H
#define TESSERA_STELLARSUBDIVISION_H

#include <memory>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Stellar subdivision of a triangulation
///
/// Wraps a base topology and refines it by a single **stellar subdivision**
/// (a \f$ 1 \to (n+1) \f$ Pachner move): one top \f$ n \f$-simplex
/// \f$ \tau = \{v_0, \ldots, v_n\} \f$ is starred at a fresh interior vertex
/// \f$ c \f$ — \f$ \tau \f$ is removed and replaced by the \f$ n+1 \f$ simplices
/// \f$ \{c\} \cup (\tau \setminus \{v_i\}) \f$. To keep the result deterministic
/// the starred simplex is the lexicographically smallest top-simplex tuple.
///
/// Stellar moves are PL homeomorphisms, so the subdivided complex is the *same
/// manifold* as the base with *identical homology*, yet it is a genuinely
/// distinct labelled complex (one extra vertex, \f$ n \f$ extra top simplices),
/// hence **not** isomorphic to the base. This makes it the canonical way to
/// produce a second, inequivalent triangulation of a fixture — e.g. a retriangulated
/// \f$ T^3 = \text{StellarSubdivision}(S^1 \times S^1 \times S^1) \f$ — for
/// triangulation-invariance checks. Mirrors :class:`SimplicialProduct` as a
/// composable wrapper over other topologies.
///
/// Exact and pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class StellarSubdivision : public Topology {
  public:
    explicit StellarSubdivision(std::shared_ptr<Topology> base)
        : base_(std::move(base)) {}

    /// A stellar subdivision is a PL homeomorphism, so it preserves dimension.
    [[nodiscard]] int dimension() const override { return base_->dimension(); }

    /// Build the base topology and apply one stellar subdivision.
    /// ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    std::shared_ptr<Topology> base_;
};

} // namespace tessera::spacetime

#endif // TESSERA_STELLARSUBDIVISION_H
