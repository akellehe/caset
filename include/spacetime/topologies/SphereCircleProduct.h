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

#ifndef TESSERA_SPHERECIRCLEPRODUCT_H
#define TESSERA_SPHERECIRCLEPRODUCT_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Sphere–circle product \f$ S^2 \times S^1 \f$
///
/// The closed, oriented 3-manifold \f$ S^2 \times S^1 \f$, triangulated by the
/// staircase (Eilenberg–Zilber) product of the minimal sphere triangulations
/// \f$ S^2 = \partial\Delta^3 \f$ and \f$ S^1 = \partial\Delta^2 \f$ — i.e.
/// exactly ``SimplicialProduct(SimplexBoundarySphere(2),
/// SimplexBoundarySphere(1))``, exposed as a named topology for symmetry with
/// :class:`Toroid`. The result has 12 vertices and 36 tetrahedra, Euler
/// characteristic \f$ \chi = 0 \f$, and Betti numbers \f$ b = (1, 1, 1, 1) \f$.
///
/// It is the negative control for the triple cup product: \f$ H^1(S^2 \times
/// S^1) \f$ is one-dimensional, so the cup-cube \f$ \alpha \cup \alpha \cup
/// \alpha \f$ vanishes on it (unlike \f$ T^3 \f$, where the three independent
/// 1-classes pair to the fundamental class). The Dijkgraaf–Witten state sum is
/// therefore insensitive to the cup-product cocycle here, \f$ Z_\text{sign} =
/// Z_\text{triv} \f$.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class SphereCircleProduct : public Topology {
  public:
    SphereCircleProduct() = default;

    /// \f$ S^2 \times S^1 \f$ is a closed 3-manifold; its top cells are tetrahedra.
    [[nodiscard]] int dimension() const override { return 3; }

    /// Build \f$ S^2 \times S^1 \f$ via the staircase product. ``numSimplices``
    /// is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_SPHERECIRCLEPRODUCT_H
