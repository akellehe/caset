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

#ifndef TESSERA_SIMPLICIALPRODUCT_H
#define TESSERA_SIMPLICIALPRODUCT_H

#include <memory>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Simplicial product \f$ K \times L \f$
///
/// The product of two triangulations, triangulated by the standard *staircase*
/// (Eilenberg–Zilber) construction. Vertices are pairs \f$ (u, v) \in V(K)
/// \times V(L) \f$; a product cell \f$ \sigma^p \times \tau^q \f$ is cut into
/// \f$ \binom{p+q}{p} \f$ simplices of dimension \f$ p+q \f$, one per monotone
/// lattice path \f$ (0,0) \to (p,q) \f$, using each factor's vertex order so the
/// pieces glue consistently across shared faces.
///
/// Used to build product manifolds for the cobordism fixtures — e.g.
/// \f$ S^2 \times S^2 = \partial\Delta^3 \times \partial\Delta^3 \f$ (a closed
/// oriented 4-manifold with \f$ b_2 = 2 \f$ and signature 0), or \f$ T^2 =
/// S^1 \times S^1 \f$. Exact and pre-geometric (coordinate-free); ``build()``
/// ignores ``numSimplices``.
///
/// The factor vertex set is taken to be \f$ \{0, \ldots, |V|-1\} \f$ (as
/// produced by the coordinate-free fixture topologies); product vertex
/// \f$ (u, v) \f$ is assigned id \f$ u \cdot |V(L)| + v \f$.
class SimplicialProduct : public Topology {
  public:
    SimplicialProduct(std::shared_ptr<Topology> left, std::shared_ptr<Topology> right)
        : left_(std::move(left)), right_(std::move(right)) {}

    /// Build \f$ K \times L \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    std::shared_ptr<Topology> left_;
    std::shared_ptr<Topology> right_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SIMPLICIALPRODUCT_H
