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

#ifndef TESSERA_SIMPLEXBOUNDARYSPHERE_H
#define TESSERA_SIMPLEXBOUNDARYSPHERE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Simplex-boundary sphere \f$ S^n = \partial\Delta^{n+1} \f$
///
/// The minimal triangulation of the n-sphere: the boundary of the
/// (n+1)-simplex, i.e. \f$ n+2 \f$ vertices with every \f$ (n+1) \f$-subset a
/// top n-simplex. f-vector \f$ \binom{n+2}{k+1} \f$ for \f$ k = 0..n \f$
/// (e.g. \f$ S^4 \f$: (6, 15, 20, 15, 6)); \f$ \chi = 1 + (-1)^n \f$.
///
/// This is the exact, fixed minimal triangulation used by the cobordism
/// fixtures — distinct from :class:`Sphere`, which grows a CDT-style
/// \f$ S^{d-1}\times S^1 \f$ initial condition of a requested size. Its
/// ``build()`` ignores ``numSimplices`` and is *pre-geometric* (coordinate-free
/// vertices; see ``Topology::buildExplicit``).
class SimplexBoundarySphere : public Topology {
  public:
    /// @param n Dimension of the sphere (n >= 1).
    explicit SimplexBoundarySphere(int n) : n_(n) {}

    [[nodiscard]] int n() const noexcept { return n_; }

    /// Build \f$ S^n = \partial\Delta^{n+1} \f$. ``numSimplices`` is ignored
    /// (the triangulation is fixed).
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int n_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SIMPLEXBOUNDARYSPHERE_H
