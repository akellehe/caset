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

#ifndef TESSERA_SOLIDSIMPLEX_H
#define TESSERA_SOLIDSIMPLEX_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Solid simplex \f$ \Delta^n \f$ (closed n-ball)
///
/// A single top n-simplex on \f$ n+1 \f$ vertices with all its faces — a
/// triangulated closed n-ball whose boundary is \f$ S^{n-1} = \partial\Delta^n \f$.
/// f-vector \f$ \binom{n+1}{k+1} \f$ for \f$ k = 0..n \f$; \f$ \chi = 1 \f$
/// (contractible). \f$ \Delta^4 \f$ is the smallest cobordism filling
/// \f$ S^3 \to \emptyset \f$.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class SolidSimplex : public Topology {
  public:
    /// @param n Dimension of the simplex (n >= 1).
    explicit SolidSimplex(int n) : n_(n) {}

    [[nodiscard]] int n() const noexcept { return n_; }

    /// \f$ \Delta^n \f$ is an n-ball (n-manifold with boundary); its single top
    /// cell is an n-simplex.
    [[nodiscard]] int dimension() const override { return n_; }

    /// Build the solid n-simplex. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int n_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SOLIDSIMPLEX_H
