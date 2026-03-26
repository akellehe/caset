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

#ifndef CASET_TOROID_H
#define CASET_TOROID_H

#include "Topology.h"

namespace caset {
class Spacetime;

/// # Toroidal Topology \f$ T^{d-1} \f$
///
/// Spatial slices have the topology of a \f$(d\!-\!1)\f$-torus, giving a spacetime
/// manifold \f$ \mathcal{M} \cong T^{d-1} \times S^1 \f$ with periodic boundary
/// conditions in both space and time. This is the default topology for CDT
/// simulations and the most commonly used in the literature.
///
/// Periodic time means the last time slice \f$ t = T \f$ is identified with the
/// first \f$ t = 0 \f$, so the triangulation has no temporal boundaries.
///
/// The build creates multiple time layers, each grown from a seed \f$ d \f$-simplex
/// via iterated coning in random (forward/backward) time directions.
///
class Toroid : public Topology {
  public:
    /// Build a toroidal triangulation with multiple time layers.
    ///
    /// Cones exterior facets in random \f$ \pm t \f$ directions within each layer,
    /// then advances to the next time slice. The result is a multi-layer simplicial
    /// complex where each layer spans one unit of coordinate time.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // caset

#endif //CASET_TOROID_H
