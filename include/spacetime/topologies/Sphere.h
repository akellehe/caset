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

#ifndef TESSERA_SPHERE_H
#define TESSERA_SPHERE_H

#include "Topology.h"

namespace tessera {

/// # Spherical Topology \f$ S^{d-1} \f$
///
/// Spatial slices have the topology of a \f$(d\!-\!1)\f$-sphere, giving a
/// spacetime manifold \f$ \mathcal{M} \cong S^{d-1} \times S^1 \f$. This is
/// the topology used in most 4D CDT simulations, where spatial slices are
/// three-spheres \f$ S^3 \f$.
///
/// The Euclidean de Sitter solution (the round four-sphere \f$ S^4 \f$) naturally
/// decomposes into \f$ S^3 \f$ spatial slices, making this topology the natural
/// choice for studying de Sitter quantum gravity. The volume profile of each
/// slice follows
///
/// \f[
///   V_3(t) \propto \cos^3\!\left(\frac{\pi\, t}{T}\right)
/// \f]
///
/// for the continuum \f$ S^4 \f$ geometry.
///
/// The build alternates coning direction between layers to close the manifold.
///
class Sphere : public Topology {
  public:
    /// Build a spherical triangulation by coning in alternating \f$ \pm t \f$ directions.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // tessera

#endif //TESSERA_SPHERE_H
