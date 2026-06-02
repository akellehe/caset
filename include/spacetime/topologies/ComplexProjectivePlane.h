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

#ifndef TESSERA_COMPLEXPROJECTIVEPLANE_H
#define TESSERA_COMPLEXPROJECTIVEPLANE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Complex projective plane \f$ \mathbb{CP}^2 \f$ (minimal 9-vertex)
///
/// Kühnel's 9-vertex triangulation of \f$ \mathbb{CP}^2 \f$ — the unique
/// minimal triangulation of any manifold that is neither a sphere nor a
/// boundary of a simplex. A closed, orientable, smooth 4-manifold with
/// f-vector \f$ (9, 36, 84, 90, 36) \f$, Euler characteristic
/// \f$ \chi = 3 \f$, Betti numbers \f$ (1, 0, 1, 0, 1) \f$, and a definite
/// intersection form of rank one — so the signature has absolute value one,
/// \f$ |\sigma| = 1 \f$. Its symmetry group has order 54.
///
/// ## Construction
///
/// The 36 four-simplices are the orbits of twelve base simplices under the
/// order-three permutation \f$ S = (1\,4\,7)(2\,5\,8)(3\,6\,9) \f$ of the nine
/// vertices (each orbit has three members, \f$ 12 \times 3 = 36 \f$). The base
/// list is the one tabulated by Kühnel and Banchoff and reproduced by Schwartz,
/// "Trisecting the 9-vertex complex projective plane". Vertices are relabeled
/// from the literature's \f$ 1\ldots 9 \f$ to tessera's \f$ 0\ldots 8 \f$, under
/// which \f$ S \f$ becomes \f$ (0\,3\,6)(1\,4\,7)(2\,5\,8) \f$.
///
/// ## Orientation note
///
/// \f$ \mathbb{CP}^2 \f$ and its orientation reversal \f$ \overline{\mathbb{CP}}^2 \f$
/// are the *same* simplicial complex; they differ only by a choice of
/// fundamental class (which generator of \f$ \ker \partial_4 \f$). The
/// signature's magnitude \f$ |\sigma| = 1 \f$ is the orientation-independent
/// invariant; its sign is a convention fixed by how the fundamental class is
/// selected.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class ComplexProjectivePlane : public Topology {
  public:
    ComplexProjectivePlane() = default;

    /// Build the 9-vertex \f$ \mathbb{CP}^2 \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_COMPLEXPROJECTIVEPLANE_H
