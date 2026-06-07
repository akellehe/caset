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

#ifndef TESSERA_REALPROJECTIVESPACE_H
#define TESSERA_REALPROJECTIVESPACE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Real projective 3-space \f$ \mathbb{RP}^3 = L(2,1) \f$ (minimal 11-vertex)
///
/// Walkup's vertex- and facet-minimal triangulation of \f$ \mathbb{RP}^3 \f$ —
/// the unique simultaneously vertex- and facet-minimal triangulation, with
/// f-vector \f$ (11, 51, 80, 40) \f$ (11 vertices, 40 tetrahedra) and Euler
/// characteristic \f$ \chi = 0 \f$. A combinatorial triangulation of
/// \f$ \mathbb{RP}^3 \f$ needs at least 11 vertices, so this is as small as it
/// gets. The explicit facet list is the one tabulated by Lutz and shipped by
/// SageMath's ``RealProjectiveSpace(3)``, with vertex labels shifted from the
/// literature's \f$ 1\ldots 11 \f$ down to tessera's \f$ 0\ldots 10 \f$.
///
/// A closed, orientable 3-manifold with Betti numbers \f$ (1,0,0,1) \f$ over
/// \f$ \mathbb{Q} \f$ and \f$ (1,1,1,1) \f$ over \f$ \mathbb{Z}/2 \f$; the gap
/// is the 2-torsion \f$ H_1(\mathbb{RP}^3;\mathbb{Z}) = \mathbb{Z}/2 \f$, which
/// distinguishes it from \f$ S^2 \times S^1 \f$ (same \f$ \mathbb{Z}/2 \f$ Betti,
/// no torsion). Being closed and orientable (\f$ b_3 = 1 \f$) it carries a
/// fundamental class.
///
/// ## Dijkgraaf–Witten positive control (P3)
///
/// \f$ \mathbb{RP}^3 \f$ is the canonical small manifold on which the
/// \f$ \mathbb{Z}_2 \f$ sign cocycle \f$ \omega(a,b,c) = (-1)^{abc} \f$ *does*
/// distinguish: its mod-2 cohomology ring is \f$ \mathbb{Z}_2[t]/t^4 \f$ with
/// \f$ t^3 \neq 0 \f$, so the cup-cube pairing \f$ \langle g^3, [W] \rangle \f$
/// is nonzero. Hence \f$ Z_\text{Sign}(\mathbb{RP}^3) = 0 \neq 1 =
/// Z_\text{Trivial}(\mathbb{RP}^3) \f$ — unlike the negative controls
/// \f$ T^3 \f$ and \f$ S^2 \times S^1 \f$, where every 1-class has \f$ g^3 = 0
/// \f$ and the two state sums agree. With 11 vertices the flat space has
/// \f$ \dim Z^1 = 11 \le 24 \f$, small enough for the brute-force state sum.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class RealProjectiveSpace : public Topology {
  public:
    RealProjectiveSpace() = default;

    /// Build the 11-vertex \f$ \mathbb{RP}^3 \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_REALPROJECTIVESPACE_H
