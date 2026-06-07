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

#ifndef TESSERA_REALPROJECTIVEPLANE_H
#define TESSERA_REALPROJECTIVEPLANE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Real projective plane \f$ \mathbb{RP}^2 \f$ (minimal 6-vertex)
///
/// The unique minimal triangulation of \f$ \mathbb{RP}^2 \f$ — the
/// hemi-icosahedron, i.e. the 10 triangles of the icosahedron modulo the
/// antipodal map, on the complete graph \f$ K_6 \f$. f-vector (6, 15, 10),
/// \f$ \chi = 1 \f$, **non-orientable** (\f$ w_1^2[\mathbb{RP}^2] = 1 \f$).
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class RealProjectivePlane : public Topology {
  public:
    RealProjectivePlane() = default;

    /// \f$ \mathbb{RP}^2 \f$ is a closed surface; its top cells are triangles.
    [[nodiscard]] int dimension() const override { return 2; }

    /// Build the 6-vertex \f$ \mathbb{RP}^2 \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_REALPROJECTIVEPLANE_H
