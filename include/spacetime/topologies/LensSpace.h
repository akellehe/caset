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

#ifndef TESSERA_LENSSPACE_H
#define TESSERA_LENSSPACE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Lens space \f$ L(p,q) \f$ (vertex-minimal simplicial triangulation)
///
/// A closed orientable 3-manifold, the quotient of \f$ S^3 \f$ by the free
/// \f$ \mathbb{Z}_p \f$ action \f$ (z_1, z_2) \mapsto (e^{2\pi i/p} z_1,
/// e^{2\pi i q/p} z_2) \f$. Its only nontrivial reduced homology is
/// \f$ H_1(L(p,q);\mathbb{Z}) = \mathbb{Z}_p \f$, so the rational Betti numbers
/// are \f$ (1,0,0,1) \f$ and \f$ \mathrm{torsion}(H_1) = [p] \f$. Being closed
/// and orientable (\f$ b_3 = 1 \f$) it carries a fundamental class, and like
/// every closed odd-dimensional manifold it has \f$ \chi = 0 \f$.
///
/// The hardcoded facet lists are Lutz's vertex-minimal combinatorial
/// triangulations (the spherical-space-form section of "The Manifold Page"),
/// relabeled from the literature's \f$ 1\ldots n \f$ down to tessera's
/// \f$ 0\ldots n-1 \f$. The homology is the proof the facet lists are right:
///
/// | \f$ (p,q) \f$ | f-vector            | \f$ \dim Z^1 \f$ |
/// |---------------|---------------------|------------------|
/// | \f$ (3,1) \f$ | \f$ (12,66,108,54)\f$ | 11             |
/// | \f$ (4,1) \f$ | \f$ (14,84,140,70)\f$ | 14             |
/// | \f$ (5,2) \f$ | \f$ (14,86,144,72)\f$ | 13             |
///
/// The minimal \f$ L(2,1) = \mathbb{RP}^3 \f$ is supplied separately as
/// ``RealProjectiveSpace`` (Walkup's 11-vertex complex). Distinct
/// \f$ L(p,q) \f$ that share a \f$ p \f$ (e.g. \f$ L(5,1) \f$ and \f$ L(5,2)
/// \f$) have the same homology — they are told apart by the linking form /
/// Reidemeister torsion, not the chain complex.
///
/// Every flat space here has \f$ \dim Z^1 \le 24 \f$, small enough for the
/// brute-force Dijkgraaf–Witten state sum.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class LensSpace : public Topology {
  public:
    /// Construct \f$ L(p,q) \f$. Throws ``std::invalid_argument`` unless
    /// \f$ (p,q) \f$ is one of the supplied triangulations: ``(3,1)``,
    /// ``(4,1)``, ``(5,2)``.
    LensSpace(int p, int q);

    /// The order \f$ p \f$ — \f$ H_1(L(p,q)) = \mathbb{Z}_p \f$.
    int p() const { return p_; }

    /// The gluing parameter \f$ q \f$.
    int q() const { return q_; }

    /// Build the vertex-minimal \f$ L(p,q) \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int p_;
    int q_;

    /// The hardcoded triangulation of \f$ L(p,q) \f$: returns the top-simplex
    /// (tetrahedron) vertex-id list and reports the vertex count through
    /// ``numVertices``. Throws ``std::invalid_argument`` for an unsupported
    /// \f$ (p,q) \f$. Single source of truth shared by the constructor's
    /// validation and ``build()``.
    static std::vector<std::vector<std::uint64_t>> triangulation(
        int p, int q, std::size_t &numVertices);
};

} // namespace tessera::spacetime

#endif // TESSERA_LENSSPACE_H
