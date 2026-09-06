// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_POLYGONCIRCLE_H
#define TESSERA_POLYGONCIRCLE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {

class Spacetime;

/// # Polygon circle \f$ S^1 \f$ with \f$ n \f$ vertices
///
/// The circle triangulated as an \f$ n \f$-gon: vertices \f$ 0, \ldots, n-1 \f$
/// and the \f$ n \f$ edges \f$ \{i, i+1 \bmod n\} \f$. Exact, fixed and
/// pre-geometric (coordinate-free vertices; see ``Topology::buildExplicit``);
/// ``build()`` ignores ``numSimplices``. \f$ n = 3 \f$ is the triangle circle
/// ``SimplexBoundarySphere(1)`` = \f$ \partial\Delta^2 \f$, which stays the
/// minimal case; this class exists for \f$ n \geq 3 \f$ in general.
///
/// WHY a resolution knob on the circle: ``SimplicialProduct`` triangulates
/// \f$ K \times L \f$ by the Eilenberg–Zilber staircase, so
/// ``SimplicialProduct(PolygonCircle(nx), PolygonCircle(ny))`` is the
/// \f$ n_x \times n_y \f$ grid torus with every square cut along one diagonal —
/// the standard flat-torus mesh. The simplicial-qubit representation
/// (``observables::SimplicialQubit``) needs that mesh at *several* resolutions:
/// its period ratio must be refinement-invariant on a flat torus and its
/// complex-structure residual must vanish under refinement elsewhere, and a
/// family of tori indexed by \f$ (n_x, n_y) \f$ is what those statements are
/// tested on. With only \f$ \partial\Delta^2 \f$ available the product torus
/// is fixed at \f$ 3 \times 3 \f$.
class PolygonCircle : public Topology {
  public:
    /// @param n Number of vertices (and edges) of the polygon; \f$ n \geq 3 \f$
    ///          so that every edge is a genuine 1-simplex (no repeated vertex
    ///          pair) and the product with another circle is a simplicial
    ///          complex.
    explicit PolygonCircle(int n) : n_(n) {}
    [[nodiscard]] int n() const noexcept { return n_; }
    /// \f$ S^1 \f$ is a 1-manifold; its top cells are edges.
    [[nodiscard]] int dimension() const override { return 1; }
    /// Build the \f$ n \f$-gon. ``numSimplices`` is ignored (the triangulation
    /// is fixed by \f$ n \f$).
    /// @throws std::invalid_argument when \f$ n < 3 \f$.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int n_;
};

} // namespace tessera::spacetime

#endif // TESSERA_POLYGONCIRCLE_H
