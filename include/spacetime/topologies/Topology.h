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

#ifndef TESSERA_TOPOLOGY_H
#define TESSERA_TOPOLOGY_H

#include <cstddef>
#include <cstdint>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class Spacetime;

/// # Spatial Topology
///
/// Abstract base class for the topology of spatial slices in a foliated spacetime.
/// In CDT, the spacetime manifold \f$ \mathcal{M} \f$ has the product structure
///
/// \f[
///   \mathcal{M} \cong \Sigma \times I
/// \f]
///
/// where \f$ \Sigma \f$ is a compact spatial manifold and \f$ I \f$ is either
/// an interval \f$[0, T]\f$ (cylinder) or a circle \f$ S^1 \f$ (periodic time).
/// The topology of \f$ \Sigma \f$ is fixed throughout the simulation and
/// determines the boundary conditions and initial triangulation.
///
/// Subclasses implement `build()` to construct an initial triangulation matching
/// their spatial topology via the coning mechanism: a seed \f$ d \f$-simplex is
/// created at each time layer, and exterior facets are iteratively coned to new
/// vertices to grow the complex.
///
class Topology {
  public:
    virtual ~Topology();

    /// Build an initial triangulation with the given spatial topology.
    ///
    /// Creates \f$ d \f$-simplices across multiple time layers using iterated
    /// coning from seed simplices. The resulting simplicial complex
    /// \f$ \mathcal{K} \f$ has the combinatorial structure of the chosen topology.
    ///
    /// @param spacetime The spacetime in which to build the triangulation
    /// @param numSimplices Target number of top-dimensional simplices to create
    virtual void build(Spacetime *spacetime, int numSimplices) = 0;

  protected:
    /// Build an explicit, *pre-geometric* triangulation from a combinatorial
    /// description: create `numVertices` **coordinate-free** vertices
    /// (ids 0..numVertices-1) and one top simplex per vertex-id tuple in
    /// `topSimplices`.
    ///
    /// No coordinates are assigned — the cobordism capabilities (homology,
    /// characteristic numbers, cobordism existence) are purely combinatorial;
    /// geometry (vertex coordinates / edge lengths) is layered on only when a
    /// geometric capability (reconstruction, Regge) actually needs it. Edges
    /// are still materialized by ``createSimplex`` so the incidence structure
    /// is complete; any squared length they carry is an unused placeholder, not
    /// meaningful geometry.
    ///
    /// Shared by the exact, fixed-triangulation topologies
    /// (``SimplexBoundarySphere``, ``SolidSimplex``, ``RealProjectivePlane``,
    /// …) whose ``build()`` ignores ``numSimplices``.
    static void buildExplicit(
        Spacetime *spacetime, std::size_t numVertices,
        const std::vector<std::vector<std::uint64_t>> &topSimplices);
};

} // namespace tessera::spacetime

#endif //TESSERA_TOPOLOGY_H
