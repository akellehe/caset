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

#ifndef TESSERA_TESSERA_SRC_EDGE_H_
#define TESSERA_TESSERA_SRC_EDGE_H_

#include "mesh/Fingerprint.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/EdgeKey.h"

#include <random>
#include <memory>


// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
/// # Edge Disposition
///
/// There are two things that determine the disposition (spacelike, timelike, light/null-like). The first is the squared
/// edge length. If the squared length is negative in a (-, +, +, +) signature it's timelike. A negative edge length in
/// a (+, -, -, -) signature is spacelike. A 0-length in either is lightlike/null.
///
/// The second thing that determines the edge disposition is whether the vertices are at the same time (spacelike,
/// within a spatial slice) or at different times (timelike, crossing between slices). See "Quantum Gravity from Causal
/// Dynamical Triangulations: A Review" by R. Loll, 2019. Figure 1. CDT does not treat the lightlike case.
enum class EdgeDisposition : uint8_t {
  Spacelike = 0,
  Timelike = 1,
  Lightlike = 2
};


/// # Edge Class
///
/// An edge that links two points (vertices) in spacetime. When we merge two vertices in the process of connecting two
/// adjacent simplices; we cannot modify the edges in place without first removing them from their containers. Otherwise
/// avoiding the necessary re-hashing results in undefined behavior. We should keep as little state as possible on the
/// edge in favor of maintaining that state on the Vertex.
///
/// @param source_If this Edge represents a directed Edge; then this is the Vertex from which the Edge originates. For
///   undirected edges; it's just one of two Vertices that define the Edge.
/// @param target_If this Edge represents a directed Edge; then this is the Vertex at which the Edge terminates. For
///   undirected edges; it's just one of two Vertices that define the Edge.
/// @param squaredLength_ The squared length of the edge according to whatever spacetime metric is being used. We work
///   in squared lengths to allow the use of imaginary Edge lengths (they have negative values).
///
class Edge {
  public:
    Edge(
      const VertexPtr &source,
      const VertexPtr &target,
      double squaredLength_
    );

    Edge(
      const VertexPtr &source,
      const VertexPtr &target
    );

    /// Every edge has a beginning and an end. Many have two! And by that I mean they're undirected, so the beginning is
    /// the end and the end, the beginning. Edges are bidirectional, so it doesn't really matter if you consider them
    /// directed or undirected. If you want to use a directed edge; in your code you should just specify that you only
    /// traverse `Vertex::outEdges` and avoid `Vertex::inEdges` when you traverse around.
    [[nodiscard]] const VertexPtr &getSource() const noexcept;

    /// `getTarget` is `getSource`'s better half. All good things come to an end, with a wonderful journey left to
    /// memory. But seriously, though, `getTarget` gives the vertex on one end, and `getSource` gives the other.
    [[nodiscard]] const VertexPtr &getTarget() const noexcept;

    /// We work in squared edge lengths because imaginary numbers don't play so nicely with floating point arithmetic.
    /// To be less cryptic: timelike edges have imaginary length. Their squared edge length is negative. Something I've
    /// always thought was kind of neat is a right triangle with the opposite and adjacent edges of length \f$ i \f$
    /// and \f$ 1 \f$ respectively. So the hypotenuse is zero. So timelike edges have imaginary length, spacelike edges
    /// have a positive length, and lightlike edges have zero length.
    ///
    /// @return The square of the length of the edge.
    [[nodiscard]] double getSquaredLength() const noexcept;

#ifdef TESSERA_VERBOSE
    [[nodiscard]] std::string toString() const noexcept;
#else
    [[nodiscard]] std::string toString() const noexcept {
      return "";
    };
#endif

    /// Replace the source vertex in-place and update the fingerprint.
    ///
    /// WARNING: The caller MUST extract this edge from EdgeList BEFORE calling,
    /// then reinsert after. Modifying the fingerprint while the edge is in a
    /// hash-keyed container causes undefined behavior (stale bucket placement).
    /// See Spacetime::swapVertexLabels for the correct extract/update/reinsert pattern.
    void replaceSourceVertex(const VertexPtr &newSource);

    /// Replace the target vertex in-place and update the fingerprint.
    ///
    /// WARNING: Same container-safety requirement as replaceSourceVertex.
    void replaceTargetVertex(const VertexPtr &newTarget);

    ///
    /// Check whether or not this Edge has a particular Vertex. The comparison is against source/target node IDs, so
    /// don't worry too much about accidentally comparing pointers. This is mostly a convenience method to make your
    /// code more clear and avoid typing.
    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool hasVertex(std::uint64_t vertexId) const;
    bool hasVertex(const VertexPtr &vertex) const;

    bool operator==(const Edge &other) const;

    [[nodiscard]] std::uint64_t toHash() const;

    Fingerprint fingerprint{};

    /// If you want to compare two edges by value; you can compare their keys. Assume two Edges with the same EdgeKey
    /// are, for all intents and purposes, equal. This will change if we begin storing state on the Edge, but at the
    /// moment let's focus on storing as much state on the Vertex as possible. Edges have potentially MUCH higher
    /// cardinality than Vertices, so as much state as we can fit on the Vertex, we should fit on the Vertex. This
    /// should be at the expense of slight inconvenience.
    ///
    /// @returns A tuple of {sourceId, targetId}.
    EdgeKey getKey() const noexcept;

    /// Set the squared edge length.  Used by the Regge solver to optimize
    /// the geometry without rebuilding the mesh.
    void setSquaredLength(double sq) noexcept { squaredLength = sq; }

    /// Index into EdgeList::liveVec_ (maintained by EdgeList).
    std::uint32_t liveIdx_{UINT32_MAX};

    /// Simplices currently containing this edge (the edge's "cofaces" in
    /// the codim-1 sense). Mirror of ``Vertex::simplices`` but at edge
    /// granularity, used by ``Vertex::removeOutEdge`` /
    /// ``Vertex::removeInEdge`` to drop the edge from just the simplices
    /// that actually contain it — instead of iterating every simplex
    /// touching the endpoint and filtering by ``hasVertex``. Surfaced by
    /// the v0.2 finite-size profile: hasVertex was ≈22% of `thermalize`
    /// wall time even after the per-call cache.
    ///
    /// Maintained in lockstep with ``Simplex::edges``:
    ///   * Spacetime::registerSimplex registers the simplex on each of
    ///     its edges
    ///   * Spacetime::unregisterSimplex removes it
    ///   * Simplex::addEdge / Simplex::removeEdge mirror the same
    ///     callbacks at runtime
    ///
    /// Callers that intend to mutate the index from inside an iteration
    /// loop (e.g. ``simplex->removeEdge(this)`` invalidates ``simplices_``)
    /// MUST use ``simplicesCopy()`` to snapshot first.
    void registerSimplex(SimplexPtr s);
    void unregisterSimplex(SimplexPtr s) noexcept;
    [[nodiscard]] Simplices const& simplices() const noexcept { return simplices_; }
    [[nodiscard]] Simplices simplicesCopy() const { return simplices_; }

  private:
    VertexPtr source = nullptr;
    VertexPtr target = nullptr;

    double squaredLength;

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
