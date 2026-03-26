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

#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Fingerprint.h"
#include "ForwardDeclarations.h"
#include "EdgeKey.h"

#include <random>
#include <memory>


namespace caset {
/// # Edge Disposition
///
/// There are two things that determine the disposition (spacelike, timelike, light/null-like). The first is the squared
/// edge length. If the squared length is negative in a (-, +, +, +) signature it's timelike. A negative edge length in
/// a (+, -, -, -) signature is spacelike. A 0-length in either is lightlike/null.
///
/// The second thing that determines the edge disposition is whether the vertices exist both in space (lightlike), both
/// at the same time (timelike), or one in space and one in time (spacelike). See "Quantum Gravity from Causal Dynamical
/// Triangulations: A Review" by R. Loll, 2019. Figure 1. There's no discussion of lightlike edges since CDT does not
/// treat that case. I'm making that up to fill in the gaps. If there's some existing discussion around this in the
/// literature I'm not aware at the time of this writing.
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
class Edge : public std::enable_shared_from_this<Edge> {
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

#ifdef CASET_VERBOSE
    [[nodiscard]] std::string toString() const noexcept;
#else
    [[nodiscard]] std::string toString() const noexcept {
      return "";
    };
#endif

    /// This method changes the target source in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup. This method also updates the fingerprint hastily. If you want to update in
    /// batches remove the fingerprint.refresh() call.
    void replaceSourceVertex(const VertexPtr &newSource);

    /// This method changes the target Vertex in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup.
    /// CRITICAL: TODO: we need to remove edges from their containers before changing their fingerprints!
    /// Same as replaceSourceVertex above, but for targets.
    void replaceTargetVertex(const VertexPtr &newTarget);

    ///
    /// Check whether or not this Edge has a particular Vertex. The comparison is against source/target node IDs, so
    /// don't worry too much about accidentally comparing pointers. This is mostly a convenience method to make your
    /// code more clear and avoid typing.
    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool hasVertex(std::uint64_t vertexId);
    bool hasVertex(const VertexPtr &vertex);

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

  private:
    VertexPtr source = nullptr;
    VertexPtr target = nullptr;

    double squaredLength;
};

}

#endif //CASET_CASET_SRC_EDGE_H_
