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
#include "EdgeKey.h"
#include "ForwardDeclarations.h"

#include <vector>
#include <random>
#include <memory>

inline double random_uniform(double min = -1.0, double max = 1.0) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

namespace caset {
class Simplex;

/// # Edge Class
///
/// An edge that links two points (vertices) in spacetime.
///
/// ## Edge Disposition
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
    Fingerprint fingerprint;

    Edge(VertexPtr source_, VertexPtr target_, double squaredLength_);
    Edge(VertexPtr source_, VertexPtr target_);

    [[nodiscard]] VertexPtr getSource() const;
    [[nodiscard]] VertexPtr getTarget() const;
    [[nodiscard]] double getSquaredLength() const noexcept;
    [[nodiscard]] std::string toString() const noexcept;

    void copyInPlaceTo(Edge *other);

    /// This method changes the target source in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup.
    void replaceSourceVertex(VertexPtr source_);

    /// This method changes the target Vertex in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup.
    void replaceTargetVertex(VertexPtr target_);

    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool hasVertex(VertexPtr vertexId) const;

    ///
    /// @param from the ID of a vertex to or from which this Edge should no longer point.
    /// @param to the ID of a source or target vertex to which this Edge should now point.
    void redirect(VertexPtr from, VertexPtr to);

    bool operator==(const Edge &other) const;

    [[nodiscard]] std::uint64_t toHash() const;

    [[nodiscard]] EdgeKey getKey() const noexcept;

    [[nodiscard]] std::vector<Simplex *> getSimplices() const noexcept;

    void addSimplex(Simplex *simplex) noexcept;

    void removeSimplex(Simplex *simplex) noexcept;

  private:
    VertexPtr source;
    VertexPtr target;
    std::vector<Simplex *> simplices;

    /// We use fingerprints for fast hashing by the equivalence class of sets of vertices. This method updates the
    /// fingerprint for this Edge after replacing a source or target vertex in-place.
    void refreshFingerprint() noexcept;

    double squaredLength;
};

using EdgeHash = FingerprintHash<Edge>;
using EdgeEq = FingerprintEq<Edge>;
using EdgePtr = std::shared_ptr<Edge>;
using EdgeRawPtr = Edge *;
using Edges = std::vector<EdgeRawPtr>;
using EdgeSet = std::unordered_set<EdgeRawPtr>;
}

#endif //CASET_CASET_SRC_EDGE_H_
