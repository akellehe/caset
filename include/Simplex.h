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

#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H

#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <coroutine>
#include <deque>

#include "Logger.h"
#include "Edge.h"
#include "EdgeList.h"
#include "VertexList.h"
#include "Fingerprint.h"
#include "Vertex.h"

namespace caset {
///
///
/// @param timeOrientation
enum class TimeOrientation : uint8_t {
  FUTURE = 0,
  PRESENT = 1,
  UNKNOWN = 2
};

class SimplexOrientation;
using SimplexOrientationPtr = std::shared_ptr<SimplexOrientation>;
using SimplexOrientations = std::vector<SimplexOrientationPtr>;

class SimplexOrientation {
  public:
    ///
    /// The orientation of a simplex is determined by how many vertices lie on the initial and final time slice for the
    /// simplex. The orientation is largely only relevant for Lorentzian/CDT complexes where causality is preserved. Those
    /// complexes restrict to allowed orientations that ensure progression forward in time and "fit together" (so they share
    /// faces without gaps in the complex).
    ///
    /// The convention was established in Ambjorn-Loll's "Causal Dynamical Triangulations" paper from 1998-2001. Every
    /// d-simplex must have its vertices split across two adjacent time slices, t and t+1. That means every simplex has
    /// a split
    ///
    /// \f$ (n, d + 1 - n) \f$
    ///
    /// @param ti_ The number of vertices on the initial time slice.
    /// @param tf_ The number of vertices on the final time slice.
    ///
    SimplexOrientation(uint8_t ti_, uint8_t tf_) : ti(ti_), tf(tf_) {
      k = ti_ + tf_ - 1;
    }
    SimplexOrientation() noexcept = default;

    [[nodiscard]] std::pair<uint8_t, uint8_t> numeric() const {
      return {ti, tf};
    }

    bool isDegenerate() const {
      return ti == 0 || tf == 0;
    }

    [[nodiscard]] uint16_t fingerprint() const {
      return (static_cast<uint16_t>(ti) << 8) | static_cast<uint16_t>(tf);
    }

    [[nodiscard]] SimplexOrientationPtr flip() const {
      return std::make_shared<SimplexOrientation>(tf, ti);
    }

    [[nodiscard]]
    SimplexOrientationPtr decTi() const {
      auto newTi = static_cast<uint8_t>(ti - 1);
      // constructor recomputes k automatically
      return std::make_shared<SimplexOrientation>(newTi, tf);
    }

    [[nodiscard]]
    SimplexOrientationPtr decTf() const {
      auto newTf = static_cast<uint8_t>(tf - 1);
      return std::make_shared<SimplexOrientation>(ti, newTf);
    }

    [[nodiscard]] std::string toString() const noexcept {
      return "<SimplexOrientation: (" + std::to_string(ti) + ", " + std::to_string(tf) + ")>";
    }

    bool operator==(const SimplexOrientation &other) const noexcept {
      return ti == other.ti && tf == other.tf;
    }

    [[nodiscard]] TimeOrientation getOrientation() const {
      if (ti == 0 || tf == 0) return TimeOrientation::UNKNOWN;
      if (ti == tf) return TimeOrientation::UNKNOWN;
      if (ti > tf) return TimeOrientation::PRESENT;
      return TimeOrientation::FUTURE;
    }

    [[nodiscard]] std::vector<SimplexOrientationPtr> getFacialOrientations() const {
      if (ti + tf == 0) return {};
      if (ti == 0) return {decTf()};
      if (tf == 0) return {decTi()};
      std::vector<SimplexOrientationPtr> orientations;
      orientations.reserve(2);
      orientations.push_back(decTi());
      orientations.push_back(decTf());
      return orientations;
    }

    /// A k-simplex has \f$ k+1 \f$ vertices.
    [[nodiscard]] uint8_t getK() const {
      return k;
    }

    static SimplexOrientationPtr orientationOf(const Vertices &vertices) {
      uint8_t tiVertices = 0;
      uint8_t tfVertices = 0;
      double ti = std::numeric_limits<double>::max();
      double tf = -1;
      double initial = -1;
      int unassigned = 0;
      for (const auto &vertex : vertices) {
        double t = vertex->getTime();
        ti = std::min(ti, t);
        tf = std::max(tf, t);
        if (ti == tf) {
          initial = t;
          unassigned++;
        } else if (t == ti) {
          tiVertices++;
        } else {
          tfVertices++;
        }
      }
      if (initial == ti) {
        tiVertices += unassigned;
      } else {
        tfVertices += unassigned;
      }
      return std::make_shared<SimplexOrientation>(tiVertices, tfVertices);
    }

  private:
    uint8_t ti{0};
    uint8_t tf{0};
    uint8_t k{0};
};

struct SimplexOrientationHash {
  using is_transparent = void; // enables heterogeneous lookup
  size_t operator()(const SimplexOrientation &o) const noexcept {
    return std::hash<std::uint16_t>{}(o.fingerprint());
  }
  size_t operator()(const std::shared_ptr<SimplexOrientation> &o) const noexcept {
    return std::hash<std::uint16_t>{}(o->fingerprint());
  }
  size_t operator()(const std::shared_ptr<const SimplexOrientation> &o) const noexcept {
    return std::hash<std::uint16_t>{}(o->fingerprint());
  }
  size_t operator()(uint64_t fp) const noexcept { return std::hash<std::uint16_t>{}(fp); }
};

struct SimplexOrientationEq {
  using is_transparent = void;
  bool operator()(const SimplexOrientation &a, const SimplexOrientation &b) const noexcept {
    return a.fingerprint() == b.fingerprint();
  }
  bool operator()(const SimplexOrientation &a, uint64_t fp) const noexcept { return a.fingerprint() == fp; }
  bool operator()(uint64_t fp, const SimplexOrientation &a) const noexcept { return fp == a.fingerprint(); }

  bool operator()(const std::shared_ptr<SimplexOrientation> &a,
                  const std::shared_ptr<SimplexOrientation> &b) const noexcept {
    return a->fingerprint() == b->fingerprint();
  }
  bool operator()(const std::shared_ptr<SimplexOrientation> &a, uint64_t fp) const noexcept {
    return a->fingerprint() == fp;
  }
  bool operator()(uint64_t fp, const std::shared_ptr<SimplexOrientation> &a) const noexcept {
    return fp == a->fingerprint();
  }

  bool operator()(const std::shared_ptr<const SimplexOrientation> &a,
                  const std::shared_ptr<const SimplexOrientation> &b) const noexcept {
    return a->fingerprint() == b->fingerprint();
  }
  bool operator()(const std::shared_ptr<const SimplexOrientation> &a, uint64_t fp) const noexcept {
    return a->fingerprint() == fp;
  }
  bool operator()(uint64_t fp, const std::shared_ptr<const SimplexOrientation> &a) const noexcept {
    return fp == a->fingerprint();
  }
};

inline bool operator==(const SimplexOrientationPtr &a, const SimplexOrientationPtr &b) {
  if (!a || !b) return !a && !b;
  return *a == *b; // compare underlying objects
}
}

template<>
struct std::hash<caset::SimplexOrientation> {
  size_t operator()(const caset::SimplexOrientation &s) const noexcept {
    auto [ti, tf] = s.numeric(); // OK now that getOrientation() is const
    std::uint16_t packed = (static_cast<std::uint16_t>(ti) << 8) | static_cast<std::uint16_t>(tf);
    return std::hash<std::uint16_t>{}(packed); // perfect for all (ti, tf)
  }
};

namespace caset {
/// # Simplex Class
///
/// A simplex is a generalization of the concept of a triangle or tetrahedron to arbitrary dimensions. Each simplex
/// is defined by it's vertices and edges. Each edge connects two vertices in spacetime.
///
/// Each simplex has a volume \f$ V_s \f$, which can represent various physical properties depending on the context.
///
///
class Simplex : public std::enable_shared_from_this<Simplex> {
  using SimplexPtr = std::shared_ptr<Simplex>;
  using SimplexPair = std::pair<SimplexPtr, SimplexPtr>;
  using OptionalSimplexPair = std::optional<SimplexPair>;
  using Simplices = std::vector<SimplexPtr>;
  using SimplexSet = std::unordered_set<SimplexPtr, SimplexHash, SimplexEq>;

  public:
    ///
    /// @param vertices_
    explicit Simplex(const Vertices &vertices_, Edges edges_);

    Simplex(const Vertices &vertices_, Edges edges_ ,const SimplexOrientationPtr &orientation_);

    void initialize(const std::shared_ptr<Simplex> &simplex);

    std::string toString() const;

    /// Returns the hinges of the simplex. A hinge is a simplex contained within a higher dimensional simplex. The hinge
    /// is one dimension lower than the "parent" simplex.
    /// For a 4-simplex, \f$ \sigma = {v_0, ..., v_4} \f$ there are 10 edges and 10 triangular hinges.
    /// In this case a hinge is any triangle \f$ {v_i, v_j, v_k} \f$. There are \f$ \binom{5}{3} = 10 \f$ such
    /// triangles.
    ///
    /// The curvature at the hinge is the deficit angle.
    ///
    // const std::vector<std::shared_ptr<Simplex> > getHinges() const;

    /// Assuming the simplex is a hinge; returns the deficit angle associated with the hinge.
    ///
    /// The deficit angle is given by:
    ///
    /// \f[
    /// \epsilon = 2 \pi - \sum_{\sigma \supset h} \theta_h^{(\sigma)}
    /// \f]
    ///
    /// \f$ \theta_h^{(\sigma)} \f$ is the 4D dihedral angle between the two tetrahedral faces of simplex \f$ \sigma \f$
    /// meeting along triangle (hinge) \f$ h \f$.
    ///
    /// Or in english; the deficit angle is equal to \f$ 2 \pi \f$ minus the sum of the 4D dihedral angle of each
    /// simplex between the two tetrahedral faces meeting along triangle \f$ h \f$.
    ///
    /// When the hinge is exterior/on a boundary; the \f$ 2 \pi \f$ is replaced with \f$ \pi \f$.
    ///
    // const double getDeficitAngle() const;

    /// Compute dihedral angles from edge lengths.
    /// ///
    /// Let \f$ C \f$ be the cofactors of \f$ G \f$, \f$ C = cof(G) \f$ (a matrix of cofactors). Then the dihedral angle
    /// between the two tetrahedral faces opposite vertices \f$ i \f$ and \f$ j \f$ is given by:
    ///
    /// \f[
    /// cos(\theta_{ij}) = - \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}, i \neq j, i, j \in {0, ..., n}
    /// \f]
    ///
    /// Map \f$ (i, j) \f$ to the hinge (triangle for a 4-simplex) opposite that pair.
    ///
    // const double computeDihedralAngles() const;

    [[nodiscard]] SimplexOrientationPtr getOrientation() const noexcept;

    ///
    /// @return A list of Vertex (es) in traversal order. You can iterate these to walk the Face.
    [[nodiscard]] Vertices getVertices() const noexcept;

    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] bool isTimelike() const;

    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k);

    template<typename T>
    T binomial(unsigned n, unsigned k) const;

    ///
    /// A k-simplex is the convex hull of k + 1 affinely independent points. Each has faces of all dimensions from 0 up
    /// to k–1. A k-1 simplex is called a Facet.
    ///
    /// A j-face is a j-simplex incorporating a subset (of size j) of the k-simplex vertices.
    ///
    /// The number of j-faces ( \f$ \sigma^j \f$ ) of a k-simplex \f$ \sigma^k \f$ is given by
    ///
    /// \f[
    /// \binom{k+1}{j+1}
    /// \f]
    ///
    /// And the total number of faces of all dimensions is
    /// \f$ \sum_{j=0}^{k-1} \binom{k+1}{j+1} = 2^{k+1} - 2 \f$
    ///
    std::size_t getNumberOfFaces(std::size_t j) const;

    ///
    ///
    /// A Face, \f$ \sigma^{k-1} \subset \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
    /// the k-simplex.
    ///
    /// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
    /// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
    /// by the orientation of those respective `Simplex`es.
    /// attach it
    ///
    /// The Facets are the \f$ \sigma^{k-1} \subset \sigma^{k} \f$ faces on which we'll most commonly join two simplices
    /// to form a simplicial complex \f$ K \f$.
    ///
    /// @return /// all k-1 simplices contained within this k-simplex.
    std::vector<std::shared_ptr<Simplex> > getFacets();

    std::size_t getNumberOfEdges() const;

    Fingerprint fingerprint;

    ///
    /// The co-face of a k-simplex \f$ \sigma_i^k \f$ is another k-simplex, \f$ \sigma_j^k \f$ that shares a k-1 simplex
    /// \f$ \sigma^{k-1} \f$ with \f$ \sigma_i^k \f$.
    ///
    /// We define a face as a set of shared vertices. The face of any given k-simplex \f$ \sigma^k \f$ is a k-1 simplex,
    /// \f$ \sigma^{k-1} \f$ such that \f$ \sigma^{k-1} \subset \sigma^k \f$.
    void addCoface(const std::shared_ptr<Simplex> &simplex);

    [[nodiscard]] bool hasCoface(const std::shared_ptr<Simplex> &simplex) const;


    /// @returns Edges in traversal order (the order of input vertices).
    [[nodiscard]] Edges getEdges() const;

    /// When we talk about pairty it's in the context of the orientation of a simplex's vertices within _the same time
    /// slice_. So a 2-simplex with 3 vertices at t=0 does NOT have the same orientation as a 2-simplex with it's
    /// vertices at t=1.
    [[nodiscard]]
    std::optional<Vertices>
    getVerticesWithParityTo(const std::shared_ptr<Simplex> &other) const;

    /// This method computes Edge (s) of the Simplex in traversal order. Note that the edges are effectively undirected
    /// since it can point either way as the direction relates to vertex order. So it's possible for e.g. vertices
    /// \f$ \{v_0, v_1, v_2\} \f$ to correspond to edges \f$ \{ e_{0 \rightarrow 1}, e_{2 \rightarrow 1)}, e_{2 \rightarrow 0} \} \f$
    // void computeEdges();

    [[nodiscard]] bool hasEdge(const EdgePtr &edge) const;
    [[nodiscard]] bool hasEdge(const IdType vertexAId, const IdType vertexBId) const;

    [[nodiscard]] bool hasVertex(IdType vertexId) const;

    void validate() const;

    [[nodiscard]] bool hasEdgeContaining(IdType vertexId) const;

    ///
    /// Simplices have an orientation which is given by the ordering of its Vertex (es). For a k-simplex,
    /// \f$ \sigma^k = [v_0, v_1, ..., v_k] \f$ even permutations have the _same_ orientation. Odd permutations have
    /// _opposite_ orientation.
    ///
    /// In order to glue two simplices they must have opposite orientation.
    ///
    /// For Face (s); orientation is inherited from the parent Simplex.
    ///
    /// \f[
    /// \partial\sigma^k = \partial[v_0, v_1, ..., v_k] = \sum_{i=0}^k (-1)^i [v_0, v_1, ..., v_k]
    /// \f]
    ///
    /// This method counts the number of cycles that result from mapping one set of vertex IDs to another set. That
    /// number reflects the number of "swaps" of vertices required to get from one configuration to another. Each of
    /// those swaps changes the sign of the orientation once. An odd number of swaps gives an opposite orientation; an
    /// even number gives the same orientation.
    ///
    int8_t checkParity(const std::shared_ptr<Simplex> &other) const;

    ///
    /// Co-faces are maintained as state rather than computed on the fly. This means any time a Simplex is attached to
    /// another Simplex; it must be added to the face at which it's attached as a co-face. If a Simplex, Edge, or Vertex
    /// within that Face is removed at any point; that effect should cascade up the ownership tree, which goes
    /// \f[
    /// Vertex \subset Edge \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplices that share this face.
    [[nodiscard]] std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> getCofaces() const noexcept;

    bool operator==(const Simplex &other) const noexcept;

    /// This method just returns whether or not the simplex has fewer than 2 co-faces. If it does; then it is available.
    bool isCausallyAvailable() const noexcept;

    ///
    /// This method iterates over all faces of this Simplex; and counts the number of co-faces for each face. If a face
    /// has fewer than 2 co-faces; it's available to glue. We limit to 2 co-faces because we want to preserve
    /// manifoldness. There's nothing wrong with internal simplicies from the perspective of simplicial algebra, but
    /// there is from the perspective of relativity.
    ///
    /// @return Whether or not this Simplex is available to glue. A face is only available when it has less than 2
    /// co-faces.
    bool hasCausallyAvailableFacet();

    bool isInternal() const noexcept;

    static std::shared_ptr<Simplex> create(const Vertices &vertices_, const Edges &edges_);
    static std::shared_ptr<Simplex> create(const Vertices &vertices_, const Edges &edges_, const SimplexOrientationPtr &orientation_);

    /// This method computes the maximum number of k+1 co-faces that can be joined to this k-Simplex _in general_.
    /// Do not use this method the purpose of causal gluing in CDT. It would create internal/non-manifold simplices and
    /// hence violate causality. If that's your goal then you want to use `isCausallyAvailable`
    ///
    /// For a given k-simplex \f$ \sigma^k \f$, a co-face is defined as an m-simplex, \f$ \sigma^m \f$ such that \f$ m > k \f$
    /// and \f$ \sigma^k \subset \sigma^m \f$. The maximum number of co-faces that can be joined to a k-simplex is in
    /// general unbounded, but for our purposes we set it to the number of faces of the simplex, so we impose the
    /// constraint that the coface no be _generally_ \f$ m > k \f$, but exactly \f$ k + 1 \f$, so \f$ m = k + 1 \f$.
    ///
    /// This can be confusing because for the purpose of causally gluing simplices we look at a face, \f$ \sigma^k \f$
    /// of the (k+1)-simplex, \f$ \sigma{k+1} \f$ where to that (k-) face we want to glue another (k+1)-simplex on one
    /// of it's k-faces. So the maximum number of co-faces that can be joined to a k-simplex is the number of faces of
    /// that simplex.
    ///
    /// @return
    std::size_t maxKPlusOneCofaces() const;

    ///
    /// @return A set of orientations for faces that can be glued to this Simplex. We look at it's available faces, and
    ///   create a unique set of the orientations. That set can be used to look up a corresponding Simplex in the
    ///   `externalSimplices` of the Spacetime.
    ///
    SimplexOrientations getGluableFaceOrientations();

    bool operator==(const std::shared_ptr<Simplex> &other) const noexcept;

    VertexIdMap getVertexIdLookup() const noexcept;

    void attach(const VertexPtr &unattached, const VertexPtr &attached, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList);

    SimplexSet getAvailableFacetsByOrientation(const SimplexOrientationPtr &orientation);

    /// Marks a facet as un(causally)available on it's cofaces by orientation. This way when we request available
    /// facets by orientation this facet won't be returned.
    void markAsUnavailable();

    void markFacetAsUnavailable(const SimplexPtr &facet) {
      availableFacetsByOrientation[facet->getOrientation()].erase(facet);
    }

  private:
    SimplexOrientationPtr orientation{};
    VertexIdMap vertexIdLookup{};
    Vertices vertices{};
    Edges edges{};

    std::vector<std::shared_ptr<Simplex> > facets{};
    std::unordered_map<SimplexOrientationPtr, SimplexSet, SimplexOrientationHash, SimplexOrientationEq> availableFacetsByOrientation{};
    std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> cofaces{};

    /// This method replaces the vertex only, Edge (s) should be replaced by the Spacetime, because it maintains the
    /// global lookup for Edge (s). If the Edge source/target is replaced; it's not enough to update the Edge, since
    /// squaredLength data could be lost.
    bool replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex);

    template<typename Method, typename... Args>
    bool cascade(Method method, bool up, bool down, Args &&... args);

};

using SimplexPtr = std::shared_ptr<Simplex>;
using SimplexPair = std::pair<SimplexPtr, SimplexPtr>;
using OptionalSimplexPair = std::optional<SimplexPair>;
using Simplices = std::vector<SimplexPtr>;
using SimplexSet = std::unordered_set<SimplexPtr, SimplexHash, SimplexEq>;
}

template<>
struct std::hash<caset::Simplex> {
  size_t operator()(const caset::Simplex &s) const noexcept {
    return std::hash<std::uint64_t>{}(s.fingerprint.fingerprint());
  }
};

#endif //CASET_SIMPLEX_H
