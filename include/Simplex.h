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

#include "ForwardDeclarations.h"
#include <memory>
#include <vector>
#include <functional>

#include "Logger.h"
#include "EdgeList.h"
#include "Fingerprint.h"
#include "SimplexOrientation.h"
#include "utils.h"

namespace caset {

/// # Simplex Class
///
/// A simplex is a generalization of the concept of a triangle or tetrahedron to arbitrary dimensions. Each simplex
/// is defined by its vertices.
///
/// Each simplex has a volume \f$ V_s \f$, which can represent various physical properties depending on the context.
///
/// ### Some Implementation Details
///
/// A k-simplex, \f$ \sigma^k \f$, within a simplicial complex, \f$ K \f$ is defined as a set of k+1 vertices.
/// Simplicial complex construction is a bit of a bottleneck in simulation of spacetime. At the moment; we declare some
/// vertices, then use coning to create a Simplex from those vertices. Those vertices are passed to the Simplex along
/// with the edges used to connect them as a performance optimization.
///
/// Most of the time building the simplicial complex is spent calculating facets from all subsets of Simplex Vertices. A
/// faster method for building the complex would be to avoid computing those vertices and edges; and just compute the
/// simplex as an abstraction with faces, cofaces, and an orientation. We'll leave this for a "Version 2 feature".
///
class Simplex : public std::enable_shared_from_this<Simplex> {
  public:
    // ==================== Static Factory Methods ====================
    static std::shared_ptr<Simplex> create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_);
    static std::shared_ptr<Simplex> create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_, const SimplexOrientation &orientation_);
    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k);

    // ==================== Constructors & Initialization ====================
    ///
    /// @param vertices_
    explicit Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_);
    Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_ ,const SimplexOrientation &orientation_);

    std::uint64_t size() const noexcept;

    /// The initialize step is necessary because the canonical owner of the Simplex object is the Spacetime, and ideally
    /// that canonical owner is the only one to permanently hang onto a std::shared_ptr. So when we initialize with this
    /// method we add the std::shared_ptr<Simplex> (aka SimplexPtr) to all the Vertex (es) that are members of the
    /// Simplex. Again; we define a Simplex abstractly as a set of vertices with a time orientation.
    /// When you construct a Spacetime which can abstractly be considered a Simplicial complex; having access to the
    /// Simplex by Vertex is pretty handy for bookkeeping.
    void initialize(const std::shared_ptr<Simplex> &simplex);

    // ==================== String Representation ====================
#ifdef CASET_VERBOSE
    std::string toString() const noexcept;
#else
    std::string toString() const noexcept {
      return "";
    }
#endif

    // ==================== Basic Getters ====================
    /// Each simplex has an associated _orientation_ in the case you're preserving causality with your work. You can
    /// find specifics of the SimplexOrientation abstractly and concretely/computationally in the documentation for the
    /// SimplexOrientation
    [[nodiscard]] SimplexOrientation getOrientation() const noexcept;

    // ==================== Vertex Queries ====================
    ///
    /// @return A list of Vertex (es) in traversal order. You can iterate these to walk the Face.
    [[nodiscard]] VertexPtrs getVertices() const noexcept;

    /// This method is self-explanatory. O(1) lookups for who has what.
    [[nodiscard]] bool hasVertex(const VertexPtr &vertex) const;

    /// This method produces a lookup table \f$ Id \rightarrow Vertex \f$. The only place it's used at the moment is for
    /// verifying state in our Python unit tests.
    [[nodiscard]] VertexIdMap getVertexIdLookup() const noexcept;

    /// This method should be used when you have two simplices and you want to find some Vertices at which to join one
    /// Simplex to the other. Two sets of ordered vertices, \f$ \mathcal{V_i} \f$ and \f$ \mathcal{V_j} \f$ "have parity"
    /// when there is an ordered subset, \f$ V_i \f$ and \f$ V_j \f$ for which Vertex::getTime returns the same value
    /// for each corresponding element. More specifically;
    ///
    /// Two sets of vertices "have pairty" iff
    ///   - They have the same cardinality, \f$ N \f$
    ///   - There exists an element \f$ v_m \memberof \f$ V_i \f$ for which there exists an an element
    ///     \f$ v_n \memberof V_j \f$ such that \f$ v_m::getTime() = v_n::getTime() \forall (m, n) s.t. m = n \f$.
    [[nodiscard]] std::optional<VertexPtrs> getVerticesWithParityTo(const std::shared_ptr<Simplex> &other) const;

    // ==================== Edge Queries ====================
    /// @returns Edges in traversal order (the order of input vertices).
    [[nodiscard]] EdgePtrSet getEdges() const;
    [[nodiscard]] std::size_t getNumberOfEdges() const;
    /// This method computes Edge (s) of the Simplex in traversal order. Note that the edges are effectively undirected
    /// since it can point either way as the direction relates to vertex order. So it's possible for e.g. vertices
    /// \f$ \{v_0, v_1, v_2\} \f$ to correspond to edges \f$ \{ e_{0 \rightarrow 1}, e_{2 \rightarrow 1)}, e_{2 \rightarrow 0} \} \f$
    [[nodiscard]] bool hasEdge(const EdgePtr &edge) const;
    [[nodiscard]] bool hasEdge(const VertexPtr &vertexA, const VertexPtr &vertexB) const;
    [[nodiscard]] bool hasEdgeContaining(IdType vertexId) const;

    // ==================== Face & Facet Queries ====================
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
    /// A Face, \f$ \sigma^{k-1} \subset \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
    /// the k-simplex.
    ///
    /// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
    /// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
    /// by the orientation of those respective `Simplex`es.
    ///
    /// The Facets are the \f$ \sigma^{k-1} \subset \sigma^{k} \f$ faces on which we'll most commonly join two simplices
    /// to form a simplicial complex \f$ K \f$.
    ///
    /// @return all k-1 simplices contained within this k-simplex.
    [[nodiscard]] const Simplices &getFacets();

    bool hasFacets() const;
    bool hasStoredFacet(const SimplexPtr &facet);
    bool addFacet(const SimplexPtr &facet);
    bool removeFacet(const SimplexPtr &facet);

    // ==================== Coface Queries & Management ====================
    ///
    /// A simplex, \f$ \sigma \memberof K \f$ with vertices \f$ V_{\sigma} \f$  is a coface of \f$ \tau \memberof K \f$
    /// with vertices \f$ V_{\tau} \f$ iff \f$ V_{\tau} \subset V_{\sigma} \f$. For our purposes, however, we confine
    /// cofaces to those of dimensionality \f$ k+1 \f$ compared to the facet of dimension \f$ k \f$
    ///
    /// We define a _facet_ as a set of shared vertices. The facet of any given k-simplex \f$ \sigma^k \f$ is a k-1
    /// simplex, such that  \f$ \sigma{k} \f$ is a coface of \f$ \sigma_{k-1} \f$.
    ///
    void addCoface(const std::shared_ptr<Simplex> &simplex);
    void removeCoface(const std::shared_ptr<Simplex> &simplex);
    bool isCofaceTo(const SimplexPtr &simplex, bool shallow=true) const;

    Simplices clearCofaces();
    Simplices clearFacets();

    // static void unregisterFromCofaces(const std::shared_ptr<Simplex> &facet);
    // static void registerToCofaces(const std::shared_ptr<Simplex> &facet);

    [[nodiscard]] bool hasCoface(const std::shared_ptr<Simplex> &simplex) const;

    ///
    /// Co-faces are maintained as state rather than computed on the fly. This means any time a Simplex is attached to
    /// another Simplex; it must be added to the face at which it's attached as a co-face. If a Simplex, Edge, or Vertex
    /// within that Face is removed at any point; that effect should cascade up the ownership tree, which goes
    /// \f[
    /// Vertex \subset Edge \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplices that share this face.
    [[nodiscard]] SimplexPtrSet getCofaces() const noexcept;

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

    // ==================== State Queries ====================
    [[nodiscard]] bool isTimelike() const;
    /// This method just returns whether or not the simplex has fewer than 2 co-faces. If it does; then it is available.
    bool isCausallyAvailable() const noexcept;
    bool referencesSimplex(const SimplexPtr &simplex);
    static std::tuple<SimplexPtr, Simplices, Simplices> breakReferences(const SimplexPtr &simplex);
    static void restoreReferences(SimplexPtr &simplex, const Simplices &cofaces, const Simplices &facets);
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
    std::uint64_t hash() const noexcept;

    // ==================== Computational & Utility Methods ====================
    template<typename T> T binomial(unsigned n, unsigned k) const;

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
    /// @return A set of orientations for faces that can be glued to this Simplex. We look at it's available faces, and
    ///   create a unique set of the orientations. That set can be used to look up a corresponding Simplex in the
    ///   `externalSimplices` of the Spacetime.
    ///
    SimplexOrientationSet getGluableFaceOrientations();

    // ==================== Modification Methods ====================
    bool addEdge(const EdgePtr &edge);
    bool removeEdge(const EdgePtr &edge);
    static void registerToVertices(const SimplexPtr &simplex);
    static void registerToFacets(const SimplexPtr &coface);
    static void unregisterFromVertices(const SimplexPtr &simplex);
    static Simplices unregisterFromFacets(const SimplexPtr &simplex);

    // ==================== Validation ====================
    void validate() const;

    // ==================== Operators ====================
    bool operator==(const Simplex &other) const noexcept;
    bool operator==(const std::shared_ptr<Simplex> &other) const noexcept;

    // ==================== Public Data ====================
    Fingerprint fingerprint{};
    bool initialized{false};
    OwnershipManager<IdType, SimplexPtr, SimplexPtrHash, SimplexPtrEq> ownershipManager{};

    // ==================== Commented Out / Future Methods ====================
    /// Returns the hinges of the simplex. A hinge is a simplex contained within a higher dimensional simplex. The hinge
    /// is one dimension lower than the "parent" simplex.
    /// For a 4-simplex, \f$ \sigma = {v_0, ..., v_4} \f$ there are 10 edges and 10 triangular hinges.
    /// In this case a hinge is any triangle \f$ {v_i, v_j, v_k} \f$. There are \f$ \binom{5}{3} = 10 \f$ such
    /// triangles.
    ///
    /// The curvature at the hinge is the deficit angle.
    ///
    // const Simplices getHinges() const;

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
    ///
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
    // void computeEdges();

    /// This method replaces the vertex only, Edge (s) should be replaced by the Spacetime, because it maintains the
    /// global lookup for Edge (s). If the Edge source/target is replaced; it's not enough to update the Edge, since
    /// squaredLength data could be lost.
    ///
    /// WARNING: This Simplex must be removed from it's containers prior to calling this method. NOT removing it from it's
    ///   containers _first_ (and adding back in after) results in UNDEFINED BEHAVIOR!
    ///
    /// @param oldVertex The Vertex to replace
    /// @param newVertex The vertex with which to replace it.
    /// @return
    bool replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex);
  private:
    Spacetime *spacetime{nullptr};
    SimplexOrientation orientation{};

    VertexIdToIndex vertexIdToIndex{};
    VertexIndexToId vertexIndexToId{};
    VertexPtrs vertices{};

    EdgePtrSet edges{};

    Simplices facets{};
    SimplexPtrSet cofaces{};

    bool _isTimelike{false};
    template<typename Method, typename... Args>
    bool cascade(Method method, bool up, bool down, Args &&... args);
};

}

#endif //CASET_SIMPLEX_H
