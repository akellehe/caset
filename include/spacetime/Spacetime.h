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

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "topologies/Topology.h"
#include "observables/Observable.h"
#include "EdgeList.h"
#include "VertexList.h"
#include "Metric.h"
#include "Simplex.h"
#include "topologies/Toroid.h"

namespace caset {
enum class SpacetimeType : uint8_t {
  CDT = 0,
  REGGE = 1,
  COSET = 2,
  REGGE_PACHNER = 3,
  GFT_SPIN_FOAM = 4,
  RICCI_FLOW_DISCRETIZATION = 5
};

///
/// # Spacetime
///
/// The Spacetime class provides methods to create and manipulate the basic building blocks of a simplicial
/// complex.
///
/// The Spacetime Topology is responsible for constructing Simplex(es) and the Topology (subclass) is responsible for
/// building the complex to match that topology.
///
/// Any assertions or state needed by the Topology to build the complex should be implemented in the Simplex.
///
class Spacetime {
  public:
    using Bucket = std::unordered_set<SimplexPtr, SimplexHash, SimplexEq>;

    Spacetime() {
      Signature signature(4, SignatureType::Lorentzian);
      metric = std::make_shared<Metric>(true, signature);
      spacetimeType = SpacetimeType::CDT;
      alpha = 1.;
      topology = std::make_shared<Toroid>();
    }

    Spacetime(
      std::shared_ptr<Metric> metric_,
      const SpacetimeType spacetimeType_,
      std::optional<double> alpha_,
      std::optional<std::shared_ptr<Topology> > topology_) : metric(metric_), spacetimeType(spacetimeType_) {
      alpha = alpha_.value_or(1.);
      topology = topology_.value_or(std::make_shared<Toroid>());
    }

    SimplexPtr createSimplex(const Vertices &vertices, const Edges &edges);
    SimplexPtr createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation);
    SimplexPtr createSimplex(std::size_t k);
    VertexPtr createVertex(const std::uint64_t id) noexcept;
    VertexPtr createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept;
    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept { return spacetimeType; }
    [[nodiscard]] double getCurrentTime() const noexcept { return static_cast<double>(currentTime); }
    [[nodiscard]] std::shared_ptr<EdgeList> getEdgeList() noexcept { return edgeList; }
    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept { return metric; }
    [[nodiscard]] std::shared_ptr<VertexList> getVertexList() noexcept { return vertexList; }
    double incrementTime() noexcept {
      currentTime++;
      return static_cast<double>(currentTime);
    }
    std::shared_ptr<Edge> createEdge(const std::uint64_t src, const std::uint64_t tgt);
    std::shared_ptr<Edge> createEdge(const std::uint64_t src, const std::uint64_t tgt, double squaredLength) noexcept;
    void addObservable(const std::shared_ptr<Observable> &observable) { observables.push_back(observable); }

    ///
    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    void build(int numSimplices=3);

    ///
    /// This method identifies a pair of faces (one from each simplex) that can be glued together while preserving the
    /// orientation of the simplices. The method checks for matching orientations and edge lengths to ensure
    /// compatibility.
    ///
    /// Before simplices are glued into the complex we consider them 'detached', so it doesn't matter if we're attaching
    /// a (3, 2) or a (2, 3). There's a pairty building method, `Simplex::getVerticesWithParityTo(yourFace)`, that finds
    /// the right order to use when attaching the Simplex to the Simplicial complex.
    ///
    /// @param unattachedSimplex A simplex not yet attached to the simplicial complex.
    /// @param attachedSimplex An attached simplex to which you would like to glue the first. Orientation is based on
    ///   this simplex.
    ///
    /// @return {unattached, attached} faces that can be glued together.
    [[nodiscard]] OptionalSimplexPair
    getGluableFaces(const SimplexPtr &unattachedSimplex, const SimplexPtr &attachedSimplex);

    void moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to);

    void moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to);

    bool removeIfIsolated(const VertexPtr &vertex) {
      if (vertex->degree() == 0) {
        CLOG(DEBUG_LEVEL, "Removing vertex: ", vertex->toString());
        vertexList->remove(vertex);
        return true;
      }
      CLOG(DEBUG_LEVEL, "NOT Removing vertex: ", vertex->toString());
      return false;
    }

    void attachAtVertices(
      const SimplexPtr &unattached,
      const SimplexPtr &attached,
      const std::vector<std::pair<VertexPtr, VertexPtr> > &vertexPairs // {unattached, attached}
    );

    ///
    /// This method is a simplicial isomorphism between two faces. Specifically; it takes two Simplex Face (s),
    /// \f$ \sigma^{k-1}_{myFace} \f$ and \f$ \sigma^{k-1}_{yourFace} \f$ as inputs and creates a new face
    /// \f$ \sigma^{k-1}_{newFace} \f$ indicating their adjacency in the simplicial complex while preserving the
    /// orientation of both their cofaces.
    ///
    /// This method runs within the context of an n-dimensional simplicial manifold; each (n-1) simplex (where faces are
    /// codimension-1) is incident to exactly 2 n-simplices for interior faces and exactly 1 n-simplex for faces along
    /// the boundary.
    ///
    /// Because this method is (causal) orientation-aware; it's intended only to be used when we're building causal
    /// simplicial complexes.
    ///
    /// If any face is shared by 3 or more n-simplices; then the neighborhood of some point becomes interior and is no
    /// longer homeomorphic to \f$ \mathcal{R}^n \f$ or a half-space
    /// \f$ \mathcal{R}^{n-1} \multiply [0, \inf] \f$, (the boundary points) so the spacetime effectively branches,
    /// causing it to lose it's manifold properties.
    ///
    /// The building blocks of a 4D causal simplicial complex are (4, 1) and (3, 2) simplices. The (4, 1) simplex has 4
    /// vertices on t=t and 1 on t=t + 1. The (3, 2) simplex has 3 vertices on t=t and 2 on t=t + 1. We build out the
    /// complex by gluing these simplices together along their faces.
    ///
    /// The Face (s) or Facets of the simplex are all sets of vertices of cardinality k-1. So for a 4-simplex
    /// \f$ \sigma^{(1, 4)}_{ab} \f$ we have vertices \f$ \{a_0, a_1, a_2, a_3, b_0\} \f$ and the
    /// 4-faces are the (ordered) combinations of the 4 vertices, where \f$ a \f$ vertices are at \f$ t=t \f$ and \f$ b \f$
    /// vertices are at \f$ t=t+1 \f$.
    ///
    /// \f[
    /// \sigma^{(4, 0)}_0 = \{a_1, a_2, a_3, b_0\}
    /// \sigma^{(3, 1)}_1 = \{a_0, a_2, a_3, b_0\}
    /// \sigma^{(3, 1)}_2 = \{a_0, a_1, a_3, b_0\}
    /// \sigma^{(3, 1)}_3 = \{a_0, a_1, a_2, b_0\}
    /// \sigma^{(3, 1)}_4 = \{a_0, a_1, a_2, a_3\}
    /// \f]
    ///
    /// When a face has vertices from both sets \f$ {a_i \in A} \f$ and \f$ {b_i \in B} \f$; the face is spacelike. When
    /// it has only vertices from one or the other; it's timelike.
    ///
    /// Let \f$ \sigma^{(3, 2)}_{cd} \f$ have vertices \f$ \{c_i \in C | i \in [0, 4]\} \f$ and \f$ \{d_i \in D | i \in [0, 4]\} \f$
    /// where vertices of \f$ C \f$ have \f$ t=t \f$ and in \f$ D \f$ they have \f$ t = t+1 \f$. Then \f$ \sigma^{(3, 2)}_{cd} \f$ has
    /// faces,
    ///
    /// \f[
    /// \{\sigma^{(2, 2)}_0, \sigma^{(2, 2)}_1, \sigma^{(2, 2)}_2, \sigma{(3, 1)}_3, \sigma{(3, 1)_4\} \memberof \sigma^{(3, 1)}_{cd}
    /// \f]
    ///
    /// We can only join faces of the same shape, e.g. (3, 1) in this case.
    ///
    /// For a detailed picture; see "Quantum Gravity from Causal Dynamical Triangulations: A Review", R. Loll, 2019.
    /// Figure 1.
    ///
    ///
    /// The process of attaching faces amounts to moving external in-edges and out-edges from the vertices of the
    /// unattachedFace to the analogous (parity matches) vertices of the attachedFace.
    ///
    /// This means we identify the vertices that match, in order. There are two classes of edges now. Those internal to
    /// the unattachedFace, and those that have a vertex outside the unattachedFace. The internal edges have analogous
    /// edges in the attachedFace, so we can delete those edges, replacing them with their analogous counterparts in the
    /// attachedFace. The external edges need to be moved from the unattachedFace vertex to the attachedFace vertex.
    ///
    /// @param attachedFace The Face of this Simplex to attach to `unattachedFace` of the other Simplex
    /// @param unattachedFace The Face of the other Simplex to attach to `attachedFace` of this Simplex.
    /// @returns {attachedFace, succeeded} The `attachedFace` after attachment and whether the attachment succeeded.
    std::tuple<SimplexPtr, bool> causallyAttachFaces(const SimplexPtr &attachedFace, const SimplexPtr &unattachedFace);

    ///
    /// @return Simplices around the boundary of the simplicial complex to which they belong. These simplices have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplices as you might expect, but 5-simplices since those are the standard
    /// building blocks. You can get the 2-simplices by calling `getFacets()` on the 5-simplices and their facets until
    /// \f$ k=2 \f$.
    [[nodiscard]]
    SimplexSet getExternalSimplices() noexcept {
      SimplexSet simplices{};
      for (const auto &[facialOrientation, bucket] : externalSimplices) {
        for (const auto &simplex : bucket) {
          simplices.insert(simplex);
        }
      }
      return simplices;
    }

    void embedEuclidean(int dimensions, double epsilon);

    /// This method chooses a simplex from the boundary of the simplicial complex to which `unattachedSimplex` can be
    /// glued. It does this by iterating through the `externalSimplices` and checking for compatible orientations and
    /// edge lengths.
    ///
    /// To the extent the hashing function for vertex fingerprinting is good; this should be pretty well pseudo-random.
    /// If you want something truly random, though, you should probably implement that.
    ///
    /// @returns A pair of \f$ k-1 \f$ simplices (faces) if a compatible k-simplex was found. None otherwise.
    OptionalSimplexPair chooseSimplexFacesToGlue(const SimplexPtr &unattachedSimplex);

    /// This method is for testing only, very poor runtime performance.
    SimplexSet getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
      SimplexOrientationPtr o = std::make_shared<
        SimplexOrientation>(std::get<0>(orientation), std::get<1>(orientation));
      SimplexSet result{};
      for (const auto &bucket : externalSimplices | std::views::values) {
        for (const auto &simplex : bucket) {
          for (const auto &simplexFacialOrientation : simplex->getOrientation()->getFacialOrientations()) {
            if (simplex->getOrientation() == o) result.insert(simplex);
          }
        }
      }
      return result;
    }

    [[nodiscard]] std::vector<Vertices> getConnectedComponents() const;

    void mergeVertices(const VertexPtr &into, const VertexPtr &from);

  private:
    std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();

    IdType vertexIdCounter = 0;
    SpacetimeType spacetimeType;
    double alpha = 1.;
    std::shared_ptr<Metric> metric;
    std::shared_ptr<Topology> topology;
    std::uint64_t currentTime = 0;

    ///
    /// These are simplices on the boundary of a simplicial complex. They have at least one external face, and hence can
    /// be glued to other simplices. The externalSimplices are organized by the orientation of their available faces. If
    /// a face is available; the orientation of that face can be found as a key corresponding to a SimplexSet containing
    /// the Simplex to which that Face belongs.
    ///
    /// This makes for fast lookups when gluing simplices together to form a complex.
    std::unordered_map<SimplexOrientationPtr, SimplexSet, SimplexOrientationHash, SimplexOrientationEq> externalSimplices{};

    ///
    /// These are simplices that are fully internal to the simplicial complex. They have no external faces, and hence
    /// cannot be glued to other simplices.
    ///
    /// A Simplex becomes _internal_ when all it's _external_ faces have been glued. At that point it is no longer
    /// relevant to store that simplex by the orientation of any given face, so _internal_ simplices are stored by the
    /// orientation of the Simplex itself.
    std::unordered_map<SimplexOrientationPtr, SimplexSet, SimplexOrientationHash, SimplexOrientationEq> internalSimplices{};
    std::vector<std::shared_ptr<Observable> > observables{};
};
} // caset

#endif //CASET_SPACETIME_H
