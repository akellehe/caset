//
// Created by andrew on 10/23/25.
//

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>
#include <optional>

#include "topologies/Topology.h"
#include "observables/Observable.h"
#include "EdgeList.h"
#include "VertexList.h"
#include "Metric.h"
#include "Simplex.h"
#include "topologies/Toroid.h"
#include "Logger.h"

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
    using Bucket = std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq>;

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

    ///
    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    void build() {
      // return topology->build(this);
    }

    void addObservable(const std::shared_ptr<Observable> &observable) {
      observables.push_back(observable);
    }

    std::shared_ptr<Vertex> createVertex(const std::uint64_t id) noexcept {
      manual = true;
      return vertexList->add(id);
    }

    std::shared_ptr<Vertex> createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      manual = true;
      return vertexList->add(id, coords);
    }

    std::shared_ptr<Edge> createEdge(
      const std::uint64_t src,
      const std::uint64_t tgt
    ) {
      manual = true;
      std::shared_ptr<Edge> edge = edgeList->add(src, tgt);
      vertexList->get(src)->addOutEdge(edge);
      vertexList->get(tgt)->addInEdge(edge);
      return edge;
    }

    std::shared_ptr<Edge> createEdge(
      const std::uint64_t src,
      const std::uint64_t tgt,
      double squaredLength
    ) noexcept {
      manual = true;
      std::shared_ptr<Edge> edge = edgeList->add(src, tgt, squaredLength);
      vertexList->get(src)->addOutEdge(edge);
      vertexList->get(tgt)->addInEdge(edge);
      return edge;
    }

    std::shared_ptr<Simplex> createSimplex(
      std::vector<std::shared_ptr<Vertex> > &vertices
    ) noexcept {
      manual = true;
      const SimplexOrientation orientation = SimplexOrientation::orientationOf(vertices);
      auto &bucket = externalSimplexes.try_emplace(orientation /*key*/).first->second;
      std::vector<IdType> _ids = {};
      _ids.reserve(vertices.size());
      for (const auto &vertex : vertices) {
        _ids.push_back(vertex->getId());
      }
      auto [fingerprint, n, ids] = Fingerprint::computeFingerprint(_ids);
      if (!bucket.contains(fingerprint)) {
        std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices);
        bucket.insert(simplex);
        return simplex;
      }
      return *bucket.find(fingerprint);
    }

    std::shared_ptr<Simplex> createSimplex(std::size_t k) {
      if (manual) {
        throw std::runtime_error(
          "You can't mix user-defined vertex/edge/simplex definitions with internal definitions. This happens when you call createSimplex(k) after you've called createVertex/createEdge/createSimplex(vertices)");
      }
      double squaredLength = alpha;
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < k; i++) {
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        std::shared_ptr<Vertex> newVertex = vertexList->add(vertexList->size());
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices);
      externalSimplexes[simplex->getOrientation()].insert(simplex);
      return simplex;
    }

    void setManual(bool manual_) noexcept {
      manual = manual_;
    }

    std::shared_ptr<Simplex> createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
      if (manual) {
        throw std::runtime_error(
          "You can't mix user-defined vertex/edge/simplex definitions with internal definitions. This happens when you call createSimplex(k) after you've called createVertex/createEdge/createSimplex(vertices)");
      }
      double squaredLength = alpha;
      double timelikeSquaredLength = alpha;
      SimplexOrientation orientation(std::get<0>(numericOrientation), std::get<1>(numericOrientation));
      std::uint8_t k = orientation.getK();
      CASET_LOG(INFO_LEVEL,
                "creating a ",
                std::to_string(k),
                "-simplex from a numeric orientation (",
                std::to_string(std::get<0>(numericOrientation)),
                ", ",
                std::to_string(std::get<1>(numericOrientation)),
                ")");
      auto [ti, tf] = orientation.numeric();
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < ti; i++) {
        // Create ti Timelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        CASET_LOG(INFO_LEVEL, "Creating a new vertex with ID=", vertexList->size());
        std::shared_ptr<Vertex> newVertex = vertexList->add(vertexList->size(), {static_cast<double>(currentTime)});
        if (getMetric()->getSignature()->getSignatureType() == SignatureType::Lorentzian) {
          timelikeSquaredLength = -alpha;
        }
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList->
              add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          CASET_LOG(INFO_LEVEL,
                    "Creating a new timelike edge from ",
                    existingVertex->getId(),
                    "->",
                    newVertex->getId());
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      for (int i = 0; i < tf; i++) {
        // Create ti Spacelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        CASET_LOG(INFO_LEVEL, "Creating a new vertex with ID=", vertexList->size());
        std::shared_ptr<Vertex> newVertex = vertexList->add(vertexList->size(), {static_cast<double>(currentTime + 1)});
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge;
          if (existingVertex->getTime() < newVertex->getTime()) {
            CASET_LOG(INFO_LEVEL,
                      "Creating a new spacelike edge (L^2 = \\alpha) from ",
                      existingVertex->getId(),
                      "->",
                      newVertex->getId());
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
          } else {
            CASET_LOG(INFO_LEVEL,
                      "Creating a new timelike edge (L^2 = -\\alpha) from ",
                      existingVertex->getId(),
                      "->",
                      newVertex->getId());
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          }
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices, orientation);
      externalSimplexes[orientation].insert(simplex);
      return simplex;
    }

    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept {
      return spacetimeType;
    }

    [[nodiscard]] std::shared_ptr<EdgeList> getEdgeList() noexcept {
      return edgeList;
    }
    [[nodiscard]] std::shared_ptr<VertexList> getVertexList() noexcept {
      return vertexList;
    }

    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept { return metric; }

    [[nodiscard]] double getCurrentTime() const noexcept {
      return static_cast<double>(currentTime);
    }

    double incrementTime() noexcept {
      currentTime++;
      return static_cast<double>(currentTime);
    }

    [[nodiscard]] static std::optional<std::pair<std::shared_ptr<Simplex>, std::shared_ptr<Simplex> > >
    getGluablePair(const std::shared_ptr<Simplex> &sA, const std::shared_ptr<Simplex> &sB) {
      auto facetsA = sA->getFacets(); // vector<shared_ptr<Simplex>>
      auto facetsB = sB->getFacets();

      for (auto &fA : facetsA) {
        auto [pA, qA] = fA->getOrientation().numeric();
        if (pA + qA != 4) continue; // just to be explicit it's a tetrahedron

        for (auto &fB : facetsB) {
          auto [pB, qB] = fB->getOrientation().numeric();
          if (pB != pA || qB != qA) continue; // requires same (1,3) type

          // Now check orientation on the shared face:
          // checkPairty should be -1 for opposite orientation.
          // int8_t parity = fA->checkPairty(fB);
          // if (parity != -1) {
          // Either same orientation (+1) or they don’t match at all (0).
          // continue;

          // (Optionally) check edge lengths match within epsilon
          // to enforce metric consistency...
          return std::make_optional(std::make_pair(fA, fB));
        }
      }

      return std::nullopt;
    }

    void moveVertex(const VertexPtr &from, const VertexPtr &to) {
      for (const auto &edge : from->getInEdges()) {
        from->removeInEdge(edge);
        edgeList->remove(edge);
        edge->replaceTargetVertex(to->getId());
        to->addInEdge(edgeList->add(edge));
        removeIfIsolated(from);
      }
    }

    void removeIfIsolated(const VertexPtr &vertex) {
      if (vertex->degree() == 0) {
        vertexList->remove(vertex);
      }
    }

    void moveEdges(
      const VertexPtr &fromVertex,
      const std::shared_ptr<Simplex> &fromSimplex,
      const VertexPtr &toVertex,
      const std::shared_ptr<Simplex> &toSimplex,
      bool moveInEdges
      ) {
      const auto edges = moveInEdges ? fromVertex->getInEdges() : fromVertex->getOutEdges();
      for (const auto &edge : edges) {
        if (fromSimplex->hasEdge(edge->getSourceId(), edge->getTargetId())) {
          const VertexPtr &sourceVertex = vertexList->get(edge->getSourceId());
          const VertexPtr &targetVertex = vertexList->get(edge->getTargetId());
          sourceVertex->removeOutEdge(edge);
          targetVertex->removeInEdge(edge);
          edgeList->remove(edge);
          removeIfIsolated(sourceVertex);
          removeIfIsolated(targetVertex);
          continue;
        }

        moveVertex(fromVertex, toVertex);
        fromSimplex->replaceVertex(fromVertex, toVertex);
        // TODO: Move stuff from cofaces.
        // for (const auto &coface : yourFace->getCofaces()) {
        // coface->replaceVertex(yourVertex, myVertex);
        // }
      }
    }

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
    /// \f$ \mathcal{R}^{n-1} \multiply [0, \inf] \f$, (the boundary points) so the spacetime effectively branches.
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
    /// @param myFace The Face of this Simplex to attach to `yourFace` of the other Simplex
    /// @param yourFace The Face of the other Simplex to attach to `myFace` of this Simplex.
    /// @return myFace, updated with the second simplex glued.
    std::tuple<std::shared_ptr<Simplex>, bool> causallyAttachFaces(std::shared_ptr<Simplex> &myFace,
                                                                   const std::shared_ptr<Simplex> &yourFace) {
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(myFace->size());
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(myFace->size());

      // Two vertices are compatible to attach iff they share the same time value.
      std::vector<std::pair<std::shared_ptr<Vertex>, std::shared_ptr<Vertex> > > vertexPairs{};
      vertexPairs.reserve(myFace->size());

      // These are in order of traversal, you can iterate them to walk the Face:
      const auto &yourVertices = yourFace->getVertices();

      // myVertices and yourVertices should have a sequence that lines up, but they're not necessarily at the correct
      // starting node. We should shuffle through until they are either compatible or we've tried all possible orders.
      const std::optional<std::vector<std::shared_ptr<Vertex> > > myOrderedVerticesOptional = myFace->
          getVerticesWithParityTo(yourFace);

      if (!myOrderedVerticesOptional.has_value()) {
        CASET_LOG(WARN_LEVEL, "No compatible vertex order found for myFace and yourFace.");
        return {nullptr, false};
      }

      const std::vector<std::shared_ptr<Vertex> > &myOrderedVertices = myOrderedVerticesOptional.value();
      for (auto i = 0; i < myOrderedVertices.size(); i++) {
        const auto &myVertex = myOrderedVertices[i];
        const auto &yourVertex = yourVertices[i];
        CASET_LOG(INFO_LEVEL, "-----------------------------------------------------------------------");
        // Move the in-edges from the vertices on yourFace to the corresponding vertex on myFace, but only if those
        // Edges aren't part of `yourFace`, since yourFace is going away completely. The edge does not need to be moved
        // from myFace, but deleted entirely.
        moveEdges(yourVertex, yourFace, myVertex, myFace, true);
        moveEdges(yourVertex, yourFace, myVertex, myFace, false);
        removeIfIsolated(yourVertex);
      }
      auto newCoface = *(yourFace->getCofaces().begin());
      myFace->addCoface(newCoface);
      return {myFace, true};
    }

    ///
    /// @return Simplexes around the boundary of the simplicial complex to which they belong. These simplexes have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplexes as you might expect, but 5-simplexes since those are the standard
    /// building blocks. You can get the 2-simplexes by calling `getFacets()` on the 5-simplexes and their facets until
    /// \f$ k=2 \f$.
    [[nodiscard]]
    std::vector<std::shared_ptr<Simplex> > getExternalSimplexes() noexcept {
      std::vector<std::shared_ptr<Simplex> > simplexes;
      for (const auto &[key, bucket] : externalSimplexes) {
        for (const auto &simplex : bucket) {
          simplexes.push_back(simplex);
        }
      }
      return simplexes;
    }

    void embedEuclidean(int dimensions, double epsilon);

  private:
    std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();
    std::unordered_map<SimplexOrientation, Bucket> externalSimplexes{};
    std::unordered_map<SimplexOrientation, Bucket> internalSimplexes{};

    std::vector<std::shared_ptr<Observable> > observables{};

    std::shared_ptr<Metric> metric;
    SpacetimeType spacetimeType;
    std::shared_ptr<Topology> topology;
    double alpha = 1.;
    std::uint64_t currentTime = 0;
    bool manual = false;
};
} // caset

#endif //CASET_SPACETIME_H
