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
#include "Face.h"
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
      std::vector<std::shared_ptr<Vertex> > &vertices,
      std::vector<std::shared_ptr<Edge> > &edges
    ) noexcept {
      manual = true;
      const SimplexOrientation orientation = SimplexOrientation::orientationOf(vertices);
      auto &bucket = simplicialComplex.try_emplace(orientation /*key*/).first->second;
      std::vector<IdType> _ids = {};
      _ids.reserve(vertices.size());
      for (const auto &vertex : vertices) {
        _ids.push_back(vertex->getId());
      }
      auto [fingerprint, n, ids] = Fingerprint::computeFingerprint(_ids);
      if (!bucket.contains(fingerprint)) {
        std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices, edges);
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
      std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices, edges);
      simplicialComplex[simplex->getOrientation()].insert(simplex);
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
      CASET_LOG(INFO_LEVEL, "creating a ", std::to_string(k), "-simplex from a numeric orientation (", std::to_string(std::get<0>(numericOrientation)), ", ", std::to_string(std::get<1>(numericOrientation)), ")");
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
          std::shared_ptr<Edge> edge = edgeList->add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          CASET_LOG(INFO_LEVEL, "Creating a new timelike edge from ", existingVertex->getId(), "->", newVertex->getId());
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
            CASET_LOG(INFO_LEVEL, "Creating a new spacelike edge (L^2 = \\alpha) from ", existingVertex->getId(), "->", newVertex->getId());
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
          } else {
            CASET_LOG(INFO_LEVEL, "Creating a new timelike edge (L^2 = -\\alpha) from ", existingVertex->getId(), "->", newVertex->getId());
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          }
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices, edges, orientation);
      simplicialComplex[orientation].insert(simplex);
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

    void replaceVertex(const std::shared_ptr<Vertex> &toRemove, const std::shared_ptr<Vertex> &toAdd) {
      for (auto &edge : toRemove->getInEdges()) {
        edgeList->remove(edge);
        toRemove->removeInEdge(edge);
        edge->replaceTargetVertex(toAdd->getId());
        toAdd->addInEdge(edgeList->add(edge));
      }
      for (auto &edge : toRemove->getOutEdges()) {
        edgeList->remove(edge);
        toRemove->removeOutEdge(edge);
        edge->replaceSourceVertex(toAdd->getId());
        toAdd->addOutEdge(edgeList->add(edge));
      }
      vertexList->replace(toRemove, toAdd);
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
    /// To avoid these problems this method takes the following steps:
    ///
    ///   1. First it checks to ensure `myFace` and `yourFace` are not already attached to another Simplex Face,
    ///   1. then it creates a new set of Vertex (es) to match the dimensionality of `myFace` and `yourFace`.
    ///   1. Next, it maps Vertex (es) \f$ \mathcal{V}_{myFace} \f$ and \f$ \mathcal{V}_{yourFace} \f$ of `myFace`
    ///     and `yourFace` 1:1 respectively (2:1 in all) to the new vertices based on their TimeOrientation
    ///   1. Next, it validates the resulting Edge (es) are of compatible length within an Epsilon value.
    ///   1. Next, it maps Edges (es) \f$ \mathcal{E}_{myFace} \f$ and \f$ \mathcal{E}_{yourFace} \f$ of `myFace` and
    ///     `yourFace` 1:1 respectively (2:1 in all) to the new Edges based on their shared Vertex (es).
    ///   1. Next, it removes the replaced Edge (s) and Vertex (es) from the Spacetime (simplicial complex).
    ///
    /// @param myFace The Face of this Simplex to attach to `yourFace` of the other Simplex
    /// @param yourFace The Face of the other Simplex to attach to `myFace` of this Simplex.
    /// @return myFace, updated with the second simplex glued.
    std::tuple<std::shared_ptr<Face>, bool> causallyAttachFaces(std::shared_ptr<Face> &myFace,
                                                                const std::shared_ptr<Face> &yourFace) {
      if (!myFace->isAvailable()) {
        CASET_LOG(ERROR_LEVEL, "You're attempting to attach a Face that has two or more co-Faces!");
        CASET_LOG(ERROR_LEVEL, "The cofaces are: ");
        for (const auto &coface : myFace->getCofaces()) {
          CASET_LOG(ERROR_LEVEL, coface->toString());
        }
        throw std::runtime_error("(myFace) You attempted to attach faces that are not available to attach.");
      }
      if (!yourFace->isAvailable()) {
        CASET_LOG(ERROR_LEVEL, "You're attempting to attach a Face that has two or more co-Faces!");
        CASET_LOG(ERROR_LEVEL, "The cofaces are: ");
        for (const auto &coface : yourFace->getCofaces()) {
          CASET_LOG(ERROR_LEVEL, coface->toString());
        }
        throw std::runtime_error("(yourFace) You attempted to attach faces that are not available to attach.");
      }
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(myFace->size());
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(myFace->size());

      // Two vertexes are compatible to attach iff they share the same time value.
      std::vector<std::pair<std::shared_ptr<Vertex>, std::shared_ptr<Vertex> > > vertexPairs = {};
      vertexPairs.reserve(myFace->size());

      // if (myFace->checkPairty(yourFace) != -1) {
      // CASET_LOG(WARN_LEVEL, "Pairty check failed. myFace and yourFace do not have opposite pairty.");
      // return {nullptr, false};
      // }

      // These are in order of traversal, you can iterate them to walk the Face:
      const auto &myVertices = myFace->getVertices();
      const auto &yourVertices = yourFace->getVertices();

      // myVertices and yourVertices should have a sequence that lines up, but they're not necessarily at the correct
      // starting node. We should shuffle through until they are either compatible or we've tried all possible orders.

      std::deque<int> myVertexIdxs{};
      for (auto i = 0; i < myVertices.size(); i++) {
        myVertexIdxs.push_back(i);
      }

      int attempts = 0;
      while (attempts < myVertexIdxs.size()) {
        for (auto i : myVertexIdxs) {
          int j = i - attempts;

          const auto &myVertex = myVertices[i];
          const auto &yourVertex = yourVertices[j];

          if (myVertex->getTime() != yourVertex->getTime()) {
            // The two vertices were not in the expected causal disposition.
            CASET_LOG(WARN_LEVEL,
                "Vertex ",
                myVertex->toString(),
                " and ",
                yourVertex->toString(),
                " do not have the same causal disposition! ",
                myVertex->toString(),
                " has t=",
                myVertex->getTime(),
                " ",
                yourVertex->toString(),
                " has t=",
                yourVertex->getTime());
            int front = myVertexIdxs.front();
            myVertexIdxs.pop_front();
            myVertexIdxs.push_back(front);
            attempts++;
            continue;
          }
          /// Create a new vertex, v3 with all the edges of v1 and v2 combined, but exclude those edges of v2 that lie on
          /// `yourFace` since they are being replaced by those corresponding edges on `myFace`.
          for (const auto &edge : yourVertex->getInEdges()) {
            // Move the in-edges from the vertices on yourFace to the corresponding vertex on myFace, but only if those
            // Edges aren't part of `yourFace`.
            if (yourFace->hasEdge(
              edge->getSourceId(),
              edge->getTargetId()
              )) continue;
            yourVertex->removeInEdge(edge);
            edgeList->remove(edge);
            edge->replaceTargetVertex(myVertex->getId());
            myVertex->addInEdge(edgeList->add(edge));
          }
          for (const auto &edge : yourVertex->getOutEdges()) {
            if (yourFace->hasEdge(edge->getSourceId(), edge->getTargetId())) continue;
            yourVertex->removeOutEdge(edge);
            edgeList->remove(edge);
            edge->replaceSourceVertex(myVertex->getId());
            myVertex->addOutEdge(edgeList->add(edge));
          }
          if (yourVertex->degree() == 0) vertexList->replace(yourVertex, myVertex);
        }
        return {myFace, true};
      }
      return {nullptr, false};
    }

    std::vector<std::shared_ptr<Simplex> > getSimplexes() noexcept {
      std::vector<std::shared_ptr<Simplex> > simplexes;
      for (const auto &[key, bucket] : simplicialComplex) {
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
    std::unordered_map<SimplexOrientation, Bucket> simplicialComplex{};

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
