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

    void addObservable(std::shared_ptr<Observable> &observable) {
      observables.push_back(observable);
    }

    std::shared_ptr<Vertex> createVertex(const std::uint64_t id) noexcept {
      manual = true;
      return vertexList.add(id);
    }

    std::shared_ptr<Vertex> createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      manual = true;
      return vertexList.add(id, coords);
    }

    std::shared_ptr<Edge> createEdge(
      const std::uint64_t src,
      const std::uint64_t tgt
    ) {
      manual = true;
      std::shared_ptr<Edge> edge = edgeList.add(src, tgt);
      vertexList[src]->addOutEdge(edge);
      vertexList[tgt]->addInEdge(edge);
      return edge;
    }

    std::shared_ptr<Edge> createEdge(
      const std::uint64_t src,
      const std::uint64_t tgt,
      double squaredLength
    ) noexcept {
      manual = true;
      std::shared_ptr<Edge> edge = edgeList.add(src, tgt, squaredLength);
      vertexList[src]->addOutEdge(edge);
      vertexList[tgt]->addInEdge(edge);
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
        throw new std::runtime_error(
          "You can't mix user-defined vertex/edge/simplex definitions with internal definitions. This happens when you call createSimplex(k) after you've called createVertex/createEdge/createSimplex(vertices)");
      }
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      std::cout << "A " << k << "-simplex" << " has " << Simplex::computeNumberOfEdges(k) << " edges" << std::endl;
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < k; i++) {
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        std::shared_ptr<Vertex> newVertex = vertexList.add(vertexList.size());
        for (auto existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList.add(existingVertex->getId(), newVertex->getId());
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
        throw new std::runtime_error(
          "You can't mix user-defined vertex/edge/simplex definitions with internal definitions. This happens when you call createSimplex(k) after you've called createVertex/createEdge/createSimplex(vertices)");
      }
      SimplexOrientation orientation(std::get<0>(numericOrientation), std::get<1>(numericOrientation));
      std::uint8_t k = orientation.getK();
      auto [ti, tf] = orientation.numeric();
      std::vector<std::shared_ptr<Vertex> > vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      std::cout << "A " << k << "-simplex" << " has " << Simplex::computeNumberOfEdges(k) << " edges" << std::endl;
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < ti; i++) {  // Create ti Timelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        std::shared_ptr<Vertex> newVertex = vertexList.add(vertexList.size(), {static_cast<double>(currentTime)});
        for (auto existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList.add(existingVertex->getId(), newVertex->getId());
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      incrementTime();
      for (int i = 0; i < tf; i++) {  // Create ti Timelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        std::shared_ptr<Vertex> newVertex = vertexList.add(vertexList.size(), {static_cast<double>(currentTime)});
        for (auto existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList.add(existingVertex->getId(), newVertex->getId());
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

    [[nodiscard]] EdgeList getEdgeList() noexcept {
      return edgeList;
    }
    [[nodiscard]] VertexList getVertexList() noexcept {
      return vertexList;
    }

    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept { return metric; }

    [[nodiscard]] double getCurrentTime() const noexcept {
      return currentTime;
    }

    double incrementTime() noexcept {
      currentTime++;
      return currentTime;
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
    /// @return A new Face representing the shared face of both simplexes.
    std::shared_ptr<Face> attachFaces(std::shared_ptr<Face> &myFace, std::shared_ptr<Face> &yourFace) {
      if (!myFace->isAvailable() || !yourFace->isAvailable()) {
        throw new std::runtime_error("You attempted to attach faces that are not available to attach.");
      }
      std::vector<std::shared_ptr<Vertex>> vertices = {};
      vertices.reserve(myFace->size());
      std::vector<std::shared_ptr<Edge>> edges = {};
      edges.reserve(myFace->size());

      // Two vertexes are compatible to attach iff they share the same time value.
      std::vector<std::pair<std::shared_ptr<Vertex>, std::shared_ptr<Vertex>>> vertexPairs= {};
      vertexPairs.reserve(myFace->size());

      if (myFace->checkPairty(yourFace) != -1) {

      }
      auto myVertices = myFace->getVertices();
      auto yourVertices = yourFace->getVertices();
      for (auto v1 = myVertices.begin(); v1 != myVertices.end(); ++v1) {
        for (auto v2 = yourVertices.begin(); v2 != yourVertices.end(); ++v2) {
          if ((*v1)->getTime() == (*v2)->getTime()) {  // Compatible if the neighbors are.
            // We should probably be iterating over edges instead of vertices, and assign a convention to the
            // orientation of an edge. A simplex/chain/wedge product has an orientation defined by the order of
            // traversal of its nodes. Probably the least abrasive way to do this would just be to encode those rules
            // and then describe the orientation as either "out" or "in" and "up" or "down" where we call time
            // increasing the "up" direction and decreasing "down" and clockwise "in" and counterclockwise "out" or
            // whatever happens to be the prevailing convention in the literature.

            // I think for two faces to join they have to be "out" and "in" respectively, and both have to have the same
            // causal orientation. What about e.g. entirely timelike or entirely spacelike faces? The preceding idea
            // applies to spacelike faces, for timelike faces i think the attachment might be arbitrary. We can get the
            // ordering of the vertices on the face by traversing their edges.

          }
        }
      }
    }

    std::vector<std::shared_ptr<Simplex>> getSimplexes() noexcept {
      std::vector<std::shared_ptr<Simplex>> simplexes;
      for (const auto &[key, bucket] : simplicialComplex) {
        for (const auto &simplex : bucket) {
          simplexes.push_back(simplex);
        }
      }
      return simplexes;
    }

  private:
    EdgeList edgeList = EdgeList{};
    VertexList vertexList = VertexList{};
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
