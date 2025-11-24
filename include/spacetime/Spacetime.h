//
// Created by andrew on 10/23/25.
/**/ //

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

    ///
    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    void build() {
      // TODO: Implement topologies instead of the default.
      // return topology->build(this);
      std::vector<std::tuple<uint8_t, uint8_t> > orientations = {{1, 4}, {2, 3}};
      createSimplex(orientations[1]);
      for (int i = 0; i < 100000; i++) {
        SimplexPtr rightSimplex = createSimplex(orientations[i % 2]);
        OptionalSimplexPair leftFaceRightFace = chooseSimplexFacesToGlue(rightSimplex);
        if (!leftFaceRightFace.has_value()) {
          return;
        }
        auto [leftFace, rightFace] = leftFaceRightFace.value();
        auto [left, succeeded] = causallyAttachFaces(leftFace, rightFace);
      }
    }

    void addObservable(const std::shared_ptr<Observable> &observable) {
      observables.push_back(observable);
    }

    VertexPtr createVertex(const std::uint64_t id) noexcept {
      return vertexList->add(id);
    }

    VertexPtr createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      return vertexList->add(id, coords);
    }

    std::shared_ptr<Edge> createEdge(
      const std::uint64_t src,
      const std::uint64_t tgt
    ) {
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
      std::shared_ptr<Edge> edge = edgeList->add(src, tgt, squaredLength);
      vertexList->get(src)->addOutEdge(edge);
      vertexList->get(tgt)->addInEdge(edge);
      return edge;
    }

    SimplexPtr createSimplex(
      Vertices &vertices
    ) noexcept {
      const SimplexOrientationPtr orientation = SimplexOrientation::orientationOf(vertices);
      auto &bucket = externalSimplices.try_emplace(orientation /*key*/).first->second;
      std::vector<IdType> _ids = {};
      _ids.reserve(vertices.size());
      for (const auto &vertex : vertices) {
        _ids.push_back(vertex->getId());
      }

      auto [fingerprint, n, ids] = Fingerprint::computeFingerprint(_ids);
      if (!bucket.contains(fingerprint)) {
        SimplexPtr simplex = std::make_shared<Simplex>(vertices);
        bucket.insert(simplex);
        for (const auto &o : simplex->getOrientation()->getFacialOrientations()) {
          externalSimplices[o].insert(simplex);
        }
        return simplex;
      }
      return *bucket.find(fingerprint);
    }

    SimplexPtr createSimplex(std::size_t k) {
      double squaredLength = alpha;
      Vertices vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < k; i++) {
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      return createSimplex(vertices);
    }

    SimplexPtr createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
      double squaredLength = alpha;
      double timelikeSquaredLength = alpha;
      SimplexOrientationPtr orientation = std::make_shared<SimplexOrientation>(
        std::get<0>(numericOrientation),
        std::get<1>(numericOrientation));
      std::uint8_t k = orientation->getK();
      auto [ti, tf] = orientation->numeric();
      Vertices vertices = {};
      vertices.reserve(k);
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(Simplex::computeNumberOfEdges(k));
      for (int i = 0; i < ti; i++) {
        // Create ti Timelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
        if (getMetric()->getSignature()->getSignatureType() == SignatureType::Lorentzian) {
          timelikeSquaredLength = -alpha;
        }
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge = edgeList->
              add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      for (int i = 0; i < tf; i++) {
        // Create ti Spacelike vertices
        // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
        /// We can't just use the vertexList .size() here, because some vertices can be removed. We need to keep a
        /// counter:
        VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime + 1)});
        for (const auto &existingVertex : vertices) {
          std::shared_ptr<Edge> edge;
          if (existingVertex->getTime() < newVertex->getTime()) {
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
          } else {
            edge = edgeList->add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
          }
          existingVertex->addOutEdge(edge);
          newVertex->addInEdge(edge);
          edges.push_back(edge);
        }
        vertices.push_back(newVertex);
      }
      return createSimplex(vertices);
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
    getGluableFaces(const SimplexPtr &unattachedSimplex, const SimplexPtr &attachedSimplex) {
      auto unattachedFacets = unattachedSimplex->getFacets(); // vector<shared_ptr<Simplex>>
      auto attachedFacets = attachedSimplex->getFacets();

      for (auto &unattachedFace : unattachedFacets) {
        const auto [tia, tfa] = unattachedFace->getOrientation()->numeric();
        for (auto &attachedFace : attachedFacets) {
          const auto [tib, tfb] = attachedFace->getOrientation()->numeric();
          if (tia == tfb && tfa == tib) {
            return std::make_optional(std::make_pair(unattachedFace, attachedFace));
          }
          if (tia == tib && tfa == tfb) {
            return std::make_optional(std::make_pair(unattachedFace, attachedFace));
          }
        }
      }

      return std::nullopt;
    }

    void moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
      for (const auto &edge : from->getInEdges()) {
        const VertexPtr originalSource = vertexList->get(edge->getSourceId());
        originalSource->removeOutEdge(edge);
        from->removeInEdge(edge);
        edgeList->remove(edge);
        edge->replaceTargetVertex(to->getId());
        const EdgePtr newEdge = edgeList->add(edge);
        to->addInEdge(newEdge);
        originalSource->addOutEdge(newEdge);
      }
    }

    void moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
      for (const auto &edge : from->getOutEdges()) {
        const VertexPtr originalTarget = vertexList->get(edge->getTargetId());
        originalTarget->removeInEdge(edge);
        from->removeOutEdge(edge);
        edgeList->remove(edge);
        edge->replaceSourceVertex(to->getId());
        const EdgePtr newEdge = edgeList->add(edge);
        to->addOutEdge(newEdge);
        originalTarget->addInEdge(newEdge);
      }
    }

    void removeIfIsolated(const VertexPtr &vertex) {
      if (vertex->degree() == 0) {
        vertexList->remove(vertex);
      }
    }

    void moveEdges(
      const VertexPtr &fromVertex,
      const SimplexPtr &fromSimplex,
      const VertexPtr &toVertex,
      bool moveInEdges
    ) {
      /// TODO: This method is not discriminating between in and out edgtes during the actual move.
      CASET_LOG(DEBUG_LEVEL,
                "Moving ",
                std::to_string(fromVertex->getInEdges().size()),
                " in-edges and ",
                std::to_string(fromVertex->getOutEdges().size()),
                " out-edges from vertex ",
                fromVertex->toString(),
                " to vertex ",
                toVertex->toString());
      const auto edges = moveInEdges ? fromVertex->getInEdges() : fromVertex->getOutEdges();
      for (const auto &edge : edges) {
        if (fromSimplex->hasEdge(edge->getSourceId(), edge->getTargetId())|| fromSimplex->hasEdge(edge->getTargetId(), edge->getSourceId())) {
          const VertexPtr &sourceVertex = vertexList->get(edge->getSourceId());
          const VertexPtr &targetVertex = vertexList->get(edge->getTargetId());
          sourceVertex->removeOutEdge(edge);
          targetVertex->removeInEdge(edge);
          edgeList->remove(edge);
          removeIfIsolated(sourceVertex);
          removeIfIsolated(targetVertex);
          continue;
        }

        if (moveInEdges) {
          moveInEdgesFromVertex(fromVertex, toVertex);
        } else {
          moveOutEdgesFromVertex(fromVertex, toVertex);
        }
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
    std::tuple<SimplexPtr, bool> causallyAttachFaces(const SimplexPtr &myFace,
                                                     const SimplexPtr &yourFace) {
#if defined(CASET_DEBUG)
      if (myFace->fingerprint.fingerprint() == yourFace->fingerprint.fingerprint()) {
        CASET_LOG(ERROR_LEVEL, "Faces are already attached!");
        return {myFace, false};
      }
      if (myFace->getOrientation() != yourFace->getOrientation()) {
        CASET_LOG(ERROR_LEVEL,
                  "Faces have different orientations: ",
                  myFace->getOrientation()->toString(),
                  " vs ",
                  yourFace->getOrientation()->toString());
        return {myFace, false};
      }
      for (const auto &myCoface : myFace->getCofaces()) {
        for (const auto &yourCoFace : yourFace->getCofaces()) {
          if (myCoface->fingerprint.fingerprint() == yourCoFace->fingerprint.fingerprint()) {
            CASET_LOG(ERROR_LEVEL, "Faces share a coface!");
            return {myFace, false};
          }
        }
      }
#endif
      Vertices vertices = {};
      vertices.reserve(myFace->size());
      std::vector<std::shared_ptr<Edge> > edges = {};
      edges.reserve(myFace->size());

      // Two vertices are compatible to attach iff they share the same time value.
      std::vector<std::pair<VertexPtr, VertexPtr> > vertexPairs{};
      vertexPairs.reserve(myFace->size());

      // These are in order of traversal, you can iterate them to walk the Face:
      const auto &yourVertices = yourFace->getVertices();

      // myVertices and yourVertices should have a sequence that lines up, but they're not necessarily at the correct
      // starting node. We should shuffle through until they are either compatible or we've tried all possible orders.
      const std::optional<Vertices> myOrderedVerticesOptional = myFace->
          getVerticesWithParityTo(yourFace);

      if (!myOrderedVerticesOptional.has_value()) {
        CASET_LOG(WARN_LEVEL, "No compatible vertex order found for myFace and yourFace.");
        return {nullptr, false};
      }

      const Vertices &myOrderedVertices = myOrderedVerticesOptional.value();
      for (auto i = 0; i < myOrderedVertices.size(); i++) {
        const auto &myVertex = myOrderedVertices[i];
        const auto &yourVertex = yourVertices[i];
        CASET_LOG(DEBUG_LEVEL, "-----------------------------------------------------------------------");

        // Move the in-edges from the vertices on yourFace to the corresponding vertex on myFace, but only if those
        // Edges aren't part of `yourFace`, since yourFace is going away completely. The edge does not need to be moved
        // from myFace, but deleted entirely.
        if (yourVertex->getInEdges().size() > 0) moveEdges(yourVertex, yourFace, myVertex, true);
        if (yourVertex->getOutEdges().size() > 0) moveEdges(yourVertex, yourFace, myVertex, false);

        yourFace->replaceVertex(yourVertex, myVertex);

        // TODO: Move stuff from cofaces. Don't know that this covers all cases.
        for (const auto &coface : yourFace->getCofaces()) {
          coface->replaceVertex(yourVertex, myVertex);
        }
        removeIfIsolated(yourVertex);
      }

      auto newCoface = *(yourFace->getCofaces().begin());
      myFace->addCoface(newCoface);
      if (!yourFace->isCausallyAvailable()) {
        for (const auto &facialOrientation : yourFace->getOrientation()->getFacialOrientations()) {
          externalSimplices[facialOrientation].erase(yourFace);
        }
        internalSimplices[yourFace->getOrientation()].insert(yourFace);
      }
      return {myFace, true};
    }

    ///
    /// @return Simplices around the boundary of the simplicial complex to which they belong. These simplices have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplices as you might expect, but 5-simplices since those are the standard
    /// building blocks. You can get the 2-simplices by calling `getFacets()` on the 5-simplices and their facets until
    /// \f$ k=2 \f$.
    [[nodiscard]]
    SimplexSet getExternalSimplices() noexcept {
      SimplexSet simplices;
      for (const auto &[facialOrientation, bucket] : externalSimplices) {
        for (const auto &simplex : bucket) {
          simplices.insert(simplex);
        }
      }
      return simplices;
    }

    void embedEuclidean(int dimensions, double epsilon);

    /// This method chooses a simplex from the boundary of the simplicial complex to which `mySimplex` can be glued. It
    /// does this by iterating through the `externalSimplices` and checking for compatible orientations and edge lengths.
    ///
    /// To the extent the hashing function for vertex fingerprinting is good; this should be pretty well pseudo-random.
    /// If you want something truly random, though, you should probably implement that.
    ///
    /// @returns A pair of \f$ k-1 \f$ simplices (faces) if a compatible k-simplex was found. None otherwise.
    OptionalSimplexPair chooseSimplexFacesToGlue(const SimplexPtr &unattachedSimplex) {
      CASET_LOG(DEBUG_LEVEL, "------------------------------------------------------------------------");
      for (const auto &facialOrientation : unattachedSimplex->getGluableFaceOrientations()) {
        const auto &bucket = externalSimplices[facialOrientation];
        if (bucket.empty()) continue;
        CASET_LOG(DEBUG_LEVEL,
                  "Found ",
                  std::to_string(bucket.size()),
                  " simplices in bucket for facial orientation ",
                  facialOrientation->toString());
        for (auto attachedSimplexIt = bucket.begin(); attachedSimplexIt != bucket.end(); ++attachedSimplexIt) {
          if ((*attachedSimplexIt)->fingerprint.fingerprint() == unattachedSimplex->fingerprint.fingerprint()) {
            CASET_LOG(DEBUG_LEVEL, "Skipping simplex ", (*attachedSimplexIt)->toString(), " because it's the same as mySimplex");
            continue;
          }
          OptionalSimplexPair gluablePair = getGluableFaces(unattachedSimplex, *attachedSimplexIt);
          if (gluablePair.has_value()) {
            CASET_LOG(INFO_LEVEL,
                      "Found gluable faces for simplex \n",
                      (*attachedSimplexIt)->toString(),
                      " \nand mySimplex \n",
                      unattachedSimplex->toString(), "\n---\n", std::get<0>(gluablePair.value())->toString(), "\nAND\n", std::get<1>(gluablePair.value())->toString());
            return gluablePair;
          }
          CASET_LOG(DEBUG_LEVEL,
                    "No gluable faces found to attach simplex ",
                    (*attachedSimplexIt)->toString(),
                    " to ",
                    unattachedSimplex->toString());
        }
      }
      return std::nullopt;
    }

    SimplexSet getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
      SimplexOrientationPtr o = std::make_shared<
        SimplexOrientation>(std::get<0>(orientation), std::get<1>(orientation));
      if (externalSimplices.contains(o)) {
        return externalSimplices[o];
      }
      if (internalSimplices.contains(o)) {
        return internalSimplices[o];
      }
      return {};
    }

  private:
    std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();

    ///
    /// These are simplices on the boundary of a simplicial complex. They have at least one external face, and hence can
    /// be glued to other simplices. The externalSimplices are organized by the orientation of their available faces. If
    /// a face is available; the orientation of that face can be found as a key corresponding to a SimplexSet containing
    /// the Simplex to which that Face belongs.
    ///
    /// This makes for fast lookups when gluing simplices together to form a complex.
    std::unordered_map<SimplexOrientationPtr, SimplexSet, SimplexOrientationHash, SimplexOrientationEq>
    externalSimplices{};

    ///
    /// These are simplices that are fully internal to the simplicial complex. They have no external faces, and hence
    /// cannot be glued to other simplices.
    ///
    /// A Simplex becomes _internal_ when all it's _external_ faces have been glued. At that point it is no longer
    /// relevant to store that simplex by the orientation of any given face, so _internal_ simplices are stored by the
    /// orientation of the Simplex itself.
    ///
    /// TODO: I don't think SimplexOrientationPtr is correctly hashing!!
    std::unordered_map<SimplexOrientationPtr, SimplexSet, SimplexOrientationHash, SimplexOrientationEq>
    internalSimplices{};

    std::vector<std::shared_ptr<Observable> > observables{};

    std::shared_ptr<Metric> metric;
    SpacetimeType spacetimeType;
    std::shared_ptr<Topology> topology;
    double alpha = 1.;
    std::uint64_t currentTime = 0;
    IdType vertexIdCounter = 0;
};
} // caset

#endif //CASET_SPACETIME_H
