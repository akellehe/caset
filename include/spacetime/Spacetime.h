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

namespace caset {
enum class SpacetimeType : uint8_t {
  CDT = 0,
  REGGE = 1,
  COSET = 2,
  REGGE_PACHNER = 3,
  GFT_SPIN_FOAM = 4,
  RICCI_FLOW_DISCRETIZATION = 5
};

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
    /// matching the chosen topology.
    void build() {
      // return topology->build(this);
    }

    void addObservable(std::shared_ptr<Observable> &observable) {
      observables.push_back(observable);
    }

    static std::shared_ptr<Vertex> createVertex(const std::uint64_t id) noexcept {
      return vertexList->add(id);
    }

    static std::shared_ptr<Vertex> createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      return vertexList->add(id, coords);
    }

    static std::shared_ptr<Edge> createEdge(
      std::shared_ptr<Vertex> &src,
      std::shared_ptr<Vertex> &tgt
      ) noexcept {
      std::shared_ptr<Edge> edge = edgeList->add(src, tgt);
      src->addOutEdge(edge);
      tgt->addInEdge(edge);
      return edge;
    }

    static std::shared_ptr<Edge> createEdge(
      std::shared_ptr<Vertex> &src,
      std::shared_ptr<Vertex> &tgt,
      double squaredLength
      ) noexcept {
      std::shared_ptr<Edge> edge = edgeList->add(src, tgt, squaredLength);
      src->addOutEdge(edge);
      tgt->addInEdge(edge);
      return edge;
    }

    static std::shared_ptr<Simplex> createSimplex(std::vector<std::shared_ptr<Vertex> > &vertices) noexcept {
      const SimplexShape shape = SimplexShape::shapeOf(vertices);
      auto &bucket = simplicialComplex.try_emplace(shape /*key*/).first->second; // creates empty set if missing
      auto [fingerprint, n, ids] = Simplex::computeFingerprint(vertices);
      if (!bucket.contains(fingerprint)) {
        std::shared_ptr<Simplex> simplex = std::make_shared<Simplex>(vertices);
        bucket.insert(simplex);
        return simplex;
      }
      return *bucket.find(fingerprint);
    }

    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept {
      return spacetimeType;
    }

    [[nodiscard]] static std::shared_ptr<EdgeList> getEdgeList() noexcept {
      return edgeList;
    }
    [[nodiscard]] static std::shared_ptr<VertexList> getVertexList() noexcept {
      return vertexList;
    }

    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept { return metric; }

  private:
    static inline std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    static inline std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();
    static inline std::unordered_map<SimplexShape, Bucket> simplicialComplex{};

    std::vector<std::shared_ptr<Observable> > observables{};

    std::shared_ptr<Metric> metric;
    SpacetimeType spacetimeType;
    std::shared_ptr<Topology> topology;
    double alpha = 1.;
};
} // caset

#endif //CASET_SPACETIME_H
