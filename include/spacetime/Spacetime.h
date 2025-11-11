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

    static void addVertex(std::shared_ptr<Vertex> vertex) noexcept {
      vertexList->add(vertex);
    }

    static void addEdge(std::shared_ptr<Edge> edge) noexcept {
      edgeList->add(edge);
    }

    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept {
      return spacetimeType;
    }

    static std::shared_ptr<EdgeList> getEdgeList() {
      return edgeList;
    }
    static std::shared_ptr<VertexList> getVertexList() {
      return vertexList;
    }

    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept { return metric; }

  private:
    std::shared_ptr<Metric> metric;
    static inline std::shared_ptr<EdgeList> edgeList;
    static inline std::shared_ptr<VertexList> vertexList;
    static inline std::unordered_map<SimplexShape, std::unordered_set<Simplex>> simplicialComplex;
    SpacetimeType spacetimeType;
    std::vector<std::shared_ptr<Observable> > observables;
    std::shared_ptr<Topology> topology;
    double alpha;
};
} // caset

#endif //CASET_SPACETIME_H
