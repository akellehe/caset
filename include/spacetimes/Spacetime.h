//
// Created by andrew on 10/23/25.
//

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>

#include "../observables/Observable.h"
#include "../EdgeList.h"
#include "../VertexList.h"
#include "../Metric.h"
#include "../Simplex.h"

enum class SpacetimeType : uint8_t {
  CDT = 0,
  REGGE = 1,
  COSET = 2,
  REGGE_PACHNER = 3,
  GFT_SPIN_FOAM = 4,
  RICCI_FLOW_DISCRETIZATION = 5
};

namespace caset {
class Spacetime {
 public:
    Spacetime(std::shared_ptr<Metric> metric_, const SpacetimeType spacetimeType_) : metric(metric_), spacetimeType(spacetimeType_) {
      alpha = 1.;
    }
    Spacetime(std::shared_ptr<Metric> metric_, const SpacetimeType spacetimeType_, double alpha_) : metric(metric_), spacetimeType(spacetimeType_), alpha(alpha_) {}

    virtual ~Spacetime() = default;

    ///
    /// Builds an n-dimensional (depending on your metric) triangulation with edge lengths equal to alpha.
    void build() {

    }

    ///
    /// `tune` is the initial stage of building the spacetime. Some example are:
    ///
    /// The stage in CDT where we approach the desired cosmological constant by continuously adjusting it based on the
    /// desired spacetime volume.
    ///
    /// In Regge calculus we build an initial triangulation of the Spacetime.
    ///
    virtual void tune();

    ///
    /// `thermalize` implements some kind of adjustment to the initial lattice.
    ///
    /// In the case of CST (Causal Set Theory) this amounts to executing a poisson "Sprinkling" of Vertexes to preserve
    /// Lorentz invariance.
    ///
    /// For Regge calculus this can be a randomly applied variation in an initially fixed edge length triangulation.
    ///
    virtual void thermalize();

    void addObservable(std::shared_ptr<Observable> &observable) {
      observables.push_back(observable);
    }

    static void addVertex(std::shared_ptr<Vertex> vertex) noexcept {
      vertexList->add(vertex);
    }

    static void addEdge(std::shared_ptr<Edge> edge) noexcept {
      edgeList->add(edge);
    }

    SpacetimeType getSpacetimeType() const noexcept {
      return spacetimeType;
    }

    ///
    /// In the paper "Simulating CDT quantum gravity", J. Brunekreef et. al; they describe simplexes in 2D as having
    /// "time orientations" (1, 2) or (2, 1). With the (2, 1) orientation; the "time edge" is on the bottom so it is a
    /// right-side up triangle. In the (2, 1) case it's on top, so it's an upside-down triangle.
    ///
    /// Moving to one higher dimension; the 2-vertex edge becomes a 3-vertex face. As a side note; let's remember to
    /// avoid the misnomer that everything with 2 vertices is always an edge and everything with 3 vertices is always a
    /// face since technically a face is defined as sub-simplices of k-1 vertices and a hinge is k-2.
    ///
    /// In this case; the 3-vertex time face is on the bottom in the (3, 1) orientation and on the top in the (1, 3)
    /// orientation.
    ///
    /// Extending to the 4D case; the time tetrad is on the bottom in the (4, 1) orientation in top in the (1, 4)
    /// orientation.
    ///
    /// Given all that; let's just refer to the two possible (standard) orientations as `Up` for the time (k-1) simplex
    /// being oriented up and `Down` for the (k-1) time simplex being oriented down.
    ///
    /// @param timeOrientation
    static void addSimplex(TimeOrientation timeOrientation) noexcept {}

    static std::shared_ptr<EdgeList> getEdgeList() {
      return edgeList;
    }
    static std::shared_ptr<VertexList> getVertexList() {
      return vertexList;
    }

    std::shared_ptr<Metric> getMetric() const noexcept { return metric; }
 private:
    std::shared_ptr<Metric> metric;
    static inline std::shared_ptr<EdgeList> edgeList;
    static inline std::shared_ptr<VertexList> vertexList;
    SpacetimeType spacetimeType;
    std::vector<std::shared_ptr<Observable>> observables;
    double alpha;

};
} // caset

#endif //CASET_SPACETIME_H