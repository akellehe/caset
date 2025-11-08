//
// Created by andrew on 10/23/25.
//

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>

#include "EdgeList.h"
#include "VertexList.h"
#include "Metric.h"

enum class TimeOrientation : uint8_t{
  UP = 0,
  DOWN = 1,
  UNKNOWN = 2
};

namespace caset {
class Spacetime {
 public:
    explicit Spacetime(std::shared_ptr<Metric> metric_) : metric(metric_) {}

    static void addVertex(std::shared_ptr<Vertex> vertex) noexcept {
      vertexList->add(vertex);
    }

    static void addEdge(std::shared_ptr<Edge> edge) noexcept {
      edgeList->add(edge);
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

};
} // caset

#endif //CASET_SPACETIME_H