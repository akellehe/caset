//
// Created by Andrew Kelleher on 12/30/25.
//

#ifndef CASET_PLOTTER_H
#define CASET_PLOTTER_H
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "ForwardDeclarations.h"
#include "Edge.h"

namespace py = pybind11;

namespace caset {

template<int dimensions>
class Plotter {
  public:

    /// @param pyplot Axes object to which collections and such should be added.
    explicit Plotter(py::object &ax);

    /// Use this method to define what vertices are being plotted. This class will plot each vertex with a label. You can
    /// totally print orphaned vertices, so don't worry about it if there's no corresponding Edge.
    ///
    /// This method does not assume the vertices you're adding have already been embedded in a coordinate space, but if
    /// you don't embed it by the time you call ::plot() then they'll all appear on the same point in the best case, and
    /// you'll get some kind of undefined error in the worst case. You can check examplex/plot4D.py if you want to see how
    /// to embed vertices in a Euclidean space.
    ///
    /// @param vertices The vertices you would like to plot, independently of Edges. They are appended to the existing
    ///   vertices.
    void addVertices(const VertexPtrs &newVertices);

    /// You can use this method to set what Edges to plot. Edges and vertices are plotted independently, but the
    /// coordinates of the Edge are derived from the coordinates of it's source/target Vertex. If those don't have
    /// coordinates set then you're going to have a bad time.
    ///
    /// Timelike and Spacelike edges are inferred by the sign on their squared edgelength. Timelike edges will be plotted
    /// as a red line segment connecting it's source and target vertices. Spacelike edges will be blue. Because Timelike
    /// edges have negative squared edge length, you will normally want to embed them with the absolute value of their
    /// edge length to avoid weirdness.
    ///
    /// Note that edges are plotted ONLY as line segments. If you want to print a Vertex at the end you need to add the
    /// vertices of the edge via setVertices().
    ///
    /// @param The edges you would like to plot. They are appended to the existing edges.
    void addEdges(const Edges &newEdges);

    /// This method does the actual plotting using mpl_toolkits Line3DCollection. Note that mpl_toolkits is a dependency
    /// for this method. Specifically we need mpl_toolkits.mplot3d.art3d.Line3DCollection.
    void addCollections();

    /// Allows you to add a Simplex to be plotted in the context of this spacetime/plot. The benefit here is that the
    /// simplex will be plotted in what ever color you pass as `color`. Note that if you add edges or vertices that
    /// belong to `simplex` after calling this method; you'll effectively double-paint the simplex edges/vertices.
    ///
    /// Simplices are painted with a slightly thicker line width than the default to make them more apparent. Also note
    /// that if you add multiple simplices that share edges/vertices those edges/vertices will take the color of the
    /// last simplex you added.
    ///
    /// @param simplex A simplex to paint a particular color.
    void addSimplex(const SimplexPtr &simplex, std::string color);

    [[nodiscard]] std::tuple<double, double, double> to3D(const VertexPtr &vertex) const;
  private:
    /// Go through the vertices and add an x, y, z, t label to them.
    void labelVertices() const;

    /// Labels the edges like labelVertices
    void labelEdges() const;

    /// Goes through all vertices and sets the boundaries of the plot to match the outer bounds for every dimension.
    void setBounds() const;

    pybind11::object ax{};
    VertexPtrSet vertices{};
    EdgePtrSet timelikeEdges{};
    EdgePtrSet spacelikeEdges{};
    VertexPtrSet simplexVertices{};
    EdgePtrSet simplexEdges{};
    std::unordered_map<SimplexPtr, std::string, SimplexPtrHash, SimplexPtrEq> simplexColors{};
};
} // caset

#endif //CASET_PLOTTER_H