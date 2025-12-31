#include "Plotter.h"

#include <filesystem>

#include "Simplex.h"
#include "Vertex.h"

namespace py = pybind11;

namespace caset {
template<int dimensions>
Plotter<dimensions>::Plotter(py::object &ax_) : ax(ax_) {
}

template<int dimensions>
void Plotter<dimensions>::addVertices(const VertexPtrs &newVertices) {
  vertices.insert(newVertices.begin(), newVertices.end());
}

template<int dimensions>
void Plotter<dimensions>::addEdges(const Edges &newEdges) {
  for (const auto &edge : newEdges) {
    if (edge->getSquaredLength() < 0) {
      timelikeEdges.insert(edge);
    } else {
      spacelikeEdges.insert(edge);
    }
  }
}

template<int dimensions>
std::tuple<double, double, double> Plotter<dimensions>::to3D(const VertexPtr &vertex) const {
  if (dimensions == 4) {
    double alpha = .7;
    double beta = .7;
    auto norm = std::sqrt(alpha * alpha + beta * beta);
    alpha = alpha / norm;
    beta = beta / norm;
    auto coords = vertex->getCoordinates();
    auto x_p = coords[1];
    auto y_p = coords[2];
    auto z_p = coords[0]; //alpha * coords[0] + beta * coords[3];
    return {x_p, y_p, z_p};
  }
  if (dimensions == 3) {
    return {vertex->getCoordinates()[0], vertex->getCoordinates()[1], vertex->getCoordinates()[2]};
  }
  if (dimensions == 2) {
    return {vertex->getCoordinates()[0], vertex->getCoordinates()[1], 0};
  }
  if (dimensions == 1) {
    return {vertex->getCoordinates()[0], 0, 0};
  }
  CLOG(WARN_LEVEL, "Unknown dimension: ", dimensions, " for vertex: ", vertex->toString());
  return {0, 0, 0};
}

template<int dimensions>
void Plotter<dimensions>::labelVertices() const {
  for (const auto &vertex : vertices) {
    if (simplexVertices.contains(vertex)) continue;
    auto [x, y, z] = to3D(vertex);
    ax.attr("text")(
      x,
      y,
      z,
      std::to_string(vertex->getId()),
      py::arg("color") = "black",
      py::arg("fontsize") = 7,
      py::arg("ha") = "center"
    );
  }
  for (const auto &[simplex, color] : simplexColors) {
    for (const auto &vertex : simplex->getVertices()) {
      auto [x, y, z] = to3D(vertex);
      ax.attr("text")(
        x,
        y,
        z,
        std::to_string(vertex->getId()),
        py::arg("color") = color,
        py::arg("fontsize") = 7,
        py::arg("ha") = "center"
      );
    }
  }
}

template<int dimensions>
void Plotter<dimensions>::labelEdges() const {
  for (const auto &edge : timelikeEdges) {
    auto [sx, sy, sz] = to3D(edge->getSource());
    auto [tx, ty, tz] = to3D(edge->getTarget());
    ax.attr("text")(
      (sx + tx) / 2.,
      (sy + ty) / 2.,
      (sz + tz) / 2.,
      std::format("{:.1f}", edge->getSquaredLength()),
      py::arg("color") = "grey",
      py::arg("fontsize") = 7
    );
  }
  for (const auto &edge : spacelikeEdges) {
    auto [sx, sy, sz] = to3D(edge->getSource());
    auto [tx, ty, tz] = to3D(edge->getTarget());
    ax.attr("text")(
      (sx + tx) / 2.,
      (sy + ty) / 2.,
      (sz + tz) / 2.,
      std::format("{:.1f}", edge->getSquaredLength()),
      py::arg("color") = "grey",
      py::arg("fontsize") = 7
    );
  }
}

template<int dimensions>
void Plotter<dimensions>::setBounds() const {
  double xMin = std::numeric_limits<double>::max();
  double xMax = std::numeric_limits<double>::lowest();
  double yMin = std::numeric_limits<double>::max();
  double yMax = std::numeric_limits<double>::lowest();
  double zMin = std::numeric_limits<double>::max();
  double zMax = std::numeric_limits<double>::lowest();
  for (const auto &vertex : vertices) {
    auto [x, y, z] = to3D(vertex);
    xMin = std::min(xMin, x);
    xMax = std::max(xMax, x);
    yMin = std::min(yMin, y);
    yMax = std::max(yMax, y);
    zMin = std::min(zMin, z);
    zMax = std::max(zMax, z);
  }

  ax.attr("set_xlim")(xMin, xMax);
  ax.attr("set_ylim")(yMin, yMax);
  ax.attr("set_zlim")(zMin, zMax);
  ax.attr("set_xlabel")("X");
  ax.attr("set_ylabel")("Y");
  ax.attr("set_zlabel")("Z");


}

template<int dimensions>
void Plotter<dimensions>::addCollections() {
  labelEdges();
  labelVertices();
  setBounds();

  std::array<double, dimensions> upperBounds{};
  std::array<double, dimensions> lowerBounds{};

  py::module art3d = py::module::import("mpl_toolkits.mplot3d.art3d");
  py::object Line3DCollection = art3d.attr("Line3DCollection");

  py::list pyTimelikeEdges{};
  for (const auto &edge : timelikeEdges) {
    if (simplexEdges.contains(edge)) continue;
    auto source = to3D(edge->getSource());
    auto target = to3D(edge->getTarget());
    pyTimelikeEdges.append(
      py::make_tuple(py::cast(source), py::cast(target)));
  }
  py::list pySpacelikeEdges{};
  for (const auto &edge : spacelikeEdges) {
    if (simplexEdges.contains(edge)) continue;
    auto source = to3D(edge->getSource());
    auto target = to3D(edge->getTarget());
    pySpacelikeEdges.append(
      py::make_tuple(py::cast(source), py::cast(target)));
  }

  py::list pyVertices{};
  for (const auto &vertex : vertices) {
    if (simplexVertices.contains(vertex)) continue;
    auto coords = to3D(vertex);
    pyVertices.append(py::cast(coords));
  }

  py::object timelikeCollection = Line3DCollection(
    pyTimelikeEdges,
    py::arg("linewidths") = .7,
    py::arg("colors") = "blue"
  );

  py::object spacelikeCollection = Line3DCollection(
    pySpacelikeEdges,
    py::arg("linewidths") = .7,
    py::arg("colors") = "red"
  );

  ax.attr("add_collection")(timelikeCollection);
  ax.attr("add_collection")(spacelikeCollection);

  for (const auto &[simplex, color] : simplexColors) {
    py::object simplexCollection = Line3DCollection(
      simplex->getEdges(),
      py::arg("linewidths") = .9,
      py::arg("colors") = color
    );
    ax.attr("add_collection")(simplexCollection);
  }
}

template<int dimensions>
void Plotter<dimensions>::addSimplex(const SimplexPtr &simplex, std::string color) {
  simplexColors[simplex] = color;
  for (const auto &v : simplex->getVertices()) {
    vertices.erase(v);
    simplexVertices.insert(v);
  }
  for (const auto &e : simplex->getEdges()) {
    if (e->getSquaredLength() < 0) timelikeEdges.erase(e);
    else spacelikeEdges.erase(e);
    simplexEdges.insert(e);
  }
}

template class Plotter<1>;
template class Plotter<2>;
template class Plotter<3>;
template class Plotter<4>;

} // caset
