// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "spacetime/topologies/Topology.h"
#include "spacetime/topologies/Sphere.h"
#include "spacetime/topologies/Toroid.h"
#include "spacetime/Spacetime.h"
#include "VertexList.h"
#include "EdgeList.h"
#include "Signature.h"
#include "Vertex.h"
#include "Edge.h"
#include "Simplex.h"
#include "Metric.h"

#include <vector>

#define DIMENSIONS 4;

namespace py = pybind11;

using namespace caset;

PYBIND11_MODULE(caset, m) {
  py::class_<Edge4D, EdgePtr4D>(m, "Edge")
      .def(
        py::init<
          const VertexPtr4D &,
          const VertexPtr4D &>(),
        py::arg("source"),
        py::arg("target")
      )
      .def(
        py::init<
          const VertexPtr4D &,
          const VertexPtr4D &,
          double>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength")
      )
      .def("__str__", &Edge4D::toString)
      .def("__repr__", &Edge4D::toString)
      .def("__eq__", &Edge4D::operator==)
      .def("__hash__", &Edge4D::toHash)
      .def("getSource", &Edge4D::getSource)
      .def("getSquaredLength", &Edge4D::getSquaredLength)
      .def("getTarget", &Edge4D::getTarget);

  py::class_<Vertex4D, VertexPtr4D>(m, "Vertex")
      .def("__eq__", &Vertex4D::operator==)
      .def("__repr__", &Vertex4D::toString)
      .def("__str__", &Vertex4D::toString)
      .def("addInEdge", &Vertex4D::addInEdge, py::arg("edge"))
      .def("addOutEdge", &Vertex4D::addOutEdge, py::arg("edge"))
      .def("degree", &Vertex4D::degree)
      .def("getCoordinates", &Vertex4D::getCoordinates)
      .def("getEdges", &Vertex4D::getEdges)
      .def("getId", &Vertex4D::getId)
      .def("getInEdges", &Vertex4D::getInEdges)
      .def("getOutEdges", &Vertex4D::getOutEdges)
      .def("getSimplices", &Vertex4D::getSimplices)
      .def("getTime", &Vertex4D::getTime)
      .def("moveEdgesTo", &Vertex4D::moveEdgesTo)
      .def("removeInEdge", &Vertex4D::removeInEdge)
      .def("removeOutEdge", &Vertex4D::removeOutEdge)
      .def("setCoordinates", &Vertex4D::setCoordinates, py::arg("coordinates"))
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"));

  py::class_<VertexList4D, std::shared_ptr<VertexList4D> >(m, "VertexList")
      .def(py::init<>())
      .def("__getitem__", &VertexList4D::operator[])
      .def("get", &VertexList4D::get)
      .def("add", py::overload_cast<const VertexPtr4D &>(&VertexList4D::add))
      .def("add",
           py::overload_cast<const std::uint64_t, const std::vector<double> &>(&VertexList4D::add))
      .def("add", py::overload_cast<const std::uint64_t>(&VertexList4D::add))
      .def("replace", &VertexList4D::replace)
      .def("size", &VertexList4D::size)
      .def("toVector", &VertexList4D::toVector);

  py::class_<EdgeList4D, std::shared_ptr<EdgeList4D> >(m, "EdgeList")
      .def(py::init<>())
      .def("add", py::overload_cast<const EdgePtr4D &>(&EdgeList4D::add))
      .def("add", py::overload_cast<const VertexPtr4D &, const VertexPtr4D &, double>(&EdgeList4D::add))
      .def("add", py::overload_cast<const VertexPtr4D &, const VertexPtr4D &>(&EdgeList4D::add))
      .def("remove", py::overload_cast<const EdgePtr4D &>(&EdgeList4D::remove), py::arg("edge"))
      .def("size", &EdgeList4D::size)
      .def("toVector", &EdgeList4D::toVector);

  py::class_<Topology4D, std::shared_ptr<Topology4D>>(m, "Topology");

  py::class_<Sphere4D, Topology4D, std::shared_ptr<Sphere4D> >(m, "Sphere")
      .def(py::init<>())
      .def("build", &Sphere4D::build);

  py::class_<Toroid4D, Topology4D, std::shared_ptr<Toroid4D> >(m, "Toroid")
      .def(py::init<>())
      .def("build", &Toroid4D::build);

  py::class_<SimplexOrientation, std::shared_ptr<SimplexOrientation> >(m, "SimplexOrientation")
      .def(py::init<uint8_t, uint8_t>())
      .def("getOrientation", &SimplexOrientation::getOrientation)
      .def("__hash__", &SimplexOrientation::hash)
      .def("__eq__", &SimplexOrientation::operator==)
      .def("__str__", &SimplexOrientation::toString)
      .def("__repr__", &SimplexOrientation::toString)
      .def("numeric", &SimplexOrientation::numeric);

  py::class_<SimplexOrientationHash, std::shared_ptr<SimplexOrientationHash> >(m, "SimplexOrientationHash")
      .def(py::init<>());
  py::class_<SimplexOrientationEq, std::shared_ptr<SimplexOrientationEq> >(m, "SimplexOrientationEq")
      .def(py::init<>());

  py::class_<Simplex4D, std::shared_ptr<Simplex4D> >(m, "Simplex")
      .def("__repr__", &Simplex4D::toString)
      .def("__str__", &Simplex4D::toString)
      .def("__hash__", &Simplex4D::hash)
      .def("__eq__",
           static_cast<bool (Simplex4D::*)(const SimplexPtr &) const noexcept>(&Simplex4D::operator==))
      .def("__eq__",
           static_cast<bool (Simplex4D::*)(const Simplex &) const noexcept>(&Simplex4D::operator==))
      .def("getCofaces", &Simplex4D::getCofaces)
      .def("getEdges", &Simplex4D::getEdges)
      .def("getFacets", &Simplex4D::getFacets)
      .def("getNumberOfFaces", &Simplex4D::getNumberOfFaces)
      .def("getOrientation", &Simplex4D::getOrientation)
      .def("getVertexIdLookup", &Simplex4D::getVertexIdLookup)
      .def("getVertices", &Simplex4D::getVertices)
      .def("hasVertex", &Simplex4D::hasVertex)
      .def("isCofaceTo", &Simplex4D::isCofaceTo, py::arg("facet"), py::arg("shallow") = true)
      .def("isInitialized", &Simplex4D::isInitialized)
      .def("isTimelike", &Simplex4D::isTimelike)
      .def("replaceVertex", &Simplex4D::replaceVertex, py::arg("oldVertex"), py::arg("newVertex"))
      .def("validate", &Simplex4D::validate);

  py::class_<SimplexHash, std::shared_ptr<SimplexHash> >(m, "SimplexHash")
      .def(py::init<>());
  py::class_<SimplexEq, std::shared_ptr<SimplexEq> >(m, "SimplexEq")
      .def(py::init<>());

  py::class_<Metric, std::shared_ptr<Metric> >(m, "Metric")
      .def(py::init<bool, Signature &>(),
           py::arg("coordinateFree"),
           py::arg("signature"))
      .def("getSquaredLength", &Metric::getSquaredLength);

  py::enum_<SignatureType>(m, "SignatureType")
      .value("Lorentzian", SignatureType::Lorentzian)
      .value("Euclidean", SignatureType::Euclidean)
      .export_values();

  py::enum_<Foliation>(m, "Foliation")
      .value("PREFERRED", Foliation::PREFERRED)
      .value("NONE", Foliation::NONE)
      .export_values();

  py::class_<Signature, std::shared_ptr<Signature> >(m, "Signature")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signature_type"))
      .def("getDiagonal", &Signature::getDiagonal);

  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m, "Spacetime")
      .def(py::init<
             std::shared_ptr<Metric>,
             const SpacetimeType,
             std::optional<double>,
             std::optional<double>,
             Foliation,
             std::optional<std::shared_ptr<Topology> >
           >(),
           py::arg("metric"),
           py::arg("spacetimeType"),
           py::arg("alpha"),
           py::arg("a"),
           py::arg("foliation"),
           py::arg("topology")
      )
      .def(py::init<>())
      .def("getVertexList", &Spacetime::getVertexList)
      .def("getSimplicesWithOrientation",
           &Spacetime::getSimplicesWithOrientation,
           py::arg("orientation"))
      .def("getEdgeList", &Spacetime::getEdgeList)
      .def("getConnectedComponents", &Spacetime::getConnectedComponents)
      .def("build", &Spacetime::build)
      .def("getSimplices", &Spacetime::getExternalSimplices)
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"))
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &, double) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::arg("squaredLength"))
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t) const noexcept>(
             &Spacetime::createVertex))
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t, const std::vector<double> &) const noexcept>(
             &Spacetime::createVertex))
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"))
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &, const std::vector<EdgePtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::arg("edges"))
      .def("createSimplex",
           py::overload_cast<const std::tuple<uint8_t, uint8_t> &>(&Spacetime::createSimplex),
           py::arg("orientation"));
           */

  m.doc() = "A C++ library for simulating lattice spacetime and causal sets";
}
