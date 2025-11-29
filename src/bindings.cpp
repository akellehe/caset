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

namespace py = pybind11;

using namespace caset;

PYBIND11_MODULE(caset, m) {
  py::class_<Edge, std::shared_ptr<Edge> >(m, "Edge")
      .def(
        py::init<
          std::uint64_t,
          std::uint64_t
        >(),
        py::arg("source"),
        py::arg("target")
      )
      .def(
        py::init<
          std::uint64_t,
          std::uint64_t,
          double>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength")
      )
      .def("__str__", &Edge::toString)
      .def("__repr__", &Edge::toString)
      .def("__eq__", &Edge::operator==)
      .def("__hash__", &Edge::toHash)
      .def("getSourceId", &Edge::getSourceId)
      .def("getSquaredLength", &Edge::getSquaredLength)
      .def("redirect", &Edge::redirect)
      .def("getTargetId", &Edge::getTargetId);

  py::class_<Vertex, std::shared_ptr<Vertex> >(m, "Vertex")
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"))
      .def("__eq__", &Vertex::operator==)
      .def("__repr__", &Vertex::toString)
      .def("__str__", &Vertex::toString)
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"))
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"))
      .def("degree", &Vertex::degree)
      .def("getCoordinates", &Vertex::getCoordinates)
      .def("setCoordinates", &Vertex::setCoordinates, py::arg("coordinates"))
      .def("getEdges", &Vertex::getEdges)
      .def("getId", &Vertex::getId)
      .def("getInEdges", &Vertex::getInEdges)
      .def("getOutEdges", &Vertex::getOutEdges)
      .def("removeInEdge", &Vertex::removeInEdge)
      .def("removeOutEdge", &Vertex::removeOutEdge)
      .def("getTime", &Vertex::getTime)
      .def("moveTo", &Vertex::moveTo);

  py::class_<VertexList, std::shared_ptr<VertexList> >(m, "VertexList")
      .def(py::init<>())
      .def("__getitem__", &VertexList::operator[])
      .def("get", &VertexList::get)
      .def("add", py::overload_cast<const std::shared_ptr<Vertex> &>(&VertexList::add))
      .def("add", py::overload_cast<const std::uint64_t, const std::vector<double> &>(&VertexList::add))
      .def("add", py::overload_cast<const std::uint64_t>(&VertexList::add))
      .def("replace", &VertexList::replace)
      .def("size", &VertexList::size)
      .def("toVector", &VertexList::toVector);

  py::class_<EdgeList, std::shared_ptr<EdgeList> >(m, "EdgeList")
      .def(py::init<>())
      .def("add", py::overload_cast<const std::shared_ptr<Edge> &>(&EdgeList::add))
      .def("add", py::overload_cast<const std::uint64_t, const std::uint64_t, double>(&EdgeList::add))
      .def("add", py::overload_cast<const std::uint64_t, const std::uint64_t>(&EdgeList::add))
      .def("remove", py::overload_cast<const EdgeKey &>(&EdgeList::remove), py::arg("edgeKey"))
      .def("remove", py::overload_cast<const EdgePtr &>(&EdgeList::remove), py::arg("edge"))
      .def("size", &EdgeList::size)
      .def("toVector", &EdgeList::toVector);

  py::class_<Topology, std::shared_ptr<Topology> >(m, "Topology");

  py::class_<Sphere, Topology, std::shared_ptr<Sphere> >(m, "Sphere")
      .def(py::init<>())
      .def("build", &Sphere::build);

  py::class_<Toroid, Topology, std::shared_ptr<Toroid> >(m, "Toroid")
      .def(py::init<>())
      .def("build", &Toroid::build);


  py::class_<SimplexOrientation, std::shared_ptr<SimplexOrientation> >(m, "SimplexOrientation")
      .def(py::init<uint8_t, uint8_t>())
      .def("getOrientation", &SimplexOrientation::getOrientation)
      .def("__hash__", &SimplexOrientation::fingerprint)
      .def("__eq__", &SimplexOrientation::operator==)
      .def("numeric", &SimplexOrientation::numeric);

  py::class_<SimplexOrientationHash, std::shared_ptr<SimplexOrientationHash>>(m, "SimplexOrientationHash")
    .def(py::init<>());
  py::class_<SimplexOrientationEq, std::shared_ptr<SimplexOrientationEq>>(m, "SimplexOrientationEq")
    .def(py::init<>());

  py::class_<Simplex, std::shared_ptr<Simplex> >(m, "Simplex")
      .def(py::init<const std::vector<std::shared_ptr<Vertex> >>(),
           py::arg("vertices"))
      .def("__repr__", &Simplex::toString)
      .def("__str__", &Simplex::toString)
      .def("checkPairty", &Simplex::checkParity)
      .def("getCofaces", &Simplex::getCofaces)
      .def("getDeficitAngle", &Simplex::getDeficitAngle)
      .def("getEdges", &Simplex::getEdges)
      .def("getFacets", &Simplex::getFacets)
      .def("getHinges", &Simplex::getHinges)
      .def("getNumberOfFaces", &Simplex::getNumberOfFaces)
      .def("getOrientation", &Simplex::getOrientation)
      .def("getVertexIdLookup", &Simplex::getVertexIdLookup)
      .def("getVertices", &Simplex::getVertices)
      .def("getVerticesWithPairtyTo", &Simplex::getVerticesWithParityTo, py::arg("other"))
      .def("getVolume", &Simplex::getVolume)
      .def("hasEdge", py::overload_cast<const EdgePtr &>(&Simplex::hasEdge), py::arg("edge"))
      .def("hasEdge", py::overload_cast<const IdType, const IdType>(&Simplex::hasEdge), py::arg("source"), py::arg("target"))
      .def("isTimelike", &Simplex::isTimelike)
      .def("removeEdge", py::overload_cast<const EdgeKey &>(&Simplex::removeEdge), py::arg("edgeKey"))
      .def("removeEdge", py::overload_cast<const EdgePtr &>(&Simplex::removeEdge), py::arg("edge"))
      .def("removeVertex", &Simplex::removeVertex)
      .def("replaceVertex", &Simplex::replaceVertex);

  py::class_<SimplexHash, std::shared_ptr<SimplexHash>>(m, "SimplexHash")
    .def(py::init<>());
  py::class_<SimplexEq, std::shared_ptr<SimplexEq>>(m, "SimplexEq")
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

  py::class_<Signature, std::shared_ptr<Signature> >(m, "Signature")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signature_type"))
      .def("getDiagonal", &Signature::getDiagonal);

  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m, "Spacetime")
      .def(py::init<
             std::shared_ptr<Metric>,
             const SpacetimeType,
             std::optional<double>,
             std::optional<std::shared_ptr<Topology> >
           >(),
           py::arg("metric"),
           py::arg("spacetimeType"),
           py::arg("alpha"),
           py::arg("topology")
      )
      .def(py::init<>())
      .def("getVertexList", &Spacetime::getVertexList)
      .def("getSimplicesWithOrientation", &Spacetime::getSimplicesWithOrientation, py::arg("orientation"))
      .def("getEdgeList", &Spacetime::getEdgeList)
      .def("getGluableFaces", &Spacetime::getGluableFaces)
      .def("embedEuclidean", &Spacetime::embedEuclidean, py::arg("dimensions") = 4, py::arg("epsilon") = 1e-8)
      .def("getConnectedComponents", &Spacetime::getConnectedComponents)
      .def("build", &Spacetime::build)
      .def("getSimplices", &Spacetime::getExternalSimplices)
      .def("chooseSimplexFacesToGlue", &Spacetime::chooseSimplexFacesToGlue, py::arg("simplex"))
      .def("createVertex",
           py::overload_cast<const std::uint64_t>(&Spacetime::createVertex),
           py::arg("id"))
      .def("createVertex",
           py::overload_cast<const std::uint64_t, const std::vector<double> &>(
             &Spacetime::createVertex),
           py::arg("id"),
           py::arg("coordinates"))
      .def("createEdge",
           py::overload_cast<
             const std::uint64_t,
             const std::uint64_t>(
             &Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"))
      .def("createEdge",
           py::overload_cast<
             const std::uint64_t,
             const std::uint64_t,
             double>(&Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::arg("squaredLength"))
      .def("createSimplex",
           py::overload_cast<std::vector<std::shared_ptr<Vertex> > &>(&Spacetime::createSimplex),
           py::arg("vertices"))
      .def("createSimplex",
           py::overload_cast<const std::tuple<uint8_t, uint8_t> &>(&Spacetime::createSimplex),
           py::arg("orientation"))
      .def("attachAtVertices", &Spacetime::attachAtVertices, py::arg("simplex"), py::arg("vertexA"), py::arg("vertexB"))
      .def("moveInEdgesFromVertex", &Spacetime::moveInEdgesFromVertex, py::arg("fromVertex"), py::arg("toVertex"))
      .def("moveOutEdgesFromVertex", &Spacetime::moveOutEdgesFromVertex, py::arg("fromVertex"), py::arg("toVertex"))
      .def("causallyAttachFaces", &Spacetime::causallyAttachFaces);

  m.doc() = "A C++ library for simulating lattice spacetime and causal sets";
}
