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
      .def("getTargetId", &Edge::getTargetId);

  py::class_<Vertex, std::shared_ptr<Vertex> >(m, "Vertex")
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"))
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"))
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"))
      .def("getEdges", &Vertex::getEdges)
      .def("getCoordinates", &Vertex::getCoordinates)
      .def("moveTo", &Vertex::moveTo)
      .def("getId", &Vertex::getId)
      .def("__str__", &Vertex::toString)
      .def("__repr__", &Vertex::toString)
      .def("__eq__", &Vertex::operator==)
      .def("getInEdges", &Vertex::getInEdges)
      .def("getTime", &Vertex::getTime)
      .def("getOutEdges", &Vertex::getOutEdges);

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
      .def("remove", &EdgeList::remove)
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
      .def("numeric", &SimplexOrientation::numeric);

  py::class_<Simplex, std::shared_ptr<Simplex> >(m, "Simplex")
      .def(py::init<const std::vector<std::shared_ptr<Vertex> >>(),
           py::arg("vertices"))
      .def("__repr__", &Simplex::toString)
      .def("__str__", &Simplex::toString)
      .def("checkPairty", &Simplex::checkPairty)
      .def("getCofaces", &Simplex::getCofaces)
      .def("getDeficitAngle", &Simplex::getDeficitAngle)
      .def("getEdges", &Simplex::getEdges)
      .def("getFacets", &Simplex::getFacets)
      .def("getHinges", &Simplex::getHinges)
      .def("getNumberOfFaces", &Simplex::getNumberOfFaces)
      .def("getOrientation", &Simplex::getOrientation)
      .def("getVertices", &Simplex::getVertices)
      .def("getVertices", &Simplex::getVertices)
      .def("getVolume", &Simplex::getVolume)
      .def("isTimelike", &Simplex::isTimelike);

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
      .def("getEdgeList", &Spacetime::getEdgeList)
      .def_static("getGluablePair", &Spacetime::getGluablePair)
      .def("embedEuclidean", &Spacetime::embedEuclidean, py::arg("dimensions") = 4, py::arg("epsilon") = 1e-8)
      .def("getSimplexes", &Spacetime::getSimplexes)
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
      .def("causallyAttachFaces", &Spacetime::causallyAttachFaces)
      .def("setManual", &Spacetime::setManual);

  m.doc() = "A C++ library for simulating lattice spacetime and causal sets";
}
