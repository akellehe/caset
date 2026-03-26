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
#include "simulations/CDT.h"
#include "observables/VolumeProfile.h"
#include "spacetime/Spacetime.h"
#include "VertexList.h"
#include "EdgeList.h"
#include "Signature.h"
#include "Vertex.h"
#include "Edge.h"
#include "Simplex.h"
#include "Metric.h"

#include <vector>
#include <algorithm>

namespace py = pybind11;

using namespace caset;

PYBIND11_MODULE(caset, m) {
  py::class_<Edge, std::unique_ptr<Edge, py::nodelete>>(m, "Edge")
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &>(),
        py::arg("source"),
        py::arg("target")
      )
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &,
          double>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength")
      )
      .def("__str__", &Edge::toString)
      .def("__repr__", &Edge::toString)
      .def("__eq__", &Edge::operator==)
      .def("__hash__", &Edge::toHash)
      .def("getSource", &Edge::getSource, py::return_value_policy::reference)
      .def("getSquaredLength", &Edge::getSquaredLength)
      .def("getTarget", &Edge::getTarget, py::return_value_policy::reference);

  py::class_<Vertex, std::unique_ptr<Vertex, py::nodelete>>(m, "Vertex")
      .def("__eq__", &Vertex::operator==)
      .def("__repr__", &Vertex::toString)
      .def("__str__", &Vertex::toString)
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"))
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"))
      .def("degree", &Vertex::degree)
      .def("getCoordinates", &Vertex::getCoordinates)
      .def("getEdges", &Vertex::getEdges, py::return_value_policy::reference)
      .def("getId", &Vertex::getId)
      .def("getInEdges", &Vertex::getInEdges, py::return_value_policy::reference)
      .def("getOutEdges", &Vertex::getOutEdges, py::return_value_policy::reference)
      .def("getSimplices", &Vertex::getSimplices, py::return_value_policy::reference)
      .def("getTime", &Vertex::getTime)
      .def("moveEdgesTo", &Vertex::moveEdgesTo)
      .def("removeInEdge", &Vertex::removeInEdge)
      .def("removeOutEdge", &Vertex::removeOutEdge)
      .def("setCoordinates", &Vertex::setCoordinates, py::arg("coordinates"))
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"));

  py::class_<VertexList, std::shared_ptr<VertexList> >(m, "VertexList")
      .def(py::init<>())
      .def("__getitem__", &VertexList::operator[], py::return_value_policy::reference)
      .def("get", &VertexList::get, py::return_value_policy::reference)
      .def("add",
           py::overload_cast<const std::uint64_t, const std::vector<double> &>(&VertexList::add),
           py::return_value_policy::reference)
      .def("add", py::overload_cast<const std::uint64_t>(&VertexList::add),
           py::return_value_policy::reference)
      .def("replace", &VertexList::replace)
      .def("size", &VertexList::size)
      .def("toVector", &VertexList::toVector, py::return_value_policy::reference);

  py::class_<EdgeList, std::shared_ptr<EdgeList> >(m, "EdgeList")
      .def(py::init<>())
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &, double>(&EdgeList::add),
           py::return_value_policy::reference)
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &>(&EdgeList::add),
           py::return_value_policy::reference)
      .def("remove", py::overload_cast<const EdgePtr &>(&EdgeList::remove), py::arg("edge"))
      .def("size", &EdgeList::size)
      .def("toVector", &EdgeList::toVector, py::return_value_policy::reference);

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
      .def("__hash__", &SimplexOrientation::hash)
      .def("__eq__", &SimplexOrientation::operator==)
      .def("__str__", &SimplexOrientation::toString)
      .def("__repr__", &SimplexOrientation::toString)
      .def("numeric", &SimplexOrientation::numeric);

  py::class_<SimplexOrientationHash, std::shared_ptr<SimplexOrientationHash> >(m, "SimplexOrientationHash")
      .def(py::init<>());
  py::class_<SimplexOrientationEq, std::shared_ptr<SimplexOrientationEq> >(m, "SimplexOrientationEq")
      .def(py::init<>());

  py::class_<Simplex, std::unique_ptr<Simplex, py::nodelete>>(m, "Simplex")
      .def("__repr__", &Simplex::toString)
      .def("__str__", &Simplex::toString)
      .def("__hash__", &Simplex::hash)
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex*) const noexcept>(&Simplex::operator==))
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex &) const noexcept>(&Simplex::operator==))
      .def("getCofaces", &Simplex::getCofaces, py::return_value_policy::reference)
      .def("getEdges", &Simplex::getEdges, py::return_value_policy::reference)
      .def("getFacets", &Simplex::getFacets, py::return_value_policy::reference)
      .def("getNumberOfFaces", &Simplex::getNumberOfFaces)
      .def("getOrientation", &Simplex::getOrientation)
      .def("getVertexIdLookup", &Simplex::getVertexIdLookup, py::return_value_policy::reference)
      .def("getVertices", &Simplex::getVertices, py::return_value_policy::reference)
      .def("hasVertex", &Simplex::hasVertex)
      .def("isCofaceTo", &Simplex::isCofaceTo, py::arg("facet"), py::arg("shallow") = true)
      .def("isInitialized", &Simplex::isInitialized)
      .def("isTimelike", &Simplex::isTimelike)
      .def("replaceVertex", &Simplex::replaceVertex, py::arg("oldVertex"), py::arg("newVertex"))
      .def("validate", &Simplex::validate);

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

  py::enum_<SpacetimeType>(m, "SpacetimeType")
      .value("CDT", SpacetimeType::CDT)
      .value("REGGE", SpacetimeType::REGGE)
      .value("COSET", SpacetimeType::COSET)
      .export_values();

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
           py::arg("orientation"),
           py::return_value_policy::reference)
      .def("getEdgeList", &Spacetime::getEdgeList)
      .def("getConnectedComponents", &Spacetime::getConnectedComponents, py::return_value_policy::reference)
      .def("build", &Spacetime::build, py::call_guard<py::gil_scoped_release>())
      .def("getSimplices", &Spacetime::getSimplices, py::return_value_policy::reference)
      .def("getExternalSimplices", &Spacetime::getExternalSimplices, py::return_value_policy::reference)
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::return_value_policy::reference)
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &, double) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::arg("squaredLength"),
           py::return_value_policy::reference)
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t) const noexcept>(
             &Spacetime::createVertex),
           py::return_value_policy::reference)
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t, const std::vector<double> &) const noexcept>(
             &Spacetime::createVertex),
           py::return_value_policy::reference)
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::return_value_policy::reference)
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &, const std::vector<EdgePtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::arg("edges"),
           py::return_value_policy::reference)
      .def("createSimplex",
           py::overload_cast<const std::tuple<uint8_t, uint8_t> &>(&Spacetime::createSimplex),
           py::arg("orientation"),
           py::return_value_policy::reference)
      .def("getSimplexCount", &Spacetime::getSimplexCount)
      .def("getVertexCount", &Spacetime::getVertexCount)
      .def("getN41", &Spacetime::getN41)
      .def("getN32", &Spacetime::getN32)
      .def("getRandomSimplex", &Spacetime::getRandomSimplex, py::return_value_policy::reference)
      .def("getRandomTopSimplex", &Spacetime::getRandomTopSimplex, py::return_value_policy::reference)
      .def("getRandomVertex", &Spacetime::getRandomVertex, py::return_value_policy::reference)
      .def("removeSimplex", &Spacetime::removeSimplex, py::arg("simplex"));

  py::class_<CDT, std::shared_ptr<CDT> >(m, "CDTSimulation")
      .def(py::init<std::shared_ptr<Spacetime>, double, double, double, double, std::size_t, bool>(),
           py::arg("spacetime"),
           py::arg("k0"),
           py::arg("k4"),
           py::arg("delta"),
           py::arg("epsilon"),
           py::arg("targetN41"),
           py::arg("quadraticVolumeFix") = true)
      .def("add", &CDT::add)
      .def("remove", &CDT::remove)
      .def("flip", &CDT::flip)
      .def("iflip", &CDT::iflip)
      .def("shift", &CDT::shift)
      .def("ishift", &CDT::ishift)
      .def("sweep", [](CDT &self, int n_sweeps, py::object progress) {
          int total = 0;
          for (int i = 0; i < n_sweeps; i++) {
              int accepted;
              {
                  py::gil_scoped_release release;
                  accepted = self.sweep();
              }
              total += accepted;
              if (!progress.is_none()) {
                  progress(i + 1, n_sweeps);
              }
          }
          return total;
      }, py::arg("n_sweeps") = 1, py::arg("progress") = py::none())
      .def("tune", &CDT::tune, py::call_guard<py::gil_scoped_release>())
      .def("thermalize", &CDT::thermalize, py::call_guard<py::gil_scoped_release>())
      .def("computeAction", &CDT::computeAction)
      .def("getVolumeProfile", &CDT::getVolumeProfile)
      .def("getAcceptanceRates", &CDT::getAcceptanceRates)
      .def("getSpacetime", &CDT::getSpacetime)
      .def("getK0", &CDT::getK0)
      .def("getK4", &CDT::getK4)
      .def("getDelta", &CDT::getDelta);

  py::class_<VolumeProfile, std::shared_ptr<VolumeProfile> >(m, "VolumeProfile")
      .def(py::init<>())
      .def("compute", &VolumeProfile::compute, py::arg("spacetime"))
      .def("getProfile", &VolumeProfile::getProfile)
      .def("getAverageProfile", &VolumeProfile::getAverageProfile)
      .def("measure", &VolumeProfile::measure, py::arg("spacetime"))
      .def("reset", &VolumeProfile::reset);

  m.doc() = "A C++ library for simulating lattice spacetime and causal sets";

#ifdef CASET_VERSION
  m.attr("__version__") = CASET_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif
}
