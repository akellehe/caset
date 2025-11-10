#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "../include/spacetime/Spacetime.h"
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
          std::shared_ptr<Vertex>,
          std::shared_ptr<Vertex>
        >(),
        py::arg("source"),
        py::arg("target")
      )
      .def(
        py::init<
          std::shared_ptr<Vertex>,
          std::shared_ptr<Vertex>,
          double>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength")
      )
      .def("getSource", &Edge::getSource)
      .def("getTarget", &Edge::getTarget);

  py::class_<Simplex, std::shared_ptr<Simplex> >(m, "Simplex")
      .def(py::init<const std::shared_ptr<Spacetime> &, std::vector<std::shared_ptr<Vertex> > &>(),
           py::arg("spacetime"),
           py::arg("vertices"))
      .def("getDeficitAngle", &Simplex::getDeficitAngle)
      .def("getHinges", &Simplex::getHinges)
      .def("getVolume", &Simplex::getVolume);

  py::class_<Vertex, std::shared_ptr<Vertex> >(m, "Vertex")
      .def(py::init<std::vector<double> &>(), py::arg("coordinates"))
      .def("getCoordinates", &Vertex::getCoordinates)
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"))
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"))
      .def("getEdges", &Vertex::getEdges)
      .def("getOutEdges", &Vertex::getOutEdges)
      .def("getInEdges", &Vertex::getInEdges);

  py::class_<Metric, std::shared_ptr<Metric> >(m, "Metric")
      .def(py::init<bool, Signature &>(), py::arg("coordinateFree"), py::arg("signature"))
      .def("getSquaredLength", &Metric::getSquaredLength);

  py::enum_<SignatureType>(m, "SignatureType")
      .value("Lorentzian", SignatureType::Lorentzian)
      .value("Euclidean", SignatureType::Euclidean)
      .export_values();

  py::class_<Signature, std::shared_ptr<Signature> >(m, "Signature")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signature_type"))
      .def("getDiagonal", &Signature::getDiagonal);

  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m, "Spacetime")
    .def(py::init<std::shared_ptr<Metric>>(), py::arg("metric"));

  m.doc() = "A C++ library for simulating lattice spacetime and causal sets";
}
