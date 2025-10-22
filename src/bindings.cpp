#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "Vertex.h"
#include "Edge.h"
#include "Simplex.h"

#include <vector>

namespace py = pybind11;

using namespace caset;

PYBIND11_MODULE(caset, m) {
  py::class_<Edge, std::shared_ptr<Edge> >(m, "Edge")
      .def(py::init<std::shared_ptr<Vertex>, std::shared_ptr<Vertex>, double>())
      .def("getLength", &Edge::getLength);

  py::class_<Simplex>(m, "Simplex")
      .def(py::init<std::vector<std::shared_ptr<Edge> > &>())
      .def("getDeficitAngle", &Simplex::getDeficitAngle)
      .def("getHinges", &Simplex::getHinges)
      .def("getVolume", &Simplex::getVolume)
      .def("getGramMatrix", &Simplex::getGramMatrix)
      .def("getGramCofactor", &Simplex::getGramCofactor);

  py::class_<Vertex>(m, "Vertex")
      .def(py::init<int, std::vector<double> >())
      .def("getCoordinates", &Vertex::getCoordinates)
      .def("getId", &Vertex::getId);

  m.doc() = "A C++ library for simulating lattice spacetimes and causal sets";
}
