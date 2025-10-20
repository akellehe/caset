#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "Vertex.h"
#include "Edge.h"

#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(caset, m) {

  py::class_<Vertex>(m, "Vertex")
      .def(py::init<int, std::vector<double>>())
      .def("getId", &Vertex::getId)
      .def("getCoordinates", &Vertex::getCoordinates);


py::class_<Edge, std::shared_ptr<Edge>>(m, "Edge")
.def(py::init<std::shared_ptr<Vertex>, std::shared_ptr<Vertex>, double>())
.def("getWeight", &Edge::getWeight);


  m.doc() = "A C++ library for simulating lattice spacetimes and causal sets";
}