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
        py::class_<Edge<4>, std::shared_ptr<Edge<4> > >(m, "Edge")
                        .def(py::init<std::shared_ptr<Vertex<4> >, std::shared_ptr<Vertex<4> > >())
                        .def("getLength", &Edge<4>::getLength);

        py::class_<Simplex<4> >(m, "Simplex")
                        .def(py::init<std::vector<std::shared_ptr<Edge<4> > > &>())
                        .def("getDeficitAngle", &Simplex<4>::getDeficitAngle)
                        .def("getHinges", &Simplex<4>::getHinges)
                        .def("getVolume", &Simplex<4>::getVolume)
                        .def("getGramMatrix", &Simplex<4>::getGramMatrix)
                        .def("getGramCofactor", &Simplex<4>::getGramCofactor);

        py::class_<Vertex<4> >(m, "Vertex")
                        .def(py::init<std::array<double, 4> &>())
                        .def("getCoordinates", &Vertex<4>::getCoordinates);

        m.doc() = "A C++ library for simulating lattice spacetimes and causal sets";
}
