#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "Signature.h"
#include "Vertex.h"
#include "Edge.h"
#include "Simplex.h"
#include "Metric.h"

#include <vector>

namespace py = pybind11;

using namespace caset;

#include "caset_config.hpp"

static constexpr int kN = CASET_DIMENSIONS;

#if   defined(CASET_SIGNATURE_Lorentzian) && CASET_SIGNATURE_Lorentzian
static constexpr SignatureType kSig = SignatureType::Lorentzian;
#elif defined(CASET_SIGNATURE_Euclidean)  && CASET_SIGNATURE_Euclidean
static constexpr SignatureType kSig = SignatureType::Euclidean;
#else
# error "Exactly one of CASET_SIGNATURE_{Lorentzian,Euclidean} must be defined"
#endif

PYBIND11_MODULE(caset, m) {
  py::class_<Edge<kN>, std::shared_ptr<Edge<kN> > >(m, "Edge")
      .def(
        py::init<
          std::shared_ptr<Vertex<kN> >,
          std::shared_ptr<Vertex<kN> >
        >(),
        py::arg("source"),
        py::arg("target")
      )
      .def("getSource", &Edge<kN>::getSource)
      .def("getTarget", &Edge<kN>::getTarget);

  py::class_<Simplex<kN, kSig>, std::shared_ptr<Simplex<kN, kSig> > >(m, "Simplex")
      .def(py::init<std::vector<std::shared_ptr<Edge<kN> > > &>(), py::arg("edges"))
      .def("getDeficitAngle", &Simplex<kN, kSig>::getDeficitAngle)
      .def("getHinges", &Simplex<kN, kSig>::getHinges)
      .def("getVolume", &Simplex<kN, kSig>::getVolume);

  py::class_<Vertex<kN>, std::shared_ptr<Vertex<kN> > >(m, "Vertex")
      .def(py::init<std::array<double, kN> &>(), py::arg("coordinates"))
      .def("getCoordinates", &Vertex<kN>::getCoordinates);

  py::class_<Metric<kN, kSig>, std::shared_ptr<Metric<kN, kSig> > >(m, "Metric")
    .def(py::init<>())
    .def("getSquaredLength", &Metric<kN, kSig>::getSquaredLength);

  m.doc() = "A C++ library for simulating lattice spacetimes and causal sets";
}
