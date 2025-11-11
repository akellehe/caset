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
    py::class_<Topology, std::shared_ptr<Topology> >(m, "Topology");

    py::class_<Sphere, Topology, std::shared_ptr<Sphere> >(m, "Sphere")
            .def(py::init<>())
            .def("build", &Sphere::build);

    py::class_<Toroid, Topology, std::shared_ptr<Toroid> >(m, "Toroid")
            .def(py::init<>())
            .def("build", &Toroid::build);

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
            .def(py::init<const std::vector<std::shared_ptr<Vertex> > &>(),
                 py::arg("vertices"))
            .def("getDeficitAngle", &Simplex::getDeficitAngle)
            .def("getHinges", &Simplex::getHinges)
            .def("getVolume", &Simplex::getVolume)
            .def("fingerprint", &Simplex::fingerprint);

    py::class_<Vertex, std::shared_ptr<Vertex> >(m, "Vertex")
            .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"))
            .def("addInEdge", &Vertex::addInEdge, py::arg("edge"))
            .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"))
            .def("getCoordinates", &Vertex::getCoordinates)
            .def("getEdges", &Vertex::getEdges)
            .def("getId", &Vertex::getId)
            .def("getInEdges", &Vertex::getInEdges)
            .def("getOutEdges", &Vertex::getOutEdges);

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
            .def_static("createVertex", py::overload_cast<const std::uint64_t>(&Spacetime::createVertex), py::arg("id"))
            .def_static("createVertex",
                        py::overload_cast<const std::uint64_t, const std::vector<double> &>(&Spacetime::createVertex),
                        py::arg("id"),
                        py::arg("coordinates"))
            .def_static("createEdge",
                        py::overload_cast<std::shared_ptr<Vertex> &, std::shared_ptr<Vertex> &>(&Spacetime::createEdge),
                        py::arg("source"),
                        py::arg("target"))
            .def_static("createEdge",
                        py::overload_cast<std::shared_ptr<Vertex> &, std::shared_ptr<Vertex> &,
                                          double>(&Spacetime::createEdge),
                        py::arg("source"),
                        py::arg("target"),
                        py::arg("squaredLength"))
            .def_static("createSimplex", &Spacetime::createSimplex);

    m.doc() = "A C++ library for simulating lattice spacetime and causal sets";
}
