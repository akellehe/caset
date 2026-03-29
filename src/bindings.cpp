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
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/topologies/Sphere.h"
#include "spacetime/topologies/Toroid.h"
#include "simulations/CDT.h"
#include "simulations/ReggeSolver.h"
#include "matter/MatterConfiguration.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "spacetime/Spacetime.h"
#include "ForceLayout.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"
#include "spacetime/Signature.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "spacetime/Metric.h"
#include "Renderer.h"

#include <vector>
#include <algorithm>

namespace py = pybind11;

using namespace caset;

PYBIND11_MODULE(caset, m) {
  m.doc() = R"doc(
caset -- Causal Set and CDT simulation library.

A C++ library (with Python bindings) for Causal Dynamical Triangulations
(CDT) and causal set theory simulations in arbitrary dimension.

Typical usage::

    import caset

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(500)
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
    cdt.tune()
    cdt.sweep(100)

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
)doc";

  // ========================================
  // Edge
  // ========================================
  py::class_<Edge, std::unique_ptr<Edge, py::nodelete>>(m, "Edge",
      R"doc(An edge connecting two vertices in the simplicial complex.

Edges are 1-simplices linking a source and target vertex.  In CDT the
squared length determines the edge disposition: positive = spacelike,
negative = timelike, zero = lightlike.

Edges are identified by an order-independent fingerprint of their
endpoint vertex IDs, so Edge(v1, v2) == Edge(v2, v1).)doc")
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &>(),
        py::arg("source"),
        py::arg("target"),
        "Create an edge between two vertices with a random squared length."
      )
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &,
          double>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength"),
        "Create an edge between two vertices with a specified squared length."
      )
      .def("__str__", &Edge::toString)
      .def("__repr__", &Edge::toString)
      .def("__eq__", &Edge::operator==, py::arg("other"))
      .def("__hash__", &Edge::toHash)
      .def("getSource", &Edge::getSource, py::return_value_policy::reference,
           "Return the source vertex of this edge.")
      .def("getSquaredLength", &Edge::getSquaredLength,
           R"doc(Return the squared edge length.

Positive = spacelike, negative = timelike, zero = lightlike.
We work in squared lengths to avoid complex arithmetic for
timelike (imaginary-length) edges.)doc")
      .def("getTarget", &Edge::getTarget, py::return_value_policy::reference,
           "Return the target vertex of this edge.");

  // ========================================
  // Vertex
  // ========================================
  py::class_<Vertex, std::unique_ptr<Vertex, py::nodelete>>(m, "Vertex",
      R"doc(A point in the simplicial spacetime, identified by a unique integer ID.

Each vertex carries a time coordinate (the first element of its
coordinate vector) and maintains references to its incident edges
and containing simplices.)doc")
      .def("__eq__", &Vertex::operator==, py::arg("other"))
      .def("__repr__", &Vertex::toString)
      .def("__str__", &Vertex::toString)
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"),
           "Register an incoming edge on this vertex.")
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"),
           "Register an outgoing edge on this vertex.")
      .def("degree", &Vertex::degree,
           "Return the total number of edges incident to this vertex (in + out).")
      .def("getCoordinates", &Vertex::getCoordinates,
           "Return the coordinate vector of this vertex.")
      .def("getEdges", &Vertex::getEdges, py::return_value_policy::copy,
           "Return all edges (both in and out) incident to this vertex.")
      .def("getId", &Vertex::getId,
           "Return the unique integer ID of this vertex.")
      .def("getInEdges", &Vertex::getInEdges, py::return_value_policy::copy,
           "Return edges where this vertex is the target.")
      .def("getOutEdges", &Vertex::getOutEdges, py::return_value_policy::copy,
           "Return edges where this vertex is the source.")
      .def("getSimplices", &Vertex::getSimplices, py::return_value_policy::copy,
           "Return all simplices (of any dimension) containing this vertex.")
      .def("getTime", &Vertex::getTime,
           "Return the time coordinate of this vertex (first coordinate).")
      .def("moveEdgesTo", &Vertex::moveEdgesTo, py::arg("vertex"), py::arg("spacetime"),
           "Transfer all edges from this vertex to another vertex.")
      .def("removeInEdge", &Vertex::removeInEdge, py::arg("edge"),
           "Remove an incoming edge from this vertex and its containing simplices.")
      .def("removeOutEdge", &Vertex::removeOutEdge, py::arg("edge"),
           "Remove an outgoing edge from this vertex and its containing simplices.")
      .def("setCoordinates", &Vertex::setCoordinates, py::arg("coordinates"),
           "Set the coordinate vector of this vertex.")
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"),
           "Create a vertex with the given ID and coordinates.");

  // ========================================
  // VertexList
  // ========================================
  py::class_<VertexList, std::shared_ptr<VertexList> >(m, "VertexList",
      "Container mapping vertex IDs to Vertex objects.")
      .def(py::init<>())
      .def("__getitem__", &VertexList::operator[], py::arg("vertexId"), py::return_value_policy::reference,
           "Look up a vertex by ID (operator[]).")
      .def("get", &VertexList::get, py::arg("id"), py::return_value_policy::reference,
           "Look up a vertex by its integer ID.  Raises if not found.")
      .def("add",
           py::overload_cast<const std::uint64_t, const std::vector<double> &>(&VertexList::add),
           py::arg("id"), py::arg("coordinates"),
           py::return_value_policy::reference,
           "Create and insert a vertex with the given ID and coordinates.")
      .def("add", py::overload_cast<const std::uint64_t>(&VertexList::add),
           py::arg("id"),
           py::return_value_policy::reference,
           "Create and insert a vertex with the given ID (no coordinates).")
      .def("replace", &VertexList::replace, py::arg("toRemove"), py::arg("toAdd"),
           "Replace a vertex in the list (same ID, new object).")
      .def("size", &VertexList::size,
           "Return the number of vertices in the list.")
      .def("toVector", &VertexList::toVector, py::return_value_policy::reference,
           "Return all vertices as a list.");

  // ========================================
  // EdgeList
  // ========================================
  py::class_<EdgeList, std::shared_ptr<EdgeList> >(m, "EdgeList",
      "Container storing all edges in the spacetime, keyed by fingerprint.")
      .def(py::init<>())
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &, double>(&EdgeList::add),
           py::arg("source"), py::arg("target"), py::arg("squaredLength"),
           py::return_value_policy::reference,
           "Add an edge with a specified squared length, or return existing if duplicate.")
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &>(&EdgeList::add),
           py::arg("source"), py::arg("target"),
           py::return_value_policy::reference,
           "Add an edge with auto-computed squared length, or return existing if duplicate.")
      .def("remove", py::overload_cast<const EdgePtr &>(&EdgeList::remove), py::arg("edge"),
           "Remove an edge from the list.")
      .def("size", &EdgeList::size,
           "Return the total number of edges.")
      .def("toVector", &EdgeList::toVector, py::return_value_policy::reference,
           "Return all edges as a list.");

  // ========================================
  // Topologies
  // ========================================
  py::class_<Topology, std::shared_ptr<Topology> >(m, "Topology",
      "Base class for spatial topologies (Toroid, Sphere, etc.).");

  py::class_<Sphere, Topology, std::shared_ptr<Sphere> >(m, "Sphere",
      "Spherical spatial topology S^{d-1}.")
      .def(py::init<>())
      .def("build", &Sphere::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a spherical initial triangulation with the given number of simplices.");

  py::class_<Cylinder, Topology, std::shared_ptr<Cylinder> >(m, "Cylinder",
      "Cylindrical spatial topology with open time boundaries.")
      .def(py::init<>())
      .def("build", &Cylinder::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a cylindrical triangulation with the given number of simplices.");

  py::class_<Toroid, Topology, std::shared_ptr<Toroid> >(m, "Toroid",
      R"doc(Toroidal spatial topology (periodic boundary conditions).

Uses the staircase product triangulation: each time slab contains
d*(d+1) simplices covering all CDT orientation types (d,1), (d-1,2),
..., (1,d).  For d=4 this gives 20 simplices per slab with equal
numbers of (4,1), (3,2), (2,3), and (1,4) types, enabling all five
Pachner moves (add, remove, flip, iflip, shift).)doc")
      .def(py::init<>())
      .def("build", &Toroid::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a toroidal staircase triangulation with the given number of simplices.");

  // ========================================
  // SimplexOrientation
  // ========================================
  py::class_<SimplexOrientation, std::shared_ptr<SimplexOrientation> >(m, "SimplexOrientation",
      R"doc(CDT simplex orientation (ti, tf) counting vertices at each time slice.

For a d-simplex spanning times t and t+1:
  - ti = number of vertices at time t
  - tf = number of vertices at time t+1
  - ti + tf = d + 1

Valid CDT orientations: (d,1), (1,d), (d-1,2), (2,d-1).
For d=4: (4,1), (1,4), (3,2), (2,3).)doc")
      .def(py::init<uint8_t, uint8_t>(), py::arg("ti"), py::arg("tf"))
      .def("getOrientation", &SimplexOrientation::getOrientation,
           "Return the (ti, tf) orientation as a pair.")
      .def("__hash__", &SimplexOrientation::hash)
      .def("__eq__", &SimplexOrientation::operator==, py::arg("other"))
      .def("__str__", &SimplexOrientation::toString)
      .def("__repr__", &SimplexOrientation::toString)
      .def("numeric", &SimplexOrientation::numeric,
           "Return the orientation as a Python tuple (ti, tf).");

  py::class_<SimplexOrientationHash, std::shared_ptr<SimplexOrientationHash> >(m, "SimplexOrientationHash")
      .def(py::init<>());
  py::class_<SimplexOrientationEq, std::shared_ptr<SimplexOrientationEq> >(m, "SimplexOrientationEq")
      .def(py::init<>());

  // ========================================
  // Simplex
  // ========================================
  py::class_<Simplex, std::unique_ptr<Simplex, py::nodelete>>(m, "Simplex",
      R"doc(A k-simplex in the simplicial complex.

A k-simplex has k+1 vertices, C(k+1,2) edges, and k+1 facets
(each a (k-1)-simplex).  Top-dimensional simplices in d-dimensional
CDT have d+1 vertices (e.g. 5 vertices for d=4).

Simplices are identified by an order-independent fingerprint of
their vertex IDs.)doc")
      .def("__repr__", &Simplex::toString)
      .def("__str__", &Simplex::toString)
      .def("__hash__", &Simplex::hash)
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex*) const noexcept>(&Simplex::operator==),
           py::arg("other"))
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex &) const noexcept>(&Simplex::operator==),
           py::arg("other"))
      .def("getCofaces", &Simplex::getCofaces, py::return_value_policy::copy,
           "Return all simplices of one dimension higher that contain this simplex as a face.")
      .def("getEdges", &Simplex::getEdges, py::return_value_policy::copy,
           "Return the edges (1-faces) of this simplex.")
      .def("getFacets", &Simplex::getFacets, py::return_value_policy::copy,
           R"doc(Return the (k-1)-dimensional faces of this k-simplex.

For a top d-simplex with d+1 vertices, returns d+1 facets each with
d vertices.  Also registers coface relationships so that
facet.getCofaces() includes this simplex.)doc")
      .def("getNumberOfFaces", &Simplex::getNumberOfFaces,
           "Return the number of sub-faces at each dimension.")
      .def("getOrientation", &Simplex::getOrientation,
           "Return the CDT orientation (ti, tf) of this simplex.")
      .def("getVertexIdLookup", &Simplex::getVertexIdLookup, py::return_value_policy::copy,
           "Return the internal vertex-ID-to-index mapping.")
      .def("getVertices", &Simplex::getVertices, py::return_value_policy::copy,
           "Return the vertices of this simplex.")
      .def("hasVertex", &Simplex::hasVertex, py::arg("vertex"),
           "Return True if this simplex contains the given vertex.")
      .def("isCofaceTo", &Simplex::isCofaceTo, py::arg("facet"), py::arg("shallow") = true,
           "Return True if this simplex is a coface of the given facet.")
      .def("isInitialized", &Simplex::isInitialized,
           "Return True if this simplex has been properly initialized with vertices.")
      .def("isSpatial", &Simplex::isSpatial,
           "Return True if all vertices lie on the same time slice (purely spatial simplex).")
      .def("isTimelike", &Simplex::isTimelike,
           "Deprecated: misnamed. Returns True for *spatial* simplices (all same time). Use isSpatial().")
      .def("replaceVertex", &Simplex::replaceVertex, py::arg("oldVertex"), py::arg("newVertex"),
           "Replace a vertex in this simplex (updates fingerprint and internal maps).")
      .def("validate", &Simplex::validate,
           "Run internal consistency checks on this simplex.")
      .def("gramMatrix", &Simplex::gramMatrix,
           "Gram matrix from edge lengths (flat d*d row-major, Wick-rotated).")
      .def("dihedralAngle", &Simplex::dihedralAngle,
           py::arg("hinge"),
           "Dihedral angle at a hinge within this simplex.")
      .def("deficitAngle", &Simplex::deficitAngle,
           "Deficit angle at this hinge (2*pi - sum of dihedral angles).")
      .def("area", &Simplex::area,
           "Area of this triangle (hinge) via Heron's formula.");

  py::class_<SimplexHash, std::shared_ptr<SimplexHash> >(m, "SimplexHash")
      .def(py::init<>());
  py::class_<SimplexEq, std::shared_ptr<SimplexEq> >(m, "SimplexEq")
      .def(py::init<>());

  // ========================================
  // Metric
  // ========================================
  py::class_<Metric, std::shared_ptr<Metric> >(m, "Metric",
      R"doc(Spacetime metric defining edge lengths and causal structure.

In coordinate-free mode (the default for CDT), edge squared lengths
are stored directly on the edges rather than computed from coordinates.

Args:
    coordinateFree: If True, squared lengths are stored on edges directly.
    signature: The metric signature (Lorentzian or Euclidean).)doc")
      .def(py::init<bool, Signature &>(),
           py::arg("coordinateFree"),
           py::arg("signature"))
      .def("getSquaredLength", &Metric::getSquaredLength,
           py::arg("sourceCoords"), py::arg("targetCoords"),
           "Compute the squared length of an edge from vertex coordinates.");

  // ========================================
  // Enums
  // ========================================
  py::enum_<SignatureType>(m, "SignatureType",
      "Metric signature: Lorentzian (-,+,+,...) or Euclidean (+,+,+,...).")
      .value("Lorentzian", SignatureType::Lorentzian)
      .value("Euclidean", SignatureType::Euclidean)
      .export_values();

  py::enum_<Foliation>(m, "Foliation",
      "Time foliation type for CDT spacetimes.")
      .value("PREFERRED", Foliation::PREFERRED)
      .value("NONE", Foliation::NONE)
      .export_values();

  // ========================================
  // Signature
  // ========================================
  py::class_<Signature, std::shared_ptr<Signature> >(m, "Signature",
      R"doc(Metric signature specifying dimension and type.

Args:
    dimensions: Number of spacetime dimensions d (e.g. 4 for 4D CDT).
    signature_type: Lorentzian or Euclidean.)doc")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signature_type"))
      .def("getDiagonal", &Signature::getDiagonal,
           "Return the diagonal entries of the metric signature tensor.");

  py::enum_<SpacetimeType>(m, "SpacetimeType",
      "Type of spacetime simulation.")
      .value("CDT", SpacetimeType::CDT)
      .value("REGGE", SpacetimeType::REGGE)
      .value("COSET", SpacetimeType::COSET)
      .export_values();

  // ========================================
  // Spacetime
  // ========================================
  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m, "Spacetime",
      R"doc(The simplicial spacetime manifold.

Holds the full simplicial complex: vertices, edges, and simplices of all
dimensions, along with the metric and topology.  Provides methods for
building the initial triangulation and manipulating the complex.

Typical construction::

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(500))doc")
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
           py::arg("topology"),
           R"doc(Construct a spacetime with the given metric, type, and topology.

Args:
    metric: The Metric object defining edge lengths.
    spacetimeType: CDT, REGGE, or COSET.
    alpha: Wick rotation parameter (1.0 for Lorentzian CDT).
    a: Lattice spacing parameter.
    foliation: PREFERRED for CDT (fixed time slicing).
    topology: Spatial topology (Toroid or Sphere).)doc"
      )
      .def(py::init<>(), "Create an empty spacetime with default 4D Lorentzian CDT settings.")
      .def("getVertexList", &Spacetime::getVertexList,
           "Return the VertexList containing all vertices.")
      .def("getSimplicesWithOrientation",
           &Spacetime::getSimplicesWithOrientation,
           py::arg("orientation"),
           py::return_value_policy::reference,
           "Return all top simplices with the given CDT orientation.")
      .def("getEdgeList", &Spacetime::getEdgeList,
           "Return the EdgeList containing all edges.")
      .def("getConnectedComponents", &Spacetime::getConnectedComponents, py::return_value_policy::reference,
           "Return the connected components of the simplicial complex.")
      .def("getDualAdjacency", &Spacetime::getDualAdjacency,
           R"doc(Return the dual graph of the top-dimensional triangulation as COO arrays.

Two top simplices are adjacent when they share a (d-1)-face.  Returns
(rows, cols, N) where rows[k] and cols[k] are 0-based indices into the
internal top-simplex array and N is the number of top simplices.

This is much faster than iterating over simplices/facets/cofaces from
Python because it makes a single C++ call instead of O(N) round trips.)doc")
      .def("getTimeSlices", &Spacetime::getTimeSlices,
           "Return sorted list of integer time values in the triangulation.")
      .def("getVerticesAtTime", &Spacetime::getVerticesAtTime,
           py::arg("t"), py::return_value_policy::reference,
           "Return all vertices at integer time t.")
      .def("getSpatialSubgraph", &Spacetime::getSpatialSubgraph,
           py::arg("t"), py::return_value_policy::reference,
           "Return (vertices, spacelike_edges) at time t.")
      .def("bfsDistances", &Spacetime::bfsDistances,
           py::arg("center"), py::arg("maxDepth") = -1,
           "BFS distances from center through spacelike edges.  Returns {id: dist}.")
      .def("build", &Spacetime::build, py::arg("numSimplices") = 3, py::call_guard<py::gil_scoped_release>(),
           R"doc(Build the initial triangulation with approximately n_simplices top simplices.

Uses the topology's builder (e.g. Toroid staircase triangulation) to
create the initial simplicial complex.  The actual number of simplices
may differ slightly due to slab quantization.)doc")
      .def("getSimplices", &Spacetime::getSimplices, py::return_value_policy::copy,
           "Return all top-dimensional simplices in the complex.")
      .def("getExternalSimplices", &Spacetime::getExternalSimplices, py::return_value_policy::copy,
           "Return simplices on the boundary of the complex.")
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::return_value_policy::reference,
           "Create an edge between two vertices (squared length computed from metric).")
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &, double) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::arg("squaredLength"),
           py::return_value_policy::reference,
           "Create an edge between two vertices with a specified squared length.")
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t) const noexcept>(
             &Spacetime::createVertex),
           py::arg("id"),
           py::return_value_policy::reference,
           "Create a vertex with the given ID (auto-assigned coordinates).")
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t, const std::vector<double> &) const noexcept>(
             &Spacetime::createVertex),
           py::arg("id"), py::arg("coordinates"),
           py::return_value_policy::reference,
           "Create a vertex with the given ID and coordinates (first = time).")
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::return_value_policy::reference,
           R"doc(Create a simplex from the given vertices.

Returns (simplex, created) where created is True if the simplex was
new, False if it already existed (dedup by fingerprint).)doc")
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &, const std::vector<EdgePtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::arg("edges"),
           py::return_value_policy::reference,
           "Create a simplex from the given vertices and edges.")
      .def("createSimplex",
           py::overload_cast<const std::tuple<uint8_t, uint8_t> &>(&Spacetime::createSimplex),
           py::arg("orientation"),
           py::return_value_policy::reference,
           R"doc(Create a seed simplex with the given orientation.

Automatically creates vertices at times 0 and 1.  Useful for building
minimal test lattices, e.g. createSimplex((1, 4)) for a (1,4) simplex.)doc")
      .def("getSimplexCount", &Spacetime::getSimplexCount,
           "Return N4 = N41 + N32, the total number of top-dimensional simplices.")
      .def("getVertexCount", &Spacetime::getVertexCount,
           "Return N0, the total number of vertices.")
      .def("getN41", &Spacetime::getN41,
           R"doc(Return N41: the count of (d,1) + (1,d) type simplices.

This is the volume-fixing target per [RU] eq. 6.)doc")
      .def("getN32", &Spacetime::getN32,
           "Return N32: the count of (d-1,2) + (2,d-1) type simplices.")
      .def("getRandomSimplex", &Spacetime::getRandomSimplex, py::return_value_policy::reference,
           "Return a uniformly random simplex from the complex (any dimension).")
      .def("getRandomTopSimplex", &Spacetime::getRandomTopSimplex, py::return_value_policy::reference,
           "Return a uniformly random top-dimensional simplex.")
      .def("getRandomVertex", &Spacetime::getRandomVertex, py::return_value_policy::reference,
           "Return a uniformly random vertex.")
      .def("removeSimplex", &Spacetime::removeSimplex, py::arg("simplex"),
           "Remove a top-dimensional simplex from the complex.")
      .def("swapVertexLabels", &Spacetime::swapVertexLabels, py::arg("v1"), py::arg("v2"),
           R"doc(Swap the integer IDs of two vertices ([BGL] Sec. 2.2.1).

Atomically re-keys all dependent data structures: VertexList,
Edge fingerprints, Simplex vertex-ID maps, and all hash tables.
Handles shared simplices, transient fingerprint collisions, and
sub-simplex ownership correctly.

No-op if v1 and v2 are the same vertex.)doc")
      .def("save", [](const Spacetime &st, const std::string &path,
                       int panelSize, int layoutIters,
                       double tilt, int spin, int precession,
                       int nFrames, int delayCentiseconds) {
          renderSpacetime(st, path, panelSize, layoutIters,
                          tilt, spin, precession, nFrames, delayCentiseconds);
      }, py::arg("path"), py::arg("panel_size") = 800,
         py::arg("layout_iters") = 500,
         py::arg("tilt") = 25.0,
         py::arg("spin") = 1,
         py::arg("precession") = 1,
         py::arg("n_frames") = 36,
         py::arg("delay_cs") = 15,
           R"doc(Render the spacetime to an image file.

Uses a force-directed layout (time fixed, spatial coordinates
optimized via spring + repulsion) then projects onto 2D.

If the path ends in .gif, produces an animated GIF whose rotation
is controlled by three parameters:

    tilt        – cone half-angle in degrees (default 25).
    spin        – full Y-axis rotations per loop (default 1).
    precession  – precession cycles per loop (default 1).

Per-frame rotation:
    ry = 2π · spin · t
    rx = tilt · cos(2π · precession · t)
    rz = tilt · sin(2π · precession · t)

Integer values for spin and precession guarantee a perfect loop.

For .graphml or .dot/.gv, exports the graph structure with vertex
attributes (id, time, degree) and edge attributes (squared_length,
timelike).  Layout/rotation parameters are ignored for these formats.

Otherwise produces a static PNG with four panels
(no rotation, 40° X, 40° Y, 40° Z).

The layout is computed internally and does not modify vertex state.

Args:
    path: Output file path (.png, .gif, .graphml, .dot, or .gv).
    panel_size: Pixel size of each panel (default 800).
    layout_iters: Maximum force-directed iterations (default 500).
    tilt: Precession cone half-angle in degrees (default 25).
    spin: Y-axis rotations per loop, integer for perfect loop (default 1).
    precession: Precession cycles per loop, integer for perfect loop (default 1).
    n_frames: Number of GIF frames (default 36).
    delay_cs: Frame delay in centiseconds (default 7).)doc");

  // ========================================
  // CDTSimulation
  // ========================================
  py::class_<CDT, std::shared_ptr<CDT> >(m, "CDTSimulation",
      R"doc(Causal Dynamical Triangulations Monte Carlo simulation.

Implements the five Pachner moves (add, remove, flip, iflip, shift) with
Metropolis-Hastings acceptance including combinatorial prefactors per
[BGL] eq. 11, 26, 27.

The Regge action is ([RU] eq. 2)::

    S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41
        + (k4 + delta)*N32 + epsilon*(N41 - target)^2

Args:
    spacetime: A built Spacetime object.
    k0: Bare inverse Newton's constant.
    k4: Cosmological constant coupling (tuned to pseudo-critical value).
    delta: Asymmetry parameter between timelike and spacelike edges.
    epsilon: Volume-fixing strength.
    targetN41: Target (d,1)-type four-volume for volume-fixing ([RU] eq. 6).
    quadraticVolumeFix: If True (default), use epsilon*(N41 - target)^2;
        if False, use epsilon*|N41 - target| ([RU] eq. 6).)doc")
      .def(py::init<std::shared_ptr<Spacetime>, double, double, double, double, std::size_t, bool>(),
           py::arg("spacetime"),
           py::arg("k0"),
           py::arg("k4"),
           py::arg("delta"),
           py::arg("epsilon"),
           py::arg("targetN41"),
           py::arg("quadraticVolumeFix") = true)
      .def("add", &CDT::add,
           R"doc(Attempt one (2,2d) vertex insertion move ([BGL] Sec. 2.3.1).

Picks a random N41 simplex, finds a spatial face with a partner, and
proposes inserting a new vertex.  Accepted via Metropolis with prefactor
N41/(N0+1).  On acceptance, the new vertex is relabeled uniformly.

Returns True if accepted, False if rejected.)doc")
      .def("remove", &CDT::remove,
           R"doc(Attempt one (2d,2) vertex deletion move ([BGL] Sec. 2.3.1).

Picks a random vertex via blind guessing, checks if it has order 2d
(all N41-type), and proposes removing it.  Inverse of add().

Returns True if accepted, False if rejected.)doc")
      .def("flip", &CDT::flip,
           R"doc(Attempt one (2,d) flip move ([BGL] Sec. 2.3.2).

Picks a random top simplex, picks a random facet, and proposes
replacing the 2 simplices sharing that facet with d new simplices.
dN0 = 0, dN4 = d - 2 = +2 in 4D.

Returns True if accepted, False if rejected.)doc")
      .def("iflip", &CDT::iflip,
           R"doc(Attempt one (d,2) inverse flip move ([BGL] Sec. 2.3.2).

Picks a random top simplex, picks a random edge, and proposes
replacing the d simplices sharing that edge with 2 new simplices.
dN0 = 0, dN4 = -(d - 2) = -2 in 4D.  Inverse of flip().

Returns True if accepted, False if rejected.)doc")
      .def("shift", &CDT::shift,
           R"doc(Attempt one (3,3) shift move ([BGL] Sec. 2.3.3).

Replaces 3 simplices sharing a (d-2)-face with 3 new simplices
sharing the complementary (d-2)-face.  Self-inverse: dN0 = 0, dN4 = 0.
Combinatorial prefactor is 1 (symmetric selection).

Returns True if accepted, False if rejected.)doc")
      .def("ishift", &CDT::ishift,
           R"doc(Attempt one inverse shift move (same as shift, since (3,3) is self-inverse).

Returns True if accepted, False if rejected.)doc")
      .def("sweep", [](CDT &self, int n_sweeps, py::object progress) {
          int total = 0;
          if (progress.is_none()) {
              // No callback — release the GIL for the entire loop so
              // multiple threads get true parallelism.
              py::gil_scoped_release release;
              for (int i = 0; i < n_sweeps; i++) {
                  total += self.sweep();
              }
          } else {
              for (int i = 0; i < n_sweeps; i++) {
                  int accepted;
                  {
                      py::gil_scoped_release release;
                      accepted = self.sweep();
                  }
                  total += accepted;
                  progress(i + 1, n_sweeps);
              }
          }
          return total;
      }, py::arg("n_sweeps") = 1, py::arg("progress") = py::none(),
           R"doc(Run one or more Monte Carlo sweeps.

Each sweep proposes N4 moves uniformly among all 5 types
(add, remove, flip, iflip, shift).

Args:
    n_sweeps: Number of sweeps to perform (default 1).
    progress: Optional callback(i, n) called after each sweep.

Returns the total number of accepted moves across all sweeps.)doc")
      .def("tune", &CDT::tune, py::call_guard<py::gil_scoped_release>(),
           R"doc(Tune k4 to its pseudo-critical value ([BGL] Sec. 3.3.1).

Computes an initial estimate of k4 from the coupling constants, then
performs 20 feedback sweeps adjusting k4 to drive N41 toward the target.
Call this before sweep() for stable simulations.)doc")
      .def("thermalize", &CDT::thermalize, py::call_guard<py::gil_scoped_release>(),
           R"doc(Thermalize the simulation until the action stabilizes.

Runs sweeps until the relative change in action is < 1%, with a minimum
of 20 sweeps.  Use after tune() to reach thermal equilibrium before
taking measurements.)doc")
      .def("computeAction", &CDT::computeAction,
           R"doc(Compute the Regge action S from current counts ([RU] eq. 2).

S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41
    + (k4 + delta)*N32 + volume_fix_term)doc")
      .def("getVolumeProfile", &CDT::getVolumeProfile, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return the spatial volume profile as a list of simplex counts per time slice.

Each entry is the number of top simplices whose minimum vertex time
equals that slice.  The sum equals N4.)doc")
      .def("getAcceptanceRates", &CDT::getAcceptanceRates, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return acceptance rates for each move type as a dict.

Keys: 'add', 'remove', 'flip', 'iflip', 'shift', 'ishift'.
Values: fraction of attempts accepted (0.0 to 1.0).)doc")
      .def("getSpacetime", &CDT::getSpacetime,
           "Return the underlying Spacetime object.")
      .def("getK0", &CDT::getK0,
           "Return the bare inverse Newton's constant k0.")
      .def("getK4", &CDT::getK4,
           "Return the current cosmological coupling k4 (may differ from initial after tune).")
      .def("getDelta", &CDT::getDelta,
           "Return the asymmetry parameter delta.")
      .def("setRelabelVertices", &CDT::setRelabelVertices, py::arg("enabled"),
           R"doc(Enable or disable vertex relabeling after add/remove moves.

Enabled by default per [BGL] Sec. 2.2.1.  Disable for deterministic
tests that compare simplex fingerprints before and after moves.)doc");

  // ========================================
  // VolumeProfile
  // ========================================
  py::class_<VolumeProfile, std::shared_ptr<VolumeProfile> >(m, "VolumeProfile",
      R"doc(Observable measuring the spatial volume profile N(t).

Counts top simplices per time slice.  Can accumulate measurements over
multiple configurations for averaging.)doc")
      .def(py::init<>())
      .def("compute", &VolumeProfile::compute, py::arg("spacetime"),
           "Compute the volume profile for the current configuration.")
      .def("getProfile", &VolumeProfile::getProfile,
           "Return the most recent volume profile as a list.")
      .def("getAverageProfile", &VolumeProfile::getAverageProfile,
           "Return the time-averaged volume profile (over all measure() calls).")
      .def("measure", &VolumeProfile::measure, py::arg("spacetime"),
           "Compute and accumulate a volume profile measurement for averaging.")
      .def("reset", &VolumeProfile::reset,
           "Reset the accumulated measurements.");

  // ========================================
  // WilsonLoop
  // ========================================
  py::enum_<WilsonMode>(m, "WilsonMode",
      "Evaluation mode for Wilson loops.")
      .value("COMBINATORIAL", WilsonMode::COMBINATORIAL,
             "Dual-graph topology only (loop length, enclosed hinges).")
      .value("DEFICIT_ANGLE", WilsonMode::DEFICIT_ANGLE,
             "Deficit-angle based: W = ((d-2)+2cos(epsilon))/d.")
      .value("CAUSAL", WilsonMode::CAUSAL,
             "CDT causal orientation changes around the loop.");

  py::enum_<LoopType>(m, "LoopType",
      "Which loop-shape generator to use.")
      .value("HINGE", LoopType::HINGE,
             "Elementary loop around a (d-2)-simplex.")
      .value("DUAL_LATTICE", LoopType::DUAL_LATTICE,
             "BFS-discovered loop of a target size.")
      .value("GEODESIC", LoopType::GEODESIC,
             "Shortest cycle through a start simplex.");

  py::class_<LoopPath>(m, "LoopPath",
      "A closed path through the dual graph (sequence of top-simplices).")
      .def_readonly("simplices", &LoopPath::simplices,
                    py::return_value_policy::reference)
      .def_readonly("facets", &LoopPath::facets,
                    py::return_value_policy::reference)
      .def("__len__", [](const LoopPath &lp) { return lp.simplices.size(); });

  py::class_<WilsonResult>(m, "WilsonResult",
      "Result of evaluating a Wilson loop.")
      .def_readonly("value", &WilsonResult::value,
                    "Primary scalar value.")
      .def_readonly("loopSize", &WilsonResult::loopSize,
                    "Number of simplices in the loop.")
      .def_readonly("enclosedHinges", &WilsonResult::enclosedHinges,
                    "Hinges enclosed by the loop.")
      .def_readonly("contractible", &WilsonResult::contractible,
                    "Whether the loop is contractible.")
      .def_readonly("causalWindingNumber", &WilsonResult::causalWindingNumber,
                    "Net time-orientation changes (causal mode).");

  py::class_<WilsonLoop, std::shared_ptr<WilsonLoop>>(m, "WilsonLoop",
      R"doc(Wilson loop observable on a triangulated spacetime.

Computes holonomy-like quantities around closed paths in the dual graph.
Three evaluation modes at increasing levels of geometric commitment:
  COMBINATORIAL -- dual-graph topology only
  DEFICIT_ANGLE -- curvature via deficit angles
  CAUSAL        -- CDT causal structure

Three loop-shape generators:
  hingeLoop()       -- elementary loop around a hinge
  dualLatticeLoop() -- BFS-discovered loop of target size
  geodesicLoop()    -- shortest cycle through a simplex)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"))
      .def("evaluate", &WilsonLoop::evaluate,
           py::arg("loop"), py::arg("mode"),
           "Evaluate the Wilson loop in the given mode.")
      .def("evaluateCombinatorial", &WilsonLoop::evaluateCombinatorial,
           py::arg("loop"),
           "Evaluate using dual-graph topology only.")
      .def("evaluateDeficitAngle", &WilsonLoop::evaluateDeficitAngle,
           py::arg("loop"),
           "Evaluate using deficit angles.")
      .def("evaluateCausal", &WilsonLoop::evaluateCausal,
           py::arg("loop"),
           "Evaluate using CDT causal orientation changes.")
      .def("hingeLoop", &WilsonLoop::hingeLoop,
           py::arg("hinge"),
           "Generate the loop of top-simplices around a hinge.")
      .def("dualLatticeLoop", &WilsonLoop::dualLatticeLoop,
           py::arg("start"), py::arg("targetLength"),
           "BFS-discovered loop of approximately targetLength simplices.")
      .def("geodesicLoop", &WilsonLoop::geodesicLoop,
           py::arg("start"),
           "Shortest cycle through start in the dual graph.")
      .def("measure", &WilsonLoop::measure,
           py::arg("loop"), py::arg("mode"),
           "Evaluate and record a measurement.")
      .def("measureAllHinges", &WilsonLoop::measureAllHinges,
           py::arg("mode"),
           "Measure all hinge loops in the spacetime.")
      .def("reset", &WilsonLoop::reset,
           "Clear accumulated measurements.")
      .def("getMeasurements", &WilsonLoop::getMeasurements,
           "Return all recorded WilsonResult objects.")
      .def("getAverageBySize", &WilsonLoop::getAverageBySize,
           "Return average Wilson value for each loop size.");

  // ========================================
  // MatterConfiguration
  // ========================================
  py::enum_<HingeType>(m, "HingeType")
      .value("SPATIAL", HingeType::SPATIAL)
      .value("TIMELIKE", HingeType::TIMELIKE);

  py::class_<MatterConfiguration>(m, "MatterConfiguration",
      R"doc(Intrinsic (coordinate-free) specification of stress-energy on a triangulation.

Matter is defined relationally: by assigning energy densities to vertices,
simplices, or as a function of geodesic distance from a reference vertex.

For point particles, the matter action is the proper-time action:
S_matter = -M Σ √(-ℓ²) along the worldline.)doc")
      .def(py::init<>())
      .def("setWorldlineMass", &MatterConfiguration::setWorldlineMass,
           py::arg("center"), py::arg("mass"), py::arg("spacetime"),
           R"doc(Assign a static point mass along its worldline through all time slices.

Traces a worldline from center through the foliation by following
timelike edges.  The matter action is the proper-time action:
S_matter = -M Σ √(-ℓ²) along the worldline.

Args:
    center: A vertex on the worldline (any time slice).
    mass: The mass in geometrized units (G=c=1).
    spacetime: The spacetime to trace through.)doc")
      .def("setEnergyDensity", &MatterConfiguration::setEnergyDensity,
           py::arg("simplex"), py::arg("rho"),
           R"doc(Assign energy density to a top-simplex.

Args:
    simplex: The simplex to assign density to.
    rho: Energy density in geometrized units.)doc")
      .def("setRadialProfile", &MatterConfiguration::setRadialProfile,
           py::arg("center"), py::arg("rho_of_r"),
           R"doc(Assign energy density as a function of geodesic distance.

Args:
    center: The reference vertex.
    rho_of_r: A callable taking distance (float) and returning density (float).)doc")
      .def_static("buildWorldline", &MatterConfiguration::buildWorldline,
           py::arg("center"), py::arg("spacetime"),
           py::return_value_policy::copy,
           R"doc(Trace a worldline from center through all time slices.

Returns a list of vertices, one per time slice, ordered by time.)doc")
      .def_static("classifyHinge", &MatterConfiguration::classifyHinge,
           py::arg("hinge"),
           R"doc(Classify a hinge as SPATIAL (all vertices at one time) or TIMELIKE.)doc");

  // ========================================
  // ReggeSolver
  // ========================================
  py::class_<ReggeSolver>(m, "ReggeSolver",
      R"doc(Regge equation solver.

Adjusts edge lengths so that the Regge equations (∂S/∂ℓ² = 0) are satisfied.
The total action is S = S_grav + S_matter where:
  S_grav = Σ_h A_h ε_h   (Regge gravitational action)
  S_matter = -M Σ √(-ℓ²)  (proper-time action along worldlines)

Minimizes F = ||∇S||² to find stationary points of S (the discrete
Einstein equations).  F ≥ 0, and F = 0 at the solution.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, MatterConfiguration>(),
           py::arg("spacetime"), py::arg("matter"))
      .def("dihedralAngle", &ReggeSolver::dihedralAngle,
           py::arg("sigma"), py::arg("hinge"),
           "Dihedral angle at hinge h within top-simplex sigma.")
      .def("deficitAngle", &ReggeSolver::deficitAngle,
           py::arg("hinge"),
           "Deficit angle at a hinge: 2π minus sum of dihedral angles.")
      .def_static("hingeArea", &ReggeSolver::hingeArea,
           py::arg("hinge"),
           "Area of a triangular hinge (Heron's formula).")
      .def("reggeAction", &ReggeSolver::reggeAction,
           "Gravitational Regge action: S_grav = Σ_h A_h · ε_h.")
      .def("matterAction", &ReggeSolver::matterAction,
           "Point-particle matter action: S_matter = -M Σ √(-ℓ²) along worldlines.")
      .def("totalAction", &ReggeSolver::totalAction,
           "Total action: S = S_grav + S_matter.  Stationary point = Einstein eqs.")
      .def("actionGradientNorm", &ReggeSolver::actionGradientNorm,
           "||∇S||² = Σ_e (∂S/∂ℓ²_e)².  Zero = Regge equations solved.")
      .def("step", &ReggeSolver::step,
           py::arg("learning_rate") = 0.001,
           "One gradient-descent step on F = ||∇S||². Returns F before the update.")
      .def("solve", [](ReggeSolver &self, double tol, int maxIters,
                        double learningRate, py::object progress) {
          ReggeSolver::ProgressCallback cb = nullptr;
          if (!progress.is_none()) {
              cb = [&progress](int iter, double F) {
                  py::gil_scoped_acquire acquire;
                  progress(iter, F);
              };
          }
          py::gil_scoped_release release;
          return self.solve(tol, maxIters, learningRate, cb);
      },
           py::arg("tol") = 1e-8,
           py::arg("max_iters") = 5000,
           py::arg("learning_rate") = 0.001,
           py::arg("progress") = py::none(),
           R"doc(Find stationary point of the total Regge action (discrete Einstein eqs).

Minimizes F = ||∇S||² until F < tol or max_iters reached.

Args:
    tol: Convergence tolerance on F = ||∇S||².
    max_iters: Maximum number of iterations.
    learning_rate: Gradient descent step size.
    progress: Optional callback(iter, F) called after each iteration.

Returns:
    Tuple of (converged: bool, final_F: float, iterations: int).)doc");

  // ========================================
  // ForceLayout
  // ========================================
  m.def("forceLayout3D", &forceLayout3D,
        py::arg("n"),
        py::arg("edges"),
        py::arg("centerIdx") = -1,
        py::arg("initPos") = std::vector<double>{},
        py::arg("restLengths") = std::vector<double>{},
        py::arg("springK") = 0.01,
        py::arg("repulsionK") = 0.5,
        py::arg("iters") = 300,
        py::arg("cooling") = 0.995,
        py::arg("repulsionCap") = 200,
        py::arg("seed") = 42,
        R"doc(Spring-electrical force-directed layout in 3D.

Returns a flat list of n*3 floats (row-major x,y,z positions).
Reshape to (n, 3) with numpy: ``np.array(result).reshape(n, 3)``.

Args:
    n: Number of nodes.
    edges: List of (i, j) index pairs.
    centerIdx: Pin this node at the origin (-1 to disable).
    initPos: Flat initial positions (length n*3). Random if empty.
    restLengths: Per-edge rest lengths. Unit if empty.
    springK: Spring constant (default 0.01).
    repulsionK: Coulomb constant (default 0.5).
    iters: Number of iterations (default 300).
    cooling: Step-size decay per iteration (default 0.995).
    repulsionCap: Max nodes for O(n^2) repulsion (default 200).
    seed: Random seed (default 42).)doc");

#ifdef CASET_VERSION
  m.attr("__version__") = CASET_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif
}
