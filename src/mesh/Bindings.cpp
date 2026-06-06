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
#include "spacetime/PachnerMove.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include "simulations/ReggeSolver.h"
#include "matter/MatterConfiguration.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/SparseGraph.h"
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
using namespace tessera;
using namespace tessera::mesh;

// Registers all tessera::mesh classes into the `m` submodule
// (i.e. `tessera.mesh`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_mesh(py::module_ m) {
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
      .def("getPhase", &Edge::getPhase,
           R"doc(Return the U(1) connection phase carried by this edge (radians).

Paired with the signed squared length it gives the complex edge weight
squaredLength * exp(i * phase) used by the Hermitian-weighted Laplacian.
The default of 0 leaves an ordinary real-weighted CDT edge unchanged.)doc")
      .def("setPhase", &Edge::setPhase, py::arg("phase"),
           "Set the U(1) connection phase carried by this edge (radians).")
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
      .def("tryAdd", &EdgeList::tryAdd,
           py::arg("source"), py::arg("target"), py::arg("squaredLength"),
           py::return_value_policy::reference,
           R"doc(Insert if absent, otherwise return the existing edge.

Returns ``(edge, inserted)`` where ``inserted`` is ``True`` on a fresh
insert and ``False`` on a dedupe-hit.  Used by transactional Pachner
moves to record which edges they freshly created (so rollback knows
which to remove).)doc")
      .def("remove", py::overload_cast<const EdgePtr &>(&EdgeList::remove), py::arg("edge"),
           "Remove an edge from the list.")
      .def("size", &EdgeList::size,
           "Return the total number of edges.")
      .def("toVector", &EdgeList::toVector, py::return_value_policy::reference,
           "Return all edges as a list.");
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
      .def("getVertices", &Simplex::getVertices, py::return_value_policy::reference_internal,
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
  // SimplexFilter (predicate over top simplices)
  // ========================================
  //
  // Held as ``std::shared_ptr<SimplexFilter>`` so the same object can
  // be referenced from HolographyConfig.simplexFilter and survive
  // copy-by-value of the config. Python subclassing is not supported
  // in this build — extend via C++ subclass + binding instead.
  py::class_<SimplexFilter, std::shared_ptr<SimplexFilter>>(m, "SimplexFilter",
      R"doc(Predicate over top simplices (abstract base).

Selects which top simplices participate in a downstream observable
(``Spacetime.getSpectralDimensionOnSkeleton``). Use one of the
concrete subclasses below; in this build, custom filters require a
C++ subclass + binding.)doc")
      .def("accept", &SimplexFilter::accept, py::arg("simplex"),
           "Return True iff the simplex should participate in the "
           "downstream observable.")
      .def("name", &SimplexFilter::name,
           "Human-readable name; appears in JSON output for "
           "reproducibility.")
      .def("__repr__", [](SimplexFilter const& self) {
        return "<" + self.name() + ">";
      });
  py::class_<AllSimplexFilter, SimplexFilter,
             std::shared_ptr<AllSimplexFilter>>(m, "AllSimplexFilter",
      R"doc(Accepts every top simplex.

Default for the holographic-dual measurement (issue #31). Registration
via ``Spacetime.createSimplex`` already implies combinatorial
constructibility (the ``k + 1`` vertices form a complete subgraph in
the edge set), so this filter intentionally ignores edge-length
geometry.)doc")
      .def(py::init<>());
  py::class_<PositiveGramDeterminantFilter, SimplexFilter,
             std::shared_ptr<PositiveGramDeterminantFilter>>(m, "PositiveGramDeterminantFilter",
      R"doc(Accepts simplices whose Gram matrix has positive determinant.

Restricts the measurement to metrically valid (non-degenerate,
non-collapsed) Euclidean cells. Stricter alternative to the default
``AllSimplexFilter``.)doc")
      .def(py::init<>());
}
