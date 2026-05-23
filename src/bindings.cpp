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

#ifdef TESSERA_QUANTUM
// Defined in src/quantum/bindings.cpp. Conditionally compiled when the
// TESSERA_QUANTUM CMake option is on — see CMakeLists.txt for the wiring.
void register_quantum_bindings(py::module_ m);
#endif

PYBIND11_MODULE(_tessera, m) {
  m.doc() = R"doc(
tessera -- Causal Set and CDT simulation library.

A C++ library (with Python bindings) for Causal Dynamical Triangulations
(CDT) and causal set theory simulations in arbitrary dimension.

Typical usage::

    import tessera

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(500)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
    cdt.tune()
    cdt.sweep(100)

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
)doc";

  // ========================================
  // Subsystem submodules — 1:1 with C++ namespaces
  // ========================================
  //
  // Each subsystem's classes are registered in its own Python submodule
  // (`tessera.mesh`, `tessera.spacetime`, etc.), mirroring the C++ namespace
  // layout (`tessera::mesh::`, `tessera::spacetime::`, ...). The classes
  // are re-exported at the top-level `tessera` module via
  // `tessera/__init__.py` for backward compatibility, so `tessera.Spacetime`
  // and `tessera.spacetime.Spacetime` both work.
  auto m_mesh        = m.def_submodule("mesh",
      "Vertex / Edge / Simplex primitives, SimplexFilter, and ID typedefs.");
  auto m_spacetime   = m.def_submodule("spacetime",
      "Spacetime simplicial complex, Metric, Signature, Foliation, topologies, Pachner moves.");
  auto m_observables = m.def_submodule("observables",
      "Observables on a Spacetime: SparseGraph, ModularityOptimizer, "
      "WilsonLoop, VolumeProfile.");
  auto m_simulations = m.def_submodule("simulations",
      "Monte Carlo simulations: CDT, ReggeSolver, Simulation base class.");

  // ========================================
  // Edge
  // ========================================
  py::class_<Edge, std::unique_ptr<Edge, py::nodelete>>(m_mesh, "Edge",
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
  py::class_<Vertex, std::unique_ptr<Vertex, py::nodelete>>(m_mesh, "Vertex",
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
  py::class_<VertexList, std::shared_ptr<VertexList> >(m_mesh, "VertexList",
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
  py::class_<EdgeList, std::shared_ptr<EdgeList> >(m_mesh, "EdgeList",
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
  // Topologies
  // ========================================
  py::class_<Topology, std::shared_ptr<Topology> >(m_spacetime, "Topology",
      "Base class for spatial topologies (Toroid, Sphere, etc.).");

  py::class_<Sphere, Topology, std::shared_ptr<Sphere> >(m_spacetime, "Sphere",
      "Spherical spatial topology S^{d-1}.")
      .def(py::init<>())
      .def("build", &Sphere::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a spherical initial triangulation with the given number of simplices.");

  py::class_<Cylinder, Topology, std::shared_ptr<Cylinder> >(m_spacetime, "Cylinder",
      "Cylindrical spatial topology with open time boundaries.")
      .def(py::init<>())
      .def("build", &Cylinder::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a cylindrical triangulation with the given number of simplices.");

  py::class_<Toroid, Topology, std::shared_ptr<Toroid> >(m_spacetime, "Toroid",
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
  py::class_<SimplexOrientation, std::shared_ptr<SimplexOrientation> >(m_mesh, "SimplexOrientation",
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

  py::class_<SimplexOrientationHash, std::shared_ptr<SimplexOrientationHash> >(m_mesh, "SimplexOrientationHash")
      .def(py::init<>());
  py::class_<SimplexOrientationEq, std::shared_ptr<SimplexOrientationEq> >(m_mesh, "SimplexOrientationEq")
      .def(py::init<>());

  // ========================================
  // Simplex
  // ========================================
  py::class_<Simplex, std::unique_ptr<Simplex, py::nodelete>>(m_mesh, "Simplex",
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

  py::class_<SimplexHash, std::shared_ptr<SimplexHash> >(m_mesh, "SimplexHash")
      .def(py::init<>());
  py::class_<SimplexEq, std::shared_ptr<SimplexEq> >(m_mesh, "SimplexEq")
      .def(py::init<>());

  // ========================================
  // Metric
  // ========================================
  py::class_<Metric, std::shared_ptr<Metric> >(m_spacetime, "Metric",
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
  py::enum_<SignatureType>(m_spacetime, "SignatureType",
      "Metric signature: Lorentzian (-,+,+,...) or Euclidean (+,+,+,...).")
      .value("Lorentzian", SignatureType::Lorentzian)
      .value("Euclidean", SignatureType::Euclidean)
      .export_values();

  py::enum_<Foliation>(m_spacetime, "Foliation",
      "Time foliation type for CDT spacetimes.")
      .value("PREFERRED", Foliation::PREFERRED)
      .value("NONE", Foliation::NONE)
      .export_values();

  // ========================================
  // Signature
  // ========================================
  py::class_<Signature, std::shared_ptr<Signature> >(m_spacetime, "Signature",
      R"doc(Metric signature specifying dimension and type.

Args:
    dimensions: Number of spacetime dimensions d (e.g. 4 for 4D CDT).
    signatureType: Lorentzian or Euclidean.)doc")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signatureType"))
      .def("getDiagonal", &Signature::getDiagonal,
           "Return the diagonal entries of the metric signature tensor.");

  py::enum_<SpacetimeType>(m_spacetime, "SpacetimeType",
      "Type of spacetime simulation.")
      .value("CDT", SpacetimeType::CDT)
      .value("REGGE", SpacetimeType::REGGE)
      .value("COSET", SpacetimeType::COSET)
      .export_values();

  // ========================================
  // Spacetime
  // ========================================
  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m_spacetime, "Spacetime",
      R"doc(The simplicial spacetime manifold.

Holds the full simplicial complex: vertices, edges, and simplices of all
dimensions, along with the metric and topology.  Provides methods for
building the initial triangulation and manipulating the complex.

Typical construction::

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
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
      .def("setSeed", &Spacetime::setSeed, py::arg("seed"),
           R"doc(Seed the spacetime's internal RNG deterministically.

The RNG drives ``getRandomVertex`` / ``getRandomSimplex`` /
``getRandomTopSimplex`` — i.e. the first-step sigma selection in
every Pachner move (ShiftMove, FlipMove, IFlipMove, AddMove,
RemoveMove). Use this in tests that need byte-identical reproducibility
across processes; otherwise the default seed comes from
``std::random_device`` and varies per construction.)doc")
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
      .def("getDualGraph", &Spacetime::getDualGraph,
           R"doc(Return the dual graph as a SparseGraph.

Equivalent to ``SparseGraph::fromCOO(*getDualAdjacency())`` but
avoids the intermediate Python conversion.)doc")
      .def("modularityOnSkeleton", &Spacetime::modularityOnSkeleton,
           py::arg("M"),
           R"doc(Newman-Girvan modularity Q on the vertex/edge 1-skeleton.

Implicit labels: ``label(v) = v.id() % M``.  Returns 0 if M < 2,
the graph has no edges, or the spacetime has no vertices.)doc")
      .def("getSpectralDimensionOnSkeleton",
           &Spacetime::getSpectralDimensionOnSkeleton,
           py::arg("sigmas"), py::arg("krylovDim"),
           py::arg("filter"), py::arg("topK") = 4,
           py::arg("skeletonDim") = 1,
           R"doc(D_S(σ) on the weighted 1-skeleton of top simplices that
pass ``filter``.

Walks the simplex list keeping those with ``size() == topK + 1`` for
which ``filter.accept(s)`` is true, unions their edges with weights
``w_uv = I_max · exp(-sqrt(|squaredLength_uv|))``, builds the
unnormalised weighted Laplacian ``L = D - W``, and returns
``SpectralGraph.spectralDimension`` of the heat-kernel return
probability. Sits next to ``modularityOnSkeleton``.

``skeletonDim`` reserves API space for higher-k skeletons; only
``skeletonDim == 1`` is currently supported.)doc")
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
      .def("createSimplexTracked",
           [](Spacetime &self, const std::vector<VertexPtr> &vertices) {
               auto r = self.createSimplexTracked(vertices);
               return py::make_tuple(r.simplex, r.created, r.newEdges);
           },
           py::arg("vertices"),
           py::return_value_policy::reference,
           R"doc(Like createSimplex(vertices) but also returns the edges
that this call freshly inserted into the EdgeList.

Returns (simplex, created, newEdges) where:
  - simplex: SimplexPtr (existing or freshly created)
  - created: True if newly created, False if found existing
  - newEdges: list of edges this call freshly inserted (empty when
              created=False, or when all needed edges already existed)

Used by transactional Pachner moves so rollback can undo edge
insertions.)doc")
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
      }, py::arg("path"), py::arg("panelSize") = 800,
         py::arg("layoutIters") = 500,
         py::arg("tilt") = 25.0,
         py::arg("spin") = 1,
         py::arg("precession") = 1,
         py::arg("nFrames") = 36,
         py::arg("delayCs") = 15,
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
    panelSize: Pixel size of each panel (default 800).
    layoutIters: Maximum force-directed iterations (default 500).
    tilt: Precession cone half-angle in degrees (default 25).
    spin: Y-axis rotations per loop, integer for perfect loop (default 1).
    precession: Precession cycles per loop, integer for perfect loop (default 1).
    nFrames: Number of GIF frames (default 36).
    delayCs: Frame delay in centiseconds (default 7).)doc");

  // ========================================
  // PachnerMove (transactional Pachner moves)
  // ========================================
  py::class_<PachnerMove>(m_spacetime, "PachnerMove",
      R"doc(Abstract base class for transactional Pachner moves.

Each subclass (ShiftMove, FlipMove, IFlipMove, AddMove, RemoveMove)
exposes a propose / apply / rollback lifecycle:

  - propose():  read-only target selection + validation.  Returns
                False if no eligible target is found.
  - apply():    commits the move; builds an internal undo log.
  - rollback(): replays the undo log; restores the spacetime to its
                pre-apply state.  Idempotent.

After a successful propose(), the move publishes its combinatorial
deltas (dN0, dN41, dN32) and Metropolis log prefactor so callers can
plug them into action-based acceptance criteria.

See ``docs/source/modularity-plan.md`` for the design rationale.)doc")
      .def("propose", &PachnerMove::propose,
           "Pick a target and validate.  No state change.  Returns "
           "True on success.")
      .def("apply", &PachnerMove::apply,
           "Commit the proposed move; build the undo log.  Returns "
           "True on success.")
      .def("rollback", &PachnerMove::rollback,
           "Restore the spacetime to its pre-apply state.  Idempotent.")
      .def("isApplied", &PachnerMove::isApplied,
           "True iff apply() has been called and not rolled back.")
      .def("dN0", &PachnerMove::dN0,
           "Combinatorial change in vertex count.")
      .def("dN41", &PachnerMove::dN41,
           "Combinatorial change in N41-type top-simplex count.")
      .def("dN32", &PachnerMove::dN32,
           "Combinatorial change in N32-type top-simplex count.")
      .def("metropolisLogPrefactor", &PachnerMove::metropolisLogPrefactor,
           "log of the Metropolis combinatorial prefactor.")
      .def("touchedVertexIds", &PachnerMove::touchedVertexIds,
           "IDs of the vertices whose neighborhood the move re-arranges. "
           "Used for informed (community-aware) proposals.")
      .def("moveType", &PachnerMove::moveType,
           "Move-type tag: one of 'add', 'remove', 'flip', 'iflip', "
           "'shift'.");

  py::class_<AddMove, PachnerMove>(m_spacetime, "AddMove",
      R"doc(Transactional (2,2d) add (vertex insertion) move.

Picks a random N41 simplex, finds its spatial face and the adjacent
simplex of opposite orientation, and inserts a new vertex at the
spatial time slice.  ``dN0 = +1``; ``dN41 = +(2d-2) = +6`` in 4D;
``dN32 = 0``.

Vertex relabeling (per [BGL] Sec. 2.2.1) is enabled by default.
Pass ``relabel=False`` to disable for tests that need stable
fingerprints across moves.)doc")
      .def(py::init<Spacetime *, std::uint64_t, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("relabel") = true,
           py::keep_alive<1, 2>(),
           "Construct an add move bound to ``spacetime`` with a fresh "
           "RNG seeded from ``seed``.  ``relabel`` controls whether the "
           "new vertex's ID is swap-relabeled with a random existing "
           "vertex on apply().");

  py::class_<FlipMove, PachnerMove>(m_spacetime, "FlipMove",
      R"doc(Transactional (2,d) flip move.

Removes 2 d-simplices sharing a (d-1)-face and creates d new
d-simplices sharing an edge.  ``dN0 = 0``; ``ΔN4 = d - 2 = +2`` in 4D.
Inverse: :class:`IFlipMove`.)doc")
      .def(py::init<Spacetime *, std::uint64_t>(),
           py::arg("spacetime"), py::arg("seed"),
           py::keep_alive<1, 2>(),
           "Construct a (2,d) flip move bound to ``spacetime`` with a "
           "fresh ``std::mt19937`` seeded with ``seed``.");

  py::class_<RemoveMove, PachnerMove>(m_spacetime, "RemoveMove",
      R"doc(Transactional (2d, 2) remove (vertex deletion) move.

Picks a random vertex with order 2d, removes the 2d incident
N41-type simplices and the vertex, and creates 2 replacement
simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D;
``dN32 = 0``.  Inverse: :class:`AddMove`.

Rollback recreates the deleted vertex (with original ID and
coordinates), reinserts its incident edges (with original squared
lengths), and recreates the 2d removed simplices.)doc")
      .def(py::init<Spacetime *, std::uint64_t>(),
           py::arg("spacetime"), py::arg("seed"),
           py::keep_alive<1, 2>(),
           "Construct a remove move bound to ``spacetime`` with a fresh "
           "RNG seeded from ``seed``.");

  py::class_<IFlipMove, PachnerMove>(m_spacetime, "IFlipMove",
      R"doc(Transactional inverse (d, 2) flip move.

Removes d d-simplices sharing an edge and creates 2 new d-simplices
sharing a (d-1)-face.  ``dN0 = 0``; ``ΔN4 = -(d - 2) = -2`` in 4D.
Inverse: :class:`FlipMove`.

Includes a manifold-preservation check in propose() — rejects if
either new simplex would already exist in the lattice.)doc")
      .def(py::init<Spacetime *, std::uint64_t>(),
           py::arg("spacetime"), py::arg("seed"),
           py::keep_alive<1, 2>(),
           "Construct an inverse flip bound to ``spacetime`` with a "
           "fresh ``std::mt19937`` seeded with ``seed``.");

  py::class_<ShiftMove, PachnerMove>(m_spacetime, "ShiftMove",
      R"doc(Transactional (3,3) shift move.

Picks a random top simplex and a random (d-2)-face.  If exactly d-1
top simplices share that face, replaces them with d-1 new simplices
sharing the complementary (d-2)-face.  Self-inverse — dN0 = 0 and
dN41 + dN32 = 0.)doc")
      .def(py::init<Spacetime *, std::uint64_t>(),
           py::arg("spacetime"), py::arg("seed"),
           py::keep_alive<1, 2>(),
           R"doc(Construct a shift move bound to ``spacetime``, using a
fresh ``std::mt19937`` seeded with ``seed`` for the proposal.

For sweeps that share a single Markov chain across many moves, drive
moves via ``CDT.proposeShift()`` instead.)doc");

  // ========================================
  // CDTSimulation
  // ========================================
  py::class_<CDT, std::shared_ptr<CDT> >(m_simulations, "CDTSimulation",
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
      .def("proposeAdd", &CDT::proposeAdd,
           R"doc(Construct a transactional AddMove bound to this
simulation's spacetime + RNG, with propose() already called.  Returns
None if no eligible target.  Caller drives apply()/rollback().  Does
NOT update acceptance counters.)doc")
      .def("proposeRemove", &CDT::proposeRemove,
           "Like proposeAdd() for the (2d,2) remove move.")
      .def("proposeFlip", &CDT::proposeFlip,
           "Like proposeAdd() for the (2,d) flip move.")
      .def("proposeIflip", &CDT::proposeIflip,
           "Like proposeAdd() for the (d,2) inverse-flip move.")
      .def("proposeShift", &CDT::proposeShift,
           "Like proposeAdd() for the (3,3) shift move.")
      .def("ishift", &CDT::ishift,
           R"doc(Attempt one inverse shift move (same as shift, since (3,3) is self-inverse).

Returns True if accepted, False if rejected.)doc")
      .def("sweep", [](CDT &self, int nSweeps, py::object progress) {
          int total = 0;
          if (progress.is_none()) {
              // No callback — release the GIL for the entire loop so
              // multiple threads get true parallelism.
              py::gil_scoped_release release;
              for (int i = 0; i < nSweeps; i++) {
                  total += self.sweep();
              }
          } else {
              for (int i = 0; i < nSweeps; i++) {
                  int accepted;
                  {
                      py::gil_scoped_release release;
                      accepted = self.sweep();
                  }
                  total += accepted;
                  progress(i + 1, nSweeps);
              }
          }
          return total;
      }, py::arg("nSweeps") = 1, py::arg("progress") = py::none(),
           R"doc(Run one or more Monte Carlo sweeps.

Each sweep proposes N4 moves uniformly among all 5 types
(add, remove, flip, iflip, shift).

Args:
    nSweeps: Number of sweeps to perform (default 1).
    progress: Optional callback(i, n) called after each sweep.

Returns the total number of accepted moves across all sweeps.)doc")
      .def("tune", [](CDT &self, py::object progress) {
          if (progress.is_none()) {
              py::gil_scoped_release release;
              self.tune();
          } else {
              self.tune([&](int i, int n) {
                  py::gil_scoped_acquire acquire;
                  progress(i, n);
              });
          }
      }, py::arg("progress") = py::none(),
           R"doc(Tune k4 to its pseudo-critical value ([BGL] Sec. 3.3.1).

Computes an initial estimate of k4 from the coupling constants, then
performs 20 feedback sweeps adjusting k4 to drive N41 toward the target.
Call this before sweep() for stable simulations.

Args:
    progress: Optional callback(i, n) called after each tuning sweep.)doc")
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
tests that compare simplex fingerprints before and after moves.)doc")
      .def("setSeed",
           [](CDT& self, std::uint32_t seed) { self.setSeed(seed); },
           py::arg("seed"),
           R"doc(Re-seed the internal RNG.

The default constructor pulls a seed from std::random_device — fine
for production MC sweeps but flaky for tests whose outcome depends
on a specific growth pattern. Pass a fixed seed at the top of such
tests to make them reproducible.)doc");

  // ========================================
  // SparseGraph (for modularity / spectral dimension)
  // ========================================
  py::class_<SparseGraph>(m_observables, "SparseGraph",
      R"doc(Undirected sparse graph in CSR form.

Built from the COO output of ``Spacetime.getDualAdjacency`` (or
similar).  Used by the modularity sweep to compute spectral
dimension on the dual graph.)doc")
      .def_static("fromCOO", &SparseGraph::fromCOO,
                  py::arg("rows"), py::arg("cols"), py::arg("n"),
                  "Construct from COO arrays + node count.")
      .def("nNodes", &SparseGraph::nNodes,
           "Number of nodes.")
      .def("nEdges", &SparseGraph::nEdges,
           "Number of undirected edges.")
      .def("isBipartite", &SparseGraph::isBipartite,
           "True iff the graph is 2-colorable (no odd cycle).")
      .def("diagonalHeatKernel",
           [](const SparseGraph &self,
              const std::vector<std::uint32_t> &starts,
              const std::vector<double> &times,
              int krylovDim) {
             auto flat = self.diagonalHeatKernel(starts, times, krylovDim);
             // Reshape to list-of-lists for Python convenience.
             std::vector<std::vector<double>> out(starts.size(),
                                                   std::vector<double>(times.size()));
             for (std::size_t s = 0; s < starts.size(); ++s) {
               for (std::size_t j = 0; j < times.size(); ++j) {
                 out[s][j] = flat[s * times.size() + j];
               }
             }
             return out;
           },
           py::arg("starts"), py::arg("times"), py::arg("krylovDim") = 30,
           R"doc(Diagonal of the heat kernel ``e^{-t L_sym}`` for each
(start, t) pair.  Returns a list of lists with shape
(len(starts), len(times)).)doc")
      .def("spectralDimension",
           [](const SparseGraph &self, int nWalks, double maxSigma,
              std::uint64_t seed, double tailFraction, int nTimes,
              double tMin, int krylovDim) {
             std::mt19937 rng(seed);
             return self.spectralDimension(nWalks, maxSigma, &rng,
                                           tailFraction, nTimes, tMin,
                                           krylovDim);
           },
           py::arg("nWalks"), py::arg("maxSigma"), py::arg("seed") = 0,
           py::arg("tailFraction") = 0.2, py::arg("nTimes") = 40,
           py::arg("tMin") = 0.5, py::arg("krylovDim") = 30,
           R"doc(Estimate spectral dimension at small / large diffusion times.

Returns ``(D_S_small, D_S_large)``.  Mirrors the Python implementation
in ``examples/modularity.py:Graph.spectral_dimension``.)doc");

  // ========================================
  // SimplexFilter (predicate over top simplices)
  // ========================================
  //
  // Held as ``std::shared_ptr<SimplexFilter>`` so the same object can
  // be referenced from HolographyConfig.simplexFilter and survive
  // copy-by-value of the config. Python subclassing is not supported
  // in this build — extend via C++ subclass + binding instead.
  py::class_<SimplexFilter, std::shared_ptr<SimplexFilter>>(m_mesh, "SimplexFilter",
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
             std::shared_ptr<AllSimplexFilter>>(m_mesh, "AllSimplexFilter",
      R"doc(Accepts every top simplex.

Default for the holographic-dual measurement (issue #31). Registration
via ``Spacetime.createSimplex`` already implies combinatorial
constructibility (the ``k + 1`` vertices form a complete subgraph in
the edge set), so this filter intentionally ignores edge-length
geometry.)doc")
      .def(py::init<>());
  py::class_<PositiveGramDeterminantFilter, SimplexFilter,
             std::shared_ptr<PositiveGramDeterminantFilter>>(m_mesh, "PositiveGramDeterminantFilter",
      R"doc(Accepts simplices whose Gram matrix has positive determinant.

Restricts the measurement to metrically valid (non-degenerate,
non-collapsed) Euclidean cells. Stricter alternative to the default
``AllSimplexFilter``.)doc")
      .def(py::init<>());

  // ========================================
  // ModularityOptimizer
  // ========================================
  py::class_<ModularityMeasurement>(m_observables, "ModularityMeasurement",
      R"doc(One recorded point on the (Q, D_S) trajectory.

Mirrors examples/modularity.py:Measurement.)doc")
      .def_readonly("Q", &ModularityMeasurement::Q)
      .def_readonly("dsSmall", &ModularityMeasurement::dsSmall)
      .def_readonly("dsLarge", &ModularityMeasurement::dsLarge)
      .def_readonly("nVertices", &ModularityMeasurement::nVertices)
      .def_readonly("nEdges", &ModularityMeasurement::nEdges)
      .def_readonly("nSimplices", &ModularityMeasurement::nSimplices)
      .def_readonly("iter", &ModularityMeasurement::iter)
      .def_readonly("direction", &ModularityMeasurement::direction);

  py::class_<ModularityOptimizerConfig>(m_observables, "ModularityOptimizerConfig")
      .def(py::init<>())
      .def_readwrite("targetDq", &ModularityOptimizerConfig::targetDq)
      .def_readwrite("maxIterations",
                     &ModularityOptimizerConfig::maxIterations)
      .def_readwrite("nDiffusionWalks",
                     &ModularityOptimizerConfig::nDiffusionWalks)
      .def_readwrite("maxSigma", &ModularityOptimizerConfig::maxSigma)
      .def_readwrite("negativeRetryMax",
                     &ModularityOptimizerConfig::negativeRetryMax)
      .def_readwrite("epsilonQMax",
                     &ModularityOptimizerConfig::epsilonQMax)
      .def_readwrite("krylovDim", &ModularityOptimizerConfig::krylovDim)
      .def_readwrite("targetNModules",
                     &ModularityOptimizerConfig::targetNModules);

  py::class_<ModularityOptimizer>(m_observables, "ModularityOptimizer",
      R"doc(Modularity sweep on a CDT spacetime, driven by transactional
Pachner moves with Q-direction acceptance.

Each iteration:
  1. Picks a random move type from {add, remove, flip, iflip, shift}.
  2. Calls cdt.proposeXxx() — if no eligible target, tries another type.
  3. Snapshots Q on the spacetime 1-skeleton.
  4. Calls move.apply() to commit the move.
  5. Computes new Q.  If direction matches, keeps the move; else
     calls move.rollback().
  6. If Q crossed the next target_dq threshold, builds the dual graph
     and measures D_S.

See docs/source/modularity-plan.md for the design rationale.)doc")
      .def(py::init<ModularityOptimizerConfig, std::uint64_t>(),
           py::arg("config"), py::arg("seed") = 0)
      .def("sweep",
           [](ModularityOptimizer &self, CDT &cdt,
              const std::string &direction, py::object progress) {
             ModularityOptimizer::ProgressCallback cb = nullptr;
             if (!progress.is_none()) {
               cb = [&progress](int it, int maxIt, double q,
                                std::size_t n) {
                 py::gil_scoped_acquire acquire;
                 progress(it, maxIt, q, n);
               };
             }
             py::gil_scoped_release release;
             return self.sweep(cdt, direction, cb);
           },
           py::arg("cdt"), py::arg("direction"),
           py::arg("progress") = py::none(),
           "Drive `cdt` to walk Q in direction ('up' or 'down'). "
           "Returns list of ModularityMeasurement.")
      .def("getNAccepted", &ModularityOptimizer::getNAccepted,
           "Number of moves applied + kept (since the last sweep).")
      .def("getNRolledBack", &ModularityOptimizer::getNRolledBack,
           "Number of moves applied then rolled back.")
      .def("getNNoMove", &ModularityOptimizer::getNNoMove,
           "Number of iterations with no eligible move.")
      .def("getNMeasurements", &ModularityOptimizer::getNMeasurements,
           "Number of D_S measurements taken.");

  // ========================================
  // VolumeProfile
  // ========================================
  py::class_<VolumeProfile, std::shared_ptr<VolumeProfile> >(m_observables, "VolumeProfile",
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
  py::enum_<WilsonMode>(m_observables, "WilsonMode",
      "Evaluation mode for Wilson loops.")
      .value("COMBINATORIAL", WilsonMode::COMBINATORIAL,
             "Dual-graph topology only (loop length, enclosed hinges).")
      .value("DEFICIT_ANGLE", WilsonMode::DEFICIT_ANGLE,
             "Deficit-angle based: W = ((d-2)+2cos(epsilon))/d.")
      .value("CAUSAL", WilsonMode::CAUSAL,
             "CDT causal orientation changes around the loop.");

  py::enum_<LoopType>(m_observables, "LoopType",
      "Which loop-shape generator to use.")
      .value("HINGE", LoopType::HINGE,
             "Elementary loop around a (d-2)-simplex.")
      .value("DUAL_LATTICE", LoopType::DUAL_LATTICE,
             "BFS-discovered loop of a target size.")
      .value("GEODESIC", LoopType::GEODESIC,
             "Shortest cycle through a start simplex.");

  py::class_<LoopPath>(m_observables, "LoopPath",
      "A closed path through the dual graph (sequence of top-simplices).")
      .def_readonly("simplices", &LoopPath::simplices,
                    py::return_value_policy::reference)
      .def_readonly("facets", &LoopPath::facets,
                    py::return_value_policy::reference)
      .def("__len__", [](const LoopPath &lp) { return lp.simplices.size(); });

  py::class_<WilsonResult>(m_observables, "WilsonResult",
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

  py::class_<WilsonLoop, std::shared_ptr<WilsonLoop>>(m_observables, "WilsonLoop",
      R"doc(Wilson loop observable on a triangulated spacetime.

A Wilson loop is the trace of a parallel-transport operator around a
closed path. On a curved triangulation without an explicit gauge field
tessera computes the Levi-Civita holonomy analogue: closed walks on the
dual graph (top-simplices as nodes, shared facets as edges), with the
loop value determined by the deficit angles of enclosed hinges.

Three evaluation modes:

* ``COMBINATORIAL``  — dual-graph topology only. ``value`` is the loop
  length; ``enclosedHinges`` counts hinges contained in every loop
  simplex; ``contractible`` is True iff ``enclosedHinges == 0``.
* ``DEFICIT_ANGLE``  — Regge-curvature holonomy. For a hinge loop
  enclosing one hinge h:
      W = ((d-2) + 2 cos(eps_h)) / d
  For multi-hinge loops the U(1) approximation is used:
      W = product_{h in enclosed} cos(eps_h).
  W = 1 corresponds to a flat loop; deviation from 1 measures local
  curvature.
* ``CAUSAL``  — CDT causal-orientation winding. ``causalWindingNumber``
  is the signed net change in foliation index around the loop; non-
  zero values mark loops that cross a CDT slice boundary.

Three loop-shape generators:

* ``hingeLoop(h)``        — cyclically ordered loop of top-simplices
                            around the (d-2)-simplex ``h``. Encloses
                            exactly one hinge (``h``), so the
                            DEFICIT_ANGLE formula above is exact.
* ``dualLatticeLoop(start, L)``  — BFS-discovered loop of approximately
                            ``L`` simplices through ``start``. Suitable
                            for population sweeps at fixed loop scale.
* ``geodesicLoop(start)``  — shortest cycle through ``start`` in the
                            dual graph (girth at that simplex).

Measurement bookkeeping: ``measure()`` and ``measureAllHinges()`` append
``WilsonResult`` entries to an internal list. ``getMeasurements()``
returns the accumulated list; ``getAverageBySize()`` aggregates by loop
length (the standard form for Creutz-ratio-style analyses);
``reset()`` clears.

See ``docs/source/wilson_loops.md`` for an end-to-end tutorial with
curvature-scan, contractibility-statistics, and causal-winding
examples.
)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Construct a Wilson-loop calculator bound to a Spacetime.")
      .def("evaluate", &WilsonLoop::evaluate,
           py::arg("loop"), py::arg("mode"),
           R"doc(Evaluate the Wilson loop in the given mode.

Dispatches to ``evaluateCombinatorial``, ``evaluateDeficitAngle``, or
``evaluateCausal`` depending on ``mode``.
)doc")
      .def("evaluateCombinatorial", &WilsonLoop::evaluateCombinatorial,
           py::arg("loop"),
           R"doc(Evaluate using dual-graph topology only.

Returns a ``WilsonResult`` with ``value = loopSize``,
``enclosedHinges`` = count of hinges shared by every loop simplex, and
``contractible`` = True iff no hinge is enclosed.
)doc")
      .def("evaluateDeficitAngle", &WilsonLoop::evaluateDeficitAngle,
           py::arg("loop"),
           R"doc(Evaluate using Regge deficit angles.

For a hinge loop (exactly one enclosed hinge h):
    W = ((d - 2) + 2 cos(eps_h)) / d.
For multi-hinge loops the U(1) approximation:
    W = product_{h in enclosed} cos(eps_h).
``value`` carries W; ``enclosedHinges`` carries the count.
)doc")
      .def("evaluateCausal", &WilsonLoop::evaluateCausal,
           py::arg("loop"),
           R"doc(Evaluate using CDT causal-orientation changes.

Walks the loop and accumulates a signed winding count from the
final-time stamps of consecutive simplices. ``causalWindingNumber`` is
the net winding; ``value`` carries the same number as a double.
)doc")
      .def("hingeLoop", &WilsonLoop::hingeLoop,
           py::arg("hinge"),
           R"doc(Loop of top-simplices around a hinge, ordered cyclically.

Encloses exactly one hinge (the input). This is the natural loop for
``DEFICIT_ANGLE`` mode and is what ``measureAllHinges`` uses internally.
)doc")
      .def("dualLatticeLoop", &WilsonLoop::dualLatticeLoop,
           py::arg("start"), py::arg("targetLength"),
           R"doc(BFS-discovered loop of approximately ``targetLength`` simplices.

Not guaranteed to be exactly ``targetLength`` — the BFS may overshoot
or return a shorter loop if local connectivity doesn't permit closing
at the target size. Suitable for population-level scans at a fixed
loop scale (analogous to specifying Wilson-loop side length in lattice
gauge theory).
)doc")
      .def("geodesicLoop", &WilsonLoop::geodesicLoop,
           py::arg("start"),
           R"doc(Shortest cycle through ``start`` in the dual graph.

The cycle length is the local girth of the dual graph at ``start``.
Useful when you want the natural shortest loop without specifying a
target size.
)doc")
      .def("measure", &WilsonLoop::measure,
           py::arg("loop"), py::arg("mode"),
           "Evaluate ``loop`` in ``mode`` and append the result to the "
           "internal measurement list.")
      .def("measureAllHinges", &WilsonLoop::measureAllHinges,
           py::arg("mode"),
           R"doc(Walk every (d-2)-simplex of the spacetime, generate its hinge
loop, and record the evaluation in ``mode``. Skips degenerate hinges
whose loop has fewer than 2 distinct simplices. Bulk shortcut for a
curvature scan.
)doc")
      .def("reset", &WilsonLoop::reset,
           "Clear all accumulated measurements.")
      .def("getMeasurements", &WilsonLoop::getMeasurements,
           "Return the full list of accumulated ``WilsonResult`` entries.")
      .def("getAverageBySize", &WilsonLoop::getAverageBySize,
           R"doc(Mean ``value`` grouped by loop size, as a ``{size: mean}`` dict.

The standard form for Creutz-ratio-style analyses: fix loop size L,
read off the population-averaged Wilson value at that scale.
)doc");

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
           py::arg("center"), py::arg("rhoOfR"),
           R"doc(Assign energy density as a function of geodesic distance.

Args:
    center: The reference vertex.
    rhoOfR: A callable taking distance (float) and returning density (float).)doc")
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
  py::class_<ReggeSolver>(m_simulations, "ReggeSolver",
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
           py::arg("learningRate") = 0.001,
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
           py::arg("maxIters") = 5000,
           py::arg("learningRate") = 0.001,
           py::arg("progress") = py::none(),
           R"doc(Find stationary point of the total Regge action (discrete Einstein eqs).

Minimizes F = ||∇S||² until F < tol or maxIters reached.

Args:
    tol: Convergence tolerance on F = ||∇S||².
    maxIters: Maximum number of iterations.
    learningRate: Gradient descent step size.
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

#ifdef TESSERA_VERSION
  m.attr("__version__") = TESSERA_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

#ifdef TESSERA_QUANTUM
  // Register the Schwinger / DMRG bindings as a `quantum` submodule so users
  // call them as `tessera._tessera.quantum.computeGroundState(...)` (typically
  // routed through `tessera.quantum` — see tessera/quantum/__init__.py).
  register_quantum_bindings(m.def_submodule("quantum",
      "Schwinger model + DMRG (docs/source/quantum-plan.md)."));
#endif
}
