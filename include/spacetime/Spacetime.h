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

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "topologies/Topology.h"
#include "observables/Observable.h"
#include "EdgeList.h"
#include "Foliation.h"
#include "VertexList.h"
#include "Metric.h"
#include "Simplex.h"

namespace caset {
enum class SpacetimeType : uint8_t {
  CDT = 0,
  REGGE = 1,
  COSET = 2,
  REGGE_PACHNER = 3,
  GFT_SPIN_FOAM = 4,
  RICCI_FLOW_DISCRETIZATION = 5
};

///
/// # Spacetime
///
/// The Spacetime class provides methods to create and manipulate the basic building blocks of a simplicial
/// complex \f$ \mathcal{K} \f$.
///
/// The Spacetime manages the simplicial complex structure, including vertices \f$ V \f$, edges \f$ E \f$, and
/// simplices \f$ \{\sigma^k_i\} \f$ of varying dimensions. It is responsible for constructing and maintaining
/// the topological relationships between these elements.
///
/// The Spacetime Topology is responsible for constructing Simplex(es) and the Topology (subclass) is responsible for
/// building the complex to match that topology.
///
/// Any assertions or state needed by the Topology to build the complex should be implemented in the Simplex.
///
template<int D>
class Spacetime {
  public:
    // ========================================
    // Constructors
    // ========================================

    /// Default constructor. Creates a 4D Lorentzian spacetime with CDT type and Toroid topology.
    Spacetime<D>();

    /// Parameterized constructor.
    /// @param metric_ The metric tensor defining the signature and dimension
    /// @param spacetimeType_ The type of quantum gravity formulation (CDT, Regge, etc.)
    /// @param alpha_ The fundamental length scale coefficient, the number of times the length of a timelike edge is the
    ///   length of a spacelike edge
    /// @param a_ The fundamental length unit for a spacelike edge.
    /// @param foliation_ The type of foliation for the spacetime. Preferred means spacelike slices are separated by
    ///   timelike slices. None means they can be interspersed.
    /// @param topology_ The topology of the spatial slices (default: Toroid)
    Spacetime<D>(
      std::shared_ptr<Metric<D>> metric_,
      SpacetimeType spacetimeType_,
      std::optional<double> alpha_,
      std::optional<double> a_,
      Foliation foliation_,
      std::optional<std::shared_ptr<Topology<D>> > topology_);

    // ========================================
    // Creation Methods
    // ========================================

    std::pair<SimplexPtr<D>, bool> createSimplex(const VertexPtrs<D> &vertices);

    /// Creates a simplex \f$ \sigma^k \f$ from explicit vertices and edges.
    /// @param vertices The vertices \f$ \{v_0, \ldots, v_k\} \f$ of the simplex
    /// @param edges The edges \f$ \{e_{ij}\} \f$ connecting the vertices
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created, false if already existed
    std::pair<SimplexPtr<D>, bool> createSimplex(const VertexPtrs<D> &vertices, const Edges<D> &edges);

    /// Creates a simplex \f$ \sigma^{(t_i, t_f)} \f$ with the given causal orientation.
    /// @param numericOrientation Tuple (timelike_initial, timelike_final) defining the orientation
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr<D>, bool> createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation);

    /// Creates a k-simplex with randomly positioned vertices and edges of length \f$ \alpha \f$.
    /// @param k The dimension of the simplex (k+1 vertices)
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr<D>, bool> createSimplex(std::size_t k);

    /// Creates a vertex \f$ v \f$ with the next available ID at the current time.
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr<D> createVertex() noexcept;
    [[nodiscard]] VertexPtr<D> createVertex(const std::vector<double> &coords) noexcept;


    /// Creates a vertex \f$ v \f$ with the given ID at the current time.
    /// @param id The unique identifier for the vertex
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr<D> createVertex(std::uint64_t id) const noexcept;

    /// Creates a vertex \f$ v \f$ with the given ID and coordinates \f$ (t, x^1, \ldots, x^{n-1}) \f$.
    /// @param id The unique identifier for the vertex
    /// @param coords The spacetime coordinates of the vertex
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr<D> createVertex(std::uint64_t id, const std::vector<double> &coords) const noexcept;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ with squared length computed from the metric.
    /// The squared length is \f$ \ell^2 = g_{\mu\nu}(x^t - x^s)^\mu(x^t - x^s)^\nu \f$.
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @return Shared pointer to the created edge
    [[nodiscard]] EdgePtr<D> createEdge(const VertexPtr<D> &src, const VertexPtr<D> &tgt) const noexcept;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ with explicit squared length.
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @param squaredLength The squared length \f$ \ell^2 \f$ of the edge
    /// @return Shared pointer to the created edge
    [[nodiscard]] EdgePtr<D> createEdge(const VertexPtr<D> &src, const VertexPtr<D> &tgt, double squaredLength) const noexcept;

    // ========================================
    // Complex Building Methods
    // ========================================

    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    ///
    /// Constructs an initial simplicial complex \f$ \mathcal{K}_0 \f$ by iteratively gluing \f$ n \f$ simplices.
    /// @param numSimplices The number of simplices to add to the initial complex
    void build(int numSimplices=3);

    // ========================================
    // Query Methods
    // ========================================

    /// @return The type of quantum gravity formulation (CDT, Regge, etc.)
    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept;

    /// @return The current time coordinate \f$ t \f$ for time-slicing
    [[nodiscard]] double getCurrentTime() const noexcept;

    /// @return The edge list \f$ E \f$ containing all edges in the complex
    [[nodiscard]] std::shared_ptr<EdgeList<D>> getEdgeList() const noexcept;

    /// @return The metric tensor \f$ g_{\mu\nu} \f$ defining the geometry
    [[nodiscard]] std::shared_ptr<Metric<D>> getMetric() const noexcept;

    /// @return The vertex list \f$ V \f$ containing all vertices in the complex
    [[nodiscard]] std::shared_ptr<VertexList<D>> getVertexList() const noexcept;

    /// @return Simplices around the boundary of the simplicial complex. These simplices have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplices as you might expect, but 5-simplices since those are the standard
    /// building blocks. You can get the 2-simplices by calling `getFacets()` on the 5-simplices and their facets until
    /// \f$ k=2 \f$.
    [[nodiscard]] SimplexSet<D> getExternalSimplices() noexcept;

    /// Retrieves all simplices with a specific causal orientation.
    /// @param orientation The orientation tuple (timelike_initial, timelike_final)
    /// @return Set of simplices \f$ \{\sigma^{(t_i, t_f)}\} \f$ with the given orientation
    /// @note This method is for testing only and has poor runtime performance.
    SimplexSet<D> getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation);

    /// Returns the foliation of the spacetime. Either NONE or PREFERRED. With PREFERRED foliation; each spatial slice
    /// lies between a time slice and vis versa. Otherwise they can be any-which-a-way.
    /// @return Foliation::PREFERRED or Foliation::NONE.
    [[nodiscard]] Foliation getFoliation() const noexcept;

    /// Computes the connected components of the vertex graph.
    ///
    /// Uses depth-first search to identify connected components in the graph
    /// \f$ G = (V, E) \f$ where vertices are connected by edges.
    ///
    /// @return Vector of connected components, each containing a list of vertices
    [[nodiscard]] std::vector<VertexPtrs<D>> getConnectedComponents() const;

    /// Retrieves a simplex from the complex by pointer lookup.
    /// @param simplex The simplex to find
    /// @return The canonical simplex from the complex, or nullptr if not found
    [[nodiscard]] SimplexPtr<D> getSimplex(SimplexPtr<D> simplex) const;

    /// Retrieves a simplex from the complex by fingerprint.
    /// @param fingerprint The fingerprint hash \f$ h(\sigma) \f$ of the simplex
    /// @return The simplex with the given fingerprint, or nullptr if not found
    [[nodiscard]] SimplexPtr<D> getSimplex(std::uint64_t fingerprint) const;

    // ========================================
    // Manipulation & Helper Methods
    // ========================================

    /// Increments the current time coordinate.
    /// @return The new current time \f$ t \leftarrow t + 1 \f$
    double incrementTime() noexcept;

    /// Removes a vertex if it has no incident edges.
    ///
    /// Checks if \f$ \deg(v) = 0 \f$ and removes \f$ v \f$ from the vertex list if so.
    ///
    /// @param vertex The vertex \f$ v \f$ to potentially remove
    /// @return true if the vertex was removed, false if it has incident edges
    [[nodiscard]] bool removeIfIsolated(const VertexPtr<D> &vertex) const noexcept;

    /// Adds an observable to track during evolution.
    ///
    /// Observables are measured after each update step to monitor the system's properties.
    ///
    /// @param observable The observable \f$ \mathcal{O} \f$ to track
    void addObservable(const std::shared_ptr<Observable<D>> &observable);

    // ========================================
    // Internal Management
    // ========================================

    /// Registers a simplex in the spacetime's internal data structures.
    ///
    /// Adds the simplex to:
    /// - The main simplex set \f$ K \f$
    /// - The appropriate orientation-indexed sets (external or internal)
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to register
    /// @param internal true if the simplex is fully internal (all faces glued), false otherwise
    /// @return The registered simplex
    SimplexPtr<D> registerSimplex(const SimplexPtr<D> &simplex, bool internal);


    void reserve(int nSimplices);

    /// Alpha is the coefficient that determines the ratio of timelike edge lengths to space like edge lengths. That
    /// relationship is
    ///
    /// \f[
    ///  l_s = +a
    /// \f]
    /// and
    /// \f[
    ///  l_t = - \alpha a
    /// \f]
    ///
    /// @return The coefficient representing the number of times the timelike length is compared to the spatial length.
    [[nodiscard]] double getAlpha() const noexcept;

    /// `a` is the coefficient that sets the fixed edge length for spacelike edges according to
    ///
    /// \f[
    ///  l_s = +a
    /// \f]
    /// for spacelike edges and
    /// \f[
    ///  l_t = - \alpha a
    /// \f]
    ///
    /// for timelike edges.
    ///
    /// @return The constant spacelike edge length.
    [[nodiscard]] double getA() const noexcept;

    /// Unregisters a simplex from the spacetime's internal data structures.
    ///
    /// Removes the simplex from all indexed sets. This should be called before
    /// modifying a simplex's fingerprint to avoid hash table corruption.
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to unregister
    void unregisterSimplex(const SimplexPtr<D> &simplex);

  private:
    std::shared_ptr<EdgeList<D>> edgeList = std::make_shared<EdgeList<D>>();
    std::shared_ptr<VertexList<D>> vertexList = std::make_shared<VertexList<D>>();

    IdType vertexIdCounter = 0;
    SpacetimeType spacetimeType;
    double alpha = 1.;
    double a = 1.;
    Foliation foliation = Foliation::PREFERRED;
    std::shared_ptr<Metric<D>> metric;
    std::shared_ptr<Topology<D>> topology;
    std::uint64_t currentTime = 0;

    SimplexSet<D> simplices{};

    ///
    /// These are simplices on the boundary of a simplicial complex. They have at least one external face, and hence can
    /// be glued to other simplices. The externalSimplices are organized by the orientation of their available faces. If
    /// a face is available; the orientation of that face can be found as a key corresponding to a SimplexSet containing
    /// the Simplex to which that Face belongs.
    ///
    /// This makes for fast lookups when gluing simplices together to form a complex.
    std::unordered_map<SimplexOrientation<D>, SimplexSet<D>, SimplexOrientationHash<D>, SimplexOrientationEq<D>>
    externalSimplicesByFacialOrientation{};

    std::vector<std::shared_ptr<Observable<D>> > observables{};
};
} // caset

#endif //CASET_SPACETIME_H
