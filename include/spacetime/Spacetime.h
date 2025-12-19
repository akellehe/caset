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
#include "VertexList.h"
#include "Metric.h"
#include "Simplex.h"
#include "topologies/Toroid.h"

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
class Spacetime {
  public:
    // ========================================
    // Constructors
    // ========================================

    /// Default constructor. Creates a 4D Lorentzian spacetime with CDT type and Toroid topology.
    Spacetime();

    /// Parameterized constructor.
    /// @param metric_ The metric tensor defining the signature and dimension
    /// @param spacetimeType_ The type of quantum gravity formulation (CDT, Regge, etc.)
    /// @param alpha_ The fundamental length scale (edge length parameter)
    /// @param topology_ The topology of the spatial slices (default: Toroid)
    Spacetime(
      std::shared_ptr<Metric> metric_,
      const SpacetimeType spacetimeType_,
      std::optional<double> alpha_,
      std::optional<std::shared_ptr<Topology> > topology_);

    // ========================================
    // Creation Methods
    // ========================================

    /// Creates a simplex \f$ \sigma^k \f$ from explicit vertices and edges.
    /// @param vertices The vertices \f$ \{v_0, \ldots, v_k\} \f$ of the simplex
    /// @param edges The edges \f$ \{e_{ij}\} \f$ connecting the vertices
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created, false if already existed
    std::pair<SimplexPtr, bool> createSimplex(const VertexPtrs &vertices, const Edges &edges);

    /// Creates a simplex \f$ \sigma^{(t_i, t_f)} \f$ with the given causal orientation.
    /// @param numericOrientation Tuple (timelike_initial, timelike_final) defining the orientation
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr, bool> createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation);

    /// Creates a k-simplex with randomly positioned vertices and edges of length \f$ \alpha \f$.
    /// @param k The dimension of the simplex (k+1 vertices)
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr, bool> createSimplex(std::size_t k);

    /// Creates a vertex \f$ v \f$ with the given ID at the current time.
    /// @param id The unique identifier for the vertex
    /// @return Shared pointer to the created vertex
    VertexPtr createVertex(const std::uint64_t id) noexcept;

    /// Creates a vertex \f$ v \f$ with the given ID and coordinates \f$ (t, x^1, \ldots, x^{n-1}) \f$.
    /// @param id The unique identifier for the vertex
    /// @param coords The spacetime coordinates of the vertex
    /// @return Shared pointer to the created vertex
    VertexPtr createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ with squared length computed from the metric.
    /// The squared length is \f$ \ell^2 = g_{\mu\nu}(x^t - x^s)^\mu(x^t - x^s)^\nu \f$.
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @return Shared pointer to the created edge
    EdgePtr createEdge(const VertexPtr &src, const VertexPtr &tgt) const;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ with explicit squared length.
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @param squaredLength The squared length \f$ \ell^2 \f$ of the edge
    /// @return Shared pointer to the created edge
    EdgePtr createEdge(const VertexPtr &src, const VertexPtr &tgt, double squaredLength) const noexcept;

    // ========================================
    // Complex Building Methods
    // ========================================

    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    ///
    /// Constructs an initial simplicial complex \f$ \mathcal{K}_0 \f$ by iteratively gluing \f$ n \f$ simplices.
    /// @param numSimplices The number of simplices to add to the initial complex
    void build(int numSimplices=3);

    /// This method identifies a pair of faces (one from each simplex) that can be glued together while preserving the
    /// orientation of the simplices. The method checks for matching orientations and edge lengths to ensure
    /// compatibility.
    ///
    /// For faces \f$ \sigma^{k-1}_1 \subset \sigma^k_{\text{unattached}} \f$ and
    /// \f$ \sigma^{k-1}_2 \subset \sigma^k_{\text{attached}} \f$, this returns the pair if:
    /// - They have the same orientation (or complementary orientations)
    /// - They have compatible timelike/spacelike character
    /// - \f$ \sigma^{k-1}_2 \f$ is causally available (has < 2 cofaces)
    ///
    /// Before simplices are glued into the complex we consider them 'detached', so it doesn't matter if we're attaching
    /// a (3, 2) or a (2, 3). There's a parity building method, `Simplex::getVerticesWithParityTo(yourFace)`, that finds
    /// the right order to use when attaching the Simplex to the Simplicial complex.
    ///
    /// @param unattachedSimplex A simplex not yet attached to the simplicial complex.
    /// @param attachedSimplex An attached simplex to which you would like to glue the first. Orientation is based on
    ///   this simplex.
    ///
    /// @return {unattached, attached} faces that can be glued together.
    [[nodiscard]] OptionalSimplexPtrPair
    getGluableFaces(const SimplexPtr &unattachedSimplex, const SimplexPtr &attachedSimplex);

    /// Attaches two simplices by identifying corresponding vertices.
    ///
    /// Given vertex pairs \f$ \{(v_i, w_i)\}_{i=0}^{k-1} \f$, this method:
    /// 1. Redirects all external edges from \f$ v_i \f$ to \f$ w_i \f$
    /// 2. Updates all simplices containing \f$ v_i \f$ to use \f$ w_i \f$ instead
    /// 3. Removes isolated vertices
    ///
    /// @param unattached The simplex containing vertices to be identified
    /// @param attached The simplex whose vertices will replace the unattached vertices
    /// @param vertexPairs Pairs of vertices {unattached_vertex, attached_vertex} to identify
    void attachAtVertices(
      const SimplexPtr &unattached,
      const SimplexPtr &attached,
      const std::vector<std::pair<VertexPtr, VertexPtr> > &vertexPairs
    );

    ///
    /// This method is a simplicial isomorphism between two faces. Specifically; it takes two Simplex Face (s),
    /// \f$ \sigma^{k-1}_{myFace} \f$ and \f$ \sigma^{k-1}_{yourFace} \f$ as inputs and creates a new face
    /// \f$ \sigma^{k-1}_{newFace} \f$ indicating their adjacency in the simplicial complex while preserving the
    /// orientation of both their cofaces.
    ///
    /// This method runs within the context of an n-dimensional simplicial manifold; each (n-1) simplex (where faces are
    /// codimension-1) is incident to exactly 2 n-simplices for interior faces and exactly 1 n-simplex for faces along
    /// the boundary.
    ///
    /// Because this method is (causal) orientation-aware; it's intended only to be used when we're building causal
    /// simplicial complexes.
    ///
    /// If any face is shared by 3 or more n-simplices; then the neighborhood of some point becomes interior and is no
    /// longer homeomorphic to \f$ \mathcal{R}^n \f$ or a half-space
    /// \f$ \mathcal{R}^{n-1} \times [0, \infty) \f$, (the boundary points) so the spacetime effectively branches,
    /// causing it to lose it's manifold properties.
    ///
    /// The building blocks of a 4D causal simplicial complex are (4, 1) and (3, 2) simplices. The (4, 1) simplex has 4
    /// vertices on t=t and 1 on t=t + 1. The (3, 2) simplex has 3 vertices on t=t and 2 on t=t + 1. We build out the
    /// complex by gluing these simplices together along their faces.
    ///
    /// The Face (s) or Facets of the simplex are all sets of vertices of cardinality k-1. So for a 4-simplex
    /// \f$ \sigma^{(1, 4)}_{ab} \f$ we have vertices \f$ \{a_0, a_1, a_2, a_3, b_0\} \f$ and the
    /// 4-faces are the (ordered) combinations of the 4 vertices, where \f$ a \f$ vertices are at \f$ t=t \f$ and \f$ b \f$
    /// vertices are at \f$ t=t+1 \f$.
    ///
    /// \f[
    /// \begin{aligned}
    /// \sigma^{(4, 0)}_0 &= \{a_1, a_2, a_3, b_0\} \\
    /// \sigma^{(3, 1)}_1 &= \{a_0, a_2, a_3, b_0\} \\
    /// \sigma^{(3, 1)}_2 &= \{a_0, a_1, a_3, b_0\} \\
    /// \sigma^{(3, 1)}_3 &= \{a_0, a_1, a_2, b_0\} \\
    /// \sigma^{(3, 1)}_4 &= \{a_0, a_1, a_2, a_3\}
    /// \end{aligned}
    /// \f]
    ///
    /// When a face has vertices from both sets \f$ \{a_i \in A\} \f$ and \f$ \{b_i \in B\} \f$; the face is spacelike. When
    /// it has only vertices from one or the other; it's timelike.
    ///
    /// Let \f$ \sigma^{(3, 2)}_{cd} \f$ have vertices \f$ \{c_i \in C \mid i \in [0, 4]\} \f$ and \f$ \{d_i \in D \mid i \in [0, 4]\} \f$
    /// where vertices of \f$ C \f$ have \f$ t=t \f$ and in \f$ D \f$ they have \f$ t = t+1 \f$. Then \f$ \sigma^{(3, 2)}_{cd} \f$ has
    /// faces,
    ///
    /// \f[
    /// \{\sigma^{(2, 2)}_0, \sigma^{(2, 2)}_1, \sigma^{(2, 2)}_2, \sigma^{(3, 1)}_3, \sigma^{(3, 1)}_4\} \subset \sigma^{(3, 2)}_{cd}
    /// \f]
    ///
    /// We can only join faces of the same shape, e.g. (3, 1) in this case.
    ///
    /// For a detailed picture; see "Quantum Gravity from Causal Dynamical Triangulations: A Review", R. Loll, 2019.
    /// Figure 1.
    ///
    ///
    /// The process of attaching faces amounts to moving external in-edges and out-edges from the vertices of the
    /// unattachedFace to the analogous (parity matches) vertices of the attachedFace.
    ///
    /// This means we identify the vertices that match, in order. There are two classes of edges now. Those internal to
    /// the unattachedFace, and those that have a vertex outside the unattachedFace. The internal edges have analogous
    /// edges in the attachedFace, so we can delete those edges, replacing them with their analogous counterparts in the
    /// attachedFace. The external edges need to be moved from the unattachedFace vertex to the attachedFace vertex.
    ///
    /// @param attachedFace The Face of this Simplex to attach to `unattachedFace` of the other Simplex
    /// @param unattachedFace The Face of the other Simplex to attach to `attachedFace` of this Simplex.
    /// @returns {attachedFace, succeeded} The `attachedFace` after attachment and whether the attachment succeeded.
    std::tuple<SimplexPtr, bool> causallyAttachFaces(const SimplexPtr &attachedFace, const SimplexPtr &unattachedFace);

    /// This method chooses a simplex from the boundary of the simplicial complex to which `unattachedSimplex` can be
    /// glued. It does this by iterating through the `externalSimplices` and checking for compatible orientations and
    /// edge lengths.
    ///
    /// To the extent the hashing function for vertex fingerprinting is good; this should be pretty well pseudo-random.
    /// If you want something truly random, though, you should probably implement that.
    ///
    /// @param unattachedSimplex The simplex \f$ \sigma \f$ for which to find a gluable face
    /// @returns A pair of \f$ k-1 \f$ simplices (faces) if a compatible k-simplex was found. None otherwise.
    OptionalSimplexPtrPair chooseSimplexFacesToGlue(const SimplexPtr &unattachedSimplex);

    // ========================================
    // Query Methods
    // ========================================

    /// @return The type of quantum gravity formulation (CDT, Regge, etc.)
    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept;

    /// @return The current time coordinate \f$ t \f$ for time-slicing
    [[nodiscard]] double getCurrentTime() const noexcept;

    /// @return The edge list \f$ E \f$ containing all edges in the complex
    [[nodiscard]] std::shared_ptr<EdgeList> getEdgeList() const noexcept;

    /// @return The metric tensor \f$ g_{\mu\nu} \f$ defining the geometry
    [[nodiscard]] std::shared_ptr<Metric> getMetric() const noexcept;

    /// @return The vertex list \f$ V \f$ containing all vertices in the complex
    [[nodiscard]] std::shared_ptr<VertexList> getVertexList() const noexcept;

    /// @return Simplices around the boundary of the simplicial complex. These simplices have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplices as you might expect, but 5-simplices since those are the standard
    /// building blocks. You can get the 2-simplices by calling `getFacets()` on the 5-simplices and their facets until
    /// \f$ k=2 \f$.
    [[nodiscard]] SimplexSet getExternalSimplices() noexcept;

    /// Retrieves all simplices with a specific causal orientation.
    /// @param orientation The orientation tuple (timelike_initial, timelike_final)
    /// @return Set of simplices \f$ \{\sigma^{(t_i, t_f)}\} \f$ with the given orientation
    /// @note This method is for testing only and has poor runtime performance.
    SimplexSet getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation);

    /// Computes the connected components of the vertex graph.
    ///
    /// Uses depth-first search to identify connected components in the graph
    /// \f$ G = (V, E) \f$ where vertices are connected by edges.
    ///
    /// @return Vector of connected components, each containing a list of vertices
    [[nodiscard]] std::vector<VertexPtrs> getConnectedComponents() const;

    /// Retrieves a simplex from the complex by pointer lookup.
    /// @param simplex The simplex to find
    /// @return The canonical simplex from the complex, or nullptr if not found
    SimplexPtr getSimplex(SimplexPtr simplex) const;

    /// Retrieves a simplex from the complex by fingerprint.
    /// @param fingerprint The fingerprint hash \f$ h(\sigma) \f$ of the simplex
    /// @return The simplex with the given fingerprint, or nullptr if not found
    SimplexPtr getSimplex(std::uint64_t fingerprint) const;

    // ========================================
    // Manipulation & Helper Methods
    // ========================================

    /// Increments the current time coordinate.
    /// @return The new current time \f$ t \leftarrow t + 1 \f$
    double incrementTime() noexcept;

    /// Redirects all incoming edges to a vertex.
    ///
    /// For each edge \f$ e = (u, v_{\text{from}}) \f$, creates a new edge \f$ e' = (u, v_{\text{to}}) \f$
    /// and removes the old edge.
    ///
    /// @param from The source vertex \f$ v_{\text{from}} \f$ whose in-edges will be moved
    /// @param to The target vertex \f$ v_{\text{to}} \f$ that will receive the in-edges
    void moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to);

    /// Redirects all outgoing edges from a vertex.
    ///
    /// For each edge \f$ e = (v_{\text{from}}, u) \f$, creates a new edge \f$ e' = (v_{\text{to}}, u) \f$
    /// and removes the old edge.
    ///
    /// @param from The source vertex \f$ v_{\text{from}} \f$ whose out-edges will be moved
    /// @param to The target vertex \f$ v_{\text{to}} \f$ that will receive the out-edges
    void moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to);

    /// Removes a vertex if it has no incident edges.
    ///
    /// Checks if \f$ \deg(v) = 0 \f$ and removes \f$ v \f$ from the vertex list if so.
    ///
    /// @param vertex The vertex \f$ v \f$ to potentially remove
    /// @return true if the vertex was removed, false if it has incident edges
    bool removeIfIsolated(const VertexPtr &vertex);

    /// Embeds the simplicial complex in Euclidean space using force-directed layout.
    ///
    /// Uses gradient descent to find coordinates \f$ \{x_i\} \f$ that minimize:
    /// \f[
    /// L = \sum_{e_{ij} \in E} \left( \|x_i - x_j\|^2 - \ell_{ij}^2 \right)^2
    /// \f]
    ///
    /// @param dimensions The embedding dimension \f$ n \f$
    /// @param epsilon Convergence threshold for the optimization
    void embedEuclidean(int dimensions, double epsilon);

    /// Adds an observable to track during evolution.
    ///
    /// Observables are measured after each update step to monitor the system's properties.
    ///
    /// @param observable The observable \f$ \mathcal{O} \f$ to track
    void addObservable(const std::shared_ptr<Observable> &observable);

    // ========================================
    // Internal Management
    // ========================================

    /// Registers a simplex in the spacetime's internal data structures.
    ///
    /// Adds the simplex to:
    /// - The main simplex set \f$ \{\sigma\} \f$
    /// - The appropriate orientation-indexed sets (external or internal)
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to register
    /// @param internal true if the simplex is fully internal (all faces glued), false otherwise
    /// @return The registered simplex
    SimplexPtr registerSimplex(const SimplexPtr &simplex, bool internal);

    /// Unregisters a simplex from the spacetime's internal data structures.
    ///
    /// Removes the simplex from all indexed sets. This should be called before
    /// modifying a simplex's fingerprint to avoid hash table corruption.
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to unregister
    void unregisterSimplex(const SimplexPtr &simplex);
  private:
    std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();

    IdType vertexIdCounter = 0;
    SpacetimeType spacetimeType;
    double alpha = 1.;
    std::shared_ptr<Metric> metric;
    std::shared_ptr<Topology> topology;
    std::uint64_t currentTime = 0;

    SimplexSet simplices{};

    ///
    /// These are simplices on the boundary of a simplicial complex. They have at least one external face, and hence can
    /// be glued to other simplices. The externalSimplices are organized by the orientation of their available faces. If
    /// a face is available; the orientation of that face can be found as a key corresponding to a SimplexSet containing
    /// the Simplex to which that Face belongs.
    ///
    /// This makes for fast lookups when gluing simplices together to form a complex.
    std::unordered_map<SimplexOrientation, SimplexSet, SimplexOrientationHash, SimplexOrientationEq> externalSimplicesByFacialOrientation{};

    ///
    /// These are simplices that are fully internal to the simplicial complex. They have no external faces, and hence
    /// cannot be glued to other simplices.
    ///
    /// A Simplex becomes _internal_ when all it's _external_ faces have been glued. At that point it is no longer
    /// relevant to store that simplex by the orientation of any given face, so _internal_ simplices are stored by the
    /// orientation of the Simplex itself.
    std::unordered_map<SimplexOrientation, SimplexSet, SimplexOrientationHash, SimplexOrientationEq> internalSimplicesByOrientation{};
    std::vector<std::shared_ptr<Observable> > observables{};
};
} // caset

#endif //CASET_SPACETIME_H
