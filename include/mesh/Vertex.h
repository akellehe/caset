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

//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef TESSERA_TESSERA_SRC_VERTEX_H_
#define TESSERA_TESSERA_SRC_VERTEX_H_

#include <vector>
#include <memory>
#include <unordered_set>
#include "mesh/ForwardDeclarations.h"
#include "mesh/Fingerprint.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

///
/// \brief Represents a vertex in a causal set (causet) spacetime discretization
///
/// # Physical Context
///
/// In lattice gauge theory, vertices represent discrete points in spacetime where gauge fields
/// and matter fields are defined. The coupling parameters at each vertex determine the strength
/// of interactions:
///
/// - **Strong Force**: Described by quantum chromodynamics (QCD) with running coupling
///   \f$ \alpha_s(Q^2) \f$ that varies with energy scale \f$ Q^2 \f$
/// - **Weak Force**: Governed by the electroweak coupling \f$ g_W \f$
/// - **Electromagnetic Force**: Characterized by the fine structure constant \f$ \alpha_{EM} \approx 1/137 \f$
///
/// A key challenge is that QCD exhibits **asymptotic freedom**: the coupling becomes weaker at
/// high energies (short distances) and stronger at low energies (long distances), preventing
/// perturbative calculations in the infrared regime. This is modeled through "running coupling"
/// theories.
///
/// # Gauge Invariance
///
/// Observables in gauge theory must be gauge-invariant. The electromagnetic 4-potential
/// \f$ A_\mu \f$ is gauge-variant and thus not directly observable. However, field strengths
/// \f$ F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu \f$ are gauge-invariant observables.
///
/// # Implementation Details
///
/// This class represents vertices in a **directed graph** structure where:
/// - Edges have direction (source → target) to represent causal relationships
/// - Vertices can have coordinates in arbitrary dimensions (though time calculation has constraints)
/// - Each vertex maintains bidirectional edge lists (incoming and outgoing)
/// - Simplices are registered to their constituent vertices for efficient topology queries
///
/// The vertex class uses shared_from_this to enable safe shared_ptr creation from member functions.
///
class Vertex {
    public:
        // ========================================
        // Constructors
        // ========================================

        /// Default constructor creating a vertex with ID 0
        Vertex() noexcept;

        /// Virtual destructor — Vertex is a polymorphic base
        /// (QuantumVertex lives in tessera_quantum) and VertexList
        /// owns instances through std::unique_ptr<Vertex>, so the
        /// destructor must be virtual for safe polymorphic deletion.
        virtual ~Vertex() = default;

        ///
        /// \brief Construct vertex with ID and spatial coordinates
        /// \param id_ Unique identifier for this vertex
        /// \param coords Position in spacetime (arbitrary dimension)
        ///
        /// Creates a vertex with specified coordinates. The dimensionality is determined by coords.size():
        /// - 1D: Time is \f$ |x_0| \f$
        /// - 4D+: Time is \f$ \sqrt{\sum_{i=0}^{N-1} x_i^2} \f$
        /// - 2D, 3D: Invalid - getTime() will throw std::out_of_range
        ///
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept;

        ///
        /// \brief Construct coordinate-independent vertex with ID only
        /// \param id_ Unique identifier for this vertex
        ///
        /// Creates a vertex without coordinate information. Calling getCoordinates() on such
        /// a vertex will throw std::runtime_error.
        ///
        explicit Vertex(const std::uint64_t id_) noexcept;

        // ========================================
        // Core Properties
        // ========================================

        ///
        /// \brief Get the unique identifier of this vertex
        /// \return The vertex ID
        ///
        std::uint64_t getId() const noexcept;

        /// Set the vertex ID. Used by Spacetime::swapVertexLabels for
        /// Brunekreef vertex relabeling (Sec. 2.2.1, 2.3.1).
        /// WARNING: caller must update all containing data structures
        /// (VertexList, Simplex fingerprints, etc.) after calling this.
        void setId(std::uint64_t newId) noexcept { id = newId; }

        ///
        /// \brief Compute the temporal coordinate in arbitrary dimensions
        ///
        /// # Mathematical Definition
        ///
        /// The time coordinate is computed based on coordinate dimensionality:
        ///
        /// - **0D** (empty): Returns 0
        /// - **1D**: \f$ t = |x_0| \f$
        /// - **4D+**: \f$ t = \sqrt{\sum_{i=0}^{N-1} x_i^2} \f$ (Euclidean norm)
        /// - **2D, 3D**: Throws std::out_of_range (unsupported)
        ///
        /// # Rationale
        ///
        /// In standard 4D Minkowski spacetime, we typically separate time and space.
        /// For higher-dimensional theories (e.g., Kaluza-Klein, string theory), this
        /// convention uses the Euclidean magnitude across all temporal dimensions,
        /// with spatial dimensions handled separately by the embedding geometry.
        ///
        /// \return The time coordinate
        /// \throws std::out_of_range if coordinate vector has length 2 or 3
        ///
        [[nodiscard]] double getTime() const;

        void setTime(double time) noexcept;

        ///
        /// \brief Get the coordinate vector for this vertex
        ///
        /// \return Vector of coordinate values
        /// \throws std::runtime_error if vertex is coordinate-independent (empty coordinates)
        ///
        /// # Usage Notes
        ///
        /// Not all vertices need coordinates - some algorithms work purely with combinatorial
        /// structure. Only call this if you're certain the vertex has coordinate data.
        ///
        const std::vector<double> &getCoordinates() const;

        ///
        /// \brief Set new coordinates for this vertex
        /// \param coords New coordinate vector
        ///
        /// This operation does not update any cached values in edges or simplices.
        /// Use with caution if edge lengths depend on coordinates.
        ///
        void setCoordinates(const std::vector<double> &coords) noexcept;

        ///
        /// \brief Get the total degree (number of incident edges)
        /// \return Sum of in-degree and out-degree
        ///
        /// For a directed graph, this returns \f$ \deg(v) = \deg^-(v) + \deg^+(v) \f$
        ///
        std::size_t degree() const noexcept;

        // ========================================
        // Edge Management
        // ========================================

        ///
        /// \brief Find a specific edge incident to this vertex
        /// \param edge Edge to search for (compared by ID)
        /// \return Shared pointer to the edge if found, nullptr otherwise
        ///
        /// Searches both inEdges and outEdges. Useful for verifying edge membership
        /// without needing to know direction.
        ///
        EdgePtr getEdge(const EdgePtr &edge) const;

        ///
        /// \brief Get all incident edges (both incoming and outgoing)
        /// \return Set containing all edges where this vertex is source or target
        ///
        /// The returned vector is a copy. Complexity: O(|inEdges| + |outEdges|)
        ///
        Edges getEdges() const noexcept;

        ///
        /// \brief Get all edges targeting this vertex
        /// \return Set of incoming edges \f$ \{e \mid e.target = v\} \f$
        ///
        const Edges &getInEdges() const noexcept;

        ///
        /// \brief Get all edges originating from this vertex
        /// \return Set of outgoing edges \f$ \{e \mid e.source = v\} \f$
        ///
        const Edges &getOutEdges() const noexcept;

        ///
        /// \brief Add an incoming edge to this vertex
        /// \param edge Edge where this vertex is the target
        ///
        /// **Caveat**: Does not verify that edge->getTarget() == this. Caller must ensure consistency.
        ///
        void addInEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Add an outgoing edge from this vertex
        /// \param edge Edge where this vertex is the source
        ///
        /// **Caveat**: Does not verify that edge->getSource() == this. Caller must ensure consistency.
        ///
        void addOutEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Remove an incoming edge and update all affected simplices
        /// \param edge The edge to remove from inEdges
        /// \return Set of simplices that contained this edge (now modified)
        ///
        /// # Implementation Details
        ///
        /// 1. Removes the edge from all simplices that contain it via Simplex::removeEdge()
        /// 2. Removes the edge from this vertex's inEdges set
        /// 3. Returns affected simplices for caller to handle (e.g., re-validation)
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Aborts if edge is nullptr
        /// - Aborts if edge is not in inEdges
        ///
        void removeInEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Remove an outgoing edge and update all affected simplices
        /// \param edge The edge to remove from outEdges
        ///
        /// Symmetric to removeInEdge() but operates on outEdges.
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Aborts if edge is nullptr
        /// - Aborts if edge is not in outEdges
        ///
        void removeOutEdge(const EdgePtr &edge) noexcept;

        // ========================================
        // Simplex Management
        // ========================================

        ///
        /// \brief Get all simplices that contain this vertex
        /// \return Set of simplices where this vertex is a constituent
        ///
        /// A vertex belongs to a simplex if it's one of the simplex's vertices.
        /// This is the inverse relationship: vertex → simplices containing it.
        ///
        const Simplices &getSimplices() const noexcept;

        ///
        /// \brief Register a simplex as containing this vertex
        /// \param simplex The simplex to add
        /// \return true if simplex was newly added, false if already present
        ///
        /// # Duplicate Detection
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Checks for null simplex pointer
        /// - Checks for null 'this' pointer
        /// - Calls checkDuplicates() before and after insertion
        /// - Aborts on any violation
        ///
        /// This bidirectional relationship (Simplex ↔ Vertex) must be maintained
        /// consistently: when a simplex is created with vertices, it must call
        /// addSimplex() on each vertex.
        ///
        bool addSimplex(const SimplexPtr &simplex);

        ///
        /// \brief Unregister a simplex from this vertex
        /// \param simplex The simplex to remove
        /// \return true if simplex was removed, false if not found
        ///
        /// Called during simplex destruction or vertex replacement operations.
        /// Removes the simplex from the internal simplices set.
        ///
        bool removeSimplex(const SimplexPtr &simplex);

        ///
        /// \brief Debug utility to detect duplicate simplices
        /// \param msg Error message to log/throw if duplicates found
        ///
        /// **Assertions Only**: Only performs checks when TESSERA_ASSERTIONS is defined.
        /// Scans all registered simplices and checks for duplicate fingerprints.
        /// Logs at CRITICAL_LEVEL and throws std::runtime_error if duplicates exist.
        ///
        void checkDuplicates(const std::string &msg) const;

        // ========================================
        // Vertex Operations (Edge Migration)
        // ========================================

        ///
        /// \brief Move all edges (both in and out) to another vertex
        /// \param vertex Target vertex to receive edges
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// # Algorithm
        ///
        /// For each edge connected to this vertex:
        /// 1. Remove edge from source/target vertex's edge lists
        /// 2. Remove edge from spacetime's edge registry
        /// 3. Create new edge with 'vertex' substituted for 'this'
        /// 4. Insert new edge into spacetime
        ///
        /// This is equivalent to "redirecting" all edges to point to/from the new vertex
        /// while preserving edge properties (e.g., squared length).
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Throws std::runtime_error if spacetime is nullptr
        ///
        /// \see moveInEdgesTo(), moveOutEdgesTo()
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        ///
        /// \brief Move only incoming edges to another vertex
        /// \param vertex Target vertex to receive incoming edges
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// # Example
        ///
        /// Before: \f$ u \rightarrow \text{this} \f$
        /// After:  \f$ u \rightarrow \text{vertex} \f$
        ///
        /// The source vertices remain unchanged; only the target is redirected.
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Throws std::runtime_error if spacetime is nullptr
        /// - Verifies sourceVertex != this (logic error if violated)
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveInEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        ///
        /// \brief Move only outgoing edges to another vertex
        /// \param vertex Target vertex to become new source
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// # Example
        ///
        /// Before: \f$ \text{this} \rightarrow u \f$
        /// After:  \f$ \text{vertex} \rightarrow u \f$
        ///
        /// The target vertices remain unchanged; only the source is redirected.
        ///
        /// **Assertions**: When TESSERA_ASSERTIONS is defined:
        /// - Throws std::runtime_error if spacetime is nullptr
        /// - Verifies targetVertex != this (logic error if violated)
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveOutEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        // ========================================
        // Operators and Utilities
        // ========================================

        ///
        /// \brief Equality comparison based on vertex ID
        /// \param vertex Vertex to compare against
        /// \return true if IDs match, false otherwise
        ///
        /// Two vertices are considered equal iff they have the same ID, regardless
        /// of coordinates or topology. This is consistent with the hash function.
        ///
        bool operator==(const Vertex &vertex) const noexcept;

        ///
        /// \brief Generate human-readable string representation
        /// \return LaTeX-formatted UTF-8 string describing the vertex
        ///
        /// # Format
        ///
        /// When TESSERA_VERBOSE is defined:
        /// - Shows vertex ID, in-degree, out-degree, and time
        /// - Example: \f$ V_{42}^{in=3} _{out=5}~(t=1.0) \f$
        ///
        /// When not defined: returns empty string for performance
        ///
#ifdef TESSERA_VERBOSE
        std::string toString() const noexcept;
#else
        std::string toString() const noexcept {
            return "";
        };
#endif

        // ========================================
        // Public Members
        // ========================================

        ///
        /// \brief Fingerprint for hashing and equality testing
        ///
        /// The fingerprint is computed from the vertex ID and used in hash tables
        /// for SimplexPtrSet, VertexPtrSet, etc. This enables O(1) lookup.
        ///
        /// **Note**: This is public to allow direct access for performance-critical code.
        ///
        Fingerprint fingerprint;

        /// Index into VertexList::liveVec_ (maintained by VertexList).
        std::uint32_t liveIdx_{UINT32_MAX};

    private:
        // ========================================
        // Private Implementation
        // ========================================

        /// Edge direction for internal moveEdgesToImpl implementation
        enum class EdgeDirection { In, Out };

        ///
        /// \brief Internal implementation for moving edges
        /// \param recipient Target vertex
        /// \param spacetime Spacetime context
        /// \param direction Whether to move In or Out edges
        /// \return Pair of (old edges, new edges)
        ///
        std::pair<EdgePtrSet, EdgePtrSet>
        moveEdgesToImpl(const VertexPtr &recipient, Spacetime *spacetime, EdgeDirection direction);

        // ========================================
        // Private Members
        // ========================================

        Edges outEdges{};              ///< Edges where this vertex is the source
        Edges inEdges{};               ///< Edges where this vertex is the target
        Simplices simplices{};         ///< Simplices containing this vertex
        std::uint64_t id;              ///< Unique identifier
        std::vector<double> coordinates{};  ///< Spacetime position (may be empty)
};

}  // namespace tessera::mesh

// ========================================
// Standard Library Specializations
// ========================================

namespace std {

///
/// \brief Hash function specialization for tessera::mesh::Vertex
///
/// Enables Vertex objects to be used as keys in std::unordered_set and std::unordered_map.
/// The hash is computed from the vertex ID, ensuring consistent hashing across equal vertices.
///
/// # Complexity
/// O(1) - delegates to std::hash<std::uint64_t>
///
template<>
struct hash<tessera::mesh::Vertex> {
    size_t operator()(const tessera::mesh::Vertex &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex.getId());
    }
};

///
/// \brief Hash function specialization for tessera::mesh::Vertex*
///
/// Enables VertexPtr (Vertex*) to be used as keys in hash tables.
/// Hashes the underlying vertex ID, not the pointer address.
///
/// # Important
/// Two pointers pointing to vertices with the same ID will hash to the same value,
/// even if they are different pointer instances.
///
template<>
struct hash<tessera::mesh::Vertex*> {
    size_t operator()(tessera::mesh::Vertex* const &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex->getId());
    }
};

///
/// \brief Equality comparison specialization for tessera::mesh::Vertex
///
/// Used by standard library containers to compare Vertex objects.
/// Two vertices are equal iff they have the same ID.
///
/// # Note
/// This is consistent with the hash specialization above.
///
template<>
struct equal_to<tessera::mesh::Vertex> {
    size_t operator()(const tessera::mesh::Vertex &a, const tessera::mesh::Vertex &b) const noexcept {
        return a.getId() == b.getId();
    }
};

///
/// \brief Equality comparison specialization for tessera::mesh::Vertex*
///
/// Compares vertices by ID, not by pointer address.
/// Consistent with the hash specialization for Vertex*.
///
template<>
struct equal_to<tessera::mesh::Vertex*> {
    size_t operator()(tessera::mesh::Vertex* const &a, tessera::mesh::Vertex* const &b) const noexcept {
        return a->getId() == b->getId();
    }
};

}  // namespace std
#endif //TESSERA_TESSERA_SRC_VERTEX_H_

