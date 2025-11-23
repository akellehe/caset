#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H

#include <memory>
#include <vector>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <coroutine>

#include <torch/torch.h>

#include "Logger.h"
#include "Edge.h"
#include "Fingerprint.h"
#include "Vertex.h"
#include "Logger.h"

namespace caset {
///
///
/// @param timeOrientation
enum class TimeOrientation : uint8_t {
  FUTURE = 0,
  PRESENT = 1,
  UNKNOWN = 2
};

class SimplexOrientation {
  public:
    ///
    /// The orientation of a simplex is determined by how many vertices lie on the initial and final time slice for the
    /// simplex. The orientation is largely only relevant for Lorentzian/CDT complexes where causality is preserved. Those
    /// complexes restrict to allowed orientations that ensure progression forward in time and "fit together" (so they share
    /// faces without gaps in the complex).
    ///
    /// The convention was established in Ambjorn-Loll's "Causal Dynamical Triangulations" paper from 1998-2001. Every
    /// d-simplex must have its vertices split across two adjacent time slices, t and t+1. That means every simplex has
    /// a split
    ///
    /// \f$ (n, d + 1 - n) \f$
    ///
    /// @param ti The number of vertices on the initial time slice.
    /// @param tf The number of vertices on the final time slice.
    ///
    SimplexOrientation(uint8_t ti_, uint8_t tf_) : ti(ti_), tf(tf_) {
      k = ti_ + tf_;
    }
    constexpr SimplexOrientation() noexcept = default;

    [[nodiscard]] std::pair<uint8_t, uint8_t> numeric() const {
      return {ti, tf};
    }

    constexpr bool operator==(const SimplexOrientation &other) const noexcept {
      return ti == other.ti && tf == other.tf;
    }

    [[nodiscard]] TimeOrientation getOrientation() const {
      if (ti == tf) return TimeOrientation::UNKNOWN;
      if (ti > tf) return TimeOrientation::PRESENT;
      return TimeOrientation::FUTURE;
    }

    uint8_t getK() const {
      return k;
    }

    static SimplexOrientation orientationOf(const Vertices &vertices) {
      uint8_t tiVertices = 0;
      uint8_t tfVertices = 0;
      double ti = std::numeric_limits<double>::max();
      double tf = -1;
      double initial = -1;
      int unassigned = 0;
      for (const auto vertex : vertices) {
        double t = vertex->getTime();
        ti = std::min(ti, t);
        tf = std::max(tf, t);
        if (ti == tf) {
          initial = t;
          unassigned++;
        } else if (t == ti) {
          tiVertices++;
        } else {
          tfVertices++;
        }
      }
      if (initial == ti) {
        tiVertices += unassigned;
      } else {
        tfVertices += unassigned;
      }
      return SimplexOrientation(tiVertices, tfVertices);
    }

  private:
    uint8_t ti{0};
    uint8_t tf{0};
    uint8_t k{0};
};
}

namespace std {
template<>
struct hash<caset::SimplexOrientation> {
  size_t operator()(const caset::SimplexOrientation &s) const noexcept {
    auto [ti, tf] = s.numeric(); // OK now that getOrientation() is const
    std::uint16_t packed = (std::uint16_t(ti) << 8) | std::uint16_t(tf);
    return std::hash<std::uint16_t>{}(packed); // perfect for all (ti, tf)
  }
};
}

namespace caset {
/// # Simplex Class
///
/// A simplex is a generalization of the concept of a triangle or tetrahedron to arbitrary dimensions. Each simplex
/// is defined by it's vertices and edges. Each edge connects two vertices in spacetime.
///
/// Each simplex has a volume \f$ V_s \f$, which can represent various physical properties depending on the context.
///
///
class Simplex : public std::enable_shared_from_this<Simplex> {
  public:
    ///
    /// @param vertices_
    Simplex(
      Vertices vertices_
    ) : orientation(SimplexOrientation(0, 0)), vertices(vertices_), fingerprint({}) {
      orientation = SimplexOrientation::orientationOf(vertices_);
      std::vector<IdType> ids = {};
      ids.reserve(vertices_.size());
      vertexIdLookup.reserve(vertices_.size());
      for (int i = 0; i < vertices_.size(); i++) {
        ids.push_back(vertices_[i]->getId());
        vertexIdLookup.insert({vertices_[i]->getId(), vertices_[i]});
      }
      fingerprint = Fingerprint(ids);
      computeEdges();
    }

    Simplex(
      Vertices vertices_,
      SimplexOrientation orientation_
    ) : orientation(orientation_), vertices(vertices_), fingerprint({}) {
      std::vector<IdType> ids = {};
      ids.reserve(vertices_.size());
      vertexIdLookup.reserve(vertices_.size());
      for (int i = 0; i < vertices_.size(); i++) {
        ids.push_back(vertices_[i]->getId());
        vertexIdLookup.insert({vertices_[i]->getId(), vertices_[i]});
      }
      fingerprint = Fingerprint(ids);
      computeEdges();
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "<";
      ss << std::to_string(getOrientation().getK());
      ss << "-Simplex (";
      for (const auto &v : vertices) {
        ss << v->toString() << "→";
      }
      ss << vertices[0]->toString() << ")>";
      return ss.str();
    }

    /// Computes the volume of the simplex, \f$ V_{\sigma} \f$
    ///
    double getVolume() const {
      return 0;
    }

    /// Returns the hinges of the simplex. A hinge is a simplex contained within a higher dimensional simplex. The hinge
    /// is one dimension lower than the "parent" simplex.
    /// For a 4-simplex, \f$ \sigma = {v_0, ..., v_4} \f$ there are 10 edges and 10 triangular hinges.
    /// In this case a hinge is any triangle \f$ {v_i, v_j, v_k} \f$. There are \f$ \binom{5}{3} = 10 \f$ such
    /// triangles.
    ///
    /// The curvature at the hinge is the deficit angle.
    ///
    const std::vector<std::shared_ptr<Simplex> > getHinges() const {
      return {};
    }

    /// Assuming the simplex is a hinge; returns the deficit angle associated with the hinge.
    ///
    /// The deficit angle is given by:
    ///
    /// \f[
    /// \epsilon = 2 \pi - \sum_{\sigma \supset h} \theta_h^{(\sigma)}
    /// \f]
    ///
    /// \f$ \theta_h^{(\sigma)} \f$ is the 4D dihedral angle between the two tetrahedral faces of simplex \f$ \sigma \f$
    /// meeting along triangle (hinge) \f$ h \f$.
    ///
    /// Or in english; the deficit angle is equal to \f$ 2 \pi \f$ minus the sum of the 4D dihedral angle of each
    /// simplex between the two tetrahedral faces meeting along triangle \f$ h \f$.
    ///
    /// When the hinge is exterior/on a boundary; the \f$ 2 \pi \f$ is replaced with \f$ \pi \f$.
    ///
    const double getDeficitAngle() const {
      return 0;
    }

    /// Compute dihedral angles from edge lengths.
    /// ///
    /// Let \f$ C \f$ be the cofactors of \f$ G \f$, \f$ C = cof(G) \f$ (a matrix of cofactors). Then the dihedral angle
    /// between the two tetrahedral faces opposite vertices \f$ i \f$ and \f$ j \f$ is given by:
    ///
    /// \f[
    /// cos(\theta_{ij}) = - \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}, i \neq j, i, j \in {0, ..., n}
    /// \f]
    ///
    /// Map \f$ (i, j) \f$ to the hinge (triangle for a 4-simplex) opposite that pair.
    ///
    const double computeDihedralAngles() const {
      return 0.;
    };

    [[nodiscard]] SimplexOrientation getOrientation() const noexcept {
      return orientation;
    }

    ///
    /// @return A list of Vertex (es) in traversal order. You can iterate these to walk the Face.
    [[nodiscard]] Vertices getVertices() const noexcept { return vertices; };

    [[nodiscard]] std::size_t size() const noexcept {
      return vertices.size();
    }

    [[nodiscard]] bool isTimelike() const {
      for (const auto &edge : edges) {
        const auto &src = vertexIdLookup.find(edge->getSourceId())->second;
        const auto &tgt = vertexIdLookup.find(edge->getTargetId())->second;
        if (src->getTime() != tgt->getTime()) {
          return false;
        }
      }
      return true;
    }

    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k) {
      if (k == 4) return 6;
      if (k == 3) return 3;
      if (k == 2) return 1;
      if (k == 0 || k == 1) return 0;

      int n = 0;
      for (int i = 0; i < k; i++) {
        n = n + i;
      }
      return n;
    }

    template<typename T>
    T binomial(unsigned n, unsigned k) {
      if (k > n) return 0;
      k = std::min(k, n - k);

      T result = 1;
      for (unsigned i = 1; i <= k; ++i) {
        result = result * (n - (k - i));
        result /= i;
      }

      return result;
    }

    ///
    /// A k-simplex is the convex hull of k + 1 affinely independent points. Each has faces of all dimensions from 0 up
    /// to k–1. A k-1 simplex is called a Facet.
    ///
    /// A j-face is a j-simplex incorporating a subset (of size j) of the k-simplex vertices.
    ///
    /// The number of j-faces ( \f$ \sigma^j \f$ ) of a k-simplex \f$ \sigma^k \f$ is given by
    ///
    /// \f[
    /// \binom{k+1}{j+1}
    /// \f]
    ///
    /// And the total number of faces of all dimensions is
    /// \sum_{j=0}^{k-1} \binom{k+1}{j+1} = 2^{k+1} - 2
    ///
    std::size_t getNumberOfFaces(std::size_t j) {
      auto k = getOrientation().getK();
      return binomial<std::size_t>(k + 1, j + 1);
    }

    ///
    ///
    /// A Face, \f$ \sigma^{k-1} \subset \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
    /// the k-simplex.
    ///
    /// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
    /// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
    /// by the orientation of those respective `Simplex`es.
    /// attach it
    ///
    /// The Facets are the \f$ \sigma^{k-1} \subset \sigma^{k} \f$ faces on which we'll most commonly join two simplices
    /// to form a simplicial complex \f$ K \f$.
    ///
    /// @return /// all k-1 simplices contained within this k-simplex.
    [[nodiscard]] std::vector<std::shared_ptr<Simplex> > getFacets() noexcept;

    Fingerprint fingerprint;

    ///
    /// The co-face of a k-simplex \f$ \sigma_i^k \f$ is another k-simplex, \f$ \sigma_j^k \f$ that shares a k-1 simplex
    /// \f$ \sigma^{k-1} \f$ with \f$ \sigma_i^k \f$.
    ///
    /// We define a face as a set of shared vertices. The face of any given k-simplex \f$ \sigma^k \f$ is a k-1 simplex,
    /// \f$ \sigma^{k-1} \f$ such that \f$ \sigma^{k-1} \subset \sigma^k \f$.
    void addCoface(const std::shared_ptr<Simplex> &simplex) {
      cofaces.insert(simplex);
    }

    [[nodiscard]] bool hasVertex(const IdType vertexId) const {
      return vertexIdLookup.contains(vertexId);
    }

    /// @returns Edges in traversal order (the order of input vertices).
    [[nodiscard]] Edges getEdges() const {
      return edges;
    }

    // TODO: Do we want to remove the vertex here as well? I don't think so, the method is just called removeEdge
    bool removeEdge(const EdgePtr &edge) {
      auto index = edgeIndexMap.find({edge->getSourceId(), edge->getTargetId()});
      if (index == edgeIndexMap.end()) return false;
      edges.erase(edges.begin() + index->second);
      edgeIndexMap.erase(index);
      return true;
    }

    bool removeVertex(const VertexPtr &vertex) {
      const long vertexIndex = vertexIndexLookup.find(vertex->getId())->second;
      if (vertexIndex > vertices.size()) return false;
      std::vector<IdType> vertexIds{};
      vertexIds.reserve(vertices.size() - 1);
      vertices.erase(vertices.begin() + vertexIndex);
      vertexIdLookup.erase(vertex->getId());
      vertexIndexLookup.erase(vertex->getId());
      for (const auto &v : vertices) {
        vertexIds.push_back(v->getId());
      }
      fingerprint.refreshFingerprint(vertexIds);
      return true;
    }

    [[nodiscard]]
    std::optional<Vertices>
    getVerticesWithParityTo(const std::shared_ptr<Simplex> &other) {
      const auto &mine = vertices;
      const auto &theirs = other->getVertices();

      const std::size_t n = mine.size();
      if (n != theirs.size()) {
        throw std::runtime_error("You can only compare simplices of the same size!");
      }
      if (n == 0) {
        return Vertices{}; // or std::nullopt, your call
      }
      if (n == 1) {
        if (mine[0]->getTime() != theirs[0]->getTime()) {
          return std::nullopt;
        }
        return mine; // already aligned
      }

      auto try_alignment =
          [&](std::size_t start,
              bool reversed)
        -> std::optional<Vertices> {
        CASET_LOG(INFO_LEVEL, "Trying alignment...");
        Vertices result;
        result.reserve(n);

        for (std::size_t k = 0; k < n; ++k) {
          std::size_t idx;
          if (!reversed) {
            // orientation-preserving: walk forward
            idx = (start + k) % n;
          } else {
            // orientation-reversing: walk backward
            // k = 0 -> idx = start
            // k = 1 -> idx = start - 1 (mod n)
            idx = (start + n - k) % n;
          }

          if (mine[idx]->getTime() != theirs[k]->getTime()) {
            CASET_LOG(INFO_LEVEL, "Alignment failed.");
            return std::nullopt; // mismatch, this alignment fails
          }

          result.push_back(mine[idx]);
        }
        CASET_LOG(INFO_LEVEL, "Alignment finished.");
        return result; // success
      };

      // Try all starting positions where times match theirs[0]
      for (std::size_t i = 0; i < n; ++i) {
        if (mine[i]->getTime() != theirs[0]->getTime()) {
          continue;
        }

        // 1. Try same orientation
        if (auto aligned = try_alignment(i, /*reversed=*/false)) {
          return aligned;
        }

        // 2. Try reversed orientation
        if (auto aligned_rev = try_alignment(i, /*reversed=*/true)) {
          return aligned_rev;
        }
      }

      // No alignment found
      return std::nullopt;
    }

    /// This method computes Edge (s) of the Simplex in traversal order. Note that the edges are effectively undirected
    /// since it can point either way as the direction relates to vertex order. So it's possible for e.g. vertices
    /// \f$ \{v_0, v_1, v_2\} \f$ to correspond to edges \f$ \{ e_{0 \rightarrow 1}, e_{2 \rightarrow 1)}, e_{2 \rightarrow 0} \} \f$
    void computeEdges() {
      auto nEdges = computeNumberOfEdges(getOrientation().getK());

      if (!edges.empty()) {
        edges = Edges{};
        edgeIndexMap = EdgeIndexMap{};
      }

      edges.reserve(nEdges);
      edgeIndexMap.reserve(nEdges);

      CASET_LOG(INFO_LEVEL, "Getting edges for simplex ", toString());
      VertexPtr origin = nullptr;

      // The direction of the edges can be either way; source -> target or target -> source. Just ensure we move across
      // the vertices in the correct order
      for (const auto &cursor : getVertices()) {
        for (const auto &e : cursor->getInEdges()) {
          if (hasVertex(e->getSourceId()) && hasVertex(e->getTargetId())) {
            CASET_LOG(INFO_LEVEL, "For vertex", cursor->toString(), " found in-edge ", e->toString());
            EdgeKey edgeKey{e->getSourceId(), e->getTargetId()};
            edgeIndexMap.insert_or_assign(edgeKey, edges.size());
            edges.push_back(e);
          }
        }
      }
    }

    ///
    /// @param vertexAId
    /// @param vertexBId
    /// @return
    [[nodiscard]] bool hasEdge(const IdType vertexAId, const IdType vertexBId) {
      return edgeIndexMap.contains({vertexAId, vertexBId}) || edgeIndexMap.contains({vertexBId, vertexAId});
    }

    [[nodiscard]] bool hasVertex(const VertexPtr &vertex) const {
      return vertexIdLookup.contains(vertex->getId());
    }

    ///
    /// Simplices have an orientation which is given by the ordering of its Vertex (es). For a k-simplex,
    /// \f$ \sigma^k = [v_0, v_1, ..., v_k] \f$ even permutations have the _same_ orientation. Odd permutations have
    /// _opposite_ orientation.
    ///
    /// In order to glue two simplices they must have opposite orientation.
    ///
    /// For Face (s); orientation is inherited from the parent Simplex.
    ///
    /// \f[
    /// \partial\sigma^k = \partial[v_0, v_1, ..., v_k] = \sum_{i=0}^k (-1)^i [v_0, v_1, ..., v_k]
    /// \f]
    ///
    /// This method counts the number of cycles that result from mapping one set of vertex IDs to another set. That
    /// number reflects the number of "swaps" of vertices required to get from one configuration to another. Each of
    /// those swaps changes the sign of the orientation once. An odd number of swaps gives an opposite orientation; an
    /// even number gives the same orientation.
    ///
    int8_t checkPairty(std::shared_ptr<Simplex> &other) {
      std::size_t K = vertices.size();

      // Build vertex -> position map for 'a'
      // For small K (≤4,5) you could linear search; this is generic.
      std::unordered_map<IdType, int> positionByVertexIdInA{};
      positionByVertexIdInA.reserve(K);
      for (int i = 0; i < K; ++i) {
        positionByVertexIdInA[vertices[i]->getId()] = i;
      }

      Vertices otherVertices = other->getVertices();
      std::vector<IdType> otherIds{};
      otherIds.reserve(K);
      for (int i = 0; i < K; ++i) {
        otherIds[i] = otherVertices[i]->getId();
      }

      std::vector<int> perm{};
      perm.reserve(K);
      for (int i = 0; i < K; ++i) {
        IdType otherId = otherIds[i];
        if (!positionByVertexIdInA.contains(otherId)) {
          CASET_LOG(WARN_LEVEL, "Other face contains ", otherId, " but this face does not!");
          for (auto &v : otherIds) {
            CASET_LOG(WARN_LEVEL, "Other face contains ", otherId);
          }
          for (auto &v : vertices) {
            CASET_LOG(WARN_LEVEL, "This face contains ", v->getId());
          }
          return 0;
        }
        perm[i] = positionByVertexIdInA[otherId];
      }

      // Count cycles of perm on {0..K-1}
      std::vector<bool> visited{};
      visited.reserve(K);
      for (int i = 0; i < K; i++) {
        visited[i] = false;
      }
      int cycles = 0;
      for (int i = 0; i < K; ++i) {
        if (visited[i]) continue;
        ++cycles;
        int j = i;
        while (!visited[j]) {
          visited[j] = true;
          j = perm[j];
        }
      }

      int N = K;
      int transpositionsMod2 = (N - cycles) & 1;
      return transpositionsMod2 ? -1 : +1;
    }

    ///
    /// Co-faces are maintained as state rather than computed on the fly. This means any time a Simplex is attached to
    /// another Simplex; it must be added to the face at which it's attached as a co-face. If a Simplex, Edge, or Vertex
    /// within that Face is removed at any point; that effect should cascade up the ownership tree, which goes
    /// \f[
    /// Vertex \subset Edge \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplices that share this face.
    [[nodiscard]] std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> getCofaces() const noexcept {
      return cofaces;
    }

    constexpr bool operator==(const Simplex &other) const noexcept {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    constexpr bool operator==(const std::shared_ptr<Simplex> &other) const noexcept {
      return fingerprint.fingerprint() == other->fingerprint.fingerprint();
    }

    /// This method replaces the vertex only, Edge (s) should be replaced by the Spacetime, because it maintains the
    /// global lookup for Edge (s). If the Edge source/target is replaced; it's not enough to update the Edge, since
    /// squaredLength data could be lost.
    ///
    void replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
      if (!hasVertex(oldVertex) || hasVertex(newVertex)) {
        return;
      }
      std::vector<IdType> vertexIds = {};
      vertexIds.reserve(vertices.size());
      for (int i = 0; i < vertices.size(); i++) {
        if (vertices[i]->getId() == oldVertex->getId()) {
          vertices[i] = newVertex;
          vertexIdLookup.erase(oldVertex->getId());
          vertexIdLookup.insert({newVertex->getId(), newVertex});
          vertexIds.push_back(oldVertex->getId());
          fingerprint.refreshFingerprint(vertexIds);
          return;
        }
      }
      throw std::runtime_error("This simplex does not contain the vertex you're trying to replace!");
    }

  private:
    // TODO: I think we can get rid of the vertexIndexLookup, the only place we appear to use it is removeVertex, which
    //  appears to be unused.
    VertexIndexMap vertexIndexLookup{};
    Vertices vertices{};
    SimplexOrientation orientation{};
    EdgeIndexMap edgeIndexMap{};
    Edges edges{};

    std::vector<std::shared_ptr<Simplex> > facets{};
    std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> cofaces{};
    static std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> facetRegistry;
    std::unordered_map<IdType, VertexPtr> vertexIdLookup{};
};

using SimplexPtr = std::shared_ptr<Simplex>;
using SimplexPair = std::pair<SimplexPtr, SimplexPtr>;
using OptionalSimplexPair = std::optional<SimplexPair>;
using Simplices = std::vector<SimplexPtr>;
}

namespace std {
template<>
struct hash<caset::Simplex> {
  size_t operator()(const caset::Simplex &s) const noexcept {
    return std::hash<std::uint64_t>{}(s.fingerprint.fingerprint());
  }
};
}

#endif //CASET_SIMPLEX_H
