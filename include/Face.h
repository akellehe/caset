//
// Created by Andrew Kelleher on 11/13/25.
//

#ifndef CASET_FACE_H
#define CASET_FACE_H
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include "Fingerprint.h"
#include "Vertex.h"
#include "Logger.h"

namespace caset {
class Simplex;

///
/// # Face
/// A Face, \f$ \sigma^{k-1} \subset \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
/// the k-simplex.
///
/// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
/// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
/// by the orientation of those respective `Simplex`es.
/// attach it
class Face {
  public:
    Face(std::vector<std::shared_ptr<const Simplex> > cofaces_,
         std::vector<std::shared_ptr<Vertex> > vertices_) : cofaces(cofaces_), vertices(vertices_), fingerprint({}) {
      vertexIdLookup.reserve(vertices_.size());
      std::vector<IdType> ids{};
      ids.reserve(vertices_.size());
      for (int i = 0; i < vertices.size(); i++) {
        ids.push_back(vertices[i]->getId());
        vertexIdLookup.insert({vertices[i]->getId(), vertices[i]});
      }
      fingerprint = Fingerprint(ids);
    }

    ///
    /// Simplexes have an orientation which is given by the ordering of its Vertex (es). For a k-simplex,
    /// \f$ \sigma^k = [v_0, v_1, ..., v_k] \f$ even permutations have the _same_ orientation. Odd permutations have
    /// _opposite_ orientation.
    ///
    /// In order to glue two simplexes they must have opposite orientation.
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
    int8_t checkPairty(std::shared_ptr<Face> &other) {
      std::size_t K = vertices.size();

      // Build vertex -> position map for 'a'
      // For small K (≤4,5) you could linear search; this is generic.
      std::unordered_map<IdType, int> positionByVertexIdInA{};
      positionByVertexIdInA.reserve(K);
      for (int i = 0; i < K; ++i) {
        positionByVertexIdInA[vertices[i]->getId()] = i;
      }

      std::vector<std::shared_ptr<Vertex> > otherVertexes = other->getVertices();
      std::vector<IdType> otherIds{};
      otherIds.reserve(K);
      for (int i = 0; i < K; ++i) {
        otherIds[i] = otherVertexes[i]->getId();
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

    [[nodiscard]] std::size_t size() const noexcept {
      return vertices.size();
    }

    ///
    /// The co-face of a k-simplex \f$ \sigma_i^k \f$ is another k-simplex, \f$ \sigma_j^k \f$ that shares a k-1 simplex
    /// \f$ \sigma^{k-1} \f$ with \f$ \sigma_i^k \f$.
    ///
    /// We define a face as a set of shared vertices. The face of any given k-simplex \f$ \sigma^k \f$ is a k-1 simplex,
    /// \f$ \sigma^{k-1} \f$ such that \f$ \sigma^{k-1} \subset \sigma^k \f$.
    void addCoface(const std::shared_ptr<const Simplex> &simplex) {
      cofaces.push_back(simplex);
    };

    ///
    /// This method runs within the context of an n-dimensional simplicial manifold; each (n-1) simplex (where faces are
    /// codimension-1) is incident to exactly 2 n-simplices for interior faces and exactly 1 n-simplex for faces along
    /// the boundary.
    [[nodiscard]] bool isAvailable() const {
      if (cofaces.size() < 2) {
        // For an interior simplex.
        return true;
      }
      return false;
    }

    [[nodiscard]] std::string toString() const {
      std::stringstream ss;
      ss << "<Face (";
      for (const auto &v : vertices) {
        ss << v->toString() << "→";
      }
      ss << vertices[0]->toString() << ")>";
      return ss.str();
    }

    [[nodiscard]] bool hasVertex(const IdType vertexId) const {
      return vertexIdLookup.contains(vertexId);
    }

    /// TODO: This method needs some optimization. We should build an edge lookup, but I'm having a little trouble doing
    /// that at the moment.
    ///
    /// @param vertexAId
    /// @param vertexBId
    /// @return
    [[nodiscard]] bool hasEdge(const IdType vertexAId, const IdType vertexBId) {
      if (!hasVertex(vertexAId) || !hasVertex(vertexBId)) {
        return false;
      }
      for (const auto &edge : getEdges()) {
        IdType sid = edge->getSourceId();
        IdType tid = edge->getTargetId();
        if (vertexAId == sid || vertexAId == tid) {
          if (vertexBId == sid || vertexBId == tid) {
            return true;
          }
        }
      }
      return false;
    }

    [[nodiscard]] bool hasVertex(const std::shared_ptr<Vertex> &vertex) const {
      return vertexIdLookup.contains(vertex->getId());
    }

    /// This method returns Edge (s) of the Simplex in traversal order. Note that the edges are effectively undirected
    /// since it can point either way as the direction relates to vertex order. So it's possible for e.g. vertices
    /// \f$ \{v_0, v_1, v_2\} \f$ to correspond to edges \f$ \{ e_{0 \rightarrow 1}, e_{2 \rightarrow 1)}, e_{2 \rightarrow 0} \} \f$
    [[nodiscard]] std::vector<const std::shared_ptr<Edge>> getEdges() {
      if (!edges.empty()) return edges;
      std::shared_ptr<Vertex> origin = nullptr;

      // The direction of the edges can be either way; source -> target or target -> source. Just ensure we move across
      // the vertexes in the correct order
      for (int currentIndex = 0; currentIndex < vertices.size(); ++currentIndex) {
        const std::shared_ptr<Vertex> cursor = vertices[currentIndex];
        for (const auto &e : cursor->getInEdges()) {
          IdType cursorId = cursor->getId();
          IdType sourceId = e->getSourceId();
          IdType targetId = e->getTargetId();
          if (sourceId == cursorId || targetId == cursorId) {
            const std::shared_ptr<Vertex> &next = vertices[(currentIndex + 1) % vertices.size()];
            const IdType nextId = next->getId();
            if (nextId == sourceId || nextId == targetId) {
              edges.push_back(e);
            }
          }
        }
        for (const auto &e : cursor->getOutEdges()) {
          IdType cursorId = cursor->getId();
          IdType sourceId = e->getSourceId();
          IdType targetId = e->getTargetId();
          if (sourceId == cursorId || targetId == cursorId) {
            const std::shared_ptr<Vertex> &next = vertices[(currentIndex + 1) % vertices.size()];
            const IdType nextId = next->getId();
            if (nextId == sourceId || nextId == targetId) {
              edges.push_back(e);
            }
          }
        }
      }
      return edges;
    }

    [[nodiscard]] bool isTimelike() const {
      double lower = std::numeric_limits<double>::infinity();
      double upper = -std::numeric_limits<double>::infinity();
      for (const auto &vertex : vertices) {
        lower = std::min(lower, vertex->getTime());
        upper = std::max(upper, vertex->getTime());
      }
      return lower < upper;
    }

    ///
    /// Co-faces are maintained as state rather than computed on the fly. This means any time a Simplex is attached to
    /// another Simplex; it must be added to the face at which it's attached as a co-face. If a Simplex, Edge, or Vertex
    /// within that Face is removed at any point; that effect should cascade up the ownership tree, which goes
    /// \f[
    /// Vertex \subset Edge \subset Face \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplexes that share this face.
    [[nodiscard]] std::vector<std::shared_ptr<const Simplex> > getCofaces() const noexcept { return cofaces; }

    ///
    /// @return A list of Vertex (es) in traversal order. You can iterate these to walk the Face.
    [[nodiscard]] std::vector<std::shared_ptr<Vertex> > getVertices() const noexcept { return vertices; };

    Fingerprint fingerprint;

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::vector<std::shared_ptr<const Simplex> > cofaces;
    std::vector<const std::shared_ptr<Edge>> edges;
    std::unordered_map<IdType, std::shared_ptr<Vertex> > vertexIdLookup{};
};

using FaceHash = FingerprintHash<Face>;
using FaceEq = FingerprintEq<Face>;
} // caset

namespace std {
template<>
struct hash<caset::Face> {
  size_t operator()(const caset::Face &face) const noexcept {
    return std::hash<std::uint64_t>{}(face.fingerprint.fingerprint());
  }
};
}

#endif //CASET_FACE_H
