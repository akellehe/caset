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
      std::vector<IdType> ids{};
      ids.reserve(vertices_.size());
      for (const auto &v : vertices_) {
        ids.push_back(v->getId());
      }
      fingerprint = Fingerprint(ids);
    }

    ///
    /// Simplexes have an orientation which is given by the ordering of its Vertex (es). For a k-simplex, \f$ \sigma^k = [v_0, v_1, ..., v_k] \f$
    /// even permutations have the _same_ orientation. Odd permutations have _opposite_ orientation.
    ///
    /// For Face (s); orientation is inherited from the parent Simplex.
    ///
    /// \f[
    /// \partial\sigma^k = \partial[v_0, v_1, ..., v_k] = \sum_{i=0}^k (-1)^i [v_0, v_1, ..., v_k]
    /// \f]
    ///
    /// Because Simplices are constructed via coning; the ordering of the Simplex `vertices` is a little tricky, but
    /// follows a predictable ordering.
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
      std::cout << "Reserving " << K << " elements for perm" << std::endl;
      for (int i = 0; i < K; ++i) {
        IdType otherId = otherIds[i];
        if (!positionByVertexIdInA.contains(otherId)) return 0;
        std::cout << "Setting position " << i << " to " << positionByVertexIdInA[otherId] << std::endl;
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

      std::cout << "Perms: " << std::endl;
      for (int i = 0; i < K; ++i) {
        std::cout << perm[i] << std::endl;
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
    [[nodiscard]] std::vector<std::shared_ptr<Vertex> > getVertices() const noexcept { return vertices; };
    Fingerprint fingerprint;

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::vector<std::shared_ptr<const Simplex> > cofaces;
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
