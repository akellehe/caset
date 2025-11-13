#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H

#include <memory>
#include <vector>
#include <functional>

#include <torch/torch.h>

#include "Edge.h"
#include "Coface.h"
#include "Face.h"
#include "Fingerprint.h"

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

    static SimplexOrientation orientationOf(const std::vector<std::shared_ptr<Vertex> > &vertices) {
      uint8_t tiVertexes = 0;
      uint8_t tfVertexes = 0;
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
          tiVertexes++;
        } else {
          tfVertexes++;
        }
      }
      if (initial == ti) {
        tiVertexes += unassigned;
      } else {
        tfVertexes += unassigned;
      }
      return SimplexOrientation(tiVertexes, tfVertexes);
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
class Simplex {
  public:
    ///
    /// @param vertices_
    Simplex(
      std::vector<std::shared_ptr<Vertex> > vertices_,
      std::vector<std::shared_ptr<Edge> > edges_
    ) : orientation(SimplexOrientation(0, 0)), vertices(vertices_), edges(edges_), fingerprint({}) {
      orientation = SimplexOrientation::orientationOf(vertices_);
      std::vector<IdType> ids = {};
      ids.reserve(vertices_.size());
      for (const auto &vertex : vertices_) {
        ids.push_back(vertex->getId());
      }
      fingerprint = Fingerprint(ids);
    }

    Simplex(
      std::vector<std::shared_ptr<Vertex> > vertices_,
      std::vector<std::shared_ptr<Edge> > edges_,
      SimplexOrientation orientation_
    ) : orientation(orientation_), vertices(vertices_), edges(edges_), fingerprint({}){
      std::vector<IdType> ids = {};
      ids.reserve(vertices_.size());
      for (const auto &vertex : vertices_) {
        ids.push_back(vertex->getId());
      }
      fingerprint = Fingerprint(ids);
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

    SimplexOrientation getOrientation() const noexcept {
      return orientation;
    }

    std::vector<std::shared_ptr<Vertex> > getVertices() const noexcept {
      return vertices;
    }

    std::vector<std::shared_ptr<Edge> > getEdges() const noexcept {
      return edges;
    }

    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k) {
      // k=0 -> 0
      // k=1 -> 0
      // k=2 -> 1
      // k=3 -> 1 + 2 = 3
      // k=4 -> 1 + 2 + 3 = 6
      // k=5 -> 1 + 2 + 3 + 4 = 10
      // k=6 -> 1 + 2 + 3 + 4 + 5 = 15
      // k=7 -> 21
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

    ///
    /// This method is a simplicial isomorphism between two faces. Specifically; it takes two Simplex Face (s),
    /// \f$ \sigma^{k-1}_{myFace} \f$ and \f$ \sigma^{k-1}_{yourFace} \f$ as inputs and creates a new face
    /// \f$ \sigma^{k-1}_{newFace} \f$ represented by a Coface instance indicating their adjacency in the simplicial
    /// complex while preserving the orientation of their immediate parent Simplex (es).
    ///
    ///   1. First it checks to ensure `myFace` and `yourFace` are not already attached to another Simplex Face,
    ///   1. then it creates a new set of Vertex (es) to match the dimensionality of `myFace` and `yourFace`.
    ///   1. Next, it maps Vertex (es) \f$ \mathcal{V}_{myFace} \f$ and \f$ \mathcal{V}_{yourFace} \f$ of `myFace` and `yourFace` 1:1 respectively (2:1 in all) to the new vertices based on their TimeOrientation
    ///   1. Next, it validates the resulting Edge (es) are of compatible length within an Epsilon value.
    ///   1. Next, it maps Edges (es) \f$ \mathcal{E}_{myFace} \f$ and \f$ \mathcal{E}_{yourFace} \f$ of `myFace` and `yourFace` 1:1 respectively (2:1 in all) to the new Edges based on their shared Vertex (es).
    ///   1. Next, it removes the replaced Edge (s) and Vertex (es) from the Spacetime (simplicial complex).
    ///
    /// @param myFace The Face of this Simplex to attach to `yourFace` of the other Simplex
    /// @param yourFace The Face of the other Simplex to attach to `myFace` of this Simplex.
    /// @return
    std::shared_ptr<Coface> attachFaces(std::shared_ptr<Face> &myFace, std::shared_ptr<Face> &yourFace) {
      cofaces.insert(coface);
    }

    Fingerprint fingerprint;

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::vector<std::shared_ptr<Edge> > edges;

    SimplexOrientation orientation;

    std::unordered_set<std::shared_ptr<Coface>, CofaceHash, CofaceEq> cofaces;
};

using SimplexHash = VertexFingerprintHash<Simplex>;
using SimplexEq = VertexFingerprintEq<Simplex>;

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
