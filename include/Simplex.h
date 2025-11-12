#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H

#include <memory>
#include <vector>
#include <functional>

#include <torch/torch.h>

#include "Edge.h"

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
    using VertId = std::uint64_t;

    static constexpr std::size_t kMax = 64; // supports K <= 64

    static inline std::uint64_t mix64(VertId x) noexcept {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    static std::tuple<std::uint8_t, std::uint64_t, std::array<VertId, kMax> > computeFingerprint(
      const std::vector<std::shared_ptr<Vertex> > &vertices_) {
      if (vertices_.size() > kMax) throw std::length_error("Simplex: too many vertices");
      std::uint8_t n = static_cast<std::uint8_t>(vertices_.size());
      std::array<VertId, kMax> ids{};
      for (std::size_t i = 0; i < n; ++i) {
        ids[i] = vertices_[i]->getId();
      }
      std::sort(ids.begin(), ids.begin() + n);
      auto it = std::unique(ids.begin(), ids.begin() + n);
      n = static_cast<std::uint8_t>(std::distance(ids.begin(), it));
      std::uint64_t h = 0xcbf29ce484222325ull ^ n;
      for (std::uint8_t i = 0; i < n; ++i) {
        h ^= mix64(ids[i] + 0x9e3779b97f4a7c15ull);
        h *= 0x100000001b3ull; // FNV-ish step
      }
      return {h, n, ids};
    }

    ///
    /// @param vertices_
    Simplex(
      std::vector<std::shared_ptr<Vertex> > vertices_,
      std::vector<std::shared_ptr<Edge> > edges_
    ) : orientation(SimplexOrientation(0, 0)), vertices(vertices_), edges(edges_) {
      setFingerprint(vertices_);
      std::cout << "Vertices going into simplex: " << std::endl;
      for (const auto &v : vertices) {
        std::cout << "Vertex: " << v->getId() << std::endl;
      }
      std::cout << std::endl;
      orientation = SimplexOrientation::orientationOf(vertices_);
    }

    Simplex(
      std::vector<std::shared_ptr<Vertex> > vertices_,
      std::vector<std::shared_ptr<Edge> > edges_,
      SimplexOrientation orientation_
    ) : orientation(orientation_), vertices(vertices_), edges(edges_) {
      setFingerprint(vertices_);
    }

    void setFingerprint(const std::vector<std::shared_ptr<Vertex> > &vertices_) {
      std::tie(h_, n_, ids_) = Simplex::computeFingerprint(vertices_);
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

    bool operator==(const Simplex &o) const noexcept {
      if (n_ != o.n_) return false;
      if (h_ != o.h_) return false; // fast reject
      return std::memcmp(ids_.data(), o.ids_.data(), n_ * sizeof(VertId)) == 0;
    }
    bool operator!=(const Simplex &o) const noexcept { return !(*this == o); }

    std::uint64_t fingerprint() const noexcept { return h_; }

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

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::vector<std::shared_ptr<Edge> > edges;

    std::array<VertId, kMax> ids_{};
    std::uint8_t n_{0};
    std::uint64_t h_{kSeed};
    static constexpr std::uint64_t kSeed = 0xcbf29ce484222325ull;
    SimplexOrientation orientation;
};

struct SimplexHash {
  using is_transparent = void; // enables heterogeneous lookup
  size_t operator()(const Simplex &s) const noexcept { return size_t(s.fingerprint()); }
  size_t operator()(const std::shared_ptr<Simplex> &s) const noexcept { return size_t(s->fingerprint()); }
  size_t operator()(uint64_t fp) const noexcept { return size_t(fp); }
};
struct SimplexEq {
  using is_transparent = void;
  bool operator()(const Simplex &a, const Simplex &b) const noexcept { return a == b; }
  bool operator()(const Simplex &a, uint64_t fp) const noexcept { return a.fingerprint() == fp; }
  bool operator()(uint64_t fp, const Simplex &a) const noexcept { return fp == a.fingerprint(); }

  bool operator()(const std::shared_ptr<Simplex> &a, const std::shared_ptr<Simplex> &b) const noexcept {
    return a->fingerprint() == b->fingerprint();
  }
  bool operator()(const std::shared_ptr<Simplex> &a, uint64_t fp) const noexcept { return a->fingerprint() == fp; }
  bool operator()(uint64_t fp, const std::shared_ptr<Simplex> &a) const noexcept { return fp == a->fingerprint(); }
};
}

namespace std {
template<>
struct hash<caset::Simplex> {
  size_t operator()(const caset::Simplex &s) const noexcept {
    return std::hash<std::uint64_t>{}(s.fingerprint());
  }
};
}

#endif //CASET_SIMPLEX_H
