#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "Edge.h"
#include "spacetime/Spacetime.h"

namespace caset {
///
///
/// @param timeOrientation
enum class TimeOrientation : uint8_t {
  FUTURE = 0,
  PRESENT = 1,
  UNKNOWN = 2
};

class SimplexShape {
  public:
    ///
    /// The shape of a simplex is determined by how many vertices lie on the initial and final time slice for the
    /// simplex. The shape is largely only relevant for Lorentzian/CDT complexes where causality is preserved. Those
    /// complexes restrict to allowed shapes that ensure progression forward in time and "fit together" (so they share
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
    SimplexShape() : ti(0), tf(0) {
    }
    SimplexShape(uint8_t ti_, uint8_t tf_) : ti(ti_), tf(tf_) {
    }

    std::pair<uint8_t, uint8_t> getShape() {
      return {ti, tf};
    }

    bool operator==(const SimplexShape &other) const {
      return ti == other.ti && tf == other.tf;
    }

    TimeOrientation getOrientation() const {
      if (ti == tf) return TimeOrientation::UNKNOWN;
      if (ti > tf) return TimeOrientation::PRESENT;
      return TimeOrientation::FUTURE;
    }

  private:
    uint8_t ti;
    uint8_t tf;

    // Allow std::hash to access private members:
    // friend struct std::hash<SimplexShape>;
};

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

    Simplex(
      const std::shared_ptr<Spacetime> &spacetime_,
      const std::vector<std::shared_ptr<Vertex> > &vertices_
    ) : spacetime(spacetime_), shape(SimplexShape(0, 0)), vertices(vertices_) {
      if (vertices_.size() > kMax) throw std::length_error("Simplex: too many vertices");
      n_ = static_cast<std::uint8_t>(vertices_.size());
      for (std::size_t i = 0; i < n_; ++i) {
        ids_[i] = vertices_[i]->getId();
      }
      std::sort(ids_.begin(), ids_.begin() + n_);
      auto it = std::unique(ids_.begin(), ids_.begin() + n_);
      n_ = static_cast<std::uint8_t>(std::distance(ids_.begin(), it));
      h_ = compute_fingerprint(ids_.data(), n_);
      this->setShape();
    }

    void setShape() {
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
      shape = SimplexShape(tiVertexes, tfVertexes);
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

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::shared_ptr<Spacetime> spacetime;

    std::array<VertId, kMax> ids_{};
    std::uint8_t n_{0};
    std::uint64_t h_{kSeed};
    static constexpr std::uint64_t kSeed = 0xcbf29ce484222325ull;
    SimplexShape shape;

    static inline std::uint64_t mix64(VertId x) noexcept {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    static std::uint64_t compute_fingerprint(
      const VertId *ids,
      std::uint8_t n
    ) noexcept {
      std::uint64_t h = 0xcbf29ce484222325ull ^ n;
      for (std::uint8_t i = 0; i < n; ++i) {
        h ^= mix64(ids[i] + 0x9e3779b97f4a7c15ull);
        h *= 0x100000001b3ull; // FNV-ish step
      }
      return h;
    }

    friend struct std::hash<Simplex>;
};
}

#endif //CASET_SIMPLEX_H
