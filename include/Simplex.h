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
    SimplexShape() : ti(0), tf(0) {}
    SimplexShape(uint8_t ti_, uint8_t tf_) : ti(ti_), tf(tf_) {}

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
    Simplex(
      const std::shared_ptr<Spacetime> &spacetime_,
      const std::vector<std::shared_ptr<Vertex> > &vertices_,
      const std::vector<std::shared_ptr<Edge> > &edges_
    ) : spacetime(spacetime_), shape(SimplexShape(0, 0)), vertices(vertices_), edges(edges_) {
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

    ///
    /// Builds the Cayley-Menger matrix using pairwise distances.

    ///
    /// Builds the Gram matrix \f$ G \f$ of edge vectors (this is Euclidean) using squared lengths.
    ///
    /// \f[
    /// G_{ij} = \frac{1}{2} (l_{0i}^2 + l_{0j}^2 - l_{ij}^2), i, j \in {1, ..., n}
    /// \f]
    ///
    /// where \f$ n \f$ is the dimension of the simplex (e.g., for a 4-simplex, \f$ n = 4 \f$) and \f$ l_{ij} \f$ is the
    /// distance between vertices \f$ i \f$ and \f$ j \f$.
    ///
    /// We expect \f$ G_{ij} \f$ to be positive-definite for non-degenerate Euclidean 4-simplices (\f$ det(G_{ij}) > 0 \f$).
    ///
    torch::Tensor getGramMatrix() const {
      // Compute the Gram matrix G from the edge lengths.
      torch::Tensor gramMatrix;
      auto edgeList = spacetime->getEdgeList();
      auto nEdges = edgeList->size();
      for (int i = 0; i < nEdges; i++) {
        for (int j = 0; j < nEdges; j++) {
          // G_ij = 1/2 (l_0i^2 + l_0j^2 - l_ij^2)
          double l_0i = spacetime->getMetric()->getSquaredLength(edgeList->get(i));
          // TODO: Not sure if gram matrix works with squared lengths.
          double l_0j = spacetime->getMetric()->getSquaredLength(edgeList->get(j));
          double l_ij = spacetime->getMetric()->getSquaredLength(edgeList->get(std::abs(i - j)));
          // Placeholder; replace with actual edge lookup
          gramMatrix[i][j] = 0.5 * (l_0i * l_0i + l_0j * l_0j - l_ij * l_ij);
        }
      }
      return gramMatrix;
    }

    ///
    /// @param gramMatrix The Gram matrix constructed from the edge lengths of the simplex.
    /// @returns The cofactor matrix of the Gram matrix.
    ///
    torch::Tensor getGramCofactor(const torch::Tensor &gramMatrix) const {
      TORCH_CHECK(gramMatrix.dim() >= 2, "G must be at least 2D");
      auto n = gramMatrix.size(-1);
      TORCH_CHECK(gramMatrix.size(-2) == n, "G must be square in the last two dims");
      TORCH_CHECK(gramMatrix.dtype() == torch::kFloat || gramMatrix.dtype() == torch::kDouble,
                  "G must be float/double");

      // Cholesky factorization (lower-triangular L with G = L L^T)
      // If your data might include near-singular simplices, catch exceptions and handle them.
      auto L = torch::linalg_cholesky(gramMatrix);

      // det(G) = (prod(diag(L)))^2 ; compute in log-space to avoid overflow
      auto diagL = L.diagonal(0, -2, -1);
      auto logdetG = 2.0 * diagL.abs().log().sum(/*dim=*/-1, /*keepdim=*/true);
      auto detG = logdetG.exp(); // shape (..., 1)

      // Inverse via Cholesky (much stabler than generic inverse for PD matrices)
      auto Ginv = torch::cholesky_inverse(L); // (..., n, n)

      // Cofactor = det(G) * (G^{-1})^T
      auto cof = detG.unsqueeze(-1) * Ginv.transpose(-2, -1); // broadcast det
      return cof;
    }

  private:
    std::vector<std::shared_ptr<Vertex> > vertices;
    std::vector<std::shared_ptr<Edge> > edges;
    std::shared_ptr<Spacetime> spacetime;
    SimplexShape shape;
};
}

#endif //CASET_SIMPLEX_H
