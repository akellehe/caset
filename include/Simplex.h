//
// Created by andrew on 10/21/25.
//

#ifndef CASET_SIMPLEX_H
#define CASET_SIMPLEX_H
#include <memory>
#include <vector>

#include "Edge.h"

namespace caset {

/// Simplex Class
///
/// A simplex is a generalization of the concept of a triangle or tetrahedron to arbitrary dimensions. Each simplex
/// is defined by it's edges. Each edge connects two vertices in spacetime.
///
/// Each simplex has a volume \f$ V_s \f$, which can represent various physical properties depending on the context.
///
class Simplex {
  public:
    explicit Simplex(const std::vector<std::shared_ptr<Edge>> &edges_) : edges(edges_) {}

    /// Computes the volume of the simplex, \f$ V_s \f$
    ///
    double getVolume() const {
      return 0;
    }

    /// Returns the hinges of the simplex. A hinge is a simplex contained within a higher dimensional simplex. The hinge
    /// is one dimension lower than the "parent" simplex.
    /// For a 4-simplex, \f$ \sigma = {v_0, ..., v_4} \f$ there are 10 edges and 10 triangular hinges.
    /// In this case a hinge is any triangle \f$ {v_i, v_j, v_k} \ff. There are \f$ \binom{5}{3} = 10 \f$ such
    /// triangles.
    ///
    /// The curvature at the hinge is the deficit angle.
    ///
    const std::vector<std::shared_ptr<Simplex>> getHinges() const;

    /// Assuming the simplex is a hinge; returns the deficit angle associated with the hinge.
    /// The deficit angle is given by:
    /// \f[
    /// \epsilon = 2 \pi - \sum_{\sigma \supset h} \theta_h^{(\sigma)}
    /// \f]
    /// or in english; the deficit angle is equal to \f$ 2 \pi \f$ minus the sum of the dihedral angles at the hinge.
    /// When the hinge is exterior; the \f$ 2 \pi \f$ is replaced with \f$ \pi \f$.
    ///
    const double getDeficitAngle() const;

  private:
    std::vector<std::shared_ptr<Edge>> edges;
};
}

#endif //CASET_SIMPLEX_H