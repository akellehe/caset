// QuantumVertex — mesh::Vertex carrying a density matrix.
//
// QuantumVertex extends mesh::Vertex with an Eigen-typed density
// matrix in its local Hilbert space. The matrix dimension is set at
// construction time (no compile-time template), so different
// QuantumVertex objects in the same VertexList can carry states of
// different sizes — e.g. the A, B vertices of a KI-interaction cell
// carry ρ_A, ρ_B (dim d_A, d_B), the Σ vertex carries ρ_Σ on the KI
// core, and the A', B' tail vertices carry ρ_{A'}, ρ_{B'}.
//
// QuantumVertex lives in tessera_quantum because density matrices
// require Eigen, and tessera_core is deliberately Eigen-free.
// VertexList stores its instances polymorphically through
// std::unique_ptr<Vertex>, so a QuantumVertex* can be downcast from
// a Vertex* via dynamic_cast or via the helper ``require``.

#pragma once

#include "mesh/Vertex.h"

#include <Eigen/Dense>

#include <stdexcept>
#include <utility>
#include <vector>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

class QuantumVertex : public ::tessera::mesh::Vertex {
  public:
    // Default constructor — empty state, id = 0. Needed so the pool
    // free-list path that placement-constructs into a slot has a
    // valid default. State must be assigned before use.
    QuantumVertex() noexcept = default;

    // Construct with id and a density matrix.
    QuantumVertex(std::uint64_t id, Eigen::MatrixXcd state) noexcept
      : Vertex(id), state_(std::move(state)) {}

    QuantumVertex(std::uint64_t id,
                  const std::vector<double>& coords,
                  Eigen::MatrixXcd state) noexcept
      : Vertex(id, coords), state_(std::move(state)) {}

    /// The vertex's quantum state ρ, in its local basis.
    [[nodiscard]] const Eigen::MatrixXcd& getState() const noexcept {
      return state_;
    }

    /// Replace the density matrix. Used by KI factories that
    /// allocate a vertex first and fill in its state once the
    /// decomposition is computed.
    void setState(Eigen::MatrixXcd state) noexcept {
      state_ = std::move(state);
    }

    [[nodiscard]] int stateDim() const noexcept {
      return static_cast<int>(state_.rows());
    }

    /// Compute the Van Raamsdonk distance d_VR = −log(I / iMax)
    /// between this vertex and ``other``. I is the mutual
    /// information of the bipartite *product joint*
    /// ρ_this ⊗ ρ_other, so this method gives the correct d_VR for
    /// any pair that has no inherited correlation (which is every
    /// pair the KI factories build except the input (A, B) edge).
    /// Two product marginals have I = 0, so the returned distance
    /// is +∞ except in trivial cases.
    ///
    /// The KI-cell factory uses this method for the nine
    /// non-(A, B) edges and computes the (A, B) edge separately
    /// from its input joint ρ_AB. The factory then writes
    /// d_VR² into ``Edge::squaredLength`` (the canonical edge
    /// length-property field).
    ///
    /// ``other`` must be a QuantumVertex — throws otherwise.
    [[nodiscard]] double
    vanRaamsdonkDistanceTo(const ::tessera::mesh::Vertex* other,
                           double                          iMax) const;

    /// The Van Raamsdonk metric law: the **spacelike** signed squared length
    /// for a given mutual information ``I`` — the value to store in
    /// ``Edge::squaredLength`` for a same-time-slice edge. Returns
    /// (−log(I/iMax))², with the length floored to −log(epsilon) (so the
    /// squared length is finite) when I < epsilon·iMax, and likewise when
    /// iMax ≤ 0 or I ≤ 0. Always ≥ 0 (spacelike). This is the static law used
    /// when a mutual information is already in hand; ``vanRaamsdonkDistanceTo``
    /// computes I from the product joint of two vertices.
    [[nodiscard]] static double
    vanRaamsdonkSquaredLength(double I, double iMax,
                              double epsilon = 1e-10) noexcept;

    /// Time-aware Van Raamsdonk **signed squared length** to ``other`` for
    /// the one-forward-step convention — the value written into
    /// ``Edge::squaredLength`` (spacelike > 0, null = 0, timelike < 0):
    ///
    ///   • different time slices (a forward-time *worldline* edge, e.g.
    ///     t=0 → t=1): returns 0 (null). A system propagating to its own
    ///     future is perfectly correlated with itself (I = iMax ⇒
    ///     d_VR = 0); its timelike content is carried by rapidity in the
    ///     Regge action, not by a spacelike length.
    ///   • same time slice (a spacelike edge): returns +d_VR² with
    ///     d_VR = −log(I/iMax), floored so d_VR ≤ −log(epsilon) (finite)
    ///     when I < epsilon·iMax.
    ///
    /// The time slice is read from ``Vertex::getTime`` (a vertex with no
    /// coordinates reads as t=0, the boundary slice). ``other`` must be a
    /// QuantumVertex — throws otherwise.
    [[nodiscard]] double
    vanRaamsdonkSquaredLengthTo(const ::tessera::mesh::Vertex* other,
                                double                          iMax,
                                double                          epsilon = 1e-10) const;

    /// Convenience downcast: pulls a QuantumVertex* out of a
    /// Vertex* via dynamic_cast and throws if the dynamic type is
    /// wrong.
    static QuantumVertex* require(::tessera::mesh::Vertex* v) {
      auto* qv = dynamic_cast<QuantumVertex*>(v);
      if (qv == nullptr) {
        throw std::invalid_argument(
          "QuantumVertex::require: vertex is not a QuantumVertex");
      }
      return qv;
    }

  private:
    Eigen::MatrixXcd state_{};
};

} // namespace tessera::quantum
