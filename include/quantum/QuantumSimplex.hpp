// QuantumSimplex — Simplex subclass for the 5-vertex KI-interaction cell.
//
// A QuantumSimplex is a 5-vertex mesh::Simplex that carries the quantum
// states for each of its vertices and a record of the Koashi–Imoto
// decomposition that produced them. The five positions are:
//
//     A       — the input "left" system, with state ρ_A
//     B       — the input "right" system, with state ρ_B
//     Sigma   — the KI core ρ_{Σ_{AB}}
//     APrime  — the KI tail on the A side, ρ_{A'}
//     BPrime  — the KI tail on the B side, ρ_{B'}
//
// The simplex has all 10 edges of a 4-simplex (C(5, 2) = 10), each with
// length equal to the Van Raamsdonk metric d_VR = -log(I / I_max) where
// I is the mutual information between the two endpoint vertices' states.
//
// Storage: QuantumSimplex objects are value-stored in a process-static
// deque inside this library (stable addresses; never moved or freed for
// the life of the process). The Spacetime tracks them as base Simplex*
// pointers via Spacetime::registerSimplex(qs, /*internal=*/true). This
// keeps tessera_core free of Eigen and of any quantum-domain types.

#pragma once

#include "mesh/Simplex.h"
#include "quantum/KoashiImoto.hpp"
#include "spacetime/Spacetime.h"

#include <Eigen/Dense>

#include <array>
#include <map>
#include <utility>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

class QuantumSimplex : public ::tessera::mesh::Simplex {
  public:
    // Vertex roles inside the 5-vertex simplex. The integer values are
    // the canonical positions in the underlying ``vertices_`` order.
    enum Position : int {
        A       = 0,
        B       = 1,
        Sigma   = 2,
        APrime  = 3,
        BPrime  = 4,
    };

    // Static factory. Constructs the five vertices, the ten edges, and
    // the QuantumSimplex itself; registers the simplex with
    // ``spacetime`` via ``registerSimplex(..., /*internal=*/true)``.
    //
    // Inputs:
    //   spacetime — the Spacetime that owns the vertices / edges and
    //               registers the simplex.
    //   rhoAB     — the joint density matrix on H_A ⊗ H_B, given in
    //               (A ⊗ B) ordering (row index = a · dimB + b).
    //   dimA, dimB — the dimensions of the A and B Hilbert spaces.
    //   iMax      — the simulation-wide information ceiling used in the
    //               Van Raamsdonk metric d_VR = -log(I / iMax). Pass
    //               whatever convention the caller wants; common choices
    //               are log(dimA · dimB) (the maximum joint entropy) or
    //               2·log(min(dimA, dimB)) (the maximum mutual info).
    //   tol       — KI numerical tolerances; defaults to 1e-10 each.
    //
    // Returns: a non-owning pointer to the QuantumSimplex, whose
    // lifetime is the process. The caller does NOT delete this pointer.
    [[nodiscard]] static QuantumSimplex*
    fromKIInteraction(::tessera::spacetime::Spacetime& spacetime,
                      const Eigen::MatrixXcd&          rhoAB,
                      int                              dimA,
                      int                              dimB,
                      double                           iMax,
                      const KoashiImotoTolerances&     tol = {});

    // The vertex at position p (A, B, Sigma, APrime, BPrime).
    [[nodiscard]] ::tessera::mesh::VertexPtr
    vertexAt(Position p) const noexcept { return positions_[p]; }

    // The quantum state at position p, as a density matrix in the
    // vertex's local basis.
    [[nodiscard]] const Eigen::MatrixXcd&
    stateAt(Position p) const noexcept { return states_[p]; }

    // Mutual information (nats) between the two endpoint states for
    // the edge connecting positions p and q. Symmetric; either order
    // works.
    [[nodiscard]] double
    mutualInfoFor(Position p, Position q) const;

    // Van Raamsdonk distance d_VR (nats) for the edge connecting
    // positions p and q. Symmetric; either order works. Returns +∞
    // when the corresponding MI is zero (the edge is "disconnected"
    // in the metric, though the simplex still has the edge in its
    // topology).
    [[nodiscard]] double
    vanRaamsdonkDistanceFor(Position p, Position q) const;

    // The full KI decomposition that produced this simplex.
    [[nodiscard]] const KoashiImotoResult&
    kiResult() const noexcept { return kiResult_; }

    [[nodiscard]] double iMax() const noexcept { return iMax_; }

    // Public so std::deque can emplace one. Callers should use the
    // static factory above; this is only public to satisfy the
    // allocator_traits::construct path through emplace_back.
    QuantumSimplex(::tessera::spacetime::Spacetime*       spacetime,
                   const ::tessera::mesh::VertexPtrs&     verts,
                   ::tessera::mesh::Edges                 edges,
                   std::array<::tessera::mesh::VertexPtr, 5> positions,
                   std::array<Eigen::MatrixXcd, 5>           states,
                   std::map<std::pair<int, int>, double>     mi,
                   KoashiImotoResult                         ki,
                   double                                    iMax);

  private:

    std::array<::tessera::mesh::VertexPtr, 5> positions_;
    std::array<Eigen::MatrixXcd, 5>           states_;
    std::map<std::pair<int, int>, double>     mi_;
    KoashiImotoResult                         kiResult_;
    double                                    iMax_;
};

} // namespace tessera::quantum
