// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_TESSERA_SRC_EDGE_H_
#define TESSERA_TESSERA_SRC_EDGE_H_

#include "mesh/Fingerprint.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/EdgeKey.h"
// walkLoop calls Vertex::getId() non-dependently, so Vertex must be COMPLETE at
// its definition. Vertex.h only forward-declares Edge, so this include is
// acyclic.
#include "mesh/Vertex.h"

#include <complex>
#include <random>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>


// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
/// # Edge Disposition
///
/// There are two things that determine the disposition (spacelike, timelike, light/null-like). The first is the squared
/// edge length. If the squared length is negative in a (-, +, +, +) signature it's timelike. A negative edge length in
/// a (+, -, -, -) signature is spacelike. A 0-length in either is lightlike/null.
///
/// The second thing that determines the edge disposition is whether the vertices are at the same time (spacelike,
/// within a spatial slice) or at different times (timelike, crossing between slices). See "Quantum Gravity from Causal
/// Dynamical Triangulations: A Review" by R. Loll, 2019. Figure 1. CDT does not treat the lightlike case.
enum class EdgeDisposition : uint8_t {
  Spacelike = 0,
  Timelike = 1,
  Lightlike = 2
};


/// # Edge Class
///
/// An edge that links two points (vertices) in spacetime. When we merge two vertices in the process of connecting two
/// adjacent simplices; we cannot modify the edges in place without first removing them from their containers. Otherwise
/// avoiding the necessary re-hashing results in undefined behavior. We should keep as little state as possible on the
/// edge in favor of maintaining that state on the Vertex.
///
/// @param source_If this Edge represents a directed Edge; then this is the Vertex from which the Edge originates. For
///   undirected edges; it's just one of two Vertices that define the Edge.
/// @param target_If this Edge represents a directed Edge; then this is the Vertex at which the Edge terminates. For
///   undirected edges; it's just one of two Vertices that define the Edge.
/// @param squaredLength_ The squared length of the edge according to whatever spacetime metric is being used. We work
///   in squared lengths to allow the use of imaginary Edge lengths (they have negative values).
///
class Edge {
  public:
    /// Construct from the (possibly complex) squared length \f$l^2\f$ — the exact metric
    /// value, stored verbatim. The complex length is derived as \f$\sqrt{l^2}\f$ (real =
    /// spacelike, imaginary = timelike). A real `double` binds here as a real \f$l^2\f$
    /// (`complex(l2, 0)`) — that is the intended meaning, not a length.
    Edge(
      const VertexPtr &source,
      const VertexPtr &target,
      std::complex<double> squaredLength
    );

    Edge(
      const VertexPtr &source,
      const VertexPtr &target
    );

    /// Every edge has a beginning and an end. Many have two! And by that I mean they're undirected, so the beginning is
    /// the end and the end, the beginning. Edges are bidirectional, so it doesn't really matter if you consider them
    /// directed or undirected. If you want to use a directed edge; in your code you should just specify that you only
    /// traverse `Vertex::outEdges` and avoid `Vertex::inEdges` when you traverse around.
    [[nodiscard]] const VertexPtr &getSource() const noexcept;

    /// `getTarget` is `getSource`'s better half. All good things come to an end, with a wonderful journey left to
    /// memory. But seriously, though, `getTarget` gives the vertex on one end, and `getSource` gives the other.
    [[nodiscard]] const VertexPtr &getTarget() const noexcept;

    /// The U(1) connection phase carried on this edge's stored source->target orientation (and
    /// negated on reversal). With the signed `squaredLength` magnitude it forms the complex edge
    /// weight \f$ \text{squaredLength}\cdot e^{i\,\text{phase}} \f$ read by the Hermitian-weighted
    /// Laplacian. The default (`phase = 0`) leaves an ordinary real-weighted CDT edge unchanged.
    ///
    /// @return The U(1) connection phase, in radians.
    [[nodiscard]] double getPhase() const noexcept;

    /// The exact (possibly complex) squared length \f$l^2\f$ — stored verbatim, NOT
    /// `getLength()*getLength()`. ALL geometry/action math reads this so it never incurs
    /// a `sqrt`→`square` round-trip; that ~1-ULP round-trip detonates in the
    /// ill-conditioned Lorentzian action at near-degenerate simplices (dual-volume
    /// circumradius blows up as the Cayley–Menger determinant → 0). Real-signed for an
    /// ordinary Lorentzian edge; complex for an analytically-continued (saddle) geometry.
    [[nodiscard]] std::complex<double> getSquaredLength() const noexcept;

    /// The (possibly complex) edge length — the causal DOF, distinct from the U(1)
    /// `phase` and from \f$l^2\f$. Real for spacelike, imaginary for timelike, general
    /// complex for the Picard–Lefschetz saddle. Causal character is read from THIS
    /// (`Im(length)`) — the timelike disambiguation a real signed \f$l^2\f$ cannot give.
    [[nodiscard]] std::complex<double> getLength() const noexcept;

    /// Causal character read from the LENGTH, not the fragile `sign(l^2)`: an edge
    /// is timelike iff its length has a (non-negligible) imaginary part. A genuinely
    /// spacelike (real) length has `Im == 0`; the epsilon only guards float noise.
    /// These supersede the scattered `getSquaredLength() < 0` / `>= 0` tests.
    static constexpr double kCausalEpsilon = 1e-12;
    [[nodiscard]] bool isTimelike() const noexcept;
    [[nodiscard]] bool isSpacelike() const noexcept;
    [[nodiscard]] bool isNull() const noexcept;
    [[nodiscard]] EdgeDisposition disposition() const noexcept;

#ifdef TESSERA_VERBOSE
    [[nodiscard]] std::string toString() const noexcept;
#else
    [[nodiscard]] std::string toString() const noexcept {
      return "";
    };
#endif

    /// Replace the source vertex in-place and update the fingerprint.
    ///
    /// WARNING: The caller MUST extract this edge from EdgeList BEFORE calling,
    /// then reinsert after. Modifying the fingerprint while the edge is in a
    /// hash-keyed container causes undefined behavior (stale bucket placement).
    /// See Spacetime::swapVertexLabels for the correct extract/update/reinsert pattern.
    void replaceSourceVertex(const VertexPtr &newSource);

    /// Replace the target vertex in-place and update the fingerprint.
    ///
    /// WARNING: Same container-safety requirement as replaceSourceVertex.
    void replaceTargetVertex(const VertexPtr &newTarget);

    ///
    /// Check whether or not this Edge has a particular Vertex. The comparison is against source/target node IDs, so
    /// don't worry too much about accidentally comparing pointers. This is mostly a convenience method to make your
    /// code more clear and avoid typing.
    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool hasVertex(std::uint64_t vertexId) const;
    bool hasVertex(const VertexPtr &vertex) const;

    bool operator==(const Edge &other) const;

    [[nodiscard]] std::uint64_t toHash() const;

    Fingerprint fingerprint{};

    /// If you want to compare two edges by value; you can compare their keys. Assume two Edges with the same EdgeKey
    /// are, for all intents and purposes, equal. This will change if we begin storing state on the Edge, but at the
    /// moment let's focus on storing as much state on the Vertex as possible. Edges have potentially MUCH higher
    /// cardinality than Vertices, so as much state as we can fit on the Vertex, we should fit on the Vertex. This
    /// should be at the expense of slight inconvenience.
    ///
    /// @returns A tuple of {sourceId, targetId}.
    EdgeKey getKey() const noexcept;

    /// Set the exact (complex) squared length \f$l^2\f$; the complex length is kept in
    /// sync as \f$\sqrt{l^2}\f$. Prefer this when the geometry is specified by a squared
    /// value (CDT, Van Raamsdonk, the backreaction scan) so \f$l^2\f$ is stored exactly
    /// and the action never sees a round-trip.
    ///
    /// **Ordinary-Lorentzian convention (#580/#589):** \f$l^2\f$ is real and signed
    /// (spacelike > 0, timelike < 0, null 0); the geometry stack reads it through
    /// `getRealSquaredLength()`, which enforces the on-axis invariant loudly (#597)
    /// instead of projecting. The complexified (Picard–Lefschetz) theory is unbuilt;
    /// the dynamics keeps \f$l^2\f$ on the real axis by construction
    /// (`MultiCobordism::runStage2`), and storage round-trips a general complex
    /// value exactly (rollback records, saddle bookkeeping, historical dumps) —
    /// only geometry consumption is on-axis.
    void setSquaredLength(std::complex<double> l2) noexcept {
      squaredLength_ = l2;
      length_ = std::sqrt(l2);
    }

    /// The geometry stack's read of \f$l^2\f$ (#589/#597): the real signed value,
    /// with the ordinary-Lorentzian on-axis invariant enforced. A nonzero
    /// \f$\mathrm{Im}\,l^2\f$ reaching the Gram/Cayley–Menger/action/register
    /// pipeline is an upstream bug that a silent `.real()` projection would mask,
    /// so it throws instead of truncating. Storage
    /// (`getSquaredLength`/`setSquaredLength`) stays general-complex — use it, not
    /// this, wherever a complex value is legitimate (Wick \f$|l^2|\f$ reads,
    /// snapshots, rollback records, dump rehydration).
    [[nodiscard]] double getRealSquaredLength() const {
      if (squaredLength_.imag() != 0.0)
        throw std::runtime_error(
            "Edge(" + std::to_string(source->getId()) + "," +
            std::to_string(target->getId()) + "): nonzero Im l^2 = " +
            std::to_string(squaredLength_.imag()) +
            " reached the geometry stack — the ordinary-Lorentzian convention "
            "(#589) keeps l^2 real and signed, so this is an upstream bug, not "
            "a value to project away");
      return squaredLength_.real();
    }

    /// Set the (complex) edge length; the squared length is kept in sync as `l*l`. Use
    /// when the geometry is specified by a length directly. Real for spacelike,
    /// imaginary for timelike — the two cases of the ordinary-Lorentzian convention
    /// (see `setSquaredLength`); the off-axis (Picard–Lefschetz) saddle is unbuilt.
    void setLength(std::complex<double> l) noexcept {
      length_ = l;
      squaredLength_ = l * l;
    }

    /// Set the U(1) connection phase (radians).  Used by the Hermitian-weighted
    /// Laplacian and its gauge transform to rephase the edge without rebuilding the mesh.
    void setPhase(double p) noexcept { phase = p; }

    /// Walk a closed loop of ordered directed steps (each Edge's
    /// getSource()->getTarget() is one traversal step). Invokes f(sourceId,
    /// targetId, sign) per step; sign = +1 if sourceId < targetId (canonical
    /// orientation) else -1.
    template <typename F>
    static void walkLoop(const std::vector<Edge> &loop, F &&f) {
      for (const Edge &step : loop) {
        const std::uint64_t u = step.getSource()->getId();
        const std::uint64_t v = step.getTarget()->getId();
        f(u, v, (u < v) ? 1.0 : -1.0);
      }
    }

    /// The Van Raamsdonk metric law: the spacelike signed squared length for a
    /// given mutual information ``I`` — the value to store as ``squaredLength``
    /// on a same-time-slice edge. Returns (−log(I/iMax))², with the length
    /// floored to −log(epsilon) (so the squared length stays finite) when
    /// I < epsilon·iMax (and when iMax ≤ 0 or I ≤ 0). Always ≥ 0 (spacelike).
    [[nodiscard]] static double
    vanRaamsdonkSquaredLength(double I, double iMax,
                              double epsilon = 1e-10) noexcept;

    /// Time-aware Van Raamsdonk signed squared length for THIS edge, given the
    /// mutual information ``I`` between its endpoints (the one-forward-step
    /// convention): a worldline edge whose endpoints lie on different time
    /// slices (``Vertex::getTime``) is null and returns 0; a same-slice edge is
    /// spacelike and returns ``vanRaamsdonkSquaredLength(I, iMax, epsilon)``.
    [[nodiscard]] double
    vanRaamsdonkSquaredLengthFor(double I, double iMax,
                                 double epsilon = 1e-10) const;

    /// Index into EdgeList::liveVec_ (maintained by EdgeList).
    std::uint32_t liveIdx_{UINT32_MAX};

    /// Simplices currently containing this edge (the edge's "cofaces" in
    /// the codim-1 sense). Mirror of ``Vertex::simplices`` but at edge
    /// granularity, used by ``Vertex::removeOutEdge`` /
    /// ``Vertex::removeInEdge`` to drop the edge from just the simplices
    /// that actually contain it — instead of iterating every simplex
    /// touching the endpoint and filtering by ``hasVertex``. Surfaced by
    /// the v0.2 finite-size profile: hasVertex was ≈22% of `thermalize`
    /// wall time even after the per-call cache.
    ///
    /// Maintained in lockstep with ``Simplex::edges``:
    ///   * Spacetime::registerSimplex registers the simplex on each of
    ///     its edges
    ///   * Spacetime::unregisterSimplex removes it
    ///   * Simplex::addEdge / Simplex::removeEdge mirror the same
    ///     callbacks at runtime
    ///
    /// Callers that intend to mutate the index from inside an iteration
    /// loop (e.g. ``simplex->removeEdge(this)`` invalidates ``simplices_``)
    /// MUST use ``simplicesCopy()`` to snapshot first.
    void registerSimplex(SimplexPtr s);
    void unregisterSimplex(SimplexPtr s) noexcept;
    [[nodiscard]] Simplices const& simplices() const noexcept { return simplices_; }
    [[nodiscard]] Simplices simplicesCopy() const { return simplices_; }

  private:
    VertexPtr source = nullptr;
    VertexPtr target = nullptr;

    /// The exact (possibly complex) squared length \f$l^2\f$ — the metric value read by
    /// the action and all geometry math. `length_` is its principal `sqrt`; the two are
    /// kept in sync by `setSquaredLength`/`setLength`.
    std::complex<double> squaredLength_{};
    /// The complex length (causal DOF, distinct from the U(1) `phase`). Causal character
    /// is `Im(length_)`; carries the sqrt-branch a real \f$l^2\f$ cannot express.
    std::complex<double> length_{};
    double phase = 0.0;

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
