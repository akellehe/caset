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
/// @param length_ The complex length of the edge according to whatever spacetime metric is
///   being used. Real for spacelike, imaginary for timelike; the squared length is derived
///   by squaring it and is never stored (#639).
///
class Edge {
  public:
    /// Construct from the (possibly complex) LENGTH \f$l\f$ — real for spacelike,
    /// imaginary for timelike, general complex off the real-Lorentzian locus. This is
    /// the edge's one degree of freedom; \f$l^2\f$ is derived by squaring, never stored
    /// (#639). Callers holding an \f$l^2\f$ pass ``std::sqrt(l2)`` and so choose the
    /// branch explicitly rather than having one chosen for them.
    Edge(
      const VertexPtr &source,
      const VertexPtr &target,
      std::complex<double> length
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

    /// The (possibly complex) edge length — the causal DOF, distinct from the U(1)
    /// `phase` and from \f$l^2\f$. Real for spacelike, imaginary for timelike, general
    /// complex for the Picard–Lefschetz saddle. Causal character is read from THIS
    /// (`Im(length)`) — the timelike disambiguation a real signed \f$l^2\f$ cannot give.
    [[nodiscard]] std::complex<double> getLength() const noexcept;

    /// Causal character read from the LENGTH, not the fragile `sign(l^2)`: an edge
    /// is timelike iff its length has a (non-negligible) imaginary part. A genuinely
    /// spacelike (real) length has `Im == 0`; the epsilon only guards float noise.
    /// These supersede the scattered `sign(l^2)` tests.
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

    /// Set the (complex) edge LENGTH \f$l\f$ — the edge's one degree of freedom.
    /// Real for spacelike, imaginary for timelike, general complex off the
    /// real-Lorentzian locus.
    ///
    /// There is no squared-length setter (#639). \f$l^2\f$ is not stored, so it cannot
    /// drift out of sync with \f$l\f$, and a caller holding an \f$l^2\f$ writes
    /// ``setLength(std::sqrt(l2))`` — picking the branch explicitly instead of having
    /// one picked silently. \f$l\f$ is the right primitive: \f$l \mapsto l^2\f$ is
    /// two-to-one, so \f$l^2\f$ cannot express which of \f$\pm l\f$ this edge is.
    ///
    /// **Cost, accepted:** a geometry SPECIFIED by a squared value (CDT, Van
    /// Raamsdonk, the backreaction scan) now round-trips through
    /// \f$\sqrt{\cdot}\f$ and back, so consumers see \f$l^2 \pm 1\f$ ULP rather
    /// than the exact value the old verbatim store gave them. That matters most in the
    /// ill-conditioned regime where the Cayley-Menger determinant approaches zero.
    void setLength(std::complex<double> l) noexcept {
      length_ = l;
      ++lengthRevision_;
    }

    /// Monotone per-edge write counter, bumped by every ``setLength``.
    /// ``Simplex``'s length-derived geometry cache keys on the sum of its
    /// edges' revisions, so an unchanged key proves no incident length changed
    /// since the cache was filled. ``setPhase`` deliberately does NOT bump it:
    /// the cache holds only length-derived data (Gram / Cayley-Menger), and
    /// the U(1) phase never enters those.
    [[nodiscard]] std::uint64_t lengthRevision() const noexcept {
      return lengthRevision_;
    }

    /// Set the U(1) connection phase (radians).  Used by the Hermitian-weighted
    /// Laplacian and its gauge transform to rephase the edge without rebuilding the mesh.
    void setPhase(double p) noexcept {
      phase = p;
      ++phaseRevision_;
    }

    /// Monotone ``setPhase`` counter, the phase analogue of ``lengthRevision``.
    /// The k=0 Hermitian Laplacian reads phases, so the shared spectrum cache
    /// keys on BOTH counters; the Simplex geometry cache (Gram/Cayley-Menger)
    /// keys on lengths alone and deliberately ignores this one.
    [[nodiscard]] std::uint64_t phaseRevision() const noexcept {
      return phaseRevision_;
    }

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
    /// given mutual information ``I`` — the value to store via ``setLength`` on a
    /// same-time-slice edge. Returns −log(I/iMax), floored at −log(epsilon) (so the
    /// length stays finite) when I < epsilon·iMax (and when iMax ≤ 0 or I ≤ 0).
    /// Always real and ≥ 0, i.e. spacelike.
    [[nodiscard]] static double
    vanRaamsdonkLength(double I, double iMax,
                       double epsilon = 1e-10) noexcept;

    /// Time-aware Van Raamsdonk length for THIS edge, given the mutual information
    /// ``I`` between its endpoints (the one-forward-step convention): a worldline edge
    /// whose endpoints lie on different time slices (``Vertex::getTime``) is null and
    /// returns 0; a same-slice edge is spacelike and returns
    /// ``vanRaamsdonkLength(I, iMax, epsilon)``.
    [[nodiscard]] double
    vanRaamsdonkLengthFor(double I, double iMax,
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

    /// The complex edge length \f$l\f$ — the edge's ONE stored degree of freedom
    /// (distinct from the U(1) `phase`). Causal character is `Im(length_)`.
    /// \f$l^2\f$ is derived by squaring at the point of use, never stored (#639).
    std::complex<double> length_{};
    /// Monotone ``setLength`` counter read by ``lengthRevision()``; see there.
    std::uint64_t lengthRevision_{0};
    /// Monotone ``setPhase`` counter read by ``phaseRevision()``; see there.
    std::uint64_t phaseRevision_{0};
    double phase = 0.0;

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
