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
  Lightlike = 2,
  /// A genuinely complex \f$ l^2 \f$ — an argument at none of the three definite
  /// values. The honest reading of an edge that has not acquired a causal
  /// character, and the common case for a randomly seeded complex.
  Mixed = 3,
  /// An absent edge, \f$ l = 0 \f$. Not a causal type.
  Degenerate = 4
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

    /// The \f$\mathbb{C}^{*}\f$ connection phase \f$\varphi\f$ carried on this edge's stored
    /// source->target orientation. It is the SECOND edge field, independent of the geometry:
    /// the link variable is \f$ U_{xy} = e^{i\varphi} \in \mathbb{C}^{*} \f$ with
    /// \f$ U_{yx} = U_{xy}^{-1} \f$ (the INVERSE, not the conjugate — the two coincide only
    /// for real \f$\varphi\f$), and a gauge transformation \f$ g \f$ acts by
    /// \f$ U_{xy} \mapsto g_x^{-1} U_{xy} g_y \f$, leaving `length_` and every metric weight
    /// built from it untouched.
    ///
    /// \f$\varphi\f$ is COMPLEX because the structure group is
    /// \f$ \mathbb{C}^{*} = U(1)\times\mathbb{R}^{+} \f$:
    /// \f$ e^{i\varphi} = e^{i\operatorname{Re}\varphi}\,e^{-\operatorname{Im}\varphi} \f$.
    /// `Re` is the compact U(1) angle in radians — the only part with winding, hence the only
    /// part that quantizes and the only part a Wilson loop reads. `Im` is the non-compact
    /// \f$\mathbb{R}^{+}\f$ local scale and carries no quantum number.
    ///
    /// It twists the HOPPING of the Aharonov-Bohm operator (`HodgeLaplacian::connectionLaplacian`)
    /// and never rescales a metric weight: the geometric Hodge operator `laplacian(k)` is built
    /// from `length_` alone and is blind to \f$\varphi\f$ at every degree. Writing \f$\varphi\f$
    /// into the weight would make the metric gauge-variant and destroy the derived form of
    /// \f$ L_k \f$. The default (`phase = 0`) leaves an ordinary untwisted CDT edge unchanged.
    ///
    /// @return The \f$\mathbb{C}^{*}\f$ connection phase; `Re` in radians, `Im` the log-scale.
    [[nodiscard]] std::complex<double> getPhase() const noexcept;

    /// The (possibly complex) edge length — the causal DOF, distinct from the
    /// connection `phase` and from \f$l^2\f$. Real for spacelike, imaginary for timelike, general
    /// complex for the Picard–Lefschetz saddle. Causal character is the LORENTZIAN
    /// magnitude of this, \f$\mathrm{Re}(l^2)\f$ — see the predicates below.
    [[nodiscard]] std::complex<double> getLength() const noexcept;

    /// # Causal character, from the ARGUMENT of \f$ l^2 \f$
    ///
    /// Classifying on \f$ \mathrm{Re}(l^2) \f$ alone would discard
    /// \f$ \mathrm{Im}(l^2) \f$, which is real physics here rather than a residue —
    /// the same error as building a diagnostic on \f$ \mathrm{Re}\,S \f$ alone. It
    /// bites exactly at the case of interest: at the lightlike point
    /// \f$ \mathrm{Im}(l^2) = 2x^2 \neq 0 \f$, so \f$ l^2 \f$ is purely imaginary and
    /// NONZERO. A fully null \f$ l^2 = 0 \f$ does not exist non-trivially.
    ///
    /// Both components are accounted for by classifying on the argument. Writing
    /// \f$ l = |l| e^{i a} \f$,
    /// \f[ l^2 = |l|^2 e^{2ia}, \quad
    ///     \mathrm{Re}(l^2) = |l|^2 \cos 2a, \quad
    ///     \mathrm{Im}(l^2) = |l|^2 \sin 2a. \f]
    ///
    /// | \f$ a = \arg l \f$ | \f$ l^2 \f$ | disposition |
    /// |---|---|---|
    /// | \f$ 0 \f$ (mod \f$\pi\f$) | real positive | spacelike |
    /// | \f$ \pi/2 \f$ (mod \f$\pi\f$) | real negative | timelike |
    /// | \f$ \pi/4 \f$ (mod \f$\pi/2\f$) | purely imaginary | lightlike |
    /// | anything else | genuinely complex | **mixed** |
    /// | \f$ l = 0 \f$ | — | degenerate: an absent edge, not a causal type |
    ///
    /// A generic argument is reported as MIXED, never assigned to the nearest of the
    /// three — that would invent a definiteness the geometry does not have. Under a
    /// uniformly drawn argument almost every edge is mixed, and the mixed FRACTION is
    /// a diagnostic: if relaxation imposes causal character, it should fall.
    ///
    /// `squaredArgument()` carries \f$ \arg(l^2) \f$ as the measured quantity, so a
    /// consumer can see where an edge actually sits rather than only which bucket it
    /// fell in.
    ///
    /// The canonical cases are unchanged: a real length is spacelike and an imaginary
    /// one timelike. What changes is that a genuinely complex length is no longer read
    /// as timelike merely for having an imaginary part.
    ///
    /// `isDegenerate` is separate from `isNull` on purpose: a null edge is a physical
    /// lightlike ray, a degenerate one is absent, and conflating them was the defect
    /// this replaces. Exactly one of the five predicates holds for any edge.

    /// Absolute floor on the EUCLIDEAN modulus below which an edge is degenerate
    /// rather than any causal type. Dimensions of LENGTH. This is the one place the
    /// Euclidean modulus is the right norm — an edge with no extent is absent.
    static constexpr double kDegenerateEpsilon = 1e-12;

    /// Angular half-width, in radians, within which an argument counts as definite.
    ///
    /// ANGULAR because it is dimensionless and therefore scale-free. An absolute
    /// floor on \f$ \mathrm{Re}(l^2) \f$ could not work: that quantity has dimensions
    /// of length SQUARED while the superseded `kCausalEpsilon` compared
    /// \f$ |\mathrm{Im}(l)| \f$, a LENGTH, so one constant cannot serve both and
    /// either choice silently reclassifies edges near the cone as a complex refines.
    ///
    /// In \f$ \arg(l^2) \f$ the three definite dispositions sit at \f$ 0 \f$,
    /// \f$ \pm\pi/2 \f$ and \f$ \pi \f$ — equivalently \f$ a = 0, \pi/4, \pi/2 \f$ —
    /// so ONE half-width applied at each treats them symmetrically, which is the
    /// property to want. The buckets stay disjoint for any width below
    /// \f$ \pi/4 \f$.
    ///
    /// The value is a numerical-noise guard, not a bucket width: a deliberately
    /// constructed disposition lands on its argument to within a few ulp
    /// (\f$ \sim 10^{-16} \f$ rad), so \f$ 10^{-9} \f$ sits some seven orders above
    /// float noise and eight below the \f$ \pi/4 \f$ collision bound. Widening it
    /// would start absorbing genuinely mixed edges into definite buckets, which is
    /// exactly the invented definiteness this classification refuses.
    static constexpr double kCausalAngularEpsilon = 1e-9;

    /// \f$ \arg(l^2) \in (-\pi, \pi] \f$ — the measured quantity every predicate
    /// below classifies. Zero is spacelike, \f$ \pm\pi/2 \f$ lightlike, \f$ \pi \f$
    /// timelike, anything else mixed.
    [[nodiscard]] double squaredArgument() const noexcept;
    /// \f$ \mathrm{Re}(l^2) = x^2 - t^2 \f$, carried for consumers that want the
    /// interval itself. It does NOT decide the disposition on its own.
    [[nodiscard]] double lorentzianMagnitude() const noexcept;
    [[nodiscard]] bool isTimelike() const noexcept;
    [[nodiscard]] bool isSpacelike() const noexcept;
    [[nodiscard]] bool isNull() const noexcept;
    /// A genuinely complex \f$ l^2 \f$: no definite causal character.
    [[nodiscard]] bool isMixed() const noexcept;
    /// An absent edge (\f$ |l|_E \approx 0 \f$), which is not a causal type.
    [[nodiscard]] bool isDegenerate() const noexcept;
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
    /// the connection phase never enters those.
    [[nodiscard]] std::uint64_t lengthRevision() const noexcept {
      return lengthRevision_;
    }

    /// Set the \f$\mathbb{C}^{*}\f$ connection phase: `Re` the compact U(1) angle in radians,
    /// `Im` the non-compact log-scale. Used by the Aharonov-Bohm operator and its gauge
    /// transform to re-twist the edge without rebuilding the mesh. A real argument converts
    /// implicitly and reproduces the untwisted-geometry, real-angle case exactly.
    void setPhase(std::complex<double> p) noexcept {
      phase = p;
      ++phaseRevision_;
    }

    /// Monotone ``setPhase`` counter, the phase analogue of ``lengthRevision``.
    /// The Aharonov-Bohm operator reads phases, so the shared spectrum cache
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
    std::complex<double> phase{0.0, 0.0};

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
