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
/// The microscopic edge fields are stored without choosing a branch:
///
/// * ``squaredLength_`` is the complex squared length \f$z_e\in\mathbb C\f$;
/// * ``canonicalLink_`` is the non-zero multiplicative link
///   \f$U_{xy}\in\mathbb C^*\f$ on the canonical ``min(id)->max(id)``
///   orientation.  The reverse link is its inverse, never its conjugate.
///
/// A length or additive phase is a legacy/presentation view.  Neither is a
/// stored bulk field.
///
class Edge {
  public:
    /// Construct from the complex SQUARED length \f$z_e\f$.  The value is
    /// stored verbatim; no square-root sheet is selected.
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

    /// LEGACY VIEW: a principal logarithm of the source->target link, defined
    /// by ``U=exp(i*phase)``.  This chooses a logarithm branch and must not be
    /// used by complex-first bulk code, cache identities, or replay.
    [[nodiscard]] std::complex<double> getPhase() const noexcept;

    /// LEGACY VIEW: the principal square root of ``squaredLength()``.  This is
    /// presentation-only and is never a replay identity.
    [[nodiscard]] std::complex<double> getLength() const noexcept;

    /// The exact branch-free geometric datum \f$z_e\f$.
    [[nodiscard]] std::complex<double> squaredLength() const noexcept {
      return squaredLength_;
    }

    /// Replace \f$z_e\f$ verbatim; no root, sign, or real-section choice.
    void setSquaredLength(std::complex<double> z) noexcept {
      squaredLength_ = z;
      ++geometryRevision_;
    }

    /// Link on the canonical endpoint orientation ``min(id)->max(id)``.
    [[nodiscard]] std::complex<double> canonicalLink() const noexcept {
      return canonicalLink_;
    }

    /// Link on an explicitly requested endpoint orientation.  ``from`` and
    /// ``to`` must be this edge's distinct endpoint ids.
    [[nodiscard]] std::complex<double> link(std::uint64_t from,
                                             std::uint64_t to) const;

    /// Set the canonical link.  Zero is rejected because the link lives in
    /// \f$\mathbb C^*\f$.
    void setCanonicalLink(std::complex<double> nonzeroU);

    /// Set the link on an explicitly requested orientation; storage is
    /// canonicalized without a logarithm or argument.
    void setLink(std::uint64_t from, std::uint64_t to,
                 std::complex<double> nonzeroU);

    /// Left-trivialized tangent coordinate \f$U^{-1}\,\delta U\f$ on an
    /// explicit orientation. This is a differential coordinate, not a
    /// logarithm of the resident link, and therefore has no branch.
    [[nodiscard]] std::complex<double> linkLogTangent(
        std::uint64_t from, std::uint64_t to,
        std::complex<double> deltaU) const {
      return deltaU / link(from, to);
    }

    /// Re-express the stored canonical link after endpoint identifiers have
    /// changed.  The oriented link between the endpoint objects is preserved.
    void recanonicalizeLink(std::uint64_t oldSourceId,
                            std::uint64_t oldTargetId) noexcept;

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

    /// LEGACY MUTATOR: accept a length and store its square.  New code uses
    /// ``setSquaredLength``.  This direction is branch-free, but it preserves
    /// the old call shape only while callers are migrated.
    void setLength(std::complex<double> l) noexcept {
      setSquaredLength(l * l);
    }

    /// Legacy spelling of ``geometryRevision``.
    [[nodiscard]] std::uint64_t lengthRevision() const noexcept {
      return geometryRevision_;
    }

    [[nodiscard]] std::uint64_t geometryRevision() const noexcept {
      return geometryRevision_;
    }

    /// LEGACY MUTATOR: convert an additive coordinate with
    /// ``U=exp(i*phase)``. This conversion is intentionally isolated here;
    /// direct complex-first code calls ``setLink``/``setCanonicalLink``.
    void setPhase(std::complex<double> p);

    /// Legacy spelling of ``linkRevision``.
    [[nodiscard]] std::uint64_t phaseRevision() const noexcept {
      return linkRevision_;
    }

    [[nodiscard]] std::uint64_t linkRevision() const noexcept {
      return linkRevision_;
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

    /// The Van Raamsdonk length view for a given mutual information ``I``.
    /// A direct-z caller stores its square with ``setSquaredLength``. Returns
    /// -log(I/iMax), floored at -log(epsilon) when needed.
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

    /// Exact complex squared length.  This is the geometry, not a selected
    /// square-root sheet.
    std::complex<double> squaredLength_{};
    /// Link on min(endpoint id)->max(endpoint id).  Always non-zero.
    std::complex<double> canonicalLink_{1.0, 0.0};
    std::uint64_t geometryRevision_{0};
    std::uint64_t linkRevision_{0};

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
