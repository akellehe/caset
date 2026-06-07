// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TESSERA_COBORDISM_COBORDISM_H
#define TESSERA_COBORDISM_COBORDISM_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// A list of simplices given by their sorted vertex-id tuples.
using SimplexList = std::vector<std::vector<std::uint64_t>>;

/// Outcome code for a cobordism-verification check.
enum class CobordismCheck {
  Ok,                              ///< W is a valid cobordism from M1 to M2.
  BoundaryChainNotClosed,          ///< the boundary-of-a-boundary is not zero (a malformed complex)
  WrongNumberOfBoundaryComponents, ///< the boundary doesn't split into the right number of pieces
  BoundaryNotIsomorphic,           ///< a boundary piece does not match M1 or M2 as a triangulation
};

/// Result of verifying a cobordism: whether it passed, a machine-readable code,
/// and a human-readable explanation when it failed.
struct CobordismResult {
  bool ok{false};
  CobordismCheck code{CobordismCheck::Ok};
  std::string detail{};
};

/// # Cobordism verification
///
/// A *cobordism* from one manifold \f$ M_1 \f$ to another \f$ M_2 \f$ is a
/// higher-dimensional manifold \f$ W \f$ whose boundary is exactly the two of
/// them placed side by side (\f$ \partial W = M_1 \sqcup M_2 \f$). For example
/// a solid cylinder is a cobordism from a circle to a circle, and a solid disk
/// is a cobordism from a circle to "nothing" (the empty manifold).
///
/// This class checks whether a given \f$ W \f$ really is such a cobordism. It
/// is a static-only utility (no state) operating on triangulations
/// (`Spacetime`s). This first version checks the **boundary structure**: that
/// the boundary of \f$ W \f$ breaks into connected pieces that match
/// \f$ M_1 \f$ and \f$ M_2 \f$ as triangulations, plus the basic
/// chain-complex sanity check that the boundary of the boundary is empty.
/// The two further checks the full specification calls for — that \f$ W \f$ is
/// a genuine manifold (every vertex's neighborhood looks like a ball or sphere)
/// and that the boundary's orientation is consistent — are separate follow-ups.
class Cobordism {
  public:
    Cobordism() = delete;

    /// The boundary of \f$ W \f$: the codimension-one faces (one dimension below
    /// the top simplices) that belong to exactly one top simplex. Returned as
    /// sorted vertex-id tuples. An interior face belongs to two top simplices
    /// and is excluded; a boundary face belongs to just one.
    [[nodiscard]] static SimplexList boundaryFaces(const Spacetime &W);

    /// Split a list of same-dimensional simplices into connected pieces. Two
    /// simplices are in the same piece when they share a codimension-one face
    /// (i.e. they are glued along a common facet), extended transitively.
    [[nodiscard]] static std::vector<SimplexList> connectedComponents(
        const SimplexList &simplices);

    /// True when two triangulations, each given as a list of top-simplex vertex
    /// tuples, are isomorphic — i.e. some relabeling of the vertices of one
    /// turns its simplex set into the other's.
    [[nodiscard]] static bool areIsomorphic(const SimplexList &a, const SimplexList &b);

    /// Verify that \f$ W \f$ is a cobordism from \f$ M_1 \f$ to \f$ M_2 \f$:
    /// its boundary splits into connected pieces that are, up to relabeling, the
    /// triangulations \f$ M_1 \f$ and \f$ M_2 \f$ (either of which may be the
    /// empty manifold). Also checks the boundary-of-a-boundary is empty.
    [[nodiscard]] static CobordismResult verify(const Spacetime &W,
                                                const Spacetime &M1,
                                                const Spacetime &M2);

    /// # Gluing (composition of cobordisms)
    ///
    /// Form the composite \f$ W_2 \cup_{\Sigma_C} W_1 \f$ of two cobordisms that
    /// share a boundary surface \f$ \Sigma_C \f$: a boundary component of
    /// \f$ W_1 \f$ that is isomorphic (as a triangulation) to one of \f$ W_2 \f$.
    /// The two copies of \f$ \Sigma_C \f$ are identified vertex-for-vertex by the
    /// order-preserving simplicial isomorphism between them (the same
    /// correspondence `areIsomorphic` certifies), the two complexes are reindexed
    /// into one dense vertex range, and their top simplices are merged. The
    /// glued \f$ \Sigma_C \f$ faces, now shared by one top simplex from each
    /// side, become interior; the result's boundary is the remaining components
    /// \f$ \partial W_1 \sqcup \partial W_2 \f$ with the two \f$ \Sigma_C \f$
    /// copies removed. Returned as a fresh pre-geometric `Spacetime` ready for
    /// `DijkgraafWitten`. The first isomorphic pair (in the deterministic
    /// component order) is used as \f$ \Sigma_C \f$.
    /// @throws std::invalid_argument if either input is empty, the top
    ///   dimensions differ, or no shared boundary surface exists.
    [[nodiscard]] static std::shared_ptr<Spacetime> glue(const Spacetime &W1,
                                                         const Spacetime &W2);

    /// # Twisted gluing along a *specified* boundary identification
    ///
    /// Like `glue(W1, W2)`, but the shared surface \f$ \Sigma_C \f$ is identified
    /// by the **caller-supplied** vertex bijection rather than the canonical
    /// order-preserving isomorphism. `boundaryBijection` must map the vertex ids
    /// of one whole boundary component of \f$ W_2 \f$ onto the vertex ids of one
    /// whole boundary component of \f$ W_1 \f$, and do so as a *simplicial
    /// isomorphism* (it carries the \f$ W_2 \f$ component's face set exactly onto
    /// the \f$ W_1 \f$ component's). This is the building block for gluing through
    /// a non-trivial mapping-class element (a Dehn twist of the boundary torus):
    /// pass a twist composed with the canonical correspondence. With the identity
    /// (order-preserving) correspondence it reproduces `glue(W1, W2)` exactly.
    /// @throws std::invalid_argument if the inputs are empty / differ in top
    ///   dimension, or the bijection is not such a simplicial isomorphism.
    [[nodiscard]] static std::shared_ptr<Spacetime> glue(
        const Spacetime &W1, const Spacetime &W2,
        const std::map<std::uint64_t, std::uint64_t> &boundaryBijection);

    /// Close a cobordism by gluing its two boundary components to each other
    /// (the categorical trace / mapping torus): \f$ \partial W \f$ must have
    /// exactly two isomorphic components \f$ \Sigma_C^{(0)}, \Sigma_C^{(1)} \f$,
    /// which are identified by the order-preserving isomorphism, yielding a
    /// closed manifold. The collar must be thick enough that no single top
    /// simplex touches both components (a top simplex spanning both would gain a
    /// repeated vertex on identification); e.g. \f$ T^2\times[0,T] \f$ needs at
    /// least three layers in the \f$ [0,T] \f$ direction so the glued circle is a
    /// non-degenerate triangulation.
    /// @throws std::invalid_argument if \f$ \partial W \f$ does not have exactly
    ///   two isomorphic components; `std::runtime_error` if the identification
    ///   collapses a top simplex (collar too thin).
    [[nodiscard]] static std::shared_ptr<Spacetime> selfGlue(const Spacetime &W);

    /// # Twisted self-gluing (the mapping torus of a boundary self-map)
    ///
    /// Like `selfGlue(W)`, but the two boundary components are identified by the
    /// caller-supplied `boundaryBijection` instead of the canonical
    /// order-preserving isomorphism. Its key set must be exactly one boundary
    /// component's vertex ids and its value set exactly the other's, and it must
    /// be a simplicial isomorphism between them; the keyed component is folded
    /// onto the valued one. Closing \f$ \Sigma\times[0,T] \f$ this way realizes
    /// the **mapping torus** \f$ \Sigma\times_\varphi S^1 \f$ of the boundary
    /// self-map \f$ \varphi \f$ (the bijection): with the identity it is the
    /// trivial bundle \f$ \Sigma\times S^1 \f$, and with a non-trivial torus
    /// automorphism it is a non-trivial torus bundle. The collar must be thick
    /// enough (no top simplex touching both components), exactly as for
    /// `selfGlue(W)`.
    /// @throws std::invalid_argument if \f$ \partial W \f$ does not have exactly
    ///   two components or the bijection is not a simplicial isomorphism between
    ///   them; `std::runtime_error` if the identification collapses a top simplex.
    [[nodiscard]] static std::shared_ptr<Spacetime> selfGlue(
        const Spacetime &W,
        const std::map<std::uint64_t, std::uint64_t> &boundaryBijection);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISM_H
