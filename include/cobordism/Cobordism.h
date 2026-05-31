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
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISM_H
