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

    /// # Disjoint union (the disconnected cobordism)
    ///
    /// The disjoint union \f$ W_1 \sqcup W_2 \f$ of two triangulations:
    /// \f$ W_2 \f$'s vertices are shifted into a fresh id range above
    /// \f$ W_1 \f$'s so the two complexes share no vertex, and their top
    /// simplices are concatenated into a single `Spacetime`. Nothing is
    /// identified, so \f$ \partial(W_1\sqcup W_2) = \partial W_1 \sqcup
    /// \partial W_2 \f$ and the bulk is disconnected. Two solid tori
    /// \f$ S^1\times D^2 \f$ (each \f$ \partial = T^2 \f$) thus give a bulk with
    /// \f$ \partial W = T^2 \sqcup T^2 \f$: the **cap-and-create** cobordism
    /// \f$ T^2 \to T^2 \f$ that is *not* a mapping cylinder. Its DW map
    /// factorizes over the components to the rank-1 outer product
    /// \f$ |st\rangle\langle st| \f$ (\f$ |st\rangle \f$ the solid-torus boundary
    /// state) — a **non-invertible** boundary map, unreachable by any cylinder
    /// (whose map is always an invertible permutation). The result is a fresh
    /// pre-geometric `Spacetime` ready for `DijkgraafWitten::map()`, whose
    /// two-component requirement it satisfies while the bulk need not be
    /// connected.
    /// @throws std::invalid_argument if either input is empty or the two top
    ///   dimensions differ.
    [[nodiscard]] static std::shared_ptr<Spacetime> disjointUnion(
        const Spacetime &W1, const Spacetime &W2);

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

    /// # Twisted cylinder (a non-identity DW boundary map)
    ///
    /// The \f$ \varphi \f$-twisted product cobordism \f$ \Sigma\times[0,T] :
    /// \Sigma \to \Sigma \f$, whose two boundary copies of the surface
    /// \f$ \Sigma \f$ are threaded through a finite-order *simplicial
    /// automorphism* \f$ \varphi \f$ of the triangulated \f$ \Sigma \f$. The
    /// ordinary product cylinder — the one `glue`/`selfGlue` produce, identified
    /// by the order-preserving (identity) isomorphism — has DW map the identity
    /// on \f$ Z(\Sigma) \f$; that identity identification is the single home of
    /// the "\f$ \to I \f$" collapse. Putting \f$ \varphi \f$ there instead makes
    /// the interior monodromy from the bottom boundary to the top boundary equal
    /// \f$ \varphi \f$, so `DijkgraafWitten::map()` returns the permutation
    /// \f$ \varphi \f$ induces on the holonomy classes
    /// \f$ Z(\Sigma)=\mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$. For the
    /// coordinate-swap \f$ \varphi:(x,y)\mapsto(y,x) \f$ of a square-product
    /// torus \f$ T^2=S^1\times S^1 \f$ this transposes the holonomy classes
    /// \f$ [a]\leftrightarrow[b] \f$ while fixing \f$ [0] \f$ and \f$ [a{+}b] \f$
    /// — a non-identity \f$ 4\times4 \f$ permutation (\f$ \varphi\bmod 2 \f$ is
    /// the modular \f$ S=\bigl(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr)
    /// \f$).
    ///
    /// Construction: three stacked copies of \f$ \Sigma \f$ (levels 0,1,2 at
    /// vertex ids \f$ x,\;|V|{+}x,\;2|V|{+}x \f$). The level-0\f$ \to \f$1 prism
    /// is the ordinary Eilenberg–Zilber product \f$ \Sigma\times[0,1] \f$ (the
    /// same staircase `SimplicialProduct` uses); the level-1\f$ \to \f$2 prism is
    /// the same product but its lower face is glued to the shared seam through
    /// \f$ \varphi \f$ (the level-1 vertex of \f$ \varphi(x) \f$ carries the seam
    /// id of \f$ x \f$). Because \f$ \varphi \f$ is a simplicial automorphism the
    /// two prisms induce the *same* triangulation on the seam, so it closes up;
    /// the seam (level 1) is interior, and \f$ \partial W \f$ is the two
    /// \f$ \Sigma \f$ copies at levels 0 and 2 (\f$ \geq 3 \f$ layers, so no top
    /// simplex touches both boundaries). The result is a fresh pre-geometric
    /// `Spacetime` ready for `DijkgraafWitten::map()`.
    ///
    /// \f$ \varphi \f$ of order \f$ n \f$ gives a DW map of order \f$ n \f$
    /// (`map`\f$ {}^n=I \f$), and composing twists composes the permutations —
    /// the functoriality of the DW functor. Only the finite-order modular
    /// elements (orders 2,3,4,6) are realizable this way: the Dehn twist
    /// \f$ T=\bigl(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\bigr) \f$ is
    /// infinite-order, hence *not* a finite-order simplicial automorphism of any
    /// fixed triangulation — which is exactly why a fixed-triangulation cobordism
    /// could only ever realize the identity until now.
    ///
    /// @param sigma a closed triangulated surface \f$ \Sigma \f$ whose vertices
    ///   are exactly \f$ 0..|V|-1 \f$ (as the product-torus fixtures produce).
    /// @param phi a vertex permutation (`phi[x]` is the image of vertex
    ///   \f$ x \f$) that is a simplicial automorphism of \f$ \Sigma \f$ — both
    ///   validated.
    /// @throws std::invalid_argument if \f$ \Sigma \f$ is not a 2-dimensional
    ///   triangulation with vertices \f$ 0..|V|-1 \f$, or `phi` is not a length-
    ///   \f$ |V| \f$ permutation, or `phi` is not a simplicial automorphism of
    ///   \f$ \Sigma \f$ (some top triangle's image is not a triangle).
    [[nodiscard]] static std::shared_ptr<Spacetime> twistedCylinder(
        const Spacetime &sigma, const std::vector<std::uint64_t> &phi);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISM_H
