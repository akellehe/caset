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

#ifndef TESSERA_COBORDISM_TORUSTWIST_H
#define TESSERA_COBORDISM_TORUSTWIST_H

#include <array>
#include <cstdint>
#include <map>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # Mapping-class elements of the torus \f$ T^2 \f$ (the modular group)
///
/// The mapping class group of the torus is \f$ \mathrm{GL}(2,\mathbb{Z}) \f$
/// (orientation-preserving part \f$ \mathrm{SL}(2,\mathbb{Z}) \f$), acting on
/// \f$ H_1(T^2)=\mathbb{Z}^2 \f$. `TorusTwist` wraps an integer
/// \f$ 2\times 2 \f$ matrix \f$ \begin{psmallmatrix} a & b \\ c & d
/// \end{psmallmatrix} \f$ and realizes its action on the product-lattice torus
/// `SimplicialProduct(S¹,S¹)` — whose vertices are the pairs
/// \f$ (i,j)\in\mathbb{Z}_n\times\mathbb{Z}_n \f$ with id \f$ i\cdot n + j \f$
/// (\f$ n = |V(S^1)| \f$) — as the vertex permutation
/// \f$ (i,j)\mapsto(ai+bj,\,ci+dj)\bmod n \f$.
///
/// The two \f$ \mathrm{SL}(2,\mathbb{Z}) \f$ generators are provided: the Dehn
/// twist \f$ T=\begin{psmallmatrix}1&1\\0&1\end{psmallmatrix} \f$ (the shear
/// \f$ (i,j)\mapsto(i+j,j) \f$) and \f$ S=\begin{psmallmatrix}0&-1\\1&0
/// \end{psmallmatrix} \f$ (the \f$ (i,j)\mapsto(-j,i) \f$ rotation). They obey
/// the modular relations \f$ S^4=I \f$ and \f$ (ST)^3=S^2 \f$ as exact integer
/// matrices (see `satisfiesModularRelations`).
///
/// ## Simplicial realization (the subtle point)
///
/// A vertex permutation is a *simplicial* automorphism only when it carries the
/// triangulation's top simplices to top simplices. For the staircase product
/// torus this is genuinely restrictive: that triangulation is **not**
/// vertex-transitive (its diagonals follow the global vertex order), so its only
/// linear simplicial automorphisms are the identity and the coordinate flip
/// \f$ (i,j)\mapsto(j,i) \f$ (`flip()`). Neither \f$ S \f$ (order 4) nor the
/// Dehn twist \f$ T \f$ preserves it — consistent with the classical fact that a
/// parabolic/finite-order-incompatible mapping class cannot be a simplicial
/// automorphism of a *fixed* torus triangulation. `isSimplicialAutomorphism`
/// reports this per matrix. The realizable `flip()` self-map is what
/// `Cobordism::selfGlue` uses to build a genuine (non-trivial, non-orientable)
/// torus bundle; \f$ S,T \f$ are still well-defined boundary vertex
/// permutations (their action on \f$ H_1 \f$ is exact) for the explicit-bijection
/// gluing once a compatible (e.g. layered) triangulation is supplied.
class TorusTwist {
  public:
    /// The mapping class with matrix \f$ \begin{psmallmatrix} a & b \\ c & d
    /// \end{psmallmatrix} \f$ acting on \f$ H_1(T^2)=\mathbb{Z}^2 \f$.
    TorusTwist(long a, long b, long c, long d) : a_(a), b_(b), c_(c), d_(d) {}

    /// The identity \f$ I \f$.
    [[nodiscard]] static TorusTwist identity();
    /// The \f$ \mathrm{SL}(2,\mathbb{Z}) \f$ generator
    /// \f$ S=\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix} \f$ (order 4).
    [[nodiscard]] static TorusTwist S();
    /// The Dehn twist \f$ T=\begin{psmallmatrix}1&1\\0&1\end{psmallmatrix} \f$
    /// (the shear \f$ (i,j)\mapsto(i+j,j) \f$).
    [[nodiscard]] static TorusTwist T();
    /// The coordinate flip \f$ \begin{psmallmatrix}0&1\\1&0\end{psmallmatrix} \f$
    /// — the one non-trivial linear *simplicial* automorphism of the staircase
    /// product torus (orientation-reversing; its mapping torus is a non-trivial
    /// torus bundle).
    [[nodiscard]] static TorusTwist flip();

    /// Matrix product \f$ (\text{this})\cdot(\text{rhs}) \f$.
    [[nodiscard]] TorusTwist compose(const TorusTwist &rhs) const;
    /// The \f$ k \f$-th power (\f$ k\ge 0 \f$; \f$ k=0 \f$ is the identity).
    [[nodiscard]] TorusTwist power(int k) const;
    /// Entry-wise equality of the two matrices.
    [[nodiscard]] bool equals(const TorusTwist &rhs) const;
    /// \f$ ad-bc \f$ (\f$ +1 \f$ orientation-preserving, \f$ -1 \f$ reversing).
    [[nodiscard]] long determinant() const;
    /// The matrix entries \f$ \{a,b,c,d\} \f$ (row-major).
    [[nodiscard]] std::array<long, 4> matrix() const;

    /// The vertex permutation \f$ (i,j)\mapsto(ai+bj,\,ci+dj)\bmod n \f$ on the
    /// \f$ n\times n \f$ product torus (vertex id \f$ i\cdot n+j \f$), as a real
    /// id→id map ready to pass to `Cobordism::selfGlue`/`glue`. Requires
    /// \f$ n\ge 1 \f$.
    [[nodiscard]] std::map<std::uint64_t, std::uint64_t> vertexPermutation(
        int n) const;

    /// Whether `permutation` (a vertex id→id bijection) carries every top
    /// simplex of `torus` to a top simplex — i.e. is a genuine simplicial
    /// automorphism of that complex.
    [[nodiscard]] static bool isSimplicialAutomorphism(
        const Spacetime &torus,
        const std::map<std::uint64_t, std::uint64_t> &permutation);

    /// Whether the defining \f$ \mathrm{SL}(2,\mathbb{Z}) \f$ relations hold:
    /// \f$ S^4=I \f$ and \f$ (ST)^3=S^2 \f$ (checked as exact integer matrices,
    /// hence independent of any triangulation).
    [[nodiscard]] static bool satisfiesModularRelations();

  private:
    long a_, b_, c_, d_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TORUSTWIST_H
