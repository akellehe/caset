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

#ifndef TESSERA_COBORDISM_DIJKGRAAFWITTEN_H
#define TESSERA_COBORDISM_DIJKGRAAFWITTEN_H

#include <complex>
#include <memory>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// The two normalized classes of \f$ Z^3(\mathbb{Z}_2; U(1)) \f$ used as the
/// Dijkgraaf–Witten weight: the trivial cocycle \f$ \omega \equiv 1 \f$ and the
/// nontrivial sign cocycle \f$ \omega(a,b,c) = (-1)^{abc} \f$ (the generator of
/// \f$ H^3(\mathbb{Z}_2; U(1)) \cong \mathbb{Z}_2 \f$, the cochain-level cup
/// cube \f$ t^3 \f$).
enum class Cocycle {
  Trivial,  ///< \f$ \omega \equiv 1 \f$ (untwisted state sum).
  Sign,     ///< \f$ \omega(a,b,c) = (-1)^{abc} \f$.
};

/// # DijkgraafWitten
///
/// The Dijkgraaf–Witten partition function of a closed oriented triangulated
/// 3-manifold \f$ W \f$ with gauge group \f$ \mathbb{Z}_2 \f$ and a 3-cocycle
/// \f$ \omega \in Z^3(\mathbb{Z}_2; U(1)) \f$:
/// \f[
///   Z(W) = \frac{1}{2^{|V|}} \sum_{\text{flat } g}
///          \prod_{\text{tetrahedra } t}
///          \omega(g_{01}, g_{12}, g_{23})^{\varepsilon_t}.
/// \f]
/// Here \f$ g \in C^1(W; \mathbb{Z}_2) \f$ is a **flat** connection — a value
/// \f$ g_e \in \{0,1\} \f$ on each edge with \f$ (dg)|_\triangle = g_{01} +
/// g_{12} - g_{02} \equiv 0 \pmod 2 \f$ on every triangle — i.e. an element of
/// \f$ \ker(d_1 \bmod 2) \f$, where the coboundary \f$ d_1 = \partial_2^\top \f$
/// is the transpose of `ChainComplex::boundaryMatrix(2)`. \f$ \varepsilon_t =
/// \pm 1 \f$ is each tetrahedron's orientation sign from the fundamental class
/// (`ChainComplex::fundamentalClass()`), and on an ordered tetrahedron
/// \f$ [v_0 < v_1 < v_2 < v_3] \f$ the weight reads the connection on its three
/// "consecutive" edges \f$ (v_0,v_1), (v_1,v_2), (v_2,v_3) \f$ — the
/// Alexander–Whitney faces — mapped to their \f$ C_1 \f$ (edge) indices.
///
/// The sum is the gauge-redundant enumeration over the whole flat space
/// \f$ \ker(d_1) \f$ (\f$ 2^{\text{nullity}} \f$ connections, materialized via
/// `gf2Span`); the division by \f$ 2^{|V|} \f$ is the gauge volume, leaving the
/// topological invariant. This brute-force enumeration is intended for the
/// small triangulations on which the invariant is checked by hand; the
/// underlying `gf2Span` refuses a nullity too large to materialize.
class DijkgraafWitten {
  public:
    /// Build the state sum over a triangulation \f$ W \f$ with the chosen
    /// cocycle. \f$ W \f$ is read at `partitionFunction()` time; the held
    /// `shared_ptr` keeps it alive.
    DijkgraafWitten(std::shared_ptr<Spacetime> W, Cocycle w);

    /// The partition function \f$ Z(W) \in \mathbb{C} \f$.
    /// @throws std::runtime_error if \f$ W \f$ is null or not a closed oriented
    ///   3-manifold (dimension \f$ \neq 3 \f$, or no fundamental class), or
    ///   `std::invalid_argument` (from `gf2Span`) if the flat space is too large
    ///   to enumerate.
    [[nodiscard]] std::complex<double> partitionFunction() const;

    /// Whether the cocycle \f$ \omega \f$ for the given choice satisfies the
    /// normalized 3-cocycle (pentagon) identity over \f$ \mathbb{Z}_2 \f$:
    /// \f$ \omega(b,c,d)\,\omega(a, b{+}c, d)\,\omega(a,b,c) =
    /// \omega(a{+}b, c, d)\,\omega(a, b, c{+}d) \f$ for every
    /// \f$ (a,b,c,d) \in \mathbb{Z}_2^4 \f$ (brute-forced over all 16 tuples).
    /// Both `Trivial` and `Sign` are genuine cocycles, so both return true;
    /// it is the prerequisite that makes the state sum a topological invariant.
    [[nodiscard]] static bool isCocycle(Cocycle w);

  private:
    std::shared_ptr<Spacetime> W_;
    Cocycle cocycle_;

    /// Evaluate \f$ \omega(a,b,c) \f$ for a cocycle choice, \f$ a,b,c \in
    /// \{0,1\} \f$.
    [[nodiscard]] static std::complex<double> omega(Cocycle w, int a, int b, int c);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DIJKGRAAFWITTEN_H
