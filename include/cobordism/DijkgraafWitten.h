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
#include <vector>

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

    /// # The Dijkgraaf–Witten state sum with boundary
    ///
    /// For a 3-manifold \f$ W \f$ **with boundary** \f$ \partial W \f$ the state
    /// sum is no longer a scalar but a vector in the boundary Hilbert space
    /// \f$ Z(\partial W) \f$. Holding the boundary connection \f$ g|_{\partial W} \f$
    /// fixed and summing the same tetrahedron product
    /// \f$ \prod_t \omega(g_{01},g_{12},g_{23})^{\varepsilon_t} \f$ over the
    /// interior flat fields produces, for each boundary flat-connection class, a
    /// complex amplitude — an element of \f$ Z(\partial W) \f$.
    ///
    /// Concretely the sum is taken over the **bulk cohomology classes**
    /// \f$ [g] \in H^1(W;\mathbb{Z}_2) \f$ (the gauge-inequivalent flat
    /// \f$ \mathbb{Z}_2 \f$ connections — the interior fields modulo gauge),
    /// binned by the class of their restriction to each boundary component. A
    /// boundary component \f$ \Sigma \f$ (a closed surface) carries the DW Hilbert
    /// space \f$ Z(\Sigma) = \mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$ of dimension
    /// \f$ 2^{b_1(\Sigma)} \f$, indexed by the holonomy class of the connection.
    /// Summing over cohomology classes (rather than the gauge-redundant cocycle
    /// space) is exactly the gauge-fixed "sum over interior fields"; no gauge
    /// volume divides it, and it is normalized so the trivial cobordism
    /// \f$ \Sigma\times[0,T] \f$ is the identity on \f$ Z(\Sigma) \f$.
    ///
    /// For the real-valued \f$ \mathbb{Z}_2 \f$ cocycles (`Trivial` \f$ \equiv 1 \f$
    /// and `Sign` \f$ = \pm 1 \f$) the orientation exponent is immaterial —
    /// \f$ x^{\pm 1} = x \f$ for \f$ x \in \{+1,-1\} \f$, and \f$ \omega^{-1} =
    /// \bar\omega = \omega \f$ — so no (relative) fundamental class is needed
    /// (none exists for a manifold with boundary) and the product matches the
    /// closed case.
    ///
    /// `boundaryVector()` returns the full element of \f$ Z(\partial W) \f$: the
    /// amplitude for every joint
    /// boundary flat-connection class, flattened row-major over the boundary
    /// components (each component ordered deterministically by its sorted
    /// top-face list). Length \f$ = \prod_i 2^{b_1(\Sigma_i)} \f$.
    /// @throws std::runtime_error if \f$ W \f$ is null, not a 3-manifold, or
    ///   closed (no boundary — use `partitionFunction()`).
    [[nodiscard]] std::vector<std::complex<double>> boundaryVector() const;

    /// The per-boundary-component Hilbert-space dimensions
    /// \f$ 2^{b_1(\Sigma_i)} \f$, in the same deterministic component order used
    /// by `boundaryVector()` / `map()`.
    [[nodiscard]] std::vector<int> boundaryDimensions() const;

    /// The boundary state sum read as a linear map \f$ Z(\Sigma_B) \to
    /// Z(\Sigma_A) \f$ when \f$ \partial W \f$ has exactly two components: a dense
    /// matrix returned as rows, `rows = 2^{b_1(\Sigma_A)}` (component 0),
    /// `cols = 2^{b_1(\Sigma_B)}` (component 1), in the flat-connection-class
    /// basis. For the trivial cobordism \f$ \Sigma\times[0,T] \f$ this is the
    /// identity \f$ \mathrm{id}_{Z(\Sigma)} \f$.
    /// @throws std::runtime_error if \f$ \partial W \f$ does not have exactly two
    ///   connected components.
    [[nodiscard]] std::vector<std::vector<std::complex<double>>> map() const;

    /// The transition amplitude \f$ \langle \psi_A | Z(W) | \psi_B \rangle \f$
    /// for boundary states \f$ \psi_A \in Z(\Sigma_A) \f$, \f$ \psi_B \in
    /// Z(\Sigma_B) \f$ given in the flat-connection-class basis: the contraction
    /// \f$ \sum_{a,b} \overline{\psi_A[a]}\, Z(W)_{ab}\, \psi_B[b] \f$ with the
    /// two-component boundary map. For the trivial cobordism (\f$ Z(W) =
    /// \mathrm{id} \f$) this reduces to the inner product
    /// \f$ \langle \psi_A | \psi_B \rangle \f$. The boundary states are prepared
    /// from the harmonic 1-forms \f$ \ker L_1(\Sigma) \f$ (the qubit of dimension
    /// \f$ b_1 \f$, kept distinct from the \f$ 2^{b_1} \f$ flat-connection count).
    /// @throws std::runtime_error if \f$ \partial W \f$ is not two components, or
    ///   `std::invalid_argument` if the state lengths do not match the map dims.
    [[nodiscard]] std::complex<double> amplitude(
        const std::vector<std::complex<double>> &psiA,
        const std::vector<std::complex<double>> &psiB) const;

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

    /// The computed boundary state sum: the per-component Hilbert dimensions and
    /// the flat (row-major over components) amplitude vector \f$ \in
    /// Z(\partial W) \f$. The shared kernel of `boundaryVector()`, `map()`,
    /// `boundaryDimensions()`, and `amplitude()`.
    struct Boundary {
      std::vector<int> dims;                          // 2^{b_1(Σ_i)} per component
      std::vector<std::complex<double>> amplitudes;   // flat, ∏ dims long
    };
    [[nodiscard]] Boundary computeBoundary() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DIJKGRAAFWITTEN_H
