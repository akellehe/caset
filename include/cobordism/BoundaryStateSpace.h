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

#ifndef TESSERA_COBORDISM_BOUNDARYSTATEPREP_H
#define TESSERA_COBORDISM_BOUNDARYSTATEPREP_H

#include <complex>
#include <memory>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # BoundaryStatePrep
///
/// The **spectral → Dijkgraaf–Witten boundary preparation map** for a closed
/// surface \f$ \Sigma \f$, and its readout (the adjoint).
///
/// Two distinct objects live on \f$ \Sigma \f$:
///
/// - The **Hodge harmonic 1-forms** \f$ \ker L_1(\Sigma) \f$ — the *spectral
///   qubit*, a complex vector space of dimension \f$ b_1(\Sigma) \f$, whose
///   elements are 1-forms (cochains of length \f$ |C_1(\Sigma)| \f$). This is
///   `HodgeLaplacian(\Sigma).harmonics(1)`.
/// - The **DW boundary Hilbert space**
///   \f$ Z(\Sigma) = \mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$ — the
///   flat-connection-class basis, of dimension \f$ 2^{b_1(\Sigma)} \f$, indexed
///   by the holonomy class of a flat \f$ \mathbb{Z}_2 \f$ connection. This is the
///   space `DijkgraafWitten::amplitude()` / `map()` / `boundaryVector()` consume.
///
/// `DijkgraafWitten::amplitude()` already sandwiches the topological state sum
/// between boundary states *prepared from* the harmonics, but that preparation
/// was caller-side and implicit. This class makes it a first-class, tested
/// operation.
///
/// ## The \f$ b_1 \f$ vs \f$ 2^{b_1} \f$ reconciliation
///
/// \f$ H^1(\Sigma;\mathbb{Z}_2) \cong (\mathbb{Z}_2)^{b_1} \f$ has \f$ b_1 \f$
/// distinguished **single-generator** classes — the weight-one holonomy patterns
/// \f$ e_i \f$ ("holonomy 1 around the \f$ i \f$-th cycle, 0 around the rest").
/// In the `gf2Span` enumeration that orders \f$ Z(\Sigma) \f$ (mask
/// \f$ 0\dots2^{b_1}-1 \f$, basis vector \f$ i \f$ alone at mask \f$ 2^i \f$),
/// these are exactly the amplitudes at indices \f$ 2^0, 2^1, \dots, 2^{b_1-1} \f$.
///
/// The preparation **embeds the spectral qubit onto those generator slots**:
/// the \f$ i \f$-th orthonormal harmonic 1-form (column \f$ i \f$ of
/// `harmonics()`, ascending-eigenvalue order) carries its amplitude to the
/// flat-connection class at \f$ Z(\Sigma) \f$-index \f$ 2^i \f$. The trivial
/// class (index 0) and **all** multi-generator classes (indices that are not
/// powers of two) carry amplitude 0. So a prepared state occupies a
/// \f$ b_1 \f$-dimensional coordinate subspace of the \f$ 2^{b_1} \f$-dimensional
/// \f$ Z(\Sigma) \f$.
///
/// ## The maps
///
/// `prepare`: \f$ \ker L_1(\Sigma) \to Z(\Sigma) \f$. Project a harmonic 1-form
/// \f$ \psi \f$ (length \f$ |C_1| \f$) onto the orthonormal harmonic basis
/// \f$ \{h_i\} \f$ to get coordinates \f$ c_i = \langle h_i, \psi\rangle \f$, then
/// scatter \f$ c_i \f$ onto the amplitude at index \f$ 2^i \f$. (Any component of
/// the input outside \f$ \ker L_1 \f$ is projected out.)
///
/// `readout`: \f$ Z(\Sigma) \to \ker L_1(\Sigma) \f$, the adjoint of `prepare`.
/// Gather the generator amplitudes \f$ c_i = \psi_{Z}[2^i] \f$ and reconstruct
/// the 1-form \f$ \sum_i c_i h_i \f$.
///
/// Because the harmonic basis is orthonormal (the `SelfAdjointEigenSolver`
/// eigenvectors of the symmetric Hodge Laplacian) `prepare` is an **isometry**
/// \f$ \langle \mathrm{prepare}(\psi)\,|\,\mathrm{prepare}(\phi)\rangle =
/// \langle\psi|\phi\rangle \f$ for \f$ \psi,\phi\in\ker L_1 \f$, and
/// `readout` \f$ \circ \f$ `prepare` is the **identity** on \f$ \ker L_1 \f$.
/// Consequently, on the trivial cobordism \f$ \Sigma\times[0,T] \f$ (where
/// \f$ Z(W)=\mathrm{id} \f$) `DijkgraafWitten::amplitude(prepare(ψ), prepare(φ))`
/// reproduces \f$ \langle\psi|\phi\rangle \f$.
///
/// Reuses `HodgeLaplacian` (\f$ k=1 \f$) for the harmonics — it does not
/// reimplement them; the basis is computed once at construction and cached.
class BoundaryStatePrep {
  public:
    /// Build the preparation map over a closed surface \f$ \Sigma \f$. The
    /// harmonic 1-form basis \f$ \ker L_1(\Sigma) \f$ is computed once
    /// (`HodgeLaplacian` at \f$ k=1 \f$) and cached; the held `shared_ptr` keeps
    /// \f$ \Sigma \f$ alive. `tol` is the \f$ |\lambda|<\text{tol} \f$ harmonic
    /// threshold and `metric` selects volume vs. unit Hodge weights — both
    /// forwarded to `HodgeLaplacian::harmonics(1, tol, metric)`; the embedding is
    /// isometric for either choice (orthonormal eigenvectors).
    /// @throws std::runtime_error if \f$ \Sigma \f$ is null, or if
    ///   \f$ b_1(\Sigma) > 24 \f$ (the `gf2Span` materialization cap — the
    ///   \f$ 2^{b_1} \f$ boundary space would be too large).
    explicit BoundaryStatePrep(std::shared_ptr<Spacetime> sigma,
                               double tol = 1e-9, bool metric = true);

    /// \f$ b_1(\Sigma) = \dim\ker L_1(\Sigma) \f$: the spectral-qubit dimension
    /// (the number of harmonic 1-forms).
    [[nodiscard]] int harmonicDimension() const;

    /// \f$ \dim Z(\Sigma) = 2^{b_1(\Sigma)} \f$: the DW boundary Hilbert-space
    /// dimension (the length of a prepared state).
    [[nodiscard]] int boundaryDimension() const;

    /// \f$ |C_1(\Sigma)| \f$: the number of edges — the length of a harmonic
    /// 1-form (the input to `prepare` / output of `readout`).
    [[nodiscard]] int numEdges() const;

    /// The orthonormal harmonic 1-form basis as a flat row-major
    /// \f$ |C_1|\times b_1 \f$ array (column \f$ i \f$, entries at indices
    /// \f$ e\cdot b_1 + i \f$, is the \f$ i \f$-th basis form) — the deterministic
    /// basis `prepare` / `readout` use, exposed for a numpy/Hodge cross-check.
    [[nodiscard]] std::vector<std::complex<double>> harmonics() const;

    /// The \f$ b_1 \f$ flat-connection-class indices in \f$ Z(\Sigma) \f$ that
    /// carry harmonic data: \f$ (2^0, 2^1, \dots, 2^{b_1-1}) \f$, the
    /// single-generator classes. (Documents the \f$ b_1 \f$ vs \f$ 2^{b_1} \f$
    /// reconciliation: harmonic \f$ i \f$ lands on index \f$ 2^i \f$.)
    [[nodiscard]] std::vector<int> generatorIndices() const;

    /// Prepare a DW boundary state from a harmonic 1-form: \f$ \ker L_1(\Sigma)
    /// \to Z(\Sigma) \f$. `form` is a 1-form of length \f$ |C_1| \f$; its
    /// harmonic-basis coordinates \f$ c_i=\langle h_i,\text{form}\rangle \f$ are
    /// scattered onto the amplitudes at indices \f$ 2^i \f$, the rest zero.
    /// Returns a vector of length \f$ 2^{b_1} \f$. A non-harmonic component of
    /// `form` is projected out (dropped).
    /// @throws std::invalid_argument if `form.size() != numEdges()`.
    [[nodiscard]] std::vector<std::complex<double>> prepare(
        const std::vector<std::complex<double>> &form) const;

    /// Read a harmonic 1-form back out of a DW boundary state (the adjoint of
    /// `prepare`): \f$ Z(\Sigma) \to \ker L_1(\Sigma) \f$. Gathers the generator
    /// amplitudes \f$ c_i = \text{state}[2^i] \f$ and returns the 1-form
    /// \f$ \sum_i c_i h_i \f$ (length \f$ |C_1| \f$). `readout(prepare(form)) ==
    /// form` for `form` \f$ \in \ker L_1(\Sigma) \f$; amplitudes off the generator
    /// indices are ignored.
    /// @throws std::invalid_argument if `state.size() != boundaryDimension()`.
    [[nodiscard]] std::vector<std::complex<double>> readout(
        const std::vector<std::complex<double>> &state) const;

  private:
    std::shared_ptr<Spacetime> sigma_;
    int numEdges_{0};
    int b1_{0};
    // Flat row-major |C_1| x b_1 orthonormal harmonic basis (columns), exactly
    // HodgeLaplacian::harmonics(1, tol, metric); element (edge e, harmonic i) is
    // at index e * b1_ + i.
    std::vector<std::complex<double>> harmonics_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BOUNDARYSTATEPREP_H
