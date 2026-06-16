// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_BOUNDARYSTATESPACE_H
#define TESSERA_COBORDISM_BOUNDARYSTATESPACE_H

#include <Eigen/Core>

#include <cstdint>
#include <memory>
#include <vector>

#include "cobordism/Cochain.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

class PreparedBoundaryState;

/// # BoundaryStateSpace
///
/// The DW boundary Hilbert space of a closed surface \f$ \Sigma \f$, as a
/// per-\f$ \Sigma \f$ context / factory — conceptually \f$ Z(\Sigma) =
/// \mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$. It owns \f$ \Sigma \f$ and the
/// cached Hodge harmonic basis of \f$ \ker L_1(\Sigma) \f$, and it manufactures
/// the value objects (`PreparedBoundaryState`) that live in it.
///
/// Two distinct objects live on \f$ \Sigma \f$:
///
/// - The **Hodge harmonic 1-forms** \f$ \ker L_1(\Sigma) \f$ — the *spectral
///   qubit*, a complex vector space of dimension \f$ b_1(\Sigma) \f$, whose
///   elements are 1-forms (`Cochain`s of degree 1, length \f$ |C_1(\Sigma)| \f$).
///   This is `HodgeLaplacian(\Sigma).harmonics(1)`, computed once at construction
///   and cached here as the orthonormal basis \f$ \{h_i\} \f$.
/// - The **DW boundary Hilbert space**
///   \f$ Z(\Sigma) = \mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$ — the
///   flat-connection-class basis, of dimension \f$ 2^{b_1(\Sigma)} \f$, indexed
///   by the holonomy class of a flat \f$ \mathbb{Z}_2 \f$ connection. This is the
///   space `PreparedBoundaryState` wraps and `DijkgraafWitten::amplitude()`
///   consumes.
///
/// ## The \f$ b_1 \f$ vs \f$ 2^{b_1} \f$ reconciliation
///
/// \f$ H^1(\Sigma;\mathbb{Z}_2) \cong (\mathbb{Z}_2)^{b_1} \f$ has \f$ b_1 \f$
/// distinguished **single-generator** classes — the weight-one holonomy patterns
/// \f$ e_i \f$ ("holonomy 1 around the \f$ i \f$-th cycle, 0 around the rest").
/// In the `gf2Span` enumeration that orders \f$ Z(\Sigma) \f$ (mask
/// \f$ 0\dots2^{b_1}-1 \f$, basis vector \f$ i \f$ alone at mask \f$ 2^i \f$),
/// these are exactly the amplitudes at indices \f$ 2^0, 2^1, \dots, 2^{b_1-1} \f$.
/// This `harmonic i \to amplitude index 2^i` convention is owned here
/// (`generatorIndex`) and is the single place it is encoded; callers never spell
/// out a power of two.
///
/// ## The factories
///
/// - `prepare(form)`: \f$ \ker L_1(\Sigma) \to Z(\Sigma) \f$. Project a harmonic
///   1-form \f$ \psi \f$ onto the orthonormal harmonic basis \f$ \{h_i\} \f$ to
///   get coordinates \f$ c_i = \langle h_i, \psi\rangle \f$, scatter \f$ c_i \f$
///   onto the amplitude at index \f$ 2^i \f$ (the trivial class and every
///   multi-generator class carry 0), and return the `PreparedBoundaryState`. A
///   component of the input outside \f$ \ker L_1 \f$ is projected out.
/// - `state(amplitudes)`: wrap a raw \f$ Z(\Sigma) \f$ amplitude vector (the
///   flat-connection-class basis, length \f$ 2^{b_1} \f$) as a
///   `PreparedBoundaryState` over this space, unchanged.
///
/// The adjoint of `prepare` — recovering the harmonic 1-form — is
/// `PreparedBoundaryState::readout()`. Because the harmonic basis is orthonormal
/// (the `SelfAdjointEigenSolver` eigenvectors of the symmetric Hodge Laplacian)
/// `prepare` is an **isometry** and `readout() \circ prepare` is the **identity**
/// on \f$ \ker L_1 \f$; consequently, on the trivial cobordism
/// \f$ \Sigma\times[0,T] \f$ (\f$ Z(W)=\mathrm{id} \f$),
/// `DijkgraafWitten::amplitude(prepare(\psi), prepare(\phi))` reproduces
/// \f$ \langle\psi|\phi\rangle \f$.
///
/// Reuses `HodgeLaplacian` (\f$ k=1 \f$) for the harmonics — it does not
/// reimplement them; the basis is computed once at construction and cached. A
/// `BoundaryStateSpace` must be owned by a `std::shared_ptr` (the
/// `PreparedBoundaryState`s it makes keep a handle back to it via
/// `shared_from_this`); construct it with `std::make_shared` in C++.
class BoundaryStateSpace
    : public std::enable_shared_from_this<BoundaryStateSpace> {
  public:
    /// Build \f$ Z(\Sigma) \f$ over a closed surface \f$ \Sigma \f$. The harmonic
    /// 1-form basis \f$ \ker L_1(\Sigma) \f$ is computed once
    /// (`HodgeLaplacian` at \f$ k=1 \f$) and cached; the held `shared_ptr` keeps
    /// \f$ \Sigma \f$ alive. `tol` is the \f$ |\lambda|<\text{tol} \f$ harmonic
    /// threshold and `metric` selects volume vs. unit Hodge weights — both
    /// forwarded to `HodgeLaplacian::harmonics(1, tol, metric)`; the embedding is
    /// isometric for either choice (orthonormal eigenvectors).
    /// @throws std::runtime_error if \f$ \Sigma \f$ is null, or if
    ///   \f$ b_1(\Sigma) > 24 \f$ (the `gf2Span` materialization cap — the
    ///   \f$ 2^{b_1} \f$ boundary space would be too large).
    explicit BoundaryStateSpace(std::shared_ptr<Spacetime> sigma,
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

    /// The cached orthonormal harmonic 1-form basis \f$ \{h_i\} \f$ as `Cochain`s
    /// (degree 1, the \f$ k=1 \f$ simplex ordering), in ascending-eigenvalue
    /// order — the deterministic basis `prepare` / `readout` use.
    [[nodiscard]] const std::vector<Cochain> &harmonics() const noexcept;

    /// The \f$ b_1 \f$ flat-connection-class indices in \f$ Z(\Sigma) \f$ that
    /// carry harmonic data: \f$ (2^0, 2^1, \dots, 2^{b_1-1}) \f$, the
    /// single-generator classes (harmonic \f$ i \f$ lands on index \f$ 2^i \f$).
    [[nodiscard]] std::vector<int> generatorIndices() const;

    /// Prepare a boundary state from a harmonic 1-form: \f$ \ker L_1(\Sigma)
    /// \to Z(\Sigma) \f$. `form` is a degree-1 `Cochain` of length \f$ |C_1| \f$;
    /// its harmonic-basis coordinates \f$ c_i=\langle h_i,\text{form}\rangle \f$
    /// are scattered onto the amplitudes at indices \f$ 2^i \f$, the rest zero. A
    /// non-harmonic component of `form` is projected out (dropped).
    /// @throws std::invalid_argument if `form.degree() != 1` or
    ///   `form.size() != numEdges()`.
    [[nodiscard]] PreparedBoundaryState prepare(const Cochain &form) const;

    /// Wrap a raw \f$ Z(\Sigma) \f$ amplitude vector (the flat-connection-class
    /// basis, length \f$ 2^{b_1} \f$) as a `PreparedBoundaryState` over this
    /// space — the direct counterpart to `prepare` for states already expressed
    /// in the holonomy-class basis.
    /// @throws std::invalid_argument if `amplitudes.size() != boundaryDimension()`.
    [[nodiscard]] PreparedBoundaryState state(
        const Eigen::VectorXcd &amplitudes) const;

  private:
    friend class PreparedBoundaryState;

    /// The single source of the `harmonic i \to Z(\Sigma) index 2^i` convention.
    [[nodiscard]] int generatorIndex(int harmonic) const noexcept;

    /// Gather the generator amplitudes \f$ c_i = \text{amplitudes}[2^i] \f$ from
    /// a \f$ Z(\Sigma) \f$ vector and rebuild the harmonic 1-form
    /// \f$ \sum_i c_i h_i \f$ as a degree-1 `Cochain` — the kernel of
    /// `PreparedBoundaryState::readout()`.
    /// @throws std::invalid_argument if `amplitudes.size() != boundaryDimension()`.
    [[nodiscard]] Cochain reconstruct(const Eigen::VectorXcd &amplitudes) const;

    std::shared_ptr<Spacetime> sigma_;
    int numEdges_{0};
    int b1_{0};
    // The degree-1 simplex ordering (sorted vertex-id edge tuples), canonical
    // ChainComplex column order; the index space a harmonic 1-form Cochain lives
    // over. Kept so `reconstruct` can build a Cochain even when b_1 = 0.
    std::vector<std::vector<std::uint64_t>> edges_{};
    // The cached ker L_1(Sigma) basis, exactly HodgeLaplacian::harmonics(1, ...).
    std::vector<Cochain> harmonics_{};
    // |C_1| x b_1 matrix, column i = harmonics_[i].coeffs(); the dense view the
    // prepare/readout projection math indexes (W_k-orthonormal columns).
    Eigen::MatrixXcd harmonicMatrix_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BOUNDARYSTATESPACE_H
