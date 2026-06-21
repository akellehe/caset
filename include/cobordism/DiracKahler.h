// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_DIRACKAHLER_H
#define TESSERA_COBORDISM_DIRACKAHLER_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # DiracKahler
///
/// The discrete **Dirac-Kahler operator** \f$ D = d + \delta \f$ on the
/// inhomogeneous cochain complex \f$ \Omega = \bigoplus_{k=0}^{n}\Omega^k \f$ of
/// a `Spacetime` — the principled, label-independent square root of the
/// `HodgeLaplacian`. It is the natural **distinguished frame** the deferred
/// operator read-out (`MergeCobordism::operatorU()`/`choiState()`) lacks, and it
/// bridges the electric-charge current \f$ j^0 \f$ to the candidate flavor/taste
/// index (the 4-fold Dirac-Kahler "doubling").
///
/// ## The operator
///
/// On the **total** cochain space \f$ \Phi \in \bigoplus_k \Omega^k \f$ (flat
/// dimension \f$ \sum_k |C_k| \f$) the operator is assembled block-by-block from
/// the same integer boundary maps \f$ \partial_k \f$ (`ChainComplex`) and
/// diagonal inner-product weights \f$ W_k \f$ (`HodgeLaplacian::weights`) the
/// `HodgeLaplacian` already uses, in the boundary convention `HodgeLaplacian`
/// squares (`include/cobordism/HodgeLaplacian.h:31-46,48-68`):
///
///  - the **boundary** part \f$ \partial \f$ (degree-lowering) has block
///    \f$ \Omega^k \to \Omega^{k-1} \f$ equal to \f$ \partial_k \f$
///    (the integer `ChainComplex::boundaryMatrix(k)`, rows \f$|C_{k-1}|\f$,
///    cols \f$|C_k|\f$);
///  - the **codifferential** part \f$ \delta = \partial^* \f$ (degree-raising)
///    has block \f$ \Omega^{k-1} \to \Omega^k \f$ equal to the metric adjoint
///    \f$ \partial_k^* = W_k^{-1}\partial_k^{\top}W_{k-1} \f$.
///
/// Then, because \f$ \partial^2 = 0 \f$ and \f$ \delta^2 = (\partial^*)^2 = 0 \f$,
/// the square is **block-diagonal per degree** and reproduces the Hodge Laplacian
/// exactly:
/// \f$ (d+\delta)^2 = \delta\partial + \partial\delta = L_k \f$, with
/// \f$ L_k = W_k^{-1}\partial_k^{\top}W_{k-1}\partial_k
///       + \partial_{k+1}W_{k+1}^{-1}\partial_{k+1}^{\top}W_k \f$ —
/// the directly-assembled `HodgeLaplacian` (`signedLaplacian`). On a complex
/// whose per-degree weights are uniform (e.g. the uniform \f$ l^2=1 \f$ metric,
/// where \f$ W_k = c_k I \f$) this coincides block-for-block with the symmetric
/// metric `HodgeLaplacian::laplacian(k, metric=true)`; the `lorentzian` path
/// reproduces the signed-weight d'Alembertian blocks (\f$ k\ge 1 \f$). The
/// matrix is rebuilt from the current edge weights on each call (matching the
/// `HodgeLaplacian` lazy convention).
///
/// The \f$ k=0 \f$ block of \f$ D^2 \f$ is the **0-form Hodge Laplacian**
/// \f$ \partial_1 W_1^{-1}\partial_1^{\top} \f$, which equals
/// `HodgeLaplacian::laplacian(0)` (the Hermitian U(1)-graph Laplacian \f$ D-A \f$)
/// only for unit, real edge weights; under the `lorentzian` (signed-weight)
/// path the graph Laplacian and the signed 0-form d'Alembertian differ, so the
/// reproduction is asserted over the \f$ k\ge 1 \f$ d'Alembertian blocks there.
///
/// ## Clifford / gamma structure and the 4-fold multiplicity
///
/// The gamma generators are the **Clifford action of unit 1-cochains** on the
/// Kahler-Atiyah form fiber \f$ \Lambda(\mathbb{R}^d) \f$ (dimension \f$ 2^d \f$):
/// \f$ \gamma^a = \varepsilon(e^a) + \eta^{aa}\,\iota(e^a) \f$, exterior
/// multiplication plus (metric) interior multiplication by the \f$ a \f$-th basis
/// 1-form. They satisfy the Clifford algebra
/// \f$ \{\gamma^a,\gamma^b\} = 2\eta^{ab} I \f$ exactly (Euclidean
/// \f$ \eta=\delta \f$, or the Lorentzian \f$ \mathrm{diag}(-1,+1,\dots) \f$ under
/// the `lorentzian` flag), and \f$ D = \gamma^\mu\partial_\mu \f$ is the discrete
/// Dirac operator. The framework spacetime dimension is \f$ d=4 \f$: the fiber is
/// \f$ 2^4 = 16 \f$-dimensional and decomposes as \f$ 16 = 4\times 4 \f$ —
/// **4 Dirac spinor components** \f$ \times \f$ **4 degenerate copies**. That
/// 4-fold "doubling" is the `multiplicity()`, the candidate **flavor/taste
/// index**.
///
/// ## The conserved current \f$ j^0 \f$ (charge density)
///
/// The U(1) Noether current of the Dirac-Kahler field is
/// \f$ j^a = \bar\Phi\,\gamma^a\Phi \f$; its time component
/// \f$ j^0 = \bar\Phi\,\gamma^0\Phi = \pm\,\Phi^\dagger\Phi \f$ is the **charge
/// density**, realized per cell as \f$ j^0_c = W_c\,|\Phi_c|^2 \f$ (the
/// \f$ W \f$-weighted modulus). Summed over a closed slice it integrates to the
/// carried U(1) charge \f$ \langle\Phi,\Phi\rangle_W \f$; on a single closed
/// harmonic \f$ \Phi \f$ (a `HodgeLaplacian::harmonicMatrix` row lifted to the
/// total space) this is the harmonic's carried charge. The eventual Gauss-law
/// cross-check (the electric-charge-density ticket) compares this to the
/// period-derived charge; until then the consistency check is that the summed
/// \f$ j^0 \f$ equals \f$ \langle\Phi,\Phi\rangle_W \f$.
///
/// ## Reduced-dimension caveat
///
/// The current `S^2 x I` cobordism is **2+1 D** (reduced): the full \f$ 4\times4 \f$
/// Dirac and the \f$ E \f$/\f$ B \f$ field-strength split need a 3+1 D
/// \f$ S^3 \f$-spatial-slice mesh. This class assembles \f$ D \f$ and the
/// \f$ D^2=L \f$ verification generically in \f$ n \f$ D from the mesh, while the
/// gamma/Clifford framework and `multiplicity()` report the **4D** values (a
/// fixed property of the Dirac-Kahler framework, not of the reduced mesh).
class DiracKahler {
  public:
    /// Construct over a triangulation. Edge weights are read lazily (at each
    /// matrix/current query), so the spacetime must outlive the operator; the
    /// held `shared_ptr` keeps it alive.
    explicit DiracKahler(std::shared_ptr<Spacetime> st);

    /// The mesh top dimension \f$ n \f$ (largest \f$ k \f$ with a \f$ k \f$-cell),
    /// or \f$ -1 \f$ for the empty complex.
    [[nodiscard]] int meshDimension() const;

    /// The total cochain dimension \f$ \sum_{k=0}^{n}|C_k| \f$ — the side length
    /// of the \f$ D \f$ matrix.
    [[nodiscard]] std::size_t totalDimension() const;

    /// Block offsets \f$ (o_0,\dots,o_{n+1}) \f$, length \f$ n+2 \f$, with
    /// \f$ o_k = \sum_{j<k}|C_j| \f$ and \f$ o_{n+1} = \text{totalDimension()} \f$:
    /// degree-\f$ k \f$ components occupy rows/cols \f$ [o_k, o_{k+1}) \f$ in the
    /// canonical `ChainComplex` cell order (`kSimplexVertices(k)`).
    [[nodiscard]] std::vector<std::size_t> blockOffsets() const;

    /// The Dirac-Kahler operator \f$ D = d+\delta \f$ as a flat row-major
    /// \f$ \text{totalDimension()}^2 \f$ complex array (real entries; imag 0).
    /// Block \f$ (\Omega^k\to\Omega^{k-1}) = \partial_k \f$ and block
    /// \f$ (\Omega^{k-1}\to\Omega^k) = W_k^{-1}\partial_k^{\top}W_{k-1} \f$.
    /// `metric = false` uses unit weights (the combinatorial Dirac operator);
    /// `lorentzian = true` uses the signed `HodgeLaplacian::weights(k, true)`
    /// (timelike cells negative), so the square is the signed d'Alembertian.
    [[nodiscard]] std::vector<std::complex<double>> matrix(
        bool metric = true, bool lorentzian = false) const;

    /// The square \f$ D^2 \f$ as a flat row-major
    /// \f$ \text{totalDimension()}^2 \f$ complex array. Block-diagonal per
    /// degree, each block the degree-\f$ k \f$ `HodgeLaplacian` (see the class
    /// docstring); the cross-degree blocks vanish (\f$ \partial^2=\delta^2=0 \f$).
    [[nodiscard]] std::vector<std::complex<double>> square(
        bool metric = true, bool lorentzian = false) const;

    /// The maximal Frobenius residual \f$ \max_k \| (D^2)_k - L_k \| \f$ between
    /// the degree-\f$ k \f$ diagonal block of \f$ D^2 \f$ and the independently
    /// assembled `HodgeLaplacian::laplacian(k, metric, lorentzian)`. Iterated over
    /// \f$ k = 0..n \f$ for the Euclidean path and \f$ k = 1..n \f$ for the
    /// `lorentzian` path (the \f$ k=0 \f$ `HodgeLaplacian` is the Hermitian graph
    /// Laplacian, not the signed 0-form d'Alembertian; see the class docstring).
    /// ~0 confirms \f$ (d+\delta)^2 = L \f$.
    [[nodiscard]] double laplacianResidual(
        bool metric = true, bool lorentzian = false) const;

    /// The framework spacetime dimension \f$ d = 4 \f$ (the 4D Dirac-Kahler
    /// framework; fixed, independent of the reduced mesh dimension).
    [[nodiscard]] int frameworkDimension() const;

    /// The Kahler-Atiyah form-fiber dimension \f$ 2^d = 16 \f$ (in 4D): the side
    /// length of each gamma matrix.
    [[nodiscard]] std::size_t gammaDimension() const;

    /// The 4-fold Dirac-Kahler **multiplicity** (the "doubling"):
    /// \f$ 2^{d/2} = 4 \f$ in 4D — the number of degenerate Dirac copies in
    /// \f$ \Lambda(\mathbb{C}^4) = 16 = 4_{\text{spinor}}\times 4_{\text{taste}} \f$.
    /// The candidate flavor/taste index. Pinned to 4 in the 4D framework.
    [[nodiscard]] int multiplicity() const;

    /// The metric signature \f$ \eta \f$ as a flat row-major
    /// \f$ d\times d \f$ array: Euclidean \f$ \delta_{ab} \f$, or the Lorentzian
    /// \f$ \mathrm{diag}(-1,+1,+1,+1) \f$ under `lorentzian = true`.
    [[nodiscard]] std::vector<double> signature(bool lorentzian = false) const;

    /// The \f$ d=4 \f$ gamma generators — the Clifford action of the unit
    /// 1-cochains on the form fiber \f$ \Lambda(\mathbb{R}^4) \f$ — as a list of
    /// \f$ \text{gammaDimension()}\times\text{gammaDimension()} \f$ flat
    /// row-major real matrices \f$ \gamma^a = \varepsilon(e^a)+\eta^{aa}\iota(e^a) \f$.
    /// They satisfy \f$ \{\gamma^a,\gamma^b\}=2\eta^{ab}I \f$ against
    /// `signature(lorentzian)`.
    [[nodiscard]] std::vector<std::vector<std::complex<double>>> gammas(
        bool lorentzian = false) const;

    /// The Clifford anticommutator residual
    /// \f$ \max_{a,b}\|\{\gamma^a,\gamma^b\} - 2\eta^{ab}I\|_F \f$ for the
    /// generators of `gammas(lorentzian)` against `signature(lorentzian)`. ~0.
    [[nodiscard]] double cliffordResidual(bool lorentzian = false) const;

    /// Lift a degree-\f$ k \f$ cochain `component` (length \f$ |C_k| \f$) into a
    /// total-space field (length `totalDimension()`), zero in every other degree.
    /// @throws std::runtime_error if `component` has the wrong length.
    [[nodiscard]] std::vector<std::complex<double>> lift(
        int k, const std::vector<std::complex<double>> &component) const;

    /// The per-cell charge density \f$ j^0_c = W_c\,|\Phi_c|^2 \f$ of a total-space
    /// field `field` (length `totalDimension()`), in the same flat block layout
    /// (`blockOffsets`). `metric = false` uses unit weights. The time component is
    /// the (positive) \f$ |\text{volume}| \f$-weighted modulus — the Dirac
    /// current's \f$ j^0 = \bar\Phi\gamma^0\Phi \f$ in any signature.
    /// @throws std::runtime_error if `field` has the wrong length.
    [[nodiscard]] std::vector<double> chargeDensity(
        const std::vector<std::complex<double>> &field, bool metric = true) const;

    /// The carried U(1) charge \f$ \sum_c j^0_c = \langle\Phi,\Phi\rangle_W \f$ —
    /// the charge density summed over the slice. On a closed harmonic this is the
    /// carried charge (the Gauss-law cross-check, deferred to the
    /// electric-charge-density ticket, compares it to the period-derived charge).
    /// @throws std::runtime_error if `field` has the wrong length.
    [[nodiscard]] double charge(
        const std::vector<std::complex<double>> &field, bool metric = true) const;

  private:
    std::shared_ptr<Spacetime> st_;

    // Per-degree cell counts |C_k| (length n+1) and the block offsets (length
    // n+2), built lazily from the ChainComplex on demand.
    [[nodiscard]] std::vector<std::size_t> cellCounts() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DIRACKAHLER_H
