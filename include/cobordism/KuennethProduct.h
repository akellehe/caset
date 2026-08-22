// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_KUENNETHPRODUCT_H
#define TESSERA_COBORDISM_KUENNETHPRODUCT_H

#include <complex>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "cobordism/Certificate.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # KuennethProduct
///
/// The exact Kronecker-sum/Künneth rule for actual product complexes (#764):
///
/// \f[ L_{A\times B} \;=\; L_A \otimes I \;+\; I \otimes L_B . \f]
///
/// **Identity and domain.** As a matrix identity the Kronecker sum is
/// algebraically exact for ANY two square operators, and because
/// \f$ L_A\otimes I \f$ and \f$ I\otimes L_B \f$ commute, its spectrum is
/// exactly the pairwise sums \f$ \{\lambda_i+\mu_j\} \f$ — no product
/// eigensolve is ever needed. As a statement about a COMPLEX it holds only
/// for an actual product cell structure with product weights: at degree
/// zero, a complex whose weighted 1-skeleton is the Cartesian product of the
/// factors' 1-skeletons (product vertices \f$ (u,v) \f$; edges
/// \f$ (u,v)\!-\!(u',v) \f$ with \f$ A \f$'s weight and
/// \f$ (u,v)\!-\!(u,v') \f$ with \f$ B \f$'s weight). A staircase-subdivided
/// `SimplicialProduct` is NOT in this domain — its diagonal edges break the
/// identity — and `productCertificate` refuses it (`holds() == false`)
/// rather than approximating. Weights stay complex/signed throughout; the
/// rule never assumes positive-definite.
///
/// This class is spectrum/matrix level only. Fock-operator structure
/// (creation/annihilation, wedge, \f$ d\Gamma \f$ as an operator) lives in
/// the exterior-algebra track; the free many-body spectra derived from these
/// one-particle rules are `OccupationSpectra`'s job.
class KuennethProduct {
  public:
    /// The Kronecker sum \f$ L_A\otimes I_{n_B} + I_{n_A}\otimes L_B \f$ as a
    /// flat row-major \f$ (n_An_B)\times(n_An_B) \f$ matrix. Product index
    /// \f$ (i_A, i_B) \mapsto i_A\,n_B + i_B \f$. Algebraically exact
    /// assembly (additions only).
    /// @throws std::invalid_argument on dimension mismatch.
    [[nodiscard]] static std::vector<std::complex<double>> kroneckerSum(
        const std::vector<std::complex<double>> &laplacianA, int dimA,
        const std::vector<std::complex<double>> &laplacianB, int dimB);

    /// The exact spectrum of the Kronecker sum from the factor spectra: all
    /// pairwise sums \f$ \lambda_i + \mu_j \f$, sorted ascending by
    /// \f$ (\mathrm{Re}, \mathrm{Im}) \f$ (the `Spectrum` convention).
    /// Output-sensitive \f$ O(n_An_B\log(n_An_B)) \f$ — never diagonalizes
    /// the product operator.
    [[nodiscard]] static std::vector<std::complex<double>> pairwiseSpectrum(
        const std::vector<std::complex<double>> &spectrumA,
        const std::vector<std::complex<double>> &spectrumB);

    /// Certify that `product` IS an actual product complex of the two
    /// factors at degree zero: its \f$ k=0 \f$ weighted graph Laplacian
    /// (`HodgeLaplacian`) equals the Kronecker sum of the factors' under the
    /// declared vertex `pairing`, entrywise, to relative `tolerance`.
    ///
    /// `pairing` lists (product vertex id, factor-A vertex id, factor-B
    /// vertex id) — the product structure is DATA carried by the caller, not
    /// discovered. The check matches vertices by identifier set (any input
    /// order, any relabeling of product ids), never by an imposed sort.
    ///
    /// The returned certificate carries the measured relative residual
    /// \f$ \max_{ij}|L_{\text{prod}} - (L_A\otimes I + I\otimes L_B)|_{ij}
    /// / \max_{ij}|L_{\text{prod}}|_{ij} \f$; `holds()` grants the
    /// Künneth/Kronecker rule for this complex, and a failed check (e.g. a
    /// staircase triangulation) reports `holds() == false`.
    ///
    /// @throws std::invalid_argument when the pairing is malformed: wrong
    ///   size, duplicate or unknown identifiers, or a missing factor pair.
    [[nodiscard]] static Certificate productCertificate(
        const std::shared_ptr<Spacetime> &product,
        const std::shared_ptr<Spacetime> &factorA,
        const std::shared_ptr<Spacetime> &factorB,
        const std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>>
            &pairing,
        double tolerance = 1e-12);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_KUENNETHPRODUCT_H
