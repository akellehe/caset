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

#ifndef TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H
#define TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "cobordism/HodgeLaplacian.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh { class Edge; }
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # EigenstateSynthesis
///
/// The §4b inverse eigenvector problem on a **fixed** complex: given a target
/// state \f$ \psi \f$, score how close the complex's current Hermitian edge
/// weights make \f$ \psi \f$ to being an eigenvector of the \f$ k=0 \f$ Hodge
/// Laplacian \f$ L = D - A \f$ (the magnitude convention, via
/// `HodgeLaplacian`), and read/write those weights so a search can perturb them.
///
/// The search itself (non-convex, multi-restart, e.g. `scipy.optimize.minimize`
/// L-BFGS-B over the flat \f$ \{w_{ij}\}\cup\{\theta_{ij}\} \f$ vector) drives
/// the cone-and-retry growth loop in a separate stage; this class is the
/// **residual + parameter access** core it calls. Growing the complex
/// (coning-in auxiliary vertices) is out of scope here.
///
/// ## Residual
///
/// For a unit target \f$ \psi \f$ the eigenvalue-agnostic residual is the
/// squared norm of the component of \f$ L\psi \f$ orthogonal to \f$ \psi \f$,
/// \f$ r(\psi) = \big\|\,(I-\psi\psi^\dagger)\,L\,\psi\,\big\|^2
///            = \|L\psi - \lambda\psi\|^2,\quad \lambda = \psi^\dagger L\psi, \f$
/// so \f$ r = 0 \iff L\psi \parallel \psi \iff \psi \f$ is an eigenvector, and
/// the realized eigenvalue is the Rayleigh quotient \f$ \lambda \f$. A non-unit
/// \f$ \psi \f$ is normalized internally (the eigenvector condition is
/// scale-invariant). The Laplacian is reassembled from the **current** edge
/// weights/phases on every call, so the residual tracks in-place perturbations
/// of `setWeights` / `setPhases`.
///
/// ## Parameters
///
/// The tunable parameters are the per-edge squared-length magnitudes
/// \f$ \{w_{ij}\} \f$ (`Edge::setSquaredLength`) and U(1) connection phases
/// \f$ \{\theta_{ij}\} \f$ (`Edge::setPhase`), in a stable edge order fixed at
/// construction (the `EdgeList` order, restricted to the edges that carry weight
/// in \f$ L \f$: both endpoints present, no self-loops). `weights()` / `phases()`
/// read them; `setWeights()` / `setPhases()` write them back in place — no mesh
/// rebuild. `psi` components are indexed in the same sorted-vertex-id order as
/// `HodgeLaplacian` (\f$ k=0 \f$), so they align with the operator.
class EigenstateSynthesis {
  public:
    /// Construct over a fixed triangulation. The vertex order (sorted id) and the
    /// tunable edge order are captured now; edge weights/phases are read live on
    /// each residual query. The held `shared_ptr` keeps the spacetime alive.
    explicit EigenstateSynthesis(std::shared_ptr<Spacetime> st);

    /// Number of vertices \f$ N \f$ — the required length of any `psi`.
    [[nodiscard]] std::size_t order() const noexcept { return order_; }

    /// Number of tunable edges — the length of `weights()` / `phases()`.
    [[nodiscard]] std::size_t numEdges() const noexcept { return edges_.size(); }

    /// The eigenvalue-agnostic residual \f$ r(\psi) = \|(I-\psi\psi^\dagger)L\psi\|^2 \f$
    /// against the current edge weights/phases. \f$ \psi \f$ is normalized
    /// internally; \f$ r = 0 \iff L\psi \parallel \psi \f$.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] double residual(const std::vector<std::complex<double>> &psi) const;

    /// The Rayleigh quotient \f$ \lambda = \psi^\dagger L\psi / \psi^\dagger\psi \f$
    /// (real; \f$ L \f$ Hermitian) — the realized eigenvalue when \f$ r = 0 \f$.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] double rayleigh(const std::vector<std::complex<double>> &psi) const;

    /// \f$ L\psi \f$ against the current edge weights/phases (no normalization),
    /// for direct \f$ L\psi \parallel \psi \f$ cross-checks. Length `order()`.
    /// @throws std::runtime_error if `psi.size() != order()`.
    [[nodiscard]] std::vector<std::complex<double>> apply(
        const std::vector<std::complex<double>> &psi) const;

    /// The edge magnitudes \f$ \{w_{ij}\} \f$ (`Edge::getSquaredLength`) in the
    /// stable edge order, length `numEdges()`.
    [[nodiscard]] std::vector<double> weights() const;

    /// The edge phases \f$ \{\theta_{ij}\} \f$ (`Edge::getPhase`) in the stable
    /// edge order, length `numEdges()`.
    [[nodiscard]] std::vector<double> phases() const;

    /// Write the edge magnitudes in place (`Edge::setSquaredLength`).
    /// @throws std::runtime_error if `w.size() != numEdges()`.
    void setWeights(const std::vector<double> &w);

    /// Write the edge phases in place (`Edge::setPhase`).
    /// @throws std::runtime_error if `theta.size() != numEdges()`.
    void setPhases(const std::vector<double> &theta);

  private:
    std::shared_ptr<Spacetime> st_;
    // The k=0 Hermitian Laplacian operator over the same complex. laplacian(0)
    // reassembles L = D - A from the live edges on each call, so perturbing the
    // edges and re-querying is honest (the eigendecomposition cache is untouched
    // by the matrix path).
    HodgeLaplacian laplacian_;
    std::size_t order_{0};  // N = |V|, the sorted-id vertex order
    // The tunable edges, in EdgeList order, restricted to those carrying weight
    // in L (both endpoints present, no self-loops). Raw pointers owned by the
    // EdgeList; valid for the fixed complex's lifetime (kept alive via st_).
    std::vector<::tessera::mesh::Edge *> edges_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_EIGENSTATESYNTHESIS_H
