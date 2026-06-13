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

#ifndef TESSERA_COBORDISM_REGISTER_H
#define TESSERA_COBORDISM_REGISTER_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "cobordism/EigenstateSynthesis.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # Register
///
/// The carried spectral register \f$ V = \ker L_1 \f$ of a surgery-grown
/// triangulated surface — the **continuous spectral** object the staged
/// spectral-gate realizability test scores against (the
/// `spectral_gate_realizability` example, the #249 scoring core). It is an
/// aggregator over the existing spectrum tooling rather than a re-derivation: it
/// holds one `EigenstateSynthesis` at \f$ k = 1 \f$ with the holonomy holes
/// opened, and every quantity reads off that complex —
/// - `harmonicMatrix()` \f$ H \f$ from `HodgeLaplacian::harmonicMatrix`,
/// - the period matrix \f$ P \f$ from `EigenstateSynthesis::cyclePeriods`,
/// - the boundary-period constraint \f$ n \f$ = `sign` from
///   `ChainComplex::endSignCovector` (the deterministic end-surface covector, not
///   a per-fill SVD null vector — which is sign-unstable),
/// - the carried representative and its residual from
///   `EigenstateSynthesis::carriedRepresentative` / `residualForPeriods`.
///
/// ## Construction
///
/// The three non-trivial \f$ \mathbb{Z}_2 \f$ holonomy classes
/// \f$ \{[a],[b],[a+b]\} \f$ live as three vertex-disjoint boundary 1-cycles on a
/// closed triangulated \f$ S^2 \f$ (the icosahedron, in the canonical register).
/// Each is a **class hole**: a triangular interior top cell. Construction opens
/// every class hole by the boundary-fixed topology-changing surgery
/// (`removeInteriorCell`), so \f$ b_1 \f$ grows \f$ 0 \to 2 \f$ and
/// \f$ \ker L_1 \f$ emerges as the \f$ S_3 \f$ standard representation (the
/// 2-dimensional Hodge register \f$ V \f$). Then, optionally, `growVertices`
/// interior vertices are added by the boundary-fixed composed stellar move
/// (`stellarSubdivideInterior`, seeded by `growSeed`) — additive growth that
/// preserves \f$ \ker L_1 \f$; the number actually added is `grown()`. Finally any
/// `extraHoles` still removable as interior top cells are opened (the \f$ b_1 \f$
/// -growth surgery the topology search drives); the subset opened is
/// `openedExtraHoles()`. (Growth/extra holes drive the example's `--retries`
/// surgery-topology search; the default register passes neither.)
///
/// ## Scoring
///
/// `harmonicForm(rawPeriods)` is the genuine carried harmonic 1-form whose
/// hole-periods are the projection of `rawPeriods` onto the carried period space,
/// plus the minimal leak so the returned cochain's periods are exactly
/// `rawPeriods` (full edge vector, cell order). `spectralResidual(rawPeriods)` is
/// that form's genuine Hodge residual on the grown bulk — \f$ \to 0 \f$ iff the
/// periods lie in the carried register \f$ V \f$. `rank()` is the rank of
/// \f$ P \f$: \f$ \text{rank} < \#\text{holes} \f$ is a genuine register (a proper
/// carried subspace), equality the saturated case.
class Register {
  public:
    /// Build the register over `st` (a triangulated surface): open each class
    /// hole (a sorted vertex-id triangle) by surgery, add `growVertices` interior
    /// vertices by the seeded stellar move, then open any `extraHoles` still
    /// removable as interior top cells. The held `shared_ptr` keeps the spacetime
    /// alive; the surgery/growth mutate `st` in place, so `spacetime()` reflects
    /// the grown bulk.
    /// @throws std::runtime_error if a class hole is not a removable interior top
    ///   cell, or the grown surface is not a closed-orientable end surface (see
    ///   `ChainComplex::endSignCovector`).
    Register(std::shared_ptr<Spacetime> st,
             const std::vector<std::vector<std::uint64_t>> &classHoles,
             const std::vector<std::vector<std::uint64_t>> &extraHoles = {},
             int growVertices = 0, std::uint64_t growSeed = 0);

    /// The surgery-grown bulk (the same spacetime passed in, holes opened) — for
    /// reading vertex/edge lists, Betti numbers, etc.
    [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const noexcept {
      return st_;
    }

    /// The operator dimension \f$ N = |C_1| \f$ — the length of a `harmonicForm`
    /// cell vector (delegates to the underlying \f$ k = 1 \f$
    /// `EigenstateSynthesis`).
    [[nodiscard]] std::size_t order() const noexcept { return synth_.order(); }

    /// The carried-register dimension \f$ \dim V = \dim\ker L_1 = b_1 \f$.
    [[nodiscard]] int dimension() const noexcept { return dim_; }

    /// The rank of the period matrix \f$ P \f$ over the holonomy holes.
    /// \f$ \text{rank} < \#\text{holes} \f$ is a genuine register; equality the
    /// saturated/degenerate case.
    [[nodiscard]] int rank() const noexcept { return rank_; }

    /// The number of interior vertices actually added by the stellar growth move.
    [[nodiscard]] int grown() const noexcept { return grown_; }

    /// The 1-cells of the grown bulk as sorted vertex-id tuples, in operator order
    /// (length `order()`) — the indexing of any `harmonicForm` cell vector and of
    /// the columns of `harmonicMatrix()`.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cells()
        const noexcept {
      return cells_;
    }

    /// The class holes (the holonomy-class triangles) as sorted vertex-id tuples.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &classHoles()
        const noexcept {
      return classHoles_;
    }

    /// The subset of the requested `extraHoles` actually opened (those still
    /// removable as interior top cells), as sorted vertex-id tuples.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &openedExtraHoles()
        const noexcept {
      return extraOpened_;
    }

    /// The period matrix \f$ P \f$ (\f$ \dim \times \#\text{holes} \f$): each
    /// harmonic's oriented loop sum (period) around each class hole
    /// (`cyclePeriods`, reshaped).
    [[nodiscard]] const Eigen::MatrixXcd &periodMatrix() const noexcept {
      return periods_;
    }

    /// The \f$ \dim \f$ harmonic 1-forms of \f$ L_1 \f$ as a
    /// \f$ \dim \times |C_1| \f$ matrix, each row a full edge vector in the bulk's
    /// cell order (`cells()`) — `HodgeLaplacian::harmonicMatrix(1)` reshaped.
    [[nodiscard]] const Eigen::MatrixXcd &harmonicMatrix() const noexcept {
      return hFull_;
    }

    /// The induced-orientation boundary-period constraint \f$ n \f$ — the
    /// `ChainComplex::endSignCovector` of the grown surface over the class holes
    /// (a \f$ \pm 1 \f$ covector, length \f$ \#\text{holes} \f$): the deterministic
    /// end-surface covector that annihilates the period rows (\f$ P n = 0 \f$).
    [[nodiscard]] const Eigen::VectorXd &constraint() const noexcept { return n_; }

    /// The induced-orientation signs — identical to `constraint()` (the
    /// \f$ \pm 1 \f$ end-sign covector) — that symmetrize the boundary-period
    /// constraint to \f$ \Sigma = 0 \f$.
    [[nodiscard]] const Eigen::VectorXd &sign() const noexcept { return sign_; }

    /// The genuine carried harmonic 1-form whose hole-periods are the projection
    /// of `rawPeriods` onto the carried period space, plus the minimal leak so the
    /// returned cochain's periods are exactly `rawPeriods` (full edge vector,
    /// length `order()`). Delegates to
    /// `EigenstateSynthesis::carriedRepresentative` over the class holes.
    /// @throws std::runtime_error if `rawPeriods.size() != classHoles().size()`.
    [[nodiscard]] std::vector<std::complex<double>> harmonicForm(
        const std::vector<std::complex<double>> &rawPeriods) const {
      return synth_.carriedRepresentative(classHoles_, rawPeriods);
    }

    /// The genuine Hodge residual \f$ \|(I-\psi\psi^\dagger)L_1\psi\|^2 \f$ of the
    /// 1-form with the given hole-periods, on the surgery-grown bulk — the
    /// continuous spectral realizability score. \f$ \to 0 \f$ iff the periods lie
    /// in the carried register \f$ V \f$. Delegates to
    /// `EigenstateSynthesis::residualForPeriods` over the class holes.
    /// @throws std::runtime_error if `rawPeriods.size() != classHoles().size()`.
    [[nodiscard]] double spectralResidual(
        const std::vector<std::complex<double>> &rawPeriods) const {
      return synth_.residualForPeriods(classHoles_, rawPeriods);
    }

  private:
    std::shared_ptr<Spacetime> st_;
    // The k=1 residual core over the grown bulk (holes opened in the ctor): the
    // genuine metric Hodge L_1, and the source of the period/residual read-outs.
    EigenstateSynthesis synth_;

    std::vector<std::vector<std::uint64_t>> classHoles_{};
    std::vector<std::vector<std::uint64_t>> extraOpened_{};
    // The 1-cells of the grown bulk (= synth_.cellSimplices()).
    std::vector<std::vector<std::uint64_t>> cells_{};

    int dim_{0};    // dim ker L_1 = number of harmonic 1-forms
    int rank_{0};   // rank of the period matrix P
    int grown_{0};  // interior vertices added by the stellar growth move

    Eigen::MatrixXcd hFull_{};    // dim x |C_1| harmonic 1-forms (rows)
    Eigen::MatrixXcd periods_{};  // dim x #holes period matrix P
    Eigen::VectorXd n_{};         // the end-sign covector over the holes (+-1)
    Eigen::VectorXd sign_{};      // == n_ (the induced-orientation signs)
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REGISTER_H
