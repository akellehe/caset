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

#ifndef TESSERA_COBORDISM_REALIZABILITYORACLE_H
#define TESSERA_COBORDISM_REALIZABILITYORACLE_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

class EigenstateSynthesis;

/// # RealizabilityOracle
///
/// The §5.0 realizability oracle: decide whether an operation
/// \f$ U:\mathcal{H}_B\to\mathcal{H}_A \f$ is realizable as a bulk cobordism
/// \f$ W_{AB} \f$ by **synthesizing the bulk spectrally** — not by TQFT
/// membership. \f$ U \f$ is bent to a boundary state via Choi–Jamiołkowski
/// (\f$ \mathrm{vec}(U) \f$, the operator-as-state); the boundaries of
/// \f$ W_{AB} \f$ are *pinned* (the synthesized \f$ \mathrm{geo} \f$'s / the
/// output surface) and the **interior** is filled — its Hermitian edge weights
/// and (via boundary-fixed Pachner growth) its topology — so the output-boundary
/// graph-Laplacian eigenvector matches the bent target, i.e. the §4b residual
/// \f$ r = \lVert (I-\psi\psi^\dagger)L\psi \rVert^2 \to 0 \f$.
///
/// **Realizability is itself the test.** A target is **realizable** iff the
/// residual can be driven below \f$ \epsilon \f$, and **non-realizable** iff it
/// *floors* away from zero under the pinned boundary — a spectral obstruction,
/// the analogue of §4b's two-vertex floor \f$ w_{\min}^2(|c_0|^2-|c_1|^2)^2 \f$.
/// The floor **is** the certificate: non-existence is certified by the residual
/// floor, **not** by exhausting triangulations.
///
/// ## Pure orchestration (no new math)
///
/// This class composes the merged building blocks; it adds no numerics of its
/// own:
///   * `quantum::ChoiJamiolkowski` — bends \f$ U \f$ to \f$ \mathrm{vec}(U) \f$.
///   * `cobordism::EigenstateSynthesis` — the fixed-boundary interior-fill engine
///     (\f$ \partial W \f$ pinned; `setInteriorWeights` / `setInteriorPhases`
///     vary only the interior; `growInterior` is the boundary-fixed Pachner add;
///     `residual` / `rayleigh` / `apply` score the target).
///   * `cobordism::LevenbergMarquardt` — the same bounded multi-restart
///     least-squares solver the §4b `GeometrySynthesizer` drives, here over the
///     **interior** parameters and the free auxiliary amplitudes.
///
/// The pipeline (one public entry point, a few private steps):
///   1. **Bend** — `vec(U)` → the normalized target \f$ \psi_U \f$ (length
///      \f$ d_A d_B \f$), the operator rendered as a boundary state.
///   2. **Pin** — the bulk's boundary edges \f$ \partial W \f$ (the synthesized
///      boundaries / output surface) are held byte-fixed throughout; the
///      fixed-boundary engine only ever writes interior edges.
///   3. **Fill** — the §4b cone-and-retry restricted to the interior: a
///      multi-restart `LevenbergMarquardt` over the interior weights/phases and
///      the auxiliary amplitudes drives \f$ r\to 0 \f$; if a pass cannot
///      converge, `growInterior` cones a fresh interior vertex (boundary-fixed)
///      and the pass retries, up to the cone budget.
///   4. **Verdict** — realizable iff \f$ r<\epsilon \f$ (with the witness state
///      and the bulk \f$ W_{AB} \f$); otherwise non-realizable, certified by the
///      residual floor reached at the explored interior complexity.
///
/// ## Boundary support convention
///
/// \f$ \psi_U \f$ is carried by the **first** \f$ d_A d_B \f$ vertices in
/// sorted-id order (the output-surface support, the
/// `GeometrySynthesizer`/`EigenstateSynthesis` embedding idiom); any remaining
/// vertices — interior vertices and the apices `growInterior` cones in (larger
/// ids, appended last) — carry **free auxiliary amplitudes** the fill solves
/// for. The caller assembles the bulk so its output surface occupies those ids
/// and pins its boundary edges (`Spacetime` is built and pinned outside the
/// synthesis classes, the established idiom). `decide` mutates the held bulk in
/// place (the accepted interior weights/phases, plus any coned-in growth), and
/// the returned `witness` is that realized \f$ W_{AB} \f$.
class RealizabilityOracle {
  public:
    /// The oracle's verdict on \f$ U \f$: the realizability decision, the
    /// residual (the obstruction floor when non-realizable), the witness state
    /// and bulk, and the interior complexity reached.
    struct Verdict {
      /// True iff the interior fill drove \f$ r<\epsilon \f$ within the cone
      /// budget — \f$ U \f$ is realizable, witnessed by `state` on `witness`.
      bool realizable{false};
      /// The best residual \f$ r=\lVert(I-\psi\psi^\dagger)L\psi\rVert^2 \f$
      /// reached. \f$ <\epsilon \f$ when realizable; the certified obstruction
      /// **floor** otherwise.
      double residual{0.0};
      /// The obstruction floor: `residual` when non-realizable, `0` when
      /// realizable. The spectral certificate that no bulk of the explored
      /// interior complexity realizes \f$ U \f$ under the pinned boundary.
      double floor{0.0};
      /// The realized eigenvalue \f$ \lambda=\psi^\dagger L\psi \f$ (Rayleigh
      /// quotient) of the witness state — meaningful when realizable.
      double eigenvalue{0.0};
      /// Interior vertices coned in to reach the verdict (the interior
      /// complexity, the §4b \f$ (|V|,|E|) \f$ analogue under fixed ends).
      std::size_t interiorVertexCount{0};
      /// Boundary-fixed cones applied during the fill (== `interiorVertexCount`
      /// for a single-top-cell seed; reported for the cone-and-retry trace).
      int conesApplied{0};
      /// The witness state: the realized full Laplacian eigenvector on
      /// \f$ W_{AB} \f$ (unit norm, length = the bulk's vertex count), whose
      /// first \f$ d_A d_B \f$ components are the output-boundary block matching
      /// `target`. A genuine eigenstate iff `realizable`.
      std::vector<std::complex<double>> state{};
      /// The bent target \f$ \psi_U=\mathrm{vec}(U)/\lVert\cdot\rVert \f$
      /// (length \f$ d_A d_B \f$) the output boundary is matched against.
      std::vector<std::complex<double>> target{};
      /// The realized bulk \f$ W_{AB} \f$ — the witness cobordism (realizable),
      /// or the complex on which the residual floors (non-realizable).
      std::shared_ptr<Spacetime> witness{};
    };

    /// Construct over the assembled bulk \f$ W_{AB} \f$: a pre-geometric
    /// `Spacetime` whose codimension-one boundary is the (pinned) output surface
    /// plus input boundaries, with its output-surface vertices occupying the
    /// smallest ids (see the boundary-support convention). The held `shared_ptr`
    /// keeps the bulk alive; `decide` realizes it in place.
    /// @throws std::invalid_argument if `bulk` is null.
    explicit RealizabilityOracle(std::shared_ptr<Spacetime> bulk);

    /// Decide whether the \f$ d_A\times d_B \f$ operator \f$ U \f$ (flat
    /// row-major, \f$ U_{ij}=U[i\,d_B+j] \f$) is realizable as a bulk cobordism:
    /// bend it to \f$ \psi_U=\mathrm{vec}(U) \f$, fill the pinned-boundary
    /// interior to drive the §4b residual to zero, and return the `Verdict`.
    /// @param epsilon  acceptance threshold on the residual (realizable iff
    ///                 \f$ r<\epsilon \f$).
    /// @param restarts random restarts per interior-fill pass (non-convex).
    /// @param maxCones boundary-fixed interior cones to try (0 ⇒ decide at the
    ///                 seed interior complexity — the bare obstruction floor).
    /// @param seed     RNG seed for the restart draws (reproducible).
    /// @throws std::invalid_argument if \f$ U \f$'s size \f$ \neq d_A d_B \f$, a
    ///   dimension is non-positive, or the bulk has fewer vertices than
    ///   \f$ d_A d_B \f$ (no room for the output-boundary support).
    [[nodiscard]] Verdict decide(const std::vector<std::complex<double>> &U,
                                 int dA, int dB, double epsilon = 1e-10,
                                 int restarts = 64, int maxCones = 4,
                                 std::uint64_t seed = 0);

    /// Decide whether a target **\f$ k=1 \f$ boundary harmonic** is realizable on
    /// the bulk \f$ W \f$ (the DW bridge lift, #176): the spectral boundary qubit
    /// is the harmonic 1-forms \f$ \ker L_1(\Sigma) \f$ of \f$ \Sigma=\partial W \f$,
    /// and the question is whether `target` extends to a harmonic form of the
    /// **bulk** \f$ \ker L_1(W) \f$ with \f$ \partial W \f$ pinned. `target` is the
    /// 1-form on \f$ \partial W \f$'s edges — length the bulk's boundary-edge count,
    /// ordered as `EigenstateSynthesis(W, 1).boundaryStateIndices()` (the boundary
    /// edges in the canonical column order). Pins \f$ \partial W \f$, fills the
    /// interior metric (interior edge weights) + free interior amplitudes to drive
    /// the harmonic residual \f$ r=\lVert L_1\psi\rVert^2\to 0 \f$, and returns the
    /// `Verdict` (realizable iff \f$ r<\epsilon \f$; otherwise the obstruction
    /// floor). `state` is the realized bulk 1-form (length \f$ |C_1(W)| \f$);
    /// `eigenvalue` is its Rayleigh quotient (\f$ \approx 0 \f$ when harmonic).
    /// @throws std::invalid_argument if `target`'s length \f$ \neq \f$ the bulk's
    ///   boundary-edge count.
    [[nodiscard]] Verdict decideBoundaryHarmonic(
        const std::vector<std::complex<double>> &target, double epsilon = 1e-10,
        int restarts = 64, int maxCones = 4, std::uint64_t seed = 0);

  private:
    // §4b search box — identical to GeometrySynthesizer so the interior fill is
    // the same machinery applied to the interior: per-edge magnitudes in
    // [kWMin, kWMax], U(1) phases in [-kThetaBound, kThetaBound]. kAuxBound
    // clamps the free auxiliary amplitudes on interior / coned-in vertices.
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kWMin = 0.1;
    static constexpr double kWMax = 10.0;
    static constexpr double kThetaBound = 2.0 * kPi;
    static constexpr double kAuxBound = 5.0;
    static constexpr int kMaxIterations = 200;  // LM iterations per descent

    std::shared_ptr<Spacetime> bulk_;

    // Bend U to its normalized boundary state psi_U = vec(U)/||vec(U)||
    // (ChoiJamiolkowski::vectorize; the residual is scale-invariant, but a unit
    // target gives a unit witness and a clean output-boundary match). Length
    // dA*dB.
    [[nodiscard]] static std::vector<std::complex<double>> bend(
        const std::vector<std::complex<double>> &U, int dA, int dB);

    // The §4b cone-and-retry restricted to the interior: drive r(psi) -> 0 for
    // the target on its first `target.size()` (boundary-support) components with
    // the remaining (interior/auxiliary) components free, varying only the
    // interior edge weights/phases via `es`; grow the interior (boundary-fixed
    // Pachner add) and retry while unconverged and within `maxCones`. Leaves
    // `es`'s complex realized at the best parameters, writes the realized unit
    // witness state to `witnessOut`, records the cones applied, and returns the
    // best residual (the floor if it never converges). Reuses LevenbergMarquardt
    // exactly as GeometrySynthesizer::runOptimizer does.
    [[nodiscard]] double fillInterior(
        EigenstateSynthesis &es,
        const std::vector<std::complex<double>> &target, double epsilon,
        int restarts, int maxCones, std::uint64_t seed,
        std::vector<std::complex<double>> &witnessOut, int &conesApplied) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REALIZABILITYORACLE_H
