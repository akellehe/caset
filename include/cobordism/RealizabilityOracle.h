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
#include <map>
#include <memory>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

class EigenstateSynthesis;
class Cochain;

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
/// ## `k=1` boundary harmonics on a 3-manifold-with-boundary (#176)
///
/// `decideHarmonic` lifts the same pipeline from the reduced \f$ k=0 \f$ 2-complex
/// to the DW-native \f$ k=1 \f$, 3-manifold setting. The target is a boundary
/// **harmonic 1-form** in \f$ \ker L_1(\partial W) \f$ (a degree-1 `Cochain`,
/// typically the readout of a `PreparedBoundaryState` — the bent \f$ U \f$
/// expressed in the prepared DW boundary basis via `BoundaryStateSpace`). The
/// bulk \f$ W \f$ is a 3-manifold-with-boundary (e.g. a solid torus, boundary
/// \f$ T^2 \f$); its boundary surface \f$ \partial W \f$ is pinned byte-fixed and
/// the interior is filled — interior edge squared-lengths (the metric content of
/// \f$ L_1 \f$ via the simplex volumes) plus boundary-fixed Pachner growth
/// (\f$ 1\!\to\!4 \f$ in 3D) — driving \f$ r = \lVert(I-\psi\psi^\dagger)L_1\psi
/// \rVert^2 \to 0 \f$ so the bulk carries the target as a (near-)harmonic whose
/// boundary restriction matches it. Realizable iff \f$ r<\epsilon \f$; otherwise
/// the floor certifies non-realizability — exactly as at \f$ k=0 \f$. The boundary
/// harmonic carried by the manifold (the restriction of \f$ \ker L_1(W) \f$, e.g.
/// the solid torus's longitude) is realizable; a class that bounds in the bulk
/// (the meridian) floors.
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
    /// How the interior fill grows when a pass cannot reach \f$ r<\epsilon \f$.
    enum class GrowthMode {
      /// The historical biased move: cone a fresh interior vertex into one top
      /// cell (`EigenstateSynthesis::growInterior`), wiring it to exactly the
      /// \f$ d+1 \f$ vertices of that cell.
      Cone,
      /// **Free interior connectivity** (#200): at each growth step search a
      /// bounded set of candidate interior connectivities for the new vertex
      /// (which existing vertices it wires to), score each by the residual it
      /// reaches after the weight/phase optimization, and keep the best — so the
      /// realized topology is what the residual selects, not a cone artifact.
      FreeConnectivity
    };

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
      /// Interior tunable edges in the realized witness — the **emergent
      /// connectivity** size. Cone growth adds exactly \f$ d+1 \f$ per vertex;
      /// free-connectivity growth can differ (this is the headline observable).
      std::size_t interiorEdgeCount{0};
      /// Candidate interior connectivities **scored per growth step**
      /// (`FreeConnectivity` only; 0 under `Cone`). The bounded breadth of the
      /// connectivity search.
      int connectivityCandidates{0};
      /// The full per-step incidence space the candidates are pruned from:
      /// \f$ 2^{N}-1 \f$ nonempty vertex subsets at the last growth step (`N` =
      /// vertices then). `connectivityCandidates` \f$ \ll \f$ this documents the
      /// cap — the search is bounded, not exhaustive, and logs what it skips.
      std::size_t connectivitySpaceSize{0};
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
    /// @param mode      `Cone` (default) keeps the historical cone-only growth;
    ///                  `FreeConnectivity` searches interior connectivity at each
    ///                  growth step (the new vertex's incidence is a free variable).
    /// @param connectivityCandidates  bounded number of candidate connectivities
    ///                  scored per growth step under `FreeConnectivity` (ignored
    ///                  under `Cone`). The candidate set is documented + logged.
    /// @throws std::invalid_argument if \f$ U \f$'s size \f$ \neq d_A d_B \f$, a
    ///   dimension is non-positive, or the bulk has fewer vertices than
    ///   \f$ d_A d_B \f$ (no room for the output-boundary support).
    [[nodiscard]] Verdict decide(const std::vector<std::complex<double>> &U,
                                 int dA, int dB, double epsilon = 1e-10,
                                 int restarts = 64, int maxCones = 4,
                                 std::uint64_t seed = 0,
                                 GrowthMode mode = GrowthMode::Cone,
                                 int connectivityCandidates = 8);

    /// Decide whether a target **boundary harmonic** \f$ k \f$-form (\f$ k =
    /// \texttt{target.degree()} \f$, the \f$ k=1 \f$ DW setting) is realizable on
    /// the held 3-manifold bulk \f$ W \f$: pin the boundary surface
    /// \f$ \partial W \f$ byte-fixed, fill the interior (interior edge
    /// squared-lengths + boundary-fixed Pachner growth) to drive the \f$ k \f$-form
    /// residual \f$ r=\lVert(I-\psi\psi^\dagger)L_k\psi\rVert^2\to 0 \f$, and return
    /// the `Verdict`. `target` is a degree-\f$ k \f$ `Cochain` over \f$ \partial W \f$'s
    /// \f$ k \f$-cells (the readout of a `PreparedBoundaryState`); it is matched to
    /// the bulk's boundary \f$ k \f$-cells by sorted vertex-id tuple, the remaining
    /// (interior) \f$ k \f$-cells carrying free auxiliary amplitudes. The witness
    /// `state` is the realized \f$ L_k(W) \f$ (near-)eigenvector whose boundary
    /// block matches `target`; realizable iff \f$ r<\epsilon \f$, else the floor
    /// certifies non-realizability at the explored complexity.
    /// @throws std::invalid_argument if `target` is empty, its degree is negative,
    ///   or none of its \f$ k \f$-cells are boundary cells of the bulk (the surface
    ///   does not match \f$ \partial W \f$).
    [[nodiscard]] Verdict decideHarmonic(const Cochain &target,
                                         double epsilon = 1e-10, int restarts = 64,
                                         int maxCones = 4, std::uint64_t seed = 0);

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

    // The §4b cone-and-retry restricted to the interior, degree-agnostic. The
    // boundary k-cells (the keys of `pinnedByTuple`, matched to es.cellSimplices()
    // by sorted vertex-id tuple) carry the fixed target amplitudes; the remaining
    // (interior) k-cells carry free auxiliary amplitudes. Drives r(psi) -> 0 by
    // varying the interior edge weights (and, at k=0 only, phases) via `es` plus
    // the auxiliary amplitudes; grows the interior (`mode`: cone, or the bounded
    // free-connectivity search) and retries while unconverged and within
    // `maxCones`, re-identifying the boundary/interior partition on the new
    // k-cell order after each grow. Leaves `es`'s complex realized at the best
    // parameters, writes the realized unit witness state to `witnessOut`, records
    // the cones applied and (free mode) the last step's candidate breadth into
    // `candidatesOut` / `spaceSizeOut`, and returns the best residual (the floor
    // if it never converges). Reuses LevenbergMarquardt exactly as
    // GeometrySynthesizer::runOptimizer does.
    [[nodiscard]] double fillInterior(
        EigenstateSynthesis &es,
        const std::map<std::vector<std::uint64_t>, std::complex<double>>
            &pinnedByTuple,
        double epsilon, int restarts, int maxCones, std::uint64_t seed,
        GrowthMode mode, int connectivityCandidates,
        std::vector<std::complex<double>> &witnessOut, int &conesApplied,
        int &candidatesOut, std::size_t &spaceSizeOut) const;

    // One fixed-complex optimization pass over the current `es`: the
    // multi-restart bounded Levenberg–Marquardt on r(psi) for the target encoded
    // by `pinnedByTuple`, seeded by `passSeed`. Leaves `es` realized at the best
    // parameters, writes the realized unit witness to `witnessOut`, and returns
    // the best residual. Extracted from the fill loop so the connectivity search
    // can score a candidate complex with exactly the same machinery.
    [[nodiscard]] double optimizePass(
        EigenstateSynthesis &es,
        const std::map<std::vector<std::uint64_t>, std::complex<double>>
            &pinnedByTuple,
        double epsilon, int restarts, std::uint64_t passSeed,
        std::vector<std::complex<double>> &witnessOut) const;

    // One free-connectivity growth step: generate a bounded set of candidate
    // interior connectivities for a fresh interior vertex (which existing
    // vertices it wires to), score each by `optimizePass` after attaching it,
    // detach, and commit the best. Reports the candidate breadth scored vs. the
    // full 2^N-1 incidence space (logged — no silent cap) into `candidatesOut` /
    // `spaceSizeOut`. Returns true if a vertex was committed (falls back to the
    // cone move if no candidate could attach).
    [[nodiscard]] bool growBestConnectivity(
        EigenstateSynthesis &es,
        const std::map<std::vector<std::uint64_t>, std::complex<double>>
            &pinnedByTuple,
        double epsilon, int restarts, std::uint64_t seed, int nCandidates,
        int &candidatesOut, std::size_t &spaceSizeOut) const;

    // The bounded, documented candidate-connectivity generator: deterministic
    // anchors (cone-equivalent — the d+1 vertices of a top cell; full-star — all
    // vertices; boundary-star — all boundary vertices) plus reproducible random
    // vertex subsets, deduplicated and capped at `nCandidates`. Each candidate is
    // the set of existing vertices the new interior vertex wires to by an edge.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> connectivityCandidates(
        const EigenstateSynthesis &es, int nCandidates, std::uint64_t seed) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REALIZABILITYORACLE_H
