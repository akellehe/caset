// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

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
      FreeConnectivity,
      /// **Surgery** (#196): the topology-CHANGING move-set. Alongside the
      /// additive attach, at each step score every interior-top-cell **removal**
      /// (`EigenstateSynthesis::removeInteriorCell`, which opens a hole/handle
      /// with \f$ \partial W \f$ held bit-exact) by the residual it reaches after
      /// the weight optimization, and commit the best **improving** move. Unlike
      /// `FreeConnectivity` (additive only — spectrally inert at \f$ k\geq 1 \f$,
      /// so \f$ b_k \f$ is frozen at the seed), removal lets the search reach an
      /// arbitrary valid complex with the fixed boundary: \f$ b_k \f$ **moves on
      /// its own**, so the realizability obstruction either falls out of the grown
      /// topology or dissolves. The intended companion of the `harmonic` criterion
      /// (a boundary class is realizable iff it is **carried** by \f$ H_k(W) \f$).
      Surgery,
      /// **Surgery + cone**: the composed move-set — additions as well as
      /// surgical cuts. Each growth step first scores every interior-top-cell
      /// removal (try → score → restore, exactly the `Surgery` step) and commits
      /// the best **improving** cut; when no cut improves, it falls back to the
      /// additive cone (`EigenstateSynthesis::growInterior`). Under this mode
      /// `maxCones` budgets the **additive commits only** (the added vertices —
      /// the resource a caller's `--max-additional-vertices` flag caps); cuts are
      /// bounded by the improving-only rule and the finite interior-cell set, so
      /// the fill terminates at convergence, an exhausted additive budget with no
      /// improving cut left, or a complex that cannot grow.
      SurgeryAndCone
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
      /// Candidate interior **edge** connectivities (singleton specs) scored per
      /// growth step (`FreeConnectivity` only; 0 under `Cone`). The bounded
      /// breadth of the 1-skeleton connectivity search — the only spectrally
      /// relevant atom at \f$ k=0 \f$ (`L_0 = D-A` sees just the 1-skeleton).
      int connectivityCandidates{0};
      /// Candidate interior **2-simplex (triangle)** connectivities scored per
      /// growth step (`FreeConnectivity` at \f$ k\geq 1 \f$ only; 0 at \f$ k=0 \f$
      /// and under `Cone`). At \f$ k\geq 1 \f$ the metric \f$ L_k \f$ depends on
      /// the 2-cells through \f$ \partial_2 \f$, so the search must also propose
      /// triangle attachments — these are **not** spectrally inert at \f$ k=1 \f$
      /// the way they are at \f$ k=0 \f$. Reported alongside `connectivityCandidates`
      /// so the full scored breadth (edges + triangles) is surfaced, never capped
      /// silently.
      int triangleCandidates{0};
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
      /// Interior top-cell **removals** committed by the surgery search
      /// (`Surgery` mode only; 0 otherwise). Each is a topology-changing move that
      /// can shift \f$ b_k \f$ of the witness — the emergent-topology trace.
      int surgeryRemovals{0};
      /// The realized geometry's **dual Lorentzian Regge action magnitude**
      /// \f$ |S_{\mathrm{Regge}}(W^{*})| \f$ — the modulus of
      /// `ReggeSolver::dualReggeAction` on the witness's circumcentric dual. The
      /// gravitational cost the mediated objective \f$ F_\beta=r_U+\beta\,|S| \f$
      /// trades against the residual; reported at every \f$ \beta \f$ (including
      /// the \f$ \beta=0 \f$ base layer, so the \f$ \beta \f$-sweep can compare the
      /// chosen fillings' action). \f$ 0 \f$ if the witness has no hinges. Not
      /// finite (and logged) if the realized geometry is degenerate.
      double reggeAction{0.0};
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
    /// @param harmonic  when `true`, score the **harmonic** residual
    ///                  \f$ \lVert L\psi\rVert^2 \f$ (distance from \f$ \ker L \f$)
    ///                  instead of the eigenvalue-agnostic
    ///                  \f$ \lVert(I-\psi\psi^\dagger)L\psi\rVert^2 \f$: realizable
    ///                  then means \f$ \psi \f$ is **carried as a harmonic**
    ///                  (eigenvalue \f$ \to 0 \f$), i.e. the boundary class lies in
    ///                  \f$ \mathrm{image}(H_k(\partial W)\to H_k(W)) \f$ — the
    ///                  physical realizability test that distinguishes topologies.
    /// @param beta      coupling of the **mediated objective**
    ///                  \f$ F_\beta=r_U+\beta\,|S_{\mathrm{Regge}}(W^{*})| \f$:
    ///                  candidate moves are scored by \f$ F_\beta \f$ (the
    ///                  realizability residual plus \f$ \beta \f$ times the dual
    ///                  Regge action **magnitude** on the candidate's dual), so
    ///                  the search prefers gravitationally cheaper fillings.
    ///                  \f$ \beta=0 \f$ (default) reproduces the base-layer search
    ///                  **bit-for-bit** (the \f$ |S| \f$ term is not even computed).
    ///                  Only the cone+surgery move-sets (`Surgery`,
    ///                  `SurgeryAndCone`) act on it; realizability is still the
    ///                  primal \f$ r_U<\epsilon \f$, so a large \f$ \beta \f$ can
    ///                  starve a gate of an improving move and drop it from the set.
    /// @param maxVertices the volume bound \f$ \abs{W} \f$: additive growth stops
    ///                  once the bulk reaches this many vertices (the conformal-mode
    ///                  regularizer). Surgery (removal) is unaffected.
    [[nodiscard]] Verdict decide(const std::vector<std::complex<double>> &U,
                                 int dA, int dB, double epsilon = 1e-10,
                                 int restarts = 64, int maxCones = 4,
                                 std::uint64_t seed = 0,
                                 GrowthMode mode = GrowthMode::Cone,
                                 int connectivityCandidates = 8,
                                 bool harmonic = false, double beta = 0.0,
                                 int maxVertices = 16);

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
    /// @param mode      `Cone` (default) keeps the historical cone-only growth
    ///                  (the boundary-fixed \f$ 1\to(d+1) \f$ Pachner add);
    ///                  `FreeConnectivity` searches interior connectivity at each
    ///                  growth step. At \f$ k\geq 1 \f$ this proposes **both** edge
    ///                  (1-simplex) and **triangle (2-simplex)** attachments — the
    ///                  triangles are what the metric \f$ L_k \f$ sees through
    ///                  \f$ \partial_2 \f$, so the free search is not 1-skeleton-inert
    ///                  the way a pure edge search would be at \f$ k\geq 1 \f$.
    /// @param connectivityCandidates  bounded number of candidate connectivities
    ///                  scored **per atom kind** per growth step under
    ///                  `FreeConnectivity` (edges and, at \f$ k\geq 1 \f$, triangles
    ///                  each capped here); ignored under `Cone`. Documented + logged.
    /// @throws std::invalid_argument if `target` is empty, its degree is negative,
    ///   or none of its \f$ k \f$-cells are boundary cells of the bulk (the surface
    ///   does not match \f$ \partial W \f$).
    /// @param harmonic  when `true`, score \f$ \lVert L_k\psi\rVert^2 \f$ (the
    ///                  distance from \f$ \ker L_k = H_k(W) \f$) so realizable
    ///                  means the boundary \f$ k \f$-form is **carried as a bulk
    ///                  harmonic** (the meridian/longitude survival test): the
    ///                  class lies in \f$ \mathrm{image}(H_k(\partial W)\to H_k(W))
    ///                  \f$. The default (`false`) keeps the eigenvalue-agnostic
    ///                  residual (any eigenvalue), which is under-constrained on
    ///                  small boundaries (it accepts a non-harmonic eigenvector).
    /// @param beta      coupling of the mediated objective
    ///                  \f$ F_\beta=r_U+\beta\,|S_{\mathrm{Regge}}(W^{*})| \f$
    ///                  (see `decide`); \f$ \beta=0 \f$ reproduces the base layer
    ///                  bit-for-bit.
    /// @param maxVertices the \f$ \abs{W} \f$ volume bound on additive growth.
    [[nodiscard]] Verdict decideHarmonic(const Cochain &target,
                                         double epsilon = 1e-10, int restarts = 64,
                                         int maxCones = 4, std::uint64_t seed = 0,
                                         GrowthMode mode = GrowthMode::Cone,
                                         int connectivityCandidates = 8,
                                         bool harmonic = false, double beta = 0.0,
                                         int maxVertices = 16);

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
        GrowthMode mode, int connectivityCandidates, bool harmonic, double beta,
        int maxVertices, std::vector<std::complex<double>> &witnessOut,
        int &conesApplied, int &candidatesOut, int &triangleCandidatesOut,
        int &surgeryRemovals, std::size_t &spaceSizeOut) const;

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
        double epsilon, int restarts, std::uint64_t passSeed, bool harmonic,
        std::vector<std::complex<double>> &witnessOut) const;

    // One surgery growth step (`Surgery` mode): score every interior-top-cell
    // removal (`EigenstateSynthesis::interiorTopCells` /`removeInteriorCell`) by
    // the residual it reaches after `optimizePass`, restoring after each trial,
    // and COMMIT the single best removal iff it strictly improves on
    // `currentResidual` (and leaves every pinned boundary tuple present). The
    // committed move is topology-changing — it can shift b_k — so the realized
    // bulk topology is what the residual selects. Returns true (and increments
    // `removalsOut`) iff a removal was committed.
    //
    // `beta` activates the **mediated** score: a removal candidate is ranked by
    // \f$ F_\beta=r_U+\beta\,|S_{\mathrm{Regge}}(W^{*})| \f$ rather than \f$ r_U \f$
    // alone, and committed iff it improves \f$ F_\beta \f$ over the current
    // complex — so the surgery the search keeps is the one that best trades
    // residual against gravitational action. With `beta == 0` the \f$ |S| \f$ term
    // is not computed and the ranking/commit rule is byte-for-byte the historical
    // residual-only one.
    [[nodiscard]] bool growBestSurgery(
        EigenstateSynthesis &es,
        const std::map<std::vector<std::uint64_t>, std::complex<double>>
            &pinnedByTuple,
        double epsilon, int restarts, std::uint64_t seed, bool harmonic,
        double currentResidual, double beta, int &removalsOut) const;

    // The mediated objective term: the magnitude of the dual Lorentzian Regge
    // action on the **current** complex held by `bulk_` (which `es` mutates in
    // place), \f$ |S_{\mathrm{Regge}}(W^{*})| =
    // |\texttt{ReggeSolver::dualReggeAction}| \f$. Builds the ReggeSolver in C++
    // (its ctor materializes the facet/coface lattice the dual volumes need; the
    // added sub-simplices are invisible to the top-cell-filtered fill, verified
    // byte-for-byte). A degenerate geometry (non-finite action) is logged and
    // returned as +infinity so the offending candidate is rejected by the
    // min-\f$ F_\beta \f$ selection — surfaced, never silently repaired.
    [[nodiscard]] double reggeActionMagnitude() const;

    // One free-connectivity growth step. Generates the bounded candidate breadth
    // — edge fans at every degree and, at k>=1, the triangle (2-simplex) fans the
    // metric L_k reads through ∂_2 — reporting the counts into `candidatesOut` /
    // `triangleCandidatesOut` and the 2^N-1 vertex-incidence space into
    // `spaceSizeOut` (logged — no silent cap).
    //
    // At k=0 the edge fans are scored by `optimizePass` (attach, score, detach)
    // and the best is committed; the historical path is byte-for-byte preserved.
    // At k>=1 every additive candidate is provably spectrally inert —
    // ChainComplex::fromSpacetime builds only the top cells' downward closure, so a
    // dangling edge/triangle is dropped from L_k, and an additive *top-cell* attach
    // is boundary-locked (it introduces new boundary edges incident to the new
    // vertex, which the bit-exact ∂W guard rejects). Both are certified by the test
    // suite, so the additive candidates are enumerated + logged but not scored
    // (scoring would only confirm no improvement while perturbing the vertex-id
    // allocator), and the step uses the one boundary-fixed move that DOES enrich
    // L_k: the stellar Pachner subdivision (`growInterior`). Returns true if a
    // vertex was committed.
    [[nodiscard]] bool growBestConnectivity(
        EigenstateSynthesis &es,
        const std::map<std::vector<std::uint64_t>, std::complex<double>>
            &pinnedByTuple,
        double epsilon, int restarts, std::uint64_t seed, int nCandidates,
        bool harmonic, int &candidatesOut, int &triangleCandidatesOut,
        std::size_t &spaceSizeOut) const;

    // The bounded, documented candidate-connectivity generator: deterministic
    // anchors (cone-equivalent — the d+1 vertices of a top cell; full-star — all
    // vertices; boundary-star — all boundary vertices) plus reproducible random
    // vertex subsets, deduplicated and capped at `nCandidates`. Each candidate is
    // the set of existing vertices the new interior vertex wires to by an edge.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> connectivityCandidates(
        const EigenstateSynthesis &es, int nCandidates, std::uint64_t seed) const;

    // The bounded, documented **triangle** candidate generator (k>=1 only): each
    // candidate is a list of 2-vertex specs {u,w} over **existing edges**, so the
    // fresh interior vertex cones a fan of 2-simplices {v_new,u,w} — the cells the
    // metric L_k reads through ∂_2. Deterministic anchors (a top cell's edges; all
    // edges; all boundary edges; all interior edges) plus reproducible random edge
    // subsets, deduplicated and capped at `nCandidates`. In 3D these never create a
    // tetrahedron, so ∂W's facet count is untouched and the attach is boundary-safe;
    // any spec that would still perturb ∂W is rejected by attachInteriorVertex and
    // skipped. Returns the spec-lists ready to hand to attachInteriorVertex.
    [[nodiscard]] std::vector<std::vector<std::vector<std::uint64_t>>>
    triangleConnectivityCandidates(const EigenstateSynthesis &es, int nCandidates,
                                   std::uint64_t seed) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REALIZABILITYORACLE_H
