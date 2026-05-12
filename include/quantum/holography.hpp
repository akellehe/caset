// Emergent spectral dimension from the Schwinger TDVP state.
//
// See docs/source/holography-causal-ordering-emergent-dimension.md for
// the scientific charter and falsification criteria. This header
// declares the C++-side classes that implement that charter:
//
//   • HolographyConfig         — composes TDVPConfig with σ-grid + ε_I
//                                + temporal-stride controls
//   • MutualInformationProfile — flat (site, snapshot) label set with
//                                a symmetric N·K × N·K MI matrix
//                                derived from per-snapshot all-pairs MI
//   • EmergentGraph            — weighted Laplacian + heat-kernel
//                                trace + Graphviz export
//   • AmbjornLollFit           — three-parameter D_S(σ) curve fit
//   • SpectralDimensionResult  — data bundle
//   • EmergentSpectralDimension — coarse-grained workflow class:
//                                 binds a HolographyConfig and exposes
//                                 ``compute()`` returning a
//                                 SpectralDimensionResult
//
// The implementation mirrors the existing SchwingerModel /
// SchwingerQuench layout: a single workflow class binds the config and
// runs the full pipeline; pure-math operations live on stateless
// utility classes.

#pragma once

#include "quantum/tdvp_runner.hpp"  // TDVPConfig, TDVPSnapshot, QuenchResult

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <string>
#include <utility>
#include <vector>

namespace tessera::quantum {

// ─── Config ──────────────────────────────────────────────────────────

// Configuration for an emergent-spectral-dimension run.
//
// Wraps a TDVPConfig and adds the σ-grid for D_S(σ) plus a
// mutual-information cutoff ε_I. Validated at construction; invalid
// configs throw std::invalid_argument before any TDVP call.
struct HolographyConfig {
    TDVPConfig tdvp;

    // σ-grid for D_S(σ); logarithmically spaced.
    double sigmaMin{1e-2};
    double sigmaMax{1e3};
    int    sigmaCount{50};

    // Mutual-information cutoff (edges with I < epsilonI are dropped).
    double epsilonI{1e-10};

    // Build cross-snapshot temporal MI edges via the Choi-state
    // construction (`ChoiPropagator`). When true, the temporal MI for
    // every snapshot pair (s, t) with t − s ≤ maxTemporalStride is
    // computed once per unique stride (the Schwinger Hamiltonian is
    // time-independent, so U_{s → t} depends only on (t − s)).
    bool   includeTemporal{false};
    int    maxTemporalStride{0};  // 0 = unlimited (all strides 1..K-1)

    // Krylov dimension for the heat-kernel trace estimator.
    int    krylovDim{30};

    // Reproducibility (TDVP itself is deterministic; this seeds the
    // Hutchinson-style trace estimator and any future stochastic
    // steps).
    int    seed{0};

    // Throws std::invalid_argument on contradictions:
    //   sigmaMin <= 0, sigmaMax <= sigmaMin, sigmaCount < 8,
    //   epsilonI < 0, maxTemporalStride < 0.
    void validate() const;
};

// ─── MutualInformationProfile ────────────────────────────────────────

// Symmetric MI matrix on the (site × snapshot) label set.
//
// Built from a vector of TDVPSnapshots that have ``mutualInformation``
// recorded (i.e. TDVPConfig::recordMutualInformation = true). Storage
// is a dense symmetric (N·K) × (N·K) matrix; zero off-diagonal for
// (site, snap)-pairs in different snapshots until temporal MI lands.
//
// Flat index convention: idx = snap * N + site, with snap ∈ [0, K) and
// site ∈ [0, N).
class MutualInformationProfile {
public:
    MutualInformationProfile(
        std::vector<TDVPSnapshot> const& snapshots,
        HolographyConfig const& config);

    [[nodiscard]] int nSites()     const noexcept { return nSites_; }
    [[nodiscard]] int nSnapshots() const noexcept { return nSnapshots_; }
    [[nodiscard]] int nLabels()    const noexcept { return nSites_ * nSnapshots_; }

    // I({site_v, snap_v} : {site_w, snap_w}). 0-based indices.
    // Returns 0 outside the dense block (different-snapshot pairs in
    // v1, or sites equal).
    [[nodiscard]] double
    at(int siteV, int snapV, int siteW, int snapW) const;

    // Flat-index accessor for callers that already have the packed idx.
    [[nodiscard]] double
    atFlat(int v, int w) const noexcept;

    // Decompose a flat label into (site, snapshot).
    [[nodiscard]] int siteOf(int label)     const noexcept { return label % nSites_; }
    [[nodiscard]] int snapshotOf(int label) const noexcept { return label / nSites_; }

    // COO arrays for the edge set with I > epsilon_I. Symmetric — each
    // edge appears twice (rows[k]=i, cols[k]=j AND rows[k']=j, cols[k']=i).
    struct COO {
        std::vector<int>    rows;
        std::vector<int>    cols;
        std::vector<double> weights;
        int                 nVertices{0};
    };
    [[nodiscard]] COO weightedAdjacency() const;

private:
    int                 nSites_{0};
    int                 nSnapshots_{0};
    double              epsilonI_{0.0};
    // Row-major (nLabels × nLabels) MI values; zero outside same-
    // snapshot blocks in v1.
    std::vector<double> mi_;
};

// ─── EmergentGraph ───────────────────────────────────────────────────

// Weighted graph on the (site × snapshot) label set with edge weights
// I(v, w). Stores the CSR adjacency, the per-vertex weighted degree,
// and the Laplacian implicitly via L · x = D · x − W · x.
class EmergentGraph {
public:
    explicit EmergentGraph(MutualInformationProfile const& profile);

    // Direct construction from a weighted edge list. Each undirected
    // edge (u, v) with weight w should appear once; the constructor
    // installs both (u → v) and (v → u) into the CSR adjacency.
    // `n` is the total vertex count. Used for the §H4 known-graph
    // acceptance tests (1D chain, 2D lattice, complete graph).
    [[nodiscard]] static EmergentGraph
    fromWeightedEdges(int n,
                       std::vector<std::tuple<int, int, double>> const& edges);

    [[nodiscard]] int nVertices() const noexcept { return n_; }
    [[nodiscard]] int nEdges()    const noexcept { return nEdges_; }

    // Sparse weighted Laplacian L = D - W. Symmetric, in CSR format.
    [[nodiscard]] Eigen::SparseMatrix<double> laplacian() const;

    // P(σ) = (1/|V|) Tr e^{-σ L} for the weighted Laplacian.
    //
    // The trace is computed exactly via diagonal Krylov-Lanczos:
    // for each vertex v as a starting vector e_v, project L onto the
    // K-dim Krylov subspace, exponentiate the resulting tridiagonal
    // T (small dense matrix), and read [exp(-σ T)]_{0,0} which gives
    // exactly e_v^T exp(-σ L) e_v. Summing over v yields the trace.
    //
    // The same pattern is used by tessera::SparseGraph::diagonalHeatKernel
    // for unweighted Laplacians; extended here to weighted L.
    [[nodiscard]] std::vector<double>
    returnProbability(std::vector<double> const& sigmas,
                       int krylovDim = 30) const;

    // D_S(σ) = -2 d log P / d log σ via centered finite differences,
    // with one-sided differences at the endpoints. Pure function; the
    // graph instance is unused. Exposed here for the workflow class to
    // call without round-tripping through Python.
    [[nodiscard]] static std::vector<double>
    spectralDimension(std::vector<double> const& sigmas,
                       std::vector<double> const& P);

    // Savitzky-Golay smoothed D_S(σ): for each interior point, fit a
    // local polynomial in (log σ, log P) over a centered window of
    // size `windowSize` and read off the slope at that σ. Endpoints
    // use a one-sided window. Returns -2 · slope at each grid point.
    //
    // Per spec §8: "Smoothing via local polynomial fit
    // (Savitzky-Golay, window 5) before differentiation is acceptable;
    // report both raw and smoothed D_S."
    //
    // windowSize must be odd and >= polyOrder + 1. Default (window 5,
    // poly order 2) matches the spec recommendation.
    [[nodiscard]] static std::vector<double>
    spectralDimensionSmoothed(std::vector<double> const& sigmas,
                               std::vector<double> const& P,
                               int windowSize = 5,
                               int polyOrder = 2);

    // Graphviz DOT representation. Mirrors Poset::toDot. Edge labels
    // are I(v,w) to 3 significant digits. Suitable for `dot -Tsvg`.
    [[nodiscard]] std::string toDot() const;

    // GraphML export. Suitable for import into Gephi or yEd. Mirrors
    // the `tessera.Spacetime.save("*.graphml")` pattern; the edge
    // weight is exported as a "weight" attribute on each edge.
    [[nodiscard]] std::string toGraphML() const;

private:
    EmergentGraph() = default;
    void buildFromCOO_(int n,
                        std::vector<int> const& rows,
                        std::vector<int> const& cols,
                        std::vector<double> const& weights);

    int n_{0};
    int nEdges_{0};
    // CSR weighted adjacency: indptr[v]..indptr[v+1] points into
    // (indices, weights). Symmetric (each undirected edge listed twice).
    std::vector<int>    indptr_;
    std::vector<int>    indices_;
    std::vector<double> weights_;
    std::vector<double> degrees_;  // Σ_w I(v, w)
};

// ─── AmbjornLollFit ──────────────────────────────────────────────────

// D_S(σ) = D_∞ - C / (B + σ), the three-parameter form used by
// Ambjorn-Loll for CDT (and by examples/spectral_dimension.py).
//
// Stateless utility class. Fit is done by Levenberg-Marquardt-style
// Gauss-Newton iteration; for our σ-grid sizes (~50) this converges in
// a few hundred microseconds and never needs an external dependency.
class AmbjornLollFit {
public:
    AmbjornLollFit() = delete;
    AmbjornLollFit(AmbjornLollFit const&) = delete;
    AmbjornLollFit& operator=(AmbjornLollFit const&) = delete;

    struct Result {
        double dInfinity{0.0};
        double C{0.0};
        double B{0.0};
        double chiSquared{0.0};  // reduced χ²; 0 if not fittable
    };

    // Fit on the chosen σ window. If sigmaFitMin/Max are negative,
    // the full grid is used. Returns zeros when nPoints < 4 or fit
    // diverges.
    [[nodiscard]] static Result
    fit(std::vector<double> const& sigmas,
        std::vector<double> const& dS,
        double sigmaFitMin = -1.0,
        double sigmaFitMax = -1.0);
};

// ─── Result ──────────────────────────────────────────────────────────

struct SpectralDimensionResult;

namespace detail {
// Tiny self-contained JSON writer for SpectralDimensionResult — keeps
// the header free of an external JSON dependency. Defined in
// holography.cpp.
[[nodiscard]] std::string
serialiseResultToJson(SpectralDimensionResult const& result,
                       HolographyConfig const& config);
} // namespace detail

struct SpectralDimensionResult {
    // σ-grid and the heat-kernel observable.
    std::vector<double> sigmas;
    std::vector<double> P;
    std::vector<double> dS;          // centered finite differences (raw)
    std::vector<double> dSSmoothed;  // Savitzky-Golay smoothed (window 5, poly 2)

    // Ambjorn-Loll fit on the smoothed D_S(σ) — the raw signal has
    // grid-spacing noise that the fit can latch onto. Spec §8
    // recommends reporting both.
    double dInfinity{0.0};
    double C{0.0};
    double B{0.0};
    double fitChiSquared{0.0};

    // Graph diagnostics.
    int graphNVertices{0};
    int graphNEdges{0};

    // TDVP summary, copied from the underlying snapshots.
    std::vector<double> snapshotTimes;
    std::vector<int>    snapshotBondDims;
    std::vector<double> snapshotEnergies;

    // Reproducibility serialisation per spec §10 — emit a single JSON
    // record with config, tdvp_summary, graph, spectral_dimension,
    // and a small provenance block.
    [[nodiscard]] std::string
    toJson(HolographyConfig const& config) const {
        return detail::serialiseResultToJson(*this, config);
    }
};

// ─── Pipeline ────────────────────────────────────────────────────────

// Coarse-grained workflow class: binds a HolographyConfig and exposes
// `compute()` returning the full result. Mirrors the
// SchwingerModel(cfg).solve() / SchwingerQuench(cfg).evolve() pattern.
class EmergentSpectralDimension {
public:
    explicit EmergentSpectralDimension(HolographyConfig config);

    [[nodiscard]] HolographyConfig const& config() const noexcept { return config_; }

    // Run the TDVP-only pipeline: DMRG ground state → q-qbar quench →
    // TDVP loop with MI recording → (site, time) graph → heat-kernel
    // trace → D_S(σ) → Ambjorn-Loll fit.
    //
    // ``recordMutualInformation`` is forced to true on the underlying
    // TDVPConfig regardless of what the caller set, because the graph
    // construction needs the all-pairs MI per snapshot. ``epsilonI``
    // controls the MI cutoff for edge construction (smaller = denser
    // graph).
    [[nodiscard]] SpectralDimensionResult compute() const;

    // Compute D_S on an already-evolved quench. Useful when the caller
    // wants to reuse a single TDVP run across multiple σ-grids or
    // ε_I values without re-running TDVP.
    [[nodiscard]] SpectralDimensionResult
    computeFromSnapshots(QuenchResult const& quench) const;

private:
    HolographyConfig config_;
};

} // namespace tessera::quantum
