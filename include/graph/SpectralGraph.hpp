// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Abstract spectral-graph interface: a vertex count plus a Laplacian
// matvec is enough to compute the diagonal of the heat kernel, the
// return probability $P(\sigma) = (1/|V|) \mathrm{Tr}\,e^{-\sigma L}$,
// and the spectral dimension $D_S(\sigma) = -2\, d\log P / d\log\sigma$.
//
// Concrete derivations:
//
// * ``tessera::observables::SparseGraph`` (``include/observables/SparseGraph.h``)
//   uses the symmetric normalised Laplacian
//   $L_\text{sym} = I - D^{-1/2} A D^{-1/2}$.
// * ``tessera::quantum::EmergentGraph``
//   (``include/quantum/Holography.hpp``) uses the weighted Laplacian
//   $L = D - W$ on a CSR adjacency.
//
// Both derivations share the Krylov-Lanczos sweep, Padé-13 dense-
// tridiagonal matrix exponential, and finite-difference $D_S$ logic
// declared below.

#pragma once

#include <cstdint>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class SpectralGraph {
public:
    virtual ~SpectralGraph() = default;

    // Vertex count. Must match the dimension expected by applyLaplacian.
    virtual int nVertices() const = 0;

    // y ← L x. ``x`` and ``y`` must both have length nVertices(); the
    // implementation is free to assume that and free to leave ``y`` un-
    // sized (or sized) at entry as long as it's correctly sized on exit.
    virtual void applyLaplacian(std::vector<double> const& x,
                                  std::vector<double>& y) const = 0;

    // ── Concrete, inherited ───────────────────────────────────────────

    // Diagonal of $e^{-\sigma L}$ for each (start vertex, σ) pair, via
    // Krylov-Lanczos with full Gram-Schmidt re-orthogonalisation. The
    // projected tridiagonal $T$ at each starting vector is exponentiated
    // with Padé-13 scaling-and-squaring; $[e^{-\sigma T}]_{0,0}$ is then
    // exactly $\langle e_v | e^{-\sigma L} | e_v\rangle$ to Krylov order.
    //
    // Returns a flat row-major matrix of shape (starts.size() × sigmas.size()):
    //   out[s * nSigmas + j] = diagonal entry at vertex starts[s] for σ[j].
    //
    // Implementation is concrete and lives in
    // ``src/graph/SpectralGraph.cpp``.
    std::vector<double> diagonalHeatKernel(
        std::vector<int> const& starts,
        std::vector<double> const& sigmas,
        int krylovDim = 30) const;

    // $P(\sigma) = (1/|V|) \mathrm{Tr}\,e^{-\sigma L}$. Estimator is the
    // mean of the diagonal heat-kernel entry over a random ``m``-subset
    // of vertices (Hutchinson-style unbiased estimator on the trace
    // diagonal). When ``m >= n``, all vertices are used and the result
    // is exact; the random subset is sampled WITHOUT replacement so
    // repeated indices don't double-count. ``m <= 0`` requests the
    // default (``min(n, 3000)``), which is the Tier-1 fix from #28 —
    // cuts the heat-kernel cost from O(n²) to O(n·m). With m=3000 at
    // n=300k this gives a 100× speedup at a modest variance penalty
    // that the Savitzky-Golay smoother absorbs.
    //
    // ``seed`` controls the subset sampling RNG for reproducibility.
    std::vector<double> returnProbability(
        std::vector<double> const& sigmas,
        int krylovDim = 30,
        int m = 0,
        std::uint64_t seed = 0) const;

    // $D_S(\sigma) = -2 \,d\log P / d\log\sigma$ via centered finite
    // differences (one-sided at endpoints). Pure function; the graph
    // instance is unused. Returns a vector aligned with ``sigmas``;
    // entries where $P \leq 0$ or non-finite are NaN.
    static std::vector<double> spectralDimension(
        std::vector<double> const& sigmas,
        std::vector<double> const& P);

    // Savitzky-Golay smoothed $D_S(\sigma)$: at each interior σ, fit a
    // local polynomial of order ``polyOrder`` over a centered window of
    // ``windowSize`` $(log \sigma, log P)$ samples and read off the
    // slope. Endpoints use a one-sided window. ``windowSize`` must be
    // odd and ≥ ``polyOrder + 1``. Default (window 5, poly 2) matches
    // the spec recommendation in
    // ``docs/source/quantum-experiments/emergent-spectral-dimension-schwinger-tdvp.md``
    // §8.
    static std::vector<double> spectralDimensionSmoothed(
        std::vector<double> const& sigmas,
        std::vector<double> const& P,
        int windowSize = 5,
        int polyOrder = 2);
};

} // namespace tessera::graph
