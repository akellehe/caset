// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_OBSERVABLES_SPARSEGRAPH_H
#define TESSERA_OBSERVABLES_SPARSEGRAPH_H

#include "graph/spectral_graph.hpp"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace tessera {

/// Undirected sparse graph in CSR (compressed sparse row) form.
///
/// Built from a (rows, cols) COO array — typically from
/// :func:`Spacetime::getDualAdjacency` for the modularity / spectral-
/// dimension observables.  All operations are read-only on the graph
/// data; the modularity sweep recomputes the graph from the spacetime
/// at every measurement rather than mutating it.
///
/// Derives from ``SpectralGraph``, inheriting the diagonal heat-kernel
/// and return-probability machinery; the override here installs the
/// symmetric normalised Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}``.
///
/// No Eigen dependency — plain ``std::vector`` storage so this header
/// is cheap to include from anywhere in tessera_core.
class SparseGraph : public SpectralGraph {
public:
  /// Build from COO.  ``rows`` and ``cols`` may contain duplicates
  /// and need not be symmetric — both directions are added once each
  /// and duplicates collapsed (binary adjacency).
  static SparseGraph fromCOO(const std::vector<std::uint32_t> &rows,
                             const std::vector<std::uint32_t> &cols,
                             std::uint32_t n);

  std::size_t nNodes() const noexcept { return nNodes_; }
  std::size_t nEdges() const noexcept { return nEdges_; }
  /// CSR index pointers, length nNodes() + 1.
  const std::vector<std::int64_t> &indptr() const noexcept { return indptr_; }
  /// CSR column indices, length 2 * nEdges() (each undirected edge
  /// stored twice).
  const std::vector<std::uint32_t> &indices() const noexcept { return indices_; }
  /// Per-node degree.  ``degree(i) == indptr_[i+1] - indptr_[i]``.
  std::uint32_t degree(std::uint32_t i) const noexcept {
    return static_cast<std::uint32_t>(indptr_[i + 1] - indptr_[i]);
  }

  // ── SpectralGraph overrides ─────────────────────────────────────────
  int nVertices() const override {
    return static_cast<int>(nNodes_);
  }

  /// y ← L_sym · x where L_sym = I - D^{-1/2} A D^{-1/2}.
  /// Isolated nodes (degree 0) get invSqrtDeg = 0, so the matvec
  /// reduces to y_i = x_i — L_sym acts as the identity on them.
  void applyLaplacian(std::vector<double> const &x,
                        std::vector<double> &y) const override;

  /// True iff the graph is 2-colorable (no odd cycle).
  /// BFS-based; empty graphs are trivially bipartite.
  bool isBipartite() const;

  /// Diagonal of the heat kernel ``e^{-t L_sym}`` for each
  /// (start, t) pair.
  ///
  /// Returns a flat row-major matrix of shape
  /// ``(starts.size(), times.size())``.  ``out[w][j]`` is the
  /// approximated ``[e^{-times[j] L_sym}]_{starts[w], starts[w]}``.
  ///
  /// Implemented as a thin uint32_t-keyed wrapper around the inherited
  /// ``SpectralGraph::diagonalHeatKernel`` so that the Python binding
  /// keeps its existing list-of-unsigned-index signature, and the
  /// nEdges_ == 0 fast path stays here (empty-graph return = 1.0 by
  /// convention).
  ///
  /// Note: ``L_rw = I - D^{-1} A`` (random-walk Laplacian) and
  /// ``L_sym = I - D^{-1/2} A D^{-1/2}`` (symmetric normalized) are
  /// related by ``L_rw = D^{-1/2} L_sym D^{1/2}``.  Their *diagonal*
  /// entries of ``e^{-tL}`` are identical (D^{1/2} D^{-1/2} = I on
  /// the diagonal) — so D_S extraction gives the same answer with
  /// either, and L_sym is symmetric, which lets us use the cheap
  /// Lanczos rather than Arnoldi.
  std::vector<double> diagonalHeatKernel(
      const std::vector<std::uint32_t> &starts,
      const std::vector<double> &times,
      int krylovDim = 30) const;

  /// Estimate the spectral dimension at small and large diffusion
  /// times via a centered finite-difference of ``-2 d log K(t) /
  /// d log t``.  Mirrors the Python implementation in
  /// ``examples/modularity.py:spectral_dimension``.
  ///
  /// Picks ``nWalks`` random start nodes uniformly without
  /// replacement (capped at ``nNodes()``).
  ///
  /// Returns ``(D_S_small, D_S_large)`` as a pair, or ``{NaN, NaN}``
  /// if the graph is too small or has no valid log-K samples.
  ///
  /// Note this overload returns a (small, large) pair via random-walk
  /// sampling, distinct from the inherited static
  /// ``SpectralGraph::spectralDimension(sigmas, P)`` which is a pure
  /// finite-difference helper on a precomputed P(σ) curve.
  std::pair<double, double> spectralDimension(
      int nWalks, double maxSigma, std::mt19937 *rng,
      double tailFraction = 0.2, int nTimes = 40,
      double tMin = 0.5, int krylovDim = 30) const;

private:
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;     // size nNodes + 1
  std::vector<std::uint32_t> indices_;   // size 2 * nEdges
  std::vector<double> invSqrtDeg_;        // size nNodes; 0.0 for isolated nodes
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_SPARSEGRAPH_H
