// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_OBSERVABLES_SPARSEGRAPH_H
#define TESSERA_OBSERVABLES_SPARSEGRAPH_H

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
/// No Eigen dependency — plain ``std::vector`` storage so this header
/// is cheap to include from anywhere in tessera_core.
class SparseGraph {
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

  /// True iff the graph is 2-colorable (no odd cycle).
  /// BFS-based; empty graphs are trivially bipartite.
  bool isBipartite() const;

  /// Diagonal of the heat kernel ``e^{-t L_sym}`` for each
  /// (start, t) pair, via Krylov-Lanczos approximation on the
  /// symmetric normalized Laplacian.
  ///
  /// Returns a flat row-major matrix of shape
  /// ``(starts.size(), times.size())``.  ``out[w][j]`` is the
  /// approximated ``[e^{-times[j] L_sym}]_{starts[w], starts[w]}``.
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
  std::pair<double, double> spectralDimension(
      int nWalks, double maxSigma, std::mt19937 *rng,
      double tailFraction = 0.2, int nTimes = 40,
      double tMin = 0.5, int krylovDim = 30) const;

private:
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;     // size nNodes + 1
  std::vector<std::uint32_t> indices_;   // size 2 * nEdges
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_SPARSEGRAPH_H
