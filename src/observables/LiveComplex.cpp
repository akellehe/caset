// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/LiveComplex.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::mesh::Edge;

std::shared_ptr<Spacetime> LiveComplex::load(
    const std::vector<std::vector<std::uint64_t>> &cells,
    const std::map<std::pair<std::uint64_t, std::uint64_t>,
                   std::complex<double>> &squaredLengths,
    const std::map<std::uint64_t, double> &vertexTimes, int dimensions) {
  if (cells.empty()) {
    throw std::invalid_argument("LiveComplex::load needs at least one top cell");
  }
  // The canonical entry point — nothing is built outside it. fromCells lays down
  // the top cells (`dimensions` is the recorded complex dimension, passed
  // through, never guessed); the metric and times are then loaded back exactly
  // as recorded, and the facet skeleton is completed for reading.
  auto st = Spacetime::fromCells(dimensions, cells, 1.0, 0.0);
  if (!vertexTimes.empty()) {
    const auto &vertexList = st->getVertexList();
    for (const auto &kv : vertexTimes) {
      vertexList->get(kv.first)->setTime(kv.second);
    }
  }
  // The edges of a freshly loaded complex are legitimately mutable — loading the
  // recorded squared lengths is the whole point of a rehydration, not a change
  // to any emergent state.
  for (Edge *e : st->getEdgeList()->toVector()) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    auto it = squaredLengths.find(a < b ? std::make_pair(a, b)
                                        : std::make_pair(b, a));
    if (it == squaredLengths.end()) {
      throw std::out_of_range(
          "LiveComplex::load: edge (" + std::to_string(std::min(a, b)) + ", " +
          std::to_string(std::max(a, b)) +
          ") of the loaded complex has no recorded squared length — a partial "
          "metric is never silently defaulted");
    }
    e->setLength(std::sqrt(it->second));
  }
  // Complete the facet/coface skeleton the dual-volume / deficit reads walk —
  // the honest direct call, never a solver (fromCells leaves only the top
  // cells; this reproduces the ReggeSolver + ChainComplex skeleton bit-for-bit).
  st->materializeFacets();
  return st;
}

std::shared_ptr<Spacetime> LiveComplex::subcomplex(
    const std::vector<std::vector<std::uint64_t>> &cells, int dimensions) {
  if (cells.empty()) {
    throw std::invalid_argument(
        "LiveComplex::subcomplex needs at least one cell");
  }
  // The cells are already SELECTED by the caller (existing ambient top cells);
  // `dimensions` is the ambient complex's canonical dimension. This only
  // re-instantiates the selection with a uniform metric through the canonical
  // fromCells (the block-residual carry diagnostic), never a build.
  return Spacetime::fromCells(dimensions, cells, 1.0, 0.0);
}

LiveComplex::Relabeled LiveComplex::relabel(const Spacetime &spacetime,
                                            std::uint64_t seed) {
  // Read the recorded geometry off the live complex (const reads only). The
  // dimension is the canonical metric-signature dimension, not a cell-size guess.
  const int dimensions = spacetime.getMetric()->getSignature()->getDimensions();
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(spacetime.getTopSimplices().size());
  for (const auto *c : spacetime.getTopSimplices()) {
    std::vector<std::uint64_t> vids;
    vids.reserve(c->getVertices().size());
    for (const auto *v : c->getVertices()) vids.push_back(v->getId());
    cells.push_back(std::move(vids));
  }
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> edges;
  for (const auto *e : spacetime.getEdgeList()->toVector()) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edges[a < b ? std::make_pair(a, b) : std::make_pair(b, a)] =
        (e->getLength() * e->getLength());
  }
  std::map<std::uint64_t, double> times;
  const auto &vertexList = spacetime.getVertexList();
  for (const auto &cell : cells) {
    for (std::uint64_t v : cell) {
      if (!times.count(v)) times[v] = vertexList->get(v)->getTime();
    }
  }

  // A random vertex-id permutation + cell-order shuffle (deterministic in seed).
  std::set<std::uint64_t> uniqueVertices;
  for (const auto &cell : cells) {
    for (std::uint64_t v : cell) uniqueVertices.insert(v);
  }
  std::vector<std::uint64_t> allVertices(uniqueVertices.begin(),
                                         uniqueVertices.end());
  std::vector<std::uint64_t> shuffled = allVertices;
  std::mt19937_64 rng(seed);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  std::map<std::uint64_t, std::uint64_t> perm;
  for (std::size_t i = 0; i < allVertices.size(); ++i) {
    perm[allVertices[i]] = shuffled[i];
  }

  std::vector<std::size_t> order(cells.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  std::vector<std::vector<std::uint64_t>> permutedCells;
  permutedCells.reserve(cells.size());
  for (std::size_t idx : order) {
    std::vector<std::uint64_t> permuted;
    permuted.reserve(cells[idx].size());
    for (std::uint64_t v : cells[idx]) permuted.push_back(perm.at(v));
    permutedCells.push_back(std::move(permuted));
  }
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      permutedEdges;
  for (const auto &kv : edges) {
    const std::uint64_t a = perm.at(kv.first.first);
    const std::uint64_t b = perm.at(kv.first.second);
    permutedEdges[a < b ? std::make_pair(a, b) : std::make_pair(b, a)] =
        kv.second;
  }
  std::map<std::uint64_t, double> permutedTimes;
  for (const auto &kv : times) permutedTimes[perm.at(kv.first)] = kv.second;

  Relabeled out;
  out.spacetime = load(permutedCells, permutedEdges, permutedTimes, dimensions);
  out.vertexMap = std::move(perm);
  return out;
}

}  // namespace tessera::observables
