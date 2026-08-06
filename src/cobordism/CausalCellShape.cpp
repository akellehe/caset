// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CausalCellShape.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <utility>

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {
namespace {

using VertexPair = std::pair<std::uint64_t, std::uint64_t>;

VertexPair orderedPair(std::uint64_t a, std::uint64_t b) {
  return {std::min(a, b), std::max(a, b)};
}

/// The timelike edges of `cell`, as ordered endpoint-id pairs.
std::set<VertexPair> timelikeEdgesOf(const Simplex &cell) {
  std::set<VertexPair> timelike;
  for (const auto &edge : cell.getEdges()) {
    if (!edge || edge->getSource() == nullptr || edge->getTarget() == nullptr)
      continue;
    if (edge->isTimelike())
      timelike.insert(orderedPair(edge->getSource()->getId(),
                                  edge->getTarget()->getId()));
  }
  return timelike;
}

std::vector<std::uint64_t> vertexIdsOf(const Simplex &cell) {
  std::vector<std::uint64_t> ids;
  ids.reserve(cell.size());
  for (const auto &vertex : cell.getVertices())
    if (vertex != nullptr) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

/// The crossing set of the split (`group`, rest): every pair with exactly one
/// endpoint in `group`.
std::set<VertexPair> crossingSet(const std::vector<std::uint64_t> &ids,
                                 const std::set<std::uint64_t> &group) {
  std::set<VertexPair> crossing;
  for (std::size_t i = 0; i + 1 < ids.size(); ++i)
    for (std::size_t j = i + 1; j < ids.size(); ++j)
      if (group.count(ids[i]) != group.count(ids[j]))
        crossing.insert(orderedPair(ids[i], ids[j]));
  return crossing;
}

}  // namespace

std::string CausalCellShape::shapeName(Shape shape) {
  switch (shape) {
    case Shape::Spacelike:    return "spacelike";
    case Shape::FourOne:      return "(4,1)";
    case Shape::ThreeTwo:     return "(3,2)";
    case Shape::NonBipartite: return "non-bipartite";
  }
  return "unknown";
}

CausalCellShape::Shape CausalCellShape::classify(const Simplex &cell) {
  const auto timelike = timelikeEdgesOf(cell);
  if (timelike.empty()) return Shape::Spacelike;

  const auto ids = vertexIdsOf(cell);
  if (ids.size() < 2) return Shape::NonBipartite;

  // Try every split with a smaller group of size 1 .. floor(n/2); a split and its
  // complement give the same crossing set, so only the smaller side is enumerated.
  // A shape is claimed only when the timelike edges are EXACTLY the crossing set —
  // matching the count alone would accept cells with the right number of timelike
  // edges arranged in no consistent temporal split.
  const std::size_t half = ids.size() / 2;
  for (std::size_t groupSize = 1; groupSize <= half; ++groupSize) {
    std::vector<bool> selector(ids.size(), false);
    std::fill(selector.begin(), selector.begin() + groupSize, true);
    std::vector<bool> combination = selector;
    std::sort(combination.begin(), combination.end());
    do {
      std::set<std::uint64_t> group;
      for (std::size_t i = 0; i < ids.size(); ++i)
        if (combination[i]) group.insert(ids[i]);
      if (crossingSet(ids, group) != timelike) continue;
      const std::size_t larger = ids.size() - groupSize;
      if (groupSize == 1 && larger == 4) return Shape::FourOne;
      if (groupSize == 2 && larger == 3) return Shape::ThreeTwo;
      // A consistent split of some other size (a complex of another dimension):
      // bipartite, but not one of the two 4-simplex shapes this reports.
      return Shape::NonBipartite;
    } while (std::next_permutation(combination.begin(), combination.end()));
  }
  return Shape::NonBipartite;
}

std::vector<int> CausalCellShape::distribution(const Spacetime &spacetime) {
  std::vector<int> counts(4, 0);
  std::size_t widest = 0;
  for (const auto &simplex : spacetime.getSimplices())
    if (simplex != nullptr) widest = std::max(widest, simplex->size());
  if (widest == 0) return counts;
  for (const auto &simplex : spacetime.getSimplices()) {
    if (simplex == nullptr || simplex->size() != widest) continue;
    counts[static_cast<std::size_t>(classify(*simplex))] += 1;
  }
  return counts;
}

CausalCellShape::DualHeightCensus CausalCellShape::dualHeightCensus(
    const Spacetime &spacetime) {
  DualHeightCensus census;

  // The top dimension: heights exist only BELOW it, since a top cell's dual is a
  // point and contributes none.
  std::size_t widest = 0;
  for (const auto &simplex : spacetime.getSimplices())
    if (simplex != nullptr) widest = std::max(widest, simplex->size());
  if (widest == 0) return census;

  for (const auto &simplex : spacetime.getSimplices()) {
    if (simplex == nullptr || simplex->size() >= widest) continue;
    // Orphans -- sub-faces a Pachner move stranded with no surviving top coface --
    // are not part of the complex and contribute no dual content.
    if (!simplex->hasTopCoface()) continue;

    const double ownCircumradius = simplex->circumradiusSquared();
    std::set<std::uint64_t> ownVertices;
    for (const auto &vertex : simplex->getVertices())
      if (vertex != nullptr) ownVertices.insert(vertex->getId());

    for (const auto &coface : simplex->getCofaces()) {
      if (!coface) continue;
      census.terms += 1;

      // The barycentric factor: `oppositeVertexSign` reads the coordinate at the
      // ONE vertex of the coface not in this simplex -- not min(barycentric).
      const auto &cofaceVertices = coface->getVertices();
      int oppositeIndex = -1;
      for (std::size_t i = 0; i < cofaceVertices.size(); ++i)
        if (cofaceVertices[i] != nullptr &&
            ownVertices.count(cofaceVertices[i]->getId()) == 0) {
          oppositeIndex = static_cast<int>(i);
          break;
        }
      bool defect = false;
      if (oppositeIndex >= 0) {
        const std::vector<double> barycentric = coface->circumcenterBarycentric();
        if (oppositeIndex < static_cast<int>(barycentric.size()))
          defect = barycentric[static_cast<std::size_t>(oppositeIndex)] < 0.0;
      }

      // The radicand factor: negative means the circumcentre separation is
      // TIMELIKE -- physical Lorentzian structure in the dual, not a defect.
      const bool timelike =
          (coface->circumradiusSquared() - ownCircumradius) < 0.0;

      if (defect) census.centerednessDefects += 1;
      if (timelike) census.timelikeSeparations += 1;
      // Exactly one negative factor flips the height; both leave it positive.
      if (defect != timelike) census.negativeHeights += 1;
    }
  }
  return census;
}

int CausalCellShape::timelikeDirectionCount(const Simplex &cell,
                                            double tolerance) {
  const int nVertices = static_cast<int>(cell.size());
  if (nVertices < 2) return 0;  // a point or an edge carries no tangent metric
  const int dimension = nVertices - 1;

  // Signature-aware Gram: a timelike edge keeps its negative l^2, so the metric
  // signature of the cell is recorded in G rather than wicked away.
  const std::vector<double> gram = cell.gramMatrix(/*wickRotate=*/false);

  // Jacobi's criterion: for a non-degenerate symmetric matrix the number of
  // negative eigenvalues equals the number of sign changes in the sequence of
  // leading principal minors 1, D_1, ..., D_d. Eigen-free, reusing the same
  // determinant helper assertSpacelikeAdmissible uses.
  int signChanges = 0;
  double previous = 1.0;
  for (int k = 1; k <= dimension; ++k) {
    std::vector<double> leading(static_cast<std::size_t>(k) * k);
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < k; ++j)
        leading[static_cast<std::size_t>(i) * k + j] =
            gram[static_cast<std::size_t>(i) * dimension + j];
    const double minor = Simplex::determinant(leading, k);
    // A vanishing minor leaves the signature undefined -- degenerate, not zero.
    if (std::abs(minor) <= tolerance) return -1;
    if ((minor < 0.0) != (previous < 0.0)) ++signChanges;
    previous = minor;
  }
  return signChanges;
}

bool CausalCellShape::isLorentzianAdmissible(const Simplex &cell,
                                             double tolerance) {
  const int timelikeDirections = timelikeDirectionCount(cell, tolerance);
  if (timelikeDirections < 0) return false;  // degenerate
  // A Lorentzian cell has exactly one timelike direction, signature (-,+,+,+).
  // A purely spacelike cell has none, which is the positive-definite condition
  // assertSpacelikeAdmissible checks, reached here through the same criterion.
  const bool anyTimelikeEdge = !timelikeEdgesOf(cell).empty();
  return anyTimelikeEdge ? (timelikeDirections == 1) : (timelikeDirections == 0);
}

}  // namespace tessera::cobordism
