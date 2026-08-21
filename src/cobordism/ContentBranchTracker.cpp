// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ContentBranchTracker.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {
using Cell = std::vector<std::uint64_t>;
using Complex = std::complex<double>;

Cell sortedIds(const SimplexPtr &simplex) {
  Cell ids;
  ids.reserve(simplex ? simplex->size() : 0);
  if (simplex)
    for (const auto &vertex : simplex->getVertices())
      if (vertex) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace

ContentBranchSnapshot ContentBranchTracker::update(const Spacetime &spacetime) {
  const ChainComplex chain = ChainComplex::fromSpacetime(spacetime);
  const std::vector<Cell> cells = chain.orientedTopSimplices();

  std::map<Cell, Complex> principalByCell;
  for (const auto &simplex : spacetime.getTopSimplices()) {
    if (!simplex) continue;
    principalByCell[sortedIds(simplex)] = simplex->volume();
  }
  for (const auto &cell : cells)
    if (!principalByCell.contains(cell))
      throw std::runtime_error(
          "ContentBranchTracker::update: canonical top cell has no simplex");

  std::map<Cell, Complex> previousByCell;
  for (std::size_t index = 0;
       index < snapshot_.cells.size() && index < snapshot_.contents.size();
       ++index)
    previousByCell[snapshot_.cells[index]] = snapshot_.contents[index];

  ContentBranchSnapshot next;
  next.cells = cells;
  next.orientation = ChainComplex::orientationLocalSystem(cells);
  next.contents.resize(cells.size(), Complex{0.0, 0.0});
  std::vector<int> sheet(cells.size(), 1);

  const double epsilon = std::numeric_limits<double>::epsilon();
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const Complex principal = principalByCell.at(cells[index]);
    const int orientationSign = next.orientation.trivialization[index];
    const Complex seeded = static_cast<double>(orientationSign) * principal;
    const auto previous = previousByCell.find(cells[index]);
    if (previous == previousByCell.end()) {
      next.contents[index] = seeded;
      ++next.seededCells;
      continue;
    }

    ++next.continuedCells;
    const double sameDistance = std::abs(seeded - previous->second);
    const double flippedDistance = std::abs(-seeded - previous->second);
    const double scale = std::max({std::abs(seeded), std::abs(previous->second), 1.0});
    if (std::abs(sameDistance - flippedDistance) <= 64.0 * epsilon * scale)
      ++next.ambiguousCells;
    if (flippedDistance < sameDistance) {
      next.contents[index] = -seeded;
      sheet[index] = -1;
      ++next.principalBranchFlips;
    } else {
      next.contents[index] = seeded;
    }
  }

  // A root sign is a local Z2 gauge choice. Move the connection into the same
  // branch gauge so the covariant Laplacian and every Wilson-loop product are
  // invariant under principal-cut crossings.
  for (std::size_t index = 0; index < next.orientation.trivialization.size(); ++index)
    next.orientation.trivialization[index] *= sheet[index];
  for (auto &transition : next.orientation.transitions) {
    transition.transport *= sheet[transition.first] * sheet[transition.second];
    transition.holonomy =
        next.orientation.trivialization[transition.first] * transition.transport *
        next.orientation.trivialization[transition.second];
  }

  snapshot_ = next;
  return snapshot_;
}

}  // namespace tessera::cobordism
