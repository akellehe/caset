// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SlicedCobordism.h"

#include <algorithm>
#include <map>
#include <utility>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {
namespace {

using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

EdgeKey keyOf(std::uint64_t a, std::uint64_t b) {
  return {std::min(a, b), std::max(a, b)};
}

/// {min,max} -> squared length over every live edge of `complex`.
std::map<EdgeKey, std::complex<double>> squaredLengthsOf(
    const Spacetime &complex) {
  std::map<EdgeKey, std::complex<double>> lengths;
  const auto &edges = complex.getEdgeList();
  if (!edges) return lengths;
  for (const auto *e : edges->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    lengths[keyOf(e->getSource()->getId(), e->getTarget()->getId())] =
        e->getSquaredLength();
  }
  return lengths;
}

}  // namespace

std::vector<std::vector<std::uint64_t>> SlicedCobordism::topCells(
    const Spacetime &complex) {
  std::size_t widest = 0;
  for (const auto &s : complex.getSimplices())
    if (s != nullptr) widest = std::max(widest, s->size());

  std::vector<std::vector<std::uint64_t>> cells;
  if (widest == 0) return cells;
  for (const auto &s : complex.getSimplices()) {
    if (s == nullptr || s->size() != widest) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(widest);
    for (const auto &v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    if (ids.size() != widest) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::shared_ptr<Spacetime> SlicedCobordism::closedSlice() {
  // dDelta^4 = S^3: the five 4-subsets of {0..4}. Every triangle lies in exactly
  // two of them, so the complex is closed -- a spatial slice, not a 3-ball.
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(5);
  for (std::uint64_t dropped = 0; dropped < 5; ++dropped) {
    std::vector<std::uint64_t> cell;
    cell.reserve(4);
    for (std::uint64_t v = 0; v < 5; ++v)
      if (v != dropped) cell.push_back(v);
    cells.push_back(std::move(cell));
  }
  // Dimension 3 -- the slice is genuinely three-dimensional until it is coned.
  return Spacetime::fromCells(3, cells, 1.0, 0.0);
}

std::pair<std::shared_ptr<Spacetime>, std::string> SlicedCobordism::coneToBulk(
    const std::shared_ptr<Spacetime> &slice,
    std::complex<double> apexEdgeSquaredLength) {
  if (!slice) return {nullptr, "no slice"};

  const auto sliceCells = topCells(*slice);
  if (sliceCells.empty()) return {nullptr, "slice has no top cells"};
  const std::size_t sliceCellSize = sliceCells.front().size();
  if (sliceCellSize != 4)
    return {nullptr,
            "slice top cells are not tetrahedra (" +
                std::to_string(sliceCellSize) + " vertices, expected 4)"};

  // A closed slice is what cone(S^3) = D^4 needs: every triangle must lie in
  // exactly two tetrahedra. A slice with boundary would cone to a complex whose
  // boundary is not the slice, which is not the object this construction means.
  std::map<std::vector<std::uint64_t>, int> triangleCofaces;
  for (const auto &cell : sliceCells)
    for (std::size_t dropped = 0; dropped < cell.size(); ++dropped) {
      std::vector<std::uint64_t> face;
      face.reserve(cell.size() - 1);
      for (std::size_t i = 0; i < cell.size(); ++i)
        if (i != dropped) face.push_back(cell[i]);
      ++triangleCofaces[face];
    }
  for (const auto &[face, count] : triangleCofaces)
    if (count != 2) return {nullptr, "slice is not closed"};

  // One fresh apex shared by every cell: cone(S^3) = D^4. Per-tetrahedron
  // apexes would give only (4,1) cells, which leave gaps and do not tile.
  std::uint64_t apex = 0;
  for (const auto &cell : sliceCells)
    for (const std::uint64_t v : cell) apex = std::max(apex, v);
  ++apex;

  std::vector<std::vector<std::uint64_t>> bulkCells;
  bulkCells.reserve(sliceCells.size());
  for (const auto &cell : sliceCells) {
    std::vector<std::uint64_t> coned = cell;
    coned.push_back(apex);
    std::sort(coned.begin(), coned.end());
    bulkCells.push_back(std::move(coned));
  }

  // The SAME gate every surgical move applies, on the candidate cell set.
  const auto verdict = ChainComplex::dualComplexIsValid(bulkCells, 4);
  if (!verdict.first) return {nullptr, verdict.second};

  auto bulk = Spacetime::fromCells(4, bulkCells, 1.0, 0.0);
  if (!bulk) return {nullptr, "fromCells returned no complex"};

  // Carry the slice's spacelike lengths over unchanged, then write the apex
  // edges -- and only those. An edge of the bulk either lies wholly in the slice
  // (carried) or touches the apex (written); there is no third case.
  const auto sliceLengths = squaredLengthsOf(*slice);
  const auto &bulkEdges = bulk->getEdgeList();
  if (!bulkEdges) return {nullptr, "bulk has no edge list"};
  for (auto *e : bulkEdges->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    if (a == apex || b == apex) {
      e->setSquaredLength(apexEdgeSquaredLength);
      continue;
    }
    const auto it = sliceLengths.find(keyOf(a, b));
    if (it != sliceLengths.end()) e->setSquaredLength(it->second);
  }

  return {bulk, "ok"};
}

}  // namespace tessera::cobordism
