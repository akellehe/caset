// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/OrientedCone.h"

#include <algorithm>

#include "cobordism/ChainComplex.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/RemoveMove.h"

namespace tessera::cobordism {

OrientedCone::OrientedCone(Spacetime *spacetime) : st_(spacetime) {}

OrientedCone::~OrientedCone() = default;

std::vector<std::vector<std::uint64_t>> OrientedCone::topCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (st_ == nullptr) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  for (const auto &s : st_->getSimplices()) {
    if (s == nullptr) continue;
    if (s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto &v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    if (ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::pair<bool, std::string> OrientedCone::validate() const {
  const auto tops = topCells();
  if (tops.empty()) return {false, "no top cells"};
  const int dim = static_cast<int>(tops.front().size()) - 1;
  // 1. Manifold (the #429 check: pseudomanifold + ridge links + n>=4 recursive
  //    vertex-link validation).
  const auto manifold = ChainComplex::dualComplexIsValid(tops, dim);
  if (!manifold.first) return {false, "not a manifold: " + manifold.second};
  // 2. Orientability: a consistent global induced orientation must exist, or a
  //    cone could flip a local sign into Im S. orientationCovector throws on a
  //    non-pseudomanifold or non-orientable propagation contradiction.
  try {
    (void)ChainComplex::orientationCovector(tops);
  } catch (const std::exception &e) {
    return {false, std::string("orientation: ") + e.what()};
  }
  return {true, "ok"};
}

std::vector<int> OrientedCone::orientationCovector() const {
  return ChainComplex::orientationCovector(topCells());
}

std::pair<bool, std::string> OrientedCone::coneIn(std::uint64_t seed) {
  if (st_ == nullptr) return {false, "no spacetime"};
  if (applied_) return {false, "a cone is already applied; rollback first"};
  // Reuse the T1 cone primitive: a pre-geometric 1 -> (d+1) stellar subdivision,
  // no vertex relabelling (the new cells carry the standard orientation from
  // their vertex times — TemporalOrientation::orientationOf — by construction).
  auto move = std::make_unique<AddMove>(
      st_, seed, /*relabelEnabled=*/false, PachnerMode::PreGeometric,
      /*boundaryFixed=*/false);
  if (!move->propose()) return {false, "cone-in did not propose"};
  if (!move->apply()) return {false, "cone-in did not apply"};
  const auto verdict = validate();
  if (!verdict.first) {
    move->rollback();
    return verdict;
  }
  lastMove_ = std::move(move);
  applied_ = true;
  return {true, "ok"};
}

std::pair<bool, std::string> OrientedCone::coneOut(std::uint64_t seed) {
  if (st_ == nullptr) return {false, "no spacetime"};
  if (applied_) return {false, "a cone is already applied; rollback first"};
  auto move = std::make_unique<RemoveMove>(
      st_, seed, PachnerMode::PreGeometric, /*boundaryFixed=*/false);
  if (!move->propose()) return {false, "cone-out did not propose"};
  if (!move->apply()) return {false, "cone-out did not apply"};
  const auto verdict = validate();
  if (!verdict.first) {
    move->rollback();
    return verdict;
  }
  lastMove_ = std::move(move);
  applied_ = true;
  return {true, "ok"};
}

bool OrientedCone::rollback() {
  if (!applied_ || lastMove_ == nullptr) return false;
  lastMove_->rollback();
  lastMove_.reset();
  applied_ = false;
  return true;
}

}  // namespace tessera::cobordism
