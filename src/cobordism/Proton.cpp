// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Proton.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <optional>
#include <utility>

#include "cobordism/MultiCobordism.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/topologies/SimplexBoundarySphere.h"

namespace tessera::cobordism {

namespace {
using cd = std::complex<double>;

// Sorted vertex-id tuple of a top simplex (the codebase's vertex-set convention;
// never sorted to impose a labeling — only to key the cell set).
std::vector<std::uint64_t> topTuple(const ::tessera::mesh::Simplex &s) {
  std::vector<std::uint64_t> ids;
  for (const auto &v : s.getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}
}  // namespace

std::shared_ptr<Spacetime> Proton::buildHost(int nRefine, std::uint64_t seed) {
  using namespace ::tessera::spacetime;
  Signature sig(4, SignatureType::Lorentzian);
  auto metric = std::make_shared<Metric>(true, sig);
  auto topo = std::make_shared<SimplexBoundarySphere>(4);
  auto st = std::make_shared<Spacetime>(
      metric, SpacetimeType::CDT, std::optional<double>(1.0),
      std::optional<double>(1.0), Foliation::PREFERRED,
      std::optional<std::shared_ptr<Topology>>(topo));
  st->build();
  // Bare host: unit spacelike edges to start; NO register/holes are placed.
  for (const auto &e : st->getEdgeList()->toVector()) e->setSquaredLength(cd(1.0));
  // Refine for surgery room (PreGeometric add-moves grow volume, not topology).
  int applied = 0;
  for (std::uint64_t s = seed;
       s < seed + static_cast<std::uint64_t>(nRefine) * 4; ++s) {
    AddMove mv(st.get(), s, /*relabel=*/false, PachnerMode::PreGeometric,
               /*boundaryFixed=*/false);
    if (mv.propose() && mv.apply()) ++applied;
    if (applied >= nRefine) break;
  }
  int i = 0;
  for (const auto &e : st->getEdgeList()->toVector())
    e->setSquaredLength(cd(1.0 + 0.01 * (i++ % 6)));
  return st;
}

std::shared_ptr<Spacetime> Proton::carveBlock(
    const std::shared_ptr<Spacetime> &st, const std::set<std::uint64_t> &verts) {
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &s : st->getTopSimplices()) {
    auto c = topTuple(*s);
    bool inside = true;
    for (auto id : c)
      if (!verts.count(id)) {
        inside = false;
        break;
      }
    if (inside) cells.push_back(std::move(c));
  }
  if (cells.size() < 2) return nullptr;
  auto sub = Spacetime::fromCells(4, cells, 1.0, 0.0);
  // Copy the RELAXED metric (a unit-metric rebuild would make downstream
  // geometric reads degenerate).
  std::map<std::pair<std::uint64_t, std::uint64_t>, cd> parent;
  for (const auto &e : st->getEdgeList()->toVector()) {
    auto a = e->getSource()->getId(), b = e->getTarget()->getId();
    parent[{std::min(a, b), std::max(a, b)}] = e->getSquaredLength();
  }
  for (const auto &e : sub->getEdgeList()->toVector()) {
    auto a = e->getSource()->getId(), b = e->getTarget()->getId();
    auto it = parent.find({std::min(a, b), std::max(a, b)});
    if (it != parent.end()) e->setSquaredLength(it->second);
  }
  sub->materializeFacets();
  return sub;
}

Proton::Proton(int nAttempts, int stage1Steps, int stage2Steps, int nRefine,
               double gamma, std::uint64_t seed0) {
  const cd w = std::exp(cd(0.0, 2.0 * M_PI / 3.0));
  // Two neutral q-qbar pairs in; a coloured diquark + antidiquark out (2-vectors).
  const std::vector<std::vector<cd>> pairs = {{cd(1), cd(-1), cd(0)},
                                              {cd(1), cd(0), cd(-1)}};
  const std::vector<cd> diquark = {cd(1), w};
  const std::vector<cd> antidiquark = {cd(1), w * w};
  // The third quark (1-vector) and the proton colour singlet (3-vector).
  const std::vector<cd> quark3 = {w * w};
  const std::vector<cd> proton = {cd(1), w, w * w};
  const std::vector<int> kDeg = {3};

  auto firstIds = [](const std::shared_ptr<Spacetime> &host, std::size_t lo,
                     std::size_t hi) {
    std::vector<std::uint64_t> out;
    const auto verts = host->getVertexList()->toVector();
    for (std::size_t i = lo; i < hi && i < verts.size(); ++i)
      out.push_back(verts[i]->getId());
    return out;
  };

  for (int a = 0; a < nAttempts; ++a) {
    attemptsUsed_ = a + 1;
    seed_ = seed0 + static_cast<std::uint64_t>(a);

    // --- Step A: recombination (pairs -> diquark + antidiquark); confirms the
    // diquark forms. The diquark STATE feeds step B as an input target. ---
    auto hostA = buildHost(nRefine, seed_);
    MultiCobordism A(hostA, pairs, {diquark, antidiquark}, kDeg, gamma, seed_);
    A.constructInputs(firstIds(hostA, 0, 2), 24);
    A.constructOutputs(firstIds(hostA, 2, 4), 24);
    A.runStage1(stage1Steps, 10, 12);
    A.relaxStage2(1.0, stage2Steps, 0.05);
    diquarkResidual_ = A.rU(A.spacetime());

    // --- Step B: formation (diquark + third quark -> proton singlet). The
    // proton output block is the proton "state". ---
    auto hostB = buildHost(nRefine, seed_ + 1000);
    MultiCobordism B(hostB, {diquark, quark3}, {proton}, kDeg, gamma,
                     seed_ + 1000);
    B.constructInputs(firstIds(hostB, 0, 2), 24);
    B.constructOutputs(firstIds(hostB, 2, 3), 24);
    B.runStage1(stage1Steps, 10, 12);
    B.relaxStage2(1.0, stage2Steps, 0.05);

    if (B.outputs().empty()) continue;
    auto block = carveBlock(B.spacetime(), B.outputs().front().verts);
    if (!block) continue;
    auto holes = MultiCobordism::emergentHoles(*block, 3);
    const double res = MultiCobordism::rState(block, 3, proton);

    // Record the latest attempt (so a non-converged Proton is still inspectable).
    st_ = B.spacetime();
    block_ = block;
    holes_ = holes;
    colorResidual_ = res;

    // Converged: the emergent proton block carries the colour singlet on >= 3
    // emergent quark holes.
    if (holes.size() >= 3 && res < 1.0) {
      converged_ = true;
      return;
    }
  }
}

}  // namespace tessera::cobordism
