// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Proton.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

namespace {
constexpr int kDim = 4;  // the closed S^4 host is a 4-manifold

// Sorted vertex-id tuple of a top simplex (mirrors MultiCobordism's `_top_tuple`).
std::vector<std::uint64_t> topTuple(const ::tessera::mesh::Simplex &simplex) {
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : simplex.getVertices()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::pair<std::uint64_t, std::uint64_t> edgeKey(const ::tessera::mesh::Edge *edge) {
  const auto a = edge->getSource()->getId();
  const auto b = edge->getTarget()->getId();
  return {std::min(a, b), std::max(a, b)};
}
}  // namespace

std::complex<double> Proton::omega() {
  // ω = exp(2πi/3); std::polar(1, θ) = cos θ + i sin θ = e^{iθ}.
  return std::polar(1.0, 2.0 * std::acos(-1.0) / 3.0);
}

std::vector<std::complex<double>> Proton::singlet() {
  const cd w = omega();
  return {cd(1.0, 0.0), w, w * w};
}

Proton::Proton(std::uint64_t seed, int hostRefinement, int registerDegree,
               double gamma)
    : baseSeed_(seed),
      hostRefinement_(hostRefinement),
      registerDegree_(registerDegree),
      gamma_(gamma) {}

std::shared_ptr<Spacetime> Proton::buildClosedS4Host(int nRefine,
                                                     std::uint64_t seed) {
  using namespace ::tessera::spacetime;
  auto metric =
      std::make_shared<Metric>(true, Signature(kDim, SignatureType::Lorentzian));
  std::shared_ptr<Topology> topology = std::make_shared<SimplexBoundarySphere>(kDim);
  auto host = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0,
                                          Foliation::PREFERRED, topology);
  host->build();
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setSquaredLength(cd(1.0, 0.0));
  // Refine the minimal ∂Δ⁵ by `nRefine` accepted PreGeometric Pachner adds so
  // surgery has room to act. Try up to 4× as many seeds as moves needed.
  int applied = 0;
  for (std::uint64_t s = seed;
       s < seed + static_cast<std::uint64_t>(nRefine) * 4; ++s) {
    AddMove move(host.get(), s, /*relabelEnabled=*/false,
                 PachnerMode::PreGeometric, /*boundaryFixed=*/false);
    if (move.propose() && move.apply()) ++applied;
    if (applied >= nRefine) break;
  }
  // A mild, deterministic non-uniform metric (matches build_closed_s4): breaks
  // exact degeneracy so the relaxation has a gradient to follow.
  int i = 0;
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setSquaredLength(cd(1.0 + 0.01 * (i++ % 6), 0.0));
  return host;
}

std::shared_ptr<Spacetime> Proton::carveRelaxedSubcomplex(
    const std::shared_ptr<Spacetime> &full,
    const std::set<std::uint64_t> &vertexSet) {
  std::vector<std::vector<std::uint64_t>> cellsInside;
  for (const auto &topSimplex : full->getTopSimplices()) {
    auto ids = topTuple(*topSimplex);
    bool inside = true;
    for (auto id : ids)
      if (!vertexSet.count(id)) {
        inside = false;
        break;
      }
    if (inside) cellsInside.push_back(std::move(ids));
  }
  if (cellsInside.empty()) return nullptr;
  auto sub = Spacetime::fromCells(kDim, cellsInside, 1.0, 0.0);
  // Copy the relaxed edge lengths in (the whole point — `fromCells` lays a unit
  // metric, which would discard the geometry the optimizer found).
  std::map<std::pair<std::uint64_t, std::uint64_t>, cd> relaxedLength;
  for (auto *edge : full->getEdgeList()->toVector())
    relaxedLength[edgeKey(edge)] = edge->getSquaredLength();
  for (auto *edge : sub->getEdgeList()->toVector()) {
    const auto found = relaxedLength.find(edgeKey(edge));
    if (found != relaxedLength.end()) edge->setSquaredLength(found->second);
  }
  return sub;
}

void Proton::build(int maxRestarts, int constructRounds, int stage1MaxSteps,
                   int stage1Candidates, int stage1Patience, double stage2Beta,
                   int stage2MaxIters, double colorTolerance, int minQuarkHoles) {
  if (attempted_) return;
  attempted_ = true;

  const cd w = omega();
  // Step A inputs: two neutral q-q̄ pairs (Σ = 0, so they carry). Step A outputs:
  // a colored diquark and antidiquark (2-vectors — NOT the singlet).
  const std::vector<std::vector<cd>> pairsA = {
      {cd(1.0, 0.0), cd(-1.0, 0.0), cd(0.0, 0.0)},
      {cd(1.0, 0.0), cd(0.0, 0.0), cd(-1.0, 0.0)}};
  const std::vector<cd> diquark = {cd(1.0, 0.0), w};
  const std::vector<cd> antidiquark = {cd(1.0, 0.0), w * w};
  // Step B inputs: the diquark (2-vec) + the third quark (1-vec). Output: the
  // proton (3-vec color singlet).
  const std::vector<cd> thirdQuark = {w * w};
  const std::vector<cd> protonSinglet = singlet();

  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination (2 → 2): diquark ⊔ antidiquark ----
    auto hostA = buildClosedS4Host(hostRefinement_, seedA);
    auto vertsA = hostA->getVertexList()->toVector();
    if (vertsA.size() < 4) continue;  // need 4 distinct construction seeds
    MultiCobordism stepA(hostA, pairsA, {diquark, antidiquark},
                         {registerDegree_}, gamma_, seedA);
    stepA.constructInputs({vertsA[0]->getId(), vertsA[1]->getId()}, constructRounds);
    stepA.constructOutputs({vertsA[2]->getId(), vertsA[3]->getId()}, constructRounds);
    stepA.runStage1(stage1MaxSteps, stage1Candidates, stage1Patience);
    stepA.runStage2(stage2Beta, stage2MaxIters);
    const double diquarkR = stepA.rU(stepA.spacetime());

    // ---- Step B — formation (2 → 1): the proton singlet ----
    auto hostB = buildClosedS4Host(hostRefinement_, seedB);
    auto vertsB = hostB->getVertexList()->toVector();
    if (vertsB.size() < 3) continue;
    MultiCobordism stepB(hostB, {diquark, thirdQuark}, {protonSinglet},
                         {registerDegree_}, gamma_, seedB);
    stepB.constructInputs({vertsB[0]->getId(), vertsB[1]->getId()}, constructRounds);
    stepB.constructOutputs({vertsB[2]->getId()}, constructRounds);
    stepB.runStage1(stage1MaxSteps, stage1Candidates, stage1Patience);
    stepB.runStage2(stage2Beta, stage2MaxIters);

    // ---- read the proton off step B's output block, with the relaxed metric ----
    auto relaxedB = stepB.spacetime();
    std::shared_ptr<Spacetime> block;
    if (!stepB.outputs().empty())
      block = carveRelaxedSubcomplex(relaxedB, stepB.outputs().front().vertices);

    double colorR;
    std::vector<std::vector<std::uint64_t>> holes;
    if (block) {
      colorR = MultiCobordism::residualOfTargetStateAgainstHarmonic(
          block, registerDegree_, protonSinglet);
      holes = MultiCobordism::emergentHoles(*block, registerDegree_);
    } else {
      colorR = 0.0;  // no sub-complex yet → full zero-filled leak ‖target‖²
      for (const auto &amplitude : protonSinglet) colorR += std::norm(amplitude);
    }

    const bool ok = static_cast<bool>(block) && colorR < colorTolerance &&
                    static_cast<int>(holes.size()) >= minQuarkHoles;

    // Keep the converged attempt, or the lowest-residual one so far otherwise —
    // the accessors always have a best-effort proton to return.
    if (ok || colorR < bestColorResidual) {
      bestColorResidual = colorR;
      converged_ = ok;
      convergedSeed_ = seedA;
      spacetime_ = relaxedB;
      block_ = block;
      quarkHoles_ = std::move(holes);
      colorResidual_ = colorR;
      diquarkResidual_ = diquarkR;
    }
    if (ok) return;  // a proton emerged — stop restarting
  }
}

void Proton::ensureBuilt() {
  if (!attempted_) build();
}

bool Proton::converged() {
  ensureBuilt();
  return converged_;
}

std::uint64_t Proton::seed() {
  ensureBuilt();
  return convergedSeed_;
}

std::shared_ptr<Spacetime> Proton::spacetime() {
  ensureBuilt();
  return spacetime_;
}

std::shared_ptr<Spacetime> Proton::block() {
  ensureBuilt();
  return block_;
}

std::vector<std::vector<std::uint64_t>> Proton::quarkHoles() {
  ensureBuilt();
  return quarkHoles_;
}

double Proton::colorResidual() {
  ensureBuilt();
  return colorResidual_;
}

double Proton::diquarkResidual() {
  ensureBuilt();
  return diquarkResidual_;
}

}  // namespace tessera::cobordism
