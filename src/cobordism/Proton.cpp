// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Proton.h"

#include <cmath>
#include <limits>
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
               double gamma, double inputWeight)
    : baseSeed_(seed),
      hostRefinement_(hostRefinement),
      registerDegree_(registerDegree),
      gamma_(gamma),
      inputResidualWeight_(inputWeight) {}

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

void Proton::build(int maxRestarts, int constructRounds, int initSteps, int evolveSteps,
                   int stage1CandidateMoves, int stage1Patience, double stage2Beta,
                   int stage2MaxIters, double colorTolerance, int minQuarkHoles) {
  if (attempted_) return;
  attempted_ = true;

  const cd w = omega();
  // Step A inputs: two neutral q-q̄ pairs (Σ = 0). Step A outputs: a colored diquark
  // {1,ω} ⊔ antidiquark {1,ω²} (2-vectors — NOT the singlet).
  const std::vector<std::vector<cd>> pairsA = {
      {cd(1.0, 0.0), cd(-1.0, 0.0), cd(0.0, 0.0)},
      {cd(1.0, 0.0), cd(0.0, 0.0), cd(-1.0, 0.0)}};
  const std::vector<cd> diquark = {cd(1.0, 0.0), w};
  const std::vector<cd> antidiquark = {cd(1.0, 0.0), w * w};
  // Step B inputs: the diquark (2-vec) + the third quark (1-vec). Output: the proton.
  const std::vector<cd> thirdQuark = {w * w};
  const std::vector<cd> protonSinglet = singlet();

  // One node's run: weight the inputs, an INITIALIZATION pass that grows the boundary
  // regions until they carry (grow_boundaries=true), an EVOLUTION pass with ∂W frozen
  // (grow_boundaries=false), then the geometric relaxation. runStage1 self-recovers
  // from unproductive grow bursts internally (revert + reseed + retry), so one call
  // per pass is as robust as many short ones.
  const auto runNode = [&](MultiCobordism &node) {
    node.setInputResidualWeight(inputResidualWeight_);
    node.runStage1(initSteps, stage1CandidateMoves, stage1Patience, /*growBoundaries=*/true);
    node.runStage1(evolveSteps, stage1CandidateMoves, stage1Patience,
                   /*growBoundaries=*/false);
    node.runStage2(stage2Beta, stage2MaxIters);
  };

  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination (2 → 2): the diquark ⊔ antidiquark form. Run as a
    // best-effort validation; its r_U is reported (diquarkResidual), not gated. ----
    double diquarkR = std::numeric_limits<double>::quiet_NaN();
    auto hostA = buildClosedS4Host(hostRefinement_, seedA);
    auto vertsA = hostA->getVertexList()->toVector();
    if (vertsA.size() >= 4) {
      MultiCobordism stepA(hostA, pairsA, {diquark, antidiquark}, {registerDegree_},
                           gamma_, seedA);
      stepA.constructInputs({vertsA[0]->getId(), vertsA[1]->getId()}, constructRounds);
      stepA.constructOutputs({vertsA[2]->getId(), vertsA[3]->getId()}, constructRounds);
      runNode(stepA);
      diquarkR = stepA.rU(stepA.spacetime());
    }

    // ---- Step B — formation (2 → 1): the proton, read off the WHOLE cobordism ----
    auto hostB = buildClosedS4Host(hostRefinement_, seedB);
    auto vertsB = hostB->getVertexList()->toVector();
    if (vertsB.size() < 3) continue;
    MultiCobordism stepB(hostB, {diquark, thirdQuark}, {protonSinglet}, {registerDegree_},
                         gamma_, seedB);
    stepB.constructInputs({vertsB[0]->getId(), vertsB[1]->getId()}, constructRounds);
    runNode(stepB);  // no constructOutputs — the single output IS the whole

    // The proton is the harmonic of the WHOLE cobordism (the inputs are held by their
    // residual, the bulk evolves to carry the singlet). Read it off the relaxed whole.
    auto whole = stepB.spacetime();
    const double colorR = MultiCobordism::residualOfTargetStateAgainstHarmonic(
        whole, registerDegree_, protonSinglet);
    auto holes = MultiCobordism::emergentHoles(*whole, registerDegree_);
    const bool ok = colorR < colorTolerance &&
                    static_cast<int>(holes.size()) >= minQuarkHoles;

    // Keep the converged attempt, or the lowest-residual one so far otherwise.
    if (ok || colorR < bestColorResidual) {
      bestColorResidual = colorR;
      converged_ = ok;
      convergedSeed_ = seedA;
      spacetime_ = whole;
      block_ = whole;  // the proton IS the whole cobordism (read off the whole)
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
