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
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {
constexpr int kDim = 4;  // framework dimension; the seed is a single Δ⁴ simplex
}  // namespace

std::complex<double> Proton::omega() {
  // ω = exp(2πi/3); std::polar(1, θ) = cos θ + i sin θ = e^{iθ}.
  return std::polar(1.0, 2.0 * std::acos(-1.0) / 3.0);
}

std::vector<std::complex<double>> Proton::singlet() {
  const complexd w = omega();
  return {complexd(1.0, 0.0), w, w * w};
}

Proton::Proton(std::uint64_t seed, int registerDegree, double gamma,
               double inputWeight, int precone, bool shouldUseDirectedSurgery,
               bool preconeTimelike, bool preconeAlternate, bool balancedEdges)
    : baseSeed_(seed),
      registerDegree_(registerDegree),
      gamma_(gamma),
      inputResidualWeight_(inputWeight),
      precone_(precone),
      shouldUseDirectedSurgery_(shouldUseDirectedSurgery),
      preconeTimelike_(preconeTimelike),
      preconeAlternate_(preconeAlternate) {
  balancedEdges_ = balancedEdges;
}

std::shared_ptr<Spacetime> Proton::buildMinimalSeed(bool balancedEdges) {
  using namespace ::tessera::spacetime;
  auto metric =
      std::make_shared<Metric>(true, Signature(kDim, SignatureType::Lorentzian));
  // A SINGLE Δ⁴ simplex (one pentatope, 5 vertices) — the most minimal seed there is.
  // Nothing is pre-built: the proton's entire topology emerges from here via the trap
  // door, and the metric is uniform (ℓ² = 1) so the geometry emerges from the
  // relaxation too. Only the seed simplex and the target color states are imposed.
  std::shared_ptr<Topology> topology = std::make_shared<SolidSimplex>(kDim);
  auto host = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0,
                                          Foliation::PREFERRED, topology);
  host->build();
  // #690: the wiring mode is stamped before ANY growth, and the seed's own
  // uniform |l^2| = 1 edges honor it too (balanced: l = sqrt(1/2)*(1+i)).
  host->setBalancedEdgeWiring(balancedEdges);
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setLength(balancedEdges
                        ? ::tessera::spacetime::Spacetime::balancedLength(1.0)
                        : std::sqrt(complexd(1.0, 0.0)));
  return host;
}

std::shared_ptr<MultiCobordism> Proton::recombinationNode(std::uint64_t seed) const {
  // Step A inputs: two neutral q-q̄ pairs (Σ = 0). Outputs: a colored diquark {1,ω} ⊔
  // antidiquark {1,ω²} (2-vectors — NOT the singlet). Seeded on a fresh single-Δ⁴ seed,
  // inputs at v0,v1 and outputs at v2,v3; NOT run (the caller drives it).
  const complexd w = omega();
  const std::vector<std::vector<complexd>> pairs = {
      {complexd(1.0, 0.0), complexd(-1.0, 0.0), complexd(0.0, 0.0)},
      {complexd(1.0, 0.0), complexd(0.0, 0.0), complexd(-1.0, 0.0)}};
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> antidiquark = {complexd(1.0, 0.0), w * w};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex IDS (not Vertex*) BEFORE constructing the node: with
  // precone_ > 0 the ctor regrows spacetime_ into a fresh complex, destroying the
  // original host's Vertex objects — but the seed ids persist through the rebuilds
  // (build() preserves vertex ids), so the input/output anchors stay valid.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, pairs, std::vector<std::vector<complexd>>{diquark, antidiquark},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  node->seedOutputs({seedVertexIds[2], seedVertexIds[3]});
  return node;
}

std::shared_ptr<MultiCobordism> Proton::formationNode(std::uint64_t seed) const {
  // Step B inputs: the diquark {1,ω} + the third quark {ω²}. Output: the proton singlet,
  // read off the WHOLE cobordism (no seedOutputs). Seeded on a fresh single-Δ⁴ seed,
  // inputs at v0,v1; NOT run (the caller drives it).
  const complexd w = omega();
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> thirdQuark = {w * w};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex IDS before constructing the node (see recombinationNode):
  // precone_ > 0 regrows the complex in the ctor, but the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, std::vector<std::vector<complexd>>{diquark, thirdQuark},
      std::vector<std::vector<complexd>>{singlet()},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  return node;
}

std::shared_ptr<MultiCobordism> Proton::directNode(std::uint64_t seed) const {
  // One-step inputs: the three bare quarks {1}, {ω}, {ω²} AND their three
  // anti-quarks — the elementwise conjugates {1}, {ω̄}, {ω̄²} (conjugation is the
  // antiparticle convention here: the antidiquark {1, ω²} is exactly the conjugate
  // of the diquark {1, ω}) — so the prepared content is three q-q̄ pairs, not three
  // quarks from nothing. Output: the proton singlet, read off the WHOLE cobordism
  // (no seedOutputs, as formationNode) — the anti-baryon partner is left to emerge
  // unpinned. Seeded on a fresh single-Δ⁴ seed; NOT run (the caller drives it).
  const complexd w = omega();
  const std::vector<std::vector<complexd>> quarksAndAntiquarks = {
      {complexd(1.0, 0.0)}, {w}, {w * w},
      {complexd(1.0, 0.0)}, {std::conj(w)}, {std::conj(w * w)}};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex IDS before constructing the node (see recombinationNode):
  // precone_ > 0 regrows the complex in the ctor, but the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, quarksAndAntiquarks, std::vector<std::vector<complexd>>{singlet()},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_);
  node->setInputResidualWeight(inputResidualWeight_);
  // Six blocks on a 5-vertex Δ⁴ seed: the anchors cycle. On the bare seed every
  // block's region is the seed's full cell-neighbourhood regardless — the anchor
  // only distinguishes one block from another — and the blocks differentiate as
  // the gated growth takes each region where its own residual wants it.
  std::vector<std::uint64_t> inputSeedVertexIds;
  for (std::size_t blockIndex = 0; blockIndex < quarksAndAntiquarks.size();
       ++blockIndex)
    inputSeedVertexIds.push_back(
        seedVertexIds[blockIndex % seedVertexIds.size()]);
  node->seedInputs(inputSeedVertexIds);
  return node;
}

void Proton::buildDirect(int maxRestarts, int initSteps, int evolveSteps,
                         int stage1CandidateMoves, double stage2Beta,
                         double colorTolerance, int minQuarkHoles) {
  if (attempted_) return;
  attempted_ = true;

  const std::vector<complexd> protonSinglet = singlet();
  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seed = baseSeed_ + static_cast<std::uint64_t>(attempt);
    auto node = directNode(seed);
    // The combined drive: every `run` iteration interleaves the stage-1 surgery
    // update with the stage-2 geometric relaxation, so the optimizer takes whichever
    // kind of progress helps at each point — an init pass growing the input regions
    // until they carry, then an evolution pass with ∂W frozen. No separate
    // relaxation pass: it is folded into every iteration.
    node->run(initSteps, stage1CandidateMoves, /*growBoundaries=*/true, stage2Beta);
    if (shouldUseDirectedSurgery_)  // deliberately open the register holes
      (void)node->directedConeOut();
    node->run(evolveSteps, stage1CandidateMoves, /*growBoundaries=*/false, stage2Beta);
    if (shouldUseDirectedSurgery_)  // select the best register (drop holes that hurt)
      (void)node->directedConeIn();

    auto whole = node->spacetime();
    const double colorR = MultiCobordism::residualOfTargetStateAgainstHarmonic(
        whole, registerDegree_, protonSinglet);
    auto holes = MultiCobordism::emergentHoles(*whole, registerDegree_);
    const bool ok = colorR < colorTolerance &&
                    static_cast<int>(holes.size()) >= minQuarkHoles;

    // Keep the converged attempt, or the lowest-residual one so far otherwise.
    if (ok || colorR < bestColorResidual) {
      bestColorResidual = colorR;
      converged_ = ok;
      convergedSeed_ = seed;
      spacetime_ = whole;
      block_ = whole;  // the proton IS the whole cobordism (read off the whole)
      quarkHoles_ = std::move(holes);
      colorResidual_ = colorR;
      diquarkResidual_ = 0.0;  // no step A in the one-step build
    }
    if (ok) return;  // a proton emerged — stop restarting
  }
}

void Proton::build(int maxRestarts, int initSteps, int evolveSteps,
                   int stage1CandidateMoves, double stage2Beta,
                   int stage2MaxIters, double colorTolerance, int minQuarkHoles) {
  if (attempted_) return;
  attempted_ = true;

  const std::vector<complexd> protonSinglet = singlet();

  // Drive one already-seeded node: an INITIALIZATION pass that grows the boundary
  // regions until they carry (grow_boundaries=true), an EVOLUTION pass with ∂W frozen
  // (grow_boundaries=false), then the geometric relaxation. (Node setup —
  // seed, targets, seeding, input weight — lives in recombinationNode/formationNode, the
  // same factories the animation drives.)
  const auto runNode = [&](MultiCobordism &node) {
    node.runStage1(initSteps, stage1CandidateMoves, /*growBoundaries=*/true);
    if (shouldUseDirectedSurgery_)  // deliberately open the register holes
      (void)node.directedConeOut();
    node.runStage1(evolveSteps, stage1CandidateMoves,
                   /*growBoundaries=*/false);
    if (shouldUseDirectedSurgery_)  // select the best register (drop holes that hurt)
      (void)node.directedConeIn();
    node.runStage2(stage2Beta, stage2MaxIters);
  };

  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination: best-effort; its r_U is reported, not gated. ----
    auto stepA = recombinationNode(seedA);
    runNode(*stepA);
    const double diquarkR = stepA->rU(stepA->spacetime());

    // ---- Step B — formation: the proton, read off the WHOLE cobordism ----
    auto stepB = formationNode(seedB);
    runNode(*stepB);
    auto whole = stepB->spacetime();
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
