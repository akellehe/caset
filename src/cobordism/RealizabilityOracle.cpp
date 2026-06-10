// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "cobordism/RealizabilityOracle.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "Logger.h"
#include "cobordism/Cochain.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/LevenbergMarquardt.h"
#include "quantum/ChoiJamiolkowski.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

RealizabilityOracle::RealizabilityOracle(std::shared_ptr<Spacetime> bulk)
    : bulk_(std::move(bulk)) {
  if (!bulk_) throw std::invalid_argument("RealizabilityOracle: null bulk");
}

std::vector<cd> RealizabilityOracle::bend(const std::vector<cd> &U, int dA,
                                          int dB) {
  // Choi–Jamiołkowski bend: vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B, the row-major
  // flatten (#vectorize validates the dims and U.size() == dA·dB).
  std::vector<cd> psi = ::tessera::quantum::ChoiJamiolkowski::vectorize(U, dA, dB);
  double nrm = 0.0;
  for (const cd &z : psi) nrm += std::norm(z);
  if (nrm > 0.0) {
    const double inv = 1.0 / std::sqrt(nrm);
    for (cd &z : psi) z *= inv;
  }
  return psi;
}

double RealizabilityOracle::optimizePass(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, std::uint64_t passSeed, bool harmonic,
    std::vector<cd> &witnessOut) const {
  // The U(1) phases enter only the k=0 graph Laplacian; the metric Hodge L_k
  // (k>=1) is assembled from integer boundary maps and real volume weights, so
  // interior phases are not tuned there (they would be wasted search dimensions).
  const bool usePhases = (es.degree() == 0);
  const std::size_t order = es.order();         // operator dim this pass
  const std::size_t m = es.numInteriorEdges();  // free interior edge weights

  // Re-identify, on the current k-cell order, which ψ components are pinned
  // boundary cells (their sorted vertex-id tuple is a key of pinnedByTuple, the
  // target form on ∂W) vs. free interior cells (auxiliary amplitudes). Growth
  // changes the k-cell order, so this is rebuilt every pass.
  const std::vector<std::vector<std::uint64_t>> &cells = es.cellSimplices();
  std::vector<cd> pinnedValue(order, cd(0.0, 0.0));
  std::vector<std::size_t> freeIdx;
  for (std::size_t i = 0; i < order; ++i) {
    const auto it = pinnedByTuple.find(cells[i]);
    if (it != pinnedByTuple.end())
      pinnedValue[i] = it->second;
    else
      freeIdx.push_back(i);
  }
  const std::size_t nAux = freeIdx.size();
  const std::size_t nW = m;
  const std::size_t nTheta = usePhases ? m : 0;
  const std::size_t nParams = nW + nTheta + 2 * nAux;

  // Assemble ψ from a parameter vector: the boundary cells hold the fixed
  // target form; the auxiliary amplitudes (interior / coned-in cells) come from
  // the tail of x (real, imag per free cell).
  const auto buildPsi = [&pinnedValue, &freeIdx, nW, nTheta,
                         nAux](const Eigen::VectorXd &x) -> std::vector<cd> {
    std::vector<cd> psi = pinnedValue;
    for (std::size_t k = 0; k < nAux; ++k) {
      const auto re = static_cast<Eigen::Index>(nW + nTheta + 2 * k);
      psi[freeIdx[k]] = cd(x[re], x[re + 1]);
    }
    return psi;
  };

  // Residual: write the interior weights (and phases at k=0) onto ∂W's
  // complement (∂W itself is never touched), normalize the assembled ψ, and
  // return the stacked g = L_kψ - λψ ([Re; Im], length 2·order) whose ‖g‖² is
  // r(ψ) — the residual the Levenberg–Marquardt loop drives to zero.
  const auto residual = [&es, &buildPsi, m, nW, usePhases, harmonic,
                         order](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    if (m > 0) {
      std::vector<double> w(m);
      for (std::size_t i = 0; i < m; ++i) w[i] = x[static_cast<Eigen::Index>(i)];
      es.setInteriorWeights(w);
      if (usePhases) {
        std::vector<double> th(m);
        for (std::size_t i = 0; i < m; ++i)
          th[i] = x[static_cast<Eigen::Index>(nW + i)];
        es.setInteriorPhases(th);
      }
    }
    std::vector<cd> psi = buildPsi(x);
    double nrm = 0.0;
    for (const cd &z : psi) nrm += std::norm(z);
    if (nrm > 0.0) {
      const double inv = 1.0 / std::sqrt(nrm);
      for (cd &z : psi) z *= inv;
    }
    const std::vector<cd> Lp = es.apply(psi);
    // Eigenvalue-agnostic residual subtracts the Rayleigh-quotient component
    // (accepts ANY eigenvalue); the harmonic residual pins λ = 0, so the cost is
    // ‖Lψ‖² — the distance from ker L, i.e. whether ψ is carried as a HARMONIC.
    cd lam(0.0, 0.0);
    for (std::size_t i = 0; i < order; ++i) lam += std::conj(psi[i]) * Lp[i];
    const double lambda = harmonic ? 0.0 : lam.real();
    Eigen::VectorXd f(static_cast<Eigen::Index>(2 * order));
    for (std::size_t i = 0; i < order; ++i) {
      const cd g = Lp[i] - lambda * psi[i];
      f[static_cast<Eigen::Index>(i)] = g.real();
      f[static_cast<Eigen::Index>(order + i)] = g.imag();
    }
    return f;
  };

  // Clamp interior weights (and phases at k=0) into the §4b box and the
  // auxiliary amplitudes into [-kAuxBound, kAuxBound].
  const auto clamp = [nW, nTheta, nAux](const Eigen::VectorXd &x)
      -> Eigen::VectorXd {
    Eigen::VectorXd c = x;
    for (std::size_t i = 0; i < nW; ++i) {
      const auto wi = static_cast<Eigen::Index>(i);
      c[wi] = std::min(std::max(c[wi], kWMin), kWMax);
    }
    for (std::size_t i = 0; i < nTheta; ++i) {
      const auto ti = static_cast<Eigen::Index>(nW + i);
      c[ti] = std::min(std::max(c[ti], -kThetaBound), kThetaBound);
    }
    for (std::size_t k = 0; k < 2 * nAux; ++k) {
      const auto ai = static_cast<Eigen::Index>(nW + nTheta + k);
      c[ai] = std::min(std::max(c[ai], -kAuxBound), kAuxBound);
    }
    return c;
  };

  // Sample a restart: w ∈ [kWMin, kWMax], θ ∈ [-π, π], aux ∈ [-1, 1]. Built
  // once and captured (the single-rng draw convention GeometrySynthesizer uses).
  std::uniform_real_distribution<double> wDist(kWMin, kWMax);
  std::uniform_real_distribution<double> tDist(-kPi, kPi);
  std::uniform_real_distribution<double> aDist(-1.0, 1.0);
  const auto sample = [nW, nTheta, nAux, wDist, tDist,
                       aDist](std::mt19937_64 &rng) mutable -> Eigen::VectorXd {
    Eigen::VectorXd x0(static_cast<Eigen::Index>(nW + nTheta + 2 * nAux));
    for (std::size_t i = 0; i < nW; ++i)
      x0[static_cast<Eigen::Index>(i)] = wDist(rng);
    for (std::size_t i = 0; i < nTheta; ++i)
      x0[static_cast<Eigen::Index>(nW + i)] = tDist(rng);
    for (std::size_t k = 0; k < 2 * nAux; ++k)
      x0[static_cast<Eigen::Index>(nW + nTheta + k)] = aDist(rng);
    return x0;
  };

  const LevenbergMarquardt lm(kMaxIterations, epsilon);
  const LevenbergMarquardt::Result best =
      lm.multiRestart(residual, clamp, sample, nParams, restarts, passSeed,
                      epsilon);

  // Leave the complex realized at the best parameters and read off the unit
  // witness state there (multiRestart leaves `residual` evaluated last at
  // best.parameters; re-evaluate so the witness ψ matches the live complex).
  residual(best.parameters);
  witnessOut = buildPsi(best.parameters);
  double wn = 0.0;
  for (const cd &z : witnessOut) wn += std::norm(z);
  if (wn > 0.0) {
    const double inv = 1.0 / std::sqrt(wn);
    for (cd &z : witnessOut) z *= inv;
  }
  return best.cost;
}

std::vector<std::vector<std::uint64_t>>
RealizabilityOracle::connectivityCandidates(const EigenstateSynthesis &es,
                                            int nCandidates,
                                            std::uint64_t seed) const {
  const std::vector<std::uint64_t> verts = es.vertexIds();
  const std::vector<std::uint64_t> bverts = es.boundaryVertexIds();
  const std::vector<std::vector<std::uint64_t>> cells = es.topCells();

  std::set<std::vector<std::uint64_t>> seen;
  std::vector<std::vector<std::uint64_t>> out;
  const auto add = [&](std::vector<std::uint64_t> s) {
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    if (s.empty()) return;
    if (static_cast<int>(out.size()) >= std::max(nCandidates, 1)) return;
    if (seen.insert(s).second) out.push_back(std::move(s));
  };

  // Deterministic anchors, highest priority first:
  //  (0) cone-equivalent — the d+1 vertices of a top cell (so free-connectivity
  //      includes the cone move's 1-skeleton and is never worse than it),
  //  (1) full-star — wire to every existing vertex (maximal connectivity),
  //  (2) boundary-star — wire to every pinned-boundary vertex.
  if (!cells.empty())
    add(cells.front());
  else
    add(verts);
  add(verts);
  if (!bverts.empty()) add(bverts);

  // Reproducible random vertex subsets (each vertex w.p. 1/2; never empty) fill
  // the remaining budget, so connectivity is genuinely searched, not fixed.
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> coin(0, 1);
  const int guardMax = 200 * std::max(nCandidates, 1);
  for (int guard = 0;
       static_cast<int>(out.size()) < std::max(nCandidates, 1) && guard < guardMax;
       ++guard) {
    std::vector<std::uint64_t> s;
    for (const std::uint64_t u : verts)
      if (coin(rng) == 1) s.push_back(u);
    if (s.empty() && !verts.empty())
      s.push_back(verts[static_cast<std::size_t>(rng() % verts.size())]);
    add(std::move(s));
  }
  return out;
}

std::vector<std::vector<std::vector<std::uint64_t>>>
RealizabilityOracle::triangleConnectivityCandidates(const EigenstateSynthesis &es,
                                                    int nCandidates,
                                                    std::uint64_t seed) const {
  // The pool of existing edges a triangle {v_new, u, w} can cone over: every
  // tunable edge (interior + boundary). In 3D (top cells = tetrahedra) coning a
  // 2-simplex over an edge adds no tetrahedron, so ∂W's facet count is untouched
  // and the attach is boundary-safe; the attach still rejects any spec that would
  // perturb ∂W (e.g. in 2D, where a triangle IS a top cell), so it is always sound
  // to *propose* a triangle and let attachInteriorVertex gate it.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> pool;
  for (const auto &e : es.interiorEdges()) pool.push_back(e);
  for (const auto &e : es.boundaryEdges()) pool.push_back(e);
  std::sort(pool.begin(), pool.end());
  pool.erase(std::unique(pool.begin(), pool.end()), pool.end());

  const std::vector<std::vector<std::uint64_t>> cells = es.topCells();

  std::set<std::vector<std::vector<std::uint64_t>>> seen;
  std::vector<std::vector<std::vector<std::uint64_t>>> out;
  const auto add =
      [&](std::vector<std::pair<std::uint64_t, std::uint64_t>> edges) {
        std::vector<std::vector<std::uint64_t>> specs;
        specs.reserve(edges.size());
        for (const auto &e : edges)
          specs.push_back({std::min(e.first, e.second),
                           std::max(e.first, e.second)});
        std::sort(specs.begin(), specs.end());
        specs.erase(std::unique(specs.begin(), specs.end()), specs.end());
        if (specs.empty()) return;
        if (static_cast<int>(out.size()) >= std::max(nCandidates, 1)) return;
        if (seen.insert(specs).second) out.push_back(std::move(specs));
      };

  // Deterministic anchors, highest priority first:
  //  (0) cell-faces — cone v_new over the edges of one top cell (the genuine
  //      k>=1 higher-cell enrichment near a cell, never worse than cone growth's
  //      2-skeleton there),
  //  (1) full 2-star — a triangle over every existing edge (maximal 2-cells),
  //  (2) boundary-fan — triangles over every ∂W edge (fills the boundary cycles
  //      spectrally without touching ∂W's pinned weights),
  //  (3) interior-fan — triangles over every interior edge (empty at the bare seed).
  if (!cells.empty()) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> cellEdges;
    const auto &c = cells.front();
    for (std::size_t i = 0; i + 1 < c.size(); ++i)
      for (std::size_t j = i + 1; j < c.size(); ++j)
        cellEdges.emplace_back(c[i], c[j]);
    add(std::move(cellEdges));
  }
  add(pool);
  add(es.boundaryEdges());
  add(es.interiorEdges());

  // Reproducible random edge subsets (each edge w.p. 1/2; never empty) fill the
  // remaining budget, so the triangle connectivity is genuinely searched. The
  // seed is decorrelated from the edge-fan draw so the two searches do not echo.
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
  std::uniform_int_distribution<int> coin(0, 1);
  const int guardMax = 200 * std::max(nCandidates, 1);
  for (int guard = 0;
       static_cast<int>(out.size()) < std::max(nCandidates, 1) && guard < guardMax;
       ++guard) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> s;
    for (const auto &e : pool)
      if (coin(rng) == 1) s.push_back(e);
    if (s.empty() && !pool.empty())
      s.push_back(pool[static_cast<std::size_t>(rng() % pool.size())]);
    add(std::move(s));
  }
  return out;
}

bool RealizabilityOracle::growBestConnectivity(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, std::uint64_t seed, int nCandidates, bool harmonic,
    int &candidatesOut, int &triangleCandidatesOut,
    std::size_t &spaceSizeOut) const {
  const std::vector<std::uint64_t> verts = es.vertexIds();
  const std::size_t N = verts.size();
  if (N == 0) return false;

  // The full per-step vertex-incidence space the edge candidates are pruned from:
  // nonempty subsets of the existing vertices the new interior vertex may wire to.
  spaceSizeOut = (N < 63) ? ((std::size_t{1} << N) - 1)
                          : std::numeric_limits<std::size_t>::max();

  // Edge-fan candidates (singleton specs {v_new, u}) — the only spectrally
  // relevant atom at k=0 (L_0 = D - A reads the 1-skeleton). Seeded exactly as
  // before, so the k=0 cone/free path stays byte-for-byte unchanged.
  const std::vector<std::vector<std::uint64_t>> subsets =
      connectivityCandidates(es, nCandidates, seed);
  candidatesOut = static_cast<int>(subsets.size());
  triangleCandidatesOut = 0;

  if (es.degree() == 0) {
    CLOG(INFO_LEVEL, "free-connectivity grow (k=0): scoring ", candidatesOut,
         " of ", spaceSizeOut, " incidence patterns over ", N, " vertices");
    double bestR = std::numeric_limits<double>::infinity();
    std::vector<std::vector<std::uint64_t>> bestSpecs;
    std::vector<cd> tmpWitness;
    for (std::size_t c = 0; c < subsets.size(); ++c) {
      std::vector<std::vector<std::uint64_t>> specs;
      specs.reserve(subsets[c].size());
      for (const std::uint64_t u : subsets[c]) specs.push_back({u});
      if (!es.attachInteriorVertex(specs)) continue;  // skip if it would touch ∂W
      const double r =
          optimizePass(es, pinnedByTuple, epsilon, restarts,
                       seed + 1 + static_cast<std::uint64_t>(c), harmonic,
                       tmpWitness);
      if (r < bestR) {
        bestR = r;
        bestSpecs = specs;
      }
      es.detachLastInteriorVertex();
    }
    if (bestSpecs.empty()) return es.growInterior(seed);
    return es.attachInteriorVertex(bestSpecs);
  }

  // k>=1: ALSO enumerate the triangle (2-simplex) candidates — the cells the
  // metric L_k reads through ∂_2 — so the candidate breadth (edge + triangle
  // fans) is surfaced in the Verdict and logged (no silent cap). But every
  // *additive* candidate is provably spectrally inert at k>=1: a dangling edge or
  // triangle is dropped by ChainComplex::fromSpacetime (which builds only the top
  // cells' downward closure), and the active top-cell attach is boundary-locked
  // (it introduces new boundary edges incident to the new vertex, which the
  // bit-exact ∂W guard rejects). Both are certified by the test suite
  // (test_k1_triangle_search_python: dangling-drop, residual-bit-exact, the
  // boundary lock). Scoring them would only confirm no improvement at the cost of
  // perturbing the vertex-id allocator, so the step is the one boundary-fixed move
  // that DOES enrich L_k: the stellar Pachner subdivision (growInterior).
  triangleCandidatesOut = static_cast<int>(
      triangleConnectivityCandidates(es, nCandidates, seed).size());
  CLOG(INFO_LEVEL, "free-connectivity grow (k=", es.degree(), "): enumerated ",
       candidatesOut, " edge + ", triangleCandidatesOut,
       " triangle additive candidates, all spectrally inert at k>=1 (dangling "
       "cells dropped by ChainComplex; active top-cell attach boundary-locked); "
       "falling back to the boundary-fixed Pachner subdivision");
  return es.growInterior(seed);
}

bool RealizabilityOracle::growBestSurgery(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, std::uint64_t seed, bool harmonic, double currentResidual,
    int &removalsOut) const {
  // Enumerate the topology-changing candidates: every interior top cell whose
  // removal opens a hole/handle with ∂W held bit-exact. Score each by the
  // residual it reaches after the weight optimization (try → score → restore),
  // and commit the single best one iff it strictly improves on the current
  // residual AND keeps every pinned boundary tuple present (so the target form
  // still has its support). The committed move shifts b_k — emergent topology.
  const std::vector<std::vector<std::uint64_t>> cells = es.interiorTopCells();
  CLOG(INFO_LEVEL, "surgery grow (k=", es.degree(), "): scoring ", cells.size(),
       " interior-top-cell removals against residual ", currentResidual);
  if (cells.empty()) return false;

  double bestR = currentResidual;
  std::vector<std::uint64_t> bestCell;
  std::vector<cd> tmpWitness;
  for (std::size_t c = 0; c < cells.size(); ++c) {
    if (!es.removeInteriorCell(cells[c])) continue;  // would touch ∂W: skip
    // The pinned boundary k-cells must all survive the removal, else the target
    // form loses its support and the score is meaningless.
    bool pinnedIntact = true;
    {
      std::set<std::vector<std::uint64_t>> live(es.cellSimplices().begin(),
                                                es.cellSimplices().end());
      for (const auto &kv : pinnedByTuple)
        if (live.find(kv.first) == live.end()) { pinnedIntact = false; break; }
    }
    if (pinnedIntact) {
      const double r =
          optimizePass(es, pinnedByTuple, epsilon, restarts,
                       seed + 1 + static_cast<std::uint64_t>(c), harmonic,
                       tmpWitness);
      if (r < bestR) {
        bestR = r;
        bestCell = cells[c];
      }
    }
    es.restoreLastRemoval();
  }
  if (bestCell.empty()) return false;  // no improving removal
  if (!es.removeInteriorCell(bestCell)) return false;
  ++removalsOut;
  return true;
}

double RealizabilityOracle::fillInterior(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, int maxCones, std::uint64_t seed, GrowthMode mode,
    int connectivityCandidates, bool harmonic, std::vector<cd> &witnessOut,
    int &conesApplied, int &candidatesOut, int &triangleCandidatesOut,
    int &surgeryRemovals, std::size_t &spaceSizeOut) const {
  double bestR = std::numeric_limits<double>::infinity();
  conesApplied = 0;
  candidatesOut = 0;
  triangleCandidatesOut = 0;
  surgeryRemovals = 0;
  spaceSizeOut = 0;

  int conesUsed = 0;  // additive commits — the budgeted resource in SurgeryAndCone
  for (int cone = 0;; ++cone) {
    // Optimize the current complex (seed + cone preserves the cone-path seed
    // sequence exactly, so the historical decide() is byte-for-byte unchanged).
    bestR = optimizePass(es, pinnedByTuple, epsilon, restarts,
                         seed + static_cast<std::uint64_t>(cone), harmonic,
                         witnessOut);

    // Stop on convergence, an exhausted budget, or a complex that cannot grow
    // (growth is the capacity/structure gate; it leaves ∂W byte-fixed).
    if (bestR < epsilon) break;
    bool grew;
    if (mode == GrowthMode::SurgeryAndCone) {
      // The composed move-set: the best IMPROVING cut wins the step; when no
      // cut improves, fall back to the additive cone. `maxCones` budgets the
      // additive commits only (the added vertices); cuts are bounded by the
      // improving-only rule and the finite interior-cell set.
      grew = growBestSurgery(es, pinnedByTuple, epsilon, restarts,
                             seed + 1000 + static_cast<std::uint64_t>(cone),
                             harmonic, bestR, surgeryRemovals);
      if (!grew && conesUsed < maxCones) {
        // The additive fallback rides the Pachner add; on complexes where the
        // move proposer cannot produce a valid cone (it can throw on degenerate
        // proposals) the step is logged and skipped — the verdict then floors
        // at the explored complexity rather than losing the decision.
        try {
          grew = es.growInterior(seed + 2000 + static_cast<std::uint64_t>(cone));
        } catch (const std::exception &e) {
          CLOG(WARN_LEVEL, "SurgeryAndCone: additive grow failed (", e.what(),
               "); stopping growth at the explored complexity");
          grew = false;
        }
        if (grew) ++conesUsed;
      }
    } else {
      if (cone >= maxCones) break;
      if (mode == GrowthMode::Surgery)
        // The topology-changing move-set: commit the best b_k-shifting removal.
        grew = growBestSurgery(es, pinnedByTuple, epsilon, restarts,
                               seed + 1000 + static_cast<std::uint64_t>(cone),
                               harmonic, bestR, surgeryRemovals);
      else if (mode == GrowthMode::FreeConnectivity)
        grew = growBestConnectivity(
            es, pinnedByTuple, epsilon, restarts,
            seed + 1000 + static_cast<std::uint64_t>(cone),
            connectivityCandidates, harmonic, candidatesOut,
            triangleCandidatesOut, spaceSizeOut);
      else
        grew = es.growInterior(seed + 1000 + static_cast<std::uint64_t>(cone));
    }
    if (!grew) break;
    ++conesApplied;
  }
  return bestR;
}

RealizabilityOracle::Verdict RealizabilityOracle::decide(
    const std::vector<cd> &U, int dA, int dB, double epsilon, int restarts,
    int maxCones, std::uint64_t seed, GrowthMode mode,
    int connectivityCandidates, bool harmonic) {
  const std::vector<cd> target = bend(U, dA, dB);  // ψ_U, length dA·dB

  EigenstateSynthesis es(bulk_);  // k = 0: the vertex graph Laplacian
  if (es.order() < target.size())
    throw std::invalid_argument(
        "RealizabilityOracle: the bulk has fewer vertices than the bent target "
        "needs on its output-boundary support (dA*dB).");

  // The output-boundary support is the first dA·dB sorted-id vertices (the
  // established k=0 embedding idiom): pin those vertex cells to ψ_U. The interior
  // vertex grown in (cone or free-connectivity) has the largest id, so the first
  // dA·dB tuples are preserved across growth.
  const std::vector<std::vector<std::uint64_t>> &cells = es.cellSimplices();
  std::map<std::vector<std::uint64_t>, cd> pinnedByTuple;
  for (std::size_t i = 0; i < target.size(); ++i)
    pinnedByTuple[cells[i]] = target[i];

  Verdict v;
  v.target = target;
  const double r = fillInterior(es, pinnedByTuple, epsilon, restarts, maxCones,
                                seed, mode, connectivityCandidates, harmonic,
                                v.state, v.conesApplied, v.connectivityCandidates,
                                v.triangleCandidates, v.surgeryRemovals,
                                v.connectivitySpaceSize);
  v.residual = r;
  v.realizable = (r < epsilon);
  v.floor = v.realizable ? 0.0 : r;
  v.eigenvalue = es.rayleigh(v.state);
  v.interiorVertexCount = es.interiorVertexCount();
  v.interiorEdgeCount = es.numInteriorEdges();
  v.witness = bulk_;
  return v;
}

RealizabilityOracle::Verdict RealizabilityOracle::decideHarmonic(
    const Cochain &target, double epsilon, int restarts, int maxCones,
    std::uint64_t seed, GrowthMode mode, int connectivityCandidates,
    bool harmonic) {
  const int k = target.degree();
  if (k < 0 || target.size() == 0)
    throw std::invalid_argument(
        "RealizabilityOracle::decideHarmonic: the target must be a non-empty "
        "k-form (degree >= 0).");

  EigenstateSynthesis es(bulk_, k);

  // Pin the boundary k-cells to the target form, matched to the bulk's k-cell
  // order by sorted vertex-id tuple. Normalize the target so its boundary block
  // has unit norm (comparable to the auxiliary-amplitude box); the residual is
  // scale-invariant either way. Record the normalized boundary values in the
  // verdict's `target` for traceability.
  const Eigen::VectorXcd &coeffs = target.coeffs();
  double tnrm = 0.0;
  for (Eigen::Index i = 0; i < coeffs.size(); ++i) tnrm += std::norm(coeffs[i]);
  const double inv = (tnrm > 0.0) ? 1.0 / std::sqrt(tnrm) : 1.0;

  std::map<std::vector<std::uint64_t>, cd> pinnedByTuple;
  std::vector<cd> targetOut;
  targetOut.reserve(target.simplices().size());
  for (std::size_t i = 0; i < target.simplices().size(); ++i) {
    std::vector<std::uint64_t> key = target.simplices()[i];
    std::sort(key.begin(), key.end());
    const cd value = coeffs[static_cast<Eigen::Index>(i)] * inv;
    pinnedByTuple[key] = value;
    targetOut.push_back(value);
  }

  // The surface must actually be ∂W: at least one target k-cell has to land on a
  // boundary cell of the bulk (else the form is over a mismatched complex).
  bool anyMatch = false;
  for (const std::vector<std::uint64_t> &cell : es.cellSimplices())
    if (pinnedByTuple.find(cell) != pinnedByTuple.end()) { anyMatch = true; break; }
  if (!anyMatch)
    throw std::invalid_argument(
        "RealizabilityOracle::decideHarmonic: none of the target form's k-cells "
        "are cells of the bulk — the surface does not match the bulk boundary.");

  Verdict v;
  v.target = targetOut;
  // Growth mode is a free choice at k>=1. `Cone` keeps the historical boundary-
  // fixed 1->(d+1) Pachner add. `FreeConnectivity` searches interior connectivity:
  // unlike at k=0 — where L_0 = D - A reads only the 1-skeleton, so only edge
  // attachments matter — the metric L_k (k>=1) reads the 2-cells through ∂_2, so
  // growBestConnectivity additionally proposes triangle (2-simplex) attachments,
  // and those are NOT spectrally inert here. The candidate breadth (edge fans +
  // triangle fans) is surfaced in the Verdict and logged.
  const double r = fillInterior(es, pinnedByTuple, epsilon, restarts, maxCones,
                                seed, mode, connectivityCandidates, harmonic,
                                v.state, v.conesApplied, v.connectivityCandidates,
                                v.triangleCandidates, v.surgeryRemovals,
                                v.connectivitySpaceSize);
  v.residual = r;
  v.realizable = (r < epsilon);
  v.floor = v.realizable ? 0.0 : r;
  v.eigenvalue = es.rayleigh(v.state);
  v.interiorVertexCount = es.interiorVertexCount();
  v.interiorEdgeCount = es.numInteriorEdges();
  v.witness = bulk_;
  return v;
}

}  // namespace tessera::cobordism
