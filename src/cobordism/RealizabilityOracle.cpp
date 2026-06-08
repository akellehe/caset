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
    int restarts, std::uint64_t passSeed, std::vector<cd> &witnessOut) const {
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
  const auto residual = [&es, &buildPsi, m, nW, usePhases,
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
    cd lam(0.0, 0.0);
    for (std::size_t i = 0; i < order; ++i) lam += std::conj(psi[i]) * Lp[i];
    const double lambda = lam.real();
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

bool RealizabilityOracle::growBestConnectivity(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, std::uint64_t seed, int nCandidates, int &candidatesOut,
    std::size_t &spaceSizeOut) const {
  const std::vector<std::uint64_t> verts = es.vertexIds();
  const std::size_t N = verts.size();
  if (N == 0) return false;

  // The full per-step incidence space the candidates are pruned from: nonempty
  // subsets of the existing vertices the new interior vertex may wire to.
  spaceSizeOut = (N < 63) ? ((std::size_t{1} << N) - 1)
                          : std::numeric_limits<std::size_t>::max();

  const std::vector<std::vector<std::uint64_t>> candidates =
      connectivityCandidates(es, nCandidates, seed);
  candidatesOut = static_cast<int>(candidates.size());

  // Log what is pruned — no silent cap (quiet unless TESSERA_VERBOSE). The same
  // counts are surfaced in the Verdict for programmatic inspection.
  CLOG(INFO_LEVEL, "free-connectivity grow: scoring ", candidates.size(),
       " of ", spaceSizeOut, " incidence patterns over ", N, " vertices");

  double bestR = std::numeric_limits<double>::infinity();
  std::vector<std::uint64_t> bestSubset;
  std::vector<cd> tmpWitness;
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    // Singleton specs: one 1-simplex {v_new, u} per chosen vertex u — the only
    // structure the k=0 graph Laplacian sees, and provably boundary-safe (a new
    // edge to a brand-new vertex creates no new top cell, so ∂W is untouched).
    std::vector<std::vector<std::uint64_t>> specs;
    specs.reserve(candidates[c].size());
    for (const std::uint64_t u : candidates[c]) specs.push_back({u});
    if (!es.attachInteriorVertex(specs)) continue;  // skip if it would touch ∂W
    const double r =
        optimizePass(es, pinnedByTuple, epsilon, restarts,
                     seed + 1 + static_cast<std::uint64_t>(c), tmpWitness);
    if (r < bestR) {
      bestR = r;
      bestSubset = candidates[c];
    }
    es.detachLastInteriorVertex();
  }

  // Fall back to the cone move if nothing could attach (keeps growth progressing).
  if (bestSubset.empty()) return es.growInterior(seed);

  std::vector<std::vector<std::uint64_t>> bestSpecs;
  bestSpecs.reserve(bestSubset.size());
  for (const std::uint64_t u : bestSubset) bestSpecs.push_back({u});
  return es.attachInteriorVertex(bestSpecs);
}

double RealizabilityOracle::fillInterior(
    EigenstateSynthesis &es,
    const std::map<std::vector<std::uint64_t>, cd> &pinnedByTuple, double epsilon,
    int restarts, int maxCones, std::uint64_t seed, GrowthMode mode,
    int connectivityCandidates, std::vector<cd> &witnessOut, int &conesApplied,
    int &candidatesOut, std::size_t &spaceSizeOut) const {
  double bestR = std::numeric_limits<double>::infinity();
  conesApplied = 0;
  candidatesOut = 0;
  spaceSizeOut = 0;

  for (int cone = 0;; ++cone) {
    // Optimize the current complex (seed + cone preserves the cone-path seed
    // sequence exactly, so the historical decide() is byte-for-byte unchanged).
    bestR = optimizePass(es, pinnedByTuple, epsilon, restarts,
                         seed + static_cast<std::uint64_t>(cone), witnessOut);

    // Stop on convergence, an exhausted budget, or a complex that cannot grow
    // (growth is the capacity/structure gate; it leaves ∂W byte-fixed).
    if (bestR < epsilon) break;
    if (cone >= maxCones) break;
    bool grew;
    if (mode == GrowthMode::FreeConnectivity)
      grew = growBestConnectivity(
          es, pinnedByTuple, epsilon, restarts,
          seed + 1000 + static_cast<std::uint64_t>(cone), connectivityCandidates,
          candidatesOut, spaceSizeOut);
    else
      grew = es.growInterior(seed + 1000 + static_cast<std::uint64_t>(cone));
    if (!grew) break;
    ++conesApplied;
  }
  return bestR;
}

RealizabilityOracle::Verdict RealizabilityOracle::decide(
    const std::vector<cd> &U, int dA, int dB, double epsilon, int restarts,
    int maxCones, std::uint64_t seed, GrowthMode mode,
    int connectivityCandidates) {
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
                                seed, mode, connectivityCandidates, v.state,
                                v.conesApplied, v.connectivityCandidates,
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
    std::uint64_t seed) {
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
  // The k>=1 harmonic path keeps cone growth: at k>=1 the metric L_k depends on
  // the higher (volume-weighted) cells, so a 1-skeleton-only connectivity search
  // would be inert — free connectivity is the k=0 graph-Laplacian story.
  int candidatesOut = 0;
  std::size_t spaceSizeOut = 0;
  const double r = fillInterior(es, pinnedByTuple, epsilon, restarts, maxCones,
                                seed, GrowthMode::Cone, /*connectivityCandidates=*/0,
                                v.state, v.conesApplied, candidatesOut,
                                spaceSizeOut);
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
