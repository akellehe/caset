// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/DiracKahler.h"

#include <Eigen/Dense>

#include <bitset>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

namespace {

// The framework spacetime dimension of the Dirac-Kahler construction: 4D. The
// gamma/Clifford structure and the taste multiplicity are fixed properties of
// this 4D framework, independent of the reduced mesh dimension.
constexpr int kFrameworkDim = 4;

// Diagonal weights W_k (length |C_k|) used by both the operator and the current.
// Mirrors HodgeLaplacian: W_0 = I; for k >= 1 the per-cell |volume| (metric) or
// signed volume (lorentzian); metric == false ⇒ unit weights (combinatorial).
std::vector<double> degreeWeights(const HodgeLaplacian &hl, int k,
                                  std::size_t count, bool metric,
                                  bool lorentzian) {
  if (!metric || k == 0)
    return std::vector<double>(count, 1.0);
  std::vector<double> w = hl.weights(k, lorentzian);
  if (w.size() != count) w.assign(count, 1.0);  // defensive: above top dim
  return w;
}

// The boundary matrix d_k (rows |C_{k-1}|, cols |C_k|) as an Eigen matrix.
Eigen::MatrixXd boundaryMatrix(const ChainComplex &cc, int k, int rows, int cols) {
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(rows, cols);
  if (rows == 0 || cols == 0) return d;
  const std::vector<long> &flat = cc.boundaryMatrix(k);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      d(r, c) = static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);
  return d;
}

}  // namespace

DiracKahler::DiracKahler(std::shared_ptr<Spacetime> st) : st_(std::move(st)) {}

std::vector<std::size_t> DiracKahler::cellCounts() const {
  if (!st_) return {};
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const int n = cc.dimension();
  std::vector<std::size_t> counts;
  counts.reserve(static_cast<std::size_t>(n + 1));
  for (int k = 0; k <= n; ++k) counts.push_back(cc.numSimplices(k));
  return counts;
}

int DiracKahler::meshDimension() const {
  if (!st_) return -1;
  return ChainComplex::fromSpacetime(*st_).dimension();
}

std::vector<std::size_t> DiracKahler::blockOffsets() const {
  const std::vector<std::size_t> counts = cellCounts();
  std::vector<std::size_t> off(counts.size() + 1, 0);
  for (std::size_t k = 0; k < counts.size(); ++k) off[k + 1] = off[k] + counts[k];
  return off;
}

std::size_t DiracKahler::totalDimension() const {
  const std::vector<std::size_t> off = blockOffsets();
  return off.empty() ? 0 : off.back();
}

std::vector<cd> DiracKahler::matrix(bool metric, bool lorentzian) const {
  const std::vector<std::size_t> counts = cellCounts();
  const std::vector<std::size_t> off = blockOffsets();
  const std::size_t total = off.empty() ? 0 : off.back();
  std::vector<cd> out(total * total, cd(0.0, 0.0));
  if (total == 0 || !st_) return out;

  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const HodgeLaplacian hl(st_);
  const int n = static_cast<int>(counts.size()) - 1;

  for (int k = 1; k <= n; ++k) {
    const int rows = static_cast<int>(counts[static_cast<std::size_t>(k - 1)]);
    const int cols = static_cast<int>(counts[static_cast<std::size_t>(k)]);
    if (rows == 0 || cols == 0) continue;
    const Eigen::MatrixXd dk = boundaryMatrix(cc, k, rows, cols);

    const std::vector<double> wk =
        degreeWeights(hl, k, static_cast<std::size_t>(cols), metric, lorentzian);
    const std::vector<double> wkm1 = degreeWeights(
        hl, k - 1, static_cast<std::size_t>(rows), metric, lorentzian);

    // Boundary block partial_k: rows in degree k-1, cols in degree k.
    const std::size_t r0 = off[static_cast<std::size_t>(k - 1)];
    const std::size_t c0 = off[static_cast<std::size_t>(k)];
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        out[(r0 + static_cast<std::size_t>(r)) * total + (c0 + static_cast<std::size_t>(c))] +=
            cd(dk(r, c), 0.0);

    // Codifferential block partial_k* = W_k^-1 d_k^T W_{k-1}: rows in degree k,
    // cols in degree k-1.
    for (int i = 0; i < cols; ++i)
      for (int j = 0; j < rows; ++j) {
        const double v = (1.0 / wk[static_cast<std::size_t>(i)]) * dk(j, i) *
                         wkm1[static_cast<std::size_t>(j)];
        out[(c0 + static_cast<std::size_t>(i)) * total + (r0 + static_cast<std::size_t>(j))] +=
            cd(v, 0.0);
      }
  }
  return out;
}

std::vector<cd> DiracKahler::square(bool metric, bool lorentzian) const {
  const std::size_t total = totalDimension();
  std::vector<cd> out(total * total, cd(0.0, 0.0));
  if (total == 0) return out;
  const std::vector<cd> Dflat = matrix(metric, lorentzian);
  Eigen::MatrixXcd D(total, total);
  for (std::size_t i = 0; i < total; ++i)
    for (std::size_t j = 0; j < total; ++j)
      D(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          Dflat[i * total + j];
  const Eigen::MatrixXcd D2 = D * D;
  for (std::size_t i = 0; i < total; ++i)
    for (std::size_t j = 0; j < total; ++j)
      out[i * total + j] =
          D2(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  return out;
}

double DiracKahler::laplacianResidual(bool metric, bool lorentzian) const {
  const std::vector<std::size_t> counts = cellCounts();
  const std::vector<std::size_t> off = blockOffsets();
  const std::size_t total = off.empty() ? 0 : off.back();
  if (total == 0 || !st_) return 0.0;

  const std::vector<cd> sq = square(metric, lorentzian);
  const HodgeLaplacian hl(st_);
  const int n = static_cast<int>(counts.size()) - 1;
  // The k=0 HodgeLaplacian is the Hermitian graph Laplacian, which equals the
  // signed 0-form d'Alembertian only for unit/real weights; skip it on the
  // lorentzian path (compare the d'Alembertian blocks k>=1).
  const int kStart = lorentzian ? 1 : 0;

  double worst = 0.0;
  for (int k = kStart; k <= n; ++k) {
    const std::size_t m = counts[static_cast<std::size_t>(k)];
    if (m == 0) continue;
    const std::vector<cd> lk = hl.laplacian(k, metric, lorentzian);
    const std::size_t o = off[static_cast<std::size_t>(k)];
    double sumSq = 0.0;
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < m; ++j) {
        const cd diff = sq[(o + i) * total + (o + j)] - lk[i * m + j];
        sumSq += std::norm(diff);
      }
    worst = std::max(worst, std::sqrt(sumSq));
  }
  return worst;
}

int DiracKahler::frameworkDimension() const { return kFrameworkDim; }

std::size_t DiracKahler::gammaDimension() const {
  return static_cast<std::size_t>(1) << kFrameworkDim;
}

int DiracKahler::multiplicity() const { return 1 << (kFrameworkDim / 2); }

std::vector<double> DiracKahler::signature(bool lorentzian) const {
  const int d = kFrameworkDim;
  std::vector<double> eta(static_cast<std::size_t>(d) * d, 0.0);
  for (int a = 0; a < d; ++a) {
    const double s = (lorentzian && a == 0) ? -1.0 : 1.0;
    eta[static_cast<std::size_t>(a) * d + a] = s;
  }
  return eta;
}

std::vector<std::vector<cd>> DiracKahler::gammas(bool lorentzian) const {
  const int d = kFrameworkDim;
  const std::size_t dim = gammaDimension();
  std::vector<std::vector<cd>> gs;
  gs.reserve(static_cast<std::size_t>(d));
  for (int a = 0; a < d; ++a) {
    std::vector<cd> g(dim * dim, cd(0.0, 0.0));
    const double etaAA = (lorentzian && a == 0) ? -1.0 : 1.0;
    const unsigned bit = 1u << a;
    const unsigned below = bit - 1u;  // bits 0..a-1
    for (unsigned S = 0; S < dim; ++S) {
      // sign = (-1)^popcount(S & bits below a) — the Koszul sign of e^a in e^S.
      const double sign =
          (std::bitset<32>(S & below).count() % 2 == 0) ? 1.0 : -1.0;
      if ((S & bit) == 0) {
        // exterior multiplication: add basis 1-form e^a (raises degree).
        const unsigned T = S | bit;
        g[static_cast<std::size_t>(T) * dim + S] += cd(sign, 0.0);
      } else {
        // interior multiplication (contraction) by e^a, with the metric eta^aa.
        const unsigned T = S & ~bit;
        g[static_cast<std::size_t>(T) * dim + S] += cd(etaAA * sign, 0.0);
      }
    }
    gs.push_back(std::move(g));
  }
  return gs;
}

double DiracKahler::cliffordResidual(bool lorentzian) const {
  const int d = kFrameworkDim;
  const std::size_t dim = gammaDimension();
  const std::vector<std::vector<cd>> gs = gammas(lorentzian);
  const std::vector<double> eta = signature(lorentzian);

  std::vector<Eigen::MatrixXcd> G;
  G.reserve(gs.size());
  for (const auto &gflat : gs) {
    Eigen::MatrixXcd M(dim, dim);
    for (std::size_t i = 0; i < dim; ++i)
      for (std::size_t j = 0; j < dim; ++j)
        M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            gflat[i * dim + j];
    G.push_back(std::move(M));
  }
  const Eigen::MatrixXcd I =
      Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(dim),
                                 static_cast<Eigen::Index>(dim));
  double worst = 0.0;
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      const Eigen::MatrixXcd anti = G[static_cast<std::size_t>(a)] * G[static_cast<std::size_t>(b)] +
                                    G[static_cast<std::size_t>(b)] * G[static_cast<std::size_t>(a)];
      const double etaAB = eta[static_cast<std::size_t>(a) * d + b];
      worst = std::max(worst, (anti - 2.0 * etaAB * I).norm());
    }
  return worst;
}

std::vector<cd> DiracKahler::lift(int k,
                                  const std::vector<cd> &component) const {
  const std::vector<std::size_t> counts = cellCounts();
  const std::vector<std::size_t> off = blockOffsets();
  const std::size_t total = off.empty() ? 0 : off.back();
  if (k < 0 || k >= static_cast<int>(counts.size()))
    throw std::runtime_error("DiracKahler::lift: degree k=" + std::to_string(k) +
                             " is out of range [0, " +
                             std::to_string(static_cast<int>(counts.size()) - 1) +
                             "]");
  const std::size_t m = counts[static_cast<std::size_t>(k)];
  if (component.size() != m)
    throw std::runtime_error(
        "DiracKahler::lift: component length " +
        std::to_string(component.size()) + " != |C_k| " + std::to_string(m));
  std::vector<cd> field(total, cd(0.0, 0.0));
  const std::size_t o = off[static_cast<std::size_t>(k)];
  for (std::size_t c = 0; c < m; ++c) field[o + c] = component[c];
  return field;
}

std::vector<double> DiracKahler::chargeDensity(const std::vector<cd> &field,
                                               bool metric) const {
  const std::vector<std::size_t> counts = cellCounts();
  const std::vector<std::size_t> off = blockOffsets();
  const std::size_t total = off.empty() ? 0 : off.back();
  if (field.size() != total)
    throw std::runtime_error(
        "DiracKahler::chargeDensity: field length " +
        std::to_string(field.size()) + " != totalDimension " +
        std::to_string(total));
  std::vector<double> density(total, 0.0);
  if (total == 0 || !st_) return density;

  const HodgeLaplacian hl(st_);
  const int n = static_cast<int>(counts.size()) - 1;
  for (int k = 0; k <= n; ++k) {
    const std::size_t m = counts[static_cast<std::size_t>(k)];
    if (m == 0) continue;
    // j^0 uses the positive |volume| charge measure (the time component of the
    // Dirac current is the positive-definite probability/charge density).
    const std::vector<double> wk =
        degreeWeights(hl, k, m, metric, /*lorentzian=*/false);
    const std::size_t o = off[static_cast<std::size_t>(k)];
    for (std::size_t c = 0; c < m; ++c)
      density[o + c] = wk[c] * std::norm(field[o + c]);  // std::norm = |.|^2
  }
  return density;
}

double DiracKahler::charge(const std::vector<cd> &field, bool metric) const {
  const std::vector<double> density = chargeDensity(field, metric);
  double q = 0.0;
  for (const double d : density) q += d;
  return q;
}

}  // namespace tessera::cobordism
