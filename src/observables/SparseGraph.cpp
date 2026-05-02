// MIT License
// Copyright (c) 2025 Andrew Kelleher

#include "observables/SparseGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>

namespace tessera {

namespace {

/// Symmetric normalized Laplacian times a vector:
/// y = (I - D^{-1/2} A D^{-1/2}) x
/// where deg = column-sum of A (undirected, no self-loops).
inline void laplacianSymApply(
    const std::vector<std::int64_t> &indptr,
    const std::vector<std::uint32_t> &indices,
    const std::vector<double> &invSqrtDeg,
    const std::vector<double> &x,
    std::vector<double> &y) {
  const std::size_t n = x.size();
  // y = x  - L * x_norm  with scaling.
  // y_i = x_i - (1/sqrt(d_i)) * sum_{j: (i,j) in E} (1/sqrt(d_j)) * x_j
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    auto p = indptr[i];
    auto q = indptr[i + 1];
    for (auto k = p; k < q; ++k) {
      std::uint32_t j = indices[k];
      s += invSqrtDeg[j] * x[j];
    }
    s *= invSqrtDeg[i];
    y[i] = x[i] - s;
  }
}

/// Compute a small dense matrix exponential e^{-t T} where T is k×k
/// symmetric tridiagonal stored as alpha (diag, length k) and
/// beta (off-diag, length k-1).
///
/// We work in the dense k×k matrix form and use Padé-13 with
/// scaling-and-squaring.  Standalone; no Eigen.
class DenseMatrixK {
public:
  std::size_t k;
  std::vector<double> data;  // row-major, k*k
  DenseMatrixK(std::size_t k_) : k(k_), data(k_ * k_, 0.0) {}
  double &at(std::size_t i, std::size_t j) { return data[i * k + j]; }
  double at(std::size_t i, std::size_t j) const { return data[i * k + j]; }

  /// One-norm = max over columns of sum of |entries| in that column.
  double oneNorm() const {
    double m = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < k; ++i) s += std::abs(at(i, j));
      if (s > m) m = s;
    }
    return m;
  }
};

DenseMatrixK matMul(const DenseMatrixK &A, const DenseMatrixK &B) {
  DenseMatrixK C(A.k);
  for (std::size_t i = 0; i < A.k; ++i) {
    for (std::size_t l = 0; l < A.k; ++l) {
      double aij = A.at(i, l);
      if (aij == 0.0) continue;
      for (std::size_t j = 0; j < A.k; ++j) {
        C.at(i, j) += aij * B.at(l, j);
      }
    }
  }
  return C;
}

void axpy(DenseMatrixK &A, double s, const DenseMatrixK &B) {
  for (std::size_t i = 0; i < A.data.size(); ++i) A.data[i] += s * B.data[i];
}

DenseMatrixK identityK(std::size_t k) {
  DenseMatrixK I(k);
  for (std::size_t i = 0; i < k; ++i) I.at(i, i) = 1.0;
  return I;
}

/// Solve A * X = B via Gauss-Jordan elimination (small k, ~30 max,
/// so this is fine).  A and B are k×k row-major.  X overwrites B.
bool solveLinearSystem(DenseMatrixK A, DenseMatrixK &B) {
  const std::size_t k = A.k;
  for (std::size_t i = 0; i < k; ++i) {
    // Find pivot.
    std::size_t piv = i;
    double pivVal = std::abs(A.at(i, i));
    for (std::size_t r = i + 1; r < k; ++r) {
      double v = std::abs(A.at(r, i));
      if (v > pivVal) { pivVal = v; piv = r; }
    }
    if (pivVal < 1e-300) return false;
    if (piv != i) {
      for (std::size_t c = 0; c < k; ++c) {
        std::swap(A.at(i, c), A.at(piv, c));
        std::swap(B.at(i, c), B.at(piv, c));
      }
    }
    double inv = 1.0 / A.at(i, i);
    for (std::size_t c = 0; c < k; ++c) {
      A.at(i, c) *= inv;
      B.at(i, c) *= inv;
    }
    for (std::size_t r = 0; r < k; ++r) {
      if (r == i) continue;
      double f = A.at(r, i);
      if (f == 0.0) continue;
      for (std::size_t c = 0; c < k; ++c) {
        A.at(r, c) -= f * A.at(i, c);
        B.at(r, c) -= f * B.at(i, c);
      }
    }
  }
  return true;
}

/// Pade-13 with scaling-and-squaring.  Adapted from Higham 2010.
DenseMatrixK matExpPade13(DenseMatrixK A) {
  const std::size_t k = A.k;
  // Coefficients of Pade-13 (numerator and denominator).
  static const double b[14] = {
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0,
    40840800.0, 960960.0, 16380.0, 182.0, 1.0
  };
  // Scaling: choose s = max(0, ceil(log2(||A||_1 / theta_13))).
  const double theta13 = 5.371920351148152;
  double normA = A.oneNorm();
  int s = 0;
  if (normA > theta13) {
    s = static_cast<int>(std::ceil(std::log2(normA / theta13)));
  }
  if (s > 0) {
    double scale = std::ldexp(1.0, -s);
    for (auto &v : A.data) v *= scale;
  }

  DenseMatrixK A2 = matMul(A, A);
  DenseMatrixK A4 = matMul(A2, A2);
  DenseMatrixK A6 = matMul(A4, A2);

  // Higham 2010 Alg. 2.3:
  //   p(A) = U + V where U has odd-power terms, V has even-power.
  //   U = A * [b1 I + b3 A^2 + b5 A^4 + b7 A^6
  //            + A^6 * (b9 A^2 + b11 A^4 + b13 A^6)]
  //   V =        b0 I + b2 A^2 + b4 A^4 + b6 A^6
  //            + A^6 * (b8 A^2 + b10 A^4 + b12 A^6)
  DenseMatrixK Ik = identityK(k);

  // U: inner = b9 A^2 + b11 A^4 + b13 A^6 (lifted by A^6)
  DenseMatrixK innerU(k);
  axpy(innerU, b[13], A6);
  axpy(innerU, b[11], A4);
  axpy(innerU, b[9], A2);
  DenseMatrixK A6_innerU = matMul(A6, innerU);

  // U_inside = b1 I + b3 A^2 + b5 A^4 + b7 A^6 + A^6 * innerU
  DenseMatrixK U_inside(k);
  axpy(U_inside, b[7], A6);
  axpy(U_inside, b[5], A4);
  axpy(U_inside, b[3], A2);
  axpy(U_inside, b[1], Ik);
  for (std::size_t i = 0; i < U_inside.data.size(); ++i)
    U_inside.data[i] += A6_innerU.data[i];
  DenseMatrixK U = matMul(A, U_inside);

  // V: inner = b8 A^2 + b10 A^4 + b12 A^6 (lifted by A^6)
  DenseMatrixK innerV(k);
  axpy(innerV, b[12], A6);
  axpy(innerV, b[10], A4);
  axpy(innerV, b[8], A2);
  DenseMatrixK A6_innerV = matMul(A6, innerV);

  // V = b0 I + b2 A^2 + b4 A^4 + b6 A^6 + A^6 * innerV
  DenseMatrixK V(k);
  axpy(V, b[6], A6);
  axpy(V, b[4], A4);
  axpy(V, b[2], A2);
  axpy(V, b[0], Ik);
  for (std::size_t i = 0; i < V.data.size(); ++i)
    V.data[i] += A6_innerV.data[i];

  // R = (V - U)^{-1} (V + U)
  DenseMatrixK numer(k), denom(k);
  for (std::size_t i = 0; i < numer.data.size(); ++i) {
    numer.data[i] = V.data[i] + U.data[i];
    denom.data[i] = V.data[i] - U.data[i];
  }
  if (!solveLinearSystem(denom, numer)) {
    // Singular — return identity as a safe fallback.
    return identityK(k);
  }
  DenseMatrixK R = numer;

  // Squaring: square R s times.
  for (int i = 0; i < s; ++i) {
    R = matMul(R, R);
  }
  return R;
}

}  // namespace

SparseGraph SparseGraph::fromCOO(
    const std::vector<std::uint32_t> &rows,
    const std::vector<std::uint32_t> &cols,
    std::uint32_t n) {
  // Build a unique-edge set with i < j.
  std::vector<std::uint64_t> packed;
  packed.reserve(rows.size());
  for (std::size_t k = 0; k < rows.size(); ++k) {
    std::uint32_t a = rows[k], b = cols[k];
    if (a == b) continue;  // ignore self-loops
    std::uint32_t u = std::min(a, b), v = std::max(a, b);
    packed.push_back((static_cast<std::uint64_t>(u) << 32)
                     | static_cast<std::uint64_t>(v));
  }
  std::sort(packed.begin(), packed.end());
  packed.erase(std::unique(packed.begin(), packed.end()), packed.end());

  // Per-node neighbor lists.
  std::vector<std::vector<std::uint32_t>> nbrs(n);
  for (auto p : packed) {
    std::uint32_t u = static_cast<std::uint32_t>(p >> 32);
    std::uint32_t v = static_cast<std::uint32_t>(p & 0xFFFFFFFFu);
    nbrs[u].push_back(v);
    nbrs[v].push_back(u);
  }

  SparseGraph g;
  g.nNodes_ = n;
  g.nEdges_ = packed.size();
  g.indptr_.resize(n + 1, 0);
  for (std::uint32_t i = 0; i < n; ++i) {
    g.indptr_[i + 1] = g.indptr_[i] + static_cast<std::int64_t>(nbrs[i].size());
  }
  g.indices_.resize(g.nEdges_ * 2);
  for (std::uint32_t i = 0; i < n; ++i) {
    auto p = g.indptr_[i];
    for (auto j : nbrs[i]) {
      g.indices_[p++] = j;
    }
  }
  return g;
}

bool SparseGraph::isBipartite() const {
  if (nNodes_ == 0 || nEdges_ == 0) return true;
  std::vector<int> color(nNodes_, -1);
  for (std::uint32_t s = 0; s < nNodes_; ++s) {
    if (color[s] != -1) continue;
    color[s] = 0;
    std::queue<std::uint32_t> q;
    q.push(s);
    while (!q.empty()) {
      auto u = q.front(); q.pop();
      auto p = indptr_[u];
      auto e = indptr_[u + 1];
      for (auto k = p; k < e; ++k) {
        auto v = indices_[k];
        if (color[v] == -1) {
          color[v] = 1 - color[u];
          q.push(v);
        } else if (color[v] == color[u]) {
          return false;
        }
      }
    }
  }
  return true;
}

std::vector<double> SparseGraph::diagonalHeatKernel(
    const std::vector<std::uint32_t> &starts,
    const std::vector<double> &times,
    int krylovDim) const {
  const std::size_t nStarts = starts.size();
  const std::size_t nT = times.size();
  std::vector<double> out(nStarts * nT, 0.0);
  if (nNodes_ == 0 || nStarts == 0) return out;

  // Precompute D^{-1/2}.  Isolated nodes (deg 0) get 0.0 for invSqrtDeg
  // — heat kernel is identity on isolated nodes (return prob = 1
  // for all t > 0).
  std::vector<double> invSqrtDeg(nNodes_, 0.0);
  for (std::uint32_t i = 0; i < nNodes_; ++i) {
    auto d = degree(i);
    if (d > 0) invSqrtDeg[i] = 1.0 / std::sqrt(static_cast<double>(d));
  }

  if (nEdges_ == 0) {
    // No edges: all return probabilities = 1.
    std::fill(out.begin(), out.end(), 1.0);
    return out;
  }

  std::vector<double> v(nNodes_), w(nNodes_), prev(nNodes_);
  std::vector<std::vector<double>> V;     // Krylov basis vectors
  std::vector<double> alpha, beta;        // tridiagonal entries

  for (std::size_t s = 0; s < nStarts; ++s) {
    std::uint32_t start = starts[s];
    if (start >= nNodes_) continue;

    // Initialize Krylov: v0 = e_start.
    std::fill(v.begin(), v.end(), 0.0);
    v[start] = 1.0;
    V.assign(1, v);
    alpha.clear();
    beta.clear();

    // Cap Krylov dimension at nNodes (can't have more orthogonal
    // vectors than the dimension of the space).
    int kMax = std::min<int>(krylovDim, static_cast<int>(nNodes_));
    for (int j = 0; j < kMax; ++j) {
      // w = L_sym * v
      laplacianSymApply(indptr_, indices_, invSqrtDeg, v, w);
      // alpha_j = <v, w>
      double a = 0.0;
      for (std::size_t i = 0; i < nNodes_; ++i) a += v[i] * w[i];
      alpha.push_back(a);
      // Last iteration — no need to compute beta_j or v_{j+1}.
      if (j + 1 == kMax) break;
      // w = w - alpha_j * v - beta_{j-1} * v_{j-1}
      for (std::size_t i = 0; i < nNodes_; ++i) {
        w[i] -= a * v[i];
        if (j > 0) w[i] -= beta.back() * prev[i];
      }
      // Reorthogonalize for numerical stability (full Gram-Schmidt
      // against existing basis).
      for (const auto &u : V) {
        double dot = 0.0;
        for (std::size_t i = 0; i < nNodes_; ++i) dot += u[i] * w[i];
        for (std::size_t i = 0; i < nNodes_; ++i) w[i] -= dot * u[i];
      }
      double normW = 0.0;
      for (std::size_t i = 0; i < nNodes_; ++i) normW += w[i] * w[i];
      normW = std::sqrt(normW);
      if (normW < 1e-12) break;  // Krylov subspace exhausted
      beta.push_back(normW);
      prev = v;
      double inv = 1.0 / normW;
      for (std::size_t i = 0; i < nNodes_; ++i) v[i] = w[i] * inv;
      V.push_back(v);
    }
    int actualK = static_cast<int>(alpha.size());
    if (actualK == 0) {
      // No iterations succeeded.
      for (std::size_t j = 0; j < nT; ++j) out[s * nT + j] = 1.0;
      continue;
    }

    // For each t: compute exp(-t T_k) and read entry [0,0].
    for (std::size_t j = 0; j < nT; ++j) {
      double t = times[j];
      // Build dense -t * T.
      DenseMatrixK Tmat(actualK);
      for (int i = 0; i < actualK; ++i) Tmat.at(i, i) = -t * alpha[i];
      for (int i = 0; i + 1 < actualK; ++i) {
        Tmat.at(i, i + 1) = -t * beta[i];
        Tmat.at(i + 1, i) = -t * beta[i];
      }
      DenseMatrixK eT = matExpPade13(Tmat);
      out[s * nT + j] = eT.at(0, 0);
    }
  }
  return out;
}

std::pair<double, double> SparseGraph::spectralDimension(
    int nWalks, double maxSigma, std::mt19937 *rng,
    double tailFraction, int nTimes, double tMin, int krylovDim) const {
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  if (nNodes_ == 0) return {NaN, NaN};

  int n = std::min<int>(nWalks, static_cast<int>(nNodes_));
  if (n <= 0) return {NaN, NaN};

  // Pick n random starts without replacement.
  std::vector<std::uint32_t> all(nNodes_);
  for (std::uint32_t i = 0; i < nNodes_; ++i) all[i] = i;
  std::shuffle(all.begin(), all.end(), *rng);
  std::vector<std::uint32_t> starts(all.begin(), all.begin() + n);

  // Log-spaced t grid in [tMin, maxSigma].
  if (nTimes < 2) return {NaN, NaN};
  std::vector<double> times(nTimes);
  double logMin = std::log(tMin);
  double logMax = std::log(maxSigma);
  for (int j = 0; j < nTimes; ++j) {
    double f = static_cast<double>(j) / (nTimes - 1);
    times[j] = std::exp(logMin + f * (logMax - logMin));
  }

  auto K = diagonalHeatKernel(starts, times, krylovDim);

  // Average K over starts.
  std::vector<double> Kavg(nTimes, 0.0);
  for (int j = 0; j < nTimes; ++j) {
    double s = 0.0;
    for (int w = 0; w < n; ++w) s += K[static_cast<std::size_t>(w) * nTimes + j];
    Kavg[j] = s / n;
  }

  // Centered finite differences on (log t, log K).  Skip
  // non-positive / non-finite samples.
  std::vector<double> logT, logK;
  logT.reserve(nTimes);
  logK.reserve(nTimes);
  for (int j = 0; j < nTimes; ++j) {
    if (Kavg[j] > 0.0 && std::isfinite(Kavg[j])) {
      logT.push_back(std::log(times[j]));
      logK.push_back(std::log(Kavg[j]));
    }
  }
  if (logT.size() < 2) return {NaN, NaN};

  std::vector<double> ds(logT.size());
  for (std::size_t i = 0; i + 1 < logT.size(); ++i) {
    if (i == 0) {
      ds[0] = (logK[1] - logK[0]) / (logT[1] - logT[0]);
    } else {
      ds[i] = (logK[i + 1] - logK[i - 1]) / (logT[i + 1] - logT[i - 1]);
    }
  }
  ds.back() = (logK[logT.size() - 1] - logK[logT.size() - 2])
            / (logT[logT.size() - 1] - logT[logT.size() - 2]);
  for (auto &d : ds) d *= -2.0;

  std::size_t nTail = std::max<std::size_t>(
      1, static_cast<std::size_t>(ds.size() * tailFraction));
  double small = 0.0, large = 0.0;
  for (std::size_t i = 0; i < nTail; ++i) {
    small += ds[i];
    large += ds[ds.size() - 1 - i];
  }
  return {small / nTail, large / nTail};
}

}  // namespace tessera
