// Implementation of the emergent-spectral-dimension submodule. See
// include/quantum/holography.hpp for the architectural overview and
// docs/source/holography-causal-ordering-emergent-dimension.md for the
// scientific charter and falsification criteria.

#include "quantum/holography.hpp"

#include "quantum/choi_state.hpp"
#include "quantum/tdvp_runner.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tessera::quantum {

// ─── HolographyConfig ────────────────────────────────────────────────

void HolographyConfig::validate() const {
    if (sigmaMin <= 0.0) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaMin must be > 0");
    }
    if (sigmaMax <= sigmaMin) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaMax must be > sigmaMin");
    }
    if (sigmaCount < 8) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaCount must be >= 8 for finite-difference D_S");
    }
    if (epsilonI < 0.0) {
        throw std::invalid_argument(
            "HolographyConfig: epsilonI must be non-negative");
    }
    if (maxTemporalStride < 0) {
        throw std::invalid_argument(
            "HolographyConfig: maxTemporalStride must be >= 0");
    }
    if (krylovDim < 4) {
        throw std::invalid_argument(
            "HolographyConfig: krylovDim must be >= 4");
    }
}

// ─── MutualInformationProfile ────────────────────────────────────────

MutualInformationProfile::MutualInformationProfile(
    std::vector<TDVPSnapshot> const& snapshots,
    HolographyConfig const& config)
    : nSnapshots_(static_cast<int>(snapshots.size())),
      epsilonI_(config.epsilonI) {
    if (nSnapshots_ == 0) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot list is empty");
    }
    if (snapshots.front().mutualInformation.empty()) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshots have no recorded MI; "
            "set TDVPConfig::recordMutualInformation = true");
    }

    // Derive N from the size of the first snapshot's MI matrix.
    const auto miFlat0 = snapshots.front().mutualInformation.size();
    int N = 0;
    while (static_cast<std::size_t>(N) * N < miFlat0) ++N;
    if (static_cast<std::size_t>(N) * N != miFlat0) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot MI buffer is not N×N");
    }
    nSites_ = N;

    const int nLabels = nSites_ * nSnapshots_;
    mi_.assign(static_cast<std::size_t>(nLabels) * nLabels, 0.0);

    // Copy per-snapshot N×N spatial-MI blocks onto the diagonal of
    // the global (nLabels × nLabels) matrix.
    for (int s = 0; s < nSnapshots_; ++s) {
        auto const& snap = snapshots[static_cast<std::size_t>(s)];
        if (static_cast<int>(snap.mutualInformation.size()) != N * N) {
            throw std::invalid_argument(
                "MutualInformationProfile: snapshot MI size inconsistent across snapshots");
        }
        const int base = s * nSites_;
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                const std::size_t srcIdx =
                    static_cast<std::size_t>(a * N + b);
                const std::size_t dstIdx =
                    static_cast<std::size_t>((base + a) * nLabels + (base + b));
                mi_[dstIdx] = snap.mutualInformation[srcIdx];
            }
        }
    }

    // Optional: populate cross-snapshot temporal MI blocks via the
    // Choi-state propagator. Because the Schwinger Hamiltonian is
    // time-independent, U_{s→t} depends only on the duration |t − s|;
    // we compute one Choi state per unique snapshot stride.
    if (!config.includeTemporal) return;
    if (nSnapshots_ < 2) return;

    SchwingerParams p;
    p.N = config.tdvp.N;
    p.a = config.tdvp.a;
    p.m = config.tdvp.m;
    p.g = config.tdvp.g;
    p.L0 = config.tdvp.L0;
    if (p.N != N) {
        // The snapshot MI matrices and the config N must agree — they
        // come from the same TDVP run.
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot N inconsistent with config.tdvp.N");
    }

    const double dtSnap =
        config.tdvp.dt * static_cast<double>(std::max(config.tdvp.snapshotEvery, 1));

    ChoiPropagator::TDVPSettings settings;
    settings.dt         = config.tdvp.dt;
    settings.maxBondDim = config.tdvp.maxBondDim;
    settings.krylovDim  = config.tdvp.krylovDim;
    settings.cutoff     = config.tdvp.cutoff;
    settings.quiet      = config.tdvp.quiet;

    const int strideCap =
        (config.maxTemporalStride <= 0)
            ? (nSnapshots_ - 1)
            : std::min(config.maxTemporalStride, nSnapshots_ - 1);

    // For each unique stride compute the Choi state once, extract the
    // N×N temporal MI matrix, and fan it out over every (s, s+stride)
    // snapshot pair. Store symmetrically so the global MI matrix stays
    // Hermitian.
    for (int stride = 1; stride <= strideCap; ++stride) {
        const double duration = stride * dtSnap;
        auto choi = ChoiPropagator::choiState(p, duration, settings);
        auto miMat = ChoiPropagator::temporalMutualInformation(choi, p.N);
        for (int s = 0; s + stride < nSnapshots_; ++s) {
            const int t = s + stride;
            const int baseS = s * nSites_;
            const int baseT = t * nSites_;
            for (int a = 0; a < N; ++a) {
                for (int b = 0; b < N; ++b) {
                    const double mi = miMat(a, b);
                    // Forward direction (s, a) → (t, b)
                    const std::size_t fwd =
                        static_cast<std::size_t>((baseS + a) * nLabels +
                                                  (baseT + b));
                    // Symmetric storage (t, b) → (s, a)
                    const std::size_t bwd =
                        static_cast<std::size_t>((baseT + b) * nLabels +
                                                  (baseS + a));
                    mi_[fwd] = mi;
                    mi_[bwd] = mi;
                }
            }
        }
    }
}

double MutualInformationProfile::at(int siteV, int snapV,
                                     int siteW, int snapW) const {
    const int v = snapV * nSites_ + siteV;
    const int w = snapW * nSites_ + siteW;
    return atFlat(v, w);
}

double MutualInformationProfile::atFlat(int v, int w) const noexcept {
    const int n = nLabels();
    if (v < 0 || w < 0 || v >= n || w >= n) return 0.0;
    return mi_[static_cast<std::size_t>(v) * n + w];
}

MutualInformationProfile::COO
MutualInformationProfile::weightedAdjacency() const {
    COO coo;
    const int n = nLabels();
    coo.nVertices = n;
    if (n == 0) return coo;

    // Reserve a heuristic upper bound. Each snapshot contributes at
    // most N(N-1)/2 distinct edges; we list each edge twice.
    coo.rows.reserve(static_cast<std::size_t>(n * nSites_));
    coo.cols.reserve(static_cast<std::size_t>(n * nSites_));
    coo.weights.reserve(static_cast<std::size_t>(n * nSites_));

    for (int v = 0; v < n; ++v) {
        for (int w = 0; w < n; ++w) {
            if (v == w) continue;
            const double wt =
                mi_[static_cast<std::size_t>(v) * n + w];
            if (wt > epsilonI_) {
                coo.rows.push_back(v);
                coo.cols.push_back(w);
                coo.weights.push_back(wt);
            }
        }
    }
    return coo;
}

// ─── EmergentGraph ───────────────────────────────────────────────────

EmergentGraph::EmergentGraph(MutualInformationProfile const& profile)
    : n_(profile.nLabels()) {
    // Build CSR weighted adjacency from the profile's COO.
    auto coo = profile.weightedAdjacency();
    // coo lists each undirected edge twice (v→w and w→v) already.
    nEdges_ = static_cast<int>(coo.rows.size()) / 2;

    // Bucket edges by source vertex to build CSR.
    std::vector<int> rowCount(static_cast<std::size_t>(n_) + 1, 0);
    for (auto r : coo.rows) ++rowCount[static_cast<std::size_t>(r) + 1];
    std::partial_sum(rowCount.begin(), rowCount.end(), rowCount.begin());
    indptr_ = std::move(rowCount);
    indices_.assign(coo.rows.size(), 0);
    weights_.assign(coo.rows.size(), 0.0);

    std::vector<int> cursor(static_cast<std::size_t>(n_), 0);
    for (std::size_t k = 0; k < coo.rows.size(); ++k) {
        const int r = coo.rows[k];
        const std::size_t pos =
            static_cast<std::size_t>(indptr_[static_cast<std::size_t>(r)]) +
            static_cast<std::size_t>(cursor[static_cast<std::size_t>(r)]++);
        indices_[pos] = coo.cols[k];
        weights_[pos] = coo.weights[k];
    }

    // Weighted degree per vertex.
    degrees_.assign(static_cast<std::size_t>(n_), 0.0);
    for (int v = 0; v < n_; ++v) {
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        double d = 0.0;
        for (int k = lo; k < hi; ++k) {
            d += weights_[static_cast<std::size_t>(k)];
        }
        degrees_[static_cast<std::size_t>(v)] = d;
    }
}

Eigen::SparseMatrix<double> EmergentGraph::laplacian() const {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(2 * nEdges_ + n_));
    for (int v = 0; v < n_; ++v) {
        trips.emplace_back(v, v, degrees_[static_cast<std::size_t>(v)]);
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        for (int k = lo; k < hi; ++k) {
            trips.emplace_back(v, indices_[static_cast<std::size_t>(k)],
                                -weights_[static_cast<std::size_t>(k)]);
        }
    }
    Eigen::SparseMatrix<double> L(n_, n_);
    L.setFromTriplets(trips.begin(), trips.end());
    L.makeCompressed();
    return L;
}

namespace {

// Apply L = D - W to a dense column vector v in-place, writing into out.
// L is implicit from (indptr, indices, weights, degrees).
void applyLaplacian(int n,
                     std::vector<int> const& indptr,
                     std::vector<int> const& indices,
                     std::vector<double> const& weights,
                     std::vector<double> const& degrees,
                     std::vector<double> const& v,
                     std::vector<double>& out) {
    for (int i = 0; i < n; ++i) {
        double s = degrees[static_cast<std::size_t>(i)] *
                   v[static_cast<std::size_t>(i)];
        const int lo = indptr[static_cast<std::size_t>(i)];
        const int hi = indptr[static_cast<std::size_t>(i) + 1];
        for (int k = lo; k < hi; ++k) {
            s -= weights[static_cast<std::size_t>(k)] *
                 v[static_cast<std::size_t>(indices[static_cast<std::size_t>(k)])];
        }
        out[static_cast<std::size_t>(i)] = s;
    }
}

// Padé-13 matrix exponential on a small dense matrix (for the projected
// Krylov tridiagonal). For the tiny sizes here (k ≤ 30) this is fine.
Eigen::MatrixXd matExp(Eigen::MatrixXd const& A) {
    // Eigen has no built-in matrix exp; use eigendecomposition since A
    // is symmetric (T is symmetric tridiagonal in the Lanczos basis).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto const& V = es.eigenvectors();
    auto const& d = es.eigenvalues();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    for (int i = 0; i < d.size(); ++i) D(i, i) = std::exp(d[i]);
    return V * D * V.transpose();
}

} // namespace

std::vector<double>
EmergentGraph::returnProbability(std::vector<double> const& sigmas,
                                   int krylovDim) const {
    const int nT = static_cast<int>(sigmas.size());
    std::vector<double> P(static_cast<std::size_t>(nT), 0.0);
    if (n_ == 0 || nT == 0) return P;

    // Krylov-Lanczos for each vertex v: project L onto a small symmetric
    // tridiagonal T_k starting from e_v, then [exp(-σ T_k)]_{0,0}
    // equals e_v^T exp(-σ L) e_v exactly to Krylov order.
    std::vector<double> v(static_cast<std::size_t>(n_));
    std::vector<double> w(static_cast<std::size_t>(n_));
    std::vector<double> prev(static_cast<std::size_t>(n_));
    std::vector<std::vector<double>> V;
    std::vector<double> alpha;
    std::vector<double> beta;

    for (int start = 0; start < n_; ++start) {
        // Initialise v0 = e_start, V = [v0], alpha = [], beta = [].
        std::fill(v.begin(), v.end(), 0.0);
        v[static_cast<std::size_t>(start)] = 1.0;
        V.assign(1, v);
        alpha.clear();
        beta.clear();

        const int kMax = std::min(krylovDim, n_);
        for (int j = 0; j < kMax; ++j) {
            // w = L · v
            applyLaplacian(n_, indptr_, indices_, weights_, degrees_, v, w);
            // alpha_j = ⟨v, w⟩
            double a = 0.0;
            for (int i = 0; i < n_; ++i) {
                a += v[static_cast<std::size_t>(i)] *
                     w[static_cast<std::size_t>(i)];
            }
            alpha.push_back(a);
            if (j + 1 == kMax) break;

            // w ← w − alpha_j · v − beta_{j-1} · v_{j-1}
            for (int i = 0; i < n_; ++i) {
                w[static_cast<std::size_t>(i)] -=
                    a * v[static_cast<std::size_t>(i)];
                if (j > 0) {
                    w[static_cast<std::size_t>(i)] -=
                        beta.back() *
                        prev[static_cast<std::size_t>(i)];
                }
            }
            // Full re-orthogonalisation against existing V (numerical
            // stability).
            for (auto const& u : V) {
                double dot = 0.0;
                for (int i = 0; i < n_; ++i) {
                    dot += u[static_cast<std::size_t>(i)] *
                           w[static_cast<std::size_t>(i)];
                }
                for (int i = 0; i < n_; ++i) {
                    w[static_cast<std::size_t>(i)] -=
                        dot * u[static_cast<std::size_t>(i)];
                }
            }
            double normW = 0.0;
            for (int i = 0; i < n_; ++i) {
                normW += w[static_cast<std::size_t>(i)] *
                         w[static_cast<std::size_t>(i)];
            }
            normW = std::sqrt(normW);
            if (normW < 1e-12) break;  // Krylov subspace exhausted.
            beta.push_back(normW);
            prev = v;
            const double inv = 1.0 / normW;
            for (int i = 0; i < n_; ++i) {
                v[static_cast<std::size_t>(i)] =
                    w[static_cast<std::size_t>(i)] * inv;
            }
            V.push_back(v);
        }

        const int actualK = static_cast<int>(alpha.size());
        if (actualK == 0) {
            // No iterations succeeded; identity element gives P = 1.
            for (int t = 0; t < nT; ++t)
                P[static_cast<std::size_t>(t)] += 1.0;
            continue;
        }

        // Build the symmetric tridiagonal T (actualK × actualK).
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(actualK, actualK);
        for (int i = 0; i < actualK; ++i) T(i, i) = alpha[i];
        for (int i = 0; i + 1 < actualK; ++i) {
            T(i, i + 1) = beta[i];
            T(i + 1, i) = beta[i];
        }

        // For each σ, [exp(-σ T)]_{0,0} contributes to the trace.
        for (int t = 0; t < nT; ++t) {
            const double sigma = sigmas[static_cast<std::size_t>(t)];
            Eigen::MatrixXd e = matExp(-sigma * T);
            P[static_cast<std::size_t>(t)] += e(0, 0);
        }
    }

    // Normalise to a return probability (trace / |V|).
    const double invN = 1.0 / static_cast<double>(n_);
    for (int t = 0; t < nT; ++t) P[static_cast<std::size_t>(t)] *= invN;
    return P;
}

std::vector<double>
EmergentGraph::spectralDimension(std::vector<double> const& sigmas,
                                   std::vector<double> const& P) {
    const int n = static_cast<int>(sigmas.size());
    std::vector<double> dS(static_cast<std::size_t>(n),
                            std::numeric_limits<double>::quiet_NaN());
    if (n < 2 || static_cast<int>(P.size()) != n) return dS;

    // Centered finite differences on (log σ, log P); one-sided at
    // endpoints. D_S = -2 · d log P / d log σ.
    std::vector<double> logSig(static_cast<std::size_t>(n));
    std::vector<double> logP(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double s = sigmas[static_cast<std::size_t>(i)];
        const double p = P[static_cast<std::size_t>(i)];
        if (s <= 0.0 || p <= 0.0) {
            // Leave dS[i] as NaN.
            logSig[static_cast<std::size_t>(i)] =
                std::numeric_limits<double>::quiet_NaN();
            logP[static_cast<std::size_t>(i)] =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        logSig[static_cast<std::size_t>(i)] = std::log(s);
        logP[static_cast<std::size_t>(i)]   = std::log(p);
    }

    auto valid = [](double x) {
        return std::isfinite(x);
    };

    for (int i = 0; i < n; ++i) {
        double slope;
        if (i == 0) {
            const double a = logP[1] - logP[0];
            const double b = logSig[1] - logSig[0];
            slope = (valid(a) && valid(b) && b != 0.0) ? a / b : std::nan("");
        } else if (i == n - 1) {
            const double a = logP[static_cast<std::size_t>(n - 1)] -
                              logP[static_cast<std::size_t>(n - 2)];
            const double b = logSig[static_cast<std::size_t>(n - 1)] -
                              logSig[static_cast<std::size_t>(n - 2)];
            slope = (valid(a) && valid(b) && b != 0.0) ? a / b : std::nan("");
        } else {
            const double a = logP[static_cast<std::size_t>(i + 1)] -
                              logP[static_cast<std::size_t>(i - 1)];
            const double b = logSig[static_cast<std::size_t>(i + 1)] -
                              logSig[static_cast<std::size_t>(i - 1)];
            slope = (valid(a) && valid(b) && b != 0.0) ? a / b : std::nan("");
        }
        dS[static_cast<std::size_t>(i)] = -2.0 * slope;
    }
    return dS;
}

std::string EmergentGraph::toDot() const {
    std::ostringstream os;
    os << "graph emergent {\n";
    os << "  // Weighted graph: edge label = mutual information (nats)\n";
    for (int v = 0; v < n_; ++v) {
        os << "  " << v << ";\n";
    }
    for (int v = 0; v < n_; ++v) {
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        for (int k = lo; k < hi; ++k) {
            const int w = indices_[static_cast<std::size_t>(k)];
            if (w <= v) continue;  // each undirected edge listed once
            os << std::fixed << std::setprecision(3)
               << "  " << v << " -- " << w
               << " [label=\"" << weights_[static_cast<std::size_t>(k)] << "\"];\n";
        }
    }
    os << "}\n";
    return os.str();
}

// ─── AmbjornLollFit ──────────────────────────────────────────────────

AmbjornLollFit::Result
AmbjornLollFit::fit(std::vector<double> const& sigmas,
                      std::vector<double> const& dS,
                      double sigmaFitMin,
                      double sigmaFitMax) {
    Result r;
    const int nAll = static_cast<int>(sigmas.size());
    if (nAll < 4 || static_cast<int>(dS.size()) != nAll) return r;

    // Filter to the chosen window and drop NaN points.
    std::vector<double> sFit, yFit;
    sFit.reserve(static_cast<std::size_t>(nAll));
    yFit.reserve(static_cast<std::size_t>(nAll));
    for (int i = 0; i < nAll; ++i) {
        const double s = sigmas[static_cast<std::size_t>(i)];
        const double y = dS[static_cast<std::size_t>(i)];
        if (!std::isfinite(s) || !std::isfinite(y)) continue;
        if (sigmaFitMin > 0.0 && s < sigmaFitMin) continue;
        if (sigmaFitMax > 0.0 && s > sigmaFitMax) continue;
        sFit.push_back(s);
        yFit.push_back(y);
    }
    const int n = static_cast<int>(sFit.size());
    if (n < 4) return r;

    // Initial guesses. D_∞ = max(y) (the large-σ asymptote), B = σ at
    // the median point, C set so the curve passes through that point.
    double dInf = *std::max_element(yFit.begin(), yFit.end());
    double B    = sFit[static_cast<std::size_t>(n / 2)];
    double yMid = yFit[static_cast<std::size_t>(n / 2)];
    double C    = std::max(0.0, (dInf - yMid) * (B + sFit[static_cast<std::size_t>(n / 2)]));

    // Gauss-Newton iteration on the residual r_i = D_∞ - C/(B+σ_i) - y_i.
    // Jacobian:
    //   ∂r_i/∂D_∞ = 1
    //   ∂r_i/∂C   = -1 / (B + σ_i)
    //   ∂r_i/∂B   = C / (B + σ_i)²
    constexpr int maxIter = 100;
    constexpr double tol  = 1e-10;
    for (int iter = 0; iter < maxIter; ++iter) {
        Eigen::MatrixXd J(n, 3);
        Eigen::VectorXd resid(n);
        for (int i = 0; i < n; ++i) {
            const double s = sFit[static_cast<std::size_t>(i)];
            const double inv = 1.0 / (B + s);
            const double f = dInf - C * inv;
            resid(i) = yFit[static_cast<std::size_t>(i)] - f;
            J(i, 0) =  1.0;
            J(i, 1) = -inv;
            J(i, 2) =  C * inv * inv;
        }
        Eigen::Matrix3d JtJ = J.transpose() * J;
        Eigen::Vector3d Jtr = J.transpose() * resid;
        // Levenberg-Marquardt damping for stability.
        JtJ += 1e-8 * Eigen::Matrix3d::Identity();
        Eigen::Vector3d delta = JtJ.ldlt().solve(Jtr);
        dInf += delta(0);
        C    += delta(1);
        B    += delta(2);
        if (delta.norm() < tol * (std::abs(dInf) + std::abs(C) + std::abs(B) + 1.0)) {
            break;
        }
    }

    // Reduced χ² assuming unit variance per point.
    double chi2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double s = sFit[static_cast<std::size_t>(i)];
        const double f = dInf - C / (B + s);
        const double e = yFit[static_cast<std::size_t>(i)] - f;
        chi2 += e * e;
    }
    chi2 /= std::max(1, n - 3);

    r.dInfinity  = dInf;
    r.C          = C;
    r.B          = B;
    r.chiSquared = chi2;
    return r;
}

// ─── EmergentSpectralDimension ───────────────────────────────────────

EmergentSpectralDimension::EmergentSpectralDimension(HolographyConfig config)
    : config_(std::move(config)) {
    config_.validate();
    // The pipeline needs all-pairs MI per snapshot — force the flag
    // here so callers can't trip themselves up by leaving it off.
    config_.tdvp.recordMutualInformation = true;
}

SpectralDimensionResult
EmergentSpectralDimension::computeFromSnapshots(QuenchResult const& quench) const {
    SpectralDimensionResult result;

    // (1) Build the MI profile on the (site, snapshot) label set.
    MutualInformationProfile profile(quench.snapshots, config_);
    EmergentGraph graph(profile);

    result.graphNVertices = graph.nVertices();
    result.graphNEdges    = graph.nEdges();

    // (2) σ-grid (log-spaced).
    const int nSig = config_.sigmaCount;
    std::vector<double> sigmas(static_cast<std::size_t>(nSig), 0.0);
    const double logMin = std::log(config_.sigmaMin);
    const double logMax = std::log(config_.sigmaMax);
    for (int i = 0; i < nSig; ++i) {
        const double frac =
            static_cast<double>(i) / static_cast<double>(nSig - 1);
        sigmas[static_cast<std::size_t>(i)] =
            std::exp(logMin + frac * (logMax - logMin));
    }
    result.sigmas = sigmas;

    // (3) P(σ) via the heat-kernel trace estimator.
    result.P  = graph.returnProbability(sigmas, config_.krylovDim);

    // (4) D_S(σ) via centered finite differences.
    result.dS = EmergentGraph::spectralDimension(sigmas, result.P);

    // (5) Ambjorn–Loll three-parameter fit on the full grid.
    auto fit = AmbjornLollFit::fit(sigmas, result.dS);
    result.dInfinity     = fit.dInfinity;
    result.C             = fit.C;
    result.B             = fit.B;
    result.fitChiSquared = fit.chiSquared;

    // (6) Snapshot summary copied across.
    const auto& snaps = quench.snapshots;
    result.snapshotTimes.reserve(snaps.size());
    result.snapshotBondDims.reserve(snaps.size());
    result.snapshotEnergies.reserve(snaps.size());
    for (auto const& s : snaps) {
        result.snapshotTimes.push_back(s.time);
        result.snapshotBondDims.push_back(s.bondDim);
        result.snapshotEnergies.push_back(s.energy);
    }
    return result;
}

SpectralDimensionResult
EmergentSpectralDimension::compute() const {
    // Run the TDVP pipeline; recordMutualInformation is forced on by
    // the constructor so the snapshots arrive with MI populated.
    const auto quench = SchwingerQuench{config_.tdvp}.evolve();
    return computeFromSnapshots(quench);
}

} // namespace tessera::quantum
