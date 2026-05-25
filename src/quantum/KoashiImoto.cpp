#include "quantum/KoashiImoto.hpp"

#include <unsupported/Eigen/KroneckerProduct>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace tessera::quantum {

Eigen::MatrixXcd partialTraceB(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB) {
    Eigen::MatrixXcd rhoA = Eigen::MatrixXcd::Zero(dimA, dimA);
    for (int i = 0; i < dimA; ++i) {
        for (int j = 0; j < dimA; ++j) {
            std::complex<double> sum(0.0, 0.0);
            for (int b = 0; b < dimB; ++b) {
                sum += rhoAB(i * dimB + b, j * dimB + b);
            }
            rhoA(i, j) = sum;
        }
    }
    return rhoA;
}

Eigen::MatrixXcd partialTraceA(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB) {
    Eigen::MatrixXcd rhoB = Eigen::MatrixXcd::Zero(dimB, dimB);
    for (int b = 0; b < dimB; ++b) {
        for (int bp = 0; bp < dimB; ++bp) {
            std::complex<double> sum(0.0, 0.0);
            for (int a = 0; a < dimA; ++a) {
                sum += rhoAB(a * dimB + b, a * dimB + bp);
            }
            rhoB(b, bp) = sum;
        }
    }
    return rhoB;
}

namespace {

double vonNeumann(const Eigen::MatrixXcd& rho, double tol = 1e-12) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    if (es.info() != Eigen::Success) return 0.0;
    double s = 0.0;
    const auto& evs = es.eigenvalues();
    for (Eigen::Index i = 0; i < evs.size(); ++i) {
        const double p = evs[i];
        if (p > tol) s -= p * std::log(p);
    }
    return s;
}

} // namespace

double mutualInformation(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB) {
    const auto rhoA = partialTraceB(rhoAB, dimA, dimB);
    const auto rhoB = partialTraceA(rhoAB, dimA, dimB);
    const double I = vonNeumann(rhoA) + vonNeumann(rhoB) - vonNeumann(rhoAB);
    return std::max(0.0, I);
}

namespace {

class UnionFind {
  public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
  private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

Eigen::MatrixXcd blockDiagonal(const std::vector<Eigen::MatrixXcd>& blocks) {
    int total = 0;
    for (const auto& b : blocks) total += static_cast<int>(b.rows());
    if (total == 0) return Eigen::MatrixXcd::Identity(1, 1);
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(total, total);
    int offset = 0;
    for (const auto& b : blocks) {
        const int n = static_cast<int>(b.rows());
        if (n > 0) out.block(offset, offset, n, n) = b;
        offset += n;
    }
    return out;
}

Eigen::MatrixXcd columnReverse(const Eigen::MatrixXcd& M) {
    Eigen::MatrixXcd out(M.rows(), M.cols());
    for (Eigen::Index c = 0; c < M.cols(); ++c) {
        out.col(c) = M.col(M.cols() - 1 - c);
    }
    return out;
}

} // namespace

KoashiImotoResult
koashiImotoDecompose(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB,
                     const KoashiImotoTolerances& tol) {
    if (dimA <= 0 || dimB <= 0) {
        throw std::invalid_argument("KI: dimA, dimB must be positive");
    }
    if (rhoAB.rows() != rhoAB.cols()) {
        throw std::invalid_argument("KI: rhoAB must be square");
    }
    if (rhoAB.rows() != static_cast<Eigen::Index>(dimA) * dimB) {
        throw std::invalid_argument(
            "KI: rhoAB dimension must equal dimA · dimB");
    }

    const Eigen::MatrixXcd rhoA = partialTraceB(rhoAB, dimA, dimB);
    const Eigen::MatrixXcd rhoB = partialTraceA(rhoAB, dimA, dimB);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> esA(rhoA);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> esB(rhoB);
    if (esA.info() != Eigen::Success || esB.info() != Eigen::Success) {
        throw std::runtime_error("KI: eigendecomposition failed");
    }
    Eigen::VectorXd  evalsA = esA.eigenvalues().reverse();
    Eigen::MatrixXcd evecsA = columnReverse(esA.eigenvectors());
    Eigen::VectorXd  evalsB = esB.eigenvalues().reverse();
    Eigen::MatrixXcd evecsB = columnReverse(esB.eigenvectors());

    Eigen::MatrixXcd U = Eigen::kroneckerProduct(evecsA, evecsB).eval();
    Eigen::MatrixXcd rhoP = U.adjoint() * rhoAB * U;

    std::vector<Eigen::MatrixXcd> condB(dimA);
    std::vector<double>           weightA(dimA);
    for (int i = 0; i < dimA; ++i) {
        Eigen::MatrixXcd block = rhoP.block(i * dimB, i * dimB, dimB, dimB);
        const double w = block.trace().real();
        weightA[i] = w;
        if (w > tol.epsKiEigen) {
            condB[i] = block / w;
        } else {
            condB[i] = Eigen::MatrixXcd::Identity(dimB, dimB)
                       / static_cast<double>(dimB);
        }
    }

    // Block-membership graph: same conditional-state group OR coherent
    // coupling via the off-diagonal block in the (A-eigvec, B-eigvec)
    // basis.
    UnionFind uf(dimA);
    for (int i = 0; i < dimA; ++i) {
        for (int j = i + 1; j < dimA; ++j) {
            const double offDiagNorm =
                rhoP.block(i * dimB, j * dimB, dimB, dimB).norm();
            const double condDiff = (condB[i] - condB[j]).norm();
            if (offDiagNorm > tol.epsKiCondState
                || condDiff < tol.epsKiCondState) {
                uf.unite(i, j);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> blockMap;
    for (int i = 0; i < dimA; ++i) {
        blockMap[uf.find(i)].push_back(i);
    }
    std::vector<std::pair<double, std::vector<int>>> ordered;
    ordered.reserve(blockMap.size());
    for (auto& [root, idxs] : blockMap) {
        std::sort(idxs.begin(), idxs.end());
        double w = 0.0;
        for (int i : idxs) w += weightA[i];
        ordered.emplace_back(w, std::move(idxs));
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& l, const auto& r) { return l.first > r.first; });

    std::vector<Eigen::MatrixXcd> condA(dimB);
    std::vector<double>           weightB(dimB);
    for (int b = 0; b < dimB; ++b) {
        const double w = evalsB[b];
        weightB[b] = w;
        Eigen::MatrixXcd s = Eigen::MatrixXcd::Zero(dimA, dimA);
        for (int i = 0; i < dimA; ++i) {
            for (int ip = 0; ip < dimA; ++ip) {
                s(i, ip) = rhoP(i * dimB + b, ip * dimB + b);
            }
        }
        if (w > tol.epsKiEigen) {
            condA[b] = s / w;
        } else {
            condA[b] = Eigen::MatrixXcd::Identity(dimA, dimA)
                       / static_cast<double>(dimA);
        }
    }

    KoashiImotoResult result;
    std::vector<Eigen::MatrixXcd> sigmaBlocks, aPrimeBlocks, bPrimeBlocks;

    auto groupByMatrixEquivalence =
        [&](const std::vector<int>& idxs,
            const std::vector<Eigen::MatrixXcd>& matrices,
            double eq) -> std::vector<std::vector<int>>
    {
        std::vector<std::vector<int>> groups;
        std::vector<bool> used(idxs.size(), false);
        for (size_t k = 0; k < idxs.size(); ++k) {
            if (used[k]) continue;
            std::vector<int> g{idxs[k]};
            used[k] = true;
            for (size_t l = k + 1; l < idxs.size(); ++l) {
                if (used[l]) continue;
                if ((matrices[idxs[k]] - matrices[idxs[l]]).norm() < eq) {
                    g.push_back(idxs[l]);
                    used[l] = true;
                }
            }
            groups.push_back(std::move(g));
        }
        return groups;
    };

    for (auto& [weight, aIdxs] : ordered) {
        KoashiImotoBlock blk;
        blk.weight = weight;

        Eigen::MatrixXcd blockSumB = Eigen::MatrixXcd::Zero(dimB, dimB);
        for (int i : aIdxs) {
            blockSumB += rhoP.block(i * dimB, i * dimB, dimB, dimB);
        }
        std::vector<int> bIdxs;
        for (int b = 0; b < dimB; ++b) {
            if (std::abs(blockSumB(b, b)) > tol.epsKiCondState) {
                bIdxs.push_back(b);
            }
        }
        if (bIdxs.empty()) continue;

        auto groupsA = groupByMatrixEquivalence(aIdxs, condB,
                                                tol.epsKiCondState);
        auto groupsB = groupByMatrixEquivalence(bIdxs, condA,
                                                tol.epsKiCondState);
        const int K_A = static_cast<int>(groupsA.size());
        const int K_B = static_cast<int>(groupsB.size());
        const int r_A = static_cast<int>(groupsA.front().size());
        const int r_B = static_cast<int>(groupsB.front().size());

        blk.dimLeftA  = K_A;
        blk.dimLeftB  = K_B;
        blk.dimRightA = r_A;
        blk.dimRightB = r_B;

        std::vector<int> lAIdx(K_A), lBIdx(K_B);
        for (int k = 0; k < K_A; ++k) lAIdx[k] = groupsA[k].front();
        for (int k = 0; k < K_B; ++k) lBIdx[k] = groupsB[k].front();

        const int coreDim = K_A * K_B;
        Eigen::MatrixXcd core = Eigen::MatrixXcd::Zero(coreDim, coreDim);
        for (int kA = 0; kA < K_A; ++kA) {
            for (int kB = 0; kB < K_B; ++kB) {
                for (int kAp = 0; kAp < K_A; ++kAp) {
                    for (int kBp = 0; kBp < K_B; ++kBp) {
                        core(kA * K_B + kB, kAp * K_B + kBp) =
                            rhoP(lAIdx[kA] * dimB + lBIdx[kB],
                                 lAIdx[kAp] * dimB + lBIdx[kBp]);
                    }
                }
            }
        }
        const double coreTrace = core.trace().real();
        if (coreTrace > tol.epsKiSvd) {
            core /= coreTrace;
        }
        blk.coreState = std::move(core);

        Eigen::MatrixXcd tA = Eigen::MatrixXcd::Zero(r_A, r_A);
        double normA = 0.0;
        for (int r = 0; r < r_A; ++r) normA += weightA[groupsA[0][r]];
        if (normA > tol.epsKiSvd) {
            for (int r = 0; r < r_A; ++r) {
                tA(r, r) = weightA[groupsA[0][r]] / normA;
            }
        } else {
            tA = Eigen::MatrixXcd::Identity(r_A, r_A)
                 / static_cast<double>(r_A);
        }
        blk.tailA = std::move(tA);

        Eigen::MatrixXcd tB = Eigen::MatrixXcd::Zero(r_B, r_B);
        double normB = 0.0;
        for (int r = 0; r < r_B; ++r) normB += weightB[groupsB[0][r]];
        if (normB > tol.epsKiSvd) {
            for (int r = 0; r < r_B; ++r) {
                tB(r, r) = weightB[groupsB[0][r]] / normB;
            }
        } else {
            tB = Eigen::MatrixXcd::Identity(r_B, r_B)
                 / static_cast<double>(r_B);
        }
        blk.tailB = std::move(tB);

        sigmaBlocks.push_back(blk.weight * blk.coreState);
        aPrimeBlocks.push_back(blk.weight * blk.tailA);
        bPrimeBlocks.push_back(blk.weight * blk.tailB);
        result.blocks.push_back(std::move(blk));
    }

    result.sigma  = blockDiagonal(sigmaBlocks);
    result.aPrime = blockDiagonal(aPrimeBlocks);
    result.bPrime = blockDiagonal(bPrimeBlocks);
    return result;
}

} // namespace tessera::quantum
