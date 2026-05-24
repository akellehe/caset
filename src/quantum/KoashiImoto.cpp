#include "quantum/KoashiImoto.hpp"

#include <unsupported/Eigen/KroneckerProduct>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace tessera::quantum {

// ── Helpers ────────────────────────────────────────────────────────────

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

Eigen::MatrixXcd conditionalB(const Eigen::MatrixXcd& rhoAB,
                              const Eigen::VectorXcd& aState,
                              int dimA, int dimB, double eps) {
    // Compute weight = ⟨a|ρ_A|a⟩.
    std::complex<double> weight(0.0, 0.0);
    for (int i = 0; i < dimA; ++i) {
        for (int j = 0; j < dimA; ++j) {
            const auto coef = std::conj(aState(i)) * aState(j);
            for (int b = 0; b < dimB; ++b) {
                weight += coef * rhoAB(i * dimB + b, j * dimB + b);
            }
        }
    }
    if (std::abs(weight) < eps) {
        // Conditional is undefined when the projection has zero weight;
        // return the maximally-mixed state as a well-defined sentinel.
        return Eigen::MatrixXcd::Identity(dimB, dimB) / static_cast<double>(dimB);
    }

    Eigen::MatrixXcd sigma = Eigen::MatrixXcd::Zero(dimB, dimB);
    for (int b = 0; b < dimB; ++b) {
        for (int bp = 0; bp < dimB; ++bp) {
            std::complex<double> sum(0.0, 0.0);
            for (int i = 0; i < dimA; ++i) {
                for (int j = 0; j < dimA; ++j) {
                    sum += std::conj(aState(i)) * aState(j)
                         * rhoAB(i * dimB + b, j * dimB + bp);
                }
            }
            sigma(b, bp) = sum / weight;
        }
    }
    return sigma;
}

// ── KI decomposition ──────────────────────────────────────────────────

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

// Reverse the columns of a matrix in place (Eigen lacks a true rowwise
// reverse for MatrixXcd that yields an MatrixXcd directly in older 3.x).
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

    // 1. Spectral decomposition of ρ_A and ρ_B (descending order, per
    //    our canonicalization convention).
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

    // 2. Transform ρ_AB into the (A-eigvec ⊗ B-eigvec) basis. This makes
    //    the marginals diagonal and exposes the bipartite block structure
    //    directly in the matrix entries.
    Eigen::MatrixXcd U = Eigen::kroneckerProduct(evecsA, evecsB).eval();
    Eigen::MatrixXcd rhoP = U.adjoint() * rhoAB * U;

    // 3. Build conditional B-states for each A-eigvec (in the eigenbasis,
    //    this is just the diagonal block(i, i) on B, normalized).
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

    // 4. Build the block-membership graph. Two A-eigvec indices i, j are
    //    in the same block iff:
    //      (a) they are coherently coupled — i.e. the off-diagonal block
    //          rhoP.block(i*dimB, j*dimB, dimB, dimB) has Frobenius norm
    //          above epsKiCondState, OR
    //      (b) they have the same conditional B-state (within
    //          epsKiCondState) — equivalent under conditioning, so they
    //          collapse into the same classical j-label with an R-multiplicity.
    //
    // (a) captures Bell-like quantum coherence; (b) captures product-like
    // structures where multiple A-eigvecs are indistinguishable via B.
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

    // 5. Collect blocks (indices grouped by union-find root, ordered by
    //    descending total weight = sum of A-eigvals in the block).
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

    // 6. For each block, extract the per-block (Σ, A', B') contributions.
    KoashiImotoResult result;
    std::vector<Eigen::MatrixXcd> sigmaBlocks, aPrimeBlocks, bPrimeBlocks;

    for (auto& [weight, aIdxs] : ordered) {
        KoashiImotoBlock blk;
        blk.weight = weight;

        // Determine which B-eigvecs are in this block: those that have
        // weight in the diagonal of rhoP restricted to the block's A
        // subspace. Sum the A-diagonal blocks and read off B-support.
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
        if (bIdxs.empty()) {
            // Degenerate: empty B-support means this block contributes
            // a trivial 1-dim core and a max-mixed A-tail.
            blk.dimLeftA  = 1;
            blk.dimLeftB  = 1;
            blk.dimRightA = static_cast<int>(aIdxs.size());
            blk.dimRightB = dimB;
            blk.coreState = Eigen::MatrixXcd::Identity(1, 1);
            blk.tailA     = Eigen::MatrixXcd::Identity(
                                blk.dimRightA, blk.dimRightA)
                            / static_cast<double>(blk.dimRightA);
            blk.tailB     = Eigen::MatrixXcd::Identity(
                                blk.dimRightB, blk.dimRightB)
                            / static_cast<double>(blk.dimRightB);
            sigmaBlocks.push_back(blk.weight * blk.coreState);
            aPrimeBlocks.push_back(blk.weight * blk.tailA);
            bPrimeBlocks.push_back(blk.weight * blk.tailB);
            result.blocks.push_back(std::move(blk));
            continue;
        }

        // Classify block as "L-dominant" (Bell-like — eigvecs coherently
        // coupled, distinct cond-states) or "R-dominant" (product-like —
        // eigvecs all share the same cond-state). The classification is
        // a heuristic that handles the common cases; mixed blocks (some
        // L and some R structure) collapse to L-dominant which over-
        // counts the L dim but stays numerically faithful.
        bool allCondStatesEqual = true;
        for (size_t i = 1; i < aIdxs.size(); ++i) {
            if ((condB[aIdxs[i]] - condB[aIdxs[0]]).norm()
                > tol.epsKiCondState) {
                allCondStatesEqual = false;
                break;
            }
        }

        if (allCondStatesEqual && aIdxs.size() > 1) {
            // R-dominant: L is trivial (dim 1), all A-eigvecs are in R.
            blk.dimLeftA  = 1;
            blk.dimLeftB  = 1;
            blk.dimRightA = static_cast<int>(aIdxs.size());
            blk.dimRightB = static_cast<int>(bIdxs.size());
            blk.coreState = Eigen::MatrixXcd::Identity(1, 1);

            // tailA: the block's contribution to ρ_A divided by weight,
            // projected onto the A-eigvec subspace of the block.
            Eigen::MatrixXcd tA = Eigen::MatrixXcd::Zero(
                blk.dimRightA, blk.dimRightA);
            for (int ri = 0; ri < blk.dimRightA; ++ri) {
                tA(ri, ri) = weightA[aIdxs[ri]] / weight;
            }
            blk.tailA = tA;

            // tailB: B-marginal of the block's rhoP, divided by weight.
            Eigen::MatrixXcd tBfull = blockSumB / weight;
            Eigen::MatrixXcd tB = Eigen::MatrixXcd::Zero(
                blk.dimRightB, blk.dimRightB);
            for (int ri = 0; ri < blk.dimRightB; ++ri) {
                for (int ci = 0; ci < blk.dimRightB; ++ci) {
                    tB(ri, ci) = tBfull(bIdxs[ri], bIdxs[ci]);
                }
            }
            blk.tailB = tB;
        } else {
            // L-dominant: A-eigvecs each carry their own L-label.
            blk.dimLeftA  = static_cast<int>(aIdxs.size());
            blk.dimLeftB  = static_cast<int>(bIdxs.size());
            blk.dimRightA = 1;
            blk.dimRightB = 1;

            const int coreDim = blk.dimLeftA * blk.dimLeftB;
            Eigen::MatrixXcd core = Eigen::MatrixXcd::Zero(coreDim, coreDim);
            for (int ri = 0; ri < blk.dimLeftA; ++ri) {
                for (int rb = 0; rb < blk.dimLeftB; ++rb) {
                    for (int ci = 0; ci < blk.dimLeftA; ++ci) {
                        for (int cb = 0; cb < blk.dimLeftB; ++cb) {
                            core(ri * blk.dimLeftB + rb,
                                 ci * blk.dimLeftB + cb) =
                                rhoP(aIdxs[ri] * dimB + bIdxs[rb],
                                     aIdxs[ci] * dimB + bIdxs[cb]);
                        }
                    }
                }
            }
            if (weight > tol.epsKiSvd) {
                core /= weight;
            }
            blk.coreState = core;
            blk.tailA = Eigen::MatrixXcd::Identity(1, 1);
            blk.tailB = Eigen::MatrixXcd::Identity(1, 1);
        }

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
