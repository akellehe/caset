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

    // 6. Build the symmetric counterparts on the B side. For each
    //    B-eigvec \f$|b\rangle\f$, the conditional A-state is
    //    \f[
    //      \sigma_b^A \;=\;
    //        \frac{\bigl(I_A \otimes \langle b|\bigr)\,
    //              \rho_{AB}\,
    //              \bigl(I_A \otimes |b\rangle\bigr)}
    //             {\mathrm{Tr}\bigl[I_A \otimes |b\rangle\langle b|
    //              \cdot \rho_{AB}\bigr]} .
    //    \f]
    //    In the AB-eigenbasis (\f$\rho'\f$ = `rhoP`), this is the
    //    \f$d_A \times d_A\f$ matrix
    //    \f$\sigma_b^A[i, i'] = \rho'[i\,d_B + b,\; i'\,d_B + b] / w_b\f$,
    //    where the weight \f$w_b = \langle b|\rho_B|b\rangle\f$ is just
    //    the corresponding B-eigenvalue (since \f$\rho_B\f$ is diagonal
    //    in this basis).
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

    // 7. Per-block construction. One single algorithm — no branches
    //    on the shape of the block, no pure/product/Bell special cases.
    //
    //    Setup: each connected component \f$j\f$ from step 5 has
    //    A-eigvec indices \f$\{i : i \in \text{block}_j\}\f$. The
    //    block's B-eigvec support is read off the diagonal of
    //    \f$\sum_{i \in \text{block}_j} \rho'[i,\cdot,i,\cdot]\f$
    //    (the per-block B-marginal in the eigenbasis is diagonal).
    //
    //    KI within block \f$j\f$:
    //    \f[
    //      \rho_{AB}\big|_{\text{block }j}
    //        \;=\; p_j \cdot \rho_{L,j}\;\otimes\;\omega_{R^A_j}
    //                       \;\otimes\;\omega_{R^B_j}
    //    \f]
    //    with the factorization \f$\mathcal{H}_{A_j} = \mathcal{H}_{L^A_j}
    //    \otimes \mathcal{H}_{R^A_j}\f$ (similarly for B).
    //
    //    The L/R dimensions on each side are determined by a single
    //    count: how many distinct conditional states the eigvecs in
    //    the block split into. Two A-eigvecs \f$|a_i\rangle, |a_{i'}\rangle\f$
    //    sharing the same conditional B-state \f$\sigma_i^B = \sigma_{i'}^B\f$
    //    are equivalent under B's view — they share an L-label and
    //    contribute a multiplicity to \f$\mathcal{H}_{R^A_j}\f$. Distinct
    //    conditional states give distinct L-labels.
    //
    //    Concretely, if a block has \f$|aIdxs| = m_A\f$ A-eigvecs that
    //    fall into \f$K_A\f$ groups (each of size \f$r_A\f$, with
    //    \f$m_A = K_A \cdot r_A\f$):
    //    \f[
    //      \dim \mathcal{H}_{L^A_j} = K_A,
    //      \qquad
    //      \dim \mathcal{H}_{R^A_j} = r_A.
    //    \f]
    //    Same on the B side with \f$K_B, r_B\f$.
    //
    //    Construction:
    //    - \f$L^A\f$ basis: one representative A-eigvec from each group.
    //    - \f$L^B\f$ basis: one representative B-eigvec from each group.
    //    - \f$\rho_{L,j}\f$: the \f$K_A K_B \times K_A K_B\f$ submatrix
    //      of \f$\rho'\f$ indexed by the cross product of L^A and L^B
    //      representatives, renormalised by \f$p_j\f$.
    //    - \f$\omega_{R^A_j}\f$: the eigenvalue spread of one A-group,
    //      \f$\mathrm{diag}(\lambda_{i_1}, \ldots, \lambda_{i_{r_A}}) /
    //         \sum_k \lambda_{i_k}\f$ (any group gives the same answer
    //      up to relabelling, by KI's R-uniformity).
    //    - \f$\omega_{R^B_j}\f$: symmetric.
    KoashiImotoResult result;
    std::vector<Eigen::MatrixXcd> sigmaBlocks, aPrimeBlocks, bPrimeBlocks;

    // Closure: group indices `idxs` by equality of `getMatrix(i)` within
    // Frobenius distance `eq`. Returns the equivalence classes as a list
    // of index-lists, where each inner list holds the original `idxs`
    // entries that share a common matrix. Used for both the A-side group
    // (by conditional B-state) and the B-side group (by conditional
    // A-state), so the partitioning rule is identical on both sides.
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

        // Identify B-side support of this block from the diagonal of
        // \f$\sum_{i \in aIdxs} \rho'\!\!\restriction_{(i, b)(i, b')}\f$.
        // In the AB-eigenbasis the off-block-diagonal entries on B are
        // negligible (by the same coupling logic that defined the block
        // on the A side), so a positive diagonal entry pins membership.
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
            // No B-support means \f$p_j \approx 0\f$ for this block;
            // it contributes a trivial 1-dim slot to \f$\Sigma\f$ and
            // empty contributions to \f$A', B'\f$. Skip it.
            continue;
        }

        // Partition A-eigvecs by conditional B-state. Each equivalence
        // class is an L-label; its size is the R-multiplicity.
        // \f[ \{|a_i\rangle : i \in \text{aIdxs}\}
        //     \;\to\;
        //     \bigsqcup_{k=1}^{K_A} \mathrm{Group}_k^A,\;
        //     \forall i \in \mathrm{Group}_k^A:\; \sigma_i^B = \sigma_k^B. \f]
        auto groupsA = groupByMatrixEquivalence(aIdxs, condB,
                                                tol.epsKiCondState);
        auto groupsB = groupByMatrixEquivalence(bIdxs, condA,
                                                tol.epsKiCondState);
        const int K_A = static_cast<int>(groupsA.size());
        const int K_B = static_cast<int>(groupsB.size());
        const int r_A = static_cast<int>(groupsA.front().size());
        const int r_B = static_cast<int>(groupsB.front().size());

        // Uniformity check. KI guarantees uniform R-multiplicities
        // within a block (every L-label has the same R-copy count).
        // Non-uniform groups indicate either numerical noise pushing
        // a near-degenerate decomposition off the manifold of clean
        // KI states, or a state that's not a single KI block. We
        // detect both and proceed with K_A, r_A as defined; the result
        // may not exactly reconstruct the input but stays trace-faithful.
        bool uniformA = true, uniformB = true;
        for (const auto& g : groupsA)
            if (static_cast<int>(g.size()) != r_A) { uniformA = false; break; }
        for (const auto& g : groupsB)
            if (static_cast<int>(g.size()) != r_B) { uniformB = false; break; }
        (void)uniformA; (void)uniformB;
        // TODO: when !uniformA || !uniformB, split the block into
        // sub-blocks (one per (groupA_size, groupB_size) signature)
        // to recover a clean KI factorisation. For valid KI-decomposable
        // inputs this branch is unreachable; for noisy inputs it
        // introduces \f$O(\text{tol})\f$ error in the reconstruction.

        blk.dimLeftA  = K_A;
        blk.dimLeftB  = K_B;
        blk.dimRightA = r_A;
        blk.dimRightB = r_B;

        // Representative L-basis: one A-eigvec per group, one B-eigvec
        // per group. Canonical choice: the first index of each group,
        // which (since `aIdxs` and `bIdxs` are sorted ascending and
        // `groupByMatrixEquivalence` preserves order) is deterministic
        // across runs.
        std::vector<int> lAIdx(K_A), lBIdx(K_B);
        for (int k = 0; k < K_A; ++k) lAIdx[k] = groupsA[k].front();
        for (int k = 0; k < K_B; ++k) lBIdx[k] = groupsB[k].front();

        // \f$\rho_{L,j}\f$: the \f$K_A K_B\f$-square core, indexed by
        // pairs (representative A, representative B). The basis order
        // is \f$|k_A\rangle \otimes |k_B\rangle\f$, i.e. row index
        // \f$k_A K_B + k_B\f$.
        // \f[
        //   \rho_{L,j}[k_A K_B + k_B, k'_A K_B + k'_B]
        //   = \frac{\rho'[\,\text{lAIdx}[k_A] \cdot d_B + \text{lBIdx}[k_B],
        //                  \;\text{lAIdx}[k'_A] \cdot d_B + \text{lBIdx}[k'_B]\,]}
        //          {p_j} .
        // \f]
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
        // \f$\rho_{L,j}\f$ is a density matrix on H_{L^A_j} ⊗ H_{L^B_j}
        // — its trace is 1 by definition, regardless of the block's
        // weight p_j. Normalise by the actual core trace (which for a
        // uniform L⊗R block carries the within-block accounting:
        // pulling one L-representative captures weight / (r_A · r_B),
        // and dividing by that recovers a trace-1 density matrix on
        // the L subspaces).
        const double coreTrace = core.trace().real();
        if (coreTrace > tol.epsKiSvd) {
            core /= coreTrace;
        }
        blk.coreState = std::move(core);

        // \f$\omega_{R^A_j}\f$: the eigenvalue spread within one A-group,
        // normalised. Pick group 0 (deterministic).
        // \f[
        //   \omega_{R^A_j}[r, r]
        //   = \frac{\lambda_{\text{groupsA}[0][r]}}
        //          {\sum_{r'} \lambda_{\text{groupsA}[0][r']}} .
        // \f]
        // For a pure block (r_A = 1) this collapses to the trivial
        // \f$1 \times 1\f$ identity; for a product block (K_A = 1)
        // this recovers the full marginal \f$\rho_A\f$ restricted to
        // the block.
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

        // \f$\omega_{R^B_j}\f$: symmetric on the B side.
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

        // Accumulate this block's contribution to \f$\Sigma, A', B'\f$.
        // The block-diagonal layout (rather than additive sum) is the
        // explicit classical register \f$\{|j\rangle\}\f$ that lets
        // downstream KI on the resulting Σ vertex decompose against
        // its (A-side, B-side) factorisation.
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
