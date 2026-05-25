// Symmetric Koashi-Imoto decomposition of a bipartite quantum state.
//
// For ρ_AB on H_A ⊗ H_B, the symmetric KI decomposition gives:
//
//   H_A = ⊕_j H_{A^L_j} ⊗ H_{A^R_j}
//   H_B = ⊕_j H_{B^L_j} ⊗ H_{B^R_j}
//   ρ_AB = ⊕_j p_j · ρ_{A^L_j B^L_j} ⊗ ω_{A^R_j} ⊗ ω_{B^R_j}
//
// The L-parts on both sides hold the joint correlation; the R-parts on
// each side are uncorrelated with the other side. j is a classical
// register both sides agree on. Single general algorithm — no
// pure/product shortcuts; those fall out as degenerate cases.
//
// References:
//   Koashi, Imoto. "Operations that do not disturb partially known
//   quantum states." Phys. Rev. A 66, 022318 (2002).
//   Hayden, Jozsa, Petz, Winter. "Structure of states which satisfy
//   strong subadditivity of quantum entropy with equality." Comm.
//   Math. Phys. 246 (2004), 359.

#pragma once

#include <Eigen/Dense>

#include <vector>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

struct KoashiImotoTolerances {
    double epsKiEigen     = 1e-10;
    double epsKiCondState = 1e-10;
    double epsKiSvd       = 1e-10;
};

struct KoashiImotoBlock {
    double           weight;
    Eigen::MatrixXcd coreState;
    Eigen::MatrixXcd tailA;
    Eigen::MatrixXcd tailB;
    int              dimLeftA{0};
    int              dimLeftB{0};
    int              dimRightA{0};
    int              dimRightB{0};
};

struct KoashiImotoResult {
    Eigen::MatrixXcd               sigma;
    Eigen::MatrixXcd               aPrime;
    Eigen::MatrixXcd               bPrime;
    std::vector<KoashiImotoBlock>  blocks;
};

[[nodiscard]] KoashiImotoResult
koashiImotoDecompose(const Eigen::MatrixXcd&        rhoAB,
                     int                            dimA,
                     int                            dimB,
                     const KoashiImotoTolerances&   tol = {});

// Partial trace over the B factor. rhoAB is (dimA·dimB)-square in (A⊗B)
// ordering (row idx = a·dimB + b); output is dimA-square.
[[nodiscard]] Eigen::MatrixXcd
partialTraceB(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

// Partial trace over the A factor.
[[nodiscard]] Eigen::MatrixXcd
partialTraceA(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

// Mutual information I(A:B) = S(A) + S(B) − S(AB) in nats.
// Floors at 0 (numerical noise can push otherwise-product inputs
// slightly negative).
[[nodiscard]] double
mutualInformation(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

} // namespace tessera::quantum
