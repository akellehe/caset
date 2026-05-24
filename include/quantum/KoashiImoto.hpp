// Symmetric Koashi-Imoto decomposition of a bipartite quantum state.
//
// Given ρ_AB on H_A ⊗ H_B, the symmetric KI decomposition gives:
//
//   H_A = ⊕_j H_{A^L_j} ⊗ H_{A^R_j}
//   H_B = ⊕_j H_{B^L_j} ⊗ H_{B^R_j}
//   ρ_AB = ⊕_j p_j · ρ_{A^L_j B^L_j} ⊗ ω_{A^R_j} ⊗ ω_{B^R_j}
//
// where the L-parts on both sides hold the joint correlation, the
// R-parts on each side are uncorrelated with the other, and the
// j-index is a classical register both sides agree on.
//
// The simulation uses this in `interact(A, B)` to produce three new
// child vertices (Σ_{A,B}, A', B') whose states are the three
// pieces of the decomposition. The decomposition is information-
// preserving — A and B retain their original states, the children
// carry the structure.
//
// Single algorithm, no shortcuts for pure or product inputs; those
// fall out as degenerate cases of the general routine.
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

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

// Numerical tolerances controlling the decomposition. Each is an
// unavoidable floating-point approximation, not a removable shortcut:
//   • epsKiEigen     — eigenvalues within this band are clustered into a
//                       single degenerate block. Sets the resolution at
//                       which the L/R structure is identified.
//   • epsKiCondState — conditional B-states (or A-states) within this
//                       Frobenius distance are treated as the same state
//                       when grouping A-eigenvectors into L/R subspaces.
//   • epsKiSvd       — singular values below this are truncated to zero
//                       in the canonicalization step.
// All default to 1e-10. Tightening or relaxing these will affect the
// decomposition on near-degenerate inputs.
struct KoashiImotoTolerances {
    double epsKiEigen     = 1e-10;
    double epsKiCondState = 1e-10;
    double epsKiSvd       = 1e-10;
};

// One block in the decomposition. Indexed by `j` in the math; this
// struct holds the per-block data so the result is inspectable from
// tests without re-running the algorithm.
struct KoashiImotoBlock {
    double           weight;       // p_j ∈ [0, 1], with Σ_j p_j = 1
    Eigen::MatrixXcd coreState;    // ρ_{A^L_j B^L_j} on H_{A^L_j} ⊗ H_{B^L_j}
    Eigen::MatrixXcd tailA;        // ω_{A^R_j} on H_{A^R_j}
    Eigen::MatrixXcd tailB;        // ω_{B^R_j} on H_{B^R_j}
    int              dimLeftA{0};  // dim H_{A^L_j}
    int              dimLeftB{0};  // dim H_{B^L_j}
    int              dimRightA{0}; // dim H_{A^R_j}
    int              dimRightB{0}; // dim H_{B^R_j}
};

// Result of one decomposition. The three child states are the
// quantities that get attached to Σ_{A,B}, A', B' on the simulation
// side; the per-block breakdown is preserved for inspection.
//
//   sigma  = ⊕_j p_j |j⟩⟨j| ⊗ coreState_j
//   aPrime = ⊕_j p_j tailA_j
//   bPrime = ⊕_j p_j tailB_j
//
// The classical register is laid out as the outermost block of `sigma`
// (one block per j, of dim dimLeftA_j · dimLeftB_j), so that downstream
// KI on Σ vs a third vertex can see the (A-side, B-side) factorization
// directly. `blocks` holds the per-j data in the same order they appear
// in `sigma`/`aPrime`/`bPrime`.
struct KoashiImotoResult {
    Eigen::MatrixXcd               sigma;
    Eigen::MatrixXcd               aPrime;
    Eigen::MatrixXcd               bPrime;
    std::vector<KoashiImotoBlock>  blocks;
};

// The decomposition itself. ρ_AB is supplied as a (dimA·dimB)-square
// matrix in the natural (A ⊗ B) ordering: row index = a·dimB + b. The
// caller specifies dimA and dimB explicitly because there's no way to
// infer the factorization from the matrix shape alone.
//
// Throws std::invalid_argument when:
//   • rhoAB is not square
//   • rhoAB.rows() != dimA · dimB
//   • dimA or dimB is non-positive
//
// Canonicalization (so the output is bitwise-reproducible across runs):
//   • eigenvalues sorted descending
//   • classical blocks ordered by descending p_j
//   • within-block basis ordered by Schmidt coefficient descending
//   • the two-sided iteration always starts with the A side
[[nodiscard]] KoashiImotoResult
koashiImotoDecompose(const Eigen::MatrixXcd&        rhoAB,
                     int                            dimA,
                     int                            dimB,
                     const KoashiImotoTolerances&   tol = {});

// ── Helpers exposed for unit tests and use by `interact()`. ─────────

// Partial trace over the B factor. Input rhoAB is (dimA·dimB)-square,
// output is dimA-square: (rhoA)[i,j] = Σ_b rhoAB[(i,b),(j,b)].
[[nodiscard]] Eigen::MatrixXcd
partialTraceB(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

// Partial trace over the A factor. Symmetric counterpart.
[[nodiscard]] Eigen::MatrixXcd
partialTraceA(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

// Mutual information I(A:B) = S(A) + S(B) - S(AB) in nats. Computed
// from rhoAB directly (no caching). Floors at 0 (numerical noise can
// produce small negatives on otherwise-product inputs).
[[nodiscard]] double
mutualInformation(const Eigen::MatrixXcd& rhoAB, int dimA, int dimB);

// Conditional B-state given A is in pure state |a⟩:
//   σ^B = Tr_A((|a⟩⟨a| ⊗ I_B) ρ_AB (|a⟩⟨a| ⊗ I_B)) / Tr(|a⟩⟨a| ⊗ I_B · ρ_AB)
// Returns the dimB × dimB matrix. When the projection weight is below
// `eps`, returns the (dimB) maximally-mixed state as a sentinel (the
// conditional is undefined when ⟨a|ρ_A|a⟩ ≈ 0).
[[nodiscard]] Eigen::MatrixXcd
conditionalB(const Eigen::MatrixXcd& rhoAB,
             const Eigen::VectorXcd& aState,
             int dimA, int dimB,
             double eps = 1e-12);

} // namespace tessera::quantum
