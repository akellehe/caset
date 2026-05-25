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

#include <utility>
#include <vector>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

/// Numerical tolerances used by ``koashiImotoDecompose``.
///
/// The KI algorithm does several rank-detection / eigendecomposition
/// steps. Each step has its own conditioning threshold, separated so
/// callers can dial them independently when chasing edge cases (in
/// practice they all sit at ~1e-10 for double precision).
class KoashiImotoTolerances {
  public:
    KoashiImotoTolerances() = default;
    KoashiImotoTolerances(double eigen, double condState, double svd) noexcept
      : epsKiEigen_(eigen), epsKiCondState_(condState), epsKiSvd_(svd) {}

    double getEpsKiEigen() const noexcept { return epsKiEigen_; }
    double getEpsKiCondState() const noexcept { return epsKiCondState_; }
    double getEpsKiSvd() const noexcept { return epsKiSvd_; }

    void setEpsKiEigen(double v) noexcept { epsKiEigen_ = v; }
    void setEpsKiCondState(double v) noexcept { epsKiCondState_ = v; }
    void setEpsKiSvd(double v) noexcept { epsKiSvd_ = v; }

  private:
    double epsKiEigen_     = 1e-10;
    double epsKiCondState_ = 1e-10;
    double epsKiSvd_       = 1e-10;
};

/// One classical block of a Koashi-Imoto decomposition.
///
/// In the symmetric KI form ρ_AB = ⊕_j p_j · ρ_{A_L B_L,j} ⊗ ω_{A_R,j} ⊗ ω_{B_R,j},
/// this stores the j-th summand: its weight p_j, the joint core
/// state on the left blocks (``coreState``), the per-side right-tail
/// states (``tailA``, ``tailB``), and the four block dimensions.
///
/// Constructed by ``koashiImotoDecompose``; immutable after build
/// (callers should treat instances as read-only outputs).
class KoashiImotoBlock {
  public:
    KoashiImotoBlock() = default;
    KoashiImotoBlock(double weight,
                     Eigen::MatrixXcd coreState,
                     Eigen::MatrixXcd tailA,
                     Eigen::MatrixXcd tailB,
                     int dimLeftA,
                     int dimLeftB,
                     int dimRightA,
                     int dimRightB) noexcept
      : weight_(weight),
        coreState_(std::move(coreState)),
        tailA_(std::move(tailA)),
        tailB_(std::move(tailB)),
        dimLeftA_(dimLeftA),
        dimLeftB_(dimLeftB),
        dimRightA_(dimRightA),
        dimRightB_(dimRightB) {}

    double getWeight() const noexcept { return weight_; }
    const Eigen::MatrixXcd& getCoreState() const noexcept { return coreState_; }
    const Eigen::MatrixXcd& getTailA() const noexcept { return tailA_; }
    const Eigen::MatrixXcd& getTailB() const noexcept { return tailB_; }
    int getDimLeftA() const noexcept { return dimLeftA_; }
    int getDimLeftB() const noexcept { return dimLeftB_; }
    int getDimRightA() const noexcept { return dimRightA_; }
    int getDimRightB() const noexcept { return dimRightB_; }

  private:
    double           weight_{0.0};
    Eigen::MatrixXcd coreState_;
    Eigen::MatrixXcd tailA_;
    Eigen::MatrixXcd tailB_;
    int              dimLeftA_{0};
    int              dimLeftB_{0};
    int              dimRightA_{0};
    int              dimRightB_{0};
};

/// Output of ``koashiImotoDecompose``. ``sigma`` is the KI core
/// ρ_{Σ_AB} (block-diagonal in the j register); ``aPrime``/``bPrime``
/// are the per-side classical-quantum tails. ``blocks`` carries each
/// j-summand explicitly for callers that want block-by-block access.
///
/// Constructed by ``koashiImotoDecompose``; immutable after build.
class KoashiImotoResult {
  public:
    KoashiImotoResult() = default;
    KoashiImotoResult(Eigen::MatrixXcd sigma,
                      Eigen::MatrixXcd aPrime,
                      Eigen::MatrixXcd bPrime,
                      std::vector<KoashiImotoBlock> blocks) noexcept
      : sigma_(std::move(sigma)),
        aPrime_(std::move(aPrime)),
        bPrime_(std::move(bPrime)),
        blocks_(std::move(blocks)) {}

    const Eigen::MatrixXcd& getSigma() const noexcept { return sigma_; }
    const Eigen::MatrixXcd& getAPrime() const noexcept { return aPrime_; }
    const Eigen::MatrixXcd& getBPrime() const noexcept { return bPrime_; }
    const std::vector<KoashiImotoBlock>& getBlocks() const noexcept {
      return blocks_;
    }

  private:
    Eigen::MatrixXcd               sigma_;
    Eigen::MatrixXcd               aPrime_;
    Eigen::MatrixXcd               bPrime_;
    std::vector<KoashiImotoBlock>  blocks_;
};

[[nodiscard]] KoashiImotoResult
koashiImotoDecompose(const Eigen::MatrixXcd&        rhoAB,
                     int                            dimA,
                     int                            dimB,
                     const KoashiImotoTolerances&   tol = {});

/// Overload taking explicit marginals ρ_A, ρ_B. Use this when the
/// caller already has the marginals on hand (e.g. they came in on
/// QuantumVertex objects) — the algorithm uses ρ_A / ρ_B for the
/// eigendecompositions that seed the KI block structure, so passing
/// them directly avoids recomputing partial traces and avoids the
/// numerical drift of partial-trace round-trips.
///
/// ``rhoA`` must be dimA-square, ``rhoB`` must be dimB-square, and
/// the partial traces of ``rhoAB`` over B (resp. A) must agree with
/// ``rhoA`` (resp. ``rhoB``) within the eigen tolerance. The check
/// is performed in debug builds; in release builds the caller is
/// trusted.
[[nodiscard]] KoashiImotoResult
koashiImotoDecompose(const Eigen::MatrixXcd&        rhoAB,
                     const Eigen::MatrixXcd&        rhoA,
                     const Eigen::MatrixXcd&        rhoB,
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

/// Overload taking ρ_A, ρ_B alongside ρ_AB. Avoids the partial
/// traces inside this function and lets callers pass marginals they
/// already have (e.g. from QuantumVertex state) without round-trip
/// numerical noise.
[[nodiscard]] double
mutualInformation(const Eigen::MatrixXcd& rhoAB,
                  const Eigen::MatrixXcd& rhoA,
                  const Eigen::MatrixXcd& rhoB);

} // namespace tessera::quantum
