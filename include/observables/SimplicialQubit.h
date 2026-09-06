// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
#define TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "observables/Observable.h"
#include "observables/Record.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::observables {

using namespace ::tessera::spacetime;

/// # SimplicialQubitRead
///
/// Everything one read of a marked torus produces, from the combinatorial
/// certificates through the harmonic zero mode to the point of
/// \f$ \mathbb{CP}^1 \f$. Fields are populated in the order the read runs, so a
/// refused read (non-empty `refusal`) still carries every certificate measured
/// before the refusal; the fields after it keep their "not measured"
/// sentinels (NaN, empty). Complex channels of `toRecord()` are split
/// `{name}_re` / `{name}_im` (#580); nothing is silently `.real()`-ed.
struct SimplicialQubitRead {
  /// f-vector of the surface and its Euler characteristic / Betti numbers
  /// (over \f$ \mathbb{Q} \f$, from the integer incidences).
  std::size_t vertices{0};
  std::size_t edges{0};
  std::size_t faces{0};
  int eulerCharacteristic{0};
  std::vector<int> betti{};

  /// Nullity of the untwisted stacked kernel at degree 1 (must be 2 on a
  /// torus) and its SVD gap certificate (NaN on the sparse path).
  int harmonicRank{0};
  double harmonicGap{std::numeric_limits<double>::quiet_NaN()};
  /// Nullity of the twisted kernel with the stored link phases ON. Equals 2
  /// exactly when the connection is pure gauge; a smaller value means the
  /// phases carry holonomy or flux and the read refuses.
  int twistedHarmonicRank{0};

  /// The harmonic basis: chains \f$ h_a = M_1 z_a \f$ (coefficients from the
  /// lengths) and their geometric images \f$ z_a \f$ (the edge integrals),
  /// \f$ n_1 \times 2 \f$ in the canonical edge order of the complex.
  Eigen::MatrixXcd harmonicChains{};
  Eigen::MatrixXcd harmonicImages{};

  /// \f$ G_{ab} = z_a^T M_1 z_b \f$: the Whitney \f$ L^2 \f$ Gram of the basis
  /// (transpose pairing; complex symmetric for complex lengths).
  Eigen::MatrixXcd gram{};
  /// \f$ R_{ab} = \langle z_a \cup z_b, [K] \rangle \f$: the intersection form
  /// on the basis (antisymmetric, metric-free, orientation-signed).
  Eigen::MatrixXcd intersection{};
  /// \f$ J = G^{-1} R^T \f$: the complex structure on the harmonic space, and
  /// \f$ \|J^2 + I\|_F \f$, the discretization residual (0 on a flat torus).
  Eigen::MatrixXcd complexStructure{};
  double complexStructureResidual{std::numeric_limits<double>::quiet_NaN()};

  /// \f$ P_{a,A}, P_{a,B} \f$: the periods of the basis over the marking,
  /// rows = basis, columns = (A, B); and the intersection number \f$ A \cdot B
  /// = R_{01} / \det P \f$ recovered from them (+1 for an admissible marking).
  Eigen::MatrixXcd periods{};
  std::complex<double> intersectionNumber{std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::quiet_NaN()};

  /// The holomorphic form: the \f$ -i \f$ eigenline of \f$ J \f$ expanded on
  /// the images (\f$ n_1 \f$ edge integrals), its periods and their ratio.
  Eigen::VectorXcd holomorphicForm{};
  std::complex<double> periodA{std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN()};
  std::complex<double> periodB{std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN()};
  std::complex<double> tau{std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN()};
  /// True when \f$ |P_A| \f$ vanished for this metric and the read switched to
  /// the marking \f$ (A', B') = (B, -A) \f$, reporting \f$ -1/\tau \f$.
  bool markingSwapped{false};

  /// \f$ |\psi\rangle = (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$, its
  /// Bloch vector (norm reported, 1 for a pure state) and density matrix.
  Eigen::VectorXcd state{};
  Eigen::VectorXd bloch{};
  double blochNorm{std::numeric_limits<double>::quiet_NaN()};
  Eigen::MatrixXcd density{};

  /// Degeneration diagnostics: condition numbers of \f$ M_1 \f$ (dense, below
  /// the crossover; NaN above) and of the \f$ 2 \times 2 \f$ Gram. Above the
  /// configured threshold the read WARNS (`nearDegenerate`, `warning`) and
  /// stays valid — a pinching cycle sends the state smoothly to a pole.
  double metricCondition{std::numeric_limits<double>::quiet_NaN()};
  double gramCondition{std::numeric_limits<double>::quiet_NaN()};
  bool nearDegenerate{false};
  std::string warning{};
  /// Empty when the read holds; otherwise the named reason it was refused.
  std::string refusal{};

  [[nodiscard]] bool holds() const noexcept { return refusal.empty(); }
  /// The JSON-able record (complex leaves split `_re`/`_im`).
  [[nodiscard]] Record toRecord() const;
};

struct MarkedTorus;

/// # SimplicialQubit
///
/// Observable: a single qubit **as the intrinsic geometry of a marked
/// triangulated torus**. The input is a closed oriented 2-complex
/// homeomorphic to \f$ T^2 \f$ carrying complex edge lengths and link phases
/// (a `Spacetime`); the output is a point of \f$ \mathbb{CP}^1 \f$.
///
/// The construction, in the chain-level Whitney pencil (`chainhodge`):
///  1. the harmonic 1-chains \f$ H_1 = M_1 \ker S \f$ of the torus — the exact
///     zero mode, whose dimension \f$ b_1 = 2 \f$ is topological and whose
///     chain coefficients come from the lengths alone (the geometric images
///     \f$ z_a \f$ are the edge integrals);
///  2. the Whitney \f$ L^2 \f$ Gram \f$ G = Z^T M_1 Z \f$ of that basis — the
///     only place the metric enters;
///  3. the intersection form \f$ R_{ab} = \langle z_a \cup z_b, [K] \rangle \f$
///     (`ChainComplex::cupProductForm`), exact and metric-free;
///  4. the complex structure \f$ J = G^{-1} R^T \f$, the discrete Hodge star
///     compressed to the harmonic space: \f$ J^2 = -I \f$ exactly on a flat
///     torus and up to a residual \f$ \|J^2 + I\|_F \f$ that vanishes under
///     refinement otherwise (reported, never symmetrized away);
///  5. the holomorphic line \f$ \ker(J + i) \f$ and its periods
///     \f$ P_A, P_B \f$ over the marked cycles, \f$ \tau = P_B / P_A \f$ in the
///     upper half plane;
///  6. \f$ |\psi\rangle = (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$,
///     Bloch vector and density matrix.
///
/// WHY the zero mode and the Whitney Gram rather than a cochain with chosen
/// coefficients: the state must be *read from* geometry so that geometric
/// relaxation can later *reach* it. The harmonic chain's coefficients are the
/// lengths' (through \f$ M_1 \f$), the Gram is the lengths' (through the
/// per-triangle \f$ \sqrt{\det \mathrm{Gram}_T} \f$, i.e. the Cayley–Menger
/// volumes), and the pairing needs no metric at all — so \f$ \tau \f$ is a
/// function of the six-per-triangle complex lengths and nothing else. Steps
/// 2–4 are the specification's "rotate each per-face vector by 90° and
/// project back" with the integrals done exactly: the rotation pairing
/// \f$ \int W(z_a)\wedge W(z_b) \f$ is the cup-product pairing of the classes,
/// and the projection is the Gram solve.
///
/// WHY the phases still matter although \f$ \tau \f$ depends on the metric
/// alone: with the link phases on, the pencil is the covariant one and its
/// zero mode is the twisted harmonic space, which on a torus is
/// two-dimensional exactly when the connection is pure gauge and collapses
/// under holonomy or flux. The read measures that rank
/// (`CovariantChainHodge::harmonicChains`) and refuses by name when it is
/// not 2: a torus whose phases are not a gauge does not carry a qubit. Complex
/// lengths are admitted throughout (the Gram is then complex symmetric; the
/// eigenline of \f$ J \f$ whose period ratio has positive imaginary part is
/// taken, and a read where neither branch does is reported, not corrected).
///
/// WHY the marking is part of the observable: \f$ \tau \f$ depends on the
/// choice of \f$ (A, B) \f$ up to \f$ SL(2, \mathbb{Z}) \f$; a canonical state
/// requires the marking fixed and stored with the complex, so it is
/// constructor state here (two closed vertex walks and an orientation flag),
/// and the read certifies it — the cycles must be independent and their
/// intersection number, recovered from \f$ R \f$ and the period matrix, must
/// be \f$ +1 \f$. Coverage is one open hemisphere of the Bloch sphere (the
/// upper half plane under the Cayley map); the other requires the opposite
/// surface orientation, the `reversed` flag.
///
/// It is deliberately NOT a relaxation: nothing here moves a length. Bringing
/// a torus's volumes to a target state is the cobordism engine's job; this
/// class is the representation it will read, plus the one exact constructor
/// in the other direction, `flatTorus`, which realizes any \f$ \tau \f$ in the
/// upper half plane on the flat torus \f$ \mathbb{C}/(\mathbb{Z} + \tau
/// \mathbb{Z}) \f$. Unitary gates have no realization as metric deformations
/// and act on `state()` as \f$ 2 \times 2 \f$ matrices.
class SimplicialQubit : public Observable {
  public:
    /// A closed vertex walk \f$ v_0, v_1, \ldots, v_{m-1} \f$ (the edge
    /// \f$ (v_{m-1}, v_0) \f$ closes it); every consecutive pair must be an
    /// edge of the complex. Traversal against the ascending-id reference
    /// orientation of an edge contributes the negative of its integral.
    using Cycle = std::vector<std::uint64_t>;

    /// @param cycleA First marked cycle (the \f$ |0\rangle \f$ direction:
    ///   \f$ P_A \f$ normalizes the holomorphic form).
    /// @param cycleB Second marked cycle, with \f$ A \cdot B = +1 \f$.
    /// @param reversed Read the surface with the opposite orientation
    ///   (negates the fundamental class): the other hemisphere.
    /// @param degeneracyThreshold Condition-number level above which the read
    ///   warns of a pinching cycle (specification §13); never a failure.
    SimplicialQubit(Cycle cycleA, Cycle cycleB, bool reversed = false,
                    double degeneracyThreshold = 1e8);

    /// The full read of a torus.
    /// @throws std::invalid_argument on a null spacetime or a marking that is
    ///   not a closed walk of edges of the complex (structural input errors);
    ///   metric- and topology-dependent failures are refused by name in the
    ///   read instead.
    [[nodiscard]] SimplicialQubitRead read(const std::shared_ptr<Spacetime> &spacetime) const;

    /// Headline scalar: the complex-structure residual \f$ \|J^2 + I\|_F \f$
    /// — the certificate that the geometry-to-state map is faithful on this
    /// mesh (0 on a flat torus, decreasing under refinement). NaN when the
    /// read is refused.
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;

    [[nodiscard]] const Cycle &cycleA() const noexcept { return cycleA_; }
    [[nodiscard]] const Cycle &cycleB() const noexcept { return cycleB_; }
    [[nodiscard]] bool reversed() const noexcept { return reversed_; }
    [[nodiscard]] double degeneracyThreshold() const noexcept { return degeneracyThreshold_; }

    /// The flat torus \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ as an
    /// \f$ n_x \times n_y \f$ grid (`SimplicialProduct` of two
    /// `PolygonCircle`s, every square cut along one diagonal), with each edge's
    /// length the Euclidean length of its lattice displacement, marked by the
    /// row loop \f$ A \f$ (along 1) and the column loop \f$ B \f$ (along
    /// \f$ \tau \f$) with the orientation flag chosen so that \f$ A \cdot B =
    /// +1 \f$. The construction is exact: the read returns \f$ \tau \f$ to
    /// rounding at every resolution, which is what makes it the reference for
    /// every other torus.
    /// @throws std::invalid_argument when \f$ \mathrm{Im}\,\tau \le 0 \f$ or a
    ///   side has fewer than 3 vertices.
    [[nodiscard]] static MarkedTorus flatTorus(std::complex<double> tau, int nx, int ny);

    /// The state, Bloch vector and density matrix of a period ratio.
    [[nodiscard]] static Eigen::VectorXcd stateOf(std::complex<double> tau);
    [[nodiscard]] static Eigen::VectorXd blochOf(std::complex<double> tau);
    [[nodiscard]] static Eigen::MatrixXcd densityOf(std::complex<double> tau);
    /// The period ratio of a state \f$ (\alpha, \beta) \f$: \f$ \beta/\alpha \f$.
    /// @throws std::invalid_argument on \f$ \alpha = 0 \f$ (the cusp: \f$ |1\rangle \f$
    ///   is \f$ \tau = \infty \f$, not a finite torus).
    [[nodiscard]] static std::complex<double> periodRatioOf(const Eigen::VectorXcd &state);

    /// Fubini–Study distance between two period ratios (distinguishability;
    /// curvature \f$ +4 \f$): \f$ \arccos\big(|1 + \bar\tau_1 \tau_2| /
    /// \sqrt{(1+|\tau_1|^2)(1+|\tau_2|^2)}\big) \f$. Finite as a cycle pinches.
    [[nodiscard]] static double fubiniStudyDistance(std::complex<double> tau1,
                                                    std::complex<double> tau2);
    /// Weil–Petersson (Poincaré) distance between two moduli (shape
    /// distance; curvature \f$ -1 \f$): \f$ \operatorname{arccosh}\big(1 +
    /// |\tau_1 - \tau_2|^2 / (2\,\mathrm{Im}\,\tau_1\,\mathrm{Im}\,\tau_2)\big) \f$.
    /// Diverges logarithmically as a cycle pinches. Kept separate from the
    /// Fubini–Study distance on purpose: conformally equivalent, not isometric.
    /// @throws std::invalid_argument unless both imaginary parts are positive.
    [[nodiscard]] static double weilPeterssonDistance(std::complex<double> tau1,
                                                      std::complex<double> tau2);

  private:
    Cycle cycleA_;
    Cycle cycleB_;
    bool reversed_;
    double degeneracyThreshold_;
};

/// A torus together with the qubit observable that reads it: what
/// `SimplicialQubit::flatTorus` returns, so the marking travels with the
/// complex it was built for.
struct MarkedTorus {
  std::shared_ptr<Spacetime> spacetime;
  SimplicialQubit qubit;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
