// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
#define TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::observables {

using namespace ::tessera::spacetime;

/// # SimplicialQubit
///
/// A single qubit state encoded as the holomorphic line in the harmonic space
/// of the metric Hodge Laplacian on a triangulated torus
/// (`docs/design/simplicial_qubit_spec.md`, followed section by section). The
/// input is intrinsic geometry — a simplicial complex \f$ K \cong T^2 \f$ with
/// real edge lengths and a marked cycle pair \f$ (A, B) \f$, \f$ A \cdot B =
/// +1 \f$; the output is a point of \f$ \mathbb{CP}^1 \f$.
///
/// What is computed, and nothing is assumed (spec §1):
///  - §2 the input is validated on load: every edge in exactly two faces,
///    \f$ n_V - n_E + n_F = 0 \f$, consistent face orientations, the strict
///    triangle inequality on every face, closed and homologically independent
///    marked cycles;
///  - §3 the incidence matrices \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and
///    \f$ d_1 \f$ (\f$ n_F \times n_E \f$) with \f$ d_1 d_0 = 0 \f$ exactly;
///  - §4 per triangle: the angles by the law of cosines, the area by Heron's
///    formula, and the intrinsic planar layout \f$ p_i = (0,0),\ p_j = (c,0),\
///    p_k = (b\cos\alpha_i, b\sin\alpha_i) \f$ that every per-face vector lives
///    in (frames of different faces are never compared);
///  - §5 the cotangent weights \f$ w_e = \tfrac12(\cot\alpha_e + \cot\beta_e) \f$,
///    \f$ M_1 = \mathrm{diag}(w_e) \f$, with negative weights and intrinsic
///    Delaunay violations \f$ \alpha_e + \beta_e > \pi \f$ flagged, and the
///    optional intrinsic Delaunay edge-flip pass `intrinsicDelaunay()`;
///  - §6 the harmonic space \f$ H = \ker[d_1;\ d_0^T M_1] \f$ (closed and
///    co-closed 1-cochains), of dimension 2 — the topological input — by a
///    dense SVD null space;
///  - §7 the \f$ L^2 \f$ inner product of 1-cochains through Whitney 1-forms:
///    barycentric gradients per face in the local frame, the interpolant
///    \f$ W_t(\omega) \f$ at the barycenter, \f$ \langle\omega,\eta\rangle =
///    \sum_t A_t\, W_t(\omega)\cdot W_t(\eta) \f$;
///  - §8 the complex structure by rotate-then-project: the Gram matrix
///    \f$ G \f$, the rotation pairing \f$ R_{ab} = \sum_t A_t\,
///    \mathrm{rot}_{90}(W_t(h_a))\cdot W_t(h_b) \f$ and \f$ J = G^{-1}R^T \f$
///    — the metric input — with the residual \f$ \|J^2 + I\|_F \f$ exposed and
///    never symmetrized away;
///  - §9 the holomorphic line: the eigenvector of \f$ J \f$ for the eigenvalue
///    nearest \f$ -i \f$ (convention \f$ \star dz = -i\,dz \f$), its periods
///    \f$ P_A, P_B \f$ over the marking, \f$ \tau = P_B/P_A \f$ in the upper
///    half plane (conjugate branch when \f$ \mathrm{Im}\,\tau < 0 \f$; the
///    marking \f$ (B, -A) \f$ and \f$ -1/\tau \f$ when \f$ |P_A| \f$ vanishes);
///  - §10 \f$ |\psi\rangle = (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$,
///    its Bloch vector (unit, asserted) and density matrix;
///  - §11 the Fubini–Study and Weil–Petersson distances, as two separate
///    functions;
///  - §13 degeneration diagnostics: \f$ \mathrm{cond}(M_1) \f$ and
///    \f$ \mathrm{cond}(G) \f$ against a configurable threshold, warning
///    rather than failing.
///
/// WHY the state is read from the geometry this way: the dimension of
/// \f$ H \f$ is topological (\f$ b_1 = 2 \f$) and the line inside it is
/// metric, so the qubit is exactly the metric's contribution — the conformal
/// structure of the torus, one point of the upper half plane once a marking
/// is fixed. Both halves are computed from the lengths and the marking and
/// nothing else, which is what lets a geometric process reach a state by
/// moving lengths. The construction is exact on flat tori and first-order
/// accurate in the mesh size otherwise (spec §15), which is why the flat torus
/// \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ of `flatTorus` is the
/// reference: it returns \f$ \tau \f$ to rounding at every resolution.
///
/// WHY the marking is stored with the complex: \f$ \tau \f$ depends on
/// \f$ (A, B) \f$ up to \f$ SL(2,\mathbb{Z}) \f$, so a canonical state needs the
/// marking fixed and carried along (spec §15). Coverage is one open hemisphere
/// of the Bloch sphere; the other requires the opposite surface orientation,
/// which is the faces given with the opposite cyclic order (or the `reversed`
/// flag of the `Spacetime` constructor, whose container stores no
/// orientation). There is no action of \f$ SU(2) \f$ on edge lengths: gates
/// act on `state()` as \f$ 2\times 2 \f$ matrices and are never pulled back.
///
/// WHY a `Spacetime` sits underneath: it is the repository's container for
/// vertices, edges and lengths, so a qubit built from the raw data structures
/// of spec §2 also exists as a `Spacetime` (`spacetime()`), and a torus that
/// already lives as a `Spacetime` can be read directly. The container sorts
/// the vertices of each face, so the consistently oriented faces of §2 are
/// held here, not there.
class SimplicialQubit {
  public:
    /// An edge \f$ (i, j) \f$ with \f$ i < j \f$, oriented \f$ i \to j \f$ (spec §2).
    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;
    /// A face \f$ (i, j, k) \f$ in its counterclockwise order (spec §2).
    using Face = std::array<std::uint64_t, 3>;
    /// One step of a marked cycle: (edge index into `edges()`, sign \f$ \pm 1 \f$)
    /// — \f$ +1 \f$ traverses the edge along its stored orientation.
    using CycleStep = std::pair<std::size_t, int>;
    using Cycle = std::vector<CycleStep>;

    /// The §2 / §14 constructor from the raw data structures.
    /// @param vertices \f$ V = [0 .. n_V - 1] \f$.
    /// @param edges \f$ E = [(i, j)] \f$, \f$ i < j \f$.
    /// @param faces \f$ F = [(i, j, k)] \f$, consistently oriented.
    /// @param lengths \f$ \ell : E \to \mathbb{R}_{>0} \f$, one per edge, in edge order.
    /// @param cycleA, cycleB The marked cycles as (edge index, sign) lists,
    ///   closed loops with \f$ A \cdot B = +1 \f$.
    /// @param degeneracyThreshold The condition-number level of spec §13 above
    ///   which a warning is recorded.
    /// @throws std::invalid_argument when a §2 validation fails;
    ///   std::runtime_error when \f$ \dim H \ne 2 \f$ (§6: not a torus, or
    ///   degenerate weights).
    SimplicialQubit(std::vector<std::uint64_t> vertices, std::vector<EdgePair> edges,
                    std::vector<Face> faces, std::vector<double> lengths, Cycle cycleA,
                    Cycle cycleB, double degeneracyThreshold = 1e8);

    /// The same qubit read from a `Spacetime` of dimension 2: its vertices
    /// (indexed by ascending id), its edges (ascending \f$ (i, j) \f$ order,
    /// which is the edge order the cycles refer to), its triangles, and its
    /// real positive edge lengths. The container stores no face orientation,
    /// so the consistent orientation is the one making the top chain a cycle
    /// (`cobordism::ChainComplex::fundamentalClass`), reversed when \p reversed
    /// is set — the "separate boolean" of spec §15 for the other hemisphere.
    /// @throws std::invalid_argument when a length is not real and positive,
    ///   the surface is not closed-orientable, or a §2 validation fails.
    SimplicialQubit(const std::shared_ptr<Spacetime> &spacetime, Cycle cycleA, Cycle cycleB,
                    bool reversed = false, double degeneracyThreshold = 1e8);

    // ----- spec §14 -----------------------------------------------------------

    /// \f$ H \f$: \f$ n_E \times 2 \f$ real, an orthonormal basis of the null
    /// space of \f$ [d_1;\ d_0^T M_1] \f$ (§6).
    [[nodiscard]] const Eigen::MatrixXd &harmonicBasis() const noexcept { return H_; }
    /// \f$ J = G^{-1}R^T \f$ in the basis \f$ \{h_1, h_2\} \f$, \f$ 2 \times 2 \f$ real (§8).
    [[nodiscard]] const Eigen::MatrixXd &complexStructure() const noexcept { return J_; }
    /// \f$ \|J^2 + I\|_F \f$: the discretization-error diagnostic (§8).
    [[nodiscard]] double jResidual() const noexcept { return jResidual_; }
    /// \f$ \omega = c_0 h_1 + c_1 h_2 \f$, the complex 1-cochain spanning the
    /// holomorphic line (§9), after the branch and marking rules.
    [[nodiscard]] const Eigen::VectorXcd &holomorphicForm() const noexcept { return omega_; }
    /// \f$ (P_A, P_B) \f$ of the holomorphic form over the marking in force (§9).
    [[nodiscard]] std::pair<std::complex<double>, std::complex<double>> periods() const noexcept {
      return {periodA_, periodB_};
    }
    /// \f$ \tau = P_B / P_A \f$ (§9).
    [[nodiscard]] std::complex<double> tau() const noexcept { return tau_; }
    /// The PERIOD FRAME of the torus (qubit cobordism spec D3): the basis
    /// \f$ (f_A, f_B) \f$ of its harmonic space with periods \f$ (1, 0) \f$
    /// and \f$ (0, 1) \f$ over the marking in force, an \f$ n_E \times 2 \f$
    /// real matrix in the torus's edge order — `harmonicBasis()` times the
    /// inverse of the period matrix, \f$ F = H\,\Pi^{-1} \f$ with
    /// \f$ \Pi_{ca} = \oint_c h_a \f$ (rows = the cycles \f$ A, B \f$,
    /// columns = the basis elements \f$ h_1, h_2 \f$), so that
    /// \f$ \oint_A f_A = 1,\ \oint_B f_A = 0,\ \oint_A f_B = 0,\ \oint_B f_B = 1 \f$.
    ///
    /// WHAT it is for: the coordinates a state is written in. Every harmonic
    /// form is \f$ \omega = P_A f_A + P_B f_B \f$ with its own periods as the
    /// coefficients, so the holomorphic line is the column combination
    /// \f$ (1, \tau) \f$ of the frame: `holomorphicForm()`
    /// \f$ = \f$ `periodFrame()` \f$ \cdot (P_A, P_B)^T = P_A\,F (1, \tau)^T \f$,
    /// the qubit \f$ |0\rangle + \tau|1\rangle \f$ of §10 read as a 1-form
    /// with \f$ f_A \leftrightarrow |0\rangle \f$ and
    /// \f$ f_B \leftrightarrow |1\rangle \f$. A two-body target \f$ \chi \f$
    /// is written in the \f$ |0\rangle, |1\rangle \f$ bases of two qubits,
    /// so the transfer of a cobordism between two tori compares with it when
    /// it is read in the period frames of the two boundary tori
    /// (`cobordism::MultiCobordism::setInputFrame`), where it is
    /// \f$ 2 \times 2 \f$. WHY over the marking in force: \f$ \tau \f$ is
    /// reported over the marking in force (§9: \f$ (B, -A) \f$ when
    /// \f$ |P_A| \f$ vanishes, `markingSwapped()`), and the frame is the
    /// basis \f$ \tau \f$ is a coordinate in, so `periods()` are always the
    /// coefficients of `holomorphicForm()` in it. The frame is real (the
    /// harmonic basis and the periods are), invariant under a common scale of
    /// the lengths (the harmonic space and the periods are), independent of
    /// the orthonormal basis §6 happens to return, and always defined: the
    /// period map of the harmonic space over a homology basis is an
    /// isomorphism, which §2's independence check guarantees.
    [[nodiscard]] const Eigen::MatrixXd &periodFrame() const noexcept { return F_; }
    /// \f$ (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$ (§10).
    [[nodiscard]] Eigen::VectorXcd state() const;
    /// \f$ (2\,\mathrm{Re}\,\tau,\ 2\,\mathrm{Im}\,\tau,\ 1 - |\tau|^2)/(1+|\tau|^2) \f$ (§10).
    [[nodiscard]] Eigen::VectorXd bloch() const;
    /// \f$ \rho = \tfrac12(I + \vec r\cdot\vec\sigma) \f$ (§10).
    [[nodiscard]] Eigen::MatrixXcd densityMatrix() const;
    /// The flat torus \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ (§12): the
    /// unit cell spanned by \f$ 1 \f$ and \f$ \tau \f$ as an \f$ n_x \times
    /// n_y \f$ grid, every square split by its diagonal, sides identified;
    /// edge lengths are the Euclidean lengths of the lattice displacements;
    /// \f$ A \f$ is the row loop along \f$ 1 \f$ and \f$ B \f$ the column loop
    /// along \f$ \tau \f$. Exact: the read returns \f$ \tau \f$ to rounding.
    /// @throws std::invalid_argument unless \f$ \mathrm{Im}\,\tau > 0 \f$ and
    ///   \f$ n_x, n_y \ge 3 \f$ (below 3 the grid is not a simplicial complex).
    [[nodiscard]] static SimplicialQubit flatTorus(std::complex<double> tau, int nx, int ny);
    /// \f$ d_{FS} = \arccos\big(|1 + \bar\tau_1\tau_2| / \sqrt{(1+|\tau_1|^2)(1+|\tau_2|^2)}\big) \f$:
    /// distinguishability of the two states (§11); curvature \f$ +4 \f$.
    [[nodiscard]] static double fubiniStudyDistance(const SimplicialQubit &q1,
                                                    const SimplicialQubit &q2);
    /// \f$ d_{WP} = \operatorname{arccosh}\big(1 + |\tau_1 - \tau_2|^2 /
    /// (2\,\mathrm{Im}\,\tau_1\,\mathrm{Im}\,\tau_2)\big) \f$: the moduli
    /// distance between the two shapes (§11); curvature \f$ -1 \f$. Kept
    /// separate from \f$ d_{FS} \f$ on purpose: conformally equivalent, not
    /// isometric.
    [[nodiscard]] static double weilPeterssonDistance(const SimplicialQubit &q1,
                                                      const SimplicialQubit &q2);

    // ----- spec §5: the optional intrinsic Delaunay preprocessing pass --------

    /// Flip every edge violating the intrinsic Delaunay condition
    /// \f$ \alpha_e + \beta_e \le \pi \f$ until none remains (the two triangles
    /// are laid out in one plane, the diagonal is replaced by the other one
    /// with its intrinsic length, the marked cycles are rerouted around the
    /// quadrilateral), and return the qubit of the flipped triangulation. The
    /// intrinsic geometry is unchanged, so on a flat torus \f$ \tau \f$ is
    /// unchanged; the cotangent weights become non-negative and the
    /// construction numerically stable. An edge whose flip would duplicate an
    /// existing edge is left alone and reported in `warnings()`.
    [[nodiscard]] SimplicialQubit intrinsicDelaunay() const;
    /// Number of flips the pass that produced this qubit performed (0 unless
    /// it came from `intrinsicDelaunay()`).
    [[nodiscard]] int delaunayFlipCount() const noexcept { return flips_; }

    // ----- inputs (§2) --------------------------------------------------------

    [[nodiscard]] const std::vector<std::uint64_t> &vertices() const noexcept { return vertices_; }
    [[nodiscard]] const std::vector<EdgePair> &edges() const noexcept { return edges_; }
    [[nodiscard]] const std::vector<Face> &faces() const noexcept { return faces_; }
    [[nodiscard]] const std::vector<double> &lengths() const noexcept { return lengths_; }
    [[nodiscard]] const Cycle &cycleA() const noexcept { return cycleA_; }
    [[nodiscard]] const Cycle &cycleB() const noexcept { return cycleB_; }
    [[nodiscard]] double degeneracyThreshold() const noexcept { return degeneracyThreshold_; }
    /// The `Spacetime` holding the vertices, edges and lengths.
    [[nodiscard]] const std::shared_ptr<Spacetime> &spacetime() const noexcept { return spacetime_; }

    // ----- intermediate quantities (§3–§9, §13) ------------------------------

    /// \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and \f$ d_1 \f$ (\f$ n_F \times n_E \f$) (§3).
    [[nodiscard]] const Eigen::MatrixXd &d0() const noexcept { return d0_; }
    [[nodiscard]] const Eigen::MatrixXd &d1() const noexcept { return d1_; }
    /// Per face \f$ (\alpha_i, \alpha_j, \alpha_k) \f$, \f$ n_F \times 3 \f$ (§4).
    [[nodiscard]] const Eigen::MatrixXd &angles() const noexcept { return angles_; }
    /// Per face the Heron area \f$ A_t \f$ (§4).
    [[nodiscard]] const Eigen::VectorXd &areas() const noexcept { return areas_; }
    /// Per face the local layout \f$ (p_i, p_j, p_k) \f$, \f$ n_F \times 6 \f$ (§4).
    [[nodiscard]] const Eigen::MatrixXd &layout() const noexcept { return layout_; }
    /// Per face \f$ (\nabla\lambda_i, \nabla\lambda_j, \nabla\lambda_k) \f$ in
    /// the local frame, \f$ n_F \times 6 \f$ (§7).
    [[nodiscard]] const Eigen::MatrixXd &barycentricGradients() const noexcept { return gradients_; }
    /// The cotangent weights \f$ w_e \f$ (§5).
    [[nodiscard]] const Eigen::VectorXd &weights() const noexcept { return weights_; }
    /// Edges with \f$ w_e < 0 \f$ and edges with \f$ \alpha_e + \beta_e > \pi \f$ (§5).
    [[nodiscard]] const std::vector<std::size_t> &negativeWeightEdges() const noexcept {
      return negativeWeightEdges_;
    }
    [[nodiscard]] const std::vector<std::size_t> &nonDelaunayEdges() const noexcept {
      return nonDelaunayEdges_;
    }
    /// \f$ G_{ab} = \langle h_a, h_b\rangle \f$ and the rotation pairing \f$ R_{ab} \f$ (§8).
    [[nodiscard]] const Eigen::MatrixXd &gram() const noexcept { return G_; }
    [[nodiscard]] const Eigen::MatrixXd &rotationPairing() const noexcept { return R_; }
    /// True when \f$ |P_A| \f$ vanished and the marking \f$ (B, -A) \f$ is in
    /// force, i.e. `tau()` is \f$ -1/\tau \f$ of the given marking (§9).
    [[nodiscard]] bool markingSwapped() const noexcept { return swapped_; }
    /// \f$ \mathrm{cond}(M_1) = \max|w_e| / \min|w_e| \f$ and the condition
    /// number of the \f$ 2 \times 2 \f$ Gram matrix (§13).
    [[nodiscard]] double conditionM1() const noexcept { return condM1_; }
    [[nodiscard]] double conditionG() const noexcept { return condG_; }
    /// True when either condition number exceeds the threshold (§13).
    [[nodiscard]] bool nearDegenerate() const noexcept { return nearDegenerate_; }
    /// Every warning the construction raised (§5 flags, §13 near-degeneracy,
    /// §9 branch notes); empty when there is nothing to report.
    [[nodiscard]] const std::vector<std::string> &warnings() const noexcept { return warnings_; }

  private:
    // Inputs.
    std::vector<std::uint64_t> vertices_;
    std::vector<EdgePair> edges_;
    std::vector<Face> faces_;
    std::vector<double> lengths_;
    Cycle cycleA_;
    Cycle cycleB_;
    double degeneracyThreshold_;
    std::shared_ptr<Spacetime> spacetime_;
    int flips_{0};

    // §3
    Eigen::MatrixXd d0_;
    Eigen::MatrixXd d1_;
    // §4, §7
    Eigen::MatrixXd angles_;
    Eigen::VectorXd areas_;
    Eigen::MatrixXd layout_;
    Eigen::MatrixXd gradients_;
    // §5
    Eigen::VectorXd weights_;
    std::vector<std::size_t> negativeWeightEdges_;
    std::vector<std::size_t> nonDelaunayEdges_;
    // §6
    Eigen::MatrixXd H_;
    // §8
    Eigen::MatrixXd G_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd J_;
    double jResidual_{0.0};
    // §9
    Eigen::VectorXcd omega_;
    std::complex<double> periodA_{0.0, 0.0};
    std::complex<double> periodB_{0.0, 0.0};
    std::complex<double> tau_{0.0, 0.0};
    bool swapped_{false};
    /// The period frame \f$ F = H\,\Pi^{-1} \f$ over the marking in force (`periodFrame`).
    Eigen::MatrixXd F_;
    // §13
    double condM1_{0.0};
    double condG_{0.0};
    bool nearDegenerate_{false};
    std::vector<std::string> warnings_;

    // Edge lookup by (min, max) vertex pair, and per edge its two
    // (face, local edge slot) incidences: slot 0 is the face's edge (i, j),
    // slot 1 is (j, k), slot 2 is (k, i).
    std::map<EdgePair, std::size_t> edgeIndex_;
    std::vector<std::vector<std::pair<std::size_t, int>>> edgeFaces_;

    // The shared body of both constructors: §2 validation through §13.
    void initialize();
    void indexEdges();                          // §2: E well-formed
    void validateCombinatorics();               // §2: incidence, chi, orientation, triangles
    void buildIncidence();                      // §3
    void validateCycles() const;                // §2: closed, independent
    void buildFaceGeometry();                   // §4, §7 (gradients)
    void buildWeights();                        // §5
    void buildHarmonicSpace();                  // §6
    void buildComplexStructure();               // §7, §8
    void buildHolomorphicLine();                // §9
    void buildPeriodFrame();                    // qubit cobordism spec D3
    void diagnoseDegeneration();                // §13
    [[nodiscard]] std::size_t edgeIndexOf(std::uint64_t u, std::uint64_t v) const;
    [[nodiscard]] std::complex<double> periodOf(const Eigen::VectorXcd &omega, const Cycle &cycle) const;
    /// \f$ W_t(\omega) \f$ at the barycenter of face \p t (§7), real or complex.
    template <typename Scalar>
    [[nodiscard]] Eigen::Matrix<Scalar, 2, 1> whitneyAtBarycenter(
        std::size_t t, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &omega) const;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
