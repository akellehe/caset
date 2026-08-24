// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_COLORFIBER_H
#define TESSERA_OBSERVABLES_COLORFIBER_H

// The exact three-edge SU(3) color kernel (issue #767, Wave 1 of the
// recursive spectral-fiber program — design spec §11 "Algorithm D", and the
// whitepaper sections "A triangle carries the exact color algebra" and
// "Quarks as modular clusters").
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • ColorFiber   — the CONSTANT color-sector algebra on three oriented
//                    edge modes: the N = 0,1,2,3 exterior-sector projectors
//                    of Λ•C³ = 1 ⊕ 3 ⊕ 3̄ ⊕ 1 (vacuum / triplet /
//                    anti-triplet / singlet), creation/annihilation
//                    matrices, E_ij = a_i†a_j, the eight normalized
//                    Gell-Mann generators on the one-occupation sector, the
//                    traceless adjoint-octet projector, the exact Fourier
//                    frame F₃ built from ω = e^{2πi/3}, the perimeter and
//                    Hilbert normalizers (distinct APIs), the color vector
//                    c = z/‖z‖₂ from stored complex squared lengths, the
//                    det(C) / det(C†C) color-wedge and singlet-Gram
//                    certificates, and the quark / anti-triplet-diquark /
//                    singlet / octet sector READS (weights only — no
//                    particle classification).
//   • OrientedTriangle / AnchorProfile / ColorAnchor
//                  — the calibrated weighted oriented-triangle anchoring
//                    kernel A_τ = |W_τ|^{1/2} R_τ Φ with an atlas score
//                    a² = Σ_τ w_τ |det A_τ|², the convex weighting {w_τ}
//                    DECLARED before any data are examined (post-hoc
//                    re-weighting is rejected), and the reported profile:
//                    maximal term, participation ratio of {|det A_τ|²}, and
//                    determinant-phase dispersion/coherence on OVERLAPPING
//                    oriented triangles — the resultant runs only over
//                    triangles that share a boundary edge with another
//                    declared triangle, so a disjoint atlas reports the
//                    coherence as unknown rather than as a value with no
//                    overlap content.  Signed sectors restrict with
//                    |W_τ|^{1/2} and report the restricted block's Krein
//                    signature separately.
//
// ─── Exact identities implemented (tested to double round-off) ──────────
//
//   • F₃†F₃ = I and |det F₃| = 1 (F₃ assembled from the ALGEBRAIC entries
//     ω^{jk}/√3, ω = (−1 + i√3)/2 — never from repeated multiplication, so
//     1 + ω + ω² cancels exactly in floating point).
//   • λ_a = λ_a†, Tr λ_a = 0, Tr(λ_a λ_b) = 2 δ_ab.
//   • [E_ij, E_kl] = δ_jk E_il − δ_il E_kj — on the 3×3 one-occupation
//     restriction AND on the full 8-dimensional Fock representation.
//   • det(gC) = det(C) for g ∈ SU(3); det(C†C) = |det C|².
//   • ‖v₁∧v₂∧v₃‖² = det[⟨v_i, v_j⟩]: the singlet wedge vanishes for
//     duplicate color modes and reaches unit Gram determinant exactly for
//     an orthonormal triad.
//   • Calibration (see ColorAnchor for the domain): with Φ†|W|Φ = I and a
//     triangle-decoupled |W| (any diagonal edge metric), each
//     |det A_τ|² = det(A_τ†A_τ) ≤ 1 because R_τ†|W_τ|R_τ ⪯ |W|, so
//     a² ∈ [0, 1] with value one exactly at full concentration on the
//     weighted edge span of the anchoring faces.
//
// The constant algebra is generated once (static locals) and checked at
// startup in debug builds (NDEBUG off) via verifyConstantAlgebra(); the
// production operation count is constant.  Everything here is a pure
// function of caller-supplied data: no solver call, no Spacetime mutation,
// nothing enters the emergence objective.
//
// Ownership boundary (#769/#770): this kernel operates on CALLER-SUPPLIED
// inputs — a rank-three frame over a component's oriented edges, the edge
// weight data, and oriented-triangle descriptors (ordered boundary edge
// indices with incidence signs).  It does not construct spectral fibers
// (#769) and contains no transport / Wilson-loop code (#770).

#include "cobordism/Certificate.h"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace tessera::observables {

/// One oriented 2-simplex descriptor for the anchoring kernel: the three
/// boundary edges in the cyclic order induced by the triangle's
/// orientation, with their incidence signs (±1: +1 when the stored oriented
/// edge agrees with the induced boundary orientation, −1 when opposed).
///
/// `edges[k]` indexes a ROW of the caller's frame / weight data — the
/// caller's own oriented-edge indexing; this kernel never sorts or
/// re-derives an order.  The orientation fixes the ordering up to a CYCLIC
/// (hence even) permutation, so det A_τ is invariant under cyclic rotation
/// of (edges, signs) and negates under an odd permutation (= the opposite
/// orientation); |det A_τ|² is invariant under both.
struct OrientedTriangle {
    /// The three ordered boundary edge indices (rows of the frame).
    std::array<Eigen::Index, 3> edges{};
    /// The three incidence signs, each ±1.
    std::array<int, 3> signs{+1, +1, +1};
};

/// The reported anchor datum — the PROFILE, not only the score (whitepaper
/// "Quarks as modular clusters").  Scores and terms are calibrated to
/// [0, 1] in the documented positive/decoupled domain; the profile fields
/// make a concentrated oracle distinguishable from an extended-but-anchored
/// atlas and from an unanchored band at equal score.
struct AnchorProfile {
    /// The calibrated atlas score a² = Σ_τ w_τ |det A_τ|².
    double score{0.0};
    /// |det A_τ|² = det(A_τ†A_τ) per declared triangle (unweighted).
    std::vector<double> terms{};
    /// max_τ |det A_τ|².
    double maxTerm{0.0};
    /// argmax_τ |det A_τ|² (0 when there are no triangles).
    std::size_t maxTermIndex{0};
    /// Participation ratio (Σ_τ t_τ)² / Σ_τ t_τ² of the distribution
    /// {t_τ = |det A_τ|²}: 1 at full concentration on one triangle, the
    /// triangle count for a uniform spread; 0 when every term vanishes.
    double participationRatio{0.0};
    /// arg det A_τ per triangle; NaN where |det A_τ| = 0 (an undefined
    /// determinant phase is reported as unknown, never as zero).
    std::vector<double> detPhases{};
    /// Determinant-phase coherence on OVERLAPPING oriented triangles: the
    /// circular resultant length |Σ_τ u_τ e^{i φ_τ}| with contribution
    /// weights u_τ ∝ w_τ t_τ, RESTRICTED to the nonzero-determinant
    /// triangles that genuinely overlap another declared triangle
    /// (∈ [0, 1]; 1 = one common determinant-line trivialization).
    /// Invariant under in-band SU(3) frame changes; a full U(3) change
    /// rotates every phase by the same det g, leaving coherence unchanged.
    /// A DISJOINT atlas has no overlap content and reports coherence as
    /// UNKNOWN (NaN), never as a value read off non-overlapping faces.
    double phaseCoherence{0.0};
    /// Circular dispersion 1 − phaseCoherence (NaN when unknown).
    double phaseDispersion{0.0};
    /// How many declared triangles genuinely overlap another declared
    /// triangle — the triangles the coherence resultant runs over.  Zero
    /// on a disjoint atlas.
    std::size_t overlappingTriangles{0};
    /// The sharing relation the overlap was recorded under: "shared-edge",
    /// the only relation an `OrientedTriangle` atlas determines (a triangle
    /// declares its three boundary EDGE rows and their incidence signs, so
    /// a vertex-sharing relation is not derivable here).
    std::string overlapRelation{"shared-edge"};
    /// Krein signature (n₊, n₀, n₋) of each triangle's restricted weight
    /// block S_τ W_τ S_τ (reported separately from the |W_τ|-restricted
    /// score; all-positive in the positive regime).
    std::vector<std::array<int, 3>> kreinSignatures{};
    /// True when every restricted block is positive (n₀ = n₋ = 0 for all
    /// τ) — the regime in which the [0, 1] calibration bound is proven.
    bool positiveRegime{true};
    /// ‖Φ†|W|Φ − I‖_max — the frame-normalization domain certificate
    /// (evaluate() rejects the read when it exceeds the tolerance).
    double frameGramResidual{0.0};
    /// max_τ λ_max(A_τ†A_τ) − 1: ≤ round-off in the decoupled domain; a
    /// positive value flags weight coupling that voids the calibration
    /// bound (score still reported honestly).
    double calibrationMargin{0.0};
    /// The pre-declared convex weighting rule that produced `score`
    /// ("uniform" or "declared"), reported with the datum per the spec's
    /// QuarkRead::anchorWeightingId.
    std::string weightingId{};
    /// The declared convex weights w_τ actually used.
    std::vector<double> weights{};
    /// The #764 certification record grading the calibrated score.
    /// Diagonal-weight path: StructureExact — the [0,1] calibration is a
    /// closed-form identity GIVEN the verified premise Φ†|W|Φ = I on a
    /// decoupled |W|; residual = max(frameGramResidual,
    /// max(0, calibrationMargin)), tolerance = the evaluate() gram
    /// tolerance.  General Hermitian-matrix path: CertifiedNumerical (the
    /// eigen-modulus computation; same residual recipe, conditioning not
    /// measured = NaN).  Regime: PositiveSemidefinite in the positive
    /// regime, HermitianIndefinite in signed sectors.  A
    /// default-constructed profile carries the never-holding
    /// HeuristicDiscovery default — no read travels bare.
    ::tessera::cobordism::Certificate certificate{};
};

/// The triangle-anchor gate the exactness contract requires before any
/// colour-specific kernel runs: "use exact 3×3 determinants and the fixed F₃
/// frame only after a rank-three band passes its triangle-anchor
/// certificate".
///
/// A DEFAULT-CONSTRUCTED gate is closed (`accepted = false`), so a caller that
/// supplies nothing is refused rather than silently admitted.  The only way to
/// open one is `ColorAnchor::gateFor`, which applies the single acceptance
/// predicate `ColorAnchor::accepts` — the SAME conjunction the quark verdict
/// uses, so the kernels and the interpretation can never drift apart.
///
/// The gate carries its own provenance so a refusal can name what failed and
/// an acceptance records what admitted it.
struct AnchorGate {
    /// Whether the profile passed `ColorAnchor::accepts`.
    bool accepted{false};
    /// The atlas score that was graded (NaN when no profile was supplied).
    double score{std::numeric_limits<double>::quiet_NaN()};
    /// The determinant-phase coherence that was graded.
    double phaseCoherence{std::numeric_limits<double>::quiet_NaN()};
    /// The pre-declared weighting rule of the profile ("" when absent).
    std::string weightingId{};
    /// Why the gate is closed ("" when accepted).
    std::string refusalReason{"triangle-anchor certificate absent"};
};

/// # ColorFiber
///
/// The constant, exactly-generated color-sector algebra of THREE oriented
/// edge modes: Λ•C³ = 1 ⊕ 3 ⊕ 3̄ ⊕ 1 with the color interpretation
/// (vacuum / fundamental triplet / antisymmetric anti-triplet / color
/// singlet) layered over the #766 exterior-algebra primitives
/// (quantum::ExteriorAlgebra — the sector projectors and CAR matrices are
/// DELEGATED, never reimplemented).
///
/// All members are static: the algebra is a constant (generated once,
/// checked at startup in debug builds).  Fock-space operators are dense
/// 8×8 matrices on the occupation basis |b⟩, b ∈ {0,1}³, indexed by
/// n(b) = Σ_i b_i 2^i (mode 0 = least-significant bit); the one-occupation
/// (triplet) sector is spanned by Fock indices {1, 2, 4} in that order,
/// identifying it with C³.
///
/// Interpretation stance: these are sector WEIGHTS and algebraic
/// certificates read from caller-supplied data.  Nothing here classifies a
/// particle, and none of it enters the emergence objective.
class ColorFiber {
  public:
    /// Double-precision complex scalar of every operator/state entry.
    using Complex = std::complex<double>;

    ColorFiber() = delete;  // constant algebra — no instances.

    // ── N = 0,1,2,3 exterior-sector projectors (Λ•C³ = 1 ⊕ 3 ⊕ 3̄ ⊕ 1) ──

    /// The 8×8 projector onto total occupation N = `occupation`
    /// (delegates to quantum::ExteriorAlgebra::sectorProjector on three
    /// modes).  Zero matrix for `occupation` > 3.
    [[nodiscard]] static Eigen::MatrixXcd sectorProjector(
        std::size_t occupation);

    /// Λ⁰: the even vacuum singlet (N = 0).
    [[nodiscard]] static Eigen::MatrixXcd vacuumProjector();
    /// Λ¹: the odd fundamental color triplet **3** (N = 1) — the quark
    /// sector read's support.
    [[nodiscard]] static Eigen::MatrixXcd tripletProjector();
    /// Λ²: the even antisymmetric anti-triplet **3̄** (N = 2) — the
    /// diquark sector read's support.
    [[nodiscard]] static Eigen::MatrixXcd antiTripletProjector();
    /// Λ³: the odd top-wedge color singlet (N = 3).
    [[nodiscard]] static Eigen::MatrixXcd singletProjector();

    // ── CAR matrices and bilinears ───────────────────────────────────────

    /// The 8×8 creation matrix a_i† (i ∈ {0,1,2}; delegates to
    /// quantum::ExteriorAlgebra::creationMatrix).
    /// @throws std::invalid_argument for `mode` ≥ 3.
    [[nodiscard]] static Eigen::MatrixXcd creationMatrix(std::size_t mode);

    /// The 8×8 annihilation matrix a_i (adjoint of creationMatrix).
    [[nodiscard]] static Eigen::MatrixXcd annihilationMatrix(std::size_t mode);

    /// The 8×8 bilinear E_ij = a_i† a_j.  Satisfies the gl(3) commutation
    /// relations [E_ij, E_kl] = δ_jk E_il − δ_il E_kj exactly on the whole
    /// Fock representation.
    [[nodiscard]] static Eigen::MatrixXcd hoppingMatrix(std::size_t i,
                                                        std::size_t j);

    /// The Fock indices {1, 2, 4} of the one-occupation basis
    /// (|100⟩, |010⟩, |001⟩ ↔ e₁, e₂, e₃) — the explicit identification of
    /// the N = 1 sector with C³ used by restrictToTriplet.
    [[nodiscard]] static std::array<std::size_t, 3> tripletBasisIndices();

    /// Restrict an 8×8 Fock operator to the one-occupation sector as a 3×3
    /// matrix in the tripletBasisIndices() basis.  Exactly
    /// restrictToTriplet(hoppingMatrix(i, j)) = e_i e_j† (the matrix unit)
    /// and restrictToTriplet(dGamma(M)) = M.
    /// @throws std::invalid_argument if `op` is not 8×8.
    [[nodiscard]] static Eigen::Matrix3cd restrictToTriplet(
        const Eigen::MatrixXcd& op);

    /// The 3×3 matrix unit E_ij on the one-occupation sector (1 at row i,
    /// column j; the gl(3) generators of the color interpretation).
    /// @throws std::invalid_argument for i or j ≥ 3.
    [[nodiscard]] static Eigen::Matrix3cd matrixUnit(std::size_t i,
                                                     std::size_t j);

    /// Second quantization dΓ(M) = Σ_ij M_ij a_i†a_j of a 3×3 one-particle
    /// matrix (delegates to quantum::ExteriorAlgebra::dGamma); its N = 1
    /// restriction is exactly M.
    [[nodiscard]] static Eigen::MatrixXcd dGamma(const Eigen::Matrix3cd& m);

    // ── the eight normalized Gell-Mann generators on N = 1 ──────────────

    /// λ_a for a ∈ {1,…,8}, assembled from the matrix units
    /// (λ₃ = H₁ = E₁₁−E₂₂, λ₈ = H₂ = (E₁₁+E₂₂−2E₃₃)/√3): Hermitian,
    /// traceless, Tr(λ_a λ_b) = 2δ_ab.
    /// @throws std::invalid_argument for `a` outside 1..8.
    [[nodiscard]] static Eigen::Matrix3cd gellMann(int a);

    // ── the traceless adjoint-octet projector (3 ⊗ 3̄ = 1 ⊕ 8) ─────────

    /// The 9×9 orthogonal projector P₈ = I₉ − vec(I)vec(I)†/3 onto the
    /// traceless (adjoint-octet) part of a 3×3 bilinear, acting on
    /// column-major vec(M) (Eigen's convention: vec index = i + 3j for
    /// M(i, j)); P₈ vec(M) = vec(M − (Tr M / 3) I).
    [[nodiscard]] static Eigen::MatrixXcd adjointOctetProjector();

    /// The traceless part M − (Tr M / 3) I — the octet component of a
    /// bilinear in 3 ⊗ 3̄ = 1 ⊕ 8.
    [[nodiscard]] static Eigen::Matrix3cd tracelessPart(
        const Eigen::Matrix3cd& m);

    // ── #774 additions BESIDE the octet projector (nothing above is
    //    re-derived): the singlet complement resolving 3 ⊗ 3̄, the literal
    //    traceless even bilinear on Fock space, and the adjoint quadratic
    //    Casimir — each an exact constant of the same algebra. ────────────

    /// The 9×9 orthogonal projector P₁ = vec(I)vec(I)†/3 onto the trace
    /// (singlet) part of a 3×3 bilinear — implemented literally as
    /// I₉ − adjointOctetProjector(), so P₁ + P₈ = I₉ is an exact (bitwise)
    /// complement: the singlet and octet projectors RESOLVE 3 ⊗ 3̄ = 1 ⊕ 8.
    /// P₁ vec(M) = vec((Tr M / 3) I).  (#774)
    [[nodiscard]] static Eigen::MatrixXcd adjointSingletProjector();

    /// The 8×8 traceless even bilinear T_ij = a_i†a_j − (δ_ij/3) N̂ on Fock
    /// space (whitepaper "Fock space as an inductive limit of
    /// interactions") — implemented literally as
    /// dGamma(tracelessPart(matrixUnit(i, j))), so the delegation is exact:
    /// T_ij = E_ij − (δ_ij/3) Σ_k E_kk, Σ_i T_ii = 0 exactly, T_ij
    /// conserves N (even fermion parity: [T_ij, (−1)^N] = 0), and the nine
    /// T_ij span the 8-dimensional octet of 3 ⊗ 3̄ = 1 ⊕ 8.  (#774)
    /// @throws std::invalid_argument for i or j ≥ 3.
    [[nodiscard]] static Eigen::MatrixXcd octetBilinear(std::size_t i,
                                                        std::size_t j);

    /// The 9×9 quadratic Casimir of the ADJOINT action on 3 ⊗ 3̄:
    /// C = Σ_a K_a² with K_a vec(M) = vec([λ_a/2, M]).  Exact identity
    /// (checked by verifyConstantAlgebra): C = 3 P₈ — the Casimir
    /// eigenvalue is 0 on the singlet and C₂(adjoint) = 3 on the octet, so
    /// the Casimir and the octet projector are the SAME certificate up to
    /// the constant 3.  (#774)
    [[nodiscard]] static Eigen::MatrixXcd adjointCasimirMatrix();

    /// The adjoint-Casimir Rayleigh quotient
    /// ⟨vec M, C vec M⟩ / ‖M‖_F² ∈ [0, 3] of a 3×3 bilinear: exactly 3 for
    /// a traceless (pure octet) M, exactly 0 for M ∝ I, and 3 × (octet
    /// weight fraction) in between.  Evaluated through the EXACT identity
    /// C = 3 P₈ as 3 ‖tracelessPart(M)‖_F² / ‖M‖_F² (the independent
    /// commutator-sum construction stays in adjointCasimirMatrix, where
    /// verifyConstantAlgebra cross-checks it).  NaN for M = 0 (an
    /// undefined quotient is reported unknown, never zero).  (#774)
    [[nodiscard]] static double adjointCasimir(const Eigen::Matrix3cd& m);

    // ── the exact Fourier color frame from ω = e^{2πi/3} ─────────────────

    /// The primitive cube root of unity as its ALGEBRAIC value
    /// ω = (−1 + i√3)/2 (never exp(2πi/3) — so 1 + ω + ω² and ω·ω̄ = 1
    /// cancel exactly in floating point).
    [[nodiscard]] static Complex omega();

    /// The exact unitary Fourier frame F₃ with entries ω^{jk}/√3 —
    /// assembled from the algebraic table {1, ω, ω²} by exponent jk mod 3,
    /// never by repeated multiplication.  F₃†F₃ = I and |det F₃| = 1.
    [[nodiscard]] static Eigen::Matrix3cd fourierFrame();

    /// Column k of F₃: the Z₃ character vector (1, ω^k, ω^{2k})/√3.
    /// @throws std::invalid_argument for k outside 0..2.
    [[nodiscard]] static Eigen::Vector3cd fourierBasisVector(int k);

    /// The existing phase pattern (1, ω, ω²)/√3 — identified as ONE color
    /// basis vector (fourierBasisVector(1)), not by itself the whole color
    /// fiber; its cyclic orbit under pointwise Z₃ powers is the exact
    /// orthonormal triad {fourierBasisVector(0), (1), (2)} = the columns
    /// of F₃.
    [[nodiscard]] static Eigen::Vector3cd omegaPhaseState();

    // ── normalization: perimeter (L¹ scale gauge) vs Hilbert (L² state) ──

    /// The triangle perimeter Σ_i |z_i|^{1/2} of three stored complex
    /// SQUARED lengths z_i = ℓ_i² (the L¹ geometric datum on the side
    /// lengths |ℓ_i| = |z_i|^{1/2}).
    [[nodiscard]] static double perimeter(const Eigen::Vector3cd& z);

    /// Rescale the squared lengths so the perimeter is one:
    /// z ↦ z / perimeter(z)² (lengths scale by 1/perimeter).  A GEOMETRIC
    /// SCALE GAUGE only — an L¹ condition that is NOT a state
    /// normalization and never replaces the L² Hilbert normalization
    /// (out-of-scope per #767: perimeter one is not ⟨c|c⟩ = 1).
    /// @throws std::invalid_argument when the perimeter vanishes.
    [[nodiscard]] static Eigen::Vector3cd perimeterNormalized(
        const Eigen::Vector3cd& z);

    /// The Hilbert L² norm ‖z‖₂ = (Σ_i |z_i|²)^{1/2}.
    [[nodiscard]] static double hilbertNorm(const Eigen::Vector3cd& z);

    /// The Hilbert-normalized state z / ‖z‖₂ with ⟨c|c⟩ = 1 — the STATE
    /// normalization, distinct from the perimeter gauge.
    /// @throws std::invalid_argument when ‖z‖₂ = 0.
    [[nodiscard]] static Eigen::Vector3cd hilbertNormalized(
        const Eigen::Vector3cd& z);

    /// The color vector formed from the stored complex squared lengths:
    /// c = z / ‖z‖₂ (exactly hilbertNormalized — the deliverable's name).
    [[nodiscard]] static Eigen::Vector3cd colorVector(
        const Eigen::Vector3cd& z);

    // ── det(C) / det(C†C) color-wedge and singlet certificates ──────────

    /// The color-wedge (singlet) amplitude det C = ε_ijk C_i1 C_j2 C_k3 of
    /// three color columns — the invariant volume S_ABC.  Invariant under
    /// a common g ∈ SU(3): det(gC) = det(C).
    [[nodiscard]] static Complex colorWedge(const Eigen::Matrix3cd& c);

    /// colorWedge of three explicit color columns.
    [[nodiscard]] static Complex colorWedge(const Eigen::Vector3cd& a,
                                            const Eigen::Vector3cd& b,
                                            const Eigen::Vector3cd& c);

    /// The singlet Gram certificate det(C†C) = |det C|² =
    /// ‖c₁∧c₂∧c₃‖² : exactly zero for duplicate color modes and exactly
    /// one for an orthonormal triad (Pauli/Gram identity).
    [[nodiscard]] static double singletGram(const Eigen::Matrix3cd& c);

    /// Certify g ∈ SU(3): ‖g†g − I‖_max ≤ tol and |det g − 1| ≤ tol.
    [[nodiscard]] static bool isSpecialUnitary(const Eigen::Matrix3cd& g,
                                               double tol = 1e-12);

    // ── sector READS (weights only — never a particle classification) ───

    /// Occupation-sector weights of an 8-dimensional Fock vector: the
    /// squared norms ‖P_N ψ‖² for N = 0..3.  `vacuum` + `quark` +
    /// `antiTriplet` + `singlet` = ‖ψ‖² exactly.
    struct SectorWeights {
        /// ‖P₀ψ‖² — the even vacuum singlet weight.
        double vacuum{0.0};
        /// ‖P₁ψ‖² — the odd fundamental-triplet (quark-sector) weight.
        double quark{0.0};
        /// ‖P₂ψ‖² — the even anti-triplet (diquark-sector) weight.
        double antiTriplet{0.0};
        /// ‖P₃ψ‖² — the odd top-wedge color-singlet weight.
        double singlet{0.0};
    };

    /// The four sector weights of `state` (must be 8-dimensional).
    /// @throws std::invalid_argument on a size mismatch.
    [[nodiscard]] static SectorWeights sectorWeights(
        const Eigen::VectorXcd& state);

    /// Frobenius split of a 3×3 bilinear under 3 ⊗ 3̄ = 1 ⊕ 8.
    struct OctetRead {
        /// ‖M − (Tr M / 3) I‖_F² — the traceless adjoint-octet weight.
        double octet{0.0};
        /// |Tr M|² / 3 — the singlet (trace) weight; octet + singlet =
        /// ‖M‖_F² exactly.
        double singlet{0.0};
    };

    /// The octet/singlet Frobenius weights of a 3×3 bilinear (for example
    /// a one-occupation density or a quark-antiquark bilinear).
    [[nodiscard]] static OctetRead octetRead(const Eigen::Matrix3cd& m);

    // ── startup self-check ───────────────────────────────────────────────

    /// Re-derive every constant-algebra identity (F₃†F₃ = I, |det F₃| = 1,
    /// λ_a Hermitian/traceless, Tr(λ_a λ_b) = 2δ_ab, the full gl(3)
    /// commutator table on both representations, projector
    /// idempotence/completeness, P₈ rank 8) and return the maximum
    /// absolute residual.  Run once at startup in debug builds (NDEBUG
    /// off); callable from tests and bindings in every build.
    [[nodiscard]] static double verifyConstantAlgebra();

    /// The #764 certificate of the constant algebra: AlgebraicallyExact /
    /// Static / PositiveSemidefinite with the measured
    /// verifyConstantAlgebra() residual against the startup tolerance
    /// 1e-12 — the same claim the debug-build startup check enforces,
    /// attached in the shared certification vocabulary.
    [[nodiscard]] static ::tessera::cobordism::Certificate
    constantAlgebraCertificate();
};

/// # ColorAnchor
///
/// The calibrated weighted oriented-triangle anchoring kernel for an
/// abstract rank-three band (whitepaper "Quarks as modular clusters";
/// design spec §11).
///
/// For each declared oriented triangle τ with restriction
/// R_τ : C₁ → C³ (the three ordered boundary edges with incidence signs)
/// and restricted weight block W_τ, the weighted anchor matrix is
///
///     A_τ = |W_τ|^{1/2} R_τ Φ,
///
/// and the atlas score is a² = Σ_τ w_τ |det A_τ|² with the convex
/// weighting {w_τ} DECLARED BEFORE the data are examined: the weighting is
/// fixed at construction (or via declareWeights BEFORE the first
/// evaluate); once any data have been evaluated, re-weighting throws —
/// post-hoc weight selection is rejected by construction.
///
/// ## Exact identity and domain
///
/// With the frame |W|-orthonormal (Φ†|W|Φ = I, verified per evaluate to
/// `gramTolerance` and reported as frameGramResidual) and |W|
/// triangle-decoupled — in particular ANY diagonal per-edge metric, the
/// production DEC/Hodge case — each
/// |det A_τ|² = det(A_τ†A_τ) ≤ 1 exactly, because R_τ†|W_τ|R_τ ⪯ |W|;
/// hence a² ∈ [0, 1], reaching 1 exactly at full concentration on the
/// weighted edge span of the anchoring faces.  A single literal triangle
/// covering the whole band is the exact oracle (a² = 1 to round-off); an
/// extended anchored fiber is the production case.  For a general
/// Hermitian (coupled) weight matrix the score is still reported, and the
/// per-read `calibrationMargin` certifies whether the ≤ 1 bound held
/// numerically rather than assuming it.
///
/// In signed sectors the restriction still uses |W_τ|^{1/2} (matrix
/// modulus of the sign-conjugated block S_τ W_τ S_τ) and the restricted
/// block's Krein signature is reported separately per triangle; the [0,1]
/// calibration is proven in the positive regime.
///
/// ## Invariances (tested)
///
/// • In-band SU(3) frame change Φ ↦ Φg: every det A_τ, the score and the
///   whole profile are invariant; a full U(3) change rotates all
///   determinant phases by the common det g (coherence/dispersion
///   invariant).
/// • Oriented edge relabeling: permuting the edge rows (frame + weights)
///   with the matching triangle re-indexing, or reversing a stored edge
///   orientation (row sign flip + incidence sign flip), changes nothing.
/// • Cyclic rotation of a triangle's (edges, signs) is even and leaves
///   det A_τ itself invariant; an odd permutation (orientation reversal)
///   negates det A_τ and fixes |det A_τ|².
///
/// Pure read: operates only on caller-supplied inputs, mutates nothing,
/// and never enters the emergence objective.
class ColorAnchor {
  public:
    /// Double-precision complex scalar of the frame/weight entries.
    using Complex = std::complex<double>;

    /// Eigenvalues of a restricted block with |λ| ≤ kreinTolerance() count
    /// as zero in the reported Krein signature.
    [[nodiscard]] static double kreinTolerance() { return 1e-12; }

    /// Declare the atlas with the UNIFORM convex weighting w_τ = 1/T.
    /// @throws std::invalid_argument on an empty atlas, a repeated edge
    ///         inside one triangle, a sign outside {−1,+1}, or a negative
    ///         edge index.
    explicit ColorAnchor(std::vector<OrientedTriangle> triangles);

    /// Declare the atlas with an EXPLICIT convex weighting (one weight per
    /// triangle, each ≥ 0, summing to one within 1e-12).
    /// @throws std::invalid_argument when the weighting is not convex or
    ///         sized to the atlas.
    ColorAnchor(std::vector<OrientedTriangle> triangles,
                std::vector<double> weights);

    /// The declared oriented triangles (immutable).
    [[nodiscard]] const std::vector<OrientedTriangle>& triangles() const {
        return triangles_;
    }

    /// Whether declared triangle `index` shares a boundary EDGE with some
    /// other declared triangle — the overlap relation the determinant-phase
    /// coherence is recorded on (whitepaper: "their coherence on
    /// OVERLAPPING triangles is recorded separately").  An
    /// `OrientedTriangle` names only its three boundary edge rows, so
    /// shared-edge is the only sharing relation this atlas determines.
    [[nodiscard]] bool overlapsAnother(std::size_t index) const {
        return index < overlapping_.size() &&
               overlapping_[index] != 0;
    }

    /// How many declared triangles overlap another declared triangle
    /// (0 on a disjoint atlas, where the coherence is UNKNOWN).
    [[nodiscard]] std::size_t overlappingTriangleCount() const {
        return overlapCount_;
    }

    /// The declared convex weights.
    [[nodiscard]] const std::vector<double>& weights() const {
        return weights_;
    }

    /// "uniform" or "declared" — reported in every AnchorProfile.
    [[nodiscard]] const std::string& weightingId() const {
        return weightingId_;
    }

    /// True once any data have been evaluated: the weighting is sealed.
    [[nodiscard]] bool sealed() const { return sealed_; }

    /// Replace the declared convex weighting — allowed ONLY before the
    /// first evaluate().  Afterwards the data have been examined and
    /// post-hoc weight selection is rejected.
    /// @throws std::logic_error after any evaluate();
    ///         std::invalid_argument when the weighting is not convex.
    void declareWeights(std::vector<double> weights);

    /// Evaluate the calibrated anchor of a rank-three frame against a
    /// DIAGONAL (possibly signed) per-edge weight vector — the production
    /// DEC/Hodge metric case, where the [0,1] calibration bound is exact.
    ///
    /// `frame` is E×3 over the component's oriented edges (rows = the
    /// caller's edge indexing; columns = the band).  `edgeWeights` has
    /// length E.  The frame must be |W|-orthonormal:
    /// ‖Φ†|W|Φ − I‖_max ≤ gramTolerance (use orthonormalizeFrame to
    /// prepare one); the residual is reported in the profile.
    /// @throws std::invalid_argument on shape mismatches, an edge index
    ///         out of range, or a violated frame normalization.
    [[nodiscard]] AnchorProfile evaluate(const Eigen::MatrixXcd& frame,
                                         const Eigen::VectorXd& edgeWeights,
                                         double gramTolerance = 1e-9);

    /// Evaluate against a general Hermitian E×E weight matrix.  |W| and
    /// each |W_τ| are the self-adjoint matrix moduli (eigen-modulus); the
    /// ≤ 1 calibration certificate is CHECKED (calibrationMargin), not
    /// assumed, because a coupled W need not satisfy R_τ†|W_τ|R_τ ⪯ |W|.
    /// @throws std::invalid_argument on shape mismatches, a non-Hermitian
    ///         weight, or a violated frame normalization.
    [[nodiscard]] AnchorProfile evaluate(const Eigen::MatrixXcd& frame,
                                         const Eigen::MatrixXcd& weight,
                                         double gramTolerance = 1e-9);

    /// The 3×3 weighted anchor matrix A_τ = |W_τ|^{1/2} R_τ Φ of one
    /// triangle against a diagonal per-edge weight vector (no
    /// normalization check — the raw restriction, for tests and callers
    /// composing their own certificates).
    [[nodiscard]] static Eigen::Matrix3cd anchorMatrix(
        const Eigen::MatrixXcd& frame, const Eigen::VectorXd& edgeWeights,
        const OrientedTriangle& tri);

    /// The |W|-orthonormalized frame Φ (Φ†|W|Φ)^{−1/2} for a diagonal
    /// per-edge weight vector — the canonical way to enter evaluate()'s
    /// domain.  @throws std::invalid_argument when the frame is
    /// rank-deficient in the |W| inner product.
    [[nodiscard]] static Eigen::MatrixXcd orthonormalizeFrame(
        const Eigen::MatrixXcd& frame, const Eigen::VectorXd& edgeWeights);

    /// Matrix-weight overload of orthonormalizeFrame (Hermitian W; uses
    /// the eigen-modulus |W|).
    [[nodiscard]] static Eigen::MatrixXcd orthonormalizeFrame(
        const Eigen::MatrixXcd& frame, const Eigen::MatrixXcd& weight);

    /// Default atlas-score floor of the acceptance predicate, mirroring the
    /// quark verdict's own configured floor.
    static constexpr double kDefaultMinScore = 0.5;
    /// Default determinant-phase coherence floor of the acceptance predicate.
    static constexpr double kDefaultMinPhaseCoherence = 0.5;

    /// **The** triangle-anchor acceptance predicate — ONE definition, shared
    /// by the quark verdict and by the colour kernels that the exactness
    /// contract gates on it, so the two can never drift apart.  A profile
    /// passes when a weighting was actually declared (an empty
    /// `weightingId` is MISSING evidence, not a zero score), its calibration
    /// certificate holds, and both the atlas score and the determinant-phase
    /// coherence meet their floors.
    [[nodiscard]] static bool accepts(
        const AnchorProfile& profile, double minScore = kDefaultMinScore,
        double minPhaseCoherence = kDefaultMinPhaseCoherence);

    /// The gate for a profile: `accepts` plus the provenance a refusal needs
    /// to name what failed.  The only way to open an `AnchorGate`.
    [[nodiscard]] static AnchorGate gateFor(
        const AnchorProfile& profile, double minScore = kDefaultMinScore,
        double minPhaseCoherence = kDefaultMinPhaseCoherence);

  private:
    static void validateTriangles(const std::vector<OrientedTriangle>& tris);
    static void validateConvex(const std::vector<double>& weights,
                               std::size_t count);
    /// Shared scoring core over precomputed per-triangle |W_τ|^{1/2}
    /// blocks and their (already exact) Krein signatures.
    /// `diagonalWeights` selects the certificate grade (StructureExact
    /// closed-form vs CertifiedNumerical eigen-modulus).
    [[nodiscard]] AnchorProfile evaluateBlocks(
        const Eigen::MatrixXcd& frame,
        const std::vector<Eigen::Matrix3cd>& sqrtBlocks,
        const std::vector<std::array<int, 3>>& signatures,
        const Eigen::MatrixXcd& gram, double gramTolerance,
        bool diagonalWeights);

    /// Fill `overlapping_` / `overlapCount_` from the declared atlas: a
    /// triangle overlaps when one of its three edge rows also appears in a
    /// DIFFERENT declared triangle.  Computed once, at declaration time,
    /// before any datum is examined.
    void markOverlaps();

    std::vector<OrientedTriangle> triangles_{};
    std::vector<double> weights_{};
    std::string weightingId_{};
    /// Per-triangle shared-edge overlap flags (char to keep an addressable
    /// element type; 1 = overlaps another declared triangle).
    std::vector<char> overlapping_{};
    std::size_t overlapCount_{0};
    bool sealed_{false};
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_COLORFIBER_H
