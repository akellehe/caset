// Majorization partial order on probability distributions, plus the
// Hasse-diagram construction we use to build the majorization poset of
// Schmidt spectra of an MPS (Phase 3 of docs/source/quantum-plan.md).
//
// The Poset / OrderAgreement / compareOrders types themselves live at
// the top of tessera (`include/Poset.h`) so they're shareable with non-
// quantum analyses; this header re-exports them under
// `tessera::quantum::` for backward compatibility with Phase 3-5 code.
//
// ─── Bibliographic references used throughout this file ─────────────────
//
// {N1999}    Nielsen, M. A. (1999), "Conditions for a class of entanglement
//            transformations", Phys. Rev. Lett. 83, 436. arXiv:quant-ph/9811053.
//            PDF in docs/source/resources/quantum/Nielsen1999_LOCC_majorization.pdf.
//            — Eq. (1) is the canonical cumulative-sum-dominance definition of
//            majorization. The main theorem (unnumbered, p. 437) is the iff
//            statement: |α⟩ → |β⟩ is achievable by deterministic LOCC iff
//            λ_α ≺ λ_β (Schmidt-spectrum majorization).
//
// {AN2008}   Aubrun, G. & Nechita, I. (2008), "Stochastic domination for
//            iterated convolutions and catalytic majorization", Comm. Math.
//            Phys. 278, 133. arXiv:0707.0211. PDF in docs/source/resources/
//            quantum/AubrunNechita2008_CatalyticMajorization.pdf.
//            — Theorem 1.1 / Proposition 2.5 give the L^p-norm-dominance
//            characterization of asymptotic / catalytic majorization. We
//            cite this for the family of *stricter-than-classical*
//            majorization variants of which `PeakRadialMajorization` below
//            is a hand-tuned member.
//
// {B2015}    Brändén, P. (2015), "Unimodality, log-concavity, real-rootedness
//            and beyond", Handbook of Enumerative Combinatorics (CRC Press).
//            arXiv:1410.6601. PDF in docs/source/resources/quantum/
//            Branden2015_Unimodality_LogConcavity.pdf. — §1 introduces
//            log-concavity (a_i² ≥ a_{i-1} · a_{i+1}); §4 surveys its
//            structural consequences. The earlier canonical reference is
//            Stanley 1989, Ann. NY Acad. Sci. 576, 500 (not on arXiv).
//
// {B1997}    Bhatia, R. (1997), "Matrix Analysis", Springer GTM 169,
//            Chapter II ("Majorisation"). The clean mathematical-textbook
//            account; cited by {N1999} as the principal majorization
//            reference. (Copyrighted; not in resources.)
//
// {MOA2011}  Marshall, A. W., Olkin, I. & Arnold, B. C. (2011),
//            "Inequalities: Theory of Majorization and Its Applications",
//            Springer (2nd ed.). The encyclopedic reference. (Copyrighted;
//            not in resources.)
//
// ─── Majorization recap ─────────────────────────────────────────────────
//
// Given two finite, non-negative sequences μ and λ, both normalised to the
// same total mass, μ *majorizes* λ (written μ ≻ λ or λ ≼ μ) iff
//
//   sum_{i=1..k} μ_i^↓  ≥  sum_{i=1..k} λ_i^↓     for every k = 1, 2, …
//
// — this is {N1999} eq. (1), with x_i^↓ denoting the entries of x sorted
// non-increasingly and the shorter vector zero-padded to the longer's
// length. Intuitively, μ is "more concentrated" than λ. For probability
// distributions the total-mass equality at k = d is automatic.
//
// The classical relation is reflexive, antisymmetric (modulo
// permutation/zero-padding), and transitive on the simplex — see
// {B1997} §II.1 for the standard treatment — so it is a partial order
// on equivalence classes of zero-padded sorted vectors.
//
// In our application the vectors are Schmidt spectra of an MPS at
// different (interval, time) labels (PLAN.md §5 Phase 3 and the
// methodology page docs/source/quantum-methodology.md). {N1999}'s main
// theorem gives the LOCC-conversion interpretation: |α⟩ can be
// deterministically converted into |β⟩ iff the Schmidt spectrum of |α⟩
// is majorized by that of |β⟩.
//
// ─── Variants ────────────────────────────────────────────────────────────
//
// Three concrete variants share the same `MajorizationPredicate` contract.
// Each is reflexive, antisymmetric, and transitive on the simplex, so
// each induces a partial order suitable for the same Hasse-poset and
// causal-comparison machinery already used for classical majorization.
// New variants are added by subclassing `MajorizationPredicate` rather
// than by editing every call site.

#pragma once

#include "Poset.h"  // top-level tessera::Poset / OrderAgreement / compareOrders

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tessera::quantum {

// Aliases keeping the Phase 3-5 code paths working unchanged. The
// canonical types live in `tessera::` so non-quantum analyses can use
// them too.
using Poset = ::tessera::Poset;
using OrderAgreement = ::tessera::OrderAgreement;
using ::tessera::compareOrders;

// ─── Variant contract ────────────────────────────────────────────────────

// Abstract base class: every concrete variant of "μ majorizes λ" honours
// this single contract. Pass instances by `const&` everywhere; ownership
// stays with the caller.
//
// Subclasses MUST satisfy the partial-order axioms on the simplex:
//   • reflexivity:    majorizes(x, x) is true for every probability x;
//   • antisymmetry:   majorizes(x, y) and majorizes(y, x) implies x and
//                     y are equivalent (sorted-padded equal, or whatever
//                     equivalence the variant uses);
//   • transitivity:   majorizes(x, y) and majorizes(y, z) implies
//                     majorizes(x, z).
// A subclass that violates any of these breaks `majorizationPoset` (the
// transitive-reduction would fail to converge to a Hasse diagram).
//
// References:
//   {N1999}, eq. (1) and main theorem — classical case, the contract this
//     interface generalises.
class MajorizationPredicate {
public:
    virtual ~MajorizationPredicate() = default;

    // True iff μ majorizes λ under this variant. Implementations are
    // responsible for sorting / zero-padding; callers pass plain
    // non-negative sequences.
    [[nodiscard]] virtual bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const = 0;

    // μ strictly majorizes λ iff μ ≻ λ but not λ ≻ μ. The default
    // implementation is two `majorizes` calls; subclasses with a
    // cheaper strict path can override.
    [[nodiscard]] virtual bool
    strictlyMajorizes(std::vector<double> const& mu,
                       std::vector<double> const& lambda) const {
        return majorizes(mu, lambda) && !majorizes(lambda, mu);
    }

    // Short identifier used in diagnostics and Python repr ("standard",
    // "log-concave", "peak-radial", …). Stable across versions; safe to
    // use as a dictionary key in result tables.
    [[nodiscard]] virtual std::string name() const = 0;

protected:
    MajorizationPredicate() = default;
    MajorizationPredicate(MajorizationPredicate const&) = default;
    MajorizationPredicate(MajorizationPredicate&&) noexcept = default;
    MajorizationPredicate& operator=(MajorizationPredicate const&) = default;
    MajorizationPredicate& operator=(MajorizationPredicate&&) noexcept = default;
};

// ─── Concrete variants ───────────────────────────────────────────────────

// Classical majorization, exactly as in {N1999} eq. (1):
//
//     μ ≻ λ   ⟺   ∑_{i=1..k} μ_i^↓  ≥  ∑_{i=1..k} λ_i^↓   ∀ k = 1..d
//                                       ∧  ∑ μ_i  =  ∑ λ_i .
//
// This is the predicate that defines the "≼_maj" order in the Phase 5
// causal-order comparison. {N1999}'s main theorem says this is exactly
// the partial order under which deterministic LOCC convertibility
// |α⟩ → |β⟩ holds (with Schmidt spectra λ_α, λ_β substituted for μ, λ).
//
// References: {N1999} eq. (1); {B1997} §II.1; {MOA2011} §1.A.
class StandardMajorization : public MajorizationPredicate {
public:
    explicit StandardMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] double tol() const noexcept { return tol_; }

protected:
    double tol_;  // shared by subclasses that want the same tolerance
};

// Standard majorization, restricted to spectra that are *log-concave*
// on their support. A non-negative sequence (a_0, a_1, …, a_{r-1}) is
// log-concave (or "PF₂", in the language of {B2015} §1) iff
//
//     a_i²  ≥  a_{i-1} · a_{i+1}     for every interior i.
//
// We require the relation only on the support — i.e. trailing zeros
// (which are forced by the descending sort) trivially satisfy it.
// Pairs where either spectrum fails log-concavity are declared
// incomparable, so this variant is a *sub-relation* of
// `StandardMajorization`.
//
// Motivation. Phase-5 numerics (docs/source/quantum-experiments/
// lightcone_vs_majorization_writeup.md, §"Reading toward Phase 6") show
// that ~50 % of `≼_maj`-related pairs lie outside the Lieb-Robinson
// cone. A natural conjecture is that much of this discordance comes
// from comparisons between unimodal "single-peak" spectra (typical of
// near-product states) and multimodal "plateau" spectra (typical
// post-pair-production states), where standard majorization happens
// to relate distributions that are physically different in shape.
// Filtering the relation to log-concave-only spectra tests this
// conjecture: if the strong-falsification fraction shrinks
// dramatically, the discordance was an artefact of cross-shape
// comparisons; if it stays, the discordance is structural.
//
// References:
//   {B2015} §1, definition of log-concavity (a_i² ≥ a_{i-1} a_{i+1});
//           §4 for structural consequences.
//   Stanley 1989, Ann. NY Acad. Sci. 576, 500 — earlier canonical
//           survey of log-concave / unimodal sequences (not on arXiv).
class LogConcaveMajorization : public StandardMajorization {
public:
    explicit LogConcaveMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    // Predicate for log-concavity of a single sorted-descending
    // spectrum. After sorting and stripping trailing zeros, requires
    // s_i² ≥ s_{i-1} · s_{i+1} for every interior i. Spectra of length
    // ≤ 2 are trivially log-concave.
    [[nodiscard]] static bool
    isLogConcave(std::vector<double> const& v, double tol = 1e-12);
};

// Peak-radial dominance: μ ≻ λ iff, after sorting both descending and
// zero-padding to a common length,
//
//     λᵢ / λ₁  ≤  μᵢ / μ₁     for every i.
//
// Equivalently (multiplying through, which is what the implementation
// does to stay clean for small peaks):
//
//     λᵢ · μ₁  ≤  μᵢ · λ₁     for every i.
//
// In words: μ decays from its peak at least as fast as λ does, in the
// relative-to-peak sense. This is *strictly stronger* than classical
// majorization — every cumulative-sum dominance can be derived from
// entrywise ratio dominance, but not vice versa, so {`peak-radial μ ≻ λ`} ⊂
// {`standard μ ≻ λ`}.
//
// Where it fits in the literature. {AN2008} Theorem 1.1 / Prop. 2.5
// characterise *catalytic* majorization (a *weakening* of the standard
// order, in which extra ancillas are allowed) by L^p-norm dominance for
// p ∈ ℝ; their characterisation runs in the opposite direction from
// ours, but the technical machinery (cross-multiplied entrywise
// inequalities) is the same shape. We are not aware of a published
// definition of the present "peak-radial" variant; it is a hand-tuned
// candidate motivated by the same Phase-5 reading-2 conjecture as
// `LogConcaveMajorization`, designed to upgrade "concentration" from
// sorted-cumulative-sum dominance to sorted-relative-to-peak
// dominance.
//
// Partial-order axioms.
//   reflexive   — λᵢ · λ₁ ≤ λᵢ · λ₁ trivially.
//   antisymmetric — if μ ≻ λ and λ ≻ μ then λᵢ μ₁ ≤ μᵢ λ₁ ≤ λᵢ μ₁ for
//                   every i, so μ and λ are scalar multiples on their
//                   support; with both normalised to total mass 1
//                   this forces μ↓ = λ↓.
//   transitive  — chain the entrywise inequalities and divide out the
//                 (positive) peaks.
//
// References:
//   {AN2008} Theorem 1.1 / Prop. 2.5 — closest published analog (different
//             direction); structural similarity to ratio/Lp dominance.
//   {N1999}  — the standard order this variant strengthens.
class PeakRadialMajorization : public MajorizationPredicate {
public:
    explicit PeakRadialMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] double tol() const noexcept { return tol_; }

private:
    double tol_;
};

// ─── Poset construction ──────────────────────────────────────────────────

// Construct the majorization poset on the given list of spectra under
// the chosen variant.
//
// `spectra[k]` becomes node k. The resulting Poset stores Hasse cover
// edges only — the transitive closure is implicit (recover with the
// usual reachability traversal).
//
// Complexity: O(M³) for M = spectra.size(), dominated by the
// transitive-reduction pass. Each predicate call is O(L log L) on the
// spectrum lengths L (sorting); for our use case L is the MPS bond
// dimension, so L ≤ a few hundred.
//
// References:
//   {N1999} main theorem — the partial order this poset Hasse-encodes
//   for the `StandardMajorization` predicate.
Poset majorizationPoset(std::vector<std::vector<double>> const& spectra,
                         MajorizationPredicate const& predicate);

// ─── Backward-compatible free-function API ───────────────────────────────
//
// These wrappers correspond to the pre-OO surface area that Phase 3-5
// code, the C++ unit tests, and the existing Python bindings already
// depend on. New code should construct an explicit
// `MajorizationPredicate` subclass and call its methods directly, or
// use the predicate-taking `majorizationPoset` overload above.

// Equivalent to `StandardMajorization{tol}.majorizes(mu, lambda)`.
// This is the predicate of {N1999} eq. (1).
bool majorizes(std::vector<double> const& mu,
               std::vector<double> const& lambda,
               double tol = 1e-12);

// Equivalent to `StandardMajorization{tol}.strictlyMajorizes(mu, lambda)`.
bool strictlyMajorizes(std::vector<double> const& mu,
                        std::vector<double> const& lambda,
                        double tol = 1e-12);

// Equivalent to `majorizationPoset(spectra, StandardMajorization{tol})`.
Poset majorizationPoset(std::vector<std::vector<double>> const& spectra,
                         double tol = 1e-12);

} // namespace tessera::quantum
