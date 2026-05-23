// Majorization partial order on probability distributions, plus the
// Hasse-diagram construction we use to build the majorization poset of
// Schmidt spectra of an MPS (see docs/source/quantum-plan.md).
//
// The Poset / OrderAgreement types themselves live at the top of tessera
// (`include/Poset.h`) so they're shareable with non-quantum analyses;
// this header re-exports them under `tessera::quantum::` for ergonomics
// and adds the quantum-specific predicate hierarchy plus the
// `Majorization` coarse-grained façade.
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
//            characterization of asymptotic / catalytic majorization.
//
// {B2015}    Brändén, P. (2015), "Unimodality, log-concavity, real-rootedness
//            and beyond", Handbook of Enumerative Combinatorics (CRC Press).
//            arXiv:1410.6601. PDF in docs/source/resources/quantum/
//            Branden2015_Unimodality_LogConcavity.pdf. — §1 introduces
//            log-concavity (a_i² ≥ a_{i-1} · a_{i+1}); §4 surveys its
//            structural consequences.
//
// {B1997}    Bhatia, R. (1997), "Matrix Analysis", Springer GTM 169,
//            Chapter II ("Majorisation"). The clean mathematical-textbook
//            account; cited by {N1999} as the principal majorization
//            reference.
//
// {MOA2011}  Marshall, A. W., Olkin, I. & Arnold, B. C. (2011),
//            "Inequalities: Theory of Majorization and Its Applications",
//            Springer (2nd ed.). The encyclopedic reference.
//
// ─── Majorization recap ─────────────────────────────────────────────────
//
// Given two finite, non-negative sequences μ and λ, both normalised to the
// same total mass, μ majorizes λ (written μ ≻ λ) iff
//
//   sum_{i=1..k} μ_i^↓  ≥  sum_{i=1..k} λ_i^↓     for every k = 1, 2, …
//
// — this is {N1999} eq. (1), with x_i^↓ denoting the entries of x sorted
// non-increasingly and the shorter vector zero-padded to the longer's
// length. Intuitively, μ is "more concentrated" than λ. For probability
// distributions the total-mass equality at k = d is automatic.

#pragma once

#include "Poset.h"  // top-level tessera::Poset / OrderAgreement / compareOrders

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tessera::quantum {

// Aliases keeping the quantum-side code paths working unchanged. The
// canonical types live in `tessera::` so non-quantum analyses can use
// them too.
using Poset          = ::tessera::Poset;
using OrderAgreement = ::tessera::OrderAgreement;

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
// A subclass that violates any of these breaks `Majorization::posetOf`
// (the transitive-reduction would fail to converge to a Hasse diagram).
//
// References:
//   {N1999}, eq. (1) and main theorem — classical case, the contract this
//     interface generalises.
class MajorizationPredicate {
public:
    virtual ~MajorizationPredicate() = default;

    [[nodiscard]] virtual bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const = 0;

    // μ strictly majorizes λ iff μ ≻ λ but not λ ≻ μ. Default
    // implementation is two `majorizes` calls; subclasses with a
    // cheaper strict path can override.
    [[nodiscard]] virtual bool
    strictlyMajorizes(std::vector<double> const& mu,
                       std::vector<double> const& lambda) const {
        return majorizes(mu, lambda) && !majorizes(lambda, mu);
    }

    // Short identifier used in diagnostics and Python repr ("standard",
    // "log-concave", "peak-radial", …). Stable across versions.
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
    double tol_;
};

// Standard majorization, restricted to spectra that are *log-concave*
// on their support: a_i² ≥ a_{i-1} · a_{i+1} ({B2015} §1). Pairs where
// either spectrum fails log-concavity are declared incomparable, so
// this is a strict sub-relation of `StandardMajorization`.
//
// References:
//   {B2015} §1, definition of log-concavity (a_i² ≥ a_{i-1} a_{i+1});
//           §4 for structural consequences.
class LogConcaveMajorization : public StandardMajorization {
public:
    explicit LogConcaveMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    // Predicate for log-concavity of a single spectrum. After sorting
    // descending and stripping trailing zeros, requires
    // s_i² ≥ s_{i-1} · s_{i+1} for every interior i. Spectra of length
    // ≤ 2 are trivially log-concave.
    [[nodiscard]] static bool
    isLogConcave(std::vector<double> const& v, double tol = 1e-12);
};

// Peak-radial dominance: μ ≻ λ iff, after sorting both descending and
// zero-padding,
//
//     λᵢ / λ₁  ≤  μᵢ / μ₁     for every i.
//
// Cross-multiplied form (used in the implementation for stability):
//     λᵢ · μ₁  ≤  μᵢ · λ₁     for every i.
//
// Strictly stronger than classical majorization.
//
// References:
//   {AN2008} Theorem 1.1 / Prop. 2.5 — closest published analog (different
//             direction); structural similarity to ratio/Lp dominance.
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

// ─── Coarse-grained façade ───────────────────────────────────────────────

// Static utility class for majorization-poset construction and pairwise
// order-agreement statistics. Stateless — not instantiable.
//
// `posetOf` builds the Hasse-cover poset on a list of spectra under a
// chosen variant of the majorization predicate. `agreement` reports
// pairwise statistics (Kendall-τ, discordant fraction, Hasse edit
// distance) between two posets on a shared label set.
class Majorization {
public:
    Majorization() = delete;
    Majorization(Majorization const&) = delete;
    Majorization& operator=(Majorization const&) = delete;

    // Build the Hasse-cover poset of the strict-majorization order under
    // an explicit predicate variant.
    //
    // `spectra[k]` becomes node k; the resulting Poset stores Hasse cover
    // edges only (transitive closure is implicit, recover with the usual
    // reachability traversal).
    //
    // Complexity: O(M³) for M = spectra.size(), dominated by the
    // transitive-reduction pass. Each predicate call is O(L log L) on
    // the spectrum lengths L.
    //
    // References:
    //   {N1999} main theorem — the partial order this poset Hasse-encodes
    //   for the `StandardMajorization` predicate.
    [[nodiscard]] static Poset posetOf(
        std::vector<std::vector<double>> const& spectra,
        MajorizationPredicate const& predicate);

    // Build the poset under the classical {N1999} majorization at the
    // given numerical tolerance.
    [[nodiscard]] static Poset posetOf(
        std::vector<std::vector<double>> const& spectra,
        double tol = 1e-12);

    // Pairwise agreement statistics between two posets on the same
    // label set of size nLabels. Delegates to ::tessera::compareOrders;
    // exists here so quantum callers don't need to reach into the
    // top-level tessera namespace.
    //
    // Complexity: O(nLabels^3) for the Floyd-Warshall transitive
    // closures, then O(nLabels^2) to count pairs.
    [[nodiscard]] static OrderAgreement agreement(
        Poset const& a, Poset const& b, int nLabels);
};

} // namespace tessera::quantum
