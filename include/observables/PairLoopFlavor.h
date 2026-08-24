// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H
#define TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H

#include <array>
#include <complex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # PairLoopFlavor
///
/// The #561/#576 pair-loop dual-basis flavor read over three emergent holes, migrated
/// as a C++ Observable (the `pair_loop_quarks.tex` §7 experiment). On a structure
/// whose periods over three emergent holes carry the color singlet `[1, ω, ω²]`:
///
///   1. ONE correlated multi-hole read — the single joint carried representative
///      `ψ = EigenstateSynthesis.carriedRepresentative(holes, σ·target)`, never
///      three independent per-hole extractions. Per-hole weight
///      `w_h = σ_h ∮_h ψ` (the five (-1)^j-signed tetrahedral facets of the
///      removed 4-cell); per-hole DK charge `q_h = Σ_{c∈∂h} W_c |ψ_c|²`.
///   2. Pair loops `γ_ij` are homologous to `[i]+[j]`: `w_i + w_j` (arithmetic,
///      no new geometry). The singlet gives the duality `[γ_ij] = -[k]`.
///   3. Criterion (a) multiplicity 2:1 via `rho`; criterion (b) odd-one-out vs
///      the recorded diquark pair (ctor provenance) — evaluated ONLY when the
///      build history supplies it, never guessed.
///
/// headline (`compute`) = `rho`; typed accessors expose the joint read and the
/// verdict. The oriented periods are reported in the propagation-root-fixed
/// convention (divided by `w0`'s unit phase) so every record leaf is GAUGE- and
/// RELABEL-invariant. Requires 3 holes on a 4-complex.
class PairLoopFlavor : public RegisterObservable {
  public:
    /// The three pair loops as (i, j) hole-index pairs; `γ_ij` encircles i, j.
    static constexpr std::array<std::pair<int, int>, 3> PAIR_LOOPS = {
        {{0, 1}, {0, 2}, {1, 2}}};
    /// Criterion (a): the closest pair's spread over its separation from the odd
    /// one must stay below this for a 2:1 (u:u:d) verdict.
    static constexpr double RHO_MAX = 0.5;
    /// Criterion-(b) status sentinels (named, never inline literals).
    static constexpr std::string_view kOddDiquarkEvaluated = "evaluated";
    static constexpr std::string_view kOddDiquarkNotEvaluable =
        "not_evaluable(no_provenance)";
    static constexpr std::string_view kRecordKey = "pair_loop_flavor";

    /// The single correlated multi-hole read (the joint read).
    struct JointRead {
      std::vector<int> sigma;                    ///< induced-orientation signs
      double rU = 0.0;                           ///< residualForPeriods of the pin
      std::vector<std::complex<double>> w;       ///< oriented per-hole weights
      std::vector<double> q;                     ///< per-hole DK charges
      std::vector<std::complex<double>> loopW;   ///< pair-loop periods (w_i+w_j)
      std::vector<double> loopQ;                 ///< pair-loop charges
      std::vector<double> dualResidual;          ///< |w_i+w_j+w_k| per loop
    };

    /// The pre-registered criteria on a finished joint read.
    struct Verdict {
      std::pair<int, int> oddLoop;               ///< the charge-odd pair loop
      int dualHole = 0;                          ///< its complementary hole
      double rho = 0.0;
      bool multiplicity21 = false;
      std::optional<bool> oddIsDiquarkLoop;      ///< empty ⇒ not evaluable
    };

    /// Read over three emergent holes with no recorded diquark pair (criterion (b)
    /// stays not-evaluable).
    PairLoopFlavor() = default;
    /// Read over three emergent holes with the step-1 hole-index pair of the diquark
    /// from the specimen's build history (makes criterion (b) decidable).
    explicit PairLoopFlavor(std::pair<int, int> diquarkPair)
        : diquarkPair_(diquarkPair) {}

    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// The derived clustering ratio `rho` divides two small charge differences
    /// and amplifies eigensolve roundoff to ~1e-13 (the source's RHO_GATE_TOL) —
    /// one tolerance covers every leaf, raw residuals reported alongside.
    [[nodiscard]] double gateTol() const override { return 1e-9; }
    [[nodiscard]] int minHoles() const override { return 3; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The joint read (typed).
    [[nodiscard]] JointRead jointRead(const RegisterContext &ctx) const;
    /// The verdict on a finished joint read.
    [[nodiscard]] Verdict evaluateCriteria(const JointRead &read) const;

    /// (odd loop index, rho): the loop whose charge sits farthest from the mean
    /// of the other two, and rho = |spread of the other two| / |that separation|.
    [[nodiscard]] static std::pair<int, double> oddOneOut(
        const std::vector<double> &loopQ);
    /// The hole index dual to the pair loop `γ_ij`: the third index (`3-i-j`).
    [[nodiscard]] static int complementHole(const std::pair<int, int> &pair) {
      return 3 - pair.first - pair.second;
    }

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;

  private:
    std::optional<std::pair<int, int>> diquarkPair_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H
