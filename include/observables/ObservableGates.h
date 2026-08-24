// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_OBSERVABLE_GATES_H
#define TESSERA_OBSERVABLES_OBSERVABLE_GATES_H

#include <cstdint>
#include <string>

#include "observables/Record.h"
#include "observables/RegisterContext.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # ObservableGates
///
/// The GAUGE and RELABEL gate harness (#593) — post-hoc validation, never a loop
/// condition. Each gate re-measures an observable on a transformed context and
/// reports the max-abs delta over every numeric leaf of its record
/// (`Record::reportDelta`); a record channel that is not gauge- and
/// relabel-invariant is a flagged leak.
///
///   * GAUGE — re-measure on `ctx.gauged(GAUGE_THETA)` (construction-free: the
///     same live complex, the register target rotated by the surviving global
///     U(1) phase).
///   * RELABEL — re-measure on a relabeled rebuild. The rebuild is a
///     construction and so lives in the loader (`LiveComplex::relabel`), never in
///     a reader: this harness loads the relabeled live complex, wraps it in a
///     (pure-reader) `RegisterContext` with the register's images matched by
///     permuted vertex SET, and maps any vertex-id-bearing provenance through the
///     permutation (`RegisterObservable::recordRelabeled`).
///
/// The self-test (`selfTest`) proves the harness actually compares: a
/// deliberately label-dependent probe is flagged by RELABEL and a deliberately
/// gauge-dependent probe is flagged by GAUGE — a silently-passing gate cannot be
/// a comparison that never happened.
class ObservableGates {
  public:
    /// GAUGE-gate angle: an incommensurate fraction of 2π, so the rotated target
    /// never lands on a symmetry of the singlet by accident.
    static constexpr double GAUGE_THETA =
        2.0 * 3.14159265358979323846 * 0.371;
    /// RELABEL-gate permutation seed.
    static constexpr std::uint64_t GATE_SEED = 3;

    /// One observable's gate verdicts.
    struct GateResult {
      double gaugeDelta = 0.0;
      double relabelDelta = 0.0;
      double gateTol = 0.0;
      bool gaugeOk = false;
      bool relabelOk = false;
    };

    /// The GAUGE residual: `reportDelta(record(ctx), record(ctx.gauged(θ)))`.
    [[nodiscard]] static double gaugeDelta(const RegisterObservable &observable,
                                           const RegisterContext &ctx);
    /// The RELABEL residual: `reportDelta(record(ctx),
    /// recordRelabeled(relabeled ctx, perm))`.
    [[nodiscard]] static double relabelDelta(
        const RegisterObservable &observable, const RegisterContext &ctx);
    /// Both gates + the `*_ok` verdicts against the observable's `gateTol`.
    [[nodiscard]] static GateResult evaluate(
        const RegisterObservable &observable, const RegisterContext &ctx);

    /// The harness self-test: the label-dependent probe must be RELABEL-flagged
    /// and the gauge-dependent probe GAUGE-flagged (both deltas > 0). Returns
    /// true iff both are flagged.
    [[nodiscard]] static bool selfTest(const RegisterContext &ctx);
};

/// A deliberately label-dependent probe: its record leaks the sum of the
/// selected holes' vertex ids, so the RELABEL gate MUST flag it (the harness
/// self-test).
class LabelLeakProbe : public RegisterObservable {
  public:
    static constexpr std::string_view kRecordKey = "label_leak_probe";
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

/// A deliberately gauge-dependent probe: its record leaks the raw register
/// target's first-component phase, so the GAUGE gate MUST flag it.
class GaugeLeakProbe : public RegisterObservable {
  public:
    static constexpr std::string_view kRecordKey = "gauge_leak_probe";
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_OBSERVABLE_GATES_H
