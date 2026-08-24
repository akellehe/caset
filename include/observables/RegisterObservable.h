// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_REGISTER_OBSERVABLE_H
#define TESSERA_OBSERVABLES_REGISTER_OBSERVABLE_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "observables/Observable.h"
#include "observables/Record.h"
#include "observables/RegisterContext.h"

namespace tessera::observables {

/// # RegisterObservable
///
/// The base for the emergent-proton readouts (#593): a pure post-hoc reader over
/// a `RegisterContext` (a live, already-built complex). It extends the existing
/// `tessera::observables::Observable` — every subclass is-an `Observable` — and
/// adds the register-aware surface the SpacetimeVolume/VolumeProfile house
/// pattern lacks:
///
///   * `record(ctx)` — the full JSON-able `Record` (bound to Python as a dict),
///     containing only GAUGE- and RELABEL-invariant channels;
///   * `compute(ctx)` — the headline scalar (also reachable through the base
///     `Observable::compute(spacetime)`, which reads a default register off the
///     spacetime);
///   * the declarative skip surface (`minHoles`/`requiredDimensions`/
///     `needsProvenance`/`needsCausalContent`) so an inapplicable observable is a
///     reported skip, never a crash;
///   * `recordRelabeled(ctx, perm)` — the RELABEL-gate hook, defaulting to
///     `record(ctx)`; observables whose configuration carries vertex ids (e.g.
///     `BlockResiduals` block regions) override it to map their provenance
///     through the permutation so the gate compares like with like.
///
/// An Observable NEVER shapes the lattice, never builds, never solves — it reads
/// and reports. The GAUGE/RELABEL gates (`ObservableGates`) are post-hoc
/// validation, never a loop condition.
class RegisterObservable : public Observable {
  public:
    // ---- skip-reason sentinels (named, never inline literals) ----
    /// The observable needs provenance it was not given.
    static constexpr std::string_view kSkipNoProvenance = "no_provenance";
    /// The observable reads causal structure the all-spacelike specimen lacks.
    static constexpr std::string_view kSkipNoCausalContent = "no_causal_content";
    /// The prefix of the hole-deficit skip reason (`holes=N < min_holes=M`).
    static constexpr std::string_view kSkipHolesPrefix = "holes=";
    static constexpr std::string_view kSkipHolesInfix = " < min_holes=";
    /// The prefix/infix of the dimension-mismatch skip reason.
    static constexpr std::string_view kSkipDimensionsPrefix = "dimensions=";
    static constexpr std::string_view kSkipDimensionsInfix = " != ";

    /// The battery record key (unique within a battery).
    [[nodiscard]] virtual std::string recordKey() const = 0;

    /// The GAUGE/RELABEL residual tolerance for the `*_ok` verdicts. Direct
    /// period/charge reads sit at ~1e-16; derived ratios amplify eigensolve
    /// roundoff; geometric aggregates re-summed in a relabeled container order
    /// carry order-ULP noise — subclasses pick the tolerance their channels
    /// warrant and the raw residuals are always reported alongside.
    [[nodiscard]] virtual double gateTol() const { return 1e-9; }

    /// The full JSON-able record of invariant channels (the pure read).
    [[nodiscard]] virtual Record record(const RegisterContext &ctx) const = 0;

    /// The headline scalar for this observable (the base-class contract).
    [[nodiscard]] double compute(const RegisterContext &ctx) const {
      return computeHeadline(ctx);
    }

    /// The RELABEL-gate hook: re-measure on the relabeled context, mapping any
    /// vertex-id-bearing configuration through `perm`. Default is identity —
    /// correct for observables whose configuration carries no vertex ids.
    [[nodiscard]] virtual Record recordRelabeled(
        const RegisterContext &ctx,
        const std::map<std::uint64_t, std::uint64_t> & /*perm*/) const {
      return record(ctx);
    }

    // ---- the declarative skip surface ----
    /// Emergent holes this observable's readout needs (0 = none).
    [[nodiscard]] virtual int minHoles() const { return 0; }
    /// The top-cell dimension this observable requires (-1 = any).
    [[nodiscard]] virtual int requiredDimensions() const { return -1; }
    /// True iff this observable needs provenance it was not given.
    [[nodiscard]] virtual bool needsProvenance() const { return false; }
    /// True iff the required provenance is present (default: nothing needed).
    [[nodiscard]] virtual bool hasProvenance() const { return true; }
    /// True iff this observable reads causal structure.
    [[nodiscard]] virtual bool needsCausalContent() const { return false; }

    /// The reason this observable cannot measure `ctx` (a string the battery
    /// reports), or empty when it can.
    [[nodiscard]] std::string skipReason(const RegisterContext &ctx) const;

    // ---- the base Observable contract ----
    /// Read a default register (`count=3`, `degree=3`, singlet target) off the
    /// live spacetime and return the headline scalar.
    /// @throws std::invalid_argument if the spacetime does not supply three
    ///   emergent holes (use `compute(ctx)` with an explicit register for
    ///   specimens with fewer).
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
    double update(const std::shared_ptr<Spacetime> &spacetime) override;

    ~RegisterObservable() override = default;

  protected:
    /// The headline scalar implementation.
    [[nodiscard]] virtual double computeHeadline(
        const RegisterContext &ctx) const = 0;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_REGISTER_OBSERVABLE_H
