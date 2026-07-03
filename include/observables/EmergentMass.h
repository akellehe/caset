// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EMERGENT_MASS_H
#define TESSERA_OBSERVABLES_EMERGENT_MASS_H

#include <string>

#include "observables/InteriorHinges.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # EmergentMass
///
/// The mass half of the #575 mass/radius battery on the relaxed 4D interior,
/// migrated as a C++ Observable. Composes the shared `InteriorHinges` core (via
/// `RegisterContext::interiorHinges`), so `EmergentMass` and `EmergentRadius`
/// read exactly one hinge selection.
///
///   * headline (`compute`) = `m_shell`, the #352/#451 intensive shell mass;
///   * typed accessors: `masses()` (m_shell/m_sum/m_action, the per-shell means,
///     and the |Im ε| boost accounting), `localization()`;
///   * `record()` = the interior census, the three masses, the localization, and
///     the r·m table with its definitional spread stated first (#451: r·m is too
///     definition-sensitive to quote as one number).
class EmergentMass : public RegisterObservable {
  public:
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// Geometric aggregates (sums over hundreds of hinges, r·m products) are
    /// re-summed in a different container order on the relabeled rebuild —
    /// order-ULP noise scales with the magnitudes, so this gate is
    /// absolute-loose while the raw residuals stay reported.
    [[nodiscard]] double gateTol() const override { return 1e-6; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The three mass readings + shell means + Im accounting.
    [[nodiscard]] InteriorHinges::Masses masses(
        const RegisterContext &ctx) const;
    /// The curvature localization (participation ratio, shell profile).
    [[nodiscard]] InteriorHinges::Localization localization(
        const RegisterContext &ctx) const;

    static constexpr std::string_view kRecordKey = "emergent_mass";

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EMERGENT_MASS_H
