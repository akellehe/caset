// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EMERGENT_RADIUS_H
#define TESSERA_OBSERVABLES_EMERGENT_RADIUS_H

#include <string>

#include "observables/InteriorHinges.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # EmergentRadius
///
/// The radius half of the #575 mass/radius battery on the relaxed 4D interior,
/// migrated as a C++ Observable. Composes the same shared `InteriorHinges` core
/// (via `RegisterContext::interiorHinges`) that `EmergentMass` reads.
///
///   * headline (`compute`) = `r_dual = V_dual^{1/4}`, the dimension-correct
///     dual-volume radius on a 4-complex;
///   * typed accessor `radii()`: `V_dual` / `V_primal`, the primal-cross-check
///     `r_primal`, and the strictly-interior-vertex count;
///   * `record()` = the radius block (dual + primal) and the hole count.
class EmergentRadius : public RegisterObservable {
  public:
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// Volume aggregates re-summed in a relabeled container order carry
    /// order-ULP noise scaling with the magnitudes (see `EmergentMass`).
    [[nodiscard]] double gateTol() const override { return 1e-6; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The dual/primal size of the interior + interior-vertex count.
    [[nodiscard]] InteriorHinges::Radii radii(const RegisterContext &ctx) const;

    static constexpr std::string_view kRecordKey = "emergent_radius";

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EMERGENT_RADIUS_H
