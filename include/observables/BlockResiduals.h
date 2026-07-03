// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H
#define TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # BlockResiduals
///
/// The #574 per-output-block carry residuals, migrated as a C++ Observable. Each
/// provenance block (a vertex region + a target, e.g. `ProtonIngredients`'s
/// output blocks or a campaign record) is scored against its OWN sub-complex
/// exactly as `ProtonIngredients::outputBlockResidual` scores it: the ambient top
/// cells whose vertices ALL lie in the region form the block's own sub-complex
/// (uniform-metric — matching how the drive's `r_U` scored the block), scored
/// with `MultiCobordism::residualOfTargetStateAgainstHarmonic`; an empty region
/// reports the full leak `‖target‖²`.
///
/// The blocks are ctor provenance (build history travels via the campaign record
/// / geometry-dump metadata — never guessed). The sub-complex is LOADED by the
/// `LiveComplex` loader (a strict selection of existing cells re-instantiated
/// through the canonical `fromCells`), never built inside this reader. Because
/// block regions carry vertex ids, `recordRelabeled` maps them through the
/// RELABEL permutation so the gate compares like with like.
class BlockResiduals : public RegisterObservable {
  public:
    /// One provenance block: a label, its emergent vertex region, and its
    /// register target.
    struct Block {
      std::string label;
      std::vector<std::uint64_t> vertices;
      std::vector<std::complex<double>> target;
    };

    explicit BlockResiduals(std::vector<Block> blocks)
        : blocks_(std::move(blocks)) {}

    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] bool needsProvenance() const override { return true; }
    [[nodiscard]] bool hasProvenance() const override {
      return !blocks_.empty();
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;
    [[nodiscard]] Record recordRelabeled(
        const RegisterContext &ctx,
        const std::map<std::uint64_t, std::uint64_t> &perm) const override;

    static constexpr std::string_view kRecordKey = "block_residuals";

  protected:
    /// The headline is the total carry leak — the sum of the block residuals.
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;

  private:
    /// The record over an explicit block list (shared by `record` and
    /// `recordRelabeled`).
    [[nodiscard]] Record recordForBlocks(
        const RegisterContext &ctx,
        const std::vector<Block> &blocks) const;
    /// One block's carry residual (the full leak when the region has no cell).
    [[nodiscard]] static double blockResidual(const RegisterContext &ctx,
                                              const Block &block,
                                              int &nCellsInRegion,
                                              double &targetNorm2);

    std::vector<Block> blocks_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H
