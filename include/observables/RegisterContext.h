// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_REGISTER_CONTEXT_H
#define TESSERA_OBSERVABLES_REGISTER_CONTEXT_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/Proton.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class InteriorHinges;  // the shared 4D hinge-selection core (InteriorHinges.h)

/// # RegisterContext
///
/// The one **validated read context** every emergent-proton observable measures
/// (#593, part of #559): a converged spacetime, its emergent holes, the
/// induced-orientation signs, and the shared per-complex caches — composed from the
/// protected readout cores (`cobordism::EigenstateSynthesis`,
/// `cobordism::MultiCobordism`, `cobordism::ChainComplex`), never refactoring them.
///
///   * **A pure reader — never a builder.** The context READS an already-built,
///     relaxed spacetime (a `Proton::block()`, a `ProtonIngredients` state, a
///     relaxed `MultiCobordism` complex, or a dump the loader already rehydrated
///     into a live complex). It never builds, solves, or materializes anything —
///     the emergent build lives exclusively in Proton/ProtonIngredients/
///     MultiCobordism and is never re-run here. The facet/coface skeleton the
///     `dualVolume()` / `deficitAngle()` reads walk is expected to be
///     already present on the live complex (it is, on every built state); the
///     construction that completes a bare `Spacetime::fromCells` skeleton — the
///     dump-rehydration and the RELABEL-gate rebuild — lives OUTSIDE this class,
///     in `LiveComplex` (the loader/transform), and only ever reads the recorded
///     geometry back, never the emergent dynamics.
///   * **Hole selection validated at ONE entry point.** The selected holes are the
///     emergent `(degree+2)`-vertex removed top cells
///     (`cobordism::MultiCobordism::emergentHoles`), in emergent-hole order. A
///     **deficit throws** (`std::invalid_argument` naming the found holes); a
///     **surplus is an explicit, recorded truncation** naming the dropped holes in
///     `selectionWarning()` (the Python binding emits it as a `UserWarning`), never
///     a silent slice. Both censuses are kept (`holesUsed()` / `holesTotal()`)
///     alongside Betti at the register degree (`bK()`) and the
///     `holesVsBettiDivergent()` flag — the campaign taught us holes and \f$ b_k \f$
///     can disagree (e.g. holes=3 / \f$ b_3 \f$=2), so the divergence is always
///     recorded, never papered over.
///   * **One orientation convention.** The induced-orientation signs
///     \f$ \varepsilon_h = \pm 1 \f$ come from
///     `cobordism::ChainComplex::endSignCovector` — the label-free orientation
///     under which every closed form's signed periods obey
///     \f$ \sum_h \varepsilon_h p_h = 0 \f$, determined up to one global sign (the
///     propagation root, \f$ -1 \in U(1) \f$ on the register — gauge, not physics).
///     Vertices are never sorted to impose a convention; holes are matched by
///     vertex SET.
///   * **One cached `EigenstateSynthesis`** per (spacetime, degree), plus the other
///     shared per-complex structures (Hodge metric weights, the canonical
///     \f$ k \f$-cell index, the Betti vector, the 4D interior-hinge selection).
///     `gauged()` copies share the caches — the gauge knob only rotates the target.
///
/// The GAUGE gate transform acts on the context, not on the observables:
/// `gauged(theta)` rotates the register target by the surviving global U(1) phase
/// (which contains the Z₃ cyclic recolor of the singlet and the orientation flip)
/// — a construction-free operation that shares the same live complex. The RELABEL
/// gate needs a rebuilt (relabeled) complex, which is a construction: that lives
/// in `LiveComplex` and is orchestrated by `ObservableGates`, never here — this
/// reader never rebuilds anything.
class RegisterContext {
  public:
    /// Read the context over the already-built `spacetime`, selecting and
    /// validating `count`
    /// emergent holes at `degree` (see the class note: deficit throws, surplus
    /// is recorded in `selectionWarning()` naming the dropped holes). `target`
    /// is the register target state, one component per hole slot — the color
    /// singlet \f$ [1, \omega, \omega^2] \f$ by default; the GAUGE gate rotates
    /// exactly this.
    /// @throws std::invalid_argument on a hole deficit (fewer than `count`
    ///   emergent holes), an empty complex, or `count < 0` / `degree < 0`.
    explicit RegisterContext(
        std::shared_ptr<Spacetime> spacetime, int count = 3, int degree = 3,
        std::vector<std::complex<double>> target = cobordism::Proton::singlet());

    /// Build the context with an EXPLICIT hole selection (e.g. a build's own
    /// census, or the relabel gate's matched images), validated with the same
    /// count semantics as the selecting constructor: fewer than `count` throws,
    /// more than `count` is a recorded truncation naming the dropped holes.
    /// `holesTotal()` still reports the complex's own emergent census.
    RegisterContext(std::shared_ptr<Spacetime> spacetime,
                    const std::vector<std::vector<std::uint64_t>> &holes,
                    int count, int degree,
                    std::vector<std::complex<double>> target);

    // ---- the validated register ----
    [[nodiscard]] const std::shared_ptr<Spacetime> &spacetime() const noexcept {
      return spacetime_;
    }
    /// The register degree \f$ k \f$ (holes are `(k+2)`-vertex removed top
    /// cells; the cached synthesis reads at this \f$ k \f$).
    [[nodiscard]] int degree() const noexcept { return degree_; }
    /// The register target state (rotated by `gauged()`).
    [[nodiscard]] const std::vector<std::complex<double>> &target() const noexcept {
      return target_;
    }
    /// The selected holes, in emergent-hole order (each a vertex-id
    /// tuple of a removed top cell, intrinsic order preserved).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &holes()
        const noexcept {
      return holes_;
    }
    /// The surplus holes dropped by the validated selection (empty when none).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &droppedHoles()
        const noexcept {
      return droppedHoles_;
    }
    /// Holes the register reads (`holes().size()`).
    [[nodiscard]] int holesUsed() const noexcept {
      return static_cast<int>(holes_.size());
    }
    /// The complex's full emergent-hole census at the register degree.
    [[nodiscard]] int holesTotal() const noexcept { return holesTotal_; }
    /// Betti at the register degree (GF(2) ranks; 0 when the Betti vector is
    /// shorter than the degree).
    [[nodiscard]] int bK() const;
    /// The full Betti vector of the complex (cached).
    [[nodiscard]] const std::vector<int> &betti() const;
    /// True when the emergent-hole census and Betti at the register degree
    /// disagree — the campaign's holes=3 / b₃=2 style finding, always recorded.
    [[nodiscard]] bool holesVsBettiDivergent() const { return holesTotal_ != bK(); }
    /// The surplus-selection warning naming the dropped holes (empty when the
    /// selection was exact). The Python binding emits it as a `UserWarning` at
    /// construction; C++ callers consult it here.
    [[nodiscard]] const std::string &selectionWarning() const noexcept {
      return selectionWarning_;
    }
    /// The canonical spacetime dimension \f$ d \f$ — the metric signature's
    /// dimension (`getMetric()->getSignature()->getDimensions()`, the same
    /// accessor WilsonLoop / ReggeSolver read). The driver's dimension gates
    /// read this.
    [[nodiscard]] int dimensions() const noexcept { return dimensions_; }
    /// The number of top cells.
    [[nodiscard]] int topCellCount() const noexcept { return topCellCount_; }
    /// True iff any edge is non-spacelike (timelike or null). At
    /// initialization no time has passed — causal structure may only emerge —
    /// so all-spacelike specimens honestly report false.
    [[nodiscard]] bool causalContent() const noexcept { return causalContent_; }

    // ---- shared per-complex caches ----
    /// The one cached `cobordism::EigenstateSynthesis(spacetime, degree)` every
    /// period readout shares (lazily built; `gauged()` copies share it).
    [[nodiscard]] cobordism::EigenstateSynthesis &synthesis() const;
    /// The induced-orientation signs \f$ \varepsilon_h = \pm 1 \f$ of the
    /// selected holes (`cobordism::ChainComplex::endSignCovector` — the ONE
    /// orientation convention; lazily built, shared across `gauged()` copies).
    [[nodiscard]] const std::vector<int> &epsilonSigns() const;
    /// The Hodge metric weights \f$ W_k \f$ at the register degree, in the
    /// canonical `ChainComplex` \f$ k \f$-cell order (lazily built, shared).
    [[nodiscard]] const std::vector<double> &hodgeWeights() const;
    /// The canonical \f$ k \f$-cell index: sorted vertex-id tuple → operator
    /// index into `synthesis().cellSimplices()` (lazily built, shared).
    [[nodiscard]] const std::map<std::vector<std::uint64_t>, std::size_t> &
    cellIndex() const;
    /// The shared 4D interior-hinge selection (`InteriorHinges` over this
    /// context's spacetime and holes; lazily built, shared — `EmergentMass` and
    /// `EmergentRadius` compose exactly this one instance).
    /// @throws std::invalid_argument if the complex is not genuinely 4D.
    [[nodiscard]] const std::shared_ptr<InteriorHinges> &interiorHinges() const;

    // ---- the GAUGE gate transform (construction-free) ----
    /// The GAUGE-gate variant: the same live complex and register with the
    /// target rotated by the global U(1) phase \f$ e^{i\theta} \f$ — the
    /// register's one surviving gauge freedom (it contains the Z₃ cyclic recolor
    /// of the singlet and the orientation flip). Shares this context's spacetime
    /// and caches (the gauge knob only rotates the target; nothing is rebuilt).
    /// The RELABEL gate — which needs a rebuilt, relabeled complex — is a
    /// construction and lives in `LiveComplex` / `ObservableGates`, never here.
    [[nodiscard]] std::shared_ptr<RegisterContext> gauged(double theta) const;

  private:
    /// The lazily-built shared structures. Held behind one shared_ptr so
    /// `gauged()` copies share every cache both ways (a cache built by either
    /// context is visible to the other — the complex is the same object).
    struct Caches;

    /// Shared constructor tail: read the live complex's censuses and validate
    /// the hole selection (`explicitHoles` null ⇒ select from the emergent
    /// census). Reads only — nothing is built or materialized.
    void initialize(int count,
                    const std::vector<std::vector<std::uint64_t>> *explicitHoles);

    std::shared_ptr<Spacetime> spacetime_;
    int degree_;
    std::vector<std::complex<double>> target_;
    std::vector<std::vector<std::uint64_t>> holes_;
    std::vector<std::vector<std::uint64_t>> droppedHoles_;
    int holesTotal_ = 0;
    int dimensions_ = -1;
    int topCellCount_ = 0;
    bool causalContent_ = false;
    std::string selectionWarning_;
    std::shared_ptr<Caches> caches_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_REGISTER_CONTEXT_H
