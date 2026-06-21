// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TOPOLOGYBUILDER_H
#define TESSERA_COBORDISM_TOPOLOGYBUILDER_H

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mesh/Edge.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;
using ::tessera::mesh::Edge;

/// # TopologyBuilder
///
/// Pluggable cobordism topology for `MergeCobordism` (#378). A builder
/// constructs the complex \f$ W \f$ (boundary \f$ \partial W \f$ + bulk) and
/// supplies the per-state read-out cycles, so `MergeCobordism` is agnostic to
/// the topology — our \f$ (T^2 - 3\,\text{holes}) \times S^1 \f$ operator
/// topology vs a \#353-style register rep topology, etc. The two encode
/// different objects (a qubit operator vs a color rep), so the read-out travels
/// with the topology rather than being hard-coded in the merge.
///
/// Usage is stateful: `build()` constructs \f$ W \f$ and may cache internal data
/// (holes, strides) that `readout()` then consumes — so call `build()` first.
class TopologyBuilder {
  public:
    /// A signed edge-loop: an ordered closed walk of directed `Edge`s, each
    /// edge's `getSource() -> getTarget()` giving its sign in the discrete
    /// \f$ \oint \f$ (the carried-period direction). Built over the mesh's
    /// vertices in `readout()`.
    using EdgeLoop = std::vector<Edge>;

    virtual ~TopologyBuilder() = default;

    /// Build \f$ W \f$ (boundary + bulk) for a state dimension \f$ d \f$.
    /// Populates `boundaryCells` (the \f$ \partial W \f$ top-cells) and returns
    /// the Spacetime, with the metric seeded off the degenerate uniform point.
    /// @param stateDim      the qudit dimension \f$ d \f$ (2 for a qubit).
    /// @param seed          RNG seed for the metric jitter.
    /// @param boundaryCells out: the \f$ \partial W \f$ top-cells (sorted ids).
    [[nodiscard]] virtual std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) = 0;

    /// The **edge-loop** pinned-state read-out (the SOFT path): signed edge-loops
    /// over \f$ \partial W \f$ and their target periods, given the states (inputs
    /// followed by outputs). Scored by `MergeCobordism` over `residualForLoops`,
    /// whose per-loop floor is non-zero — use this ONLY for cycles that are not
    /// triangle boundaries (the operator topology's \f$ S^1 \f$ time loop, which
    /// has no triangle-hole equivalent). A topology whose read-out cycles ARE
    /// triangle boundaries (the color register) must instead override
    /// `readoutHoles()`, the EXACT (`residualForPeriods`) path — never this one,
    /// so it cannot fall back into the soft loop residual. The default is empty
    /// (a register supplies only `readoutHoles()`).
    /// @param cobordism the \f$ W \f$ from `build()`, for the loop edges' vertices.
    virtual void readout(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<EdgeLoop> &loops,
        std::vector<std::complex<double>> &targets) const {
      (void)cobordism;
      (void)states;
      loops.clear();
      targets.clear();
    }

    /// The **triangle-hole** pinned-state read-out (the EXACT \#353 period path).
    /// When a topology's read-out cycles are triangle boundaries (the color
    /// register's hole-circles), it overrides this so `MergeCobordism` scores the
    /// pinned inputs over the EXACT `residualForPeriods` (period of a removed
    /// triangle, machine-zero on a carried target) and reads the emergent result
    /// block over `cyclePeriods` — never the soft edge-loop `residualForLoops`.
    /// Only the **inputs** are pinned (the supplied `states`); the result block's
    /// holes are returned separately in `resultHoles` to be READ after the relax,
    /// not pinned (the \#353 inputs -> emergent result flow). The default is empty
    /// (a topology whose cycles are not triangle boundaries — the operator's
    /// \f$ S^1 \f$ — supplies only `readout()` and is scored over loops).
    /// @param cobordism    the \f$ W \f$ from `build()`.
    /// @param states       the pinned states (inputs followed by outputs).
    /// @param inputHoles   out: the pinned states' triangle holes (sorted triples).
    /// @param inputTargets out: their target periods (induced-orientation signed).
    /// @param resultHoles  out: the emergent result block's triangle holes (read).
    /// @param resultSigns  out: the result block's induced-orientation signs
    ///   (\f$ \pm 1 \f$), applied to the emergent result periods so the read-out is
    ///   symmetric with the signed input targets and hence relabeling-invariant.
    ///   Empty means no signing (the read-out keeps the raw per-hole periods).
    virtual void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles,
        std::vector<int> &resultSigns) const {
      (void)cobordism;
      (void)states;
      inputHoles.clear();
      inputTargets.clear();
      resultHoles.clear();
      resultSigns.clear();
    }

    /// Whether the result block EMERGES from the pinned inputs alone (the \#353
    /// register: pin the neutral-pair inputs, read the emergent result block),
    /// rather than being pinned/supplied. When true, `MergeCobordism` does not
    /// require `outputStates`/`U` (the result is not a caller input but an
    /// emergent read-out via `readoutHoles`'s result block). The default is false
    /// (the operator topology pins inputs AND outputs, or derives outputs from U).
    [[nodiscard]] virtual bool emergesResult() const { return false; }

    /// The carried-object dimension this topology realizes:
    /// \f$ \dim \ker L_1(W - \partial W) \f$ (e.g. \f$ d^2 - 1 \f$ for the
    /// operator topology, \f$ b_1 \f$ on the \f$ \sum=0 \f$ hyperplane for a
    /// register). Used to size and validate the read-out.
    [[nodiscard]] virtual std::size_t carriedDim(std::size_t stateDim) const = 0;

    /// The number of read-out cycles `readout()` emits per pinned state (e.g. 2
    /// for the torus: the hole-circle and the \f$ S^1 \f$ time loop; 3 for the
    /// color register: its three hole-circles). With it the caller can split
    /// `readout()`'s flat loop list back into per-state blocks and detect a state
    /// that went unpinned — `readout()` pins at most the topology's state
    /// capacity, so `loops.size() == loopsPerState() * (#states)` holds iff every
    /// state was pinned.
    [[nodiscard]] virtual std::size_t loopsPerState() const = 0;

    /// Validate the state amplitude dimension \f$ d \f$ for this topology, so the
    /// admissible dimension travels with the topology rather than being baked
    /// into `MergeCobordism`. The default requires a power of two \f$ \geq 2 \f$
    /// (the qubit/Choi operator topology); a register overrides it (\f$ d = 3 \f$,
    /// the color triple on the \f$ \sum = 0 \f$ hyperplane).
    /// @throws std::invalid_argument if \f$ d \f$ is not admissible.
    virtual void validateStateDim(std::size_t d) const {
      if (d < 2 || (d & (d - 1)) != 0)
        throw std::invalid_argument(
            "MergeCobordism: state dimension must be a power of two >= 2");
    }

    /// Human-readable topology name, for `MergeCobordism::Stats::topology`.
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TOPOLOGYBUILDER_H
