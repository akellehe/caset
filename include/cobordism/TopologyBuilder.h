// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TOPOLOGYBUILDER_H
#define TESSERA_COBORDISM_TOPOLOGYBUILDER_H

#include <complex>
#include <cstdint>
#include <memory>
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

    /// The pinned-state read-out: the signed edge-loops over \f$ \partial W \f$
    /// and their target periods, given the states (inputs followed by outputs).
    /// Defines what the relaxation pins and what the operator/rep read-out reads.
    /// The loops' `Edge`s are built over `cobordism`'s vertices, so this needs
    /// the `cobordism` returned by a prior `build()`.
    /// @param cobordism the \f$ W \f$ from `build()`, for the loop edges' vertices.
    virtual void readout(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<EdgeLoop> &loops,
        std::vector<std::complex<double>> &targets) const = 0;

    /// The carried-object dimension this topology realizes:
    /// \f$ \dim \ker L_1(W - \partial W) \f$ (e.g. \f$ d^2 - 1 \f$ for the
    /// operator topology, \f$ b_1 \f$ on the \f$ \sum=0 \f$ hyperplane for a
    /// register). Used to size and validate the read-out.
    [[nodiscard]] virtual std::size_t carriedDim(std::size_t stateDim) const = 0;

    /// Human-readable topology name, for `MergeCobordism::Stats::topology`.
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TOPOLOGYBUILDER_H
