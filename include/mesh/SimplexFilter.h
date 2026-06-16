// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Predicate interface for selecting which top simplices participate in a
// downstream observable (e.g. ``Spacetime::getSpectralDimensionOnSkeleton``,
// ``Spacetime::skeletonGraph``).
//
// The interface intentionally has only ``accept(simplex)`` — no side
// effects, no shared state. Filters are passed by const reference so
// callers can compose them inline without ownership concerns. Stateful or
// configurable filters are free to add members; the contract is just the
// boolean predicate plus a printable name for JSON / logging.

#pragma once

#include "mesh/ForwardDeclarations.h"

#include <string>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Boolean predicate over top simplices. ``accept(s)`` returning false
/// means: do not include ``s`` in the downstream observable.
///
/// Default for holographic-dual measurements is :class:`AllSimplexFilter`,
/// which lets every registered top simplex through. The codebase reads MI
/// as a quantum-entanglement correlator that need not respect a positive-
/// volume metric (spacelike-separated subsystems can have non-zero MI), so
/// the default does not filter by metric validity. Use
/// :class:`PositiveGramDeterminantFilter` when you want to restrict the
/// measurement to metrically valid Euclidean cells.
class SimplexFilter {
public:
    virtual ~SimplexFilter() = default;

    /// Returns true if ``simplex`` should participate in the downstream
    /// observable.
    [[nodiscard]] virtual bool accept(SimplexPtr const& simplex) const = 0;

    /// Human-readable name. Emitted in JSON output so experiment runs are
    /// reproducible; printed by Python ``repr``.
    [[nodiscard]] virtual std::string name() const = 0;
};

/// Accepts every simplex. The default filter for the holographic dual
/// measurement.
///
/// Registration via :func:`Spacetime::createSimplex` already implies the
/// simplex is combinatorially constructable (its ``k + 1`` vertices form a
/// complete subgraph in the edge set — "constructable by coning"), so
/// this filter intentionally ignores edge-length geometry.
class AllSimplexFilter : public SimplexFilter {
public:
    [[nodiscard]] bool accept(SimplexPtr const&) const override {
        return true;
    }
    [[nodiscard]] std::string name() const override {
        return "AllSimplexFilter";
    }
};

/// Accepts simplices whose Gram matrix (from edge lengths via
/// :func:`Simplex::gramMatrix`) has positive determinant — i.e. metrically
/// valid (non-degenerate, non-collapsed) Euclidean cells.
///
/// Implementation reuses :func:`Simplex::determinant` on the flat
/// row-major Gram matrix. Simplices with fewer than 2 vertices, or with
/// any non-finite edge lengths, are rejected.
class PositiveGramDeterminantFilter : public SimplexFilter {
public:
    [[nodiscard]] bool accept(SimplexPtr const& simplex) const override;
    [[nodiscard]] std::string name() const override {
        return "PositiveGramDeterminantFilter";
    }
};

} // namespace tessera::mesh
