// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H
#define TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "observables/Record.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Stable, label-free identity of a discovered component.
///
/// The hash is derived from the oriented incidence structure of the
/// component's children and its parent lineage (the child hashes feed the
/// parent hash), never from raw vertex/cell numbers.  It is used for
/// persistence matching and deterministic tie-breaking, never as a physical
/// observable.  Two structurally identical (automorphic) components share a
/// hash by construction; bookkeeping that must tell such twins apart (for
/// example cache invalidation) is positional, see
/// :class:`InvalidationRead`.
///
/// ``level`` is the multilevel-aggregation depth at which the component was
/// formed: level 0 is an input cell, level ``k`` a community formed at the
/// ``k``-th aggregation round of the discovery run.
class ComponentId {
public:
  ComponentId() = default;
  ComponentId(std::string hash, std::size_t level)
      : hash_(std::move(hash)), level_(level) {}

  /// The canonical structural hash (32 lowercase hex characters).
  std::string canonicalHash() const { return hash_; }
  /// Multilevel-aggregation depth at which this component was formed.
  std::size_t level() const { return level_; }

  bool operator==(const ComponentId &o) const noexcept {
    return level_ == o.level_ && hash_ == o.hash_;
  }
  bool operator!=(const ComponentId &o) const noexcept {
    return !(*this == o);
  }
  /// Deterministic label-free ordering: (level, hash).
  bool operator<(const ComponentId &o) const noexcept {
    if (level_ != o.level_) return level_ < o.level_;
    return hash_ < o.hash_;
  }

private:
  std::string hash_;
  std::size_t level_ = 0;
};

/// Configuration for the label-free multiscale component discovery.
struct PersistentModularityConfig {
  /// Resolution parameters gamma for the scan, in scan order.  Adjacent
  /// entries are matched into persistence tracks.
  std::vector<double> resolutions{1.0};
  /// Base of the fixed restart seed sequence: restart ``t`` uses seed
  /// ``splitmix64(baseSeed + t)``.  Deterministic by construction.
  std::uint64_t baseSeed = 0;
  /// Number of deterministic multilevel restarts per resolution.  The best
  /// exact score is retained; the spread across restarts is reported
  /// honestly (no claim of the NP-hard global optimum is ever made).
  int restarts = 4;
  /// Hard cap on local-move sweeps per aggregation level.
  int maxSweepsPerLevel = 64;
  /// Minimum support overlap (Jaccard) for a persistence track to continue
  /// across adjacent resolutions.
  double overlapThreshold = 0.5;
};

/// One discovered component: canonical id, level-0 cell support, cached
/// sufficient statistics, and the exact per-component scores derived from
/// them.
struct ComponentRead {
  ComponentId id;
  /// Level-0 member cell ids (a set; listed ascending for reporting only —
  /// the ordering carries no convention and the identity never derives from
  /// these numbers).
  std::vector<std::uint64_t> support;
  /// Sigma_in: total internal adjacency weight A(C,C) counting both ordered
  /// directions (self-loop convention of aggregated levels included).
  double internalWeight = 0.0;
  /// S_C: summed weighted degree (strength) of the members.
  double strength = 0.0;
  /// Weighted conductance cut(C) / min(vol C, vol V\C); 0 by convention when
  /// the denominator vanishes (whole-graph or empty community).
  double conductance = 0.0;
  /// This community's exact additive term of Q_gamma:
  /// Sigma_in/(2m) - gamma (S_C/(2m))^2.
  double modularityContribution = 0.0;
};

/// One deterministic restart: its seed and exact best score.
struct RestartRead {
  std::uint64_t seed = 0;
  /// Exact Q_gamma of this restart's final partition (cold recompute).
  double q = 0.0;
  std::size_t communities = 0;
};

/// The discovery result at one resolution gamma.
struct ResolutionSlice {
  double gamma = 1.0;
  /// Exact Q_gamma of the winning partition, recomputed cold from the final
  /// labels.  The best score across the deterministic restarts — a heuristic
  /// proposal, not the NP-hard global optimum.
  double q = 0.0;
  /// The winning restart's incrementally accumulated score
  /// (Q_0 + sum of accepted exact delta-Q, compensated summation).  Must
  /// agree with ``q`` to double round-off; tested against it.
  double qIncremental = 0.0;
  /// Number of aggregation levels in the winning run's hierarchy.
  std::size_t levels = 0;
  /// Final-level components of the winning partition, ordered by canonical
  /// hash.
  std::vector<ComponentRead> components;
  /// The full multilevel hierarchy of the winning run: ``hierarchy[k]`` are
  /// the communities formed at aggregation level ``k + 1``, each ordered by
  /// canonical hash.  ``hierarchy.back() == components``.
  std::vector<std::vector<ComponentRead>> hierarchy;
  /// Every restart's exact score, in seed-sequence order.
  std::vector<RestartRead> restarts;
  /// max - min of the restart scores: the honestly reported restart
  /// uncertainty of the heuristic search.
  double restartSpread = 0.0;
};

/// A matched component pair across adjacent resolutions or across cobordism
/// time (two reports over a common cell-id universe).
struct ComponentMatch {
  ComponentId from;
  ComponentId to;
  /// Indices of the matched components in their source containers (the
  /// positional disambiguation for automorphic twins that share a hash).
  std::size_t fromIndex = 0;
  std::size_t toIndex = 0;
  /// Jaccard overlap of the level-0 cell supports.
  double supportOverlap = 0.0;
  /// Spectral-projector overlap — the documented interface hook.  Populated
  /// only when a projector-overlap hook has been installed via
  /// :func:`PersistentModularity::setProjectorOverlapHook`; a later ticket
  /// supplies the projectors.  Absent (nullopt) means unknown, never zero.
  std::optional<double> projectorOverlap;
};

/// A component track across the resolution scan: the same emergent support
/// followed through consecutive slices by maximum support overlap.
struct PersistenceTrack {
  /// One member per covered slice, consecutive from ``firstSlice``.
  std::vector<ComponentId> members;
  /// Positional index of each member within its slice's final components.
  std::vector<std::size_t> memberIndices;
  std::size_t firstSlice = 0;
  std::size_t lastSlice = 0;
  double gammaFirst = 0.0;
  double gammaLast = 0.0;
  /// Smallest adjacent-slice support overlap along the track (1.0 for a
  /// single-slice track).
  double minAdjacentOverlap = 1.0;
  /// Mean weighted conductance of the members.
  double meanConductance = 0.0;
  /// Downstream weight-aware gap/localization/persistence status.  Populated
  /// by the later weight-aware certificate tickets; Null means unknown —
  /// unknown is never encoded as zero.  Lifetime/overlap here are proposal
  /// diagnostics only and neither accept nor veto a fiber.
  Record weightAwareStatus;
};

/// The full resolution-scan report.
struct ScanReport {
  std::vector<ResolutionSlice> slices;
  /// Adjacent-slice best matches (slice r -> r + 1), all r.
  std::vector<ComponentMatch> matches;
  std::vector<PersistenceTrack> tracks;
};

/// Components and tracks invalidated by a local change (see
/// :func:`PersistentModularity::invalidatedAncestry`).  Positions
/// disambiguate automorphic twins that share a canonical hash.
struct InvalidationRead {
  /// Unique invalidated component ids.
  std::vector<ComponentId> components;
  /// (slice, hierarchy level index, index in level) of every invalidated
  /// component.  Hierarchy level index k refers to aggregation level k + 1.
  std::vector<std::array<std::size_t, 3>> positions;
  /// Indices into ``ScanReport::tracks`` of the affected tracks.
  std::vector<std::size_t> tracks;
};

/// Label-free discovery of connected modular components that persist across
/// resolution and cobordism time (design spec section 8, ticket #765).
///
/// **Domain and exact identities.**  The input is a finite nonnegative
/// weighted undirected similarity graph (the complex's one-skeleton under a
/// documented monotone weight map; see :class:`WeightMap`).  On that domain
/// the class evaluates generalized modularity exactly:
///
///   Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j / (2m)) [c_i = c_j],
///
/// via the per-community sufficient statistics
/// Q = sum_c [ Sigma_in(c)/(2m) - gamma (S_c/(2m))^2 ], with the aggregated
/// self-loop convention A_CC = Sigma_in(C).  Every cached local move gain is
/// the exact closed form
///
///   dQ(v: a -> b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a) / (2 m^2),
///
/// evaluated in O(deg v) from the cached community totals, so one complete
/// local-move sweep is near O(|E|) (up to revisits).  These identities are
/// exact in double arithmetic; incremental accumulations use compensated
/// summation and are tested against cold recomputation at the ~1e-15..1e-14
/// double round-off standard.
///
/// **Heuristic status (mandatory reading).**  Global modularity maximization
/// is NP-hard; the discovery is a deterministic multilevel aggregation from
/// a fixed seed sequence that retains the best exact score and reports the
/// restart spread honestly.  Nothing here claims the global optimum, and the
/// Newman-Girvan / generalized-modularity score runs on a combinatorial /
/// nonnegative one-skeleton: it is blind to signed and complex Hodge
/// weights.  Modularity is a heuristic proposal generator only.  Nothing in
/// this class may enter the emergence objective, and a modularity read may
/// never veto an otherwise certified fiber — fiber acceptance rests solely
/// on the independent weight-aware gap/localization/leakage/persistence/
/// anchor certificates (supplied by later tickets; reported here as unknown,
/// never zero).  Communities are proposals and carry no connectivity
/// guarantee.
///
/// **Label-freedom.**  Visit order and tie-breaking derive from a canonical
/// structural ranking (iterated weighted color refinement with
/// individualization by breadth-first distance), and component identity from
/// oriented incidence and lineage — never from raw vertex numbers.  The
/// discovery is a pure function of the labeled graph: edge input order never
/// changes the result.  Within a refinement class whose members are
/// structurally indistinguishable the individualization picks an arbitrary
/// representative (minimum cell id); when such classes are automorphism
/// orbits (all shipped fixtures) the discovered hierarchy under a relabeling
/// is the automorphic image — identical scores, identical per-level
/// canonical-hash multisets, supports mapped up to graph automorphism.
///
/// **Read-only.**  A pure observable: never calls a solver and never mutates
/// the spacetime it reads.  Instances are immutable after construction; the
/// canonical ranking is a lazily computed per-instance cache (the
/// sufficient-statistics caches live inside each discovery run).
class PersistentModularity {
public:
  /// The documented monotone map from complex edge magnitude to similarity
  /// weight (design spec section 8.1).  Nothing else is silently mixed into
  /// the metric.
  enum class WeightMap {
    /// w = 1 per edge: the combinatorial one-skeleton, exactly the graph the
    /// legacy Newman-Girvan reads (SparseGraph::modularity,
    /// Spacetime::modularityOnSkeleton) score.
    Unit,
    /// w = exp(-|l|), l the complex edge length: monotone decreasing in the
    /// edge magnitude (the mutual-information convention l = -log I), values
    /// in (0, 1].
    ExpNegAbsLength,
  };

  /// Build from an explicit nonnegative weighted edge list.  Cells are
  /// identified by arbitrary 64-bit ids; the node set is the union of the
  /// endpoint ids and ``isolatedCells``.  Parallel edges are consolidated by
  /// weight summation; self-loops and zero-weight edges are ignored at
  /// level 0.  Throws std::invalid_argument on negative weights or
  /// mismatched array lengths.
  static PersistentModularity fromWeightedEdges(
      const std::vector<std::uint64_t> &src,
      const std::vector<std::uint64_t> &tgt,
      const std::vector<double> &weight,
      const std::vector<std::uint64_t> &isolatedCells = {});

  /// Build the similarity graph from the spacetime's one-skeleton (vertices
  /// and edges) under the given weight map.  Read-only on the spacetime.
  static PersistentModularity fromSpacetime(
      const Spacetime &st, WeightMap map = WeightMap::ExpNegAbsLength);

  std::size_t nCells() const noexcept { return nNodes_; }
  std::size_t nEdges() const noexcept { return nEdges_; }
  /// Total adjacency weight 2m = sum_ij A_ij.
  double totalWeight2() const noexcept { return twoM_; }
  /// The cell ids in internal storage order (input first-appearance order;
  /// carries no convention).
  const std::vector<std::uint64_t> &cellIds() const noexcept {
    return cellIds_;
  }

  /// Exact generalized modularity Q_gamma of a fixed partition
  /// (``labels[i]`` labels cell ``cellIds()[i]``; distinct values are
  /// distinct communities).  The fixed-partition entry point: at gamma = 1
  /// on a Unit-weight graph this is exactly the Newman-Girvan score.
  /// Community terms are combined in canonical-hash-free ascending label
  /// order.  Throws std::invalid_argument when labels.size() != nCells().
  double modularityGamma(const std::vector<int> &labels, double gamma) const;

  /// Deterministic label-free discovery at one resolution: multilevel
  /// aggregation over ``cfg.restarts`` seeds from the fixed sequence,
  /// keeping the best exact score (ties broken by the sorted component
  /// hash lists).  ``cfg.resolutions`` is ignored here.
  ResolutionSlice discover(double gamma,
                           const PersistentModularityConfig &cfg) const;

  /// The configurable resolution-sequence scan: one slice per entry of
  /// ``cfg.resolutions``, adjacent slices matched by support overlap into
  /// persistence tracks.
  ScanReport scanResolutions(const PersistentModularityConfig &cfg) const;

  /// Match components across resolution or cobordism time by simplex-support
  /// overlap (Jaccard on level-0 cell ids; both sides must reference a
  /// common cell-id universe, e.g. the same evolving complex).  For each
  /// component of ``a`` the best-overlap partner in ``b`` is emitted
  /// (overlap > 0), ties broken by canonical hash then position.  When a
  /// projector-overlap hook is installed its value is reported per match;
  /// matching decisions remain support-based until a later ticket supplies
  /// the projectors.
  std::vector<ComponentMatch> matchComponents(
      const std::vector<ComponentRead> &a,
      const std::vector<ComponentRead> &b) const;

  /// The documented interface hook for spectral-projector overlap.  The
  /// callback receives the two component ids and returns their projector
  /// overlap in [0, 1].  This ticket only plumbs the hook: no projector is
  /// implemented here, and without a hook the field stays absent (unknown).
  using ProjectorOverlapHook =
      std::function<double(const ComponentId &, const ComponentId &)>;
  void setProjectorOverlapHook(ProjectorOverlapHook hook) {
    projectorHook_ = std::move(hook);
  }

  /// Components and tracks whose ancestry a local change touches: every
  /// component (at every hierarchy level of every slice) whose support
  /// intersects ``touchedCells``, plus the tracks containing one.  Siblings
  /// with disjoint support remain valid.  Pure bookkeeping over the report;
  /// no recomputation is triggered.
  static InvalidationRead invalidatedAncestry(
      const ScanReport &report,
      const std::vector<std::uint64_t> &touchedCells);

private:
  PersistentModularity() = default;

  // CSR similarity graph (undirected; each edge stored in both directions).
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;
  std::vector<std::uint32_t> indices_;
  std::vector<double> weights_;
  std::vector<double> strength_;          // k_i
  double twoM_ = 0.0;                     // 2m
  std::vector<std::uint64_t> cellIds_;    // internal index -> cell id
  // Consolidated edges with their stored (input) orientation: the oriented
  // incidence used for level-1 identity hashing and exact cold recomputes.
  std::vector<std::uint32_t> orientedSrc_;
  std::vector<std::uint32_t> orientedTgt_;
  std::vector<double> orientedW_;

  // Lazily computed canonical structure (invariant colors + visit ranks).
  mutable bool canonicalReady_ = false;
  mutable std::vector<std::uint64_t> stableColor_;  // pre-individualization
  mutable std::vector<std::uint32_t> rank_;          // canonical visit rank

  ProjectorOverlapHook projectorHook_;

  void ensureCanonical() const;

  struct LevelGraph;   // aggregated weighted graph with self-loops
  struct RunResult;    // one restart's full multilevel outcome
  RunResult runOnce(double gamma, std::uint64_t seed,
                    const PersistentModularityConfig &cfg) const;
  ResolutionSlice buildSlice(double gamma, const RunResult &winner,
                             std::vector<RestartRead> restarts) const;
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H
