// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_ANALYTICCACHE_H
#define TESSERA_COBORDISM_ANALYTICCACHE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cobordism/Certificate.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # TouchedStar
///
/// The publication record of one accepted move (design spec section 19): the
/// touched simplices, changed edges, and created/deleted cells, all named by
/// their vertex identifiers. `AnalyticCache::publish` intersects this record
/// with each cache entry's component vertex set — entries meeting the star
/// are invalidated, disjoint siblings survive.
///
/// Vertex identifiers are the right currency because a simplex IS its vertex
/// set (matching is by set, never by any imposed vertex order), and any two
/// cells interact through the Hodge/Regge operators only when they share a
/// vertex — so vertex-set intersection is a conservative, exact-support test
/// for "could this entry's operator have changed".
class TouchedStar {
  public:
    /// Record a simplex whose geometry or incidence changed.
    void addTouchedSimplex(const std::vector<std::uint64_t> &vertexIds);
    /// Record an edge whose complex length or phase changed.
    void addChangedEdge(std::uint64_t vertexA, std::uint64_t vertexB);
    /// Record a created cell (a combinatorial change).
    void addCreatedCell(const std::vector<std::uint64_t> &vertexIds);
    /// Record a deleted cell (a combinatorial change).
    void addDeletedCell(const std::vector<std::uint64_t> &vertexIds);

    /// The union of all recorded vertex identifiers (the star's support).
    [[nodiscard]] const std::unordered_set<std::uint64_t> &vertices() const noexcept {
      return vertices_;
    }
    /// Whether any cell was created or deleted (vs. a pure metric change).
    [[nodiscard]] bool structuralChange() const noexcept { return structural_; }
    /// Whether nothing was recorded.
    [[nodiscard]] bool empty() const noexcept { return vertices_.empty(); }

  private:
    std::unordered_set<std::uint64_t> vertices_{};
    bool structural_{false};
};

/// # AnalyticCache
///
/// Revision- and star-keyed cache for per-component analytic payloads (#764):
/// Hodge blocks, component factorizations, spectral projectors, transports,
/// covariance blocks, Wick contraction plans. Entries are keyed by
///
///  - the **component key** — the order-independent
///    `Fingerprint::fingerprintOf` hash of the component's vertex-identifier
///    set (the same convention `MultiCobordism` uses for its block-Betti
///    slots), so the key is invariant under any vertex relabeling that
///    preserves identifiers and under any input ordering;
///  - a caller-chosen **kind** string (e.g. `"hodge-block"`,
///    `"lu-factorization"`, `"spectral-projector"`); and
///  - an integer **parameter** (degree \f$ k \f$, band index, ...).
///
/// **Freshness contract.** Every entry is stamped with the spacetime's
/// `metricRevisionKey()` at store time; the cache additionally tracks the
/// revision it was last synchronized to. An entry is served when either
///
///  1. the global metric revision still equals the entry's stamp (nothing
///     anywhere changed), or
///  2. every change since the stamp was published through `publish` and this
///     entry survived every intersection test (its component is untouched).
///
/// A revision drift that was never published makes the cache serve NOTHING
/// until the next `publish`/`store` — fail-safe: an unpublished mutation can
/// only cause recomputation, never a stale hit. This composes with, and does
/// not replace, the coarser whole-complex slots already on `Spacetime`
/// (`cachedSpectralSlot`, `cachedBettiNumbers`): those invalidate globally on
/// any change; this cache keeps disjoint siblings alive across a local move.
///
/// Payloads are opaque `shared_ptr<void>` (the `Spacetime::storeSpectralSlot`
/// convention) plus the `Certificate` that grades them, so one container
/// serves every kernel without a parallel cache class per payload type.
///
/// Threading: not synchronized, on the same contract as the `Spacetime`
/// cache slots — candidate spacetimes are thread-private clones and the
/// shared complex is scored serially. Replay mode disables the cache with
/// `setEnabled(false)` and compares against the incremental path.
class AnalyticCache {
  public:
    /// Bind the cache to the spacetime whose geometry revisions gate it. The
    /// spacetime must outlive the cache; the held `shared_ptr` keeps it alive.
    explicit AnalyticCache(std::shared_ptr<Spacetime> st);

    /// The order-independent component key of a vertex-identifier set
    /// (`Fingerprint::fingerprintOf`; XOR of mixed ids — any permutation of
    /// `vertexIds` yields the same key).
    [[nodiscard]] static std::uint64_t componentKey(
        const std::vector<std::uint64_t> &vertexIds);

    /// The bound spacetime's current metric revision
    /// (`Spacetime::metricRevisionKey`): moves on any combinatorial change,
    /// any `setLength`, any `setPhase`.
    [[nodiscard]] std::uint64_t geometryRevision() const;

    /// The bound spacetime's current combinatorial revision
    /// (`Spacetime::structuralRevision`).
    [[nodiscard]] std::uint64_t structuralRevision() const;

    /// Store `payload` + `certificate` for (component vertex set, kind,
    /// parameter), stamped at the CURRENT metric revision. Overwrites any
    /// entry under the same key. The vertex-id set is retained for the
    /// `publish` intersection test.
    void store(const std::vector<std::uint64_t> &componentVertexIds,
               const std::string &kind, std::int64_t parameter,
               std::shared_ptr<void> payload, Certificate certificate);

    /// The cached payload, or nullptr when absent, disabled, or stale under
    /// the freshness contract above. Counts hits/misses.
    [[nodiscard]] std::shared_ptr<void> fetch(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter) const;

    /// The certificate stored beside a payload, or nullptr under exactly the
    /// same conditions `fetch` returns nullptr (does not count a hit/miss).
    [[nodiscard]] const Certificate *fetchCertificate(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter) const;

    /// Publish one accepted move: drop every entry whose component vertex set
    /// intersects `star.vertices()`, then mark the cache synchronized to the
    /// CURRENT metric revision. Call AFTER the mutation, with the COMPLETE
    /// record of what it touched; disjoint siblings survive. An empty star
    /// asserts the revision drift touched nothing any entry depends on.
    void publish(const TouchedStar &star);

    /// Number of live entries.
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    /// Drop every entry and resynchronize to the current revision.
    void clear();

    /// Replay-mode switch: a disabled cache serves nothing (fetch returns
    /// nullptr) but keeps accepting stores, so replay can compare the cold
    /// path against the incremental one.
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    /// Whether the cache is serving entries (see `setEnabled`).
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// Served-fetch count since construction/clear-counters.
    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    /// Empty-fetch count (absent, stale, or disabled).
    [[nodiscard]] std::uint64_t misses() const noexcept { return misses_; }
    /// Entries dropped by `publish` intersection tests.
    [[nodiscard]] std::uint64_t invalidations() const noexcept {
      return invalidations_;
    }

  private:
    struct Entry {
      std::vector<std::uint64_t> vertexIds{};
      std::shared_ptr<void> payload{};
      Certificate certificate{};
      std::uint64_t revision{0};
    };
    struct Key {
      std::uint64_t component{0};
      std::uint64_t kind{0};
      std::int64_t parameter{0};
      bool operator==(const Key &other) const noexcept {
        return component == other.component && kind == other.kind &&
               parameter == other.parameter;
      }
    };
    struct KeyHash {
      std::size_t operator()(const Key &key) const noexcept;
    };

    /// Whether an entry may be served under the freshness contract.
    [[nodiscard]] bool fresh(const Entry &entry) const;
    [[nodiscard]] static Key makeKey(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter);

    std::shared_ptr<Spacetime> st_{};
    std::unordered_map<Key, Entry, KeyHash> entries_{};
    std::uint64_t syncedRevision_{0};
    bool enabled_{true};
    mutable std::uint64_t hits_{0};
    mutable std::uint64_t misses_{0};
    std::uint64_t invalidations_{0};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_ANALYTICCACHE_H
