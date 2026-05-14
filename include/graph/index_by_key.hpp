// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Build a ``key → index`` lookup table from a vector of items.
//
// Four sites in the codebase (Spacetime::getDualAdjacency,
// Poset::fromSpacetime, ReggeSolver::flattenMeshForGpu's edge map,
// and similar) all hand-roll the same pattern:
//
//   std::unordered_map<KeyType, IndexType> map;
//   map.reserve(items.size());
//   for (IndexType i = 0; i < items.size(); ++i) {
//       map.emplace(keyExtractor(items[i]), i);
//   }
//
// This helper compresses the idiom to a single call and makes the
// key-extraction function explicit at the call site.
//
// Note: the helper assumes keys are unique across ``items``. If two
// items map to the same key, the first wins (``emplace`` semantics)
// and the duplicate is silently dropped — the same behaviour each
// original callsite already had.

#pragma once

#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tessera::graph {

template <typename IndexType = int, typename Container, typename KeyFn>
[[nodiscard]] auto
indexByKey(Container const& items, KeyFn keyFn) {
    using KeyType = std::decay_t<decltype(keyFn(*items.begin()))>;
    std::unordered_map<KeyType, IndexType> out;
    out.reserve(items.size());
    IndexType i{0};
    for (auto const& item : items) {
        out.emplace(keyFn(item), i);
        ++i;
    }
    return out;
}

} // namespace tessera::graph
