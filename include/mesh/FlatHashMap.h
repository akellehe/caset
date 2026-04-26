// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Open-addressing hash map with linear probing for uint64_t keys.
// Designed for fingerprint-keyed lookups in the CDT hot path where
// std::unordered_map's chained hashing causes cache misses.
//
// Keys are already well-mixed (via Fingerprint::mix64), so we use
// an identity hash — no re-hashing overhead.

#ifndef TESSERA_FLAT_HASH_MAP_H
#define TESSERA_FLAT_HASH_MAP_H

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace tessera {

/// Identity hash for uint64_t keys that are already well-distributed
/// (e.g. fingerprints produced by mix64).
struct IdentityHash {
    std::size_t operator()(std::uint64_t key) const noexcept { return key; }
};

/// Open-addressing hash map: uint64_t → V with linear probing.
///
/// ~3x faster than std::unordered_map for lookup-heavy workloads due to
/// cache-friendly flat layout (keys and values in contiguous arrays).
///
/// Reserves key 0 as the empty sentinel.  If 0 is a valid key, the
/// caller must offset keys by 1.  In practice, fingerprints from XOR
/// of mix64 values are never 0 for non-empty simplices.
template<typename V>
class FlatHashMap {
  public:
    static constexpr std::uint64_t EMPTY = 0;

    FlatHashMap() { resize(64); }

    void reserve(std::size_t n) {
        std::size_t need = (n * 10) / 7 + 1; // ~70% load factor
        if (need > cap_) resize(nextPow2(need));
    }

    V* find(std::uint64_t key) noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return &vals_[idx];
            if (keys_[idx] == EMPTY) return nullptr;
            idx = (idx + 1) & mask_;
        }
    }

    const V* find(std::uint64_t key) const noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return &vals_[idx];
            if (keys_[idx] == EMPTY) return nullptr;
            idx = (idx + 1) & mask_;
        }
    }

    /// Insert or update.  Returns pointer to the value.
    V& operator[](std::uint64_t key) {
        if (size_ * 10 >= cap_ * 7) grow();
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return vals_[idx];
            if (keys_[idx] == EMPTY) {
                keys_[idx] = key;
                vals_[idx] = V{};
                ++size_;
                return vals_[idx];
            }
            idx = (idx + 1) & mask_;
        }
    }

    /// Insert key-value pair.  No-op if key already exists.
    /// Returns pointer to the (existing or new) value.
    V& insert(std::uint64_t key, const V &val) {
        if (size_ * 10 >= cap_ * 7) grow();
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return vals_[idx];
            if (keys_[idx] == EMPTY) {
                keys_[idx] = key;
                vals_[idx] = val;
                ++size_;
                return vals_[idx];
            }
            idx = (idx + 1) & mask_;
        }
    }

    bool erase(std::uint64_t key) noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == EMPTY) return false;
            if (keys_[idx] == key) {
                // Backward-shift deletion to maintain probe chains
                keys_[idx] = EMPTY;
                --size_;
                std::size_t next = (idx + 1) & mask_;
                while (keys_[next] != EMPTY) {
                    std::size_t ideal = keys_[next] & mask_;
                    // Check if 'next' is displaced past 'idx' (wrapping)
                    bool displaced = (idx <= next)
                        ? (ideal <= idx || ideal > next)
                        : (ideal <= idx && ideal > next);
                    if (displaced) {
                        keys_[idx] = keys_[next];
                        vals_[idx] = std::move(vals_[next]);
                        keys_[next] = EMPTY;
                        idx = next;
                    }
                    next = (next + 1) & mask_;
                }
                return true;
            }
            idx = (idx + 1) & mask_;
        }
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

  private:
    std::vector<std::uint64_t> keys_;
    std::vector<V> vals_;
    std::size_t cap_ = 0;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;

    static std::size_t nextPow2(std::size_t n) {
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    void resize(std::size_t newCap) {
        auto oldKeys = std::move(keys_);
        auto oldVals = std::move(vals_);
        auto oldCap = cap_;
        cap_ = newCap;
        mask_ = cap_ - 1;
        keys_.assign(cap_, EMPTY);
        vals_.resize(cap_);
        size_ = 0;
        for (std::size_t i = 0; i < oldCap; ++i) {
            if (oldKeys[i] != EMPTY) {
                insert(oldKeys[i], oldVals[i]);
            }
        }
    }

    void grow() { resize(cap_ * 2); }
};

} // namespace tessera

#endif // TESSERA_FLAT_HASH_MAP_H
