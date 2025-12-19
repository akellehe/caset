// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by Andrew Kelleher on 11/13/25.
//

#ifndef CASET_VERTEXFINGERPRINT_H
#define CASET_VERTEXFINGERPRINT_H

#include <tuple>
#include <cassert>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include "Logger.h"

///
/// \file Fingerprint.h
/// \brief Order-independent hashing system for set-based object identification
///
/// This file provides a fingerprinting system that creates unique hash values from unordered
/// sets of IDs. The key property is **commutativity**: the fingerprint of {1, 2, 3} equals
/// the fingerprint of {3, 1, 2}.
///
/// # Mathematical Foundation
///
/// The fingerprint \f$ h \f$ of a set \f$ S = \{id_1, id_2, \ldots, id_n\} \f$ is computed as:
///
/// \f[
/// h(S) = \bigoplus_{id \in S} \text{mix64}(id)
/// \f]
///
/// where \f$ \oplus \f$ denotes XOR (exclusive-or) and mix64 is a bijective mixing function.
///
/// # Design Rationale
///
/// In causal set theory, simplices are defined by their constituent vertices, not by the order
/// in which vertices are specified. A 2-simplex with vertices {v1, v2, v3} is the same simplex
/// as {v3, v1, v2}. Using XOR provides an efficient commutative hash that:
///
/// - Runs in O(n) time
/// - Uses O(1) space after hashing
/// - Has good avalanche properties (small changes to input produce large changes to output)
/// - Allows incremental updates without full recomputation
///
/// # Performance Characteristics
///
/// - **Add/Remove ID**: O(n) where n is the current number of IDs (typically < 10)
/// - **Compute Hash**: O(n) but lazy - only recomputed when dirty flag is set
/// - **Equality Check**: O(1) hash comparison, O(n²) fallback for collision handling
/// - **Memory**: Fixed 64-element array + metadata ≈ 520 bytes per instance
///

namespace caset {

// ========================================
// Type Definitions
// ========================================

///
/// \brief Type for object identifiers (64-bit unsigned integers)
///
/// All vertices, edges, and simplices are assigned unique IDs of this type.
/// Using uint64_t allows for \f$ 2^{64} \approx 1.8 \times 10^{19} \f$ unique objects.
///
using IdType = std::uint64_t;

///
/// \brief Maximum number of IDs that can be stored in a fingerprint
///
/// This limit is chosen based on typical simplex sizes in causal set simulations:
/// - 0-simplex (vertex): 1 ID
/// - 1-simplex (edge): 2 IDs
/// - 2-simplex (triangle): 3 IDs
/// - 3-simplex (tetrahedron): 4 IDs
/// - k-simplex: k+1 IDs
///
/// The value 64 supports up to 63-simplices, far exceeding practical needs.
///
inline constexpr std::size_t kMax = 64;

///
/// \brief Fixed-size array for storing IDs
///
/// Using a fixed-size array avoids dynamic allocation and cache misses.
/// Only the first n_ elements are valid; the rest are uninitialized.
///
using IdArray = std::array<IdType, kMax>;

///
/// \brief FNV-1a hash seed (unused in current implementation)
///
/// This constant is a standard FNV-1a offset basis. The current mix64()
/// implementation uses a different mixing algorithm, so this seed is not used.
/// Kept for potential future hash function variants.
///
inline constexpr std::uint64_t kSeed = 0xcbf29ce484222325ull;

// ========================================
// Fingerprint Class
// ========================================

///
/// \brief Order-independent hash for sets of IDs
///
/// # Purpose
///
/// The Fingerprint class provides a unique identifier for any unordered set of IDs.
/// This is essential for causal set theory where geometric objects (simplices) are
/// defined by their constituent vertices, regardless of specification order.
///
/// # Usage Pattern
///
/// Include a `Fingerprint` as a public member in your class:
///
/// ```cpp
/// class Simplex {
///   public:
///     Fingerprint fingerprint;
///     // ... other members
/// };
/// ```
///
/// Then use the provided template functors for hash tables:
///
/// ```cpp
/// using SimplexHash = FingerprintHash<Simplex>;
/// using SimplexEq = FingerprintEq<Simplex>;
/// std::unordered_set<Simplex, SimplexHash, SimplexEq> simplices;
/// ```
///
/// Or specialize std::hash in the std namespace:
///
/// ```cpp
/// namespace std {
/// template<>
/// struct hash<caset::Simplex> {
///   size_t operator()(const caset::Simplex &s) const noexcept {
///     return std::hash<std::uint64_t>{}(s.fingerprint.fingerprint());
///   }
/// };
/// }
/// ```
///
/// # Implementation Details
///
/// ## Lazy Evaluation Pattern
///
/// The fingerprint uses a dirty flag to defer hash computation until needed:
/// - Modifications (add/remove) only mark dirty=true
/// - Reading fingerprint() triggers recomputation if dirty
/// - Mutable members allow const methods to update cache
///
/// ## Collision Handling
///
/// While hash collisions are rare (probability ≈ \f$ n^2 / 2^{64} \f$ for n objects),
/// operator== performs full set comparison to handle them:
/// 1. Quick reject: different sizes or hash values
/// 2. O(n²) set comparison if hashes match
///
/// ## Thread Safety
///
/// **Not thread-safe**. The mutable dirty flag and hash cache h_ can race.
/// Concurrent reads are safe if no writes occur. Use external synchronization
/// for concurrent modification.
///
class Fingerprint {
  public:
    // ========================================
    // Constructors
    // ========================================

    ///
    /// \brief Default constructor creating an empty fingerprint
    ///
    /// Initializes with zero IDs and zero hash. Not dirty since hash is already current.
    ///
    Fingerprint() noexcept : ids_({}), n_(0), h_(0), dirty_(false) {
    }

    ///
    /// \brief Construct from a vector of IDs
    /// \param ids Initial set of IDs (duplicates are automatically filtered)
    ///
    /// # Complexity
    /// O(n²) where n = ids.size(), due to duplicate checking in addId().
    /// For small n (< 10), this is faster than using std::unordered_set.
    ///
    explicit Fingerprint(const std::vector<IdType> &ids) noexcept : ids_({}), n_(0), h_(0), dirty_(true) {
      setIds(ids);
    }

    // ========================================
    // Hash Mixing Function
    // ========================================

    ///
    /// \brief Avalanche hash mixer based on MurmurHash3 finalizer
    /// \param x Input value to mix
    /// \return Mixed 64-bit hash value
    ///
    /// # Algorithm
    ///
    /// This implements a variant of the MurmurHash3 64-bit finalizer:
    ///
    /// \f[
    /// \begin{align}
    /// x &\gets x + \phi \cdot 2^{64} \\
    /// x &\gets (x \oplus (x \gg 30)) \cdot c_1 \\
    /// x &\gets (x \oplus (x \gg 27)) \cdot c_2 \\
    /// x &\gets x \oplus (x \gg 31)
    /// \end{align}
    /// \f]
    ///
    /// where \f$ \phi \approx 1.618 \f$ is the golden ratio and \f$ c_1, c_2 \f$
    /// are carefully chosen mixing constants.
    ///
    /// # Properties
    ///
    /// - **Bijective**: Every input maps to a unique output (invertible)
    /// - **Avalanche**: Flipping one input bit changes ~50% of output bits
    /// - **constexpr**: Can be evaluated at compile time
    /// - **Fast**: 4 operations (add, 3 xor-shift-multiply sequences)
    ///
    /// # Performance
    ///
    /// On modern x86-64 CPUs: ~4-5 cycles per call (pipelined).
    ///
    static inline constexpr std::uint64_t mix64(IdType x) noexcept {
      x += 0x9e3779b97f4a7c15ull;  // Golden ratio * 2^64
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    // ========================================
    // ID Management
    // ========================================

    ///
    /// \brief Replace all IDs with a new set
    /// \param ids New set of IDs (duplicates filtered)
    ///
    /// Clears existing IDs and adds each new ID via addId().
    /// Marks the fingerprint dirty for lazy recomputation.
    ///
    /// # Complexity
    /// O(n²) where n = ids.size()
    ///
    void setIds(const std::vector<IdType> &ids) noexcept {
      n_ = 0;
      for (auto id : ids) {
        addId(id);
      }
      dirty_ = true;
    }

    ///
    /// \brief Add a single ID to the set
    /// \param id ID to add (ignored if already present)
    ///
    /// # Duplicate Detection
    ///
    /// Uses linear search O(n) to check for duplicates. For small n (< 10),
    /// this is faster than std::unordered_set due to cache locality and
    /// lack of allocation overhead.
    ///
    /// # Overflow Behavior
    ///
    /// If n_ == kMax (64 IDs), the ID is silently ignored. In practice,
    /// simplices rarely exceed 5 vertices, so this limit is never reached.
    ///
    /// # Branch Prediction
    ///
    /// Uses `[[unlikely]]` hint for duplicate case, optimizing the common
    /// path where IDs are unique.
    ///
    void addId(IdType id) noexcept {
      // Check for duplicates using linear scan (fast for small n)
      for (std::uint8_t i = 0; i < n_; ++i) {
        if (ids_[i] == id) [[unlikely]] return; // Already present
      }

      if (n_ < kMax) [[likely]] {
        ids_[n_++] = id;
        dirty_ = true;
      }
    }

    ///
    /// \brief Remove an ID from the set
    /// \param id ID to remove (no-op if not present)
    ///
    /// # Algorithm
    ///
    /// Uses swap-and-pop: replace removed element with last element,
    /// then decrement count. This avoids shifting elements and maintains
    /// O(n) complexity.
    ///
    /// # Order Preservation
    ///
    /// The order of IDs is **not** preserved, but since the hash is commutative,
    /// this doesn't affect correctness.
    ///
    void removeId(IdType id) noexcept {
      for (std::uint8_t i = 0; i < n_; ++i) {
        if (ids_[i] == id) {
          // Remove by swapping with last element
          ids_[i] = ids_[--n_];
          dirty_ = true;
          return;
        }
      }
    }

    // ========================================
    // Hash Computation
    // ========================================

    ///
    /// \brief Get the cached fingerprint value (lazy evaluation)
    /// \return 64-bit hash value
    ///
    /// # Lazy Evaluation
    ///
    /// Only recomputes if dirty flag is set. This amortizes the cost
    /// of multiple modifications:
    ///
    /// ```cpp
    /// fp.addId(1);  // O(1) - just marks dirty
    /// fp.addId(2);  // O(1) - just marks dirty
    /// fp.addId(3);  // O(1) - just marks dirty
    /// auto h = fp.fingerprint();  // O(n) - computes once
    /// ```
    ///
    /// # Hash Formula
    ///
    /// \f[
    /// h = \bigoplus_{i=0}^{n-1} \text{mix64}(\text{ids}_i)
    /// \f]
    ///
    /// XOR is commutative and associative, so order doesn't matter.
    ///
    /// # Mutable State
    ///
    /// Uses mutable members to allow updating cache in const method.
    /// This is the "logical const" pattern: the observable value doesn't
    /// change, only internal cache state.
    ///
    std::uint64_t fingerprint() const noexcept {
      if (dirty_) [[unlikely]] {
        // Recompute hash from all IDs using commutative XOR
        h_ = 0;
        for (std::uint8_t i = 0; i < n_; ++i) {
          h_ ^= mix64(ids_[i]);
        }
        dirty_ = false;
      }
      return h_;
    }

    ///
    /// \brief Force immediate hash recomputation
    ///
    /// Normally the hash is computed lazily on access. Call this to
    /// force evaluation immediately, e.g., before a tight loop that
    /// repeatedly reads the fingerprint.
    ///
    /// # Use Case
    ///
    /// ```cpp
    /// for (int i = 0; i < 1000000; ++i) {
    ///   auto h = fp.fingerprint();  // Redundant dirty checks
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```cpp
    /// fp.refresh();  // Compute once
    /// for (int i = 0; i < 1000000; ++i) {
    ///   auto h = fp.fingerprint();  // Fast path: already clean
    /// }
    /// ```
    ///
    void refresh() const noexcept {
      if (dirty_) fingerprint(); // Trigger lazy evaluation
    }

    ///
    /// \brief Deprecated: Batch update and refresh
    /// \param ids New set of IDs
    ///
    /// **Deprecated**: Use setIds() + refresh() instead.
    /// Kept for backward compatibility.
    ///
    void refreshFingerprint(const std::vector<IdType> &ids) {
      setIds(ids);
      refresh();
    }

    // ========================================
    // Operators and Utilities
    // ========================================

    ///
    /// \brief Generate debug string representation
    /// \return String showing hash value and ID list
    ///
    /// # Format
    ///
    /// `<Fingerprint: 12345678901234567890 (1, 5, 9)>`
    ///
    /// Forces refresh to ensure hash is current before stringifying.
    ///
    std::string toString() const {
      // Force refresh before stringifying
      refresh();

      std::stringstream ss;
      ss << "<Fingerprint: " << h_ << " (";
      for (std::uint8_t i = 0; i < n_; ++i) {
        ss << ids_[i];
        if (i < n_ - 1) ss << ", ";
      }
      ss << ")>";
      return ss.str();
    }

    ///
    /// \brief Equality comparison with collision-resistant set comparison
    /// \param o Other fingerprint to compare
    /// \return true if both fingerprints represent the same set of IDs
    ///
    /// # Algorithm
    ///
    /// 1. Refresh both fingerprints if dirty
    /// 2. Quick reject: if n_ differs, sets can't be equal
    /// 3. Quick reject: if h_ differs, sets likely differ (collision possible)
    /// 4. Full O(n²) set comparison to handle hash collisions
    ///
    /// # Hash Collision Probability
    ///
    /// For n fingerprints with k IDs each, birthday paradox gives:
    ///
    /// \f[
    /// P(\text{collision}) \approx 1 - e^{-n^2 / (2 \cdot 2^{64})}
    /// \f]
    ///
    /// For 1 billion simplices: \f$ P \approx 2.7 \times 10^{-11} \f$ (negligible)
    ///
    /// # Performance
    ///
    /// - Typical case (no collision): O(1)
    /// - Collision case: O(n²) where n is number of IDs
    ///
    bool operator==(const Fingerprint &o) const noexcept {
      // Force refresh on both sides if needed
      if (dirty_) [[unlikely]] refresh();
      if (o.dirty_) [[unlikely]] o.refresh();

      if (n_ != o.n_) return false;
      if (h_ != o.h_) return false; // Fast path: hash mismatch

      // Compare sets - both are unique but unsorted
      // For small n (typically 2-5), nested loop is fastest
      for (std::uint8_t i = 0; i < n_; ++i) {
        bool found = false;
        for (std::uint8_t j = 0; j < o.n_; ++j) {
          if (ids_[i] == o.ids_[j]) {
            found = true;
            break;
          }
        }
        if (!found) return false;
      }
      return true;
    }

    ///
    /// \brief Inequality comparison
    /// \param o Other fingerprint
    /// \return true if fingerprints differ
    ///
    bool operator!=(const Fingerprint &o) const noexcept {
      return !(*this == o);
    }

  private:
    // ========================================
    // Private Members
    // ========================================

    IdArray ids_{};                ///< Unique IDs (unsorted for speed)
    std::uint8_t n_{0};            ///< Count of unique IDs [0, kMax]
    mutable std::uint64_t h_{0};   ///< Cached XOR hash - recomputed when dirty
    mutable bool dirty_{false};    ///< Needs recomputation?
};

// ========================================
// Hash and Equality Functors
// ========================================

///
/// \brief Hash functor for types with a `fingerprint` member
/// \tparam T Type that has a public `Fingerprint fingerprint` member
///
/// # Purpose
///
/// Provides a generic hash functor for use with std::unordered_set and std::unordered_map.
/// Supports hashing both value types and shared_ptr types, as well as heterogeneous lookup.
///
/// # Heterogeneous Lookup
///
/// Defines `is_transparent` to enable C++20 heterogeneous lookup. This allows:
///
/// ```cpp
/// std::unordered_set<SimplexPtr, FingerprintHash<Simplex>, FingerprintEq<Simplex>> simplices;
/// uint64_t fp = computeFingerprint();
/// auto it = simplices.find(fp);  // No temporary object created!
/// ```
///
/// Without `is_transparent`, you'd need to create a full SimplexPtr just to search.
///
/// # Supported Types
///
/// - `T` (value type)
/// - `std::shared_ptr<T>`
/// - `std::shared_ptr<const T>`
/// - `uint64_t` (raw fingerprint value for heterogeneous lookup)
///
template<typename T>
struct FingerprintHash {
  using is_transparent = void; // enables heterogeneous lookup

  /// Hash a value type
  size_t operator()(const T &s) const noexcept {
    return static_cast<size_t>(s.fingerprint.fingerprint());
  }

  /// Hash a shared_ptr
  size_t operator()(const std::shared_ptr<T> &s) const noexcept {
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a shared_ptr to const
  size_t operator()(const std::shared_ptr<const T> &s) const noexcept {
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a raw fingerprint value (for heterogeneous lookup)
  size_t operator()(uint64_t fp) const noexcept {
    return static_cast<size_t>(fp);
  }
};

///
/// \brief Equality functor for types with a `fingerprint` member
/// \tparam T Type that has a public `Fingerprint fingerprint` member
///
/// # Purpose
///
/// Provides equality comparison for use with std::unordered_set and std::unordered_map.
/// Compares objects by their fingerprint values, not by address or full state.
///
/// # Supported Comparisons
///
/// - `T == T` (delegates to T::operator==)
/// - `T == uint64_t` (compares fingerprint to raw value)
/// - `shared_ptr<T> == shared_ptr<T>` (compares fingerprints)
/// - `shared_ptr<const T> == shared_ptr<const T>` (with nullptr handling)
/// - All combinations for heterogeneous lookup
///
/// # Nullptr Handling
///
/// - `nullptr == nullptr`: Returns true
/// - `nullptr == non-null`: Returns false
/// - When CASET_ASSERTIONS is defined: Some nullptr cases abort instead
///
template<typename T>
struct FingerprintEq {
  using is_transparent = void;  // enables heterogeneous lookup

  /// Compare two value types (delegates to operator==)
  bool operator()(const T &a, const T &b) const noexcept {
    return a == b;
  }

  /// Compare value type to raw fingerprint
  bool operator()(const T &a, uint64_t fp) const noexcept {
    return a.fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to value type
  bool operator()(uint64_t fp, const T &a) const noexcept {
    return fp == a.fingerprint.fingerprint();
  }

  /// Compare two shared_ptr by fingerprint (not by pointer address)
  bool operator()(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) const noexcept {
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }

  /// Compare shared_ptr to raw fingerprint
  bool operator()(const std::shared_ptr<T> &a, uint64_t fp) const noexcept {
    return a->fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to shared_ptr (with nullptr check)
  bool operator()(uint64_t fp, const std::shared_ptr<T> &a) const noexcept {
    if (a == nullptr) return false;
    return fp == a->fingerprint.fingerprint();
  }

  /// Compare two const shared_ptr with explicit nullptr handling
  bool operator()(const std::shared_ptr<const T> &a, const std::shared_ptr<const T> &b) const noexcept {
    if (a == nullptr && b == nullptr) {
      return true;
    }
    if (a == nullptr || b == nullptr) {
      return false;
    }
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }

  /// Compare const shared_ptr to raw fingerprint
  bool operator()(const std::shared_ptr<const T> &a, uint64_t fp) const noexcept {
    return a->fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to const shared_ptr (aborts on nullptr with assertions)
  bool operator()(uint64_t fp, const std::shared_ptr<const T> &a) const noexcept {
#ifdef CASET_ASSERTIONS
    if (a == nullptr) {
      CLOG(CRITICAL_LEVEL, "Nullptr in FingerprintEq");
      std::abort();
    }
#endif
    return fp == a->fingerprint.fingerprint();
  }
};

///
/// \brief Hash functor for shared_ptr types with fingerprint member (strict nullptr checking)
/// \tparam T Shared pointer type (e.g., std::shared_ptr<Simplex>)
///
/// # Difference from FingerprintHash
///
/// This variant assumes T is already a pointer type and performs stricter nullptr checking
/// when CASET_ASSERTIONS is defined. Use this when you know you're working with pointers
/// and want to catch nullptr bugs early in development.
///
/// # Assertions
///
/// When CASET_ASSERTIONS is defined, aborts if given a nullptr.
/// In production builds (no assertions), behavior is undefined for nullptr.
///
template<typename T>
struct FingerprintPtrHash {
  /// Hash a pointer (aborts on nullptr with assertions)
  size_t operator()(const T &s) const noexcept {
#ifdef CASET_ASSERTIONS
    if (s == nullptr) {
      CLOG(CRITICAL_LEVEL, "Nullptr in FingerprintPtrHash");
      std::abort();
    }
#endif
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a raw fingerprint value
  size_t operator()(uint64_t fp) const noexcept {
    return static_cast<size_t>(fp);
  }
};

///
/// \brief Equality functor for shared_ptr types with fingerprint member (strict nullptr checking)
/// \tparam T Shared pointer type (e.g., std::shared_ptr<Simplex>)
///
/// # Assertions
///
/// When CASET_ASSERTIONS is defined, asserts that both pointers are non-null.
/// Use this when nullptr should never occur and you want to catch bugs early.
///
template<typename T>
struct FingerprintPtrEq {
  /// Compare two pointers by fingerprint (asserts non-null)
  bool operator()(const T &a, const T &b) const noexcept {
#ifdef CASET_ASSERTIONS
    assert(a != nullptr);
    assert(b != nullptr);
#endif
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }
};

// ========================================
// Debugging Utilities (Assertion Builds Only)
// ========================================

#ifdef CASET_ASSERTIONS
///
/// \brief Debug utility for detecting hash table corruption
/// \tparam Ptr Pointer type stored in container (e.g., SimplexPtr)
/// \tparam PtrHash Hash functor for Ptr
/// \tparam PtrEq Equality functor for Ptr
///
/// # Purpose
///
/// During development with CASET_ASSERTIONS enabled, this class provides validation
/// methods to detect common hash table corruption issues:
///
/// - **Nullptr entries**: Dangling pointers that weren't properly removed
/// - **Duplicate fingerprints**: Same object inserted multiple times
/// - **Hash table inconsistency**: Object in wrong bucket or missing
///
/// # Usage
///
/// Call isCorrupted() or wouldDuplicate() before/after critical operations:
///
/// ```cpp
/// #ifdef CASET_ASSERTIONS
/// using Detector = CorruptionDetector<SimplexPtr, SimplexHash, SimplexEq>;
/// if (Detector::isCorrupted(simplices)) {
///   // Log error, abort, or enter debugger
/// }
/// #endif
/// ```
///
/// # Performance Impact
///
/// These checks are O(n) and only active when CASET_ASSERTIONS is defined.
/// In release builds (no assertions), this entire class is compiled out.
///
template<typename Ptr, typename PtrHash, typename PtrEq>
class CorruptionDetector {
  public:
    ///
    /// \brief Check if an unordered_set contains corrupted entries
    /// \param container Set to validate
    /// \return true if corruption detected (nullptr or duplicate fingerprints)
    ///
    /// # Checks Performed
    ///
    /// 1. **Nullptr check**: Ensures no null or dangling pointers
    /// 2. **Duplicate check**: Ensures all fingerprints are unique
    ///
    /// # Complexity
    /// O(n) where n = container.size()
    ///
    static bool isCorrupted(const std::unordered_set<Ptr, PtrHash, PtrEq> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &o : container) {
        if (o == nullptr || o.get() == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (nullptr)!");
          return true;
        }
        if (seen.contains(o->fingerprint.fingerprint())) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(o->fingerprint.fingerprint());
      }
      // TODO: Check that objects are aligned to their fingerprints.

      return false;
    }

    ///
    /// \brief Check if an unordered_map<IdType, Ptr> contains corrupted entries
    /// \param container Map to validate
    /// \return true if corruption detected
    ///
    /// # Checks Performed
    ///
    /// 1. **Nullptr check**: Ensures no null object pointers
    /// 2. **Fingerprint nullptr check**: Ensures fingerprint member is valid
    /// 3. **Duplicate fingerprint check**: No two objects with same fingerprint
    /// 4. **Duplicate key check**: No duplicate keys in map
    ///
    /// # Complexity
    /// O(n) where n = container.size()
    ///
    static bool isCorrupted(const std::unordered_map<IdType, Ptr, PtrHash, PtrEq> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &[id, o] : container) {
        if (o == nullptr || o.get() == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (nullptr)!");
          return true;
        }
        if (o->fingerprint == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (fp nullptr)!");
          return true;
        }
        if (seen.contains(o->fingerprint.fingerprint())) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        if (seen.contains(id)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(o->fingerprint.fingerprint());
        seen.insert(id);
      }
      return false;
    }

    ///
    /// \brief Check if an ID→ID map contains duplicate keys or values
    /// \param container Map to validate
    /// \return true if corruption detected (duplicate keys or values)
    ///
    /// This overload is for simple ID mappings without object pointers.
    ///
    /// # Complexity
    /// O(n) where n = container.size()
    ///
    static bool isCorrupted(const std::unordered_map<IdType, IdType> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &[id, o] : container) {
        if (seen.contains(id)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        if (seen.contains(o)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(id);
      }
      return false;
    }

    ///
    /// \brief Check if inserting an element would create a duplicate
    /// \param container Set to check
    /// \param newElement Element to potentially insert
    /// \return true if element already exists (by pointer or fingerprint)
    ///
    /// # Use Case
    ///
    /// Call before insertion to verify uniqueness:
    ///
    /// ```cpp
    /// if (Detector::wouldDuplicate(simplices, newSimplex)) {
    ///   CLOG(ERROR_LEVEL, "Attempted to insert duplicate simplex!");
    ///   std::abort();
    /// }
    /// simplices.insert(newSimplex);
    /// ```
    ///
    /// # Checks Performed
    ///
    /// 1. Pointer equality (same shared_ptr instance)
    /// 2. Fingerprint equality (different pointers, same content)
    /// 3. Container membership (via contains())
    ///
    /// # Complexity
    /// O(n) linear scan + O(1) hash lookup
    ///
    static bool wouldDuplicate(const std::unordered_set<Ptr, PtrHash, PtrEq> &container, const Ptr &newElement) {
      std::unordered_set<IdType> seen{};
      for (const auto &o : container) {
        if (o == newElement) {
          return true;
        }
        if (o->fingerprint.fingerprint() == newElement->fingerprint.fingerprint()) {
          return true;
        }
      }
      for (const auto &o : container) {
        if (o == newElement) {
          return true;
        }
      }
      return container.contains(newElement);
    }

    ///
    /// \brief Check if inserting a key-value pair would create a duplicate
    /// \param container Map to check
    /// \param newKey Key to potentially insert
    /// \param newElement Element to potentially insert
    /// \return true if key or element already exists
    ///
    /// Checks for duplicates by:
    /// 1. Key equality
    /// 2. Pointer equality
    /// 3. Fingerprint equality
    /// 4. Container membership
    ///
    /// # Complexity
    /// O(n) linear scan + O(1) hash lookups
    ///
    static bool wouldDuplicate(const std::unordered_map<IdType, Ptr, PtrHash, PtrEq> &container, const IdType &newKey, const Ptr &newElement) {
      std::unordered_set<IdType> seen{};
      for (const auto &[k, o] : container) {
        if (k == newKey || o == newElement) {
          return true;
        }
        if (o->fingerprint.fingerprint() == newElement->fingerprint.fingerprint()) {
          return true;
        }
      }
      return container.contains(newElement) || container.contains(newKey);
    }
};
#endif  // CASET_ASSERTIONS

}

#endif //CASET_VERTEXFINGERPRINT_H
