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
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstring>

namespace caset {

using IdType = std::uint64_t;
inline constexpr std::size_t kMax = 64;
using IdArray = std::array<IdType, kMax>;
inline constexpr std::uint64_t kSeed = 0xcbf29ce484222325ull;

///
/// ## Fingerprint
/// Any category of the equivalence class defined by a unique set of (e.g. vertex) IDs can be enforced by the
/// Fingerprint class.
///
/// To implement this, you should include an instance of the Fingerprint class at the `fingerprint` public member of
/// whatever class you're implementing it on. Then you can also use the Eq and Hash templates like:
///
/// ```C++
/// using MyTypeHash = FingerprintHash<MyType>;
/// using MyTypeEq = FingerprintHash<MyType>;
/// ```
///
/// which can subsequently be used elsewhere like:
///
/// ```C++
/// std::unordered_set<MyType, MyTypeHash, MyTypeEq> myObjects;
/// ```
///
/// or to make your type hashable by std::hash<MyType>(myObject) you can define:
///
/// ```C++
/// namespace std {
/// template<>
/// struct hash<caset::Coface> {
/// size_t operator()(const caset::Coface &s) const noexcept {
///      return std::hash<std::uint64_t>{}(s.fingerprint.fingerprint());
///   }
/// };
/// }
/// ```
///
/// Note that it has to be in the std:: namespace.
///
class Fingerprint {
  public:
    Fingerprint() noexcept : ids_({}), n_(0), h_(0), dirty_(false) {}

    explicit Fingerprint(const std::vector<IdType> &ids) noexcept : ids_({}), n_(0), h_(0), dirty_(true) {
      setIds(ids);
    }

    // Optimized hash mixing - constexpr for compile-time evaluation
    static inline constexpr std::uint64_t mix64(IdType x) noexcept {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    // Batch update - replaces all IDs, marks dirty
    void setIds(const std::vector<IdType> &ids) noexcept {
      n_ = 0;
      for (auto id : ids) {
        addId(id);
      }
      dirty_ = true;
    }

    // Incremental add - stages change, marks dirty
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

    // Remove ID - stages change, marks dirty
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

    // Lazy evaluation - only compute when accessed
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

    // Force immediate refresh - call after batch updates if needed
    void refresh() const noexcept {
      if (dirty_) {
        fingerprint(); // Trigger lazy evaluation
      }
    }

    // Backward compatibility
    void refreshFingerprint(const std::vector<IdType> &ids) {
      setIds(ids);
      refresh();
    }

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

    bool operator!=(const Fingerprint &o) const noexcept {
      return !(*this == o);
    }

  private:
    IdArray ids_{};              // Unique IDs (unsorted for speed)
    std::uint8_t n_{0};          // Count of unique IDs
    mutable std::uint64_t h_{0}; // Cached XOR hash - recomputed when dirty
    mutable bool dirty_{false};  // Needs recomputation?
};

template<typename T>
struct FingerprintHash {
  // using is_transparent = void; // enables heterogeneous lookup

  size_t operator()(const T &s) const noexcept { return static_cast<size_t>(s.fingerprint.fingerprint()); }
  size_t operator()(const std::shared_ptr<T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const std::shared_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(std::shared_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const T *s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(T *s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(uint64_t fp) const noexcept { return static_cast<size_t>(fp); }
};

template<typename T>
struct FingerprintEq {
  bool operator()(const T &a, const T &b) const noexcept { return a == b; }
  bool operator()(const T *a, const T *b) const noexcept {return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();}
  bool operator()(T &a, T &b) const noexcept { return a == b; }
  bool operator()(T *a, T *b) const noexcept {return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();}
  bool operator()(T* const &a, T* const &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
  bool operator()(T* &a, T* &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
};

/// These classes are for when T is a pointer
template<typename T>
struct FingerprintPtrHash {
  size_t operator()(const T &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const std::shared_ptr<T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const std::shared_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(std::shared_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const std::unique_ptr<T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const std::unique_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(std::unique_ptr<const T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(const T *s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(T *s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(uint64_t fp) const noexcept { return static_cast<size_t>(fp); }
};

template<typename T>
struct FingerprintPtrEq {
  bool operator()(const T &a, const T &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
  bool operator()(const T *a, const T *b) const noexcept {return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();}
  bool operator()(T &a, T &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
  bool operator()(T *a, T *b) const noexcept {return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();}
  bool operator()(T* const &a, T* const &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
  bool operator()(T* &a, T* &b) const noexcept { return a->fingerprint.fingerprint() == b->fingerprint.fingerprint(); }
};

} // namespace caset

#endif //CASET_VERTEXFINGERPRINT_H
