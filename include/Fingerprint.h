//
// Created by Andrew Kelleher on 11/13/25.
//

#ifndef CASET_VERTEXFINGERPRINT_H
#define CASET_VERTEXFINGERPRINT_H

#include <tuple>
#include <array>
#include <vector>

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

    explicit Fingerprint(const std::vector<IdType> &ids_) : ids_({})  {
      setFingerprint(ids_);
    }

    static inline std::uint64_t mix64(IdType x) noexcept {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    static std::tuple<std::uint8_t, std::uint64_t, IdArray> computeFingerprint(
      const std::vector<IdType> &ids__) {
      if (ids__.size() > kMax) throw std::length_error("VertexFingerprint: Too many ids");
      std::uint8_t n = static_cast<std::uint8_t>(ids__.size());
      IdArray ids{};
      for (std::size_t i = 0; i < n; ++i) {
        ids[i] = ids__[i];
      }
      std::sort(ids.begin(), ids.begin() + n);
      auto it = std::unique(ids.begin(), ids.begin() + n);
      n = static_cast<std::uint8_t>(std::distance(ids.begin(), it));
      std::uint64_t h = 0xcbf29ce484222325ull ^ n;
      for (std::uint8_t i = 0; i < n; ++i) {
        h ^= mix64(ids[i] + 0x9e3779b97f4a7c15ull);
        h *= 0x100000001b3ull; // FNV-ish step
      }
      return {h, n, ids};
    }

    std::uint64_t fingerprint() const noexcept { return h_; }

    void setFingerprint(const std::vector<IdType> &ids__) {
      std::tie(h_, n_, ids_) = Fingerprint::computeFingerprint(ids__);
    }

    bool operator==(const Fingerprint &o) const noexcept {
      if (n_ != o.n_) return false;
      if (h_ != o.h_) return false; // fast reject
      return std::memcmp(ids_.data(), o.ids_.data(), n_ * sizeof(IdType)) == 0;
    }
    bool operator!=(const Fingerprint &o) const noexcept { return !(*this == o); }

  private:
    IdArray ids_{};
    std::uint8_t n_{0};
    std::uint64_t h_{kSeed};
};

template<typename T>
struct FingerprintHash {
  using is_transparent = void; // enables heterogeneous lookup
  size_t operator()(const T &s) const noexcept { return static_cast<size_t>(s.fingerprint.fingerprint()); }
  size_t operator()(const std::shared_ptr<T> &s) const noexcept { return static_cast<size_t>(s->fingerprint.fingerprint()); }
  size_t operator()(uint64_t fp) const noexcept { return static_cast<size_t>(fp); }
};

template<typename T>
struct FingerprintEq {
  using is_transparent = void;
  bool operator()(const T &a, const T &b) const noexcept { return a == b; }
  bool operator()(const T &a, uint64_t fp) const noexcept { return a.fingerprint.fingerprint() == fp; }
  bool operator()(uint64_t fp, const T &a) const noexcept { return fp == a.fingerprint.fingerprint(); }

  bool operator()(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) const noexcept {
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }
  bool operator()(const std::shared_ptr<T> &a, uint64_t fp) const noexcept { return a->fingerprint.fingerprint() == fp; }
  bool operator()(uint64_t fp, const std::shared_ptr<T> &a) const noexcept { return fp == a->fingerprint.fingerprint(); }
};


}

#endif //CASET_VERTEXFINGERPRINT_H
