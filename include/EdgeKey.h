//
// Created by andrew on 12/12/25.
//

#ifndef CASET_EDGEKEY_H
#define CASET_EDGEKEY_H

#include "ForwardDeclarations.h"
#include <unordered_set>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace caset {

class EdgeKey {
  public:
    IdType first;
    IdType second;

    EdgeKey(IdType sourceId_, IdType targetId_) : first(sourceId_), second(targetId_) {
#if CASET_DEBUG
      if (sourceId_ == targetId_) throw std::runtime_error("You can't create a self-reference.");
#endif
    }

    bool operator==(const EdgeKey &other) const {
      return first == other.first && second == other.second;
    }

    [[nodiscard]] std::uint64_t hash() const {
      std::size_t h1 = std::hash<IdType>{}(first);
      std::size_t h2 = std::hash<IdType>{}(second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

    std::string toString() const {
      return "(" + std::to_string(first) + ", " + std::to_string(second) + ")";
    }
};

struct EdgeKeyHash {
  std::uint64_t operator()(const EdgeKey &p) const noexcept {
    return p.hash();
  }
};

struct EdgeKeyEqual {
  bool operator()(const EdgeKey &a,
                  const EdgeKey &b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};

using EdgeKeySet = std::unordered_set<EdgeKey, EdgeKeyHash, EdgeKeyEqual>;
using EdgeKeys = std::vector<EdgeKey>;

} // caset

#endif //CASET_EDGEKEY_H