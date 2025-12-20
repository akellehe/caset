//
// Created by andrew on 12/12/25.
//

#include "EdgeKey.h"
#include "ForwardDeclarations.h"
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace caset {


    EdgeKey::EdgeKey(IdType sourceId_, IdType targetId_) : first(sourceId_), second(targetId_), fingerprint({sourceId_, targetId_}) {
#if CASET_ASSERTIONS
      if (sourceId_ == targetId_) throw std::runtime_error("You can't create a self-reference.");
#endif
      fingerprint = Fingerprint(fingerprint);
    }

    bool EdgeKey::operator==(const EdgeKey &other) const {
      return first == other.first && second == other.second;
    }

    [[nodiscard]] std::uint64_t EdgeKey::hash() const {
      std::size_t h1 = std::hash<IdType>{}(first);
      std::size_t h2 = std::hash<IdType>{}(second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

    std::string EdgeKey::toString() const noexcept {
#ifdef CASET_VERBOSE
        return "(" + std::to_string(first) + ", " + std::to_string(second) + ")";
#else
        return "";
#endif
    }
} // caset
