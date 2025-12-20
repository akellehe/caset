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
    IdType first{0};
    IdType second{0};

    EdgeKey(IdType sourceId_, IdType targetId_);

    bool operator==(const EdgeKey &other) const;

    [[nodiscard]] std::uint64_t hash() const;

    std::string toString() const noexcept;

    Fingerprint fingerprint;
};
} // caset

#endif //CASET_EDGEKEY_H
