// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/12/25.
//

#include "mesh/EdgeKey.h"
#include "mesh/ForwardDeclarations.h"
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;


    EdgeKey::EdgeKey(IdType sourceId_, IdType targetId_) : first(sourceId_), second(targetId_), fingerprint({sourceId_, targetId_}) {
#if TESSERA_ASSERTIONS
      if (sourceId_ == targetId_) throw std::runtime_error("You can't create a self-reference.");
#endif
    }

    bool EdgeKey::operator==(const EdgeKey &other) const {
      return first == other.first && second == other.second;
    }

    [[nodiscard]] std::uint64_t EdgeKey::hash() const {
      std::size_t h1 = std::hash<IdType>{}(first);
      std::size_t h2 = std::hash<IdType>{}(second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

#ifdef TESSERA_VERBOSE
    std::string EdgeKey::toString() const noexcept {
      return "(" + std::to_string(first) + ", " + std::to_string(second) + ")";
    }
#else
std::string EdgeKey::toString() const noexcept {
      return "";
    }
#endif


} // namespace tessera::mesh
