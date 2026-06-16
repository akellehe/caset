// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_FORWARD_DECLARATIONS_H
#define TESSERA_FORWARD_DECLARATIONS_H

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include "mesh/Fingerprint.h"
#include <optional>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Basic types
using IdType = std::uint64_t;

// Forward declarations
class Vertex;
class Edge;
class Simplex;
class EdgeList;
class VertexList;
class EdgeKey;

// Type aliases

// Edge Structures
using EdgePtr = Edge*;
using EdgeHash = FingerprintHash<Edge>;
using EdgeEq = FingerprintEq<Edge>;
using EdgeSet = std::unordered_set<Edge, EdgeHash, EdgeEq>;
using EdgePtrHash = FingerprintPtrHash<EdgePtr>;
using EdgePtrEq = FingerprintPtrEq<EdgePtr>;
using EdgePtrSet = std::unordered_set<EdgePtr, EdgePtrHash, EdgePtrEq>;
using Edges = std::vector<EdgePtr>;
using EdgePtrMap = std::unordered_map<std::uint64_t, EdgePtr>;

// Edge Key Structures
using EdgeKeyRawPtr = EdgeKey *;
using EdgeKeyHash = FingerprintHash<EdgeKey>;
using EdgeKeyEq = FingerprintEq<EdgeKey>;
using EdgeKeySet = std::unordered_set<EdgeKey, EdgeKeyHash, EdgeKeyEq>;
using EdgeKeys = std::vector<EdgeKey>;

// Simplex Structures
using SimplexRawPtr = Simplex *;
using SimplexPtr = Simplex*;
using SimplexHash = FingerprintHash<Simplex>;
using SimplexEq = FingerprintEq<Simplex>;
using SimplexPtrHash = FingerprintPtrHash<SimplexPtr>;
using SimplexPtrEq = FingerprintPtrEq<SimplexPtr>;
using SimplexSet = std::unordered_set<SimplexPtr, SimplexPtrHash, SimplexPtrEq>;

using SimplexPtrPair = std::pair<SimplexPtr, SimplexPtr>;
using OptionalSimplexPtrPair = std::optional<SimplexPtrPair>;
using Simplices = std::vector<SimplexPtr>;

using SimplexPtrSet = std::unordered_set<SimplexPtr, SimplexPtrHash, SimplexPtrEq>;
using SimplexPtrMap = std::unordered_map<SimplexPtr, Simplices, SimplexPtrHash, SimplexPtrEq>;


// Vertex Structures
using VertexPtr = Vertex*;
using VertexPtrs = std::vector<VertexPtr>;

using VertexPtrHash = FingerprintPtrHash<VertexPtr>;
using VertexPtrEq = FingerprintPtrEq<VertexPtr>;
using VertexPtrSet = std::unordered_set<VertexPtr, VertexPtrHash, VertexPtrEq>;

using VertexIndexMap = std::unordered_map<IdType, std::size_t>;
using VertexIdMap = std::unordered_map<IdType, VertexPtr>;
using VertexIdToIndex = std::unordered_map<IdType, std::size_t>;
using VertexIndexToId = std::unordered_map<std::size_t, IdType>;

// SimplexOrientation Structures
class SimplexOrientation;
using SimplexOrientationPtr = std::shared_ptr<SimplexOrientation>;
using SimplexOrientationPtrHash = FingerprintPtrHash<SimplexOrientationPtr>;
using SimplexOrientationPtrEq = FingerprintPtrEq<SimplexOrientationPtr>;
using SimplexOrientationHash = FingerprintHash<SimplexOrientation>;
using SimplexOrientationEq = FingerprintEq<SimplexOrientation>;
using SimplexOrientationSet = std::unordered_set<SimplexOrientation, SimplexOrientationHash, SimplexOrientationEq>;


} // namespace tessera::mesh

#endif // TESSERA_FORWARD_DECLARATIONS_H