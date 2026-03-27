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

#ifndef CASET_FORWARD_DECLARATIONS_H
#define CASET_FORWARD_DECLARATIONS_H

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include "mesh/Fingerprint.h"
#include <optional>

namespace caset {

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

class Spacetime;

} // namespace caset

#endif // CASET_FORWARD_DECLARATIONS_H