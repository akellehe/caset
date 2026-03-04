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
#include "Fingerprint.h"
#include <optional>

namespace caset {

// Basic types
using IdType = std::uint64_t;

// Forward declarations
template<int D> class Vertex;
template<int D> class Edge;

template<int D> class Simplex;
template<int D> class EdgeList;
template<int D> class VertexList;
template<int D> class EdgeKey;

// Type aliases

// Edge Structures
template<int D>
using EdgePtr = std::shared_ptr<Edge<D>>;
template<int D>
using EdgeHash = FingerprintHash<Edge<D>>;
template<int D>
using EdgeEq = FingerprintEq<Edge<D>>;
template<int D>
using EdgeSet = std::unordered_set<Edge<D>, EdgeHash<D>, EdgeEq<D>>;
template<int D>
using EdgePtrHash = FingerprintPtrHash<EdgePtr<D>>;
template<int D>
using EdgePtrEq = FingerprintPtrEq<EdgePtr<D>>;
template<int D>
using EdgePtrSet = std::unordered_set<EdgePtr<D>, EdgePtrHash<D>, EdgePtrEq<D>>;
template<int D>
using Edges = std::vector<EdgePtr<D>>;
template<int D>
using EdgePtrMap = std::unordered_map<std::uint64_t, EdgePtr<D>>;

// Edge Key Structures
template<int D>
using EdgeKeyRawPtr = EdgeKey<D> *;
template<int D>
using EdgeKeyHash = FingerprintHash<EdgeKey<D>>;
template<int D>
using EdgeKeyEq = FingerprintEq<EdgeKey<D>>;
template<int D>
using EdgeKeySet = std::unordered_set<EdgeKey<D>, EdgeKeyHash<D>, EdgeKeyEq<D>>;
template<int D>
using EdgeKeys = std::vector<EdgeKey<D>>;

// Simplex Structures
template<int D>
using SimplexRawPtr = Simplex<D> *;

template<int D>
using SimplexPtr = std::shared_ptr<Simplex<D>>;

template<int D>
using SimplexHash = FingerprintHash<Simplex<D>>;
template<int D>
using SimplexEq = FingerprintEq<Simplex<D>>;
template<int D>
using SimplexSet = std::unordered_set<std::shared_ptr<Simplex<D>>, SimplexHash<D>, SimplexEq<D>>;

template<int D>
using SimplexPtrPair = std::pair<SimplexPtr<D>, SimplexPtr<D>>;
template<int D>
using OptionalSimplexPtrPair = std::optional<SimplexPtrPair<D>>;
template<int D>
using Simplices = std::vector<SimplexPtr<D>>;

template<int D>
using SimplexPtrHash = FingerprintPtrHash<SimplexPtr<D>>;

template<int D>
using SimplexPtrEq = FingerprintPtrEq<SimplexPtr<D>>;

template<int D>
using SimplexPtrSet = std::unordered_set<SimplexPtr<D>, SimplexPtrHash<D>, SimplexPtrEq<D>>;

template<int D>
using SimplexPtrMap = std::unordered_map<SimplexPtr<D>, Simplices<D>, SimplexPtrHash<D>, SimplexPtrEq<D>>;

template<int D>
using SimplexUniquePtr = std::unique_ptr<Simplex<D>>;
template<int D>
using SimplexUniquePtrHash = FingerprintPtrHash<SimplexUniquePtr<D>>;
template<int D>
using SimplexUniquePtrEq = FingerprintPtrEq<SimplexUniquePtr<D>>;
template<int D>
using SimplexUniquePtrSet = std::unordered_set<SimplexUniquePtr<D>, SimplexUniquePtrHash<D>, SimplexUniquePtrEq<D>>;


// Vertex Structures
template<int D>
using VertexPtr = std::shared_ptr<Vertex<D>>;
template<int D>
using VertexPtrs = std::vector<VertexPtr<D>>;

template<int D>
using VertexPtrHash = FingerprintPtrHash<VertexPtr<D>>;
template<int D>
using VertexPtrEq = FingerprintPtrEq<VertexPtr<D>>;
template<int D>
using VertexPtrSet = std::unordered_set<VertexPtr<D>, VertexPtrHash<D>, VertexPtrEq<D>>;

using VertexIndexMap = std::unordered_map<IdType, std::size_t>;
template<int D>
using VertexIdMap = std::unordered_map<IdType, VertexPtr<D>>;
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

template<int D>
class Spacetime;

// using EdgeHash = FingerprintHash<Edge>;
// using EdgeEq = FingerprintEq<Edge>;
// using Edges = std::vector<EdgePtr>;

// using EdgeKey = std::pair<IdType, IdType>;
// using EdgeIdSet = std::unordered_set<EdgeKey, EdgeKeyHash, EdgeKeyEqual>;
// using EdgeIds = std::vector<EdgeKey>;

// using SimplexPtr = std::shared_ptr<Simplex>;
// using SimplexPair = std::pair<SimplexPtr, SimplexPtr>;
// using OptionalSimplexPair = std::optional<SimplexPair>;
// using Simplices = std::vector<SimplexPtr>;
// using SimplexSet = std::unordered_set<SimplexPtr, SimplexHash, SimplexEq>;
} // namespace caset

#endif // CASET_FORWARD_DECLARATIONS_H