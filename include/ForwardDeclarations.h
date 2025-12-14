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
#include <cstdint>
#include "Fingerprint.h"

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
using VertexPtr = std::shared_ptr<Vertex>;
using VertexPtrs = std::vector<VertexPtr>;
using VertexPtrSet = std::unordered_set<VertexPtr>;

using EdgeRawPtr = Edge *;
using EdgeHash = FingerprintHash<Edge>;
using EdgeEq = FingerprintEq<Edge>;
using Edges = std::vector<EdgeRawPtr>;
using EdgeSet = std::unordered_set<EdgeRawPtr, EdgeHash, EdgeEq>;

using SimplexRawPtr = Simplex *;
using SimplexHash = FingerprintHash<Simplex>;
using SimplexEq = FingerprintEq<Simplex>;
using SimplexSet = std::unordered_set<SimplexRawPtr, SimplexHash, SimplexEq>;

} // namespace caset

#endif // CASET_FORWARD_DECLARATIONS_H