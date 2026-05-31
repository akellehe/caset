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

#include "cobordism/Cobordism.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

using Face = std::vector<std::uint64_t>;

// Top simplices of a triangulation, as sorted vertex-id tuples. "Top" means
// maximal dimension (largest vertex count present).
SimplexList topSimplices(const Spacetime &complex) {
  SimplexList all;
  std::size_t topVertexCount = 0;
  for (const auto &simplex : complex.getSimplices()) {
    Face vertices;
    for (const auto &v : simplex->getVertices()) vertices.push_back(v->getId());
    std::sort(vertices.begin(), vertices.end());
    topVertexCount = std::max(topVertexCount, vertices.size());
    all.push_back(std::move(vertices));
  }
  SimplexList tops;
  for (auto &s : all)
    if (s.size() == topVertexCount) tops.push_back(std::move(s));
  return tops;
}

// All cardinality-(k) sub-faces of a sorted vertex tuple.
std::vector<Face> subfaces(const Face &simplex, std::size_t faceVertexCount) {
  std::vector<Face> faces;
  const std::size_t n = simplex.size();
  if (faceVertexCount > n) return faces;
  std::vector<std::size_t> idx(faceVertexCount);
  std::iota(idx.begin(), idx.end(), 0);
  for (;;) {
    Face face;
    face.reserve(faceVertexCount);
    for (auto i : idx) face.push_back(simplex[i]);
    faces.push_back(std::move(face));
    std::size_t pos = faceVertexCount;
    while (pos > 0) {
      --pos;
      if (idx[pos] != pos + n - faceVertexCount) {
        ++idx[pos];
        for (std::size_t j = pos + 1; j < faceVertexCount; ++j) idx[j] = idx[j - 1] + 1;
        break;
      }
      if (pos == 0) { pos = faceVertexCount + 1; break; }
    }
    if (pos == faceVertexCount + 1) break;
  }
  return faces;
}

// Relabel a simplex list's vertices to a dense range 0..(numVertices-1) and
// return the relabeled top-simplex set (sorted tuples) plus the vertex count.
struct Normalized {
  int numVertices{0};
  std::set<std::vector<int>> simplices{};
  int faceVertexCount{0};
};
Normalized normalize(const SimplexList &simplices) {
  std::set<std::uint64_t> verts;
  for (const auto &s : simplices)
    for (auto v : s) verts.insert(v);
  std::unordered_map<std::uint64_t, int> dense;
  int next = 0;
  for (auto v : verts) dense[v] = next++;
  Normalized out;
  out.numVertices = next;
  for (const auto &s : simplices) {
    std::vector<int> relabeled;
    relabeled.reserve(s.size());
    for (auto v : s) relabeled.push_back(dense.at(v));
    std::sort(relabeled.begin(), relabeled.end());
    out.faceVertexCount = static_cast<int>(relabeled.size());
    out.simplices.insert(std::move(relabeled));
  }
  return out;
}

}  // namespace

SimplexList Cobordism::boundaryFaces(const Spacetime &W) {
  const SimplexList tops = topSimplices(W);
  if (tops.empty()) return {};
  const std::size_t topVertexCount = tops.front().size();
  // Count how many top simplices each codimension-one face belongs to.
  std::map<Face, int> incidence;
  for (const auto &top : tops)
    for (auto &face : subfaces(top, topVertexCount - 1)) ++incidence[face];
  SimplexList boundary;
  for (auto &[face, count] : incidence)
    if (count == 1) boundary.push_back(face);
  return boundary;
}

std::vector<SimplexList> Cobordism::connectedComponents(const SimplexList &simplices) {
  const int n = static_cast<int>(simplices.size());
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  auto unite = [&](int a, int b) { parent[find(a)] = find(b); };

  // Two simplices are adjacent when they share a codimension-one facet.
  if (n > 0) {
    const std::size_t faceVertexCount = simplices.front().size();
    std::map<Face, int> firstOwner;  // facet -> a simplex index that has it
    for (int i = 0; i < n; ++i)
      for (auto &facet : subfaces(simplices[i], faceVertexCount - 1)) {
        auto it = firstOwner.find(facet);
        if (it == firstOwner.end()) firstOwner[facet] = i;
        else unite(i, it->second);
      }
  }
  std::map<int, SimplexList> groups;
  for (int i = 0; i < n; ++i) groups[find(i)].push_back(simplices[i]);
  std::vector<SimplexList> components;
  for (auto &[root, members] : groups) components.push_back(std::move(members));
  return components;
}

bool Cobordism::areIsomorphic(const SimplexList &a, const SimplexList &b) {
  const Normalized A = normalize(a);
  const Normalized B = normalize(b);
  if (A.numVertices != B.numVertices) return false;
  if (A.simplices.size() != B.simplices.size()) return false;
  if (A.faceVertexCount != B.faceVertexCount) return false;
  if (A.numVertices == 0) return true;  // both empty

  const int nv = A.numVertices;
  // Per-vertex degree = number of top simplices containing it.
  auto degrees = [nv](const Normalized &x) {
    std::vector<int> deg(nv, 0);
    for (const auto &s : x.simplices)
      for (int v : s) ++deg[v];
    return deg;
  };
  const std::vector<int> degA = degrees(A);
  const std::vector<int> degB = degrees(B);
  {
    std::vector<int> sa = degA, sb = degB;
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    if (sa != sb) return false;
  }

  // Which A-simplices contain each A-vertex (for incremental constraint checks).
  std::vector<std::vector<const std::vector<int> *>> aSimplicesOf(nv);
  for (const auto &s : A.simplices)
    for (int v : s) aSimplicesOf[v].push_back(&s);

  // Assign A-vertices (most-constrained first: highest degree) to B-vertices.
  std::vector<int> order(nv);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int x, int y) { return degA[x] > degA[y]; });

  std::vector<int> mapAtoB(nv, -1);
  std::vector<char> usedB(nv, 0);

  std::function<bool(int)> backtrack = [&](int pos) -> bool {
    if (pos == nv) return true;
    const int u = order[pos];
    for (int w = 0; w < nv; ++w) {
      if (usedB[w] || degB[w] != degA[u]) continue;
      mapAtoB[u] = w;
      usedB[w] = 1;
      bool consistent = true;
      // Every A-simplex through u whose vertices are all assigned must map to a
      // B-simplex.
      for (const auto *simplex : aSimplicesOf[u]) {
        std::vector<int> image;
        image.reserve(simplex->size());
        bool fullyAssigned = true;
        for (int v : *simplex) {
          if (mapAtoB[v] < 0) { fullyAssigned = false; break; }
          image.push_back(mapAtoB[v]);
        }
        if (!fullyAssigned) continue;
        std::sort(image.begin(), image.end());
        if (!B.simplices.count(image)) { consistent = false; break; }
      }
      if (consistent && backtrack(pos + 1)) return true;
      mapAtoB[u] = -1;
      usedB[w] = 0;
    }
    return false;
  };
  return backtrack(0);
}

CobordismResult Cobordism::verify(const Spacetime &W, const Spacetime &M1,
                                  const Spacetime &M2) {
  const SimplexList boundary = boundaryFaces(W);
  const std::vector<SimplexList> components = connectedComponents(boundary);

  // The boundary of a manifold-with-boundary is itself closed: every
  // codimension-one face of the boundary complex must belong to exactly two
  // boundary simplices (boundary-of-boundary is empty).
  if (!boundary.empty()) {
    const std::size_t faceVertexCount = boundary.front().size();
    if (faceVertexCount >= 1) {
      std::map<Face, int> incidence;
      for (const auto &simplex : boundary)
        for (auto &facet : subfaces(simplex, faceVertexCount - 1)) ++incidence[facet];
      for (auto &[facet, count] : incidence)
        if (count != 2)
          return {false, CobordismCheck::BoundaryChainNotClosed,
                  "boundary is not a closed manifold (a codimension-one face of "
                  "the boundary belongs to " + std::to_string(count) +
                  " boundary simplices, expected 2)"};
    }
  }

  // The expected boundary pieces are the non-empty manifolds among {M1, M2}.
  std::vector<SimplexList> expected;
  if (auto t = topSimplices(M1); !t.empty()) expected.push_back(std::move(t));
  if (auto t = topSimplices(M2); !t.empty()) expected.push_back(std::move(t));

  if (components.size() != expected.size())
    return {false, CobordismCheck::WrongNumberOfBoundaryComponents,
            "boundary has " + std::to_string(components.size()) +
            " connected component(s), but M1 ⊔ M2 has " +
            std::to_string(expected.size()) + " non-empty manifold(s)"};

  // Match boundary components to the expected manifolds (each used once).
  std::vector<char> expectedUsed(expected.size(), 0);
  for (const auto &component : components) {
    bool matched = false;
    for (std::size_t e = 0; e < expected.size(); ++e) {
      if (expectedUsed[e]) continue;
      if (areIsomorphic(component, expected[e])) {
        expectedUsed[e] = 1;
        matched = true;
        break;
      }
    }
    if (!matched)
      return {false, CobordismCheck::BoundaryNotIsomorphic,
              "a boundary component is not isomorphic to M1 or M2"};
  }
  return {true, CobordismCheck::Ok, "valid cobordism (boundary structure)"};
}

}  // namespace tessera::cobordism
