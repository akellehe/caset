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
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
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
// return the relabeled top-simplex set (sorted tuples), the vertex count, and
// the dense→real id map (denseToReal[k] is the k-th smallest real vertex id, so
// the dense labels are assigned in increasing-id order — this is what makes the
// vertex bijection below order-preserving).
struct Normalized {
  int numVertices{0};
  std::set<std::vector<int>> simplices{};
  int faceVertexCount{0};
  std::vector<std::uint64_t> denseToReal{};
};
Normalized normalize(const SimplexList &simplices) {
  std::set<std::uint64_t> verts;
  for (const auto &s : simplices)
    for (auto v : s) verts.insert(v);
  std::unordered_map<std::uint64_t, int> dense;
  int next = 0;
  Normalized out;
  for (auto v : verts) {
    dense[v] = next++;
    out.denseToReal.push_back(v);  // verts is sorted, so this is increasing
  }
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

// A vertex bijection (dense A-label → dense B-label) witnessing that two
// normalized triangulations are isomorphic, or nullopt if none exists. The
// search assigns the most-constrained (highest-degree) vertices first, breaking
// degree ties by ascending dense label and trying targets in ascending order;
// for two triangulations that are equal after normalization this finds the
// identity permutation first, so the correspondence it returns is
// order-preserving on shift-related copies (e.g. the two ends of a product
// cylinder). The backtracking is the same one `areIsomorphic` reports a bool for.
std::optional<std::vector<int>> vertexBijection(const Normalized &A,
                                                const Normalized &B) {
  if (A.numVertices != B.numVertices) return std::nullopt;
  if (A.simplices.size() != B.simplices.size()) return std::nullopt;
  if (A.faceVertexCount != B.faceVertexCount) return std::nullopt;
  if (A.numVertices == 0) return std::vector<int>{};  // both empty

  const int nv = A.numVertices;
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
    if (sa != sb) return std::nullopt;
  }

  std::vector<std::vector<const std::vector<int> *>> aSimplicesOf(nv);
  for (const auto &s : A.simplices)
    for (int v : s) aSimplicesOf[v].push_back(&s);

  std::vector<int> order(nv);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    if (degA[x] != degA[y]) return degA[x] > degA[y];
    return x < y;  // ascending-label tie-break → order-preserving preference
  });

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
  if (!backtrack(0)) return std::nullopt;
  return mapAtoB;
}

// The vertex correspondence (real `from` id → real `to` id) of the
// order-preserving isomorphism between two same-dimensional triangulations, or
// nullopt if they are not isomorphic.
std::optional<std::map<std::uint64_t, std::uint64_t>> vertexCorrespondence(
    const SimplexList &from, const SimplexList &to) {
  const Normalized F = normalize(from);
  const Normalized T = normalize(to);
  const std::optional<std::vector<int>> bijection = vertexBijection(F, T);
  if (!bijection) return std::nullopt;
  std::map<std::uint64_t, std::uint64_t> correspondence;
  for (int a = 0; a < F.numVertices; ++a)
    correspondence[F.denseToReal[static_cast<std::size_t>(a)]] =
        T.denseToReal[static_cast<std::size_t>((*bijection)[static_cast<std::size_t>(a)])];
  return correspondence;
}

// Build a fresh pre-geometric Spacetime from an explicit combinatorial
// description: `numVertices` coordinate-free vertices (ids 0..numVertices-1) and
// one top simplex per (dense) vertex-id tuple. Mirrors Topology::buildExplicit,
// reached here through the public Spacetime API so the gluing constructors need
// not be Topology subclasses — geometry is irrelevant to the homological /
// state-sum computations these complexes feed.
std::shared_ptr<Spacetime> buildSpacetime(std::size_t numVertices,
                                          const SimplexList &topSimplices) {
  // Match the spacetime's signature dimension to the glued triangulation's
  // top-cell dimension (top vertex count − 1). ``getBoundary`` — reached
  // downstream through ``verify`` / a subsequent ``glue`` — reads the top set
  // off ``topSimplicesVec``, which ``registerSimplex`` keys to the signature's
  // d+1. The default ``Spacetime()`` pins ``Signature(4)``, which would
  // silently leave ``topSimplicesVec`` (and hence the boundary) empty for a
  // glued 3-manifold. An empty input falls back to a harmless default.
  std::size_t topVertexCount = 0;
  for (const auto &simplex : topSimplices)
    topVertexCount = std::max(topVertexCount, simplex.size());
  const int d = topVertexCount > 0 ? static_cast<int>(topVertexCount) - 1 : 4;
  auto metric =
      std::make_shared<Metric>(true, Signature(d, SignatureType::Lorentzian));
  auto spacetime = std::make_shared<Spacetime>(
      metric, SpacetimeType::CDT, 1.0, 1.0, Foliation::PREFERRED, std::nullopt);
  std::vector<VertexPtr> verts;
  verts.reserve(numVertices);
  for (std::size_t i = 0; i < numVertices; ++i)
    verts.push_back(spacetime->createVertex(static_cast<std::uint64_t>(i)));
  for (const auto &simplex : topSimplices) {
    VertexPtrs sv;
    sv.reserve(simplex.size());
    for (auto id : simplex) sv.push_back(verts.at(static_cast<std::size_t>(id)));
    spacetime->createSimplex(sv);
  }
  return spacetime;
}

// Boundary surfaces of W as connected components, in the deterministic order
// (each component's faces sorted, then components sorted) that map()/
// boundaryVector() also use — so component i here is component i there.
std::vector<SimplexList> sortedBoundaryComponents(const Spacetime &W) {
  std::vector<SimplexList> components =
      Cobordism::connectedComponents(Cobordism::boundaryFaces(W));
  for (SimplexList &component : components)
    std::sort(component.begin(), component.end());
  std::sort(components.begin(), components.end());
  return components;
}

}  // namespace

SimplexList Cobordism::boundaryFaces(const Spacetime &W) {
  // Thin wrapper over the canonical, Spacetime-owned boundary derivation
  // (#162). ``Spacetime::getBoundary`` does the same facet-counting from the
  // top simplices (codimension-one faces with incidence == 1), returning the
  // identical sorted vertex-id tuples. Kept as a static entry point for the
  // existing cobordism callers (``verify``, ``connectedComponents`` consumers,
  // ``DijkgraafWitten``) and their tests.
  return W.getBoundary();
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
  // Existence of a vertex relabeling carrying a's simplex set onto b's: the
  // order-preserving search in vertexBijection (shared with glue()/selfGlue(),
  // which need the actual correspondence and not just the yes/no).
  return vertexBijection(normalize(a), normalize(b)).has_value();
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

std::shared_ptr<Spacetime> Cobordism::glue(const Spacetime &W1,
                                           const Spacetime &W2) {
  const SimplexList tops1 = topSimplices(W1);
  const SimplexList tops2 = topSimplices(W2);
  if (tops1.empty() || tops2.empty())
    throw std::invalid_argument(
        "Cobordism::glue: both inputs must be non-empty triangulations");
  if (tops1.front().size() != tops2.front().size())
    throw std::invalid_argument(
        "Cobordism::glue: the two complexes must have the same top dimension");

  // Σ_C = the first isomorphic (W1 component, W2 component) pair; corr maps the
  // W2 copy's vertex ids onto the W1 copy's, identifying the shared surface.
  const std::vector<SimplexList> components1 = sortedBoundaryComponents(W1);
  const std::vector<SimplexList> components2 = sortedBoundaryComponents(W2);
  std::optional<std::map<std::uint64_t, std::uint64_t>> correspondence;
  int sharedComponent2 = -1;
  for (std::size_t i = 0; i < components1.size() && !correspondence; ++i)
    for (std::size_t j = 0; j < components2.size(); ++j) {
      std::optional<std::map<std::uint64_t, std::uint64_t>> candidate =
          vertexCorrespondence(components2[j], components1[i]);
      if (candidate) {
        correspondence = std::move(candidate);
        sharedComponent2 = static_cast<int>(j);
        break;
      }
    }
  if (!correspondence)
    throw std::invalid_argument(
        "Cobordism::glue: no shared (isomorphic) boundary surface Σ_C between "
        "the two cobordisms");

  std::set<std::uint64_t> sharedW2;  // W2 vertices lying on Σ_C
  for (const std::vector<std::uint64_t> &face :
       components2[static_cast<std::size_t>(sharedComponent2)])
    for (std::uint64_t v : face) sharedW2.insert(v);

  // Dense reindexing: W1's vertices first (in id order), then W2's non-shared
  // vertices; each shared W2 vertex collapses onto its W1 partner.
  std::set<std::uint64_t> vertices1;
  for (const auto &top : tops1) for (std::uint64_t v : top) vertices1.insert(v);
  std::map<std::uint64_t, std::uint64_t> dense1;
  std::uint64_t next = 0;
  for (std::uint64_t v : vertices1) dense1[v] = next++;

  std::set<std::uint64_t> vertices2;
  for (const auto &top : tops2) for (std::uint64_t v : top) vertices2.insert(v);
  std::map<std::uint64_t, std::uint64_t> dense2;
  for (std::uint64_t v : vertices2)
    dense2[v] = sharedW2.count(v) ? dense1.at(correspondence->at(v)) : next++;
  const std::size_t numVertices = next;

  SimplexList merged;
  merged.reserve(tops1.size() + tops2.size());
  for (std::vector<std::uint64_t> top : tops1) {
    for (std::uint64_t &v : top) v = dense1.at(v);
    std::sort(top.begin(), top.end());
    merged.push_back(std::move(top));
  }
  for (std::vector<std::uint64_t> top : tops2) {
    for (std::uint64_t &v : top) v = dense2.at(v);
    std::sort(top.begin(), top.end());
    merged.push_back(std::move(top));
  }
  return buildSpacetime(numVertices, merged);
}

std::shared_ptr<Spacetime> Cobordism::selfGlue(const Spacetime &W) {
  const SimplexList tops = topSimplices(W);
  if (tops.empty())
    throw std::invalid_argument(
        "Cobordism::selfGlue: the input must be a non-empty triangulation");
  const std::vector<SimplexList> components = sortedBoundaryComponents(W);
  if (components.size() != 2)
    throw std::invalid_argument(
        "Cobordism::selfGlue: ∂W must have exactly two components to glue to "
        "each other; got " + std::to_string(components.size()));
  // Identify component 1 onto component 0 by the order-preserving isomorphism.
  const std::optional<std::map<std::uint64_t, std::uint64_t>> correspondence =
      vertexCorrespondence(components[1], components[0]);
  if (!correspondence)
    throw std::invalid_argument(
        "Cobordism::selfGlue: the two boundary components are not isomorphic");

  std::set<std::uint64_t> identified;  // component-1 vertices (folded into 0)
  for (const std::vector<std::uint64_t> &face : components[1])
    for (std::uint64_t v : face) identified.insert(v);

  // Dense ids for the kept vertices (everything except component 1); each
  // component-1 vertex takes the id of its component-0 partner.
  std::set<std::uint64_t> allVertices;
  for (const auto &top : tops) for (std::uint64_t v : top) allVertices.insert(v);
  std::map<std::uint64_t, std::uint64_t> dense;
  std::uint64_t next = 0;
  for (std::uint64_t v : allVertices)
    if (!identified.count(v)) dense[v] = next++;
  for (std::uint64_t v : identified)
    dense[v] = dense.at(correspondence->at(v));
  const std::size_t numVertices = next;

  SimplexList merged;
  merged.reserve(tops.size());
  for (std::vector<std::uint64_t> top : tops) {
    for (std::uint64_t &v : top) v = dense.at(v);
    std::sort(top.begin(), top.end());
    if (std::adjacent_find(top.begin(), top.end()) != top.end())
      throw std::runtime_error(
          "Cobordism::selfGlue: the identification collapsed a top simplex (a "
          "top simplex touches both boundary components — the collar is too "
          "thin; thicken the glued direction to at least three layers)");
    merged.push_back(std::move(top));
  }
  return buildSpacetime(numVertices, merged);
}

std::shared_ptr<Spacetime> Cobordism::twistedCylinder(
    const Spacetime &sigma, const std::vector<std::uint64_t> &phi) {
  // Σ's top cells must be triangles (a 2-dimensional surface).
  const SimplexList triangles = topSimplices(sigma);
  if (triangles.empty())
    throw std::invalid_argument(
        "Cobordism::twistedCylinder: Σ must be a non-empty triangulation");
  for (const std::vector<std::uint64_t> &triangle : triangles)
    if (triangle.size() != 3)
      throw std::invalid_argument(
          "Cobordism::twistedCylinder: Σ must be a 2-dimensional surface (its "
          "top cells must be triangles)");

  // The vertices must be exactly 0..n-1: φ is supplied as a flat per-id vector
  // and the level offsets below assume a dense [0, n) id range (the
  // product-torus fixtures number their vertices this way).
  std::set<std::uint64_t> vertexSet;
  for (const std::vector<std::uint64_t> &triangle : triangles)
    for (std::uint64_t v : triangle) vertexSet.insert(v);
  const std::uint64_t n = static_cast<std::uint64_t>(vertexSet.size());
  if (vertexSet.empty() || *vertexSet.rbegin() != n - 1)
    throw std::invalid_argument(
        "Cobordism::twistedCylinder: Σ's vertices must be exactly 0..|V|-1");

  // φ must be a length-n permutation of 0..n-1 ...
  if (static_cast<std::uint64_t>(phi.size()) != n)
    throw std::invalid_argument(
        "Cobordism::twistedCylinder: phi must have one entry per Σ vertex "
        "(length |V|)");
  const std::set<std::uint64_t> phiImage(phi.begin(), phi.end());
  if (static_cast<std::uint64_t>(phiImage.size()) != n ||
      *phiImage.rbegin() != n - 1)
    throw std::invalid_argument(
        "Cobordism::twistedCylinder: phi must be a permutation of 0..|V|-1");
  // ... and a simplicial automorphism: every top triangle's φ-image is a
  // triangle (this is what makes the two prisms induce one shared seam, and what
  // limits the realizable maps to the finite-order modular elements).
  const std::set<std::vector<std::uint64_t>> triangleSet(triangles.begin(),
                                                         triangles.end());
  for (const std::vector<std::uint64_t> &triangle : triangles) {
    std::vector<std::uint64_t> image{phi[triangle[0]], phi[triangle[1]],
                                     phi[triangle[2]]};
    std::sort(image.begin(), image.end());
    if (!triangleSet.count(image))
      throw std::invalid_argument(
          "Cobordism::twistedCylinder: phi is not a simplicial automorphism of "
          "Σ (a top triangle's image is not a triangle)");
  }

  // Three stacked copies of Σ: level ℓ vertex x at id ℓ·n + x. Each triangle
  // (a < b < c) spans a prism (a 2-simplex × [0,1]) cut into the three
  // Eilenberg–Zilber tetrahedra {β(a),β(b),β(c),τ(c)}, {β(a),β(b),τ(b),τ(c)},
  // {β(a),τ(a),τ(b),τ(c)} for a lower-face map β and an upper-face map τ — the
  // same staircase SimplicialProduct lays down for Σ×[0,1], the diagonal fixed
  // by Σ's vertex order so adjacent prisms (and the seam) glue consistently.
  auto addPrism = [](SimplexList &tops, const std::vector<std::uint64_t> &tri,
                     auto beta, auto tau) {
    const std::uint64_t a = tri[0], b = tri[1], c = tri[2];
    auto push = [&tops](std::vector<std::uint64_t> tet) {
      std::sort(tet.begin(), tet.end());
      tops.push_back(std::move(tet));
    };
    push({beta(a), beta(b), beta(c), tau(c)});
    push({beta(a), beta(b), tau(b), tau(c)});
    push({beta(a), tau(a), tau(b), tau(c)});
  };

  const auto level0 = [](std::uint64_t x) { return x; };
  const auto level1 = [n](std::uint64_t x) { return n + x; };
  const auto level2 = [n](std::uint64_t x) { return 2 * n + x; };
  // The seam (level 1) read through φ: the level-1 copy of vertex φ(x) carries
  // x's slot, so the monodromy from the bottom boundary (level 0) up to the top
  // boundary (level 2) is exactly φ — the whole twist lives in this one map.
  const auto seam = [n, &phi](std::uint64_t x) { return n + phi[x]; };

  SimplexList tops;
  tops.reserve(triangles.size() * 6);
  for (const std::vector<std::uint64_t> &triangle : triangles) {
    addPrism(tops, triangle, level0, level1);  // bottom prism Σ×[0,1] (identity)
    addPrism(tops, triangle, seam, level2);    // top prism Σ×[1,2] (φ-threaded)
  }
  return buildSpacetime(static_cast<std::size_t>(3 * n), tops);
}

}  // namespace tessera::cobordism
