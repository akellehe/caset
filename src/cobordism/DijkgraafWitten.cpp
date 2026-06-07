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

#include "cobordism/DijkgraafWitten.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "cobordism/Cobordism.h"
#include "cobordism/IntegerLinalg.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

DijkgraafWitten::DijkgraafWitten(std::shared_ptr<Spacetime> W, Cocycle w)
    : W_(std::move(W)), cocycle_(w) {}

std::complex<double> DijkgraafWitten::omega(Cocycle w, int a, int b, int c) {
  switch (w) {
    case Cocycle::Trivial:
      return {1.0, 0.0};
    case Cocycle::Sign:
      // (-1)^{abc}: the sign flips only when all three edge values are 1.
      return ((a & b & c) & 1) ? std::complex<double>{-1.0, 0.0}
                               : std::complex<double>{1.0, 0.0};
  }
  return {1.0, 0.0};  // unreachable; silences a maybe-no-return warning
}

bool DijkgraafWitten::isCocycle(Cocycle w) {
  // Normalized 3-cocycle (pentagon) identity over Z_2, written multiplicatively:
  //   ω(b,c,d) · ω(a, b⊕c, d) · ω(a,b,c)  ==  ω(a⊕b, c, d) · ω(a, b, c⊕d)
  // must hold for every (a,b,c,d) ∈ Z_2^4 (⊕ is addition mod 2). Brute-force all
  // 16 tuples. This is the prerequisite that makes the state sum gauge-invariant
  // (hence a topological invariant), so the partition function relies on it.
  const double tolerance = 1e-12;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c)
        for (int d = 0; d < 2; ++d) {
          const std::complex<double> left =
              omega(w, b, c, d) * omega(w, a, b ^ c, d) * omega(w, a, b, c);
          const std::complex<double> right =
              omega(w, a ^ b, c, d) * omega(w, a, b, c ^ d);
          if (std::abs(left - right) > tolerance) return false;
        }
  return true;
}

std::complex<double> DijkgraafWitten::partitionFunction() const {
  if (W_ == nullptr)
    throw std::runtime_error(
        "DijkgraafWitten::partitionFunction: the spacetime is null");

  const ChainComplex chain = ChainComplex::fromSpacetime(*W_);
  if (chain.dimension() != 3)
    throw std::runtime_error(
        "DijkgraafWitten::partitionFunction: a closed oriented 3-manifold is "
        "required (dimension must be 3)");

  const int numVertices = static_cast<int>(chain.numSimplices(0));
  const int numEdges = static_cast<int>(chain.numSimplices(1));
  const int numTriangles = static_cast<int>(chain.numSimplices(2));

  // The flat Z_2 connections g ∈ C^1 are ker(d_1 mod 2), where the coboundary
  // d_1 = ∂_2^T acts on edge-cochains. ∂_2 = boundaryMatrix(2) is
  // rows = |C_1| (edges) × cols = |C_2| (triangles); its transpose d_1 is
  // triangles × edges. A flat g (length numEdges) satisfies, on every triangle,
  // (d_1 g)|_△ = g_01 + g_12 - g_02 ≡ 0 (mod 2).
  const std::vector<long> &boundaryTwo = chain.boundaryMatrix(2);
  std::vector<int> coboundaryOne(
      static_cast<std::size_t>(numTriangles) * numEdges, 0);
  for (int edge = 0; edge < numEdges; ++edge)
    for (int triangle = 0; triangle < numTriangles; ++triangle)
      coboundaryOne[static_cast<std::size_t>(triangle) * numEdges + edge] =
          static_cast<int>(
              boundaryTwo[static_cast<std::size_t>(edge) * numTriangles +
                          triangle] &
              1L);

  const std::vector<std::vector<int>> flatBasis =
      gf2Nullspace(coboundaryOne, numTriangles, numEdges);

  // Orientation / flatness verification: every basis cocycle must satisfy
  // d_1 · g ≡ 0 (mod 2) on each triangle ((dg)|_△ = g_01 + g_12 - g_02 ≡ 0).
  // This holds by construction of the nullspace; assert it so a mis-transposed
  // coboundary or inconsistent edge ordering is caught loudly rather than
  // silently producing a wrong invariant.
  for (const std::vector<int> &basisVector : flatBasis)
    for (int triangle = 0; triangle < numTriangles; ++triangle) {
      int parity = 0;
      for (int edge = 0; edge < numEdges; ++edge)
        parity ^=
            coboundaryOne[static_cast<std::size_t>(triangle) * numEdges + edge] &
            basisVector[static_cast<std::size_t>(edge)];
      if (parity != 0)
        throw std::runtime_error(
            "DijkgraafWitten::partitionFunction: a flat-connection basis vector "
            "violates d_1·g ≡ 0 (mod 2) — the coboundary/edge indexing is "
            "inconsistent");
    }

  const std::vector<std::vector<int>> flatConnections =
      gf2Span(flatBasis, numEdges);

  // Map a sorted vertex pair (v_i, v_j) to its C_1 (edge) index. The edge
  // ordering kSimplexVertices(1) returns is exactly the row order of ∂_2 (both
  // are faceVerts_[1]), so g — indexed by ∂_2's rows — is read consistently.
  const std::vector<std::vector<std::uint64_t>> edges = chain.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> edgeIndex;
  for (int e = 0; e < static_cast<int>(edges.size()); ++e)
    edgeIndex[{edges[e][0], edges[e][1]}] = e;

  // Each tetrahedron's ordered (sorted) vertices and its orientation sign ε_t,
  // both indexed in the column order of ∂_3 (orientedTopSimplices()).
  const std::vector<std::vector<std::uint64_t>> tetrahedra =
      chain.orientedTopSimplices();
  const std::vector<int> epsilon = chain.fundamentalClass();

  // Z(W) = (1/2^{|V|}) Σ_{flat g} Π_t ω(g_01, g_12, g_23)^{ε_t}.
  std::complex<double> total{0.0, 0.0};
  for (const std::vector<int> &g : flatConnections) {
    std::complex<double> weight{1.0, 0.0};
    for (int t = 0; t < static_cast<int>(tetrahedra.size()); ++t) {
      const std::vector<std::uint64_t> &v = tetrahedra[t];  // v0 < v1 < v2 < v3
      const int g01 = g[static_cast<std::size_t>(edgeIndex.at({v[0], v[1]}))];
      const int g12 = g[static_cast<std::size_t>(edgeIndex.at({v[1], v[2]}))];
      const int g23 = g[static_cast<std::size_t>(edgeIndex.at({v[2], v[3]}))];
      std::complex<double> tetWeight = omega(cocycle_, g01, g12, g23);
      // ω^{ε_t}: ε_t = +1 leaves it, ε_t = -1 inverts it. For a U(1) phase the
      // inverse is the conjugate (a no-op for the real ±1 values of Trivial and
      // Sign, but kept so the formula is faithful for any U(1) cocycle).
      if (epsilon[static_cast<std::size_t>(t)] < 0) tetWeight = std::conj(tetWeight);
      weight *= tetWeight;
    }
    total += weight;
  }
  total /= std::pow(2.0, static_cast<double>(numVertices));
  return total;
}

std::complex<double> DijkgraafWitten::partitionFunctionByCohomology() const {
  if (W_ == nullptr)
    throw std::runtime_error(
        "DijkgraafWitten::partitionFunctionByCohomology: the spacetime is null");

  const ChainComplex chain = ChainComplex::fromSpacetime(*W_);
  if (chain.dimension() != 3)
    throw std::runtime_error(
        "DijkgraafWitten::partitionFunctionByCohomology: a closed oriented "
        "3-manifold is required (dimension must be 3)");

  const int numEdges = static_cast<int>(chain.numSimplices(1));
  const int numTriangles = static_cast<int>(chain.numSimplices(2));

  // Cocycles Z^1 = ker(d_1 mod 2), d_1 = ∂_2^T (triangles × edges) — the flat
  // ℤ₂ connections, exactly as in partitionFunction().
  const std::vector<long> &boundaryTwo = chain.boundaryMatrix(2);
  std::vector<int> coboundaryOne(
      static_cast<std::size_t>(numTriangles) * numEdges, 0);
  for (int edge = 0; edge < numEdges; ++edge)
    for (int triangle = 0; triangle < numTriangles; ++triangle)
      coboundaryOne[static_cast<std::size_t>(triangle) * numEdges + edge] =
          static_cast<int>(
              boundaryTwo[static_cast<std::size_t>(edge) * numTriangles +
                          triangle] &
              1L);
  const std::vector<std::vector<int>> cocycleBasis =
      gf2Nullspace(coboundaryOne, numTriangles, numEdges);

  // Map a sorted vertex pair (v_i, v_j) to its C_1 (edge) index, as in
  // partitionFunction() — the row order of ∂_2 / kSimplexVertices(1).
  const std::vector<std::vector<std::uint64_t>> edges = chain.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> edgeIndex;
  for (int e = 0; e < numEdges; ++e) edgeIndex[{edges[e][0], edges[e][1]}] = e;

  // Coboundaries B^1 = im(d_0): each vertex's edge-incidence vector (mod 2 it is
  // d_0 of the vertex indicator — an elementary gauge transformation).
  std::vector<std::vector<int>> coboundaryBasis;
  for (const std::vector<std::uint64_t> &vertex : chain.kSimplexVertices(0)) {
    const std::uint64_t id = vertex[0];
    std::vector<int> incidence(numEdges, 0);
    for (int e = 0; e < numEdges; ++e)
      if (edges[e][0] == id || edges[e][1] == id) incidence[e] = 1;
    coboundaryBasis.push_back(std::move(incidence));
  }

  // H^1(W;ℤ₂) = Z^1/B^1: a complement basis of representatives, enumerated to the
  // 2^{b_1} cohomology classes (one representative per gauge class).
  const std::vector<std::vector<int>> cohomologyBasis =
      gf2CohomologyBasis(cocycleBasis, coboundaryBasis, numEdges);
  const std::vector<std::vector<int>> classes =
      gf2Span(cohomologyBasis, numEdges);

  // The DW weight ∏_t ω(g_01,g_12,g_23)^{ε_t} on a connection g, read exactly as
  // in partitionFunction() (same edge map, same orientation signs ε_t from the
  // fundamental class); for the real ℤ₂ cocycles ε_t is a no-op but kept faithful.
  const std::vector<std::vector<std::uint64_t>> tetrahedra =
      chain.orientedTopSimplices();
  const std::vector<int> epsilon = chain.fundamentalClass();
  auto weightOf = [&](const std::vector<int> &g) {
    std::complex<double> weight{1.0, 0.0};
    for (int t = 0; t < static_cast<int>(tetrahedra.size()); ++t) {
      const std::vector<std::uint64_t> &v = tetrahedra[t];  // v0 < v1 < v2 < v3
      const int g01 = g[static_cast<std::size_t>(edgeIndex.at({v[0], v[1]}))];
      const int g12 = g[static_cast<std::size_t>(edgeIndex.at({v[1], v[2]}))];
      const int g23 = g[static_cast<std::size_t>(edgeIndex.at({v[2], v[3]}))];
      std::complex<double> tetWeight = omega(cocycle_, g01, g12, g23);
      if (epsilon[static_cast<std::size_t>(t)] < 0) tetWeight = std::conj(tetWeight);
      weight *= tetWeight;
    }
    return weight;
  };

  // Z(W) = (1/2^{b_0}) Σ_{[g] ∈ H^1} weight([g]). Per class, the weight is
  // computed once for the sum and re-checked against every gauge shift g ⊕ c
  // (c an elementary coboundary): a genuine 3-cocycle weight is constant on a
  // class, so a mismatch flags a non-cocycle ω or inconsistent edge indexing —
  // the same loud guard partitionFunction() applies to flatness.
  const double tolerance = 1e-9;
  std::complex<double> total{0.0, 0.0};
  for (const std::vector<int> &g : classes) {
    const std::complex<double> weight = weightOf(g);
    for (const std::vector<int> &c : coboundaryBasis) {
      std::vector<int> shifted(g.size());
      for (std::size_t e = 0; e < g.size(); ++e)
        shifted[e] = (g[e] ^ c[static_cast<std::size_t>(e)]) & 1;
      if (std::abs(weightOf(shifted) - weight) > tolerance)
        throw std::runtime_error(
            "DijkgraafWitten::partitionFunctionByCohomology: the weight is not "
            "constant on a cohomology class (gauge-shift mismatch) — ω is not a "
            "3-cocycle or the coboundary/edge indexing is inconsistent");
    }
    total += weight;
  }

  // b_0 = dim H^0 = number of connected components: the residual gauge volume
  // that makes the cohomology-class sum equal the 2^{-|V|} Σ_{flat g} brute force
  // (|B^1| = 2^{|V|-b_0} cocycles collapse to each class).
  const int b0 = chain.bettiNumbersGF2()[0];
  total /= std::pow(2.0, static_cast<double>(b0));
  return total;
}

DijkgraafWitten::StateSumTerms DijkgraafWitten::stateSumTerms() const {
  if (W_ == nullptr)
    throw std::runtime_error(
        "DijkgraafWitten::stateSumTerms: the spacetime is null");

  const ChainComplex chain = ChainComplex::fromSpacetime(*W_);
  if (chain.dimension() != 3)
    throw std::runtime_error(
        "DijkgraafWitten::stateSumTerms: a 3-manifold (dimension 3) is required");

  // dim Z^1 = b_1 + dim B^1 = b_1 + (|V| - b_0): the brute-force path enumerates
  // all 2^{dim Z^1} flat cocycles, the cohomology path only the 2^{b_1} classes.
  const std::vector<int> bettiGF2 = chain.bettiNumbersGF2();
  const int b0 = bettiGF2[0];
  const int b1 = bettiGF2[1];
  const int numVertices = static_cast<int>(chain.numSimplices(0));
  const int dimZ1 = b1 + numVertices - b0;

  StateSumTerms terms;
  terms.firstBetti = b1;
  terms.cocycleDimension = dimZ1;
  terms.cohomologyTerms = std::pow(2.0, static_cast<double>(b1));
  terms.bruteForceTerms = std::pow(2.0, static_cast<double>(dimZ1));
  terms.speedup = std::pow(2.0, static_cast<double>(dimZ1 - b1));
  return terms;
}

DijkgraafWitten::Boundary DijkgraafWitten::computeBoundary() const {
  if (W_ == nullptr)
    throw std::runtime_error(
        "DijkgraafWitten boundary state sum: the spacetime is null");

  const ChainComplex chain = ChainComplex::fromSpacetime(*W_);
  if (chain.dimension() != 3)
    throw std::runtime_error(
        "DijkgraafWitten boundary state sum: a 3-manifold (dimension 3) is "
        "required");

  const int numEdges = static_cast<int>(chain.numSimplices(1));
  const int numTriangles = static_cast<int>(chain.numSimplices(2));

  // --- GF(2) helpers (echelon basis of a span + canonical coset residue) -----
  // `echelon` returns a row-echelon basis (each row's leftmost 1 — its pivot —
  // is unique and sits left of nothing else), sorted by pivot column. `residue`
  // reduces a vector against such a basis to the canonical representative of its
  // coset modulo the span (supported on the non-pivot columns); two vectors are
  // equal modulo the span iff their residues match. Both are linear over GF(2).
  auto echelon = [](std::vector<std::vector<int>> gens, int cols) {
    std::vector<std::vector<int>> rows;
    std::vector<int> pivots;
    for (std::vector<int> g : gens) {
      for (int &c : g) c &= 1;
      for (std::size_t r = 0; r < rows.size(); ++r)
        if (g[static_cast<std::size_t>(pivots[r])])
          for (int c = pivots[r]; c < cols; ++c)
            g[static_cast<std::size_t>(c)] ^= rows[r][static_cast<std::size_t>(c)];
      int p = -1;
      for (int c = 0; c < cols; ++c)
        if (g[static_cast<std::size_t>(c)]) { p = c; break; }
      if (p < 0) continue;  // dependent on the rows already collected
      std::size_t pos = 0;
      while (pos < pivots.size() && pivots[pos] < p) ++pos;
      rows.insert(rows.begin() + static_cast<long>(pos), std::move(g));
      pivots.insert(pivots.begin() + static_cast<long>(pos), p);
    }
    return rows;
  };
  auto residue = [](std::vector<int> v,
                    const std::vector<std::vector<int>> &rows) {
    for (const std::vector<int> &row : rows) {
      int p = -1;
      for (int c = 0; c < static_cast<int>(row.size()); ++c)
        if (row[static_cast<std::size_t>(c)]) { p = c; break; }
      if (p >= 0 && (v[static_cast<std::size_t>(p)] & 1))
        for (int c = p; c < static_cast<int>(v.size()); ++c)
          v[static_cast<std::size_t>(c)] ^= row[static_cast<std::size_t>(c)];
    }
    for (int &c : v) c &= 1;
    return v;
  };
  // Representatives of H^1 = Z^1 / B^1 (the b_1 cocycles independent modulo the
  // coboundaries) come from IntegerLinalg::gf2CohomologyBasis; `echelon` and
  // `residue` above remain for the per-component class-index lookup below.

  // --- bulk: enumerate the gauge classes [g] ∈ H^1(W; ℤ₂) --------------------
  // Flat connections Z^1(W) = ker(d_1 mod 2), d_1 = ∂_2^T (triangles × edges);
  // coboundaries B^1(W) = im(d_0), each vertex's edge-incidence vector. The
  // 2^{b_1(W)} classes (materialized via gf2Span) are the interior fields modulo
  // gauge — small, so no gauge-redundant 2^{|V|+b_1} enumeration is needed.
  const std::vector<long> &boundaryTwo = chain.boundaryMatrix(2);
  std::vector<int> coboundaryOne(
      static_cast<std::size_t>(numTriangles) * numEdges, 0);
  for (int edge = 0; edge < numEdges; ++edge)
    for (int triangle = 0; triangle < numTriangles; ++triangle)
      coboundaryOne[static_cast<std::size_t>(triangle) * numEdges + edge] =
          static_cast<int>(
              boundaryTwo[static_cast<std::size_t>(edge) * numTriangles +
                          triangle] &
              1L);
  const std::vector<std::vector<int>> cocycleBasis =
      gf2Nullspace(coboundaryOne, numTriangles, numEdges);

  const std::vector<std::vector<std::uint64_t>> edges = chain.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> edgeIndex;
  for (int e = 0; e < numEdges; ++e) edgeIndex[{edges[e][0], edges[e][1]}] = e;

  std::vector<std::vector<int>> coboundaryBasis;  // B^1(W) generators
  for (const std::vector<std::uint64_t> &vertex : chain.kSimplexVertices(0)) {
    const std::uint64_t id = vertex[0];
    std::vector<int> incidence(numEdges, 0);
    for (int e = 0; e < numEdges; ++e)
      if (edges[e][0] == id || edges[e][1] == id) incidence[e] = 1;
    coboundaryBasis.push_back(std::move(incidence));
  }
  const std::vector<std::vector<int>> bulkReps =
      gf2CohomologyBasis(cocycleBasis, coboundaryBasis, numEdges);
  const std::vector<std::vector<int>> bulkClasses = gf2Span(bulkReps, numEdges);

  // --- boundary ∂W: components, each a closed surface Σ_i --------------------
  const SimplexList boundaryTriangles = Cobordism::boundaryFaces(*W_);
  if (boundaryTriangles.empty())
    throw std::runtime_error(
        "DijkgraafWitten boundary state sum: W is closed (∂W is empty); use "
        "partitionFunction() for the scalar invariant");
  std::vector<SimplexList> components =
      Cobordism::connectedComponents(boundaryTriangles);
  for (SimplexList &component : components)
    std::sort(component.begin(), component.end());
  std::sort(components.begin(), components.end());  // deterministic Σ_A, Σ_B, …

  // Per component: the class-indexing data for its DW Hilbert space Z(Σ_i) =
  // ℂ[H^1(Σ_i; ℤ₂)] (dimension 2^{b_1(Σ_i)}). A boundary connection's index is
  // the position (in gf2Span order) of its H^1(Σ_i) class.
  struct ComponentIndexer {
    std::vector<int> localToGlobalEdge;            // local edge → bulk edge index
    std::vector<std::vector<int>> coboundaryRows;  // echelon B^1(Σ_i)
    std::map<std::vector<int>, int> classOfResidue;
    int dimension{1};
  };
  std::vector<ComponentIndexer> indexers;
  indexers.reserve(components.size());
  for (const SimplexList &component : components) {
    std::set<std::pair<std::uint64_t, std::uint64_t>> edgeSet;
    std::set<std::uint64_t> vertexSet;
    for (const std::vector<std::uint64_t> &triangle : component) {
      vertexSet.insert(triangle[0]);
      vertexSet.insert(triangle[1]);
      vertexSet.insert(triangle[2]);
      edgeSet.insert({triangle[0], triangle[1]});
      edgeSet.insert({triangle[0], triangle[2]});
      edgeSet.insert({triangle[1], triangle[2]});
    }
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> localEdges(
        edgeSet.begin(), edgeSet.end());
    const int numLocalEdges = static_cast<int>(localEdges.size());
    std::map<std::pair<std::uint64_t, std::uint64_t>, int> localEdgeIndex;
    for (int i = 0; i < numLocalEdges; ++i) localEdgeIndex[localEdges[i]] = i;

    // d_1 on Σ_i (triangles × local edges) — incidence; ℤ₂ flatness is its kernel.
    std::vector<int> localCoboundary(
        static_cast<std::size_t>(component.size()) * numLocalEdges, 0);
    for (int t = 0; t < static_cast<int>(component.size()); ++t) {
      const std::vector<std::uint64_t> &tri = component[static_cast<std::size_t>(t)];
      const std::pair<std::uint64_t, std::uint64_t> triEdges[3] = {
          {tri[0], tri[1]}, {tri[0], tri[2]}, {tri[1], tri[2]}};
      for (const auto &pr : triEdges)
        localCoboundary[static_cast<std::size_t>(t) * numLocalEdges +
                        localEdgeIndex[pr]] = 1;
    }
    const std::vector<std::vector<int>> localCocycles = gf2Nullspace(
        localCoboundary, static_cast<int>(component.size()), numLocalEdges);
    std::vector<std::vector<int>> localCoboundaryBasis;
    for (std::uint64_t v : vertexSet) {
      std::vector<int> incidence(numLocalEdges, 0);
      for (int i = 0; i < numLocalEdges; ++i)
        if (localEdges[static_cast<std::size_t>(i)].first == v ||
            localEdges[static_cast<std::size_t>(i)].second == v)
          incidence[i] = 1;
      localCoboundaryBasis.push_back(std::move(incidence));
    }
    const std::vector<std::vector<int>> localReps =
        gf2CohomologyBasis(localCocycles, localCoboundaryBasis, numLocalEdges);
    const std::vector<std::vector<int>> localClasses =
        gf2Span(localReps, numLocalEdges);

    ComponentIndexer indexer;
    indexer.coboundaryRows = echelon(localCoboundaryBasis, numLocalEdges);
    indexer.dimension = static_cast<int>(localClasses.size());
    indexer.localToGlobalEdge.resize(static_cast<std::size_t>(numLocalEdges));
    for (int i = 0; i < numLocalEdges; ++i)
      indexer.localToGlobalEdge[static_cast<std::size_t>(i)] =
          edgeIndex.at(localEdges[static_cast<std::size_t>(i)]);
    for (int c = 0; c < static_cast<int>(localClasses.size()); ++c)
      indexer.classOfResidue[residue(localClasses[static_cast<std::size_t>(c)],
                                     indexer.coboundaryRows)] = c;
    indexers.push_back(std::move(indexer));
  }

  // --- bin each bulk class by its boundary-restriction class, weight Π ω ------
  // For the real ℤ₂ cocycles (Trivial ≡ 1, Sign = ±1) the orientation exponent
  // ε_t is immaterial (x^{±1} = x, ω^{-1} = ω̄ = ω), so the product matches the
  // closed case with no (relative) fundamental class. Summing over cohomology
  // classes gives the gauge-fixed amplitude; the trivial cobordism comes out the
  // identity with no normalization factor.
  std::vector<int> dims;
  long totalSize = 1;
  for (const ComponentIndexer &indexer : indexers) {
    dims.push_back(indexer.dimension);
    totalSize *= indexer.dimension;
  }
  std::vector<std::complex<double>> amplitudes(
      static_cast<std::size_t>(totalSize), {0.0, 0.0});
  const std::vector<std::vector<std::uint64_t>> tetrahedra =
      chain.orientedTopSimplices();
  for (const std::vector<int> &g : bulkClasses) {
    long jointIndex = 0;
    for (const ComponentIndexer &indexer : indexers) {
      std::vector<int> local(indexer.localToGlobalEdge.size());
      for (std::size_t i = 0; i < local.size(); ++i)
        local[i] = g[static_cast<std::size_t>(indexer.localToGlobalEdge[i])];
      const int classIndex =
          indexer.classOfResidue.at(residue(local, indexer.coboundaryRows));
      jointIndex = jointIndex * indexer.dimension + classIndex;
    }
    std::complex<double> weight{1.0, 0.0};
    for (const std::vector<std::uint64_t> &v : tetrahedra) {
      const int g01 = g[static_cast<std::size_t>(edgeIndex.at({v[0], v[1]}))];
      const int g12 = g[static_cast<std::size_t>(edgeIndex.at({v[1], v[2]}))];
      const int g23 = g[static_cast<std::size_t>(edgeIndex.at({v[2], v[3]}))];
      weight *= omega(cocycle_, g01, g12, g23);
    }
    amplitudes[static_cast<std::size_t>(jointIndex)] += weight;
  }
  return {std::move(dims), std::move(amplitudes)};
}

std::vector<std::complex<double>> DijkgraafWitten::boundaryVector() const {
  return computeBoundary().amplitudes;
}

std::vector<int> DijkgraafWitten::boundaryDimensions() const {
  return computeBoundary().dims;
}

std::vector<std::vector<std::complex<double>>> DijkgraafWitten::map() const {
  const Boundary boundary = computeBoundary();
  if (boundary.dims.size() != 2)
    throw std::runtime_error(
        "DijkgraafWitten::map: ∂W must have exactly two components for a map "
        "Z(Σ_B) → Z(Σ_A); got " + std::to_string(boundary.dims.size()));
  const int rows = boundary.dims[0];
  const int cols = boundary.dims[1];
  std::vector<std::vector<std::complex<double>>> matrix(
      static_cast<std::size_t>(rows),
      std::vector<std::complex<double>>(static_cast<std::size_t>(cols)));
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      matrix[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
          boundary.amplitudes[static_cast<std::size_t>(r) * cols + c];
  return matrix;
}

std::complex<double> DijkgraafWitten::amplitude(
    const std::vector<std::complex<double>> &psiA,
    const std::vector<std::complex<double>> &psiB) const {
  const Boundary boundary = computeBoundary();
  if (boundary.dims.size() != 2)
    throw std::runtime_error(
        "DijkgraafWitten::amplitude: ∂W must have exactly two components "
        "(⟨ψ_A| Z(W) |ψ_B⟩ needs Σ_A and Σ_B)");
  const int rows = boundary.dims[0];
  const int cols = boundary.dims[1];
  if (static_cast<int>(psiA.size()) != rows ||
      static_cast<int>(psiB.size()) != cols)
    throw std::invalid_argument(
        "DijkgraafWitten::amplitude: state lengths must match the map "
        "dimensions (|ψ_A| = 2^{b_1(Σ_A)}, |ψ_B| = 2^{b_1(Σ_B)})");
  std::complex<double> accumulator{0.0, 0.0};
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      accumulator += std::conj(psiA[static_cast<std::size_t>(r)]) *
                     boundary.amplitudes[static_cast<std::size_t>(r) * cols + c] *
                     psiB[static_cast<std::size_t>(c)];
  return accumulator;
}

}  // namespace tessera::cobordism
