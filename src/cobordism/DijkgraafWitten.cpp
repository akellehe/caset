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

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
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

}  // namespace tessera::cobordism
