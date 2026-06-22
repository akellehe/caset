// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/S3WindowSurface.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace tessera::cobordism {

namespace {

using Cell = S3WindowSurface::Cell;

// The join of two K-cycles C_K * C_K: a clean triangulated S^3 with 2K vertices
// (a_i = i, b_j = K + j) and K^2 top tetrahedra {a_i, a_{i+1 mod K}, b_j,
// b_{j+1 mod K}}. The two core circles are Hopf-linked; the rotations of either
// factor are simplicial automorphisms (the join is cyclically symmetric, unlike a
// shuffle product). Betti [1,0,0,1]. Deterministic.
std::vector<Cell> joinOfCycles(std::uint64_t K) {
  std::vector<Cell> faces;
  faces.reserve(static_cast<std::size_t>(K) * K);
  for (std::uint64_t i = 0; i < K; ++i)
    for (std::uint64_t j = 0; j < K; ++j) {
      Cell t = {i, (i + 1) % K, K + j, K + (j + 1) % K};
      std::sort(t.begin(), t.end());
      faces.push_back(std::move(t));
    }
  return faces;
}

// The hole tetrahedron seated at the diagonal grid position p (its a- and b-pair
// both start at p): {a_p, a_{p+1}, b_p, b_{p+1}} — a genuine join top cell.
Cell holeAt(std::uint64_t p, std::uint64_t K) {
  Cell t = {p % K, (p + 1) % K, K + p % K, K + (p + 1) % K};
  std::sort(t.begin(), t.end());
  return t;
}

}  // namespace

S3WindowSurface::Surface S3WindowSurface::build(int windowCount, int granularity) {
  if (windowCount < 1)
    throw std::invalid_argument(
        "S3WindowSurface: windowCount must be >= 1 (four matches the W_ABC "
        "A,B,C,R structure; one is the minimal color register)");
  if (granularity < 1)
    throw std::invalid_argument(
        "S3WindowSurface: granularity must be >= 1 (the lattice-refinement "
        "factor; larger refines the triangulation between the disjoint holes)");

  // K is a multiple of 6: divisible by 3 for the color Z_3 = tau^{K/3}, and the
  // even spacing keeps the windowCount windows' 3*windowCount holes vertex-disjoint.
  const std::uint64_t K = static_cast<std::uint64_t>(6) *
                          static_cast<std::uint64_t>(windowCount) *
                          static_cast<std::uint64_t>(granularity);
  const std::uint64_t colorStep = K / 3;  // the color-Z_3 generator sigma = tau^{K/3}

  auto faces = joinOfCycles(K);
  const std::set<Cell> faceSet(faces.begin(), faces.end());

  // Window w: the sigma-orbit (sigma = tau^{K/3}, order 3) of a seed hole at the
  // diagonal position 2w. sigma cyclically permutes the three color holes (the
  // color Z_3); tau^2 maps window w -> window w+1, so the windows are themselves
  // one symmetry orbit (not a hand-placed set). Asserts: each window a genuine Z_3
  // orbit of base tetrahedra, all holes vertex-disjoint.
  std::vector<std::vector<Cell>> windows(static_cast<std::size_t>(windowCount));
  std::set<std::uint64_t> used;
  for (int w = 0; w < windowCount; ++w) {
    const std::uint64_t seed = static_cast<std::uint64_t>(2 * w);
    Cell h0 = holeAt(seed, K);
    Cell h1 = holeAt(seed + colorStep, K);
    Cell h2 = holeAt(seed + 2 * colorStep, K);
    if (holeAt(seed + 3 * colorStep, K) != h0)
      throw std::runtime_error(
          "S3WindowSurface: color window is not a Z_3 orbit");
    for (const Cell &h : {h0, h1, h2}) {
      if (!faceSet.count(h))
        throw std::runtime_error(
            "S3WindowSurface: color hole is not a base tetrahedron");
      for (const auto v : h)
        if (!used.insert(v).second)
          throw std::runtime_error(
              "S3WindowSurface: color windows are not vertex-disjoint");
      windows[static_cast<std::size_t>(w)].push_back(h);
    }
  }

  return Surface{std::move(faces), std::move(windows)};
}

}  // namespace tessera::cobordism
