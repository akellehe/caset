// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/TorusOperatorTopology.h"

#include <algorithm>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SimplexBoundarySphere.h"
#include "spacetime/topologies/SimplicialProduct.h"
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

namespace {

// One geodesic (1 -> 4) subdivision of a triangulated surface, so a 3-fold
// vertex-disjoint hole triple exists (the minimal S^1 x S^1 torus has only 2).
std::vector<std::vector<std::uint64_t>> subdivideFaces(
    const std::vector<std::vector<std::uint64_t>> &faces) {
  std::uint64_t nxt = 0;
  for (const auto &f : faces)
    for (const auto v : f) nxt = std::max(nxt, v + 1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> mid;
  auto m = [&](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
    const auto key = std::make_pair(std::min(a, b), std::max(a, b));
    const auto it = mid.find(key);
    if (it != mid.end()) return it->second;
    const std::uint64_t id = nxt++;
    mid[key] = id;
    return id;
  };
  std::vector<std::vector<std::uint64_t>> out;
  for (const auto &f : faces) {
    const std::uint64_t a = f[0], b = f[1], c = f[2];
    const std::uint64_t ab = m(a, b), bc = m(b, c), ca = m(c, a);
    std::vector<std::vector<std::uint64_t>> tris = {
        {a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}};
    for (auto &s : tris) {
      std::sort(s.begin(), s.end());
      out.push_back(s);
    }
  }
  return out;
}

// The first k pairwise vertex-disjoint faces — the holonomy holes whose
// boundary circles become the qubit tori (each circle x S^1).
std::vector<std::vector<std::uint64_t>> disjointHoles(
    const std::vector<std::vector<std::uint64_t>> &faces, std::size_t k) {
  std::vector<std::vector<std::uint64_t>> holes;
  std::set<std::uint64_t> used;
  for (const auto &f : faces) {
    if (holes.size() >= k) break;
    bool overlap = false;
    for (const auto v : f)
      if (used.count(v)) { overlap = true; break; }
    if (!overlap) {
      holes.push_back(f);
      for (const auto v : f) used.insert(v);
    }
  }
  return holes;
}

}  // namespace

std::shared_ptr<Spacetime> TorusOperatorTopology::build(
    std::size_t stateDim, std::uint64_t seed,
    std::vector<std::vector<std::uint64_t>> &boundaryCells) {
  // dW = three qubit registers, each a torus (b_1 = 2). The bulk carries the
  // operator as the interior handles, ker L_1(W - dW) = d^2 - 1. Realized as
  // (subdivided T^2 with 3 holes) x S^1 = T^3 minus three (hole x S^1) solid
  // tori. REGGE (not CDT): edge lengths free, causal type emergent.
  Signature sig2(2, SignatureType::Lorentzian);
  auto metric2 = std::make_shared<Metric>(true, sig2);
  auto s1a = std::make_shared<SimplexBoundarySphere>(1);
  auto s1b = std::make_shared<SimplexBoundarySphere>(1);
  std::shared_ptr<Topology> t2 = std::make_shared<SimplicialProduct>(s1a, s1b);
  auto torus = std::make_shared<Spacetime>(metric2, SpacetimeType::REGGE, 1.0, 1.0,
                                           Foliation::NONE, t2);
  torus->build();

  const auto faces =
      subdivideFaces(ChainComplex::fromSpacetime(*torus).kSimplexVertices(2));
  const int holeCount = static_cast<int>(stateDim * stateDim) - 1;  // 3 for a qubit
  const auto holes = disjointHoles(faces, static_cast<std::size_t>(holeCount));
  if (static_cast<int>(holes.size()) < holeCount)
    throw std::runtime_error(
        "TorusOperatorTopology: torus base lacks enough vertex-disjoint holes");
  const std::set<std::vector<std::uint64_t>> holeSet(holes.begin(), holes.end());

  holes_ = holes;     // cache for readout(): the qubit boundary holes
  std::vector<std::vector<std::uint64_t>> holed;  // torus with the holes removed
  for (const auto &f : faces)
    if (!holeSet.count(f)) holed.push_back(f);
  std::uint64_t N = 0;  // per-layer vertex stride (matches Spacetime::prismCells)
  for (const auto &f : holed)
    for (const auto v : f) N = std::max(N, v + 1);
  layerStride_ = N;   // cache for readout(): the S^1 layer stride

  // (holed torus) x S^1 via the canonical prism (Spacetime::prismCells), then
  // glue the top layer (ids >= 3N) back to the bottom to close the S^1.
  std::set<std::vector<std::uint64_t>> cellSet;
  const std::uint64_t topOff = N * 3;  // three layers, stride N
  for (const auto &cell : Spacetime::prismCells(holed, /*layers=*/3)) {
    std::vector<std::uint64_t> c;
    c.reserve(cell.size());
    for (const auto v : cell) c.push_back(v >= topOff ? v - topOff : v);
    std::sort(c.begin(), c.end());
    cellSet.insert(std::move(c));
  }

  Signature sig3(3, SignatureType::Lorentzian);
  auto metric3 = std::make_shared<Metric>(true, sig3);
  auto cobordism = std::make_shared<Spacetime>(metric3, SpacetimeType::REGGE, 1.0,
                                               1.0, Foliation::NONE, std::nullopt);
  std::set<std::uint64_t> verts;
  for (const auto &c : cellSet)
    for (const auto v : c) verts.insert(v);
  std::map<std::uint64_t, ::tessera::mesh::Vertex *> vmap;
  for (const auto id : verts) vmap[id] = cobordism->createVertex(id);
  for (const auto &c : cellSet) {
    std::vector<::tessera::mesh::Vertex *> vs;
    vs.reserve(c.size());
    for (const auto v : c) vs.push_back(vmap[v]);
    cobordism->createSimplex(vs);
  }

  // Seed the metric off the degenerate uniform point. At l^2 = 1 a non-null
  // metric-Hodge eigenvalue sits at ~0, where the period-residual gradient
  // blows up; a small seeded spread breaks the degeneracy. l^2 stays positive
  // (spacelike) — the causal type is still emergent.
  std::mt19937 jrng(static_cast<std::uint32_t>(seed) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> jitter(0.7, 1.3);
  for (auto *e : cobordism->getEdgeList()->toVector())
    e->setSquaredLength(std::complex<double>(jitter(jrng), 0.0));

  // dW = the single-coface triangles: the three qubit tori.
  std::map<std::vector<std::uint64_t>, int> cofaceCount;
  for (const auto &c : cellSet)
    for (std::size_t i = 0; i < c.size(); ++i) {
      std::vector<std::uint64_t> tri;
      tri.reserve(c.size() - 1);
      for (std::size_t j = 0; j < c.size(); ++j)
        if (j != i) tri.push_back(c[j]);
      ++cofaceCount[tri];
    }
  boundaryCells.clear();
  for (const auto &kv : cofaceCount)
    if (kv.second == 1) boundaryCells.push_back(kv.first);
  return cobordism;
}

void TorusOperatorTopology::readout(
    const std::vector<std::vector<std::complex<double>>> &states,
    std::vector<EdgeLoop> &loops,
    std::vector<std::complex<double>> &targets) const {
  // Each state's two qubit components are the periods over its torus's two
  // cycles: the hole-circle (the removed face's boundary) and the S^1 (the time
  // loop of a hole vertex). The six cycles jointly over-determine the bulk b_1.
  loops.clear();
  targets.clear();
  if (holes_.size() < 3 || layerStride_ == 0) return;
  const std::uint64_t N = layerStride_;
  const std::size_t nStates = std::min(states.size(), holes_.size());
  for (std::size_t i = 0; i < nStates; ++i) {
    std::vector<std::uint64_t> h(holes_[i]);
    std::sort(h.begin(), h.end());
    const std::complex<double> a0 =
        states[i].size() > 0 ? states[i][0] : std::complex<double>(0);
    const std::complex<double> a1 =
        states[i].size() > 1 ? states[i][1] : std::complex<double>(0);
    // cycle 1: the hole-circle (removed face boundary) -> psi_i[0]
    loops.push_back({{h[0], h[1]}, {h[1], h[2]}, {h[2], h[0]}});
    targets.push_back(a0);
    // cycle 2: the S^1 (vertical loop of hole vertex h[0]) -> psi_i[1]
    loops.push_back(
        {{h[0], h[0] + N}, {h[0] + N, h[0] + 2 * N}, {h[0] + 2 * N, h[0]}});
    targets.push_back(a1);
  }
}

}  // namespace tessera::cobordism
