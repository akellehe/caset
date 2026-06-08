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

#include "cobordism/EigenstateSynthesis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

EigenstateSynthesis::EigenstateSynthesis(std::shared_ptr<Spacetime> st, int k)
    : st_(st), k_(k), laplacian_(std::move(st)) {
  if (k_ < 0)
    throw std::runtime_error(
        "EigenstateSynthesis: degree k must be non-negative; got k=" +
        std::to_string(k_));
  if (!st_) return;
  capture();
  classifyBoundary();
}

void EigenstateSynthesis::capture() {
  order_ = 0;
  edges_.clear();
  cellOrdering_.clear();
  if (!st_) return;

  // The psi ordering: at k=0 the sorted-id vertex set (one component per vertex),
  // matching HodgeLaplacian's k=0 indexing; at k>=1 the canonical ChainComplex
  // k-cell column order, matching the metric L_k the operator assembles. order_
  // is the operator dimension N either way.
  std::unordered_set<std::uint64_t> idset;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idset.insert(v->getId());

  if (k_ == 0) {
    std::vector<std::uint64_t> ids(idset.begin(), idset.end());
    std::sort(ids.begin(), ids.end());
    cellOrdering_.reserve(ids.size());
    for (const std::uint64_t id : ids) cellOrdering_.push_back({id});
  } else {
    cellOrdering_ = ChainComplex::fromSpacetime(*st_).kSimplexVertices(k_);
  }
  order_ = cellOrdering_.size();

  // Stable edge order: the tunable edges in EdgeList order — those that actually
  // carry weight in L = D - A (both endpoints present in the vertex set, not a
  // self-loop). This is exactly HodgeLaplacian::assemble's edge filter, so the
  // {w_ij, theta_ij} we expose are the parameters the Laplacian reads.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    if (idset.find(s->getId()) == idset.end() ||
        idset.find(t->getId()) == idset.end())
      continue;
    edges_.push_back(e);
  }
}

void EigenstateSynthesis::classifyBoundary() {
  interiorEdgeIdx_.clear();
  boundaryEdgeIdx_.clear();
  boundaryVertexIdsSorted_.clear();
  interiorVertexCount_ = 0;
  if (!st_) return;

  // Top cells have d+1 vertices (d = metric dimension), matching the
  // Spacetime's own top-simplex bookkeeping (getRandomTopSimplex) so this
  // partition agrees with the pre-geometric Pachner moves growInterior reuses.
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;

  // ∂W: codim-1 faces (a top cell with one vertex dropped) belonging to exactly
  // one top cell. An *edge* sits on the boundary only once codim-1 faces are at
  // least edges themselves (topVerts >= 3); below that there is no boundary —
  // every tunable edge is interior (the free §4b regime, owned by #134).
  std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdgeKeys;
  std::unordered_set<std::uint64_t> boundaryVertexIds;
  if (topVerts >= 3) {
    std::map<std::vector<std::uint64_t>, int> facetCount;
    for (const auto s : st_->getSimplices()) {
      if (s == nullptr) continue;
      if (s->size() != topVerts) continue;
      std::vector<std::uint64_t> ids;
      ids.reserve(topVerts);
      for (const auto v : s->getVertices())
        if (v != nullptr) ids.push_back(v->getId());
      if (ids.size() != topVerts) continue;
      std::sort(ids.begin(), ids.end());
      for (std::size_t skip = 0; skip < ids.size(); ++skip) {
        std::vector<std::uint64_t> facet;
        facet.reserve(ids.size() - 1);
        for (std::size_t i = 0; i < ids.size(); ++i)
          if (i != skip) facet.push_back(ids[i]);
        ++facetCount[facet];
      }
    }
    for (const auto &[facet, count] : facetCount) {
      if (count != 1) continue;  // interior facet (shared by two top cells)
      for (const std::uint64_t id : facet) boundaryVertexIds.insert(id);
      // facet is sorted ascending, so (facet[i], facet[j]) for i<j is ordered.
      for (std::size_t i = 0; i + 1 < facet.size(); ++i)
        for (std::size_t j = i + 1; j < facet.size(); ++j)
          boundaryEdgeKeys.insert({facet[i], facet[j]});
    }
  }

  // An edge is on ∂W iff some boundary facet contains both endpoints; otherwise
  // it is interior (free).
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const std::uint64_t a = edges_[e]->getSource()->getId();
    const std::uint64_t b = edges_[e]->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key =
        a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    if (boundaryEdgeKeys.find(key) != boundaryEdgeKeys.end())
      boundaryEdgeIdx_.push_back(e);
    else
      interiorEdgeIdx_.push_back(e);
  }

  // Interior vertices: those on no boundary face (the coned-in apexes).
  std::unordered_set<std::uint64_t> seen;
  for (const auto v : st_->getVertexList()->toVector()) {
    if (v == nullptr) continue;
    if (!seen.insert(v->getId()).second) continue;
    if (boundaryVertexIds.find(v->getId()) == boundaryVertexIds.end())
      ++interiorVertexCount_;
  }

  // Persist the boundary vertex set (sorted) for boundaryVertexIds() — the
  // candidate pool a "boundary-star" connectivity search wires into.
  boundaryVertexIdsSorted_.assign(boundaryVertexIds.begin(),
                                  boundaryVertexIds.end());
  std::sort(boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
}

std::vector<cd> EigenstateSynthesis::apply(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::apply: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  std::vector<cd> out(N, cd(0.0, 0.0));
  if (N == 0) return out;
  // L_k reassembled from the live edges on each call: at k=0 the k=0 magnitude
  // convention L = D - A; at k>=1 the symmetric metric Hodge Laplacian whose
  // volume weights W_k are read live from the edge squared-lengths. The matrix
  // path does not consult the eigendecomposition cache, so repeated
  // perturb-then-query is honest.
  const std::vector<cd> L = laplacian_.laplacian(k_);
  for (std::size_t i = 0; i < N; ++i) {
    cd acc(0.0, 0.0);
    for (std::size_t j = 0; j < N; ++j) acc += L[i * N + j] * psi[j];
    out[i] = acc;
  }
  return out;
}

double EigenstateSynthesis::residual(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::residual: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  // Normalize: r and the eigenvector condition are scale-invariant, and the spec
  // writes r for a unit target.
  double nrm2 = 0.0;
  for (const cd &c : psi) nrm2 += std::norm(c);
  if (nrm2 <= 0.0) return 0.0;
  const double inv = 1.0 / std::sqrt(nrm2);
  std::vector<cd> p(N);
  for (std::size_t i = 0; i < N; ++i) p[i] = psi[i] * inv;

  const std::vector<cd> Lp = apply(p);
  // lambda = p^dagger L p (real for Hermitian L; take the real part).
  cd lam(0.0, 0.0);
  for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
  const double lambda = lam.real();

  // r = || L p - lambda p ||^2 = ||(I - p p^dagger) L p||^2 (p unit).
  double r = 0.0;
  for (std::size_t i = 0; i < N; ++i) r += std::norm(Lp[i] - lambda * p[i]);
  return r;
}

double EigenstateSynthesis::rayleigh(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::rayleigh: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  const std::vector<cd> Lp = apply(psi);
  cd num(0.0, 0.0);
  double den = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    num += std::conj(psi[i]) * Lp[i];
    den += std::norm(psi[i]);
  }
  if (den <= 0.0) return 0.0;
  return num.real() / den;
}

std::vector<double> EigenstateSynthesis::weights() const {
  std::vector<double> w;
  w.reserve(edges_.size());
  for (const auto e : edges_) w.push_back(e->getSquaredLength());
  return w;
}

std::vector<double> EigenstateSynthesis::phases() const {
  std::vector<double> th;
  th.reserve(edges_.size());
  for (const auto e : edges_) th.push_back(e->getPhase());
  return th;
}

void EigenstateSynthesis::setWeights(const std::vector<double> &w) {
  if (w.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setWeights: got " + std::to_string(w.size()) +
        " weights, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setSquaredLength(w[i]);
}

void EigenstateSynthesis::setPhases(const std::vector<double> &theta) {
  if (theta.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setPhases: got " + std::to_string(theta.size()) +
        " phases, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setPhase(theta[i]);
}

// === Fixed-boundary interior fill (§5.0) ===

std::vector<double> EigenstateSynthesis::interiorWeights() const {
  std::vector<double> w;
  w.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) w.push_back(edges_[i]->getSquaredLength());
  return w;
}

std::vector<double> EigenstateSynthesis::interiorPhases() const {
  std::vector<double> th;
  th.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) th.push_back(edges_[i]->getPhase());
  return th;
}

void EigenstateSynthesis::setInteriorWeights(const std::vector<double> &w) {
  if (w.size() != interiorEdgeIdx_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setInteriorWeights: got " +
        std::to_string(w.size()) + " weights, expected " +
        std::to_string(interiorEdgeIdx_.size()));
  for (std::size_t k = 0; k < interiorEdgeIdx_.size(); ++k)
    edges_[interiorEdgeIdx_[k]]->setSquaredLength(w[k]);
}

void EigenstateSynthesis::setInteriorPhases(const std::vector<double> &theta) {
  if (theta.size() != interiorEdgeIdx_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setInteriorPhases: got " +
        std::to_string(theta.size()) + " phases, expected " +
        std::to_string(interiorEdgeIdx_.size()));
  for (std::size_t k = 0; k < interiorEdgeIdx_.size(); ++k)
    edges_[interiorEdgeIdx_[k]]->setPhase(theta[k]);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
EigenstateSynthesis::boundaryEdges() const {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  out.reserve(boundaryEdgeIdx_.size());
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    out.emplace_back(std::min(a, b), std::max(a, b));
  }
  return out;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
EigenstateSynthesis::interiorEdges() const {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  out.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    out.emplace_back(std::min(a, b), std::max(a, b));
  }
  return out;
}

bool EigenstateSynthesis::growInterior(std::uint64_t seed) {
  if (!st_) return false;
  // Cone a fresh interior vertex via the boundary-fixed pre-geometric Pachner
  // add (#112): a 1→(d+1) stellar subdivision, always interior, so ∂W is left
  // exactly fixed (the move never touches a boundary face).
  ::tessera::spacetime::AddMove move(
      st_.get(), seed, /*relabelEnabled=*/false,
      ::tessera::spacetime::PachnerMode::PreGeometric, /*boundaryFixed=*/true);
  if (!move.propose()) return false;
  if (!move.apply()) return false;
  // The vertex set changed: rebuild the operator over the fresh vertex order
  // (the new apex has the largest id, so it appends last in sorted order and
  // the existing psi indices are preserved), then re-capture the tunable edges
  // and the interior/boundary partition.
  laplacian_ = HodgeLaplacian(st_);
  capture();
  classifyBoundary();
  return true;
}

// === Free interior connectivity (general growth primitive, #200) ===

void EigenstateSynthesis::rollbackAttachment(const Attachment &att) {
  // Remove in dependency order: simplices reference edges/vertices, edges
  // reference vertices. removeVertex also drops any edge still incident to the
  // vertex, so removing the created edges first (some — among the spec's own
  // vertices — are not incident to the new vertex) leaves only the vertex.
  for (auto *s : att.createdSimplices)
    if (s != nullptr) st_->removeSimplex(s);
  for (auto *e : att.createdEdges)
    if (e != nullptr) st_->removeEdge(e);
  if (att.vertex != nullptr) st_->removeVertex(att.vertex);
}

bool EigenstateSynthesis::attachInteriorVertex(
    const std::vector<std::vector<std::uint64_t>> &incidentSimplices) {
  if (!st_) return false;
  if (incidentSimplices.empty()) return false;  // would isolate the new vertex

  // The spec may reference only existing vertices. Build id -> Vertex* and the
  // current max id in one pass.
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idToVert;
  std::uint64_t maxId = 0;
  bool anyVert = false;
  for (const auto v : st_->getVertexList()->toVector()) {
    if (v == nullptr) continue;
    idToVert.emplace(v->getId(), v);
    maxId = anyVert ? std::max(maxId, v->getId()) : v->getId();
    anyVert = true;
  }
  if (!anyVert) return false;
  for (const auto &spec : incidentSimplices) {
    if (spec.empty()) return false;  // a face needs >= 1 existing vertex
    // spec ∪ {new vertex} is one simplex; the Fingerprint caps a simplex at kMax
    // vertices (createSimplex would otherwise throw mid-mutation).
    if (spec.size() + 1 > ::tessera::mesh::kMax) return false;
    std::unordered_set<std::uint64_t> seen;
    for (const std::uint64_t id : spec) {
      if (idToVert.find(id) == idToVert.end()) return false;  // dangling ref
      if (!seen.insert(id).second) return false;              // duplicate
    }
  }

  // Snapshot the pinned boundary (id-pair -> (w, theta)) for the bit-exact check.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::pair<double, double>>
      boundaryBefore;
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    boundaryBefore[{std::min(a, b), std::max(a, b)}] = {
        edges_[i]->getSquaredLength(), edges_[i]->getPhase()};
  }

  // Fresh interior vertex with the largest id (sorts last; preserves the
  // boundary-support psi prefix). Mirror GeometrySynthesizer::coneInVertex's
  // maxId+1 idiom rather than the vertexIdCounter, which can be stale relative
  // to explicitly-id'd fixture vertices.
  Attachment att;
  ::tessera::mesh::Vertex *vnew = st_->createVertex(maxId + 1);
  att.vertex = vnew;

  // Create one simplex per spec; createSimplexTracked materializes the simplex's
  // full 1-skeleton (every pairwise edge) and reports the freshly inserted edges
  // so detach can undo exactly.
  for (const auto &spec : incidentSimplices) {
    ::tessera::mesh::VertexPtrs verts;
    verts.reserve(spec.size() + 1);
    for (const std::uint64_t id : spec) verts.push_back(idToVert[id]);
    verts.push_back(vnew);
    const auto res = st_->createSimplexTracked(verts);
    if (res.created && res.simplex != nullptr)
      att.createdSimplices.push_back(res.simplex);
    for (const auto e : res.newEdges)
      if (e != nullptr) att.createdEdges.push_back(e);
  }

  // Re-capture so the operator / partition track the grown complex.
  laplacian_ = HodgeLaplacian(st_);
  capture();
  classifyBoundary();

  // Validate the ONLY two invariants the experiment allows.
  // (a) Valid downward-closed complex: every pair within each new simplex carries
  //     an edge (createSimplexTracked guarantees this — assert it as a real gate).
  std::set<std::pair<std::uint64_t, std::uint64_t>> edgeKeys;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeKeys.insert({std::min(a, b), std::max(a, b)});
  }
  bool valid = true;
  for (const auto &spec : incidentSimplices) {
    std::vector<std::uint64_t> ids(spec.begin(), spec.end());
    ids.push_back(vnew->getId());
    for (std::size_t i = 0; valid && i + 1 < ids.size(); ++i)
      for (std::size_t j = i + 1; valid && j < ids.size(); ++j) {
        const std::pair<std::uint64_t, std::uint64_t> key{
            std::min(ids[i], ids[j]), std::max(ids[i], ids[j])};
        if (edgeKeys.find(key) == edgeKeys.end()) valid = false;
      }
  }
  // (b) Pinned boundary dW bit-exact: same edge set, same weights/phases.
  if (valid) {
    std::size_t matched = 0;
    for (const auto i : boundaryEdgeIdx_) {
      const std::uint64_t a = edges_[i]->getSource()->getId();
      const std::uint64_t b = edges_[i]->getTarget()->getId();
      const auto it =
          boundaryBefore.find({std::min(a, b), std::max(a, b)});
      if (it == boundaryBefore.end() ||
          it->second.first != edges_[i]->getSquaredLength() ||
          it->second.second != edges_[i]->getPhase()) {
        valid = false;
        break;
      }
      ++matched;
    }
    if (matched != boundaryBefore.size()) valid = false;  // a boundary edge vanished
  }

  if (!valid) {
    rollbackAttachment(att);
    laplacian_ = HodgeLaplacian(st_);
    capture();
    classifyBoundary();
    return false;
  }
  attachments_.push_back(std::move(att));
  return true;
}

bool EigenstateSynthesis::detachLastInteriorVertex() {
  if (!st_ || attachments_.empty()) return false;
  const Attachment att = std::move(attachments_.back());
  attachments_.pop_back();
  rollbackAttachment(att);
  laplacian_ = HodgeLaplacian(st_);
  capture();
  classifyBoundary();
  return true;
}

std::vector<std::uint64_t> EigenstateSynthesis::vertexIds() const {
  std::vector<std::uint64_t> ids;
  if (!st_) return ids;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

std::vector<std::uint64_t> EigenstateSynthesis::boundaryVertexIds() const {
  return boundaryVertexIdsSorted_;
}

std::vector<std::vector<std::uint64_t>> EigenstateSynthesis::topCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (!st_) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr) continue;
    if (s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    if (ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

}  // namespace tessera::cobordism
