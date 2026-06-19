// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/EigenstateSynthesis.h"

#include <Eigen/Dense>

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

#ifdef TESSERA_CUDA
#include "cuda/eigenstate_cuda.h"
#endif

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
  for (const auto e : edges_) w.push_back(e->getSquaredLength().real());
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
    edges_[i]->setSquaredLength(std::complex<double>{w[i], 0.0});
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
  for (const auto i : interiorEdgeIdx_)
    w.push_back(edges_[i]->getSquaredLength().real());
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
    edges_[interiorEdgeIdx_[k]]->setSquaredLength(std::complex<double>{w[k], 0.0});
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
        edges_[i]->getSquaredLength().real(), edges_[i]->getPhase()};
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
          it->second.first != edges_[i]->getSquaredLength().real() ||
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

// === Surgery: the topology-changing interior remove move (#196) ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::interiorTopCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (!st_) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    bool allInterior = true;
    for (const auto v : s->getVertices()) {
      if (v == nullptr) { allInterior = false; break; }
      ids.push_back(v->getId());
      if (bverts.find(v->getId()) != bverts.end()) allInterior = false;
    }
    if (!allInterior || ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

bool EigenstateSynthesis::removeInteriorCell(
    const std::vector<std::uint64_t> &cell) {
  if (!st_) return false;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts < 2 || cell.size() != topVerts) return false;
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());

  // The cell must be interior: no boundary vertex (so no ∂W face is removed).
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const std::uint64_t id : want)
    if (bverts.find(id) != bverts.end()) return false;

  // Locate the matching top simplex, and collect the OTHER top cells' vertex
  // sets (to tell which of `want`'s edges remain covered after removal). Refuse
  // to remove the last top cell of the top dimension (it would drop the complex
  // dimension and promote orphan facets to top cells).
  ::tessera::mesh::Simplex *target = nullptr;
  std::vector<std::vector<std::uint64_t>> otherTop;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    if (target == nullptr && ids == want)
      target = s;
    else
      otherTop.push_back(std::move(ids));
  }
  if (target == nullptr || otherTop.empty()) return false;

  // An edge {u,v} of the cell is orphaned iff no other top cell contains both
  // endpoints. Map endpoint pairs -> Edge* for the orphaned ones.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  const auto covered = [&](std::uint64_t u, std::uint64_t v) {
    for (const auto &c : otherTop) {
      const bool hu = std::find(c.begin(), c.end(), u) != c.end();
      const bool hv = std::find(c.begin(), c.end(), v) != c.end();
      if (hu && hv) return true;
    }
    return false;
  };

  Removal rem;
  rem.cell = want;
  std::vector<::tessera::mesh::Edge *> toRemove;
  for (std::size_t i = 0; i + 1 < want.size(); ++i)
    for (std::size_t j = i + 1; j < want.size(); ++j) {
      const std::uint64_t u = want[i];
      const std::uint64_t v = want[j];
      if (covered(u, v)) continue;  // edge survives in another top cell
      const auto it = edgeByPair.find({u, v});
      if (it == edgeByPair.end()) continue;  // already absent
      rem.removedEdges.emplace_back(
          u, v, it->second->getSquaredLength().real(),
          it->second->getPhase());
      toRemove.push_back(it->second);
    }

  // Snapshot ∂W (id-pair -> (w, theta)) for the bit-exact check.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::pair<double, double>>
      boundaryBefore;
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    boundaryBefore[{std::min(a, b), std::max(a, b)}] = {
        edges_[i]->getSquaredLength().real(), edges_[i]->getPhase()};
  }

  // Mutate: drop the top cell, then its orphaned edges.
  st_->removeSimplex(target);
  for (auto *e : toRemove)
    if (e != nullptr) st_->removeEdge(e);

  laplacian_ = HodgeLaplacian(st_);
  capture();
  classifyBoundary();

  // ∂W must be preserved bit-exactly: every previously-boundary edge still
  // present with the same weight/phase (newly EXPOSED boundary edges are allowed
  // — the opened hole — so this is a subset check, not equality).
  bool valid = true;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::pair<double, double>>
      liveWeights;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key{std::min(a, b),
                                                      std::max(a, b)};
    liveWeights[key] = {e->getSquaredLength().real(), e->getPhase()};
  }
  for (const auto &[key, wp] : boundaryBefore) {
    const auto it = liveWeights.find(key);
    if (it == liveWeights.end() || it->second.first != wp.first ||
        it->second.second != wp.second) {
      valid = false;
      break;
    }
  }

  if (!valid) {
    applyRestore(rem);
    laplacian_ = HodgeLaplacian(st_);
    capture();
    classifyBoundary();
    return false;
  }
  removals_.push_back(std::move(rem));
  return true;
}

bool EigenstateSynthesis::applyRestore(const Removal &rem) {
  if (!st_) return false;
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idToVert;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idToVert.emplace(v->getId(), v);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(rem.cell.size());
  for (const std::uint64_t id : rem.cell) {
    const auto it = idToVert.find(id);
    if (it == idToVert.end()) return false;  // a cell vertex vanished
    verts.push_back(it->second);
  }
  // Re-create the top cell; createSimplexTracked rebuilds exactly the missing
  // edges (the orphaned ones removed above), leaving surviving edges untouched.
  st_->createSimplexTracked(verts);
  // Restore the removed edges' weights/phases bit-exactly.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  for (const auto &[u, v, w, theta] : rem.removedEdges) {
    const auto it = edgeByPair.find({std::min(u, v), std::max(u, v)});
    if (it != edgeByPair.end()) {
      it->second->setSquaredLength(std::complex<double>{w, 0.0});
      it->second->setPhase(theta);
    }
  }
  return true;
}

bool EigenstateSynthesis::restoreLastRemoval() {
  if (!st_ || removals_.empty()) return false;
  const Removal rem = std::move(removals_.back());
  removals_.pop_back();
  applyRestore(rem);
  laplacian_ = HodgeLaplacian(st_);
  capture();
  classifyBoundary();
  return true;
}

// === Gated topology moves: the checked cut and the composed stellar move ===

std::pair<bool, std::string> EigenstateSynthesis::removeInteriorCellChecked(
    const std::vector<std::uint64_t> &cell) {
  if (!removeInteriorCell(cell))
    return {false,
            "not an interior top cell (or the removal would touch dW)"};
  const auto verdict = dualComplexValid();  // {true, "ok"} when the dual holds
  if (!verdict.first) restoreLastRemoval();
  return verdict;
}

std::pair<bool, std::string> EigenstateSynthesis::stellarSubdivideInterior(
    const std::vector<std::uint64_t> &cell) {
  if (!st_) return {false, "no spacetime"};
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts < 2 || cell.size() != topVerts)
    return {false, "cell is not a top cell (" + std::to_string(cell.size()) +
                       " vertices, expected " + std::to_string(topVerts) + ")"};

  // The facet fan: `cell` with each vertex dropped in turn (its d+1 codim-one
  // facets). The attach cones the fresh vertex onto every facet, so the parent
  // cell's 1-skeleton survives its removal (each parent edge keeps a fan
  // coface) and the subdivision is exactly 1 -> (d+1).
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());
  std::vector<std::vector<std::uint64_t>> fan;
  fan.reserve(want.size());
  for (std::size_t skip = 0; skip < want.size(); ++skip) {
    std::vector<std::uint64_t> facet;
    facet.reserve(want.size() - 1);
    for (std::size_t i = 0; i < want.size(); ++i)
      if (i != skip) facet.push_back(want[i]);
    fan.push_back(std::move(facet));
  }

  if (!attachInteriorVertex(fan))
    return {false, "attach rejected (invalid fan spec, or dW perturbed)"};
  if (!removeInteriorCell(want)) {
    detachLastInteriorVertex();
    return {false,
            "not an interior top cell (or the removal would touch dW)"};
  }
  const auto verdict = dualComplexValid();
  if (!verdict.first) {
    // LIFO rollback: the removal happened after the attach.
    restoreLastRemoval();
    detachLastInteriorVertex();
    return verdict;
  }

  // The uniform re-pin: the seeds are built with every edge at squared length 1
  // and phase 0 (the unit cochain metric), and the move must hold that by
  // construction, not by the time-rule coincidence on all-same-time seeds.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    e->setLength({1.0, 0.0});  // spacelike unit length
    e->setPhase(0.0);
  }
  return verdict;  // {true, "ok"}
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

std::pair<bool, std::string> EigenstateSynthesis::dualComplexValid() const {
  const auto tops = topCells();
  if (tops.empty()) return {false, "no top cells"};
  const int dim = static_cast<int>(tops.front().size()) - 1;
  // The dangling-facet check needs the (n-1)-cell universe; cellSimplices()
  // is exactly that when the synthesis degree sits one below the top
  // dimension (the register layers). At other degrees the facet universe is
  // not tracked here, so only the top-cell conditions are checked.
  const bool facetDegree = (k_ == dim - 1);
  return ChainComplex::dualComplexIsValid(
      tops, dim,
      facetDegree ? cellSimplices()
                  : std::vector<std::vector<std::uint64_t>>{});
}

// === The discovered operator: ker L₁(W − ∂W) (#363) ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::bulkMinusBoundaryCells() const {
  std::vector<std::vector<std::uint64_t>> out;
  if (!st_) return out;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (auto &e : ChainComplex::fromSpacetime(*st_).kSimplexVertices(1)) {
    bool interior = true;
    for (const std::uint64_t v : e)
      if (bverts.count(v)) { interior = false; break; }
    if (interior) out.push_back(e);
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::bulkMinusBoundaryHarmonicMatrix(
    double tol) const {
  using Eigen::Index;
  using Eigen::MatrixXd;
  if (!st_) return {};

  // W − ∂W is the subcomplex induced on the interior vertices (the ones on no
  // ∂W face); a cell belongs to it iff all of its vertices are interior. This is
  // the ticket's "boundary removed" — and ties to its measured "3 interior
  // vertices carry nothing" (too few interior vertices ⇒ no interior 1-cycle).
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  const auto interiorCell = [&](const std::vector<std::uint64_t> &c) {
    for (const std::uint64_t v : c)
      if (bverts.count(v)) return false;
    return true;
  };

  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const auto v0 = cc.kSimplexVertices(0);
  const auto v1 = cc.kSimplexVertices(1);
  const auto v2 = cc.kSimplexVertices(2);
  const std::size_t n0 = v0.size(), n1 = v1.size(), n2 = v2.size();
  if (n1 == 0) return {};

  // The interior-cell index lists into the full C_k orderings.
  std::vector<Index> i0, i1, i2;
  for (std::size_t i = 0; i < n0; ++i)
    if (interiorCell(v0[i])) i0.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n1; ++i)
    if (interiorCell(v1[i])) i1.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n2; ++i)
    if (interiorCell(v2[i])) i2.push_back(static_cast<Index>(i));
  const Index m1 = static_cast<Index>(i1.size());
  if (m1 == 0) return {};

  // Restrict ∂₁ (n0×n1) and ∂₂ (n1×n2) to the interior rows/columns, then the
  // combinatorial L₁ = ∂₁ᵀ∂₁ + ∂₂∂₂ᵀ (unit weights — the magnitude,
  // signature-blind register the merge reads with harmonicMatrix(1,·,False)).
  const std::vector<long> &d1 = cc.boundaryMatrix(1);
  const std::vector<long> &d2 = cc.boundaryMatrix(2);
  MatrixXd D1 = MatrixXd::Zero(static_cast<Index>(i0.size()), m1);
  for (Index r = 0; r < static_cast<Index>(i0.size()); ++r)
    for (Index c = 0; c < m1; ++c)
      D1(r, c) = static_cast<double>(
          d1[static_cast<std::size_t>(i0[r]) * n1 + static_cast<std::size_t>(i1[c])]);
  MatrixXd L = D1.transpose() * D1;
  if (!i2.empty()) {
    MatrixXd D2 = MatrixXd::Zero(m1, static_cast<Index>(i2.size()));
    for (Index r = 0; r < m1; ++r)
      for (Index c = 0; c < static_cast<Index>(i2.size()); ++c)
        D2(r, c) = static_cast<double>(
            d2[static_cast<std::size_t>(i1[r]) * n2 + static_cast<std::size_t>(i2[c])]);
    L += D2 * D2.transpose();
  }

  // ker L₁ ≅ H₁ of the interior: the |λ| < tol eigenvectors, stacked as rows in
  // ascending-eigenvalue order (matching HodgeLaplacian::harmonicMatrix).
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(L);
  const Eigen::VectorXd &lam = eig.eigenvalues();
  const MatrixXd &V = eig.eigenvectors();
  std::vector<cd> out;
  for (Index j = 0; j < lam.size(); ++j) {
    if (std::abs(lam[j]) >= tol) continue;
    for (Index i = 0; i < m1; ++i) out.push_back(cd(V(i, j), 0.0));
  }
  return out;
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleRegisterReadout(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  const auto joinIds = [](const std::vector<std::uint64_t> &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };

  RegisterReadout out;
  const std::size_t n = order_;
  // Harmonics fresh from the live complex — surgery between calls moves them,
  // and the operator's own spectral cache is keyed to construction time.
  out.H = HodgeLaplacian(st_).harmonicMatrix(k_, 1e-9, /*metric=*/true);
  if (n == 0) {
    if (!holes.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleRegisterReadout: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n) + " (the complex changed behind the synthesizer)");
  out.dim = out.H.size() / n;

  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;

  const std::size_t m = holes.size();
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: hole " + joinIds(h) +
          " has " + std::to_string(h.size()) + " vertices; a degree-" +
          std::to_string(k_) + " period needs a removed (k+1)-cell of " +
          std::to_string(hv));
    // Facets are visited in the degree's established walk order — the
    // (a,b),(b,c),(a,c) edge walk of a circle at k = 1 (so the period
    // accumulates in exactly the register layers' summation order, bit for
    // bit), the canonical drop-v_j order otherwise. The hole's leak facet is
    // the first of the walk — the (a,b) edge / the drop-v_0 facet — which
    // carries boundary sign +1 in both conventions, so adding the leak there
    // moves that hole's period by exactly the leak.
    std::vector<std::size_t> walk(hv);
    for (std::size_t i = 0; i < hv; ++i) walk[i] = i;
    if (k_ == 1) std::rotate(walk.begin(), walk.end() - 1, walk.end());
    for (std::size_t w = 0; w < hv; ++w) {
      const std::size_t j = walk[w];
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::assembleRegisterReadout: facet " +
            joinIds(f) + " of hole " + joinIds(h) +
            " is not a k-cell of the complex (not a boundary cycle here)");
      if (w == 0) out.leakColumns.push_back(it->second);
      // The induced-orientation boundary sign: facet j of the sorted hole
      // drops vertex v_j and carries (-1)^j.
      const double s = (j % 2 == 0) ? 1.0 : -1.0;
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    }
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriods(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  return assembleRegisterReadout(holes).P;
}

std::vector<cd> EigenstateSynthesis::carriedRepresentative(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  if (targetPeriods.size() != holes.size())
    throw std::runtime_error(
        "EigenstateSynthesis::carriedRepresentative: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(holes.size()) + " holes");
  return carriedFromReadout(assembleRegisterReadout(holes), targetPeriods);
}

std::vector<cd> EigenstateSynthesis::carriedFromReadout(
    const RegisterReadout &ro, const std::vector<cd> &targetPeriods) const {
  const std::size_t n = order_;
  const std::size_t m = ro.leakColumns.size();
  if (n == 0) return {};
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::carriedFromReadout: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " cycles");

  // The minimum-norm least-squares projection onto the carried period rows
  // (what numpy.linalg.lstsq returns): c = (P^T)^+ target via the SVD.
  Eigen::VectorXcd t(static_cast<Eigen::Index>(m));
  for (std::size_t q = 0; q < m; ++q)
    t[static_cast<Eigen::Index>(q)] = targetPeriods[q];
  Eigen::VectorXcd c(static_cast<Eigen::Index>(ro.dim));
  if (ro.dim > 0) {
    Eigen::MatrixXcd Pt(static_cast<Eigen::Index>(m),
                        static_cast<Eigen::Index>(ro.dim));
    for (std::size_t q = 0; q < m; ++q)
      for (std::size_t r = 0; r < ro.dim; ++r)
        Pt(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r)) =
            ro.P[r * m + q];
    c = Pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(t);
  }

  // The carried representative plus the minimal leak: psi = sum_r c_r h_r,
  // then each cycle's uncarried remainder lands on its leak column, so the
  // cochain's periods are exactly the targets.
  std::vector<cd> psi(n, cd(0.0, 0.0));
  for (std::size_t r = 0; r < ro.dim; ++r) {
    const cd cr = c[static_cast<Eigen::Index>(r)];
    for (std::size_t i = 0; i < n; ++i) psi[i] += cr * ro.H[r * n + i];
  }
  for (std::size_t q = 0; q < m; ++q) {
    cd carried(0.0, 0.0);
    for (std::size_t r = 0; r < ro.dim; ++r)
      carried += c[static_cast<Eigen::Index>(r)] * ro.P[r * m + q];
    psi[ro.leakColumns[q]] += targetPeriods[q] - carried;
  }
  return psi;
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleReadoutOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  RegisterReadout out;
  const std::size_t n = order_;
  out.H = HodgeLaplacian(st_).harmonicMatrix(k_, 1e-9, /*metric=*/true);
  if (n == 0) {
    if (!loops.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleReadoutOverLoops: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n));
  out.dim = out.H.size() / n;
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t m = loops.size();
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: loop " +
          std::to_string(q) + " is empty");
    bool first = true;
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::assembleReadoutOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (first) {
        out.leakColumns.push_back(it->second);
        first = false;
      }
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    });
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriodsOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  return assembleReadoutOverLoops(loops).P;
}

double EigenstateSynthesis::residualForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi =
      carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
  if (psi.empty()) return 0.0;
  return residual(psi);
}

std::vector<cd> EigenstateSynthesis::carriedRepresentativeOverLoops(
    const std::vector<EdgeLoop> &loops, const std::vector<cd> &targetPeriods) const {
  return carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodsOfCochainOverLoops(
    const std::vector<cd> &cochain, const std::vector<EdgeLoop> &loops) const {
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  std::vector<cd> out(loops.size(), cd(0.0, 0.0));
  for (std::size_t q = 0; q < loops.size(); ++q)
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodsOfCochainOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (it->second < cochain.size())
        out[q] += cd(s, 0.0) * cochain[it->second];
    });
  return out;
}

double EigenstateSynthesis::residualForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi = carriedRepresentative(holes, targetPeriods);
  if (psi.empty()) return 0.0;  // no k-cells to carry periods
  return residual(psi);
}

std::vector<EigenstateSynthesis::EdgeLoop> EigenstateSynthesis::holeLoops(
    const std::vector<std::vector<std::uint64_t>> &holes, const char *who) const {
  auto &vlist = *st_->getVertexList();
  auto edge = [&](std::uint64_t u, std::uint64_t v) {
    auto *vu = vlist.get(u), *vv = vlist.get(v);
    if (!vu || !vv)
      throw std::runtime_error(std::string(who) + ": hole vertex absent");
    return Edge(vu, vv, cd(1.0, 0.0));
  };
  std::vector<EdgeLoop> loops;
  loops.reserve(holes.size());
  for (const auto &h : holes) {
    if (h.size() != 3)
      throw std::runtime_error(std::string(who) + ": hole has " +
                               std::to_string(h.size()) +
                               " vertices, expected 3");
    std::vector<std::uint64_t> s(h);
    std::sort(s.begin(), s.end());
    loops.push_back({edge(s[0], s[1]), edge(s[1], s[2]), edge(s[2], s[0])});
  }
  return loops;
}

double EigenstateSynthesis::periodGapForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::size_t m = loops.size();
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  if (m == 0) return 0.0;
  const RegisterReadout ro = assembleReadoutOverLoops(loops);
  // No harmonic carried: none of the target is reachable -> the gap is all of it.
  if (ro.dim == 0) {
    double g = 0.0;
    for (const cd &t : targetPeriods) g += std::norm(t);
    return g;
  }
  // P^T (m x dim): the live harmonics' periods over the loops. The carried object
  // stays a PURE harmonic (no leak): least-squares-fit the target onto P^T's
  // column space and return the squared norm of the unreachable remainder.
  Eigen::MatrixXcd Pt(static_cast<Eigen::Index>(m),
                      static_cast<Eigen::Index>(ro.dim));
  for (std::size_t q = 0; q < m; ++q)
    for (std::size_t r = 0; r < ro.dim; ++r)
      Pt(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r)) =
          ro.P[r * m + q];
  Eigen::VectorXcd t(static_cast<Eigen::Index>(m));
  for (std::size_t q = 0; q < m; ++q)
    t[static_cast<Eigen::Index>(q)] = targetPeriods[q];
  const Eigen::VectorXcd c =
      Pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(t);
  return (Pt * c - t).squaredNorm();
}

double EigenstateSynthesis::periodGapForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  return periodGapForLoops(
      holeLoops(holes, "EigenstateSynthesis::periodGapForPeriods"),
      targetPeriods);
}

std::vector<double> EigenstateSynthesis::periodGradientOverLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  using Eigen::Index;
  using Eigen::MatrixXd;
  using Eigen::VectorXcd;
  using Eigen::VectorXd;
  const std::size_t n1 = order_;
  std::vector<double> grad(n1, 0.0);
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientOverLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // ---- chain complex, weights, and the metric Laplacian M = L1 ----
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto tris = cc.kSimplexVertices(2);
  const std::size_t n2 = tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = d1flat.size() / n1;
  const HodgeLaplacian hl(st_);
  const std::vector<double> W1v = hl.weights(1);  // n1
  const std::vector<double> W2v = hl.weights(2);  // n2
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true, /*lorentzian=*/false);

  MatrixXd M(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j].real();
  VectorXd W1(N), D1d(N), D1pd(N);  // W1, 1/sqrt(W1), sqrt(W1)
  for (std::size_t i = 0; i < n1; ++i) {
    W1[static_cast<Index>(i)] = W1v[i];
    D1pd[static_cast<Index>(i)] = std::sqrt(W1v[i]);
    D1d[static_cast<Index>(i)] = 1.0 / std::sqrt(W1v[i]);
  }
  MatrixXd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  MatrixXd d2m(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  const MatrixXd K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t) W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  const MatrixXd K2 = d2m * W2inv.asDiagonal() * d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  auto key = [](std::uint64_t a, std::uint64_t b) {
    return std::pair<std::uint64_t, std::uint64_t>(std::min(a, b), std::max(a, b));
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> cidx1;
  for (std::size_t i = 0; i < n1; ++i) cidx1[key(cells1[i][0], cells1[i][1])] = i;
  std::map<std::pair<std::uint64_t, std::uint64_t>, double> l2map;
  for (auto *e : edges_)
    l2map[key(e->getSource()->getId(), e->getTarget()->getId())] =
        e->getSquaredLength().real();
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> trisOf;
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        trisOf[key(tris[ti][i], tris[ti][j])].push_back(ti);
  auto L2 = [&](std::uint64_t a, std::uint64_t b) -> double {
    if (a == b) return 0.0;
    auto it = l2map.find(key(a, b));
    return it == l2map.end() ? 0.0 : it->second;
  };

  // ---- Q (signed edge-loop covector) + each cycle's leak column ----
  // Generalizes the removed-triangle boundary to any closed walk of oriented
  // edges: Q(q, edge) += +1 along the stored orientation, -1 against; the leak
  // is the loop's first edge.
  MatrixXd Q = MatrixXd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m);
  for (std::size_t q = 0; q < m; ++q) {
    const EdgeLoop &loop = loops[q];
    if (loop.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientOverLoops: loop " +
          std::to_string(q) + " is empty");
    const Edge &fe = loop.front();
    leakCol[q] = cidx1.at(key(fe.getSource()->getId(), fe.getTarget()->getId()));
    Edge::walkLoop(loop, [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q),
        static_cast<Index>(cidx1.at(key(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(M);
  const VectorXd lam = eig.eigenvalues();
  const MatrixXd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  VectorXd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi (via Un / Q / pseudo-inverse), p, rho, r_U ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXd A = Q * Un;                            // m x nd
  const MatrixXd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;  // nd
  VectorXcd psi = Un.cast<cd>() * c;                    // n1
  const VectorXcd carried = Q.cast<cd>() * psi;         // m
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // ---- per-edge analytic gradient d r_U / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = key(cells1[je][0], cells1[je][1]);
    const double l2 = l2map.at(ek);
    const double dW1je = (l2 >= 0.0 ? 1.0 : -1.0) / (2.0 * std::sqrt(std::abs(l2)));
    const double s1 = -0.5 * dW1je / std::pow(W1[j], 1.5);
    const double s2 = 0.5 * dW1je / std::sqrt(W1[j]);
    const VectorXd w = s1 * K1.row(j).transpose().cwiseProduct(D1d) +
                       s2 * K2.row(j).transpose().cwiseProduct(D1pd);
    // dM = fa * fb^T (symmetric, low rank): columns [e, w] then one per triangle.
    std::vector<VectorXd> colsA, colsB;
    VectorXd ev = VectorXd::Zero(N);
    ev[j] = 1.0;
    colsA.push_back(ev);
    colsB.push_back(w);
    colsA.push_back(w);
    colsB.push_back(ev);
    for (std::size_t ti : trisOf[ek]) {
      const auto &t = tris[ti];
      Eigen::Matrix2d G;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
      const double detG = G.determinant();
      const double W2ti = W2v[ti];
      if (std::abs(std::sqrt(std::abs(detG)) / 2.0 - W2ti) > 1e-9 || std::abs(detG) < 1e-12)
        continue;
      auto ind = [&](int pp, int qq) -> double {
        return (pp != qq && key(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
      };
      Eigen::Matrix2d dG;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
      const double dW2ti = (W2ti / 2.0) * (G.inverse() * dG).trace();
      const VectorXd cj = D1pd.cwiseProduct(d2m.col(static_cast<Index>(ti)));
      colsA.push_back(cj);
      colsB.push_back((-dW2ti / (W2ti * W2ti)) * cj);
    }
    const Index r = static_cast<Index>(colsA.size());
    MatrixXd fa(N, r), fb(N, r);
    for (Index k = 0; k < r; ++k) {
      fa.col(k) = colsA[static_cast<std::size_t>(k)];
      fb.col(k) = colsB[static_cast<std::size_t>(k)];
    }
    // dM p, the eigenvector perturbation dUn, the pseudo-inverse perturbation, dpsi.
    const VectorXcd dMp = fa.cast<cd>() * (fb.transpose().cast<cd>() * p);
    const MatrixXd core = (Unn.transpose() * fa) * (fb.transpose() * Un);  // nnd x nd
    const MatrixXd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXd dA = Q * dUn;                                           // m x nd
    const MatrixXd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                             // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;                       // nd
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;              // n1
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;                        // m
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M.cast<cd>() * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGapForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  // The hard-pin sibling of periodGradientOverLoops (r_U): same first-order
  // eigenvector-perturbation setup (M = L1, harmonic split Un/Unn, the per-edge
  // low-rank dM, dUn), but the score is the period GAP r_psi = ||A c - t||^2 with
  // A = Q Un and c the least-squares fit, NOT the leak'd state's non-harmonicity.
  // Least-squares optimality A^T r = 0 (envelope theorem) drops the dc term, so
  // d r_psi / d l^2 = 2 Re( r^H (Q dUn) c ) -- no leak, no dpsi chain.
  using Eigen::Index;
  using Eigen::MatrixXd;
  using Eigen::VectorXcd;
  using Eigen::VectorXd;
  const std::size_t n1 = order_;
  std::vector<double> grad(n1, 0.0);
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoopsGradient: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // ---- chain complex, weights, and the metric Laplacian M = L1 ----
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto tris = cc.kSimplexVertices(2);
  const std::size_t n2 = tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = d1flat.size() / n1;
  const HodgeLaplacian hl(st_);
  const std::vector<double> W1v = hl.weights(1);  // n1
  const std::vector<double> W2v = hl.weights(2);  // n2
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true, /*lorentzian=*/false);

  MatrixXd M(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j].real();
  VectorXd W1(N), D1d(N), D1pd(N);  // W1, 1/sqrt(W1), sqrt(W1)
  for (std::size_t i = 0; i < n1; ++i) {
    W1[static_cast<Index>(i)] = W1v[i];
    D1pd[static_cast<Index>(i)] = std::sqrt(W1v[i]);
    D1d[static_cast<Index>(i)] = 1.0 / std::sqrt(W1v[i]);
  }
  MatrixXd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  MatrixXd d2m(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  const MatrixXd K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t) W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  const MatrixXd K2 = d2m * W2inv.asDiagonal() * d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  auto key = [](std::uint64_t a, std::uint64_t b) {
    return std::pair<std::uint64_t, std::uint64_t>(std::min(a, b), std::max(a, b));
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> cidx1;
  for (std::size_t i = 0; i < n1; ++i) cidx1[key(cells1[i][0], cells1[i][1])] = i;
  std::map<std::pair<std::uint64_t, std::uint64_t>, double> l2map;
  for (auto *e : edges_)
    l2map[key(e->getSource()->getId(), e->getTarget()->getId())] =
        e->getSquaredLength().real();
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> trisOf;
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        trisOf[key(tris[ti][i], tris[ti][j])].push_back(ti);
  auto L2 = [&](std::uint64_t a, std::uint64_t b) -> double {
    if (a == b) return 0.0;
    auto it = l2map.find(key(a, b));
    return it == l2map.end() ? 0.0 : it->second;
  };

  // ---- Q (signed edge-loop covector); the gap needs no leak column ----
  MatrixXd Q = MatrixXd::Zero(static_cast<Index>(m), N);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGapForLoopsGradient: loop " +
          std::to_string(q) + " is empty");
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q), static_cast<Index>(cidx1.at(key(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(M);
  const VectorXd lam = eig.eigenvalues();
  const MatrixXd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried, gap is constant
  MatrixXd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  VectorXd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- the least-squares fit c and the period-gap residual r = A c - target ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXd A = Q * Un;                            // m x nd
  const MatrixXd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;  // nd
  const VectorXcd r = A.cast<cd>() * c - target;        // m; A^T r = 0 (optimality)

  // ---- per-edge analytic gradient d r_psi / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = key(cells1[je][0], cells1[je][1]);
    const double l2 = l2map.at(ek);
    const double dW1je = (l2 >= 0.0 ? 1.0 : -1.0) / (2.0 * std::sqrt(std::abs(l2)));
    const double s1 = -0.5 * dW1je / std::pow(W1[j], 1.5);
    const double s2 = 0.5 * dW1je / std::sqrt(W1[j]);
    const VectorXd w = s1 * K1.row(j).transpose().cwiseProduct(D1d) +
                       s2 * K2.row(j).transpose().cwiseProduct(D1pd);
    // dM = fa * fb^T (symmetric, low rank): columns [e, w] then one per triangle.
    std::vector<VectorXd> colsA, colsB;
    VectorXd ev = VectorXd::Zero(N);
    ev[j] = 1.0;
    colsA.push_back(ev);
    colsB.push_back(w);
    colsA.push_back(w);
    colsB.push_back(ev);
    for (std::size_t ti : trisOf[ek]) {
      const auto &t = tris[ti];
      Eigen::Matrix2d G;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
      const double detG = G.determinant();
      const double W2ti = W2v[ti];
      if (std::abs(std::sqrt(std::abs(detG)) / 2.0 - W2ti) > 1e-9 || std::abs(detG) < 1e-12)
        continue;
      auto ind = [&](int pp, int qq) -> double {
        return (pp != qq && key(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
      };
      Eigen::Matrix2d dG;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
      const double dW2ti = (W2ti / 2.0) * (G.inverse() * dG).trace();
      const VectorXd cj = D1pd.cwiseProduct(d2m.col(static_cast<Index>(ti)));
      colsA.push_back(cj);
      colsB.push_back((-dW2ti / (W2ti * W2ti)) * cj);
    }
    const Index rk = static_cast<Index>(colsA.size());
    MatrixXd fa(N, rk), fb(N, rk);
    for (Index k = 0; k < rk; ++k) {
      fa.col(k) = colsA[static_cast<std::size_t>(k)];
      fb.col(k) = colsB[static_cast<std::size_t>(k)];
    }
    // The harmonic-subspace perturbation dUn, then dA = Q dUn; the envelope
    // theorem (A^T r = 0) leaves only 2 Re( r^H (dA c) ).
    const MatrixXd core = (Unn.transpose() * fa) * (fb.transpose() * Un);  // nnd x nd
    const MatrixXd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXd dA = Q * dUn;                                           // m x nd
    grad[je] = 2.0 * (r.dot(dA.cast<cd>() * c)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGapForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  return periodGapForLoopsGradient(
      holeLoops(holes, "EigenstateSynthesis::periodGapForPeriodsGradient"),
      targetPeriods);
}

std::vector<double> EigenstateSynthesis::residualForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // A removed triangle's boundary IS the oriented loop h0 -> h1 -> h2 -> h0
  // (identical signed covector and leak), so route through the loop core.
  auto &vlist = *st_->getVertexList();
  auto edge = [&](std::uint64_t u, std::uint64_t v) {
    auto *vu = vlist.get(u), *vv = vlist.get(v);
    if (!vu || !vv)
      throw std::runtime_error(
          "EigenstateSynthesis::residualForPeriodsGradient: hole vertex absent");
    return Edge(vu, vv, cd(1.0, 0.0));
  };
  std::vector<EdgeLoop> loops;
  loops.reserve(holes.size());
  for (const auto &h : holes) {
    if (h.size() != 3)
      throw std::runtime_error(
          "EigenstateSynthesis::residualForPeriodsGradient: hole has " +
          std::to_string(h.size()) + " vertices, expected 3");
    std::vector<std::uint64_t> s(h);
    std::sort(s.begin(), s.end());
    loops.push_back({edge(s[0], s[1]), edge(s[1], s[2]), edge(s[2], s[0])});
  }
  return periodGradientOverLoops(loops, targetPeriods);
}

std::vector<double> EigenstateSynthesis::residualForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  return periodGradientOverLoops(loops, targetPeriods);
}

std::vector<double> EigenstateSynthesis::residualForPeriodsGradientGpu(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
#ifndef TESSERA_CUDA
  (void)holes;
  (void)targetPeriods;
  throw std::runtime_error(
      "EigenstateSynthesis::residualForPeriodsGradientGpu: tessera was built "
      "without CUDA (TESSERA_CUDA=OFF); use the CPU residualForPeriodsGradient.");
#else
  using Eigen::Index;
  using Eigen::MatrixXd;
  using Eigen::MatrixXf;
  using Eigen::VectorXcd;
  using Eigen::VectorXd;
  // The setup below MIRRORS residualForPeriodsGradient verbatim through r_U so
  // the FP64 CPU method stays the untouched correctness oracle; only the
  // dominant per-edge GEMMs are offloaded to FP32 cuBLAS (the sole
  // approximation). Any divergence here would be a bug, not a design choice.
  const std::size_t n1 = order_;
  std::vector<double> grad(n1, 0.0);
  const std::size_t m = holes.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::residualForPeriodsGradientGpu: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // ---- chain complex, weights, and the metric Laplacian M = L1 ----
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto tris = cc.kSimplexVertices(2);
  const std::size_t n2 = tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = d1flat.size() / n1;
  const HodgeLaplacian hl(st_);
  const std::vector<double> W1v = hl.weights(1);  // n1
  const std::vector<double> W2v = hl.weights(2);  // n2
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true, /*lorentzian=*/false);

  MatrixXd M(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j].real();
  VectorXd W1(N), D1d(N), D1pd(N);  // W1, 1/sqrt(W1), sqrt(W1)
  for (std::size_t i = 0; i < n1; ++i) {
    W1[static_cast<Index>(i)] = W1v[i];
    D1pd[static_cast<Index>(i)] = std::sqrt(W1v[i]);
    D1d[static_cast<Index>(i)] = 1.0 / std::sqrt(W1v[i]);
  }
  MatrixXd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  MatrixXd d2m(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  const MatrixXd K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t) W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  const MatrixXd K2 = d2m * W2inv.asDiagonal() * d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  auto key = [](std::uint64_t a, std::uint64_t b) {
    return std::pair<std::uint64_t, std::uint64_t>(std::min(a, b), std::max(a, b));
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> cidx1;
  for (std::size_t i = 0; i < n1; ++i) cidx1[key(cells1[i][0], cells1[i][1])] = i;
  std::map<std::pair<std::uint64_t, std::uint64_t>, double> l2map;
  for (auto *e : edges_)
    l2map[key(e->getSource()->getId(), e->getTarget()->getId())] =
        e->getSquaredLength().real();
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> trisOf;
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        trisOf[key(tris[ti][i], tris[ti][j])].push_back(ti);
  auto L2 = [&](std::uint64_t a, std::uint64_t b) -> double {
    if (a == b) return 0.0;
    auto it = l2map.find(key(a, b));
    return it == l2map.end() ? 0.0 : it->second;
  };

  // ---- Q (hole-boundary covector) + each hole's leak column ----
  MatrixXd Q = MatrixXd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m);
  for (std::size_t q = 0; q < m; ++q) {
    const auto &h = holes[q];
    for (int j = 0; j < 3; ++j) {
      std::vector<std::uint64_t> facet;
      for (int i = 0; i < 3; ++i)
        if (i != j) facet.push_back(h[i]);
      Q(static_cast<Index>(q), static_cast<Index>(cidx1.at(key(facet[0], facet[1])))) +=
          (j % 2 == 0 ? 1.0 : -1.0);
    }
    leakCol[q] = cidx1.at(key(h[0], h[1]));
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(M);
  const VectorXd lam = eig.eigenvalues();
  const MatrixXd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  VectorXd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi (via Un / Q / pseudo-inverse), p, rho, r_U ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXd A = Q * Un;                            // m x nd
  const MatrixXd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;  // nd
  VectorXcd psi = Un.cast<cd>() * c;                    // n1
  const VectorXcd carried = Q.cast<cd>() * psi;         // m
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();
  if (nnd == 0) return grad;  // eigenvector perturbation needs a non-null block

  // ---- upload the loop-invariant blocks (FP32, column-major) to the GPU ----
  // UnnS = Unn * diag(invlam): by associativity this hoists the per-edge
  // diagonal so dUn = UnnS * core == Unn * (invlam.asDiagonal() * core).
  const MatrixXf UnnF = Unn.cast<float>();
  const MatrixXf UnnSF = (Unn * invlam.asDiagonal()).cast<float>();
  const MatrixXf UnF = Un.cast<float>();
  const MatrixXf MF = M.cast<float>();
  MatrixXf P2F(N, 2);
  P2F.col(0) = p.real().cast<float>();
  P2F.col(1) = p.imag().cast<float>();
  // rmax: the widest per-edge low-rank dM = fa*fb^T (the 2 fixed [e,w] columns
  // plus one per incident triangle) — sizes the GPU scratch once.
  std::size_t rmax = 2;
  for (const auto &kv : trisOf) rmax = std::max(rmax, 2 + kv.second.size());

  cuda::RuGradientGpu gpu(static_cast<int>(N), static_cast<int>(nnd),
                          static_cast<int>(nd), static_cast<int>(rmax),
                          UnnF.data(), UnnSF.data(), UnF.data(), MF.data(),
                          P2F.data());

  // ---- per-edge analytic gradient d r_U / d l^2 (FP32 GEMMs on GPU) ----
  std::vector<float> dUnBuf(static_cast<std::size_t>(N) * nd);
  std::vector<float> dMp2Buf(static_cast<std::size_t>(N) * 2);
  std::vector<float> dpsi2Buf(static_cast<std::size_t>(N) * 2);
  std::vector<float> Mdpsi2Buf(static_cast<std::size_t>(N) * 2);
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = key(cells1[je][0], cells1[je][1]);
    const double l2 = l2map.at(ek);
    const double dW1je = (l2 >= 0.0 ? 1.0 : -1.0) / (2.0 * std::sqrt(std::abs(l2)));
    const double s1 = -0.5 * dW1je / std::pow(W1[j], 1.5);
    const double s2 = 0.5 * dW1je / std::sqrt(W1[j]);
    const VectorXd w = s1 * K1.row(j).transpose().cwiseProduct(D1d) +
                       s2 * K2.row(j).transpose().cwiseProduct(D1pd);
    // dM = fa * fb^T (symmetric, low rank): columns [e, w] then one per triangle.
    std::vector<VectorXd> colsA, colsB;
    VectorXd ev = VectorXd::Zero(N);
    ev[j] = 1.0;
    colsA.push_back(ev);
    colsB.push_back(w);
    colsA.push_back(w);
    colsB.push_back(ev);
    for (std::size_t ti : trisOf[ek]) {
      const auto &t = tris[ti];
      Eigen::Matrix2d G;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
      const double detG = G.determinant();
      const double W2ti = W2v[ti];
      if (std::abs(std::sqrt(std::abs(detG)) / 2.0 - W2ti) > 1e-9 || std::abs(detG) < 1e-12)
        continue;
      auto ind = [&](int pp, int qq) -> double {
        return (pp != qq && key(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
      };
      Eigen::Matrix2d dG;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
      const double dW2ti = (W2ti / 2.0) * (G.inverse() * dG).trace();
      const VectorXd cj = D1pd.cwiseProduct(d2m.col(static_cast<Index>(ti)));
      colsA.push_back(cj);
      colsB.push_back((-dW2ti / (W2ti * W2ti)) * cj);
    }
    const int r = static_cast<int>(colsA.size());
    MatrixXf fa(N, r), fb(N, r);
    for (int k = 0; k < r; ++k) {
      fa.col(k) = colsA[static_cast<std::size_t>(k)].cast<float>();
      fb.col(k) = colsB[static_cast<std::size_t>(k)].cast<float>();
    }

    // GPU stage 1 (FP32 GEMMs): dUn (N x nd) and dMp = dM*p (N, complex).
    gpu.edgeStage1(fa.data(), fb.data(), r, dUnBuf.data(), dMp2Buf.data());
    const MatrixXd dUn =
        Eigen::Map<const MatrixXf>(dUnBuf.data(), N, nd).cast<double>();
    VectorXcd dMp(N);
    for (Index i = 0; i < N; ++i)
      dMp[i] = cd(dMp2Buf[static_cast<std::size_t>(i)],
                  dMp2Buf[static_cast<std::size_t>(N + i)]);

    // Cheap small-dimension algebra stays FP64 on the host (as the oracle does):
    // pseudo-inverse perturbation, dc, dpsi, and the minimal-leak adjustment.
    const MatrixXd dA = Q * dUn;                                          // m x nd
    const MatrixXd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                            // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;                      // nd
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;             // n1
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;                       // m
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];

    // GPU stage 2 (FP32 GEMV): Mdpsi = M*dpsi for the post-leak perturbation.
    for (Index i = 0; i < N; ++i) {
      dpsi2Buf[static_cast<std::size_t>(i)] = static_cast<float>(dpsi[i].real());
      dpsi2Buf[static_cast<std::size_t>(N + i)] = static_cast<float>(dpsi[i].imag());
    }
    gpu.edgeStage2(dpsi2Buf.data(), Mdpsi2Buf.data());
    VectorXcd Mdpsi(N);
    for (Index i = 0; i < N; ++i)
      Mdpsi[i] = cd(Mdpsi2Buf[static_cast<std::size_t>(i)],
                    Mdpsi2Buf[static_cast<std::size_t>(N + i)]);

    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
#endif
}

}  // namespace tessera::cobordism
