// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "spacetime/PachnerMove.h"
#include "graph/IndexByKey.hpp"
#include "mesh/Simplex.h"
#include "mesh/Edge.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"

#ifdef TESSERA_CUDA
#include "cuda/regge_cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

// =====================================================================
// Construction
// =====================================================================

ReggeSolver::ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                         MatterConfiguration matter)
    : spacetime_(std::move(spacetime)), matter_(std::move(matter)) {
    // Materialize the facet/coface lattice down to the (d-2)-hinges, in C++.
    //
    // dualVolume() walks a hinge UP through its cofaces to a top cell, so every
    // coface link from the hinge up to the top must exist.  getFacets() on a
    // k-simplex creates its (k-1)-facets and registers itself as their coface,
    // so we must call it on every simplex of size >= d: the top d-cells (size
    // d+1) register the (d-1)-facets, and the (d-1)-facets (size d) register the
    // (d-2)-hinges.  build() does not guarantee the (d-1)-facets exist (e.g. a
    // freshly built SolidSimplex holds only its top cells), so we start from the
    // tops rather than assuming the facets are already present.
    //
    // This MUST run in C++.  The Python getFacets()/getCofaces() bindings use
    // return_value_policy::copy, so driving materialization from Python would
    // register *copies* of the sub-simplices — each carrying an incomplete
    // coface list — onto the shared vertices, and the fingerprint-keyed
    // hasCoface() guard would then block the canonical facets.  dualVolume()
    // would then see half the cofaces it should.
    //
    // getFacets() grows simplicesVec as it registers new sub-simplices, so the
    // loop re-reads size() each iteration rather than snapshotting it.
    const int d = spacetime_->getMetric()->getSignature()->getDimensions();
    for (std::size_t i = 0; i < spacetime_->getSimplices().size(); ++i) {
        auto s = spacetime_->getSimplices()[i];
        if (static_cast<int>(s->size()) >= d)
            (void)s->getFacets();
    }
}

// =====================================================================
// Geometry delegations to Simplex
// =====================================================================

double ReggeSolver::dihedralAngle(SimplexPtr sigma,
                                   SimplexPtr hinge) const {
    // Regge calculus runs on the Wick-rotated (Euclidean) geometry.
    return sigma->dihedralAngle(hinge, /*wickRotate=*/true);
}

double ReggeSolver::deficitAngle(SimplexPtr hinge) const {
    return hinge->deficitAngle();
}

double ReggeSolver::hingeArea(SimplexPtr hinge) {
    // Regge calculus runs on the Wick-rotated (Euclidean) geometry.
    return hinge->area(/*wickRotate=*/true);
}

// =====================================================================
// Collect hinges
// =====================================================================

std::vector<SimplexPtr> ReggeSolver::collectHinges() const {
    // Hinges are (d-2)-simplices. In 4D, these are triangles (3 vertices).
    // They are registered in the spacetime's simplex list (sub-simplices
    // are registered during getFacets()).
    int d = spacetime_->getMetric()->getSignature()->getDimensions();
    int hingeSize = d - 1; // (d-2)-simplex has (d-1) vertices

    std::vector<SimplexPtr> hinges;
    for (const auto &s : spacetime_->getSimplices()) {
        if (static_cast<int>(s->size()) == hingeSize)
            hinges.push_back(s);
    }
    return hinges;
}

// =====================================================================
// Actions
// =====================================================================

double ReggeSolver::reggeAction() const {
    double S = 0.0;
    for (const auto &h : collectHinges()) {
        S += hingeArea(h) * deficitAngle(h);
    }
    return S;
}

std::complex<double> ReggeSolver::dualReggeAction() const {
    // S_Regge(W*) = sum_h |*h| * eps_h: the circumcentric dual content of each
    // (d-2)-hinge weighted by its complex Lorentzian deficit. Hinges must be
    // registered (sub-simplices materialize via getFacets), as for reggeAction().
    std::complex<double> S(0.0, 0.0);
    for (const auto &h : collectHinges()) {
        S += h->dualVolume() * h->lorentzianDeficitAngle();
    }
    return S;
}

double ReggeSolver::matterAction() const {
    // Point-particle action: S_matter = -M ∫ dτ
    // Timelike edges (between slices) have ℓ² < 0; spacelike edges (within a
    // slice) have ℓ² > 0.  Proper time = √(-ℓ²).
    double S = 0.0;
    for (const auto &wl : matter_.getWorldlines()) {
        for (std::size_t i = 0; i + 1 < wl.vertices.size(); ++i) {
            auto *v1 = wl.vertices[i];
            auto *v2 = wl.vertices[i + 1];
            // Find the edge connecting consecutive worldline vertices
            for (const auto &e : v1->getEdges()) {
                auto *other = (e->getSource()->getId() == v1->getId())
                              ? e->getTarget() : e->getSource();
                if (other->getId() == v2->getId()) {
                    double sq = e->getSquaredLength().real();
                    if (sq < 0.0)  // timelike: ℓ² < 0
                        S -= wl.mass * std::sqrt(-sq);
                    break;
                }
            }
        }
    }
    return S;
}

double ReggeSolver::totalAction() const {
    return reggeAction() + matterAction();
}

// =====================================================================
// Action gradient: ∂S/∂ℓ²_e for each edge (numerical)
// =====================================================================

std::vector<double> ReggeSolver::actionGradient() const {
    auto edgeList = spacetime_->getEdgeList();
    auto edges = edgeList->toVector();
    std::vector<double> g(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::complex<double> origSq = edges[i]->getSquaredLength();
        const double W = std::abs(origSq);              // |l^2|
        const double h = std::max(W * 1e-4, 1e-8);
        const bool tl = edges[i]->isTimelike();
        auto sqAtW = [tl](double w) {                    // signed l^2 with |l^2|=w, same character
            return tl ? std::complex<double>{-w, 0.0}
                      : std::complex<double>{w, 0.0};
        };
        // Central differences in W-space, preserving edge character; perturb l^2
        // exactly and restore the original l^2 exactly (no sqrt round-trip drift).
        edges[i]->setSquaredLength(sqAtW(W + h));
        double Sp = totalAction();
        edges[i]->setSquaredLength(sqAtW(std::max(W - h, 1e-12)));
        double Sm = totalAction();
        g[i] = (Sp - Sm) / (2.0 * h);
        edges[i]->setSquaredLength(origSq);
    }
    return g;
}

std::vector<std::complex<double>> ReggeSolver::actionGradientExact() const {
    using cd = std::complex<double>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> eidx;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        eidx[{std::min(a, b), std::max(a, b)}] = i;
    }
    std::vector<cd> g(edges.size(), cd(0.0, 0.0));
    // dS/dl^2_e = sum_h [ d|*h|/dl^2_e * eps_h + |*h| * d eps_h/dl^2_e ]
    for (const auto &h : collectHinges()) {
        const cd eps = h->lorentzianDeficitAngle();
        const double dv = h->dualVolume();
        for (const auto &[e, dEps] : h->lorentzianDeficitAngleGradient()) {
            const auto it = eidx.find(e);
            if (it != eidx.end()) g[it->second] += dv * dEps;
        }
        for (const auto &[e, dDv] : h->dualVolumeGradient()) {
            const auto it = eidx.find(e);
            if (it != eidx.end()) g[it->second] += dDv * eps;
        }
    }
    return g;
}

std::vector<std::vector<std::complex<double>>>
ReggeSolver::actionHessianExact() const {
    using cd = std::complex<double>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> eidx;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        eidx[{std::min(a, b), std::max(a, b)}] = i;
    }
    const std::size_t E = edges.size();
    std::vector<std::vector<cd>> H(E, std::vector<cd>(E, cd(0.0, 0.0)));
    // d^2 S/dl^2_e dl^2_f = sum_h [ d2V_ef*eps + dV_e*dEps_f + dV_f*dEps_e
    //                              + V*d2Eps_ef ].
    for (const auto &h : collectHinges()) {
        const cd eps = h->lorentzianDeficitAngle();
        const double V = h->dualVolume();
        const auto dEps = h->lorentzianDeficitAngleGradient();
        const auto dV = h->dualVolumeGradient();
        const auto d2Eps = h->lorentzianDeficitAngleHessian();
        const auto d2V = h->dualVolumeHessian();
        for (const auto &[e, dVe] : dV) {
            const auto ie = eidx.find(e);
            if (ie == eidx.end()) continue;
            const auto dEe_it = dEps.find(e);
            const cd dEe = (dEe_it != dEps.end()) ? dEe_it->second : cd(0.0, 0.0);
            for (const auto &[f, dVf] : dV) {
                const auto if_ = eidx.find(f);
                if (if_ == eidx.end()) continue;
                const auto dEf_it = dEps.find(f);
                const cd dEf =
                    (dEf_it != dEps.end()) ? dEf_it->second : cd(0.0, 0.0);
                cd term = dVe * dEf + dVf * dEe;       // cross terms
                const auto k = std::make_pair(e, f);
                const auto d2Vit = d2V.find(k);
                if (d2Vit != d2V.end()) term += d2Vit->second * eps;
                const auto d2Eit = d2Eps.find(k);
                if (d2Eit != d2Eps.end()) term += V * d2Eit->second;
                H[ie->second][if_->second] += term;
            }
        }
    }
    return H;
}

double ReggeSolver::actionGradientNorm() const {
    auto g = actionGradient();
    double F = 0.0;
    for (double gi : g) F += gi * gi;
    return F;
}

// =====================================================================
// Incremental gradient (Pachner-local)
//
// The resident gradient g[] = ∂S/∂ℓ²_e and resident action S = Σ_h |★h|·ε_h
// are maintained by a subtract-old / add-new delta over the move's touched
// region.  A hinge's contribution depends only on the geometry of its top
// cells; a boundary-fixed Pachner move only adds/removes top cells inside the
// region spanned by ``touchedVertexIds``, so every hinge whose contribution
// changes is a (d-2)-face of a region top cell.  We re-evaluate the whole
// region (``regionHinges``) on both sides of the move: unchanged peripheral
// hinges recompute to the identical value and cancel (subtract then add), so
// the delta is exact without having to single out the changed hinges, and
// edges shared with hinges OUTSIDE the region keep their contribution because
// no edge entry is ever zeroed — only added to.
// =====================================================================

void ReggeSolver::accumulateHinge(SimplexPtr hinge, int sign) {
    using cd = std::complex<double>;
    const cd eps = hinge->lorentzianDeficitAngle();
    const double dv = hinge->dualVolume();
    const double s = static_cast<double>(sign);
    residentAction_ += cd(s, 0.0) * (cd(dv, 0.0) * eps);
    // dS/dℓ²_e = Σ_h [ |★h|·∂ε_h + ε_h·∂|★h| ], keyed by sorted-id edge.
    for (const auto &[e, dEps] : hinge->lorentzianDeficitAngleGradient())
        residentGradient_[e] += cd(s * dv, 0.0) * dEps;
    for (const auto &[e, dDv] : hinge->dualVolumeGradient())
        residentGradient_[e] += cd(s * dDv, 0.0) * eps;
}

std::vector<SimplexPtr>
ReggeSolver::regionHinges(const std::vector<std::uint64_t> &vertexIds) {
    const int d = spacetime_->getMetric()->getSignature()->getDimensions();
    const int topSize = d + 1;     // top cell: (d+1) vertices
    const int hingeSize = d - 1;   // (d-2)-hinge: (d-1) vertices
    const auto &vlist = spacetime_->getVertexList();

    // Snapshot the incident top cells FIRST (deduped): getFacets() below
    // registers freshly-materialized sub-simplices onto these same vertices'
    // simplex lists, which would invalidate an in-flight iterator.
    std::set<std::uint64_t> seenTops;
    std::vector<SimplexPtr> tops;
    for (std::uint64_t id : vertexIds) {
        Vertex *v = vlist->get(id);
        if (v == nullptr) continue;          // e.g. a vertex the move deleted
        for (const auto &s : v->getSimplices()) {
            if (static_cast<int>(s->size()) != topSize) continue;
            if (seenTops.insert(s->fingerprint.fingerprint()).second)
                tops.push_back(s);
        }
    }

    // Walk each top down to its (d-2)-hinges, materializing the facet lattice.
    std::set<std::uint64_t> seenHinges;
    std::vector<SimplexPtr> hinges;
    for (const auto &top : tops) {
        for (const auto &facet : top->getFacets()) {        // (d-1)-cells
            for (const auto &hinge : facet->getFacets()) {  // (d-2)-hinges
                if (static_cast<int>(hinge->size()) != hingeSize) continue;
                if (seenHinges.insert(hinge->fingerprint.fingerprint()).second)
                    hinges.push_back(hinge);
            }
        }
    }
    return hinges;
}

std::set<ReggeSolver::EdgeKey>
ReggeSolver::regionEdgeKeys(const std::vector<std::uint64_t> &vertexIds) const {
    std::set<EdgeKey> keys;
    const auto &vlist = spacetime_->getVertexList();
    for (std::uint64_t id : vertexIds) {
        Vertex *v = vlist->get(id);
        if (v == nullptr) continue;
        for (const auto &e : v->getEdges()) {  // returns a copy: safe to walk
            const std::uint64_t a = e->getSource()->getId();
            const std::uint64_t b = e->getTarget()->getId();
            keys.insert({std::min(a, b), std::max(a, b)});
        }
    }
    return keys;
}

void ReggeSolver::resetIncrementalGradient() {
    residentGradient_.clear();
    residentAction_ = std::complex<double>(0.0, 0.0);
    for (const auto &h : collectHinges()) accumulateHinge(h, +1);
    gradientResident_ = true;
}

void ReggeSolver::updateAround(const std::vector<std::uint64_t> &vertexIds,
                               const std::function<void()> &mutate) {
    const std::set<EdgeKey> preKeys = regionEdgeKeys(vertexIds);
    for (const auto &h : regionHinges(vertexIds)) accumulateHinge(h, -1);
    mutate();
    for (const auto &h : regionHinges(vertexIds)) accumulateHinge(h, +1);
    // Drop entries for edges the move deleted (present before, gone after) so a
    // later id-reusing rollback re-adds onto a clean slate rather than residue.
    const std::set<EdgeKey> postKeys = regionEdgeKeys(vertexIds);
    for (const auto &k : preKeys)
        if (postKeys.find(k) == postKeys.end()) residentGradient_.erase(k);
}

void ReggeSolver::applyMoveIncremental(
    ::tessera::spacetime::PachnerMove &move) {
    if (!gradientResident_)
        throw std::logic_error(
            "applyMoveIncremental: call resetIncrementalGradient() first to "
            "establish the resident-gradient baseline");
    updateAround(move.touchedVertexIds(), [&move] { (void)move.apply(); });
}

void ReggeSolver::rollbackMoveIncremental(
    ::tessera::spacetime::PachnerMove &move) {
    if (!gradientResident_)
        throw std::logic_error(
            "rollbackMoveIncremental: call resetIncrementalGradient() first to "
            "establish the resident-gradient baseline");
    updateAround(move.touchedVertexIds(), [&move] { move.rollback(); });
}

std::vector<std::complex<double>> ReggeSolver::incrementalGradient() const {
    using cd = std::complex<double>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    std::vector<cd> g(edges.size(), cd(0.0, 0.0));
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        const auto it =
            residentGradient_.find({std::min(a, b), std::max(a, b)});
        if (it != residentGradient_.end()) g[i] = it->second;
    }
    return g;
}

// =====================================================================
// GPU mesh flattening (CUDA path)
// =====================================================================

#ifdef TESSERA_CUDA
cuda::GpuMeshData ReggeSolver::flattenMeshForGpu() const {
    cuda::GpuMeshData mesh;
    int d = spacetime_->getMetric()->getSignature()->getDimensions();
    int topSize = d + 1;

    // --- Collect top-simplices ---
    std::vector<SimplexPtr> topSimplices;
    std::unordered_map<std::uint64_t, int> simplexToIdx;
    for (const auto &s : spacetime_->getSimplices()) {
        if (static_cast<int>(s->size()) == topSize) {
            simplexToIdx[s->fingerprint.fingerprint()] =
                static_cast<int>(topSimplices.size());
            topSimplices.push_back(s);
        }
    }
    mesh.n_simplices = static_cast<int>(topSimplices.size());

    // --- Collect edges and assign indices ---
    auto edgeVec = spacetime_->getEdgeList()->toVector();
    mesh.n_edges = static_cast<int>(edgeVec.size());
    auto edgeToIdx = ::tessera::graph::indexByKey(
        edgeVec, [](auto const& e) { return e->fingerprint.fingerprint(); });

    // --- Per-simplex squared-distance matrices ---
    // Also record which (simplex, row, col) positions correspond to each edge.
    mesh.simplex_sq_dist_offsets.resize(mesh.n_simplices + 1);
    mesh.simplex_n_verts.resize(mesh.n_simplices);
    std::vector<std::vector<int>> edgeDistPos(mesh.n_edges); // per edge: positions in flat array
    int sq_offset = 0;

    for (int si = 0; si < mesh.n_simplices; ++si) {
        auto verts = topSimplices[si]->getVertices();
        int nv = static_cast<int>(verts.size());
        mesh.simplex_n_verts[si] = nv;
        mesh.simplex_sq_dist_offsets[si] = sq_offset;

        std::unordered_map<std::uint64_t, double> sqMap;
        for (const auto &e : topSimplices[si]->getEdges()) {
            auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                      Fingerprint::mix64(e->getTarget()->getId());
            sqMap[fp] = std::abs(e->getSquaredLength());  // |l^2|, Wick-rotated
        }

        for (int i = 0; i < nv; ++i) {
            for (int j = 0; j < nv; ++j) {
                int pos = sq_offset + i * nv + j;
                if (i == j) {
                    mesh.simplex_sq_dist_flat.push_back(0.0);
                } else {
                    auto fp = Fingerprint::mix64(verts[i]->getId()) ^
                              Fingerprint::mix64(verts[j]->getId());
                    auto sqIt = sqMap.find(fp);
                    mesh.simplex_sq_dist_flat.push_back(
                        sqIt != sqMap.end() ? sqIt->second : 0.0);
                    // Record this position for the edge
                    auto eIt = edgeToIdx.find(fp);
                    if (eIt != edgeToIdx.end())
                        edgeDistPos[eIt->second].push_back(pos);
                }
            }
        }
        sq_offset += nv * nv;
    }
    mesh.simplex_sq_dist_offsets[mesh.n_simplices] = sq_offset;

    // --- Collect hinges ---
    auto hinges = collectHinges();
    mesh.n_hinges = static_cast<int>(hinges.size());
    std::unordered_map<std::uint64_t, int> hingeToIdx;
    for (int hi = 0; hi < mesh.n_hinges; ++hi)
        hingeToIdx[hinges[hi]->fingerprint.fingerprint()] = hi;

    // --- Hinge → simplex CSR ---
    mesh.hinge_simplex_offsets.resize(mesh.n_hinges + 1, 0);
    std::vector<std::vector<std::tuple<int,int,int>>> hingeEntries(mesh.n_hinges);
    for (int si = 0; si < mesh.n_simplices; ++si) {
        auto sigmaVerts = topSimplices[si]->getVertices();
        int nv = static_cast<int>(sigmaVerts.size());
        for (int a = 0; a < nv; ++a) {
            for (int b = a + 1; b < nv; ++b) {
                std::uint64_t hingeFp = 0;
                for (int k = 0; k < nv; ++k)
                    if (k != a && k != b) hingeFp ^= Fingerprint::mix64(sigmaVerts[k]->getId());
                auto it = hingeToIdx.find(hingeFp);
                if (it != hingeToIdx.end())
                    hingeEntries[it->second].emplace_back(si, a, b);
            }
        }
    }
    for (int hi = 0; hi < mesh.n_hinges; ++hi)
        mesh.hinge_simplex_offsets[hi+1] =
            mesh.hinge_simplex_offsets[hi] + static_cast<int>(hingeEntries[hi].size());
    int nnz_hs = mesh.hinge_simplex_offsets[mesh.n_hinges];
    mesh.hinge_simplex_ids.resize(nnz_hs);
    mesh.hinge_opposite_a.resize(nnz_hs);
    mesh.hinge_opposite_b.resize(nnz_hs);
    for (int hi = 0; hi < mesh.n_hinges; ++hi) {
        int off = mesh.hinge_simplex_offsets[hi];
        for (int k = 0; k < static_cast<int>(hingeEntries[hi].size()); ++k) {
            auto [si, a, b] = hingeEntries[hi][k];
            mesh.hinge_simplex_ids[off+k] = si;
            mesh.hinge_opposite_a[off+k] = a;
            mesh.hinge_opposite_b[off+k] = b;
        }
    }

    // --- Edge → hinge CSR (which hinges does each edge affect?) ---
    // An edge affects a hinge if they share a simplex.
    // Equivalently: edge (u,v) affects hinge h if some top-simplex contains
    // both edge vertices and all hinge vertices.
    std::vector<std::set<int>> edgeHingesSets(mesh.n_edges);
    for (int si = 0; si < mesh.n_simplices; ++si) {
        auto sigmaVerts = topSimplices[si]->getVertices();
        int nv = static_cast<int>(sigmaVerts.size());

        // Edges in this simplex
        std::vector<int> simplexEdgeIds;
        for (const auto &e : topSimplices[si]->getEdges()) {
            auto it = edgeToIdx.find(e->fingerprint.fingerprint());
            if (it != edgeToIdx.end()) simplexEdgeIds.push_back(it->second);
        }
        // Hinges in this simplex (complement of each pair)
        for (int a = 0; a < nv; ++a) {
            for (int b = a + 1; b < nv; ++b) {
                std::uint64_t hingeFp = 0;
                for (int k = 0; k < nv; ++k)
                    if (k != a && k != b) hingeFp ^= Fingerprint::mix64(sigmaVerts[k]->getId());
                auto it = hingeToIdx.find(hingeFp);
                if (it == hingeToIdx.end()) continue;
                int hid = it->second;
                for (int eid : simplexEdgeIds)
                    edgeHingesSets[eid].insert(hid);
            }
        }
    }
    mesh.edge_hinge_offsets.resize(mesh.n_edges + 1, 0);
    for (int ei = 0; ei < mesh.n_edges; ++ei)
        mesh.edge_hinge_offsets[ei+1] =
            mesh.edge_hinge_offsets[ei] + static_cast<int>(edgeHingesSets[ei].size());
    int nnz_eh = mesh.edge_hinge_offsets[mesh.n_edges];
    mesh.edge_hinge_ids.resize(nnz_eh);
    for (int ei = 0; ei < mesh.n_edges; ++ei) {
        int off = mesh.edge_hinge_offsets[ei];
        int k = 0;
        for (int hid : edgeHingesSets[ei])
            mesh.edge_hinge_ids[off + k++] = hid;
    }

    // --- Edge → dist positions CSR ---
    mesh.edge_dist_offsets.resize(mesh.n_edges + 1, 0);
    for (int ei = 0; ei < mesh.n_edges; ++ei)
        mesh.edge_dist_offsets[ei+1] =
            mesh.edge_dist_offsets[ei] + static_cast<int>(edgeDistPos[ei].size());
    int nnz_ed = mesh.edge_dist_offsets[mesh.n_edges];
    mesh.edge_dist_positions.resize(nnz_ed);
    for (int ei = 0; ei < mesh.n_edges; ++ei) {
        int off = mesh.edge_dist_offsets[ei];
        for (int k = 0; k < static_cast<int>(edgeDistPos[ei].size()); ++k)
            mesh.edge_dist_positions[off + k] = edgeDistPos[ei][k];
    }

    // --- Edge → neighbor edges CSR ---
    // Two edges are neighbors if they share at least one hinge.
    // Build reverse map: hinge → edges (inverse of edge → hinges).
    std::vector<std::vector<int>> hingeEdges(mesh.n_hinges);
    for (int ei = 0; ei < mesh.n_edges; ++ei) {
        for (int k = mesh.edge_hinge_offsets[ei];
             k < mesh.edge_hinge_offsets[ei + 1]; ++k)
            hingeEdges[mesh.edge_hinge_ids[k]].push_back(ei);
    }
    std::vector<std::set<int>> nbrSets(mesh.n_edges);
    for (int ei = 0; ei < mesh.n_edges; ++ei) {
        nbrSets[ei].insert(ei); // self-neighbor
        for (int k = mesh.edge_hinge_offsets[ei];
             k < mesh.edge_hinge_offsets[ei + 1]; ++k) {
            int hid = mesh.edge_hinge_ids[k];
            for (int nb : hingeEdges[hid])
                nbrSets[ei].insert(nb);
        }
    }
    mesh.edge_nbr_offsets.resize(mesh.n_edges + 1, 0);
    for (int ei = 0; ei < mesh.n_edges; ++ei)
        mesh.edge_nbr_offsets[ei + 1] =
            mesh.edge_nbr_offsets[ei] +
            static_cast<int>(nbrSets[ei].size());
    int nnz_nbr = mesh.edge_nbr_offsets[mesh.n_edges];
    mesh.edge_nbr_ids.resize(nnz_nbr);
    for (int ei = 0; ei < mesh.n_edges; ++ei) {
        int off = mesh.edge_nbr_offsets[ei];
        int k = 0;
        for (int nb : nbrSets[ei])
            mesh.edge_nbr_ids[off + k++] = nb;
    }

    // --- Target deficits (zero for deficit-residual kernel) ---
    mesh.target_deficits.resize(mesh.n_hinges, 0.0);

    // --- Base hinge contributions A_h * ε_h (for action gradient kernel) ---
    mesh.base_hinge_contribs.resize(mesh.n_hinges);
    for (int hi = 0; hi < mesh.n_hinges; ++hi) {
        mesh.base_hinge_contribs[hi] =
            hingeArea(hinges[hi]) * deficitAngle(hinges[hi]);
    }

    // --- Worldline mask and per-edge mass ---
    mesh.worldline_edge_mask.resize(mesh.n_edges, 0);
    mesh.worldline_edge_mass.resize(mesh.n_edges, 0.0);
    for (const auto &wl : matter_.getWorldlines()) {
        for (std::size_t i = 0; i + 1 < wl.vertices.size(); ++i) {
            auto fp = Fingerprint::mix64(wl.vertices[i]->getId()) ^
                      Fingerprint::mix64(wl.vertices[i + 1]->getId());
            auto it = edgeToIdx.find(fp);
            if (it != edgeToIdx.end()) {
                mesh.worldline_edge_mask[it->second] = 1;
                mesh.worldline_edge_mass[it->second] = wl.mass;
            }
        }
    }

    return mesh;
}
#endif

// =====================================================================
// Gradient descent step
// =====================================================================

double ReggeSolver::step(double learningRate) {
    auto edgeList = spacetime_->getEdgeList();
    auto edges = edgeList->toVector();
    int n = static_cast<int>(edges.size());

    // Minimize F = ||∇S||² = Σ_e (∂S/∂ℓ²_e)².
    // F ≥ 0 and F = 0 exactly at a stationary point of S (= Regge equations).
    // We cannot minimize S directly because it is unbounded below.

#ifdef TESSERA_CUDA
    // GPU path: 2 kernel launches total.
    //   1. Base action gradient ∂S/∂W_e  (one thread per edge)
    //   2. Fused ∂F/∂W_j  (one thread per edge, using edge neighborhoods)
    // All GPU memory allocated/uploaded/downloaded/freed in one cycle.
    auto mesh = flattenMeshForGpu();

    std::vector<double> g0(n);
    std::vector<double> dF(n, 0.0);
    cuda::compute_step_gpu(mesh, g0.data(), dF.data());

    double F = 0.0;
    for (double gi : g0) F += gi * gi;

    // Update in Wick-rotated (W) space, preserving edge signature.
    // dF[j] is ∂F/∂W_j; gradient descent: W_j -= lr · ∂F/∂W_j.
    for (int j = 0; j < n; ++j) {
        const bool tl = edges[j]->isTimelike();
        double W = std::abs(edges[j]->getSquaredLength());     // |l^2|
        double W_new = W - learningRate * dF[j];
        if (W_new < 1e-12) W_new = 1e-12;
        edges[j]->setSquaredLength(tl ? std::complex<double>{-W_new, 0.0}
                                      : std::complex<double>{W_new, 0.0});
    }
#else
    // CPU path: gradient descent on ∂S/∂ℓ² directly.
    // At the solution, ∂S/∂ℓ² = 0 (the Regge equations).
    auto g = actionGradient();
    double F = 0.0;
    for (double gi : g) F += gi * gi;

    // Update in Wick-rotated space, preserving edge signature.
    // Clamp per-edge change to at most 5% of magnitude to prevent
    // overshooting.
    for (int j = 0; j < n; ++j) {
        const bool tl = edges[j]->isTimelike();
        double W = std::abs(edges[j]->getSquaredLength());     // |l^2|
        double delta = learningRate * g[j];
        double maxDelta = W * 0.05;
        delta = std::clamp(delta, -maxDelta, maxDelta);
        double W_new = W - delta;
        if (W_new < 1e-12) W_new = 1e-12;
        edges[j]->setSquaredLength(tl ? std::complex<double>{-W_new, 0.0}
                                      : std::complex<double>{W_new, 0.0});
    }
#endif

    return F;
}

// =====================================================================
// Solve
// =====================================================================

std::tuple<bool, double, int> ReggeSolver::solve(
    double tol, int maxIters, double learningRate,
    ProgressCallback progress) {
    double F = 0.0;
    double F0 = -1.0; // initial F, for relative tolerance
    for (int i = 0; i < maxIters; ++i) {
        F = step(learningRate);   // returns ||∇S||² before the update
        if (F0 < 0.0) F0 = F;
        if (progress) progress(i, F);
        // Converge when F < tol (absolute) or F < tol * F0 (relative)
        double threshold = std::max(tol, tol * F0);
        if (F < threshold) {
            return {true, F, i + 1};
        }
    }
    return {false, F, maxIters};
}

} // namespace tessera::simulations
