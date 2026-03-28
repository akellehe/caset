// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "matter/MatterConfiguration.h"
#include "spacetime/Spacetime.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Fingerprint.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace caset {

void MatterConfiguration::setPointMass(VertexPtr vertex, double mass) {
    pointMasses_[vertex->getId()] = mass;
}

void MatterConfiguration::setWorldlineMass(VertexPtr center, double mass,
                                            const Spacetime &st) {
    auto worldline = buildWorldline(center, st);
    for (auto *v : worldline)
        pointMasses_[v->getId()] = mass;
}

void MatterConfiguration::setEnergyDensity(SimplexPtr simplex, double rho) {
    simplexRho_[simplex->fingerprint.fingerprint()] = rho;
}

// =====================================================================
// Worldline construction via spatial Chebyshev center
// =====================================================================

// Compute spacelike geodesic distances from `source` to all other vertices
// on the same time slice, using only spacelike edges (same-time neighbors).
static std::unordered_map<std::uint64_t, double>
spatialDijkstra(VertexPtr source) {
    double sourceTime = source->getTime();
    std::unordered_map<std::uint64_t, double> dist;
    using PQ = std::priority_queue<
        std::pair<double, VertexPtr>,
        std::vector<std::pair<double, VertexPtr>>,
        std::greater<>>;
    PQ pq;
    dist[source->getId()] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();
        if (d > dist[v->getId()]) continue;
        for (const auto &e : v->getEdges()) {
            VertexPtr other = (e->getSource()->getId() == v->getId())
                                  ? e->getTarget() : e->getSource();
            // Only follow spacelike edges (same time slice)
            if (std::abs(other->getTime() - sourceTime) > 0.5) continue;
            double edgeLen = std::sqrt(std::abs(e->getSquaredLength()));
            double nd = d + edgeLen;
            auto it = dist.find(other->getId());
            if (it == dist.end() || nd < it->second) {
                dist[other->getId()] = nd;
                pq.push({nd, other});
            }
        }
    }
    return dist;
}

// Find the Chebyshev center of a spatial slice: the vertex with minimum
// eccentricity (minimum of maximum geodesic distance to any other vertex).
static VertexPtr findSliceCenter(const std::vector<VertexPtr> &sliceVerts) {
    if (sliceVerts.empty()) return nullptr;
    if (sliceVerts.size() == 1) return sliceVerts[0];

    VertexPtr bestVertex = nullptr;
    double bestEccentricity = std::numeric_limits<double>::max();

    for (auto *v : sliceVerts) {
        auto dists = spatialDijkstra(v);
        double eccentricity = 0.0;
        for (auto *u : sliceVerts) {
            auto it = dists.find(u->getId());
            if (it != dists.end())
                eccentricity = std::max(eccentricity, it->second);
        }
        if (eccentricity < bestEccentricity) {
            bestEccentricity = eccentricity;
            bestVertex = v;
        }
    }
    return bestVertex;
}

std::vector<VertexPtr> MatterConfiguration::buildWorldline(
    VertexPtr center, const Spacetime &st) {

    // Group all vertices by time slice
    std::unordered_map<int, std::vector<VertexPtr>> slices;
    for (auto *v : st.getVertexList()->liveVector()) {
        int t = static_cast<int>(std::round(v->getTime()));
        slices[t].push_back(v);
    }

    // Find all unique times, sorted
    std::vector<int> times;
    times.reserve(slices.size());
    for (auto &[t, _] : slices) times.push_back(t);
    std::sort(times.begin(), times.end());

    // On the center's slice, use the provided vertex.
    // On every other slice, find the Chebyshev center independently.
    int centerTime = static_cast<int>(std::round(center->getTime()));

    std::vector<VertexPtr> worldline;
    worldline.reserve(times.size());
    for (int t : times) {
        if (t == centerTime) {
            worldline.push_back(center);
        } else {
            VertexPtr sliceCenter = findSliceCenter(slices[t]);
            if (sliceCenter) worldline.push_back(sliceCenter);
        }
    }
    return worldline;
}

// =====================================================================
// Hinge classification
// =====================================================================

HingeType MatterConfiguration::classifyHinge(SimplexPtr hinge) {
    auto verts = hinge->getVertices();
    if (verts.empty()) return HingeType::SPATIAL;

    double t0 = verts[0]->getTime();
    for (std::size_t i = 1; i < verts.size(); ++i) {
        if (std::abs(verts[i]->getTime() - t0) > 0.5)
            return HingeType::TIMELIKE;
    }
    return HingeType::SPATIAL;
}

void MatterConfiguration::setRadialProfile(
    VertexPtr center, std::function<double(double)> rho_of_r) {
    radialProfiles_.push_back({center, std::move(rho_of_r)});
}

// =====================================================================
// Geodesic distance (Dijkstra on edge-length-weighted graph)
// =====================================================================

static std::unordered_map<std::uint64_t, double>
geodesicDistances(const Spacetime &st, VertexPtr center) {
    std::unordered_map<std::uint64_t, double> dist;
    using PQ = std::priority_queue<
        std::pair<double, VertexPtr>,
        std::vector<std::pair<double, VertexPtr>>,
        std::greater<>>;
    PQ pq;
    dist[center->getId()] = 0.0;
    pq.push({0.0, center});

    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();
        if (d > dist[v->getId()]) continue;
        for (const auto &e : v->getEdges()) {
            double edgeLen = std::sqrt(std::abs(e->getSquaredLength()));
            VertexPtr neighbor = (e->getSource()->getId() == v->getId())
                                     ? e->getTarget()
                                     : e->getSource();
            double nd = d + edgeLen;
            auto it = dist.find(neighbor->getId());
            if (it == dist.end() || nd < it->second) {
                dist[neighbor->getId()] = nd;
                pq.push({nd, neighbor});
            }
        }
    }
    return dist;
}

// =====================================================================
// Compute target deficits
// =====================================================================

std::unordered_map<std::uint64_t, double>
MatterConfiguration::computeTargetDeficits(const Spacetime &st) const {
    int d = st.getMetric()->getSignature()->getDimensions();
    int hingeSize = d - 1; // (d-2)-simplex has (d-1) vertices

    // Collect all hinges
    std::unordered_map<std::uint64_t, SimplexPtr> hinges;
    for (const auto &s : st.getSimplices()) {
        if (static_cast<int>(s->size()) == hingeSize)
            hinges[s->fingerprint.fingerprint()] = s;
    }

    // Initialize targets to 0 (flat space → deficit = 0 everywhere)
    std::unordered_map<std::uint64_t, double> targets;
    for (auto &[fp, h] : hinges)
        targets[fp] = 0.0;

    // Point masses: distribute 8πM equally among SPATIAL hinges incident
    // to the vertex.  The Hamiltonian constraint sources 3D spatial curvature
    // from energy density; timelike hinges encode time evolution, not matter.
    for (auto &[vid, mass] : pointMasses_) {
        std::vector<std::uint64_t> incidentSpatialHinges;
        for (auto &[fp, h] : hinges) {
            if (classifyHinge(h) != HingeType::SPATIAL) continue;
            for (const auto &v : h->getVertices()) {
                if (v->getId() == vid) {
                    incidentSpatialHinges.push_back(fp);
                    break;
                }
            }
        }
        if (!incidentSpatialHinges.empty()) {
            double deficitPerHinge = 8.0 * std::numbers::pi * mass
                                     / static_cast<double>(incidentSpatialHinges.size());
            for (auto fp : incidentSpatialHinges)
                targets[fp] += deficitPerHinge;
        }
    }

    // Radial profiles: evaluate ρ(r) only at SPATIAL hinges.
    for (auto &profile : radialProfiles_) {
        auto dists = geodesicDistances(st, profile.center);
        for (auto &[fp, h] : hinges) {
            if (classifyHinge(h) != HingeType::SPATIAL) continue;

            double avgDist = 0.0;
            for (const auto &v : h->getVertices())
                avgDist += dists[v->getId()];
            avgDist /= static_cast<double>(h->size());

            double rho = profile.rho(avgDist);
            targets[fp] += 8.0 * std::numbers::pi * rho;
        }
    }

    return targets;
}

} // namespace caset
