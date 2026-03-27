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

void MatterConfiguration::setEnergyDensity(SimplexPtr simplex, double rho) {
    simplexRho_[simplex->fingerprint.fingerprint()] = rho;
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

    // Point masses: distribute 8πM equally among hinges incident to the vertex
    for (auto &[vid, mass] : pointMasses_) {
        // Find hinges containing this vertex
        std::vector<std::uint64_t> incidentHinges;
        for (auto &[fp, h] : hinges) {
            for (const auto &v : h->getVertices()) {
                if (v->getId() == vid) {
                    incidentHinges.push_back(fp);
                    break;
                }
            }
        }
        if (!incidentHinges.empty()) {
            double deficitPerHinge = 8.0 * std::numbers::pi * mass
                                     / static_cast<double>(incidentHinges.size());
            for (auto fp : incidentHinges)
                targets[fp] += deficitPerHinge;
        }
    }

    // Radial profiles: evaluate ρ(r) at each hinge's centroid distance
    for (auto &profile : radialProfiles_) {
        auto dists = geodesicDistances(st, profile.center);
        for (auto &[fp, h] : hinges) {
            // Average geodesic distance of hinge vertices from center
            double avgDist = 0.0;
            for (const auto &v : h->getVertices())
                avgDist += dists[v->getId()];
            avgDist /= static_cast<double>(h->size());

            double rho = profile.rho(avgDist);
            // Convert energy density to deficit angle contribution:
            // ε ≈ 8π ρ V_local / A_hinge (rough discretization)
            // For simplicity, use ε += 8π ρ (dimensionless in geometrized units)
            targets[fp] += 8.0 * std::numbers::pi * rho;
        }
    }

    return targets;
}

} // namespace caset
