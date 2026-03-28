// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "matter/MatterConfiguration.h"
#include "spacetime/Spacetime.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Fingerprint.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace caset {

void MatterConfiguration::setWorldlineMass(VertexPtr center, double mass,
                                            const Spacetime &st) {
    worldlines_.push_back({buildWorldline(center, st), mass});
}

void MatterConfiguration::setEnergyDensity(SimplexPtr simplex, double rho) {
    simplexRho_[simplex->fingerprint.fingerprint()] = rho;
}

// =====================================================================
// Worldline construction by following timelike edges
// =====================================================================

std::vector<VertexPtr> MatterConfiguration::buildWorldline(
    VertexPtr center, const Spacetime &st) {

    // Determine all time slices present in the triangulation
    std::set<int> timeSet;
    for (auto *v : st.getVertexList()->liveVector())
        timeSet.insert(static_cast<int>(std::round(v->getTime())));
    std::vector<int> times(timeSet.begin(), timeSet.end());

    int centerTime = static_cast<int>(std::round(center->getTime()));

    // Helper: among timelike neighbors of `current` on slice `targetTime`,
    // pick the one sharing the most spacelike neighbors (same spatial position).
    auto bestNeighborOnSlice = [](VertexPtr current, int targetTime) -> VertexPtr {
        double curTime = current->getTime();

        // Collect current vertex's spacelike neighbors (same slice)
        std::unordered_set<std::uint64_t> spatialNbrs;
        for (const auto &e : current->getEdges()) {
            VertexPtr other = (e->getSource()->getId() == current->getId())
                              ? e->getTarget() : e->getSource();
            if (std::abs(other->getTime() - curTime) < 0.5)
                spatialNbrs.insert(other->getId());
        }

        VertexPtr best = nullptr;
        int bestScore = -1;

        for (const auto &e : current->getEdges()) {
            VertexPtr other = (e->getSource()->getId() == current->getId())
                              ? e->getTarget() : e->getSource();
            if (static_cast<int>(std::round(other->getTime())) != targetTime)
                continue;

            // Score: number of other's same-slice neighbors that overlap
            // with current's same-slice neighbors
            int score = 0;
            for (const auto &e2 : other->getEdges()) {
                VertexPtr nbr = (e2->getSource()->getId() == other->getId())
                                ? e2->getTarget() : e2->getSource();
                if (spatialNbrs.count(nbr->getId())) ++score;
            }

            if (score > bestScore) {
                bestScore = score;
                best = other;
            }
        }
        return best;
    };

    // Find center's index in sorted times
    auto centerIt = std::find(times.begin(), times.end(), centerTime);
    int centerIdx = static_cast<int>(centerIt - times.begin());

    // Trace forward (increasing time)
    std::vector<VertexPtr> forward;
    VertexPtr current = center;
    for (int i = centerIdx + 1; i < static_cast<int>(times.size()); ++i) {
        VertexPtr next = bestNeighborOnSlice(current, times[i]);
        if (!next) break;
        forward.push_back(next);
        current = next;
    }

    // Trace backward (decreasing time)
    std::vector<VertexPtr> backward;
    current = center;
    for (int i = centerIdx - 1; i >= 0; --i) {
        VertexPtr prev = bestNeighborOnSlice(current, times[i]);
        if (!prev) break;
        backward.push_back(prev);
        current = prev;
    }

    // Combine: backward (reversed) + center + forward
    std::reverse(backward.begin(), backward.end());
    std::vector<VertexPtr> worldline;
    worldline.reserve(backward.size() + 1 + forward.size());
    worldline.insert(worldline.end(), backward.begin(), backward.end());
    worldline.push_back(center);
    worldline.insert(worldline.end(), forward.begin(), forward.end());

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

} // namespace caset
