// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "observables/WilsonLoop.h"
#include "spacetime/Spacetime.h"
#include "mesh/Simplex.h"
#include "mesh/Edge.h"
#include "mesh/Vertex.h"
#include "mesh/Fingerprint.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace tessera {

// =====================================================================
// Construction
// =====================================================================

WilsonLoop::WilsonLoop(std::shared_ptr<Spacetime> spacetime)
    : spacetime_(std::move(spacetime)),
      d_(spacetime_->getMetric()->getSignature()->getDimensions()) {
    // Ensure hinges are registered (same pattern as ReggeSolver ctor).
    auto nBefore = spacetime_->getSimplices().size();
    for (std::size_t i = 0; i < nBefore; ++i) {
        auto s = spacetime_->getSimplices()[i];
        if (static_cast<int>(s->size()) == d_)
            s->getFacets();
    }
}

// =====================================================================
// Dual-graph helpers
// =====================================================================

std::vector<SimplexPtr> WilsonLoop::dualNeighbors(SimplexPtr sigma) const {
    int topSize = d_ + 1;
    std::vector<SimplexPtr> nbrs;
    for (const auto &facet : sigma->getFacets()) {
        for (const auto &coface : facet->getCofaces()) {
            if (static_cast<int>(coface->size()) == topSize &&
                coface->fingerprint.fingerprint() !=
                    sigma->fingerprint.fingerprint()) {
                nbrs.push_back(coface);
            }
        }
    }
    return nbrs;
}

SimplexPtr WilsonLoop::findSharedFacet(SimplexPtr a, SimplexPtr b) const {
    for (const auto &facet : a->getFacets()) {
        for (const auto &coface : facet->getCofaces()) {
            if (coface->fingerprint.fingerprint() ==
                b->fingerprint.fingerprint())
                return facet;
        }
    }
    return nullptr;
}

LoopPath WilsonLoop::buildLoopPath(
    const std::vector<SimplexPtr> &simplices) const {
    LoopPath lp;
    lp.simplices = simplices;
    int n = static_cast<int>(simplices.size());
    lp.facets.resize(n);
    for (int i = 0; i < n; ++i) {
        lp.facets[i] = findSharedFacet(simplices[i],
                                        simplices[(i + 1) % n]);
        if (!lp.facets[i]) return {};  // non-adjacent pair → invalid loop
    }
    return lp;
}

// =====================================================================
// Loop generators
// =====================================================================

LoopPath WilsonLoop::hingeLoop(SimplexPtr hinge) const {
    int topSize = d_ + 1;
    auto hingeVerts = hinge->getVertices();
    if (hingeVerts.empty()) return {};

    // Collect all top-simplices containing the hinge.
    std::unordered_set<std::uint64_t> candidateFps;
    std::unordered_map<std::uint64_t, SimplexPtr> fpToSigma;
    for (const auto &sigma : hingeVerts[0]->getSimplices()) {
        if (static_cast<int>(sigma->size()) != topSize) continue;
        bool all = true;
        for (std::size_t i = 1; i < hingeVerts.size(); ++i) {
            if (!sigma->hasVertex(hingeVerts[i])) { all = false; break; }
        }
        if (all) {
            auto fp = sigma->fingerprint.fingerprint();
            candidateFps.insert(fp);
            fpToSigma[fp] = sigma;
        }
    }
    if (candidateFps.size() < 2) return {};

    // Order cyclically: walk adjacency through facets containing the hinge.
    auto containsHinge = [&](SimplexPtr facet) {
        for (const auto &hv : hingeVerts)
            if (!facet->hasVertex(hv)) return false;
        return true;
    };

    std::vector<SimplexPtr> ordered;
    auto startFp = *candidateFps.begin();
    ordered.push_back(fpToSigma[startFp]);
    std::unordered_set<std::uint64_t> visited{startFp};

    while (visited.size() < candidateFps.size()) {
        auto current = ordered.back();
        bool found = false;
        for (const auto &facet : current->getFacets()) {
            if (!containsHinge(facet)) continue;
            for (const auto &coface : facet->getCofaces()) {
                auto fp = coface->fingerprint.fingerprint();
                if (candidateFps.count(fp) && !visited.count(fp)) {
                    ordered.push_back(coface);
                    visited.insert(fp);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (!found) break;  // open chain (boundary)
    }

    return buildLoopPath(ordered);
}

LoopPath WilsonLoop::geodesicLoop(SimplexPtr start) const {
    using FP = std::uint64_t;
    FP startFp = start->fingerprint.fingerprint();
    int topSize = d_ + 1;

    std::unordered_map<FP, FP> parent;       // fp -> parent fp
    std::unordered_map<FP, SimplexPtr> fpMap; // fp -> simplex
    parent[startFp] = startFp;               // self = root
    fpMap[startFp] = start;

    std::queue<SimplexPtr> q;
    q.push(start);

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        auto curFp = cur->fingerprint.fingerprint();

        for (const auto &nbr : dualNeighbors(cur)) {
            auto nbrFp = nbr->fingerprint.fingerprint();
            if (parent.find(nbrFp) == parent.end()) {
                parent[nbrFp] = curFp;
                fpMap[nbrFp] = nbr;
                q.push(nbr);
            } else if (nbrFp != parent[curFp]) {
                // Found a cycle: trace cur→start and nbr→start, combine.
                auto trace = [&](FP fp) {
                    std::vector<SimplexPtr> path;
                    while (fp != startFp) {
                        path.push_back(fpMap[fp]);
                        fp = parent[fp];
                    }
                    path.push_back(start);
                    return path;
                };
                auto p1 = trace(curFp);
                auto p2 = trace(nbrFp);
                std::reverse(p1.begin(), p1.end());
                // p1 goes start → ... → cur, p2 goes nbr → ... → start
                // Drop the duplicate start at end of p2
                if (!p2.empty()) p2.pop_back();
                p1.insert(p1.end(), p2.begin(), p2.end());
                return buildLoopPath(p1);
            }
        }
    }
    return {};  // shouldn't happen on a closed manifold
}

LoopPath WilsonLoop::dualLatticeLoop(SimplexPtr start,
                                      int targetLength) const {
    using FP = std::uint64_t;
    FP startFp = start->fingerprint.fingerprint();
    int maxDepth = std::max(targetLength / 2, 2);

    std::unordered_map<FP, FP> parent;
    std::unordered_map<FP, int> depth;
    std::unordered_map<FP, SimplexPtr> fpMap;
    parent[startFp] = startFp;
    depth[startFp] = 0;
    fpMap[startFp] = start;

    std::queue<SimplexPtr> q;
    q.push(start);

    // Track the best cycle found
    std::vector<SimplexPtr> bestCycle;
    int bestDiff = targetLength + 1;

    auto trace = [&](FP fp) {
        std::vector<SimplexPtr> path;
        while (fp != startFp) {
            path.push_back(fpMap[fp]);
            fp = parent[fp];
        }
        path.push_back(start);
        return path;
    };

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        auto curFp = cur->fingerprint.fingerprint();
        if (depth[curFp] >= maxDepth) continue;

        for (const auto &nbr : dualNeighbors(cur)) {
            auto nbrFp = nbr->fingerprint.fingerprint();
            if (parent.find(nbrFp) == parent.end()) {
                parent[nbrFp] = curFp;
                depth[nbrFp] = depth[curFp] + 1;
                fpMap[nbrFp] = nbr;
                q.push(nbr);
            } else if (nbrFp != parent[curFp] && depth[curFp] >= 1) {
                // Found a cycle
                auto p1 = trace(curFp);
                auto p2 = trace(nbrFp);
                std::reverse(p1.begin(), p1.end());
                if (!p2.empty()) p2.pop_back();
                p1.insert(p1.end(), p2.begin(), p2.end());
                int len = static_cast<int>(p1.size());
                int diff = std::abs(len - targetLength);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestCycle = p1;
                    if (diff == 0) goto done;
                }
            }
        }
    }
done:
    if (bestCycle.empty()) return geodesicLoop(start);
    return buildLoopPath(bestCycle);
}

// =====================================================================
// Evaluation modes
// =====================================================================

WilsonResult WilsonLoop::evaluate(const LoopPath &loop,
                                   WilsonMode mode) const {
    switch (mode) {
        case WilsonMode::COMBINATORIAL: return evaluateCombinatorial(loop);
        case WilsonMode::DEFICIT_ANGLE: return evaluateDeficitAngle(loop);
        case WilsonMode::CAUSAL:        return evaluateCausal(loop);
    }
    return {};
}

// ---- Combinatorial ----

WilsonResult WilsonLoop::evaluateCombinatorial(const LoopPath &loop) const {
    WilsonResult r;
    r.loopSize = static_cast<int>(loop.simplices.size());
    if (r.loopSize < 2) return r;

    int hingeSize = d_ - 1;  // (d-2)-simplex has (d-1) vertices

    // Collect fingerprints of all loop simplices.
    std::unordered_set<std::uint64_t> loopFps;
    for (const auto &s : loop.simplices)
        loopFps.insert(s->fingerprint.fingerprint());

    // Count hinges shared by ALL loop simplices (enclosed hinges).
    // A hinge is "enclosed" if every loop simplex contains it.
    // For hinge loops this is 1 (the hinge itself); for general loops
    // we count hinges present in every simplex of the loop.
    int enclosed = 0;
    if (!loop.simplices.empty()) {
        for (const auto &h : spacetime_->getSimplices()) {
            if (static_cast<int>(h->size()) != hingeSize) continue;
            auto hv = h->getVertices();
            bool inAll = true;
            for (const auto &sigma : loop.simplices) {
                bool has = true;
                for (const auto &v : hv)
                    if (!sigma->hasVertex(v)) { has = false; break; }
                if (!has) { inAll = false; break; }
            }
            if (inAll) ++enclosed;
        }
    }
    r.enclosedHinges = enclosed;
    r.contractible = (enclosed == 0);
    r.value = static_cast<double>(r.loopSize);
    return r;
}

// ---- Deficit angle ----

WilsonResult WilsonLoop::evaluateDeficitAngle(const LoopPath &loop) const {
    WilsonResult r;
    r.loopSize = static_cast<int>(loop.simplices.size());
    if (r.loopSize < 2) return r;

    int hingeSize = d_ - 1;

    // Find hinges shared by ALL loop simplices (same as combinatorial).
    std::vector<SimplexPtr> enclosedHinges;
    for (const auto &h : spacetime_->getSimplices()) {
        if (static_cast<int>(h->size()) != hingeSize) continue;
        auto hv = h->getVertices();
        bool inAll = true;
        for (const auto &sigma : loop.simplices) {
            bool has = true;
            for (const auto &v : hv)
                if (!sigma->hasVertex(v)) { has = false; break; }
            if (!has) { inAll = false; break; }
        }
        if (inAll) enclosedHinges.push_back(h);
    }
    r.enclosedHinges = static_cast<int>(enclosedHinges.size());

    if (enclosedHinges.size() == 1) {
        // Hinge loop: exact Wilson loop value
        double eps = enclosedHinges[0]->deficitAngle();
        r.value = (static_cast<double>(d_ - 2) + 2.0 * std::cos(eps))
                  / static_cast<double>(d_);
    } else {
        // General loop: U(1) approximation — product of cos(epsilon)
        double product = 1.0;
        for (const auto &h : enclosedHinges)
            product *= std::cos(h->deficitAngle());
        r.value = product;
    }
    return r;
}

// ---- Causal ----

WilsonResult WilsonLoop::evaluateCausal(const LoopPath &loop) const {
    WilsonResult r;
    r.loopSize = static_cast<int>(loop.simplices.size());
    if (r.loopSize < 2) return r;

    // Track how the time orientation changes around the loop.
    // At each face crossing, compare tf of current simplex with tf of next.
    int winding = 0;
    int n = r.loopSize;
    for (int i = 0; i < n; ++i) {
        double tf_cur  = loop.simplices[i]->getTf();
        double tf_next = loop.simplices[(i + 1) % n]->getTf();
        if (tf_next > tf_cur + 0.5) ++winding;
        else if (tf_next < tf_cur - 0.5) --winding;
    }
    r.causalWindingNumber = winding;
    r.value = static_cast<double>(winding);
    return r;
}

// =====================================================================
// Measurement accumulation
// =====================================================================

void WilsonLoop::measure(const LoopPath &loop, WilsonMode mode) {
    measurements_.push_back(evaluate(loop, mode));
}

void WilsonLoop::measureAllHinges(WilsonMode mode) {
    int hingeSize = d_ - 1;
    for (const auto &s : spacetime_->getSimplices()) {
        if (static_cast<int>(s->size()) != hingeSize) continue;
        auto loop = hingeLoop(s);
        if (loop.simplices.size() >= 2)
            measurements_.push_back(evaluate(loop, mode));
    }
}

void WilsonLoop::reset() { measurements_.clear(); }

const std::vector<WilsonResult> &WilsonLoop::getMeasurements() const {
    return measurements_;
}

std::map<int, double> WilsonLoop::getAverageBySize() const {
    std::map<int, double> sums;
    std::map<int, int> counts;
    for (const auto &r : measurements_) {
        sums[r.loopSize] += r.value;
        counts[r.loopSize]++;
    }
    std::map<int, double> avg;
    for (const auto &[sz, s] : sums)
        avg[sz] = s / counts[sz];
    return avg;
}

} // namespace tessera
