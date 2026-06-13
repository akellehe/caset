// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "ForceLayout.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace tessera {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
} // namespace

std::vector<double> ForceLayout::layout3D(
    int n,
    const std::vector<std::pair<int, int>> &edges,
    int centerIdx,
    const std::vector<double> &initPos,
    const std::vector<double> &restLengths,
    double springK,
    double repulsionK,
    int iters,
    double cooling,
    int repulsionCap,
    int seed) {

    // Initialize positions
    std::vector<double> pos(n * 3, 0.0);
    if (!initPos.empty() && static_cast<int>(initPos.size()) == n * 3) {
        pos = initPos;
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 0.5);
        for (int i = 0; i < n * 3; ++i)
            pos[i] = normal(rng);
    }

    constexpr double eps = 1e-6;
    double step = 0.5;
    std::vector<double> forces(n * 3, 0.0);

    for (int iter = 0; iter < iters; ++iter) {
        std::fill(forces.begin(), forces.end(), 0.0);

        // Spring forces along edges
        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            int a = edges[ei].first;
            int b = edges[ei].second;
            double dx = pos[b * 3]     - pos[a * 3];
            double dy = pos[b * 3 + 1] - pos[a * 3 + 1];
            double dz = pos[b * 3 + 2] - pos[a * 3 + 2];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            dist = std::max(dist, eps);
            double rl = (!restLengths.empty()) ? restLengths[ei] : 1.0;
            double f = springK * (dist - rl) / dist;
            forces[a * 3]     += f * dx;
            forces[a * 3 + 1] += f * dy;
            forces[a * 3 + 2] += f * dz;
            forces[b * 3]     -= f * dx;
            forces[b * 3 + 1] -= f * dy;
            forces[b * 3 + 2] -= f * dz;
        }

        // Repulsion (capped for large graphs)
        int cap = std::min(n, repulsionCap);
        for (int a = 0; a < cap; ++a) {
            for (int b = a + 1; b < cap; ++b) {
                double dx = pos[a * 3]     - pos[b * 3];
                double dy = pos[a * 3 + 1] - pos[b * 3 + 1];
                double dz = pos[a * 3 + 2] - pos[b * 3 + 2];
                double d2 = dx * dx + dy * dy + dz * dz + eps;
                double dist = std::sqrt(d2);
                double f = repulsionK / d2 / dist;
                forces[a * 3]     += f * dx;
                forces[a * 3 + 1] += f * dy;
                forces[a * 3 + 2] += f * dz;
                forces[b * 3]     -= f * dx;
                forces[b * 3 + 1] -= f * dy;
                forces[b * 3 + 2] -= f * dz;
            }
        }

        // Pin center node
        if (centerIdx >= 0 && centerIdx < n) {
            forces[centerIdx * 3]     = 0.0;
            forces[centerIdx * 3 + 1] = 0.0;
            forces[centerIdx * 3 + 2] = 0.0;
        }

        // Clamp and apply forces
        for (int i = 0; i < n; ++i) {
            double fx = forces[i * 3];
            double fy = forces[i * 3 + 1];
            double fz = forces[i * 3 + 2];
            double mag = std::sqrt(fx * fx + fy * fy + fz * fz);
            if (mag > step) {
                double s = step / mag;
                fx *= s; fy *= s; fz *= s;
            }
            pos[i * 3]     += fx;
            pos[i * 3 + 1] += fy;
            pos[i * 3 + 2] += fz;
        }

        step *= cooling;
    }

    return pos;
}

std::vector<double> ForceLayout::layout2D(
    int n,
    const std::vector<std::pair<int, int>> &edges,
    const std::vector<double> &targetRadii,
    const std::vector<int> &groups,
    int centerIdx,
    const std::vector<double> &initPos,
    const std::vector<double> &restLengths,
    double springK,
    double repulsionK,
    int iters,
    double cooling,
    int repulsionCap,
    double initialStep,
    int seed) {

    constexpr double eps = 1e-6;
    const bool radial  = (static_cast<int>(targetRadii.size()) == n);
    const bool grouped = (static_cast<int>(groups.size()) == n);

    // Initialize 2D positions
    std::vector<double> pos(n * 2, 0.0);
    if (!initPos.empty() && static_cast<int>(initPos.size()) == n * 2) {
        pos = initPos;
    } else {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> angle(0.0, kTwoPi);
        std::normal_distribution<double> normal(0.0, 0.5);
        for (int i = 0; i < n; ++i) {
            if (radial) {
                double r = (i == centerIdx) ? 0.0 : targetRadii[i];
                double a = angle(rng);
                pos[i * 2]     = r * std::cos(a);
                pos[i * 2 + 1] = r * std::sin(a);
            } else {
                pos[i * 2]     = normal(rng);
                pos[i * 2 + 1] = normal(rng);
            }
        }
    }

    // Pinned radii for the radius-constrained mode.
    std::vector<double> radii;
    if (radial) {
        radii = targetRadii;
        if (centerIdx >= 0 && centerIdx < n)
            radii[centerIdx] = 0.0;
    }

    // Bucket nodes by group so repulsion can run intra-group only.
    std::vector<std::vector<int>> buckets;
    if (grouped) {
        std::unordered_map<int, int> groupToBucket;
        for (int i = 0; i < n; ++i) {
            auto it = groupToBucket.find(groups[i]);
            if (it == groupToBucket.end()) {
                groupToBucket.emplace(groups[i],
                                      static_cast<int>(buckets.size()));
                buckets.emplace_back();
                buckets.back().push_back(i);
            } else {
                buckets[it->second].push_back(i);
            }
        }
    }

    double step = initialStep;
    std::vector<double> forces(n * 2, 0.0);

    for (int iter = 0; iter < iters; ++iter) {
        std::fill(forces.begin(), forces.end(), 0.0);

        // Spring forces along edges
        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            int a = edges[ei].first;
            int b = edges[ei].second;
            double dx = pos[b * 2]     - pos[a * 2];
            double dy = pos[b * 2 + 1] - pos[a * 2 + 1];
            double dist = std::sqrt(dx * dx + dy * dy);
            dist = std::max(dist, eps);
            double rl = (!restLengths.empty()) ? restLengths[ei] : 1.0;
            double f = springK * (dist - rl) / dist;
            forces[a * 2]     += f * dx;
            forces[a * 2 + 1] += f * dy;
            forces[b * 2]     -= f * dx;
            forces[b * 2 + 1] -= f * dy;
        }

        // Coulomb repulsion: within each group if grouped, else global.
        if (grouped) {
            for (const auto &bucket : buckets) {
                int m = std::min(static_cast<int>(bucket.size()), repulsionCap);
                for (int ia = 0; ia < m; ++ia) {
                    for (int ib = ia + 1; ib < m; ++ib) {
                        repel2D(pos, forces, bucket[ia], bucket[ib],
                                repulsionK, eps);
                    }
                }
            }
        } else {
            int cap = std::min(n, repulsionCap);
            for (int a = 0; a < cap; ++a)
                for (int b = a + 1; b < cap; ++b)
                    repel2D(pos, forces, a, b, repulsionK, eps);
        }

        // Pin center node
        if (centerIdx >= 0 && centerIdx < n) {
            forces[centerIdx * 2]     = 0.0;
            forces[centerIdx * 2 + 1] = 0.0;
        }

        // Radius constraint: project out radial force so nodes move only
        // tangentially (keeps each node on its target circle).
        if (radial) {
            for (int i = 0; i < n; ++i) {
                if (i == centerIdx)
                    continue;
                double r = std::sqrt(pos[i * 2] * pos[i * 2] +
                                     pos[i * 2 + 1] * pos[i * 2 + 1]);
                if (r < eps)
                    continue;
                double rx = pos[i * 2] / r;
                double ry = pos[i * 2 + 1] / r;
                double fr = forces[i * 2] * rx + forces[i * 2 + 1] * ry;
                forces[i * 2]     -= fr * rx;
                forces[i * 2 + 1] -= fr * ry;
            }
        }

        // Clamp and apply forces
        for (int i = 0; i < n; ++i) {
            double fx = forces[i * 2];
            double fy = forces[i * 2 + 1];
            double mag = std::sqrt(fx * fx + fy * fy);
            if (mag > step) {
                double s = step / mag;
                fx *= s; fy *= s;
            }
            pos[i * 2]     += fx;
            pos[i * 2 + 1] += fy;
        }

        // Re-snap to the pinned radius after moving (radius-constrained mode).
        if (radial) {
            for (int i = 0; i < n; ++i) {
                if (i == centerIdx)
                    continue;
                double r = std::sqrt(pos[i * 2] * pos[i * 2] +
                                     pos[i * 2 + 1] * pos[i * 2 + 1]);
                if (r > eps) {
                    double s = radii[i] / r;
                    pos[i * 2]     *= s;
                    pos[i * 2 + 1] *= s;
                }
            }
        }

        step *= cooling;
    }

    return pos;
}

void ForceLayout::repel2D(std::vector<double> &pos,
                          std::vector<double> &forces,
                          int a, int b, double repulsionK, double eps) {
    double dx = pos[a * 2]     - pos[b * 2];
    double dy = pos[a * 2 + 1] - pos[b * 2 + 1];
    double d2 = dx * dx + dy * dy + eps;
    double dist = std::sqrt(d2);
    double f = repulsionK / d2 / dist;
    forces[a * 2]     += f * dx;
    forces[a * 2 + 1] += f * dy;
    forces[b * 2]     -= f * dx;
    forces[b * 2 + 1] -= f * dy;
}

} // namespace tessera
