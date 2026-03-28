// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "ForceLayout.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace caset {

std::vector<double> forceLayout3D(
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

} // namespace caset
