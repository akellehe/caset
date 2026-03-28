// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "mesh/Simplex.h"
#include "mesh/Edge.h"
#include "mesh/Vertex.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace caset {

// =====================================================================
// Construction
// =====================================================================

ReggeSolver::ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                         MatterConfiguration matter)
    : spacetime_(std::move(spacetime)), matter_(std::move(matter)) {
    // Ensure all sub-simplices down to hinges (d-2) are registered.
    // Top-simplices → facets (d-1) are registered during build().
    // We need facets of facets → hinges (d-2).
    // Use index-based iteration since getFacets() may grow simplicesVec.
    {
        int d = spacetime_->getMetric()->getSignature()->getDimensions();
        auto nBefore = spacetime_->getSimplices().size();
        for (std::size_t i = 0; i < nBefore; ++i) {
            auto s = spacetime_->getSimplices()[i];
            if (static_cast<int>(s->size()) == d) // (d-1)-simplices
                s->getFacets(); // registers (d-2)-simplices (hinges)
        }
    }
    targetDeficits_ = matter_.computeTargetDeficits(*spacetime_);
}

// =====================================================================
// Linear algebra helpers (small dense matrices, no external deps)
// =====================================================================

double ReggeSolver::determinant(const std::vector<double> &M, int n) {
    if (n == 1) return M[0];
    if (n == 2) return M[0] * M[3] - M[1] * M[2];

    // LU decomposition with partial pivoting
    std::vector<double> A(M);
    double det = 1.0;
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot = col;
        double maxVal = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > maxVal) { maxVal = val; pivot = row; }
        }
        if (maxVal < 1e-15) return 0.0;
        if (pivot != col) {
            for (int j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[pivot * n + j]);
            det = -det;
        }
        det *= A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / A[col * n + col];
            for (int j = col + 1; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
        }
    }
    return det;
}

std::vector<double> ReggeSolver::cofactorMatrix(
    const std::vector<double> &M, int n) {
    std::vector<double> C(n * n, 0.0);
    if (n == 1) { C[0] = 1.0; return C; }

    std::vector<double> sub((n - 1) * (n - 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Build (n-1)×(n-1) minor by removing row i, col j
            int si = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                int sj = 0;
                for (int c = 0; c < n; ++c) {
                    if (c == j) continue;
                    sub[si * (n - 1) + sj] = M[r * n + c];
                    sj++;
                }
                si++;
            }
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            C[i * n + j] = sign * determinant(sub, n - 1);
        }
    }
    return C;
}

// =====================================================================
// Gram matrix from edge lengths
// =====================================================================

std::vector<double> ReggeSolver::gramMatrix(SimplexPtr sigma) {
    // For a d-simplex with vertices {v0, v1, ..., vd}, the Gram matrix
    // G is d×d with entries:
    //   G_ij = ½(ℓ²(v0,vi) + ℓ²(v0,vj) - ℓ²(vi,vj))
    // for i,j ∈ {1, ..., d} (vertex 0 is the origin).

    auto verts = sigma->getVertices();
    int dPlus1 = static_cast<int>(verts.size());
    int d = dPlus1 - 1;
    if (d < 1) return {};

    // Build squared-distance lookup: sq[i][j] = ℓ²(vi, vj)
    // Use edges stored in the simplex
    std::unordered_map<std::uint64_t, double> sqMap;
    for (const auto &e : sigma->getEdges()) {
        auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                  Fingerprint::mix64(e->getTarget()->getId());
        sqMap[fp] = e->getSquaredLength();
    }

    auto getSq = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        auto fp = Fingerprint::mix64(verts[i]->getId()) ^
                  Fingerprint::mix64(verts[j]->getId());
        auto it = sqMap.find(fp);
        if (it != sqMap.end()) return it->second;
        return 0.0; // shouldn't happen in a valid simplex
    };

    // G_ij = ½(ℓ²(v0,vi+1) + ℓ²(v0,vj+1) - ℓ²(vi+1,vj+1))
    // indices shifted: i,j in [0,d-1] correspond to vertices 1..d
    std::vector<double> G(d * d, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            G[i * d + j] = 0.5 * (getSq(0, i + 1) + getSq(0, j + 1)
                                   - getSq(i + 1, j + 1));
        }
    }
    return G;
}

// =====================================================================
// Dihedral angle computation
// =====================================================================

double ReggeSolver::dihedralAngle(SimplexPtr sigma,
                                   SimplexPtr hinge) const {
    // Identify which two vertices of sigma are NOT in the hinge.
    // These are vertices i and j (in the simplex's local numbering).
    // The dihedral angle at the hinge is θ_ij.

    auto sigmaVerts = sigma->getVertices();
    auto hingeVerts = hinge->getVertices();
    int dPlus1 = static_cast<int>(sigmaVerts.size());
    int d = dPlus1 - 1;

    // Find the two vertices in sigma but not in hinge
    std::vector<int> opposite; // local indices in sigma
    for (int k = 0; k < dPlus1; ++k) {
        bool inHinge = false;
        for (const auto &hv : hingeVerts) {
            if (hv->getId() == sigmaVerts[k]->getId()) {
                inHinge = true;
                break;
            }
        }
        if (!inHinge) opposite.push_back(k);
    }

    if (opposite.size() != 2) {
        // Hinge doesn't have exactly (d-1) vertices in common with sigma
        return 0.0;
    }

    int vi = opposite[0];
    int vj = opposite[1];

    // Build the (d+1)×(d+1) Cayley-Menger-like Gram matrix for the
    // full simplex, then use cofactors to get cos(θ_ij).
    //
    // We use the FULL (d+1)×(d+1) matrix indexed by ALL vertices:
    //   B_ij = ½(ℓ²(v0,vi) + ℓ²(v0,vj) - ℓ²(vi,vj))  with B_00 = 0
    // But it's simpler to use the bordered matrix approach.
    //
    // Alternative: use the formula directly on the d×d Gram matrix G
    // (with vertex 0 as origin). The cofactors of G give the dihedral
    // angles between faces opposite to pairs of non-origin vertices.
    //
    // For the general case (any pair of vertices, not just non-origin):
    // we use the (d+1)×(d+1) distance matrix approach.

    // Distance matrix D where D_ij = ℓ²(vi, vj)
    std::unordered_map<std::uint64_t, double> sqMap;
    for (const auto &e : sigma->getEdges()) {
        auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                  Fingerprint::mix64(e->getTarget()->getId());
        sqMap[fp] = e->getSquaredLength();
    }

    auto getSq = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        auto fp = Fingerprint::mix64(sigmaVerts[i]->getId()) ^
                  Fingerprint::mix64(sigmaVerts[j]->getId());
        auto it = sqMap.find(fp);
        if (it != sqMap.end()) return it->second;
        return 0.0;
    };

    // Build the Cayley-Menger matrix (bordered):
    //     | 0  1   1   ... 1   |
    //     | 1  0   d01 ... d0n |
    // B = | 1  d10 0   ... d1n |
    //     | :  :   :   ... :   |
    //     | 1  dn0 dn1 ... 0   |
    //
    // where d_ij = ℓ²(vi, vj).
    // Size: (d+2) × (d+2).
    //
    // The dihedral angle between faces opposite vertices i and j is:
    //   cos(θ_ij) = -Δ_ij / √(Δ_ii · Δ_jj)
    // where Δ_ij is the cofactor of B at position (i+1, j+1)
    // (shifted by 1 because of the border row/col).

    int n = dPlus1 + 1; // (d+2)
    std::vector<double> B(n * n, 0.0);
    // First row and column: border
    for (int k = 1; k < n; ++k) {
        B[0 * n + k] = 1.0;
        B[k * n + 0] = 1.0;
    }
    // Fill distance entries (shifted by 1)
    for (int i = 0; i < dPlus1; ++i) {
        for (int j = 0; j < dPlus1; ++j) {
            B[(i + 1) * n + (j + 1)] = getSq(i, j);
        }
    }

    // Cofactors at positions (vi+1, vj+1), (vi+1, vi+1), (vj+1, vj+1)
    auto cof = cofactorMatrix(B, n);
    int bi = vi + 1;
    int bj = vj + 1;
    double Cij = cof[bi * n + bj];
    double Cii = cof[bi * n + bi];
    double Cjj = cof[bj * n + bj];

    double denom = std::sqrt(std::abs(Cii * Cjj));
    if (denom < 1e-15) return 0.0;

    double cosTheta = -Cij / denom;
    // Clamp to [-1, 1] for numerical safety
    cosTheta = std::clamp(cosTheta, -1.0, 1.0);
    return std::acos(cosTheta);
}

// =====================================================================
// Deficit angle
// =====================================================================

double ReggeSolver::deficitAngle(SimplexPtr hinge) const {
    // Find all top-simplices containing this hinge by checking which
    // top-simplices contain ALL vertices of the hinge.
    // (We can't rely on coface pointers because the hinge's direct cofaces
    // are (d-1)-simplices, not top-simplices.)
    int d = spacetime_->getMetric()->getSignature()->getDimensions();
    int topSize = d + 1;
    auto hingeVerts = hinge->getVertices();

    // Start from simplices incident to the first hinge vertex
    double sum = 0.0;
    if (hingeVerts.empty()) return 2.0 * std::numbers::pi;

    for (const auto &sigma : hingeVerts[0]->getSimplices()) {
        if (static_cast<int>(sigma->size()) != topSize) continue;
        // Check if sigma contains ALL hinge vertices
        bool containsAll = true;
        for (std::size_t i = 1; i < hingeVerts.size(); ++i) {
            if (!sigma->hasVertex(hingeVerts[i])) {
                containsAll = false;
                break;
            }
        }
        if (containsAll) {
            sum += dihedralAngle(sigma, hinge);
        }
    }

    return 2.0 * std::numbers::pi - sum;
}

// =====================================================================
// Hinge area (triangle area from edge lengths via Heron's formula)
// =====================================================================

double ReggeSolver::hingeArea(SimplexPtr hinge) {
    auto edges = hinge->getEdges();
    if (edges.size() < 3) return 0.0;

    // For a triangle with squared edge lengths a², b², c²:
    // 16 A² = 2a²b² + 2b²c² + 2c²a² - a⁴ - b⁴ - c⁴
    double a2 = std::abs(edges[0]->getSquaredLength());
    double b2 = std::abs(edges[1]->getSquaredLength());
    double c2 = std::abs(edges[2]->getSquaredLength());

    double val = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2)
                 - (a2 * a2 + b2 * b2 + c2 * c2);
    if (val <= 0.0) return 0.0;
    return std::sqrt(val) / 4.0;
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
// Regge action
// =====================================================================

double ReggeSolver::reggeAction() const {
    double S = 0.0;
    for (const auto &h : collectHinges()) {
        S += hingeArea(h) * deficitAngle(h);
    }
    return S;
}

// =====================================================================
// Residual
// =====================================================================

double ReggeSolver::residual() const {
    double L = 0.0;
    for (const auto &h : collectHinges()) {
        double eps = deficitAngle(h);
        auto fp = h->fingerprint.fingerprint();
        auto it = targetDeficits_.find(fp);
        double target = (it != targetDeficits_.end()) ? it->second : 0.0;
        double diff = eps - target;
        L += diff * diff;
    }
    return L;
}

// =====================================================================
// Gradient descent step
// =====================================================================

double ReggeSolver::step(double learningRate) {
    // Numerical gradient: perturb each edge length and measure residual change.
    // Uses relative perturbation to handle varying edge length scales.
    auto edgeList = spacetime_->getEdgeList();
    auto edges = edgeList->toVector();
    double L0 = residual();

    // First pass: compute gradients (without modifying edges yet)
    std::vector<double> grads(edges.size(), 0.0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        double origSq = edges[i]->getSquaredLength();
        double h = std::max(std::abs(origSq) * 1e-4, 1e-8);

        edges[i]->setSquaredLength(origSq + h);
        double Lp = residual();
        edges[i]->setSquaredLength(origSq);

        grads[i] = (Lp - L0) / h;
    }

    // Second pass: apply updates
    for (std::size_t i = 0; i < edges.size(); ++i) {
        double origSq = edges[i]->getSquaredLength();
        double newSq = origSq - learningRate * grads[i];
        edges[i]->setSquaredLength(newSq);
    }

    return residual();
}

// =====================================================================
// Solve
// =====================================================================

std::tuple<bool, double, int> ReggeSolver::solve(
    double tol, int maxIters, double learningRate,
    ProgressCallback progress) {
    double L = residual();
    for (int i = 0; i < maxIters; ++i) {
        if (L < tol) {
            if (progress) progress(i, L);
            return {true, L, i};
        }
        L = step(learningRate);
        if (progress) progress(i, L);
    }
    return {L < tol, L, maxIters};
}

} // namespace caset
