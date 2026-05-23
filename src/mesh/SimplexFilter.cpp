// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// PositiveGramDeterminantFilter — the only filter with a non-trivial
// body. AllSimplexFilter is header-only.

#include "mesh/SimplexFilter.h"
#include "mesh/Simplex.h"

#include <cmath>

namespace tessera {

bool PositiveGramDeterminantFilter::accept(SimplexPtr const& simplex) const {
    if (simplex == nullptr) return false;
    const std::uint64_t k1 = simplex->size();  // k + 1 vertices for a k-simplex
    if (k1 < 2) return false;                  // 0-simplex: nothing to check

    // gramMatrix() returns a flat (d × d) row-major matrix with d = k = size − 1.
    // Vertex 0 is the origin; G is positive-definite for a non-degenerate
    // Euclidean k-simplex, so det(G) > 0 is the validity criterion. (The
    // simplex's k-volume is √(det(G) / k!²); det(G) > 0 ⇔ real positive
    // volume.) Any NaN / inf in edge lengths propagates through the Gram
    // build and we reject the simplex defensively.
    std::vector<double> gram = simplex->gramMatrix();
    const int d = static_cast<int>(k1) - 1;
    if (static_cast<int>(gram.size()) != d * d) return false;

    for (double const& g : gram) {
        if (!std::isfinite(g)) return false;
    }

    const double det = Simplex::determinant(gram, d);
    return std::isfinite(det) && det > 0.0;
}

} // namespace tessera
