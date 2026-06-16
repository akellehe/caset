// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// PositiveGramDeterminantFilter — the only filter with a non-trivial
// body. AllSimplexFilter is header-only.

#include "mesh/SimplexFilter.h"
#include "mesh/Simplex.h"

#include <cmath>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

bool PositiveGramDeterminantFilter::accept(SimplexPtr const& simplex) const {
    if (simplex == nullptr) return false;
    const std::uint64_t k1 = simplex->size();  // k + 1 vertices for a k-simplex
    if (k1 < 2) return false;                  // 0-simplex: nothing to check

    // gramMatrix() returns a flat (d × d) row-major matrix with d = k = size − 1.
    // Vertex 0 is the origin; G is positive-definite for a non-degenerate
    // Euclidean k-simplex, so det(G) > 0 is the validity criterion. (The
    // simplex's k-volume is √(det(G) / k!²); det(G) > 0 ⇔ real positive
    // volume.) This is a Euclidean validity test, so request the Wick-rotated
    // (|l^2|) Gram — a Lorentzian cell with timelike edges would otherwise have
    // a sign-indefinite det(G). Any NaN / inf in edge lengths propagates
    // through the Gram build and we reject the simplex defensively.
    std::vector<double> gram = simplex->gramMatrix(/*wickRotate=*/true);
    const int d = static_cast<int>(k1) - 1;
    if (static_cast<int>(gram.size()) != d * d) return false;

    for (double const& g : gram) {
        if (!std::isfinite(g)) return false;
    }

    const double det = Simplex::determinant(gram, d);
    return std::isfinite(det) && det > 0.0;
}

} // namespace tessera::mesh
