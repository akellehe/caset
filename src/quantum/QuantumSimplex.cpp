#include "quantum/QuantumSimplex.hpp"

#include "mesh/Edge.h"
#include "mesh/Vertex.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tessera::quantum {

namespace {

// Process-static stable-address storage for QuantumSimplex objects.
// std::deque guarantees that previously-pushed elements keep their
// addresses on push_back, and we never erase, so any raw pointer
// obtained from this deque is valid for the life of the process.
// Spacetime tracks QuantumSimplex objects only as Simplex* pointers
// into this storage; this lets the Spacetime stay in tessera_core
// while QuantumSimplex (with its Eigen dependency) lives in
// tessera_quantum.
std::deque<QuantumSimplex>& quantumSimplexStorage() {
    static std::deque<QuantumSimplex> storage;
    return storage;
}

// All ten pairs (i, j) with i < j over the five vertex positions.
struct PositionPair { int u, v; };
constexpr PositionPair kAllEdges[10] = {
    {0, 1}, {0, 2}, {0, 3}, {0, 4},
    {1, 2}, {1, 3}, {1, 4},
    {2, 3}, {2, 4},
    {3, 4},
};

inline std::pair<int, int> orderedPair(int a, int b) {
    return (a < b) ? std::pair{a, b} : std::pair{b, a};
}

// Construct ρ_u ⊗ ρ_v in (u ⊗ v) ordering. Used to compute pairwise MI
// between two vertex states that have no inherited correlation
// (everything except the (A, B) pair, which carries the input joint).
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    const int dA = static_cast<int>(a.rows());
    const int dB = static_cast<int>(b.rows());
    Eigen::MatrixXcd out(dA * dB, dA * dB);
    for (int i = 0; i < dA; ++i)
        for (int j = 0; j < dA; ++j)
            for (int k = 0; k < dB; ++k)
                for (int l = 0; l < dB; ++l)
                    out(i * dB + k, j * dB + l) = a(i, j) * b(k, l);
    return out;
}

// d_VR(I) = -log(I / iMax). Returns +∞ when I == 0 (or iMax == 0).
double vanRaamsdonk(double I, double iMax) {
    if (!(I > 0.0) || !(iMax > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::log(I / iMax);
}

} // namespace

QuantumSimplex::QuantumSimplex(::tessera::spacetime::Spacetime*       spacetime,
                               const ::tessera::mesh::VertexPtrs&     verts,
                               ::tessera::mesh::Edges                 edges,
                               std::array<::tessera::mesh::VertexPtr, 5> positions,
                               std::array<Eigen::MatrixXcd, 5>           states,
                               std::map<std::pair<int, int>, double>     mi,
                               KoashiImotoResult                         ki,
                               double                                    iMaxVal)
    : ::tessera::mesh::Simplex(spacetime, verts, std::move(edges)),
      positions_(std::move(positions)),
      states_(std::move(states)),
      mi_(std::move(mi)),
      kiResult_(std::move(ki)),
      iMax_(iMaxVal) {}

double QuantumSimplex::mutualInfoFor(Position p, Position q) const {
    auto it = mi_.find(orderedPair(p, q));
    if (it == mi_.end()) return 0.0;
    return it->second;
}

double QuantumSimplex::vanRaamsdonkDistanceFor(Position p, Position q) const {
    return vanRaamsdonk(mutualInfoFor(p, q), iMax_);
}

QuantumSimplex*
QuantumSimplex::fromKIInteraction(::tessera::spacetime::Spacetime& spacetime,
                                  const Eigen::MatrixXcd&          rhoAB,
                                  int                              dimA,
                                  int                              dimB,
                                  double                           iMaxVal,
                                  const KoashiImotoTolerances&     tol) {
    if (dimA <= 0 || dimB <= 0) {
        throw std::invalid_argument("QuantumSimplex::fromKIInteraction: "
                                    "dimA, dimB must be positive");
    }
    if (rhoAB.rows() != rhoAB.cols()
        || rhoAB.rows() != static_cast<Eigen::Index>(dimA) * dimB) {
        throw std::invalid_argument("QuantumSimplex::fromKIInteraction: "
                                    "rhoAB dimension must equal dimA · dimB");
    }
    if (!(iMaxVal > 0.0)) {
        throw std::invalid_argument("QuantumSimplex::fromKIInteraction: "
                                    "iMax must be positive");
    }

    // Marginals + KI decomposition.
    const Eigen::MatrixXcd rhoA = partialTraceB(rhoAB, dimA, dimB);
    const Eigen::MatrixXcd rhoB = partialTraceA(rhoAB, dimA, dimB);
    KoashiImotoResult ki = koashiImotoDecompose(rhoAB, dimA, dimB, tol);

    // Per-position states. Position layout: A, B, Σ, A', B'.
    std::array<Eigen::MatrixXcd, 5> states = {
        rhoA,
        rhoB,
        ki.sigma,
        ki.aPrime,
        ki.bPrime,
    };
    std::array<int, 5> dims = {
        dimA,
        dimB,
        static_cast<int>(ki.sigma.rows()),
        static_cast<int>(ki.aPrime.rows()),
        static_cast<int>(ki.bPrime.rows()),
    };

    // Allocate five vertices via the Spacetime. Vertex IDs are
    // assigned automatically by createVertex() (no-arg overload).
    std::array<::tessera::mesh::VertexPtr, 5> positions;
    for (int p = 0; p < 5; ++p) {
        positions[p] = spacetime.createVertex();
    }
    ::tessera::mesh::VertexPtrs verts(positions.begin(), positions.end());

    // Compute the 10 pairwise MIs and edge lengths.
    //
    // The (A, B) edge uses the INPUT joint ρ_AB directly — that's the
    // pair carrying inherited correlation. All other edges use the
    // product joint of the two endpoint states (no inherited
    // correlation between, say, A and Σ — the relationship is encoded
    // in the structure of ρ_AB and falls out through KI, but the
    // pairwise joint between distinct vertices in the new cell is
    // taken as product).
    std::map<std::pair<int, int>, double> mi;
    std::map<std::pair<int, int>, double> dvr;
    for (const auto& [u, v] : kAllEdges) {
        double I;
        if ((u == A && v == B) || (u == B && v == A)) {
            I = mutualInformation(rhoAB, dimA, dimB);
        } else {
            const Eigen::MatrixXcd joint = kron(states[u], states[v]);
            I = mutualInformation(joint, dims[u], dims[v]);
        }
        const std::pair<int, int> key = orderedPair(u, v);
        mi[key]  = I;
        dvr[key] = vanRaamsdonk(I, iMaxVal);
    }

    // Create the 10 edges in the Spacetime. Edge stores squaredLength;
    // here we square the d_VR (Euclidean spacelike convention — every
    // edge contributes +ℓ²). +∞ length squares to +∞.
    ::tessera::mesh::Edges edges;
    edges.reserve(10);
    for (const auto& [u, v] : kAllEdges) {
        const double len   = dvr.at(orderedPair(u, v));
        const double lenSq = len * len;
        ::tessera::mesh::EdgePtr edge =
            spacetime.createEdge(positions[u], positions[v], lenSq);
        if (edge != nullptr) edges.push_back(edge);
    }

    // Allocate the QuantumSimplex in our stable-storage deque, then
    // register it with the Spacetime as an external simplex. The
    // initialize() hook registers the simplex back on its vertices.
    auto& storage = quantumSimplexStorage();
    storage.emplace_back(&spacetime,
                         verts,
                         std::move(edges),
                         positions,
                         std::move(states),
                         std::move(mi),
                         std::move(ki),
                         iMaxVal);
    QuantumSimplex* qs = &storage.back();
    qs->initialize(qs);
    spacetime.registerSimplex(qs, /*internal=*/true);
    return qs;
}

} // namespace tessera::quantum
