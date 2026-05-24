// Tests for the quantum-state surface on tessera::mesh::Vertex.
//
// Pins the per-vertex QuantumState member, the const/mutable accessors,
// and the isLocallyPure delegate added in issue #56.

#include "mesh/Vertex.h"
#include "quantum/QuantumState.hpp"
#include "test_helpers.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

using namespace tessera::quantum;
using namespace tessera::test_helpers_core;

namespace {

constexpr double TOL = 1e-10;

bool t_default_vertex_has_trivial_state() {
    // A default-constructed Vertex (or one constructed with just an id)
    // has the 1-dim trivial state ρ = [[1]] — purity 1, entropy 0,
    // trivially "locally pure".
    tessera::mesh::Vertex v(7);
    return expect_true(v.quantumState().dim() == 1,
            "Vertex(id).quantumState().dim() == 1")
        && expect_near(v.quantumState().purity(), 1.0, TOL,
            "Vertex(id) purity == 1 (trivial 1-dim state)")
        && expect_near(v.quantumState().entropy(), 0.0, TOL,
            "Vertex(id) entropy == 0")
        && expect_true(v.isLocallyPure(TOL),
            "Vertex(id).isLocallyPure(TOL) == true");
}

bool t_assign_state_via_mutable_accessor() {
    // The mutable quantumState() accessor returns a reference; assigning
    // through it should persist on the Vertex.
    tessera::mesh::Vertex v(11);
    auto target = QuantumState::randomMixed(4, /*S=*/1.0, /*seed=*/0xDEADBEEF);
    v.quantumState() = target;
    bool ok = expect_true(v.quantumState().dim() == 4,
        "after assignment, dim == 4");
    ok &= expect_matrix_near(v.quantumState().matrix(), target.matrix(), TOL,
        "after assignment, matrix matches target");
    ok &= expect_near(v.quantumState().entropy(), 1.0, 1e-5,
        "after assignment, entropy ≈ targetEntropy");
    return ok;
}

bool t_const_accessor_returns_same_matrix() {
    // const quantumState() should expose the same matrix as the mutable
    // one; the only difference is mutability.
    tessera::mesh::Vertex v(13);
    v.quantumState() = QuantumState::maximallyMixed(3);
    const auto& cs = static_cast<const tessera::mesh::Vertex&>(v).quantumState();
    return expect_matrix_near(cs.matrix(), v.quantumState().matrix(), TOL,
        "const and mutable accessors return the same matrix");
}

bool t_is_locally_pure_delegates_to_quantum_state() {
    // Vertex::isLocallyPure(eps) should match QuantumState::isLocallyPure(eps).
    tessera::mesh::Vertex pureV(21);
    pureV.quantumState() = QuantumState::computationalBasis(4, 1);
    bool ok = expect_true(pureV.isLocallyPure(TOL),
        "pure-state vertex is locally pure under tight eps");

    tessera::mesh::Vertex mixedV(22);
    mixedV.quantumState() = QuantumState::maximallyMixed(4);
    ok &= expect_false(mixedV.isLocallyPure(TOL),
        "I/4 vertex is not locally pure under tight eps");
    // Loose eps swallows the purity gap.
    ok &= expect_true(mixedV.isLocallyPure(0.8),
        "I/4 vertex is locally pure under loose eps (sentinel)");
    return ok;
}

bool t_per_vertex_states_are_independent() {
    // Two vertices' states should be independent — modifying one does
    // not modify the other.
    tessera::mesh::Vertex a(31);
    tessera::mesh::Vertex b(32);
    a.quantumState() = QuantumState::maximallyMixed(2);
    b.quantumState() = QuantumState::computationalBasis(2, 0);
    return expect_near(a.quantumState().entropy(), std::log(2.0), TOL,
            "vertex A has I/2 entropy")
        && expect_near(b.quantumState().entropy(), 0.0, TOL,
            "vertex B has |0><0| entropy = 0")
        && expect_true(a.quantumState().dim() == 2 && b.quantumState().dim() == 2,
            "both vertices have dim 2 but distinct states");
}

bool t_vertex_state_dim_can_grow() {
    // Vertices in the post-#56 design can carry varying dimensions
    // (Σ vertices are typically larger). A vertex's state can be
    // re-assigned to a different dimension.
    tessera::mesh::Vertex v(41);
    v.quantumState() = QuantumState::maximallyMixed(2);
    bool ok = expect_true(v.quantumState().dim() == 2,
        "initially dim 2");
    v.quantumState() = QuantumState::maximallyMixed(16);
    ok &= expect_true(v.quantumState().dim() == 16,
        "after re-assignment, dim 16 (Σ-like growth)");
    ok &= expect_near(v.quantumState().entropy(), std::log(16.0), TOL,
        "entropy matches the new dimension's max");
    return ok;
}

} // namespace

int main() {
    std::cout << "== test_vertex_quantum_state ==\n";
    bool ok = true;
    ok &= t_default_vertex_has_trivial_state();
    ok &= t_assign_state_via_mutable_accessor();
    ok &= t_const_accessor_returns_same_matrix();
    ok &= t_is_locally_pure_delegates_to_quantum_state();
    ok &= t_per_vertex_states_are_independent();
    ok &= t_vertex_state_dim_can_grow();
    std::cout << (ok ? "ALL PASSED\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
