#include "quantum/QuantumVertex.hpp"

#include "quantum/KoashiImoto.hpp"  // mutualInformation overload (ρ_AB, ρ_A, ρ_B)

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tessera::quantum {

namespace {

// ρ_A ⊗ ρ_B in (A ⊗ B) ordering.
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

} // namespace

double
QuantumVertex::vanRaamsdonkDistanceTo(
    const ::tessera::mesh::Vertex* other,
    double                          iMax) const {
    auto* qo = dynamic_cast<const QuantumVertex*>(other);
    if (qo == nullptr) {
        throw std::invalid_argument(
            "QuantumVertex::vanRaamsdonkDistanceTo: "
            "the other vertex must also be a QuantumVertex");
    }
    if (!(iMax > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::MatrixXcd joint = kron(state_, qo->state_);
    const double I = mutualInformation(joint, state_, qo->state_);
    if (!(I > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::log(I / iMax);
}

} // namespace tessera::quantum
