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

double
QuantumVertex::vanRaamsdonkSquaredLength(double I, double iMax,
                                         double epsilon) noexcept {
    const double cap = -std::log(epsilon);  // floor on d_VR ⇒ finite squared length
    const double x = (iMax > 0.0 && I > 0.0) ? (I / iMax) : 0.0;
    double dVR = (x > 0.0) ? -std::log(x)
                           : std::numeric_limits<double>::infinity();
    if (!std::isfinite(dVR) || dVR > cap) {
        dVR = cap;
    }
    return dVR * dVR;
}

double
QuantumVertex::vanRaamsdonkSquaredLengthTo(
    const ::tessera::mesh::Vertex* other,
    double                          iMax,
    double                          epsilon) const {
    if (other == nullptr) {
        throw std::invalid_argument(
            "QuantumVertex::vanRaamsdonkSquaredLengthTo: other is null");
    }
    // Forward-time worldline edge (different time slice) → null (squaredLength
    // 0). I = iMax there, so the spacelike length would be 0 anyway; we return
    // it directly and let the Lorentzian/rapidity structure (Regge action)
    // supply the timelike content.
    if (std::abs(getTime() - other->getTime()) > 1e-12) {
        return 0.0;
    }
    // Spacelike edge (same slice): +d_VR², floored to a finite value when
    // I < epsilon·iMax so a vanishing-correlation edge cannot blow the cell up.
    double dVR = vanRaamsdonkDistanceTo(other, iMax);
    const double cap = -std::log(epsilon);
    if (!std::isfinite(dVR) || dVR > cap) {
        dVR = cap;
    }
    return dVR * dVR;
}

} // namespace tessera::quantum
