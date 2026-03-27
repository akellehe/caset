// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include "mesh/ForwardDeclarations.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace caset {

class Spacetime;

/// Intrinsic (coordinate-free) specification of stress-energy on a triangulation.
///
/// Matter is defined relationally — by assigning energy densities to vertices,
/// simplices, or as a function of geodesic distance from a reference vertex.
/// The solver converts these to target deficit angles at each hinge.
class MatterConfiguration {
  public:
    /// Assign a point mass to a vertex.  The total deficit angle around
    /// hinges incident to this vertex should equal \f$8\pi M\f$ (in
    /// geometrized units where \f$G = c = 1\f$).
    void setPointMass(VertexPtr vertex, double mass);

    /// Assign energy density \f$\rho\f$ uniformly to a top-simplex.
    void setEnergyDensity(SimplexPtr simplex, double rho);

    /// Assign energy density as a function of geodesic distance from a
    /// center vertex.  The profile is sampled at each hinge during
    /// \c computeTargetDeficits().
    void setRadialProfile(VertexPtr center,
                          std::function<double(double)> rho_of_r);

    /// Compute target deficit angles at every hinge in the spacetime,
    /// based on the configured matter content.
    ///
    /// @return map from hinge fingerprint → target deficit angle
    std::unordered_map<std::uint64_t, double>
    computeTargetDeficits(const Spacetime &st) const;

  private:
    std::unordered_map<std::uint64_t, double> pointMasses_;   // vertex ID → mass
    std::unordered_map<std::uint64_t, double> simplexRho_;    // simplex fp → ρ

    struct RadialProfile {
        VertexPtr center;
        std::function<double(double)> rho;
    };
    std::vector<RadialProfile> radialProfiles_;
};

} // namespace caset
