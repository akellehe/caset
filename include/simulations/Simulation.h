//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_SIMULATION_H
#define CASET_SIMULATION_H

namespace caset {
class Simulation {
  public:

    virtual ~Simulation() = default;

    ///
    /// `tune` is the initial stage of building the spacetime. Some example are:
    ///
    /// The stage in CDT where we approach the desired cosmological constant by continuously adjusting it based on the
    /// desired spacetime volume.
    ///
    /// In Regge calculus we build an initial triangulation of the Spacetime.
    ///
    virtual void tune();

    ///
    /// `thermalize` implements some kind of adjustment to the initial lattice.
    ///
    /// In the case of CST (Causal Set Theory) this amounts to executing a poisson "Sprinkling" of Vertices to preserve
    /// Lorentz invariance.
    ///
    /// For Regge calculus this can be a randomly applied variation in an initially fixed edge length triangulation.
    ///
    virtual void thermalize();
};
}

#endif //CASET_SIMULATION_H
