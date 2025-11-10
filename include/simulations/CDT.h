//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_CDT_H
#define CASET_CDT_H

#include "simulations/Simulation.h"

namespace caset {
class CDT : public Simulation {
  public:

    void add() {

    }

    void flip() {

    }

    void shift() {

    }

    void ishift() {

    }

    ///
    /// `tune` is the initial stage of building the spacetime. Some example are:
    ///
    /// The stage in CDT where we approach the desired cosmological constant by continuously adjusting it based on the
    /// desired spacetime volume.
    ///
    /// In Regge calculus we build an initial triangulation of the Spacetime.
    ///
    void tune() override {

    }

    ///
    /// `thermalize` implements some kind of adjustment to the initial lattice.
    ///
    /// In the case of CST (Causal Set Theory) this amounts to executing a poisson "Sprinkling" of Vertexes to preserve
    /// Lorentz invariance.
    ///
    /// For Regge calculus this can be a randomly applied variation in an initially fixed edge length triangulation.
    ///
    void thermalize() override {

    }
};
}

#endif //CASET_CDT_H