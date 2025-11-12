//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_TOPOLOGY_H
#define CASET_TOPOLOGY_H

// #include "constraints/Constraint.h"

namespace caset {

class Spacetime;

class Topology {
  public:
    virtual ~Topology();
    // virtual std::vector<std::shared_ptr<Constraint>> getConstraints() = 0;

    ///
    /// Builds an initial triangulation matching this topology for t=0 on a given spacetime based on the parameters of
    /// the spacetime.
    virtual void build(Spacetime *spacetime) = 0;
};
}

#endif //CASET_TOPOLOGY_H