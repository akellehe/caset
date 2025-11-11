//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_SPHERE_H
#define CASET_SPHERE_H

#include <vector>
#include <memory>

#include "Topology.h"
#include "constraints/Constraint.h"

namespace caset {
class Sphere : public Topology {
  public:
    // std::vector<std::shared_ptr<Constraint>> getConstraints() override {
      // return std::vector<std::shared_ptr<Constraint>>();
    // }
    void build() override; // Spacetime *spacetime) override;
};
}

#endif //CASET_SPHERE_H