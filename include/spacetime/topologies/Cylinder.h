//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_CYLINDER_H
#define CASET_CYLINDER_H

#include <vector>
#include <memory>

#include "Topology.h"
#include "constraints/Constraint.h"

namespace caset {
class Cylinder : public Topology {
  public:
    std::vector<std::shared_ptr<Constraint>> getConstraints() override {
      return std::vector<std::shared_ptr<Constraint>>();
    }
};
}

#endif //CASET_CYLINDER_H