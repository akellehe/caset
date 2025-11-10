//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_TOROID_H
#define CASET_TOROID_H

#include <vector>
#include <memory>

#include "Topology.h"
#include "constraints/Constraint.h"

namespace caset {
class Toroid : public Topology {
  public:
    std::vector<std::shared_ptr<Constraint>> getConstraints() override {
      return std::vector<std::shared_ptr<Constraint>>();
    }

    void build(std::shared_ptr<Spacetime> spacetime) override {
      return;
    }
};
}

#endif //CASET_TOROID_H