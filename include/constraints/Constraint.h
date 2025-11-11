//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_CONSTRAINT_H
#define CASET_CONSTRAINT_H

#include <memory>

namespace caset {

class Spacetime;

enum class ConstraintType : uint8_t {
  PachnerMove = 0,
  All = 1
};

class Constraint {
  public:
    virtual ~Constraint() = default;

    virtual bool applies(std::shared_ptr<Spacetime> &spacetime, ConstraintType &type_);

};
} // caset

#endif //CASET_CONSTRAINT_H