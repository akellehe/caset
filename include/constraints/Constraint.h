//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_CONSTRAINT_H
#define CASET_CONSTRAINT_H

namespace caset {

enum class ConstraintType : uint8_t {
  PachnerMove = 0,
  All = 1
};

class Constraint {
  public:
    virtual ~Constraint() = default;


};
} // caset

#endif //CASET_CONSTRAINT_H