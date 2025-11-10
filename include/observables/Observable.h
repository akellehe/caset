//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_OBSERVABLE_H
#define CASET_OBSERVABLE_H

#include <memory>

namespace caset {

class Spacetime;

class Observable {
  public:
    virtual double compute(std::shared_ptr<Spacetime> &spacetime);
    virtual double update(std::shared_ptr<Spacetime> &spacetime);
    virtual ~Observable() = default;
};
}

#endif //CASET_OBSERVABLE_H