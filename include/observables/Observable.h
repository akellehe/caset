//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_OBSERVABLE_H
#define CASET_OBSERVABLE_H

#include "../spacetimes/Spacetime.h"

namespace caset {
class Observable {
  public:
    virtual double compute(Spacetime &spacetime);
    virtual double update(Spacetime &spacetime);
    virtual ~Observable() = default;
};
}

#endif //CASET_OBSERVABLE_H