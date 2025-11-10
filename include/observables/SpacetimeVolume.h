//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_SPACETIMEVOLUME_H
#define CASET_SPACETIMEVOLUME_H

#include "Observable.h"
#include "../spacetimes/Spacetime.h"

namespace caset {
class SpacetimeVolume : public Observable {
  public:
    double update(Spacetime &spacetime) override {
      return 1.;
    }
    double compute(Spacetime &spacetime) override {
      return 1.;
    }
};
}

#endif //CASET_SPACETIMEVOLUME_H