//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef CASET_SPACETIMEVOLUME_H
#define CASET_SPACETIMEVOLUME_H

#include "Observable.h"
#include "../spacetime/Spacetime.h"

namespace caset {
class SpacetimeVolume : public Observable {
  public:
    double compute(std::shared_ptr<Spacetime> &spacetime) override;
    double update(std::shared_ptr<Spacetime> &spacetime) override;
};
}

#endif //CASET_SPACETIMEVOLUME_H