//
// Created by andrew on 10/23/25.
//

#ifndef CASET_METRIC_H
#define CASET_METRIC_H

#include <memory>

#include "Signature.h"

namespace caset {
template<int N>
class Metric {
  public:
    using Sig = Signature<N>;
};
} // caset

#endif //CASET_METRIC_H