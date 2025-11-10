//
// Created by Andrew Kelleher on 11/9/25.
//

#ifndef CASET_CDT_H
#define CASET_CDT_H

#include "spacetimes/Spacetime.h"

namespace caset {
class CDT : public Spacetime {
  public:

    void tune() override {
      double lambda = 1; // Cosmological coupling
      double NTilde = 1; // Target Spacetime Volume

      double lambda0 = 1;  // Initial guess for lambda



    };
    void thermalize() override;
};
} // caset

#endif //CASET_CDT_H