//
// Created by andrew on 10/21/25.
//

#include "Simplex.h"

namespace caset {

const double Simplex::getDeficitAngle() const {
  return 0;
}

const std::vector<std::shared_ptr<Simplex>> Simplex::getHinges() const {
  return {};
}

}