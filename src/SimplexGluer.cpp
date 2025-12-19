//
// Created by andrew on 12/16/25.
//

#include "../include/SimplexGluer.h"

#include <mutex>

namespace caset {
SimplexGluer::SimplexGluer(std::mutex &mutex) : m_(mutex) {
  m_.lock();
}

SimplexGluer::~SimplexGluer() {
  m_.unlock();
}

} // caset