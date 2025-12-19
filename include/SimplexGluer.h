//
// Created by andrew on 12/16/25.
//

#ifndef CASET_SIMPLEXGLUER_H
#define CASET_SIMPLEXGLUER_H

#include <mutex>

namespace caset {
class SimplexGluer {
  public:
    SimplexGluer(std::mutex &mutex);
    ~SimplexGluer();
    SimplexGluer(const SimplexGluer &) = delete;
    SimplexGluer &operator=(const SimplexGluer &) = delete;
  private:
    std::mutex &m_;

};
} // caset

#endif //CASET_SIMPLEXGLUER_H