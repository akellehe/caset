//
// Created by Andrew Kelleher on 11/12/25.
//

#ifndef CASET_COFACE_H
#define CASET_COFACE_H

#include <memory>
#include "Simplex.h"

namespace caset {
///
/// # Coface
/// The co-face of a k-simplex \f$ \sigma_i^k \f$ is another k-simplex, \f$ \sigma_j^k \f$ that shares a k-1 simplex
/// \f$ \sigma^{k-1} \f$ with \f$ \sigma_i^k \f$.
///
/// We define a face as a set of shared vertices. The face of any given k-simplex \f$ \sigma^k \f$ is a k-1 simplex,
/// \f$ \sigma^{k-1} \f$ such that \f$ \f$ \sigma^{k-1} \subset \sigma^k \f$.
/// 
class Coface {

  public:

    Coface(
      const std::shared_ptr<Simplex> &first_,
      const std::shared_ptr<Simplex> &second_,
      const std::vector<std::shared_ptr<Vertex>> face_
      ) : first(first_), second(second_), face(face_) {};


    std::shared_ptr<Simplex> getFirst() const noexcept {
      return first;
    };

    std::shared_ptr<Simplex> getSecond() const noexcept {
      return second;
    };

    std::vector<std::shared_ptr<Vertex>> getFace() const noexcept {
      return ;
    }

  private:
    std::shared_ptr<Simplex> first;
    std::shared_ptr<Simplex> second;
    std::vector<std::shared_ptr<Vertex>> face;
};
} // caset

#endif //CASET_COFACE_H