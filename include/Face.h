//
// Created by Andrew Kelleher on 11/13/25.
//

#ifndef CASET_FACE_H
#define CASET_FACE_H
#include <memory>
#include <vector>

#include "Simplex.h"
#include "Vertex.h"

namespace caset {

///
/// # Face
/// A Face, \f$ \sigma^{k-1} \memberof \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
/// the k-simplex.
///
/// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
/// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
/// by the orientation of those respective `Simplex`es.
/// attach it
class Face {
  public:
    Face(std::shared_ptr<Simplex> simplex_, std::vector<std::shared_ptr<Vertex>> vertices_) : simplex(simplex_), vertices(vertices_) {}

    std::shared_ptr<Simplex> getSimplex() const noexcept { return simplex; }
    std::vector<std::shared_ptr<Vertex>> getVertices() const noexcept {return vertices;};
  private:
    std::vector<std::shared_ptr<Vertex>> vertices;
    std::shared_ptr<Simplex> simplex;
};
} // caset

#endif //CASET_FACE_H