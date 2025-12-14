//
// Created by andrew on 12/14/25.
//


#include <pybind11/pybind11.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "ForwardDeclarations.h"
#include "SimplexOrientation.h"
#include "Vertex.h"


namespace caset {
SimplexOrientation::SimplexOrientation(uint8_t ti_, uint8_t tf_) : ti(ti_), tf(tf_), fingerprint({ti_, tf_}) {
  k = ti_ + tf_ - 1;
  fingerprint = Fingerprint({ti_, tf_}); // TODO: Does this initialize twice?
}

[[nodiscard]] std::pair<uint8_t, uint8_t> SimplexOrientation::numeric() const {
  return {ti, tf};
}

[[nodiscard]] size_t SimplexOrientation::hash() const {
  return fingerprint.fingerprint();
}


[[nodiscard]] SimplexOrientationPtr SimplexOrientation::flip() const {
  return std::make_shared<SimplexOrientation>(tf, ti);
}

[[nodiscard]]
SimplexOrientationPtr SimplexOrientation::decTi() const {
  auto newTi = static_cast<uint8_t>(ti - 1);
  // constructor recomputes k automatically
  return std::make_shared<SimplexOrientation>(newTi, tf);
}

[[nodiscard]]
SimplexOrientationPtr SimplexOrientation::decTf() const {
  auto newTf = static_cast<uint8_t>(tf - 1);
  return std::make_shared<SimplexOrientation>(ti, newTf);
}

[[nodiscard]] std::string SimplexOrientation::toString() const noexcept {
  return "<SimplexOrientation: (" + std::to_string(ti) + ", " + std::to_string(tf) + ")>";
}

bool SimplexOrientation::operator==(const SimplexOrientation &other) const noexcept {
  return ti == other.ti && tf == other.tf;
}

[[nodiscard]] TimeOrientation SimplexOrientation::getOrientation() const {
  if (ti == tf) return TimeOrientation::UNKNOWN;
  if (ti > tf) return TimeOrientation::PRESENT;
  return TimeOrientation::FUTURE;
}

[[nodiscard]] std::vector<SimplexOrientationPtr> SimplexOrientation::getFacialOrientations() const {
  if (ti + tf == 0) return {};
  if (ti == 0) return {decTf()};
  if (tf == 0) return {decTi()};
  std::vector<SimplexOrientationPtr> orientations;
  orientations.reserve(2);
  orientations.push_back(decTi());
  orientations.push_back(decTf());
  return orientations;
}

/// A k-simplex has \f$ k+1 \f$ vertices.
[[nodiscard]] uint8_t SimplexOrientation::getK() const {
  return k;
}

SimplexOrientationPtr SimplexOrientation::orientationOf(const VertexPtrs &vertices) {
  uint8_t tiVertices = 0;
  uint8_t tfVertices = 0;
  double ti = std::numeric_limits<double>::max();
  double tf = -1;
  double initial = -1;
  int unassigned = 0;
  for (const auto &vertex : vertices) {
    double t = vertex->getTime();
    ti = std::min(ti, t);
    tf = std::max(tf, t);
    if (ti == tf) {
      initial = t;
      unassigned++;
    } else if (t == ti) {
      tiVertices++;
    } else {
      tfVertices++;
    }
  }
  if (initial == ti) {
    tiVertices += unassigned;
  } else {
    tfVertices += unassigned;
  }
  return std::make_shared<SimplexOrientation>(tiVertices, tfVertices);
}
}
