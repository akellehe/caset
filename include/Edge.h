#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Fingerprint.h"

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <random>
#include <memory>

inline double random_uniform(double min = -1.0, double max = 1.0) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

namespace caset {

/// # Edge Disposition
///
/// There are two things that determine the disposition (spacelike, timelike, light/null-like). The first is the squared
/// edge length. If the squared length is negative in a (-, +, +, +) signature it's timelike. A negative edge length in
/// a (+, -, -, -) signature is spacelike. A 0-length in either is lightlike/null.
///
/// The second thing that determines the edge disposition is whether the vertices exist both in space (lightlike), both
/// at the same time (timelike), or one in space and one in time (spacelike). See "Quantum Gravity from Causal Dynamical
/// Triangulations: A Review" by R. Loll, 2019. Figure 1. There's no discussion of lightlike edges since CDT does not
/// treat that case. I'm making that up to fill in the gaps. If there's some existing discussion around this in the
/// literature I'm not aware at the time of this writing.
enum class EdgeDisposition : uint8_t {
  Spacelike = 0,
  Timelike = 1,
  Lightlike = 2
};

struct EdgeKeyHash {
  std::size_t operator()(const std::pair<std::uint64_t, std::uint64_t>& p) const noexcept {
    // Standard-ish hash combine
    std::size_t h1 = std::hash<std::uint64_t>{}(p.first);
    std::size_t h2 = std::hash<std::uint64_t>{}(p.second);
    // from boost::hash_combine
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

// Optional: custom equality if you ever need something non-trivial
struct EdgeKeyEqual {
  bool operator()(const std::pair<std::uint64_t, std::uint64_t>& a,
                  const std::pair<std::uint64_t, std::uint64_t>& b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};


/// # Edge Class
///
/// An edge that links two points (vertices) in spacetime.
///
/// @param source_
/// @param target_
/// @param squaredLength_ The squared length of the edge according to whatever spacetime metric is being used.
///
class Edge : public std::enable_shared_from_this<Edge> {
  public:
    Edge(
      std::uint64_t sourceId_,
      std::uint64_t targetId_,
      double squaredLength_
    ) : sourceId(sourceId_), targetId(targetId_), squaredLength(squaredLength_), fingerprint({sourceId_, targetId_}) {
    }

    Edge(
      std::uint64_t sourceId_,
      std::uint64_t targetId_
    ) : sourceId(sourceId_), targetId(targetId_), fingerprint({sourceId_, targetId_}) {
      // Set squaredLength to a random value between -1 and 1
      squaredLength = random_uniform(); // TODO: Should we use a poisson dist here for coset theory?
    }

    [[nodiscard]] std::uint64_t getSourceId() const noexcept {
      return sourceId;
    }

    [[nodiscard]] std::uint64_t getTargetId() const noexcept {
      return targetId;
    }

    [[nodiscard]] double getSquaredLength() const noexcept {
      return squaredLength;
    }

    [[nodiscard]] std::string toString() const noexcept {
      return std::to_string(sourceId) + "->" + std::to_string(targetId);
    }

    void refreshFingerprint() noexcept {
      fingerprint = Fingerprint({sourceId, targetId});
    }

    void replaceSourceVertex(std::uint64_t sourceId_) {
      sourceId = sourceId_;
      refreshFingerprint();
    }

    void replaceTargetVertex(std::uint64_t targetId_) {
      targetId = targetId_;
      refreshFingerprint();
    }

    void redirect(std::uint64_t from, std::uint64_t to) {
      if (getSourceId() == from) {
        replaceSourceVertex(to);
      }
      if (getTargetId() == from) {
        replaceTargetVertex(to);
      }
    }

    bool operator==(const Edge &other) const {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    [[nodiscard]] std::uint64_t toHash() const {
      return fingerprint.fingerprint();
    }

    Fingerprint fingerprint;

    std::pair<IdType, IdType> getKey() const noexcept{
      return {sourceId, targetId};
    }

  private:
    std::uint64_t sourceId;
    std::uint64_t targetId;
    double squaredLength;
};

using EdgeHash = FingerprintHash<Edge>;
using EdgeEq = FingerprintEq<Edge>;
using EdgePtr = std::shared_ptr<Edge>;
using Edges = std::vector<EdgePtr>;
using EdgeKey = std::pair<IdType, IdType>;
using EdgeMap = std::unordered_map<EdgeKey, EdgePtr, EdgeKeyHash, EdgeKeyEqual>;
using EdgeIdSet = std::unordered_set<EdgeKey, EdgeKeyHash, EdgeKeyEqual>;
using EdgeIndexMap = std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash, EdgeKeyEqual>;

}

#endif //CASET_CASET_SRC_EDGE_H_
