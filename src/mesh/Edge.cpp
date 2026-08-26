// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/Fingerprint.h"
#include "mesh/Edge.h"
#include "mesh/EdgeKey.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "utils.h"

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <numbers>


// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class Simplex;

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_,
      std::complex<double> squaredLength
    ) : source(source_), target(target_), squaredLength_(squaredLength),
        fingerprint({source_->getId(), target_->getId()}) {
    }

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_
    ) : source(source_), target(target_), fingerprint({source_->getId(), target_->getId()}) {
      // Fallback only.  Production complex-first paths provide z explicitly.
      squaredLength_ = {random_uniform(), 0.0};
    }

    [[nodiscard]] const VertexPtr &Edge::getSource() const noexcept {
      return source;
    }

    [[nodiscard]] const VertexPtr &Edge::getTarget() const noexcept {
      return target;
    }

    [[nodiscard]] std::complex<double> Edge::getPhase() const noexcept {
      // Legacy principal-log view: phase = -i Log(U_source,target).
      return std::complex<double>(0.0, -1.0) *
             std::log(link(source->getId(), target->getId()));
    }

    [[nodiscard]] std::complex<double> Edge::getLength() const noexcept {
      // Legacy principal-root view.  Direct geometry reads squaredLength().
      return std::sqrt(squaredLength_);
    }

    [[nodiscard]] std::complex<double> Edge::link(
        std::uint64_t from, std::uint64_t to) const {
      const auto sourceId = source->getId();
      const auto targetId = target->getId();
      if (from == to ||
          !((from == sourceId && to == targetId) ||
            (from == targetId && to == sourceId))) {
        throw std::invalid_argument(
            "Edge::link orientation does not name this edge's endpoints");
      }
      return from < to ? canonicalLink_ : 1.0 / canonicalLink_;
    }

    void Edge::setCanonicalLink(std::complex<double> nonzeroU) {
      if (nonzeroU == std::complex<double>{0.0, 0.0} ||
          !std::isfinite(nonzeroU.real()) ||
          !std::isfinite(nonzeroU.imag()))
        throw std::invalid_argument("Edge link must be non-zero (C*)");
      canonicalLink_ = nonzeroU;
      ++linkRevision_;
    }

    void Edge::setLink(std::uint64_t from, std::uint64_t to,
                       std::complex<double> nonzeroU) {
      const auto sourceId = source->getId();
      const auto targetId = target->getId();
      if (from == to ||
          !((from == sourceId && to == targetId) ||
            (from == targetId && to == sourceId))) {
        throw std::invalid_argument(
            "Edge::setLink orientation does not name this edge's endpoints");
      }
      if (nonzeroU == std::complex<double>{0.0, 0.0})
        throw std::invalid_argument("Edge link must be non-zero (C*)");
      setCanonicalLink(from < to ? nonzeroU : 1.0 / nonzeroU);
    }

    void Edge::setPhase(std::complex<double> p) {
      // Legacy conversion is named and isolated here.  New paths store U.
      setLink(source->getId(), target->getId(),
              std::exp(std::complex<double>(0.0, 1.0) * p));
    }

    void Edge::recanonicalizeLink(std::uint64_t oldSourceId,
                                  std::uint64_t oldTargetId) noexcept {
      const bool oldCanonicalForward = oldSourceId < oldTargetId;
      const bool newCanonicalForward = source->getId() < target->getId();
      if (oldCanonicalForward != newCanonicalForward) {
        canonicalLink_ = 1.0 / canonicalLink_;
        ++linkRevision_;
      }
    }

    [[nodiscard]] double Edge::squaredArgument() const noexcept {
      return std::arg(squaredLength_);  // legacy presentation in (-pi, pi]
    }

    [[nodiscard]] double Edge::lorentzianMagnitude() const noexcept {
      return squaredLength_.real();
    }

    [[nodiscard]] bool Edge::isDegenerate() const noexcept {
      // The EUCLIDEAN modulus, and the one place it is the right norm: an edge with
      // no extent at all is ABSENT, not lightlike.
      return std::abs(squaredLength_) <=
             kDegenerateEpsilon * kDegenerateEpsilon;
    }

    [[nodiscard]] bool Edge::isSpacelike() const noexcept {
      // arg(l^2) ~ 0: l^2 real positive. |arg| folds the (-pi, pi] range so each
      // test below is one comparison against a single definite argument.
      if (isDegenerate()) return false;
      return std::abs(squaredArgument()) <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isTimelike() const noexcept {
      // arg(l^2) ~ +/- pi: l^2 real negative.
      if (isDegenerate()) return false;
      return std::abs(std::abs(squaredArgument()) - std::numbers::pi)
             <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isNull() const noexcept {
      // arg(l^2) ~ +/- pi/2: l^2 purely imaginary and NONZERO -- the light cone,
      // reached non-trivially at Re(l) == Im(l) != 0. Not the same as degenerate.
      if (isDegenerate()) return false;
      return std::abs(std::abs(squaredArgument()) - 0.5 * std::numbers::pi)
             <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isMixed() const noexcept {
      // No definite causal character. Deliberately NOT snapped to the nearest of the
      // three: a generic argument is genuinely mixed, and reporting it as definite
      // would invent structure the geometry does not have.
      return !isDegenerate() && !isSpacelike() && !isTimelike() && !isNull();
    }

    [[nodiscard]] EdgeDisposition Edge::disposition() const noexcept {
      if (isDegenerate()) return EdgeDisposition::Degenerate;
      if (isSpacelike()) return EdgeDisposition::Spacelike;
      if (isTimelike()) return EdgeDisposition::Timelike;
      if (isNull()) return EdgeDisposition::Lightlike;
      return EdgeDisposition::Mixed;
    }

#ifdef TESSERA_VERBOSE
    [[nodiscard]] std::string Edge::toString() const noexcept {
      return source->toString() + "->" + target->toString();
    }
#endif

    void Edge::replaceSourceVertex(const VertexPtr &newSource) {
      const auto oldSourceId = source->getId();
      const auto oldTargetId = target->getId();
      fingerprint.removeId(source->getId());
      source = newSource;
      fingerprint.addId(newSource->getId());
      fingerprint.refresh();
      recanonicalizeLink(oldSourceId, oldTargetId);
    }

    void Edge::replaceTargetVertex(const VertexPtr &newTarget) {
      const auto oldSourceId = source->getId();
      const auto oldTargetId = target->getId();
      fingerprint.removeId(target->getId());
      target = newTarget;
      fingerprint.addId(newTarget->getId());
      fingerprint.refresh();
      recanonicalizeLink(oldSourceId, oldTargetId);
    }

    bool Edge::hasVertex(std::uint64_t vertexId) const {
      if (getSource()->getId() == vertexId || getTarget()->getId() == vertexId) return true;
      return false;
    }

    bool Edge::operator==(const Edge &other) const {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    [[nodiscard]] std::uint64_t Edge::toHash() const {
      return fingerprint.fingerprint();
    }

    EdgeKey Edge::getKey() const noexcept {
      return {source->getId(), target->getId()};
    }

    void Edge::registerSimplex(SimplexPtr s) {
      // Append; the registry is duplicate-free by construction because
      // Simplex::addEdge / Spacetime::registerSimplex deduplicate before
      // calling. Idempotency under multiple registers would require a
      // scan we don't want in the hot path.
      simplices_.push_back(s);
    }

    void Edge::unregisterSimplex(SimplexPtr s) noexcept {
      // Swap-pop by fingerprint to keep the storage tight. The list is
      // small (≤ few simplices per edge in typical 4D builds) so the
      // linear scan is cheap; the canonical Simplex pointer is unique
      // per fingerprint so the first match suffices.
      const auto fp = s->fingerprint.fingerprint();
      for (std::size_t i = 0; i < simplices_.size(); ++i) {
        if (simplices_[i]->fingerprint.fingerprint() == fp) {
          simplices_[i] = simplices_.back();
          simplices_.pop_back();
          return;
        }
      }
    }

double Edge::vanRaamsdonkLength(double I, double iMax,
                                double epsilon) noexcept {
  const double cap = -std::log(epsilon);  // floor on d_VR => finite length
  const double x = (iMax > 0.0 && I > 0.0) ? (I / iMax) : 0.0;
  double dVR = (x > 0.0) ? -std::log(x)
                         : std::numeric_limits<double>::infinity();
  if (!std::isfinite(dVR) || dVR > cap) {
    dVR = cap;
  }
  return dVR;
}

double Edge::vanRaamsdonkLengthFor(double I, double iMax,
                                   double epsilon) const {
  const VertexPtr s = getSource();
  const VertexPtr t = getTarget();
  // Forward-time worldline edge (endpoints on different time slices) → null.
  if (s != nullptr && t != nullptr &&
      std::abs(s->getTime() - t->getTime()) > 1e-12) {
    return 0.0;
  }
  return vanRaamsdonkLength(I, iMax, epsilon);  // spacelike
}

}
