// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SimplicialQubit.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

namespace {

using Complex = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = std::numeric_limits<double>::epsilon();

/// rot90(x, y) = (-y, x): the +90 degree rotation of spec §7 / §8.
Eigen::Vector2d rot90(const Eigen::Vector2d &v) { return Eigen::Vector2d(-v(1), v(0)); }

std::string edgeText(std::uint64_t u, std::uint64_t v) {
  return "(" + std::to_string(u) + "," + std::to_string(v) + ")";
}

std::string faceText(const SimplicialQubit::Face &f) {
  return "(" + std::to_string(f[0]) + "," + std::to_string(f[1]) + "," + std::to_string(f[2]) + ")";
}

/// The face's boundary traversal (spec §3): local edge slots 0, 1, 2 are
/// (i, j), (j, k), (k, i); slot s is opposite the local vertex (s + 2) mod 3.
std::pair<std::uint64_t, std::uint64_t> traversal(const SimplicialQubit::Face &f, int slot) {
  switch (slot) {
    case 0: return {f[0], f[1]};
    case 1: return {f[1], f[2]};
    default: return {f[2], f[0]};
  }
}

int oppositeVertexSlot(int edgeSlot) { return (edgeSlot + 2) % 3; }

/// Spec §4: the three angles of a triangle with a = l(jk), b = l(ki), c = l(ij)
/// by the law of cosines (the cosine is clamped to [-1, 1]; with the strict
/// triangle inequality it lies strictly inside, so the clamp only absorbs
/// rounding).
std::array<double, 3> anglesOf(double a, double b, double c) {
  auto angle = [](double cosine) { return std::acos(std::clamp(cosine, -1.0, 1.0)); };
  return {angle((b * b + c * c - a * a) / (2.0 * b * c)),
          angle((c * c + a * a - b * b) / (2.0 * c * a)),
          angle((a * a + b * b - c * c) / (2.0 * a * b))};
}

int numericalRank(const Eigen::MatrixXd &m) {
  if (m.cols() == 0 || m.rows() == 0) return 0;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
  lu.setThreshold(1e-9);
  return static_cast<int>(lu.rank());
}

}  // namespace

// ============================================================================
// Constructors
// ============================================================================

SimplicialQubit::SimplicialQubit(std::vector<std::uint64_t> vertices, std::vector<EdgePair> edges,
                                 std::vector<Face> faces, std::vector<double> lengths, Cycle cycleA,
                                 Cycle cycleB, double degeneracyThreshold)
    : vertices_(std::move(vertices)),
      edges_(std::move(edges)),
      faces_(std::move(faces)),
      lengths_(std::move(lengths)),
      cycleA_(std::move(cycleA)),
      cycleB_(std::move(cycleB)),
      degeneracyThreshold_(degeneracyThreshold) {
  indexEdges();
  validateCombinatorics();
  // The container: the faces as cells, then the lengths onto its edges. The
  // container keeps its own (sorted) vertex order per face; the oriented faces
  // stay in faces_.
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(faces_.size());
  for (const Face &f : faces_) cells.push_back({f[0], f[1], f[2]});
  spacetime_ = Spacetime::fromCells(2, cells, 1.0, Complex(0.0, 0.0));
  for (mesh::Edge *edge : spacetime_->getEdgeList()->toVector()) {
    const std::uint64_t u = edge->getSource()->getId();
    const std::uint64_t v = edge->getTarget()->getId();
    edge->setLength(Complex(lengths_[edgeIndexOf(std::min(u, v), std::max(u, v))], 0.0));
    edge->setPhase(Complex(0.0, 0.0));
  }
  initialize();
}

SimplicialQubit::SimplicialQubit(const std::shared_ptr<Spacetime> &spacetime, Cycle cycleA,
                                 Cycle cycleB, bool reversed, double degeneracyThreshold)
    : cycleA_(std::move(cycleA)),
      cycleB_(std::move(cycleB)),
      degeneracyThreshold_(degeneracyThreshold),
      spacetime_(spacetime) {
  if (!spacetime_) throw std::invalid_argument("SimplicialQubit: null spacetime");

  // Vertices: V = [0 .. nV-1] by ascending id.
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : spacetime_->getVertexList()->liveVector()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::map<std::uint64_t, std::uint64_t> indexOfId;
  for (std::size_t n = 0; n < ids.size(); ++n) indexOfId[ids[n]] = n;
  vertices_.resize(ids.size());
  for (std::size_t n = 0; n < ids.size(); ++n) vertices_[n] = n;

  // Edges (i < j, ascending) with their real positive lengths.
  std::vector<std::pair<EdgePair, double>> edgeLengths;
  for (const mesh::Edge *edge : spacetime_->getEdgeList()->toVector()) {
    const std::uint64_t a = indexOfId.at(edge->getSource()->getId());
    const std::uint64_t b = indexOfId.at(edge->getTarget()->getId());
    const Complex length = edge->getLength();
    if (length.imag() != 0.0 || !(length.real() > 0.0) || !std::isfinite(length.real()))
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(std::min(a, b), std::max(a, b)) +
                                  " has length " + std::to_string(length.real()) +
                                  (length.imag() < 0 ? "-" : "+") + std::to_string(std::abs(length.imag())) +
                                  "i; the lengths must be real and positive (spec section 2)");
    edgeLengths.push_back({{std::min(a, b), std::max(a, b)}, length.real()});
  }
  std::sort(edgeLengths.begin(), edgeLengths.end());
  for (const auto &[pair, length] : edgeLengths) {
    edges_.push_back(pair);
    lengths_.push_back(length);
  }

  // Faces: the triangles, consistently oriented by the fundamental class
  // (the container sorts vertex orders, so orientation is derived here).
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &simplex : spacetime_->getSimplices()) {
    if (simplex->size() != 3) continue;
    std::vector<std::uint64_t> cell;
    for (const auto &vertex : simplex->getVertices()) cell.push_back(indexOfId.at(vertex->getId()));
    std::sort(cell.begin(), cell.end());
    cells.push_back(std::move(cell));
  }
  if (cells.empty()) throw std::invalid_argument("SimplicialQubit: the spacetime has no triangles");
  const cobordism::ChainComplex K = cobordism::ChainComplex::fromTopCells(cells);
  std::vector<int> epsilon;
  try {
    epsilon = K.fundamentalClass();
  } catch (const std::runtime_error &e) {
    throw std::invalid_argument(
        std::string("SimplicialQubit: the faces cannot be consistently oriented (") + e.what() + ")");
  }
  const auto canonical = K.kSimplexVertices(2);
  for (std::size_t t = 0; t < canonical.size(); ++t) {
    const auto &c = canonical[t];
    const bool counterclockwise = (epsilon[t] > 0) != reversed;
    faces_.push_back(counterclockwise ? Face{c[0], c[1], c[2]} : Face{c[0], c[2], c[1]});
  }

  indexEdges();
  validateCombinatorics();
  initialize();
}

void SimplicialQubit::initialize() {
  buildIncidence();
  validateCycles();
  buildFaceGeometry();
  buildWeights();
  buildHarmonicSpace();
  buildComplexStructure();
  buildHolomorphicLine();
  buildPeriodFrame();
  diagnoseDegeneration();
}

// ============================================================================
// Spec section 2: input validation
// ============================================================================

void SimplicialQubit::indexEdges() {
  const std::size_t nV = vertices_.size();
  std::vector<std::uint64_t> sorted = vertices_;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t n = 0; n < nV; ++n)
    if (sorted[n] != n)
      throw std::invalid_argument("SimplicialQubit: vertices must be exactly 0 .. nV-1");
  edgeIndex_.clear();
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const auto [i, j] = edges_[e];
    if (!(i < j) || j >= nV)
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(i, j) +
                                  " must satisfy i < j < nV");
    if (!edgeIndex_.emplace(edges_[e], e).second)
      throw std::invalid_argument("SimplicialQubit: duplicate edge " + edgeText(i, j));
  }
  if (lengths_.size() != edges_.size())
    throw std::invalid_argument("SimplicialQubit: expected one length per edge (" +
                                std::to_string(edges_.size()) + "), got " +
                                std::to_string(lengths_.size()));
  for (std::size_t e = 0; e < edges_.size(); ++e)
    if (!(lengths_[e] > 0.0) || !std::isfinite(lengths_[e]))
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(edges_[e].first, edges_[e].second) +
                                  " has length " + std::to_string(lengths_[e]) +
                                  "; lengths must be real and positive");
}

std::size_t SimplicialQubit::edgeIndexOf(std::uint64_t u, std::uint64_t v) const {
  const auto it = edgeIndex_.find({std::min(u, v), std::max(u, v)});
  if (it == edgeIndex_.end())
    throw std::invalid_argument("SimplicialQubit: " + edgeText(std::min(u, v), std::max(u, v)) +
                                " is not an edge of E");
  return it->second;
}

void SimplicialQubit::validateCombinatorics() {
  const std::size_t nV = vertices_.size();
  const std::size_t nE = edges_.size();
  const std::size_t nF = faces_.size();

  // Every face: three distinct vertices, three edges of E; record the
  // incidence (face, slot) and the traversal sign of each edge.
  edgeFaces_.assign(nE, {});
  std::vector<std::vector<int>> traversalSigns(nE);
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2] || f[0] >= nV || f[1] >= nV || f[2] >= nV)
      throw std::invalid_argument("SimplicialQubit: face " + faceText(f) +
                                  " must have three distinct vertices below nV");
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(f, slot);
      std::size_t e;
      try {
        e = edgeIndexOf(u, v);
      } catch (const std::invalid_argument &) {
        throw std::invalid_argument("SimplicialQubit: face " + faceText(f) + " uses " +
                                    edgeText(std::min(u, v), std::max(u, v)) + ", which is not in E");
      }
      edgeFaces_[e].push_back({t, slot});
      traversalSigns[e].push_back(u < v ? 1 : -1);
    }
  }

  // Every edge belongs to exactly 2 faces (closed surface), traversed in
  // opposite directions by the two (consistent orientation).
  for (std::size_t e = 0; e < nE; ++e) {
    if (edgeFaces_[e].size() != 2)
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(edges_[e].first, edges_[e].second) +
                                  " belongs to " + std::to_string(edgeFaces_[e].size()) +
                                  " faces; a closed surface needs exactly 2");
    if (traversalSigns[e][0] == traversalSigns[e][1])
      throw std::invalid_argument("SimplicialQubit: face orientations are inconsistent: faces " +
                                  faceText(faces_[edgeFaces_[e][0].first]) + " and " +
                                  faceText(faces_[edgeFaces_[e][1].first]) + " traverse edge " +
                                  edgeText(edges_[e].first, edges_[e].second) + " in the same direction");
  }

  // Euler characteristic zero.
  const long chi = static_cast<long>(nV) - static_cast<long>(nE) + static_cast<long>(nF);
  if (chi != 0)
    throw std::invalid_argument("SimplicialQubit: nV - nE + nF = " + std::to_string(chi) +
                                "; a torus needs Euler characteristic 0");

  // The strict triangle inequality on every face.
  for (const Face &f : faces_) {
    const double a = lengths_[edgeIndexOf(f[1], f[2])];
    const double b = lengths_[edgeIndexOf(f[2], f[0])];
    const double c = lengths_[edgeIndexOf(f[0], f[1])];
    if (!(a < b + c && b < c + a && c < a + b)) {
      std::ostringstream s;
      s << "SimplicialQubit: face " << faceText(f) << " violates the strict triangle inequality (a = "
        << a << ", b = " << b << ", c = " << c << ")";
      throw std::invalid_argument(s.str());
    }
  }
}

void SimplicialQubit::validateCycles() const {
  const std::size_t nV = vertices_.size();
  const std::size_t nE = edges_.size();
  auto chainOf = [&](const Cycle &cycle, const char *name) {
    if (cycle.empty())
      throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name + " is empty");
    Eigen::VectorXd chain = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nE));
    std::vector<long> net(nV, 0);
    for (const auto &[e, sign] : cycle) {
      if (e >= nE)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " refers to edge index " + std::to_string(e) + " >= nE");
      if (sign != 1 && sign != -1)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " has a sign that is not +1 or -1");
      chain(static_cast<Eigen::Index>(e)) += sign;
      const auto [i, j] = edges_[e];
      net[j] += sign;  // the edge runs i -> j; the step enters j and leaves i
      net[i] -= sign;
    }
    for (std::size_t v = 0; v < nV; ++v)
      if (net[v] != 0)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " is not closed (vertex " + std::to_string(v) + " has net degree " +
                                    std::to_string(net[v]) + ")");
    return chain;
  };
  const Eigen::VectorXd cA = chainOf(cycleA_, "A");
  const Eigen::VectorXd cB = chainOf(cycleB_, "B");

  // Independent homology classes: neither cycle, nor a combination, is a
  // boundary — the rank of [d1^T | c_A | c_B] exceeds that of d1^T by two.
  const Eigen::MatrixXd boundaries = d1_.transpose();
  Eigen::MatrixXd augmented(boundaries.rows(), boundaries.cols() + 2);
  augmented << boundaries, cA, cB;
  if (numericalRank(augmented) != numericalRank(boundaries) + 2)
    throw std::invalid_argument(
        "SimplicialQubit: the homology classes of cycles A and B are not independent");
}

// ============================================================================
// Spec section 3: incidence matrices
// ============================================================================

void SimplicialQubit::buildIncidence() {
  const Eigen::Index nV = static_cast<Eigen::Index>(vertices_.size());
  const Eigen::Index nE = static_cast<Eigen::Index>(edges_.size());
  const Eigen::Index nF = static_cast<Eigen::Index>(faces_.size());
  d0_ = Eigen::MatrixXd::Zero(nE, nV);
  for (Eigen::Index e = 0; e < nE; ++e) {
    d0_(e, static_cast<Eigen::Index>(edges_[static_cast<std::size_t>(e)].first)) = -1.0;
    d0_(e, static_cast<Eigen::Index>(edges_[static_cast<std::size_t>(e)].second)) = 1.0;
  }
  d1_ = Eigen::MatrixXd::Zero(nF, nE);
  for (Eigen::Index t = 0; t < nF; ++t)
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(faces_[static_cast<std::size_t>(t)], slot);
      d1_(t, static_cast<Eigen::Index>(edgeIndexOf(u, v))) = (u < v) ? 1.0 : -1.0;
    }
  // Check: d1 @ d0 == 0 exactly (integer arithmetic in doubles).
  if ((d1_ * d0_).cwiseAbs().maxCoeff() != 0.0)
    throw std::logic_error("SimplicialQubit: d1 d0 != 0");
}

// ============================================================================
// Spec section 4 (per-triangle geometry) and section 7 (barycentric gradients)
// ============================================================================

void SimplicialQubit::buildFaceGeometry() {
  const std::size_t nF = faces_.size();
  angles_.resize(static_cast<Eigen::Index>(nF), 3);
  areas_.resize(static_cast<Eigen::Index>(nF));
  layout_.resize(static_cast<Eigen::Index>(nF), 6);
  gradients_.resize(static_cast<Eigen::Index>(nF), 6);
  double gradientScale = 0.0, gradientDefect = 0.0;
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    const double a = lengths_[edgeIndexOf(f[1], f[2])];  // l(jk)
    const double b = lengths_[edgeIndexOf(f[2], f[0])];  // l(ki)
    const double c = lengths_[edgeIndexOf(f[0], f[1])];  // l(ij)
    const Eigen::Index row = static_cast<Eigen::Index>(t);

    const std::array<double, 3> alpha = anglesOf(a, b, c);
    angles_(row, 0) = alpha[0];
    angles_(row, 1) = alpha[1];
    angles_(row, 2) = alpha[2];

    const double s = 0.5 * (a + b + c);
    const double area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    areas_(row) = area;

    const Eigen::Vector2d pi(0.0, 0.0);
    const Eigen::Vector2d pj(c, 0.0);
    const Eigen::Vector2d pk(b * std::cos(alpha[0]), b * std::sin(alpha[0]));
    layout_.row(row) << pi(0), pi(1), pj(0), pj(1), pk(0), pk(1);

    const Eigen::Vector2d gi = rot90(pk - pj) / (2.0 * area);
    const Eigen::Vector2d gj = rot90(pi - pk) / (2.0 * area);
    const Eigen::Vector2d gk = rot90(pj - pi) / (2.0 * area);
    gradients_.row(row) << gi(0), gi(1), gj(0), gj(1), gk(0), gk(1);
    gradientScale = std::max(gradientScale, gi.norm() + gj.norm() + gk.norm());
    gradientDefect = std::max(gradientDefect, (gi + gj + gk).norm());
  }
  // Verify grad_lambda_i + grad_lambda_j + grad_lambda_k == 0.
  if (gradientDefect > 1e-10 * std::max(gradientScale, 1.0))
    throw std::logic_error("SimplicialQubit: the barycentric gradients of a face do not sum to zero");
}

// ============================================================================
// Spec section 5: cotangent weights
// ============================================================================

void SimplicialQubit::buildWeights() {
  const std::size_t nE = edges_.size();
  weights_.resize(static_cast<Eigen::Index>(nE));
  negativeWeightEdges_.clear();
  nonDelaunayEdges_.clear();
  std::vector<double> angleSums(nE, 0.0);
  for (std::size_t e = 0; e < nE; ++e) {
    double cotangentSum = 0.0;
    for (const auto &[t, slot] : edgeFaces_[e]) {
      const double angle = angles_(static_cast<Eigen::Index>(t), oppositeVertexSlot(slot));
      cotangentSum += std::cos(angle) / std::sin(angle);
      angleSums[e] += angle;
    }
    weights_(static_cast<Eigen::Index>(e)) = 0.5 * cotangentSum;
  }
  // Flags at rounding tolerance: an edge with alpha_e + beta_e = pi exactly
  // (a co-circular quadrilateral, e.g. every diagonal of a square grid) has
  // weight zero, not a negative weight, however the last bits fall.
  const double scale = std::max(weights_.cwiseAbs().maxCoeff(), 1.0);
  for (std::size_t e = 0; e < nE; ++e) {
    if (weights_(static_cast<Eigen::Index>(e)) < -1e-12 * scale) negativeWeightEdges_.push_back(e);
    if (angleSums[e] > kPi + 1e-12) nonDelaunayEdges_.push_back(e);
  }
  if (!nonDelaunayEdges_.empty() || !negativeWeightEdges_.empty()) {
    std::ostringstream w;
    w << nonDelaunayEdges_.size() << " edge(s) violate the intrinsic Delaunay condition alpha_e + beta_e "
      << "<= pi and " << negativeWeightEdges_.size() << " cotangent weight(s) are negative; the "
      << "construction is numerically stable only when the condition holds on every edge "
      << "(apply the intrinsic Delaunay edge-flip pass)";
    warnings_.push_back(w.str());
  }
}

// ============================================================================
// Spec section 6: harmonic space
// ============================================================================

void SimplicialQubit::buildHarmonicSpace() {
  const Eigen::Index nE = static_cast<Eigen::Index>(edges_.size());
  // S = vstack([d1, d0.T @ M1]); H = null_space(S).
  Eigen::MatrixXd S(d1_.rows() + d0_.cols(), nE);
  S.topRows(d1_.rows()) = d1_;
  S.bottomRows(d0_.cols()) = d0_.transpose() * weights_.asDiagonal();
  Eigen::BDCSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullV);
  const Eigen::VectorXd sigma = svd.singularValues();
  // scipy.linalg.null_space: rcond = eps * max(M, N), tolerance rcond * sigma_max.
  const double tolerance = kEps * static_cast<double>(std::max(S.rows(), S.cols())) *
                           (sigma.size() > 0 ? sigma(0) : 0.0);
  Eigen::Index rank = 0;
  for (Eigen::Index n = 0; n < sigma.size(); ++n)
    if (sigma(n) > tolerance) ++rank;
  H_ = svd.matrixV().rightCols(nE - rank);
  if (H_.cols() != 2)
    throw std::runtime_error("SimplicialQubit: dim H = " + std::to_string(H_.cols()) +
                             " != 2; the input is not a torus or the weights are degenerate");
}

// ============================================================================
// Spec section 7 (Whitney interpolant, L2 inner product) and section 8 (J)
// ============================================================================

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> SimplicialQubit::whitneyAtBarycenter(
    std::size_t t, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &omega) const {
  // W_t(omega) = (1/3) sum over the three oriented edges (u, v) of t of
  // omega_uv (grad_lambda_v - grad_lambda_u), omega_uv signed relative to the
  // stored orientation of the edge.
  const Face &f = faces_[t];
  const Eigen::Index row = static_cast<Eigen::Index>(t);
  auto gradient = [&](int vertexSlot) {
    return Eigen::Vector2d(gradients_(row, 2 * vertexSlot), gradients_(row, 2 * vertexSlot + 1));
  };
  Eigen::Matrix<Scalar, 2, 1> w = Eigen::Matrix<Scalar, 2, 1>::Zero();
  for (int slot = 0; slot < 3; ++slot) {
    const int from = slot, to = (slot + 1) % 3;
    const auto [u, v] = traversal(f, slot);
    const Scalar value = (u < v ? Scalar(1.0) : Scalar(-1.0)) *
                         omega(static_cast<Eigen::Index>(edgeIndexOf(u, v)));
    w += value * (gradient(to) - gradient(from)).template cast<Scalar>();
  }
  return w / Scalar(3.0);
}

void SimplicialQubit::buildComplexStructure() {
  const std::size_t nF = faces_.size();
  G_ = Eigen::MatrixXd::Zero(2, 2);
  R_ = Eigen::MatrixXd::Zero(2, 2);
  for (std::size_t t = 0; t < nF; ++t) {
    const double area = areas_(static_cast<Eigen::Index>(t));
    const Eigen::Vector2d w0 = whitneyAtBarycenter<double>(t, H_.col(0));
    const Eigen::Vector2d w1 = whitneyAtBarycenter<double>(t, H_.col(1));
    const std::array<Eigen::Vector2d, 2> w = {w0, w1};
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        G_(a, b) += area * w[a].dot(w[b]);
        R_(a, b) += area * rot90(w[a]).dot(w[b]);
      }
  }
  // J = G^{-1} @ R.T; the residual is exposed, never symmetrized away.
  J_ = G_.inverse() * R_.transpose();
  jResidual_ = (J_ * J_ + Eigen::MatrixXd::Identity(2, 2)).norm();
}

// ============================================================================
// Spec section 9: holomorphic line and period ratio
// ============================================================================

Complex SimplicialQubit::periodOf(const Eigen::VectorXcd &omega, const Cycle &cycle) const {
  Complex total(0.0, 0.0);
  for (const auto &[e, sign] : cycle) total += static_cast<double>(sign) * omega(static_cast<Eigen::Index>(e));
  return total;
}

void SimplicialQubit::buildHolomorphicLine() {
  // Complexify: the eigenvector of J for the eigenvalue closest to -i.
  Eigen::EigenSolver<Eigen::MatrixXd> eig(J_);
  const Eigen::VectorXcd lambda = eig.eigenvalues();
  int branch = 0;
  for (int b = 1; b < lambda.size(); ++b)
    if (std::abs(lambda(b) - Complex(0.0, -1.0)) < std::abs(lambda(branch) - Complex(0.0, -1.0)))
      branch = b;
  Eigen::VectorXcd c = eig.eigenvectors().col(branch);

  auto evaluate = [&](const Eigen::VectorXcd &coefficients, Eigen::VectorXcd &omega, Complex &pA,
                      Complex &pB, bool &swapped) {
    omega = H_.cast<Complex>() * coefficients;  // omega = c[0] h1 + c[1] h2
    const Complex rawA = periodOf(omega, cycleA_);
    const Complex rawB = periodOf(omega, cycleB_);
    // |P_A| near zero: the marking is degenerate for this metric; swap the
    // roles of A and B — the marking (B, -A) — and report -1/tau.
    swapped = std::abs(rawA) <= 1e-12 * (std::abs(rawA) + std::abs(rawB));
    if (swapped) {
      pA = rawB;
      pB = -rawA;
    } else {
      pA = rawA;
      pB = rawB;
    }
    return pB / pA;
  };

  Eigen::VectorXcd omega;
  Complex pA, pB;
  bool swapped = false;
  Complex tau = evaluate(c, omega, pA, pB, swapped);
  // Require Im(tau) > 0; otherwise the surface orientation or the eigenvalue
  // branch is flipped: take the conjugate eigenvector and recompute.
  if (tau.imag() < 0.0) {
    c = c.conjugate();
    tau = evaluate(c, omega, pA, pB, swapped);
    warnings_.push_back(
        "Im tau < 0 on the eigenvector nearest -i: the surface orientation or the eigenvalue branch is "
        "flipped; the conjugate eigenvector was taken");
  }
  if (!std::isfinite(tau.real()) || !std::isfinite(tau.imag()))
    throw std::runtime_error("SimplicialQubit: the period ratio is not finite (both periods vanish)");
  if (tau.imag() == 0.0)
    warnings_.push_back("Im tau = 0: the marking is degenerate for this metric (a pinched cycle)");
  if (swapped) warnings_.push_back("|P_A| vanished: the roles of A and B were swapped and -1/tau is reported");

  omega_ = omega;
  periodA_ = pA;
  periodB_ = pB;
  tau_ = tau;
  swapped_ = swapped;

  // Section 10's assertion: |r| == 1 to machine precision.
  const double blochNorm = bloch().norm();
  if (std::abs(blochNorm - 1.0) > 1e-12)
    throw std::logic_error("SimplicialQubit: the Bloch vector is not a unit vector (|r| = " +
                           std::to_string(blochNorm) + ")");
}

// ============================================================================
// Spec section 10: the qubit state
// ============================================================================

Eigen::VectorXcd SimplicialQubit::state() const {
  const double norm = std::sqrt(1.0 + std::norm(tau_));
  Eigen::VectorXcd psi(2);
  psi(0) = Complex(1.0, 0.0) / norm;
  psi(1) = tau_ / norm;
  return psi;
}

Eigen::VectorXd SimplicialQubit::bloch() const {
  const double N = 1.0 + std::norm(tau_);
  Eigen::VectorXd r(3);
  r(0) = 2.0 * tau_.real() / N;
  r(1) = 2.0 * tau_.imag() / N;
  r(2) = (1.0 - std::norm(tau_)) / N;
  return r;
}

Eigen::MatrixXcd SimplicialQubit::densityMatrix() const {
  const Eigen::VectorXd r = bloch();
  Eigen::MatrixXcd rho(2, 2);
  rho(0, 0) = 0.5 * (1.0 + r(2));
  rho(0, 1) = 0.5 * Complex(r(0), -r(1));
  rho(1, 0) = 0.5 * Complex(r(0), r(1));
  rho(1, 1) = 0.5 * (1.0 - r(2));
  return rho;
}

// ============================================================================
// Spec section 11: the two metrics on the state space
// ============================================================================

double SimplicialQubit::fubiniStudyDistance(const SimplicialQubit &q1, const SimplicialQubit &q2) {
  const Complex t1 = q1.tau(), t2 = q2.tau();
  const double overlap =
      std::abs(Complex(1.0, 0.0) + std::conj(t1) * t2) / std::sqrt((1.0 + std::norm(t1)) * (1.0 + std::norm(t2)));
  return std::acos(std::clamp(overlap, 0.0, 1.0));
}

double SimplicialQubit::weilPeterssonDistance(const SimplicialQubit &q1, const SimplicialQubit &q2) {
  const Complex t1 = q1.tau(), t2 = q2.tau();
  if (!(t1.imag() > 0.0) || !(t2.imag() > 0.0))
    throw std::invalid_argument(
        "SimplicialQubit::weilPeterssonDistance: both period ratios must lie in the upper half plane");
  return std::acosh(std::max(1.0 + std::norm(t1 - t2) / (2.0 * t1.imag() * t2.imag()), 1.0));
}

// ============================================================================
// Spec section 12: the flat torus C / (Z + tau Z)
// ============================================================================

SimplicialQubit SimplicialQubit::flatTorus(std::complex<double> tau, int nx, int ny) {
  if (!(tau.imag() > 0.0))
    throw std::invalid_argument("SimplicialQubit::flatTorus: tau must lie in the upper half plane");
  if (nx < 3 || ny < 3)
    throw std::invalid_argument(
        "SimplicialQubit::flatTorus: the grid needs at least 3 vertices per side to be a simplicial complex");

  // The unit cell spanned by 1 and tau; vertex (i, j) at i/nx + tau j/ny; every
  // square split by its diagonal; sides identified.
  const auto vid = [ny](int i, int j) {
    return static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j);
  };
  const Complex e1(1.0 / nx, 0.0);
  const Complex e2 = tau / static_cast<double>(ny);
  std::vector<std::uint64_t> vertices(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (std::size_t n = 0; n < vertices.size(); ++n) vertices[n] = n;

  std::map<EdgePair, double> lengthOf;
  auto addEdge = [&](std::uint64_t u, std::uint64_t v, Complex displacement) {
    lengthOf[{std::min(u, v), std::max(u, v)}] = std::abs(displacement);
  };
  std::vector<Face> faces;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) {
      const int i1 = (i + 1) % nx, j1 = (j + 1) % ny;
      addEdge(vid(i, j), vid(i1, j), e1);
      addEdge(vid(i, j), vid(i, j1), e2);
      addEdge(vid(i, j), vid(i1, j1), e1 + e2);
      // Counterclockwise for Im tau > 0: (0, e1, e1 + e2) and (0, e1 + e2, e2).
      faces.push_back({vid(i, j), vid(i1, j), vid(i1, j1)});
      faces.push_back({vid(i, j), vid(i1, j1), vid(i, j1)});
    }
  std::vector<EdgePair> edges;
  std::vector<double> lengths;
  std::map<EdgePair, std::size_t> index;
  for (const auto &[pair, length] : lengthOf) {
    index[pair] = edges.size();
    edges.push_back(pair);
    lengths.push_back(length);
  }

  // Marked cycles: A the row loop along 1 (j = 0), B the column loop along
  // tau (i = 0); a step's sign is +1 when it runs from the smaller id.
  auto step = [&](std::uint64_t from, std::uint64_t to) {
    return CycleStep{index.at({std::min(from, to), std::max(from, to)}), from < to ? 1 : -1};
  };
  Cycle cycleA, cycleB;
  for (int i = 0; i < nx; ++i) cycleA.push_back(step(vid(i, 0), vid((i + 1) % nx, 0)));
  for (int j = 0; j < ny; ++j) cycleB.push_back(step(vid(0, j), vid(0, (j + 1) % ny)));

  return SimplicialQubit(std::move(vertices), std::move(edges), std::move(faces), std::move(lengths),
                         std::move(cycleA), std::move(cycleB));
}

// ============================================================================
// Spec section 5: the optional intrinsic Delaunay edge-flip pass
// ============================================================================

SimplicialQubit SimplicialQubit::intrinsicDelaunay() const {
  std::vector<EdgePair> edges = edges_;
  std::vector<Face> faces = faces_;
  std::vector<double> lengths = lengths_;
  Cycle cycleA = cycleA_, cycleB = cycleB_;
  std::map<EdgePair, std::size_t> index = edgeIndex_;
  std::vector<std::vector<std::pair<std::size_t, int>>> incidence = edgeFaces_;
  std::vector<std::string> skipped;

  auto lengthOf = [&](std::uint64_t u, std::uint64_t v) {
    return lengths[index.at({std::min(u, v), std::max(u, v)})];
  };
  auto faceAngles = [&](const Face &f) {
    return anglesOf(lengthOf(f[1], f[2]), lengthOf(f[2], f[0]), lengthOf(f[0], f[1]));
  };
  auto oppositeAngle = [&](std::size_t t, int slot) {
    return faceAngles(faces[t])[static_cast<std::size_t>(oppositeVertexSlot(slot))];
  };
  auto detach = [&](std::size_t e, std::size_t t) {
    auto &list = incidence[e];
    list.erase(std::remove_if(list.begin(), list.end(),
                              [t](const std::pair<std::size_t, int> &p) { return p.first == t; }),
               list.end());
  };
  auto attach = [&](std::size_t t) {
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(faces[t], slot);
      incidence[index.at({std::min(u, v), std::max(u, v)})].push_back({t, slot});
    }
  };

  std::deque<std::size_t> queue;
  std::vector<bool> queued(edges.size(), true);
  for (std::size_t e = 0; e < edges.size(); ++e) queue.push_back(e);
  int flips = 0;
  const std::size_t cap = 100 * edges.size() + 100;
  std::size_t iterations = 0;
  while (!queue.empty()) {
    if (++iterations > cap)
      throw std::runtime_error("SimplicialQubit::intrinsicDelaunay: the flip pass did not terminate");
    const std::size_t e = queue.front();
    queue.pop_front();
    queued[e] = false;
    const auto &pair = incidence[e];
    const double alpha = oppositeAngle(pair[0].first, pair[0].second);
    const double beta = oppositeAngle(pair[1].first, pair[1].second);
    if (!(alpha + beta > kPi + 1e-12)) continue;

    // The two faces: t1 traverses the edge i -> j (its apex k), t2 traverses
    // j -> i (its apex l).
    const auto [u, v] = edges[e];
    std::size_t t1 = pair[0].first, t2 = pair[1].first;
    int slot1 = pair[0].second, slot2 = pair[1].second;
    if (traversal(faces[t1], slot1).first != u) {
      std::swap(t1, t2);
      std::swap(slot1, slot2);
    }
    const std::uint64_t i = u, j = v;
    const std::uint64_t k = faces[t1][static_cast<std::size_t>(oppositeVertexSlot(slot1))];
    const std::uint64_t l = faces[t2][static_cast<std::size_t>(oppositeVertexSlot(slot2))];
    if (k == l || index.count({std::min(k, l), std::max(k, l)}) != 0) {
      skipped.push_back("edge " + edgeText(i, j) + " violates the Delaunay condition but its flip would "
                        "duplicate edge " + edgeText(std::min(k, l), std::max(k, l)) + "; left in place");
      continue;
    }

    // Lay both triangles out in the frame of t1 (spec section 4): p_i = 0,
    // p_j = (c, 0), p_k above the edge, p_l below it; the new diagonal is the
    // distance between the two apexes.
    const double a = lengthOf(j, k), b = lengthOf(k, i), c = lengthOf(i, j);
    const double aPrime = lengthOf(j, l), bPrime = lengthOf(l, i);
    const double cosK = std::clamp((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0);
    const double cosL = std::clamp((bPrime * bPrime + c * c - aPrime * aPrime) / (2.0 * bPrime * c), -1.0, 1.0);
    const Eigen::Vector2d pk(b * cosK, b * std::sqrt(1.0 - cosK * cosK));
    const Eigen::Vector2d pl(bPrime * cosL, -bPrime * std::sqrt(1.0 - cosL * cosL));
    const double newLength = (pk - pl).norm();

    // Replace: faces (i,j,k), (j,i,l) -> (k,i,l), (l,j,k); edge (i,j) -> (k,l).
    for (int slot = 0; slot < 3; ++slot) {
      const auto [x, y] = traversal(faces[t1], slot);
      detach(index.at({std::min(x, y), std::max(x, y)}), t1);
      const auto [p, q] = traversal(faces[t2], slot);
      detach(index.at({std::min(p, q), std::max(p, q)}), t2);
    }
    index.erase({i, j});
    edges[e] = {std::min(k, l), std::max(k, l)};
    lengths[e] = newLength;
    index[edges[e]] = e;
    faces[t1] = {k, i, l};
    faces[t2] = {l, j, k};
    attach(t1);
    attach(t2);
    ++flips;

    // Reroute the marked cycles: a step across (i, j) becomes the path through k.
    auto reroute = [&](Cycle &cycle) {
      Cycle out;
      for (const auto &[edge, sign] : cycle) {
        if (edge != e) {
          out.push_back({edge, sign});
          continue;
        }
        const std::uint64_t from = sign > 0 ? i : j, to = sign > 0 ? j : i;
        out.push_back({index.at({std::min(from, k), std::max(from, k)}), from < k ? 1 : -1});
        out.push_back({index.at({std::min(k, to), std::max(k, to)}), k < to ? 1 : -1});
      }
      cycle = std::move(out);
    };
    reroute(cycleA);
    reroute(cycleB);

    // The four sides of the quadrilateral may have become non-Delaunay.
    for (const auto &[x, y] : {EdgePair{i, k}, EdgePair{k, j}, EdgePair{j, l}, EdgePair{l, i}}) {
      const std::size_t side = index.at({std::min(x, y), std::max(x, y)});
      if (!queued[side]) {
        queued[side] = true;
        queue.push_back(side);
      }
    }
  }

  SimplicialQubit result(vertices_, std::move(edges), std::move(faces), std::move(lengths), std::move(cycleA),
                         std::move(cycleB), degeneracyThreshold_);
  result.flips_ = flips;
  result.warnings_.insert(result.warnings_.end(), skipped.begin(), skipped.end());
  return result;
}

// ============================================================================
// Spec section 13: degeneration diagnostics
// ============================================================================

void SimplicialQubit::buildPeriodFrame() {
  // Pi_{ca} = the period of h_a over cycle c, the cycles in force: (A, B), or
  // (B, -A) when section 9 swapped the marking (|P_A| vanished), so that the
  // frame is the basis tau() is a coordinate in.
  Eigen::Matrix2d Pi;
  for (Eigen::Index a = 0; a < 2; ++a) {
    const Eigen::VectorXcd h = H_.col(a).cast<Complex>();
    const double pA = periodOf(h, cycleA_).real();
    const double pB = periodOf(h, cycleB_).real();
    Pi(0, a) = swapped_ ? pB : pA;
    Pi(1, a) = swapped_ ? -pA : pB;
  }
  // F = H Pi^{-1}: the period of column c of F over cycle c' is
  // sum_a Pi_{c'a} (Pi^{-1})_{ac} = delta_{c'c}. The period map of the
  // harmonic space over a homology basis is an isomorphism (section 2 checked
  // the cycles' independence), so Pi is invertible whenever dim H = 2.
  const double det = Pi.determinant();
  if (!std::isfinite(det) || det == 0.0)
    throw std::runtime_error("SimplicialQubit: the period matrix of the harmonic basis over the marking is "
                             "singular; the marked cycles do not span the torus's homology");
  F_ = H_ * Pi.inverse();
}

void SimplicialQubit::diagnoseDegeneration() {
  const Eigen::VectorXd magnitudes = weights_.cwiseAbs();
  const double smallest = magnitudes.minCoeff();
  condM1_ = smallest > 0.0 ? magnitudes.maxCoeff() / smallest : std::numeric_limits<double>::infinity();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(G_);
  const Eigen::VectorXd sigma = svd.singularValues();
  condG_ = sigma(1) > 0.0 ? sigma(0) / sigma(1) : std::numeric_limits<double>::infinity();
  nearDegenerate_ = condM1_ > degeneracyThreshold_ || condG_ > degeneracyThreshold_;
  if (nearDegenerate_) {
    const long zeroWeights = std::count_if(
        magnitudes.data(), magnitudes.data() + magnitudes.size(),
        [&](double w) { return w <= 1e-12 * std::max(magnitudes.maxCoeff(), 1.0); });
    std::ostringstream w;
    w << "near-degenerate: cond(M1) = " << condM1_ << ", cond(G) = " << condG_ << " against the threshold "
      << degeneracyThreshold_ << " (";
    if (zeroWeights > 0)
      w << zeroWeights << " cotangent weight(s) vanish, alpha_e + beta_e = pi on those edges";
    else
      w << "a cycle is pinching";
    w << "; the state remains valid)";
    warnings_.push_back(w.str());
  }
}

}  // namespace tessera::observables
