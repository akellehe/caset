// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SimplicialQubit.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/PolygonCircle.h"
#include "spacetime/topologies/SimplicialProduct.h"

namespace tessera::observables {

namespace {

using Complex = std::complex<double>;
using EdgeKeyPair = std::pair<std::uint64_t, std::uint64_t>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Canonical edge index of the complex by (min, max) vertex pair.
std::map<EdgeKeyPair, int> edgeIndexOf(const cobordism::ChainComplex &K) {
  std::map<EdgeKeyPair, int> index;
  const auto edges = K.kSimplexVertices(1);
  for (int j = 0; j < static_cast<int>(edges.size()); ++j)
    index[{edges[static_cast<std::size_t>(j)][0], edges[static_cast<std::size_t>(j)][1]}] = j;
  return index;
}

/// Every consecutive pair of the closed walk as (edge index, traversal sign):
/// +1 along the ascending-id reference orientation, -1 against it — the sign
/// rule `EigenstateSynthesis::EdgeLoop` and `Edge::walkLoop` use.
std::vector<std::pair<int, double>> signedEdgesOf(const SimplicialQubit::Cycle &cycle,
                                                  const std::map<EdgeKeyPair, int> &index,
                                                  const char *name) {
  if (cycle.size() < 2)
    throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                " must be a closed walk of at least two vertices");
  std::vector<std::pair<int, double>> out;
  out.reserve(cycle.size());
  for (std::size_t i = 0; i < cycle.size(); ++i) {
    const std::uint64_t u = cycle[i];
    const std::uint64_t v = cycle[(i + 1) % cycle.size()];
    if (u == v)
      throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                  " repeats vertex " + std::to_string(u) + " consecutively");
    const auto it = index.find({std::min(u, v), std::max(u, v)});
    if (it == index.end())
      throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name + " step (" +
                                  std::to_string(u) + "," + std::to_string(v) +
                                  ") is not an edge of the complex");
    out.emplace_back(it->second, u < v ? 1.0 : -1.0);
  }
  return out;
}

/// The signed sum of a cochain over a closed walk: its period.
Complex periodOf(const Eigen::VectorXcd &cochain,
                 const std::vector<std::pair<int, double>> &signedEdges) {
  Complex total(0.0, 0.0);
  for (const auto &[edge, sign] : signedEdges) total += sign * cochain(edge);
  return total;
}

std::vector<Complex> columnOf(const Eigen::MatrixXcd &m, int column) {
  std::vector<Complex> out(static_cast<std::size_t>(m.rows()));
  for (Eigen::Index i = 0; i < m.rows(); ++i) out[static_cast<std::size_t>(i)] = m(i, column);
  return out;
}

double conditionOf(const Eigen::MatrixXcd &m) {
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(m);
  const Eigen::VectorXd sv = svd.singularValues();
  if (sv.size() == 0) return kNaN;
  const double smallest = sv(sv.size() - 1);
  return smallest > 0.0 ? sv(0) / smallest : std::numeric_limits<double>::infinity();
}

std::string formatComplex(Complex z) {
  std::ostringstream s;
  s.precision(6);
  s << z.real() << (z.imag() < 0 ? "-" : "+") << std::abs(z.imag()) << "i";
  return s.str();
}

std::vector<Complex> flattenRowMajor(const Eigen::MatrixXcd &m) {
  std::vector<Complex> out;
  out.reserve(static_cast<std::size_t>(m.size()));
  for (Eigen::Index i = 0; i < m.rows(); ++i)
    for (Eigen::Index j = 0; j < m.cols(); ++j) out.push_back(m(i, j));
  return out;
}

}  // namespace

Record SimplicialQubitRead::toRecord() const {
  Record::Map map;
  map["vertices"] = Record(vertices);
  map["edges"] = Record(edges);
  map["faces"] = Record(faces);
  map["euler_characteristic"] = Record(eulerCharacteristic);
  Record::List bettiList;
  for (int b : betti) bettiList.emplace_back(b);
  map["betti"] = Record(std::move(bettiList));
  map["harmonic_rank"] = Record(harmonicRank);
  map["harmonic_gap"] = Record(harmonicGap);
  map["twisted_harmonic_rank"] = Record(twistedHarmonicRank);
  Record::splitComplex(map, "harmonic_images", flattenRowMajor(harmonicImages));
  Record::splitComplex(map, "gram", flattenRowMajor(gram));
  Record::splitComplex(map, "intersection", flattenRowMajor(intersection));
  Record::splitComplex(map, "complex_structure", flattenRowMajor(complexStructure));
  map["complex_structure_residual"] = Record(complexStructureResidual);
  Record::splitComplex(map, "periods", flattenRowMajor(periods));
  Record::splitComplex(map, "intersection_number", intersectionNumber);
  Record::splitComplex(map, "holomorphic_form", flattenRowMajor(holomorphicForm));
  Record::splitComplex(map, "period_a", periodA);
  Record::splitComplex(map, "period_b", periodB);
  Record::splitComplex(map, "tau", tau);
  map["marking_swapped"] = Record(markingSwapped);
  Record::splitComplex(map, "state", flattenRowMajor(state));
  Record::List blochList;
  for (Eigen::Index i = 0; i < bloch.size(); ++i) blochList.emplace_back(bloch(i));
  map["bloch"] = Record(std::move(blochList));
  map["bloch_norm"] = Record(blochNorm);
  Record::splitComplex(map, "density", flattenRowMajor(density));
  map["metric_condition"] = Record(metricCondition);
  map["gram_condition"] = Record(gramCondition);
  map["near_degenerate"] = Record(nearDegenerate);
  map["warning"] = Record(warning);
  map["refusal"] = Record(refusal);
  return Record(std::move(map));
}

SimplicialQubit::SimplicialQubit(Cycle cycleA, Cycle cycleB, bool reversed,
                                 double degeneracyThreshold)
    : cycleA_(std::move(cycleA)),
      cycleB_(std::move(cycleB)),
      reversed_(reversed),
      degeneracyThreshold_(degeneracyThreshold) {}

double SimplicialQubit::compute(const std::shared_ptr<Spacetime> &spacetime) {
  const SimplicialQubitRead r = read(spacetime);
  return r.holds() ? r.complexStructureResidual : kNaN;
}

SimplicialQubitRead SimplicialQubit::read(const std::shared_ptr<Spacetime> &spacetime) const {
  if (!spacetime) throw std::invalid_argument("SimplicialQubit::read: null spacetime");
  SimplicialQubitRead out;

  // ---- 1. Topology: a closed, connected, oriented surface with chi = 0 and
  //         b = (1, 2, 1), and a manifold (every edge in exactly two faces).
  const cobordism::ChainComplex K = chainhodge::WhitneyMass::complexOf(*spacetime);
  if (K.dimension() != 2) {
    out.refusal = "the complex is " + std::to_string(K.dimension()) +
                  "-dimensional; a qubit is read from a triangulated surface (dimension 2)";
    return out;
  }
  out.vertices = K.numSimplices(0);
  out.edges = K.numSimplices(1);
  out.faces = K.numSimplices(2);
  out.eulerCharacteristic = K.eulerCharacteristic();
  out.betti = K.bettiNumbers();
  const auto boundary = spacetime->getBoundary();
  if (!boundary.empty()) {
    out.refusal = "the surface has a boundary (" + std::to_string(boundary.size()) +
                  " boundary edges); a torus is closed";
    return out;
  }
  const auto [manifold, why] = cobordism::ChainComplex::dualComplexIsValid(K.kSimplexVertices(2), 2);
  if (!manifold) {
    out.refusal = "the surface is not a manifold: " + why;
    return out;
  }
  if (out.eulerCharacteristic != 0 || out.betti != std::vector<int>{1, 2, 1}) {
    std::ostringstream s;
    s << "not a torus: chi = " << out.eulerCharacteristic << ", b = (";
    for (std::size_t i = 0; i < out.betti.size(); ++i) s << (i ? ", " : "") << out.betti[i];
    s << ")";
    out.refusal = s.str();
    return out;
  }
  // The cup-product pairing needs the fundamental class; its absence names a
  // non-orientable (or disconnected) surface.
  std::optional<cobordism::ChainComplex::CupProductForm> form;
  try {
    form.emplace(K.cupProductForm(1));
  } catch (const std::runtime_error &e) {
    out.refusal = std::string("no fundamental class (the surface is not closed-orientable): ") +
                  e.what();
    return out;
  }

  // ---- 2. The marking: closed walks of edges of the complex (structural
  //         errors throw; homological failures are refused below).
  const std::map<EdgeKeyPair, int> edgeIndex = edgeIndexOf(K);
  const auto walkA = signedEdgesOf(cycleA_, edgeIndex, "A");
  const auto walkB = signedEdgesOf(cycleB_, edgeIndex, "B");

  // ---- 3. The pencil and its exact zero mode at degree 1.
  const chainhodge::SquaredLengths s = chainhodge::WhitneyMass::squaredLengthsOf(*spacetime, K);
  std::optional<chainhodge::ChainHodge> hodge;
  try {
    hodge.emplace(K, s);
  } catch (const std::exception &e) {
    out.refusal = std::string("degenerate metric (the Whitney mass could not be assembled): ") +
                  e.what();
    return out;
  }
  const chainhodge::HarmonicRead harmonic = hodge->harmonicChains(1);
  out.harmonicRank = harmonic.nullity;
  out.harmonicGap = harmonic.gap;
  if (harmonic.nullity != 2) {
    out.refusal = "harmonic rank " + std::to_string(harmonic.nullity) +
                  " at degree 1 (a torus has 2): the metric is degenerate";
    return out;
  }
  out.harmonicChains = harmonic.chains;
  out.harmonicImages = harmonic.images;

  // ---- 4. Phases on: the twisted zero mode certifies a pure-gauge connection.
  const chainhodge::Connection U = chainhodge::Connection::fromSpacetime(*spacetime, K);
  const bool trivialLinks =
      std::all_of(U.links().begin(), U.links().end(), [](Complex u) { return u == Complex(1.0, 0.0); });
  if (trivialLinks) {
    out.twistedHarmonicRank = harmonic.nullity;
  } else {
    const chainhodge::CovariantChainHodge covariant(*hodge, U, 7, false);
    out.twistedHarmonicRank = covariant.harmonicChains(1).nullity;
  }
  if (out.twistedHarmonicRank != 2) {
    out.refusal = "the link phases carry holonomy or flux: twisted harmonic rank " +
                  std::to_string(out.twistedHarmonicRank) +
                  " (a qubit needs the full rank 2, a pure-gauge connection)";
    return out;
  }

  // ---- 5. The Whitney Gram and the intersection form on the basis.
  out.gram = hodge->harmonicGram(harmonic);
  const double orientationSign = reversed_ ? -1.0 : 1.0;
  Eigen::MatrixXcd R(2, 2);
  {
    const std::vector<Complex> z0 = columnOf(harmonic.images, 0);
    const std::vector<Complex> z1 = columnOf(harmonic.images, 1);
    R(0, 0) = orientationSign * form->evaluate(z0, z0);
    R(0, 1) = orientationSign * form->evaluate(z0, z1);
    R(1, 0) = orientationSign * form->evaluate(z1, z0);
    R(1, 1) = orientationSign * form->evaluate(z1, z1);
  }
  out.intersection = R;

  // ---- 6. The complex structure J = G^{-1} R^T and its residual.
  Eigen::FullPivLU<Eigen::MatrixXcd> gramLU(out.gram);
  if (!gramLU.isInvertible()) {
    out.refusal = "the harmonic Gram is singular (an isotropic harmonic direction)";
    return out;
  }
  const Eigen::MatrixXcd J = gramLU.solve(R.transpose());
  out.complexStructure = J;
  out.complexStructureResidual = (J * J + Eigen::MatrixXcd::Identity(2, 2)).norm();

  // ---- 7. Periods of the basis over the marking; A·B recovered.
  Eigen::MatrixXcd P(2, 2);
  for (int a = 0; a < 2; ++a) {
    P(a, 0) = periodOf(harmonic.images.col(a), walkA);
    P(a, 1) = periodOf(harmonic.images.col(a), walkB);
  }
  out.periods = P;
  const Complex detP = P(0, 0) * P(1, 1) - P(1, 0) * P(0, 1);
  const double scaleP = P.norm() * P.norm();
  if (std::abs(detP) <= 1e-12 * std::max(scaleP, 1e-300)) {
    out.refusal = "the marked cycles are homologically dependent (singular period matrix)";
    return out;
  }
  // R_01 = (A·B) (P_0A P_1B - P_1A P_0B): the intersection number of the marking.
  out.intersectionNumber = R(0, 1) / detP;
  if (std::abs(out.intersectionNumber - Complex(1.0, 0.0)) > 1e-6) {
    out.refusal = "the marking has A·B = " + formatComplex(out.intersectionNumber) +
                  "; a qubit needs A·B = +1 (swap A and B, or read with `reversed`)";
    return out;
  }

  // ---- 8. The holomorphic line and the period ratio.
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eig(J);
  const Eigen::VectorXcd lambda = eig.eigenvalues();
  int branch = std::abs(lambda(0) - Complex(0.0, -1.0)) <= std::abs(lambda(1) - Complex(0.0, -1.0)) ? 0 : 1;
  const double formScale = harmonic.images.norm();
  auto evaluateBranch = [&](int b, Eigen::VectorXcd &omega, Complex &pA, Complex &pB, bool &swapped) {
    omega = harmonic.images * eig.eigenvectors().col(b);
    const Complex rawA = periodOf(omega, walkA);
    const Complex rawB = periodOf(omega, walkB);
    // |P_A| ≈ 0: the marking is degenerate for this metric; report in the
    // marking (A', B') = (B, -A), i.e. -1/tau (specification §9).
    swapped = std::abs(rawA) <= 1e-12 * formScale * std::sqrt(static_cast<double>(walkA.size())) &&
              std::abs(rawB) > std::abs(rawA);
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
  Complex tau = evaluateBranch(branch, omega, pA, pB, swapped);
  if (!(tau.imag() > 0.0)) {
    // The other eigenline (the conjugate branch for a real metric).
    Eigen::VectorXcd omegaOther;
    Complex pAOther, pBOther;
    bool swappedOther = false;
    const Complex tauOther = evaluateBranch(1 - branch, omegaOther, pAOther, pBOther, swappedOther);
    if (tauOther.imag() > 0.0 || (std::isnan(tau.imag()) && !std::isnan(tauOther.imag()))) {
      branch = 1 - branch;
      omega = omegaOther;
      pA = pAOther;
      pB = pBOther;
      swapped = swappedOther;
      tau = tauOther;
    } else {
      out.warning += "neither eigenline of J has Im tau > 0 (a complex metric): the state lies in "
                     "the opposite hemisphere; ";
    }
  }
  out.holomorphicForm = omega;
  out.periodA = pA;
  out.periodB = pB;
  out.tau = tau;
  out.markingSwapped = swapped;

  // ---- 9. The point of CP^1.
  out.state = stateOf(tau);
  out.bloch = blochOf(tau);
  out.blochNorm = out.bloch.norm();
  out.density = densityOf(tau);

  // ---- 10. Degeneration diagnostics (warn, never fail).
  out.gramCondition = conditionOf(out.gram);
  const int n1 = hodge->size(1);
  if (n1 < hodge->crossoverDimension())
    out.metricCondition = conditionOf(Eigen::MatrixXcd(hodge->Minv(1)));
  const bool gramNear = std::isfinite(out.gramCondition) ? out.gramCondition > degeneracyThreshold_
                                                         : std::isinf(out.gramCondition);
  const bool metricNear = std::isfinite(out.metricCondition) ? out.metricCondition > degeneracyThreshold_
                                                             : std::isinf(out.metricCondition);
  out.nearDegenerate = gramNear || metricNear;
  if (out.nearDegenerate) {
    std::ostringstream w;
    w << "near-degenerate torus: cond(M1) = " << out.metricCondition << ", cond(G) = "
      << out.gramCondition << " (threshold " << degeneracyThreshold_ << "); a cycle is pinching";
    out.warning += w.str();
  }
  return out;
}

MarkedTorus SimplicialQubit::flatTorus(std::complex<double> tau, int nx, int ny) {
  if (!(tau.imag() > 0.0))
    throw std::invalid_argument("SimplicialQubit::flatTorus: tau must lie in the upper half plane");
  if (nx < 3 || ny < 3)
    throw std::invalid_argument("SimplicialQubit::flatTorus: each side needs at least 3 vertices");

  // The nx x ny grid torus: product of two polygon circles, dimension-2
  // signature so the triangles register as top cells.
  auto metric = std::make_shared<Metric>(true, Signature(2, SignatureType::Lorentzian));
  auto topology = std::make_shared<SimplicialProduct>(std::make_shared<PolygonCircle>(nx),
                                                      std::make_shared<PolygonCircle>(ny));
  auto st = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0, Foliation::PREFERRED,
                                        topology);
  st->build(0);

  // Lattice vectors of the fundamental domain spanned by 1 and tau, per grid
  // step. A product vertex (i, j) has id i*ny + j; an edge's displacement is
  // its (di, dj) reduced to {-1, 0, +1} (the wrap edge (0, n-1) is a step of
  // -1), so each square's diagonal — whichever the staircase chose — gets the
  // Euclidean length of the displacement it actually spans.
  const double e1x = 1.0 / nx, e1y = 0.0;
  const double e2x = tau.real() / ny, e2y = tau.imag() / ny;
  auto reduce = [](std::int64_t delta, int n) -> int {
    if (delta == n - 1) return -1;
    if (delta == -(n - 1)) return 1;
    return static_cast<int>(delta);
  };
  for (Edge *edge : st->getEdgeList()->toVector()) {
    const std::uint64_t u = edge->getSource()->getId();
    const std::uint64_t v = edge->getTarget()->getId();
    const int di = reduce(static_cast<std::int64_t>(v / static_cast<std::uint64_t>(ny)) -
                              static_cast<std::int64_t>(u / static_cast<std::uint64_t>(ny)),
                          nx);
    const int dj = reduce(static_cast<std::int64_t>(v % static_cast<std::uint64_t>(ny)) -
                              static_cast<std::int64_t>(u % static_cast<std::uint64_t>(ny)),
                          ny);
    const double dx = di * e1x + dj * e2x;
    const double dy = di * e1y + dj * e2y;
    edge->setLength(Complex(std::sqrt(dx * dx + dy * dy), 0.0));
    edge->setPhase(Complex(0.0, 0.0));
  }

  // The marking: the row loop along 1 and the column loop along tau.
  auto vid = [ny](int i, int j) { return static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j); };
  Cycle cycleA, cycleB;
  for (int i = 0; i < nx; ++i) cycleA.push_back(vid(i, 0));
  for (int j = 0; j < ny; ++j) cycleB.push_back(vid(0, j));

  // Orientation: the triangle [(0,0), (1,0), (1,1)] traversed in ascending id
  // order runs along 1 then along tau, i.e. counterclockwise in the plane
  // (Im tau > 0). If the fundamental class gives it -1 the reference
  // orientation is the plane's opposite and the read must flip it for
  // A·B = +1.
  const cobordism::ChainComplex K = chainhodge::WhitneyMass::complexOf(*st);
  const std::vector<int> epsilon = K.fundamentalClass();
  const auto cells = K.kSimplexVertices(2);
  const std::vector<std::uint64_t> corner = {vid(0, 0), vid(1, 0), vid(1, 1)};
  const auto it = std::find(cells.begin(), cells.end(), corner);
  if (it == cells.end())
    throw std::logic_error("SimplicialQubit::flatTorus: the corner triangle is missing from the product");
  const bool reversed = epsilon[static_cast<std::size_t>(it - cells.begin())] < 0;

  return MarkedTorus{st, SimplicialQubit(std::move(cycleA), std::move(cycleB), reversed)};
}

Eigen::VectorXcd SimplicialQubit::stateOf(std::complex<double> tau) {
  const double norm = std::sqrt(1.0 + std::norm(tau));
  Eigen::VectorXcd psi(2);
  psi(0) = Complex(1.0, 0.0) / norm;
  psi(1) = tau / norm;
  return psi;
}

Eigen::VectorXd SimplicialQubit::blochOf(std::complex<double> tau) {
  const double N = 1.0 + std::norm(tau);
  Eigen::VectorXd r(3);
  r(0) = 2.0 * tau.real() / N;
  r(1) = 2.0 * tau.imag() / N;
  r(2) = (1.0 - std::norm(tau)) / N;
  return r;
}

Eigen::MatrixXcd SimplicialQubit::densityOf(std::complex<double> tau) {
  const Eigen::VectorXcd psi = stateOf(tau);
  return psi * psi.adjoint();
}

std::complex<double> SimplicialQubit::periodRatioOf(const Eigen::VectorXcd &state) {
  if (state.size() != 2)
    throw std::invalid_argument("SimplicialQubit::periodRatioOf: a qubit state has two amplitudes");
  if (state(0) == Complex(0.0, 0.0))
    throw std::invalid_argument(
        "SimplicialQubit::periodRatioOf: |1> is the cusp tau = infinity, not a finite torus");
  return state(1) / state(0);
}

double SimplicialQubit::fubiniStudyDistance(std::complex<double> tau1, std::complex<double> tau2) {
  const double overlap = std::abs(Complex(1.0, 0.0) + std::conj(tau1) * tau2) /
                         std::sqrt((1.0 + std::norm(tau1)) * (1.0 + std::norm(tau2)));
  return std::acos(std::clamp(overlap, 0.0, 1.0));
}

double SimplicialQubit::weilPeterssonDistance(std::complex<double> tau1, std::complex<double> tau2) {
  if (!(tau1.imag() > 0.0) || !(tau2.imag() > 0.0))
    throw std::invalid_argument(
        "SimplicialQubit::weilPeterssonDistance: both moduli must lie in the upper half plane");
  const double argument = 1.0 + std::norm(tau1 - tau2) / (2.0 * tau1.imag() * tau2.imag());
  return std::acosh(std::max(argument, 1.0));
}

}  // namespace tessera::observables
