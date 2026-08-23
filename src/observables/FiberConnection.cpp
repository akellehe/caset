// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/FiberConnection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "cobordism/AnalyticCache.h"
#include "cobordism/ChainComplex.h"
#include "mesh/Fingerprint.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "observables/ColorFiber.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

using cd = std::complex<double>;
using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateRegime;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 2.0 * kPi;

/// Spectral norm ||A||_2 of a small dense matrix (largest singular value).
double spectralNorm(const Eigen::MatrixXcd &a) {
  if (a.size() == 0) return 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(a);
  return svd.singularValues().size() > 0 ? svd.singularValues()[0] : 0.0;
}

/// The signature matrix J = diag(I_p, -I_q) of a band certificate.  The
/// #769 Krein-normalizable frames are stored with positives first
/// (Phi^dagger W Phi = J), so the diagonal order is structural.
Eigen::MatrixXcd signatureMatrix(int p, int q) {
  Eigen::MatrixXcd j = Eigen::MatrixXcd::Zero(p + q, p + q);
  for (int i = 0; i < p; ++i) j(i, i) = cd(1.0, 0.0);
  for (int i = 0; i < q; ++i) j(p + i, p + i) = cd(-1.0, 0.0);
  return j;
}

/// min(lowerGap, upperGap) with NaN propagation (an UNKNOWN side stays
/// unknown — it is never reported as infinitely isolated).
double isolationGap(const SpectralBandCertificate &c) {
  if (std::isnan(c.lowerGap) || std::isnan(c.upperGap)) return kNaN;
  return std::min(c.lowerGap, c.upperGap);
}

/// The paired transport regime of two endpoint bands: NonNormal dominates,
/// then HermitianIndefinite, else the shared positive regime.
CertificateRegime pairedRegime(CertificateRegime a, CertificateRegime b) {
  if (a == CertificateRegime::NonNormal || b == CertificateRegime::NonNormal)
    return CertificateRegime::NonNormal;
  if (a == CertificateRegime::HermitianIndefinite ||
      b == CertificateRegime::HermitianIndefinite)
    return CertificateRegime::HermitianIndefinite;
  return CertificateRegime::PositiveSemidefinite;
}

/// Principal phase step Arg(to / from) in (-pi, pi] between unit complexes.
double principalStep(cd fromUnit, cd toUnit) {
  return std::arg(toUnit / fromUnit);
}

/// The unit determinant of one accepted transport: det of the emitted
/// factor when present, the raw GL determinant phase otherwise.
cd unitDeterminant(const FiberTransportRead &link, bool *ok) {
  const Eigen::MatrixXcd &m =
      link.unitaryMap.size() > 0 ? link.unitaryMap : link.rawMap;
  if (m.rows() != m.cols() || m.size() == 0) {
    *ok = false;
    return cd(0.0, 0.0);
  }
  const cd det = m.determinant();
  if (!(std::abs(det) > 0.0)) {
    *ok = false;
    return cd(0.0, 0.0);
  }
  *ok = true;
  return det / std::abs(det);
}

/// The matrix a winding/holonomy consumer composes for one link.
const Eigen::MatrixXcd &linkMatrix(const FiberTransportRead &link) {
  return link.unitaryMap.size() > 0 ? link.unitaryMap : link.rawMap;
}

/// Principal square root of a small matrix through its eigendecomposition.
/// Defined only when no eigenvalue sits on the closed negative real axis;
/// `ok` reports that domain check.
Eigen::MatrixXcd principalSqrt(const Eigen::MatrixXcd &k, bool *ok) {
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(k);
  if (es.info() != Eigen::Success) {
    *ok = false;
    return Eigen::MatrixXcd();
  }
  const Eigen::VectorXcd &lambda = es.eigenvalues();
  for (Eigen::Index i = 0; i < lambda.size(); ++i) {
    const cd l = lambda[i];
    const double mag = std::abs(l);
    if (mag == 0.0 ||
        (l.real() <= 0.0 && std::abs(l.imag()) <= 1e-14 * std::max(1.0, mag))) {
      *ok = false;
      return Eigen::MatrixXcd();
    }
  }
  Eigen::VectorXcd roots(lambda.size());
  for (Eigen::Index i = 0; i < lambda.size(); ++i)
    roots[i] = std::sqrt(lambda[i]);  // principal branch
  const Eigen::MatrixXcd v = es.eigenvectors();
  *ok = true;
  return v * roots.asDiagonal() * v.inverse();
}

/// Order-SENSITIVE mix64 chain over the pieces of a cache parameter (the
/// component key itself is the order-independent part; the parameter must
/// distinguish direction and loop order, so it chains).
std::int64_t chainedParameter(int degree, int convention,
                              const std::vector<std::uint64_t> &keysInOrder) {
  std::uint64_t h = mesh::Fingerprint::mix64(
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(degree)) << 8) ^
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(convention)));
  for (const std::uint64_t k : keysInOrder)
    h = mesh::Fingerprint::mix64(h ^ k);
  return static_cast<std::int64_t>(h);
}

/// NaN-ignoring running max (std::fmax semantics) for certificate rollups.
double fmaxAccumulate(double acc, double value) { return std::fmax(acc, value); }

}  // namespace

// ---------------------------------------------------------------------------
// FiberTransportRead
// ---------------------------------------------------------------------------

std::string FiberTransportRead::describe() const {
  std::ostringstream out;
  out << "FiberTransport[deg " << degree << ", rank " << rank << ", regime ";
  switch (regime) {
    case CertificateRegime::PositiveSemidefinite: out << "positive"; break;
    case CertificateRegime::HermitianIndefinite: out << "krein"; break;
    case CertificateRegime::NonNormal: out << "non-normal"; break;
  }
  out << "] numerical rank " << numericalRank << ", leakage " << leakage
      << ", overlap cond " << overlapConditionNumber << ", gaps ("
      << toGap << ", " << fromGap << "), signatures (" << toPositiveSignature
      << "," << toNegativeSignature << ")/(" << fromPositiveSignature << ","
      << fromNegativeSignature << ")";
  if (accepted) {
    out << (unitaryMap.size() > 0 ? "; reduced (polar residual "
                                  : "; certified GL transport (polar residual ")
        << polarResidual << ")";
    if (projectiveOnly) out << " [projective-only]";
  } else {
    out << "; REJECTED: " << rejectionReason;
  }
  return out.str();
}

// ---------------------------------------------------------------------------
// construction / keys
// ---------------------------------------------------------------------------

FiberConnection::FiberConnection(FiberConnectionConfig cfg) : cfg_(cfg) {}

std::uint64_t FiberConnection::fiberKey(const SpectralFiber &fiber) {
  std::set<std::uint64_t> ids;
  for (const auto &cell : fiber.cellVertices())
    ids.insert(cell.begin(), cell.end());
  return mesh::Fingerprint::fingerprintOf(ids);
}

std::vector<std::uint64_t> FiberConnection::unionVertexIds(
    const std::vector<const SpectralFiber *> &fibers) {
  std::set<std::uint64_t> ids;
  for (const SpectralFiber *fiber : fibers)
    for (const auto &cell : fiber->cellVertices())
      ids.insert(cell.begin(), cell.end());
  return {ids.begin(), ids.end()};
}

// ---------------------------------------------------------------------------
// chain-transfer sources (wrappers over existing machinery)
// ---------------------------------------------------------------------------

Eigen::MatrixXcd FiberConnection::chainTransfer(
    const std::shared_ptr<Spacetime> &st, int degree,
    const std::vector<std::vector<std::uint64_t>> &toCells,
    const std::vector<std::vector<std::uint64_t>> &fromCells,
    cobordism::HodgeLaplacian::WeightConvention weights) {
  if (st == nullptr)
    throw std::invalid_argument("FiberConnection::chainTransfer: null spacetime");
  if (degree < 0)
    throw std::invalid_argument("FiberConnection::chainTransfer: negative degree");

  // Canonical whole-complex cell order: sorted vertex ids at degree 0 (the
  // HodgeLaplacian k = 0 convention), the ChainComplex column order at
  // degree >= 1 (the documented laplacian(k) alignment).
  std::vector<std::vector<std::uint64_t>> cells;
  if (degree == 0) {
    std::vector<std::uint64_t> ids;
    for (const auto &v : st->getVertexList()->toVector()) {
      if (v == nullptr) continue;
      ids.push_back(v->getId());
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    cells.reserve(ids.size());
    for (const std::uint64_t id : ids) cells.push_back({id});
  } else {
    const cobordism::ChainComplex cc =
        cobordism::ChainComplex::fromSpacetime(*st);
    cells = cc.kSimplexVertices(degree);
  }
  const std::size_t n = cells.size();
  if (n == 0)
    throw std::invalid_argument(
        "FiberConnection::chainTransfer: no cells at this degree");

  std::map<std::vector<std::uint64_t>, std::size_t> indexOf;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::uint64_t> key = cells[i];
    std::sort(key.begin(), key.end());
    indexOf.emplace(std::move(key), i);
  }
  const auto lookup =
      [&](const std::vector<std::uint64_t> &cell) -> std::size_t {
    std::vector<std::uint64_t> key = cell;
    std::sort(key.begin(), key.end());
    const auto it = indexOf.find(key);
    if (it == indexOf.end()) {
      std::ostringstream out;
      out << "FiberConnection::chainTransfer: unknown cell (";
      for (std::size_t i = 0; i < cell.size(); ++i)
        out << (i ? "," : "") << cell[i];
      out << ") at degree " << degree;
      throw std::invalid_argument(out.str());
    }
    return it->second;
  };

  const cobordism::HodgeLaplacian hodge(st, weights);
  const std::vector<cd> flat = hodge.laplacian(degree);
  if (flat.size() != n * n)
    throw std::invalid_argument(
        "FiberConnection::chainTransfer: operator/cell count mismatch");

  Eigen::MatrixXcd block(static_cast<Eigen::Index>(toCells.size()),
                         static_cast<Eigen::Index>(fromCells.size()));
  std::vector<std::size_t> rows;
  rows.reserve(toCells.size());
  for (const auto &cell : toCells) rows.push_back(lookup(cell));
  std::vector<std::size_t> cols;
  cols.reserve(fromCells.size());
  for (const auto &cell : fromCells) cols.push_back(lookup(cell));
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (std::size_t c = 0; c < cols.size(); ++c)
      block(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          flat[rows[r] * n + cols[c]];
  return block;
}

Eigen::MatrixXcd FiberConnection::responseTransfer(
    const cobordism::RecursiveQuotient::ResponseNetworkRead &network,
    int toComponent, int fromComponent) {
  const int count = static_cast<int>(network.stalkDimensions.size());
  if (toComponent < 0 || toComponent >= count || fromComponent < 0 ||
      fromComponent >= count)
    throw std::out_of_range(
        "FiberConnection::responseTransfer: component index out of range");
  const int rows = network.stalkDimensions[static_cast<std::size_t>(toComponent)];
  const int cols =
      network.stalkDimensions[static_cast<std::size_t>(fromComponent)];
  Eigen::MatrixXcd block = Eigen::MatrixXcd::Zero(rows, cols);
  for (const auto &edge : network.edges) {
    if (edge.from != toComponent || edge.to != fromComponent) continue;
    if (edge.block.size() != static_cast<std::size_t>(rows) *
                                 static_cast<std::size_t>(cols))
      throw std::invalid_argument(
          "FiberConnection::responseTransfer: malformed edge block");
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        block(r, c) = edge.block[static_cast<std::size_t>(r) *
                                     static_cast<std::size_t>(cols) +
                                 static_cast<std::size_t>(c)];
    return block;
  }
  return block;  // no such edge: the zero transfer of the right shape
}

// ---------------------------------------------------------------------------
// the derived transport
// ---------------------------------------------------------------------------

FiberTransportRead FiberConnection::transport(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  return deriveTransport(to, from, transfer);
}

FiberTransportRead FiberConnection::transportReverse(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  // W-adjoint reverse block T_BA = W_B^{-1} T_AB^dagger W_A — the exact
  // reverse chain transfer whenever W L is (anti)symmetric (the
  // W-self-adjoint regimes; see the header identity).
  const Eigen::VectorXcd wTo = to.weightDiagonal();
  const Eigen::VectorXcd wFrom = from.weightDiagonal();
  if (transfer.rows() != wTo.size() || transfer.cols() != wFrom.size())
    throw std::invalid_argument(
        "FiberConnection::transportReverse: transfer shape mismatch");
  Eigen::VectorXcd invWFrom(wFrom.size());
  for (Eigen::Index i = 0; i < wFrom.size(); ++i) {
    if (!(std::abs(wFrom[i]) > 0.0))
      throw std::invalid_argument(
          "FiberConnection::transportReverse: singular source weight");
    invWFrom[i] = cd(1.0, 0.0) / wFrom[i];
  }
  const Eigen::MatrixXcd reversed =
      invWFrom.asDiagonal() * transfer.adjoint() * wTo.asDiagonal();
  return deriveTransport(from, to, reversed);
}

FiberTransportRead FiberConnection::deriveTransport(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  const SpectralBandCertificate &certTo = to.certificate();
  const SpectralBandCertificate &certFrom = from.certificate();
  if (certTo.degree != certFrom.degree)
    throw std::invalid_argument(
        "FiberConnection::transport: the two bands live at different degrees");
  const auto rowsNeeded = static_cast<Eigen::Index>(to.cellVertices().size());
  const auto colsNeeded = static_cast<Eigen::Index>(from.cellVertices().size());
  if (transfer.rows() != rowsNeeded || transfer.cols() != colsNeeded)
    throw std::invalid_argument(
        "FiberConnection::transport: transfer shape mismatch (rows = "
        "destination cells, cols = source cells)");

  FiberTransportRead read;
  read.toKey = fiberKey(to);
  read.fromKey = fiberKey(from);
  read.degree = certTo.degree;
  read.rank = static_cast<int>(to.rank());
  read.toGap = isolationGap(certTo);
  read.fromGap = isolationGap(certFrom);
  read.toPositiveSignature = certTo.positiveSignature;
  read.toNegativeSignature = certTo.negativeSignature;
  read.fromPositiveSignature = certFrom.positiveSignature;
  read.fromNegativeSignature = certFrom.negativeSignature;
  read.toConditionNumber = certTo.conditionNumber;
  read.fromConditionNumber = certFrom.conditionNumber;
  read.frameConditionNumber =
      std::fmax(certTo.conditionNumber, certFrom.conditionNumber);
  read.regime = pairedRegime(certTo.certificate.regime(),
                             certFrom.certificate.regime());
  const bool nonNormal = read.regime == CertificateRegime::NonNormal;

  // Overlap: M = Phi_A^dagger W_A T Phi_B in the self-adjoint regimes,
  // M = Psi_A^dagger W_A T Phi_B on the biorthogonal path (spec 5.5).
  const Eigen::MatrixXcd leftFrame =
      nonNormal ? to.leftFrame() : to.rightFrame();
  const Eigen::MatrixXcd m = leftFrame.adjoint() *
                             to.weightDiagonal().asDiagonal() * transfer *
                             from.rightFrame();
  read.rawMap = m;

  // Pre-normalization diagnostics: rank, singular values, conditioning.
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(m, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();
  read.singularValues.assign(sigma.data(), sigma.data() + sigma.size());
  const double sigmaMax = sigma.size() > 0 ? sigma[0] : 0.0;
  const double sigmaMin = sigma.size() > 0 ? sigma[sigma.size() - 1] : 0.0;
  int numericalRank = 0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i)
    if (sigma[i] > cfg_.rankTolerance * sigmaMax) ++numericalRank;
  if (sigmaMax == 0.0) numericalRank = 0;
  read.numericalRank = numericalRank;
  read.overlapConditionNumber =
      sigma.size() == 0 ? 0.0 : (sigmaMin > 0.0 ? sigmaMax / sigmaMin : kInf);

  // Regime-appropriate leakage (spec 5.5).  In the self-adjoint regimes the
  // Krein form ||M^dagger J_A M - J_B|| is used, with J = I reproducing the
  // positive ||M^dagger M - I||; on the biorthogonal path the Euclidean
  // unitarity defect is REPORTED (the GL transport is not gated on it).
  const int rankTo = static_cast<int>(to.rank());
  const int rankFrom = static_cast<int>(from.rank());
  const bool ranksMatch = rankTo == rankFrom;
  const bool toSignatureComplete =
      certTo.positiveSignature + certTo.negativeSignature == rankTo;
  const bool fromSignatureComplete =
      certFrom.positiveSignature + certFrom.negativeSignature == rankFrom;
  Eigen::MatrixXcd jTo;
  Eigen::MatrixXcd jFrom;
  if (!nonNormal && toSignatureComplete && fromSignatureComplete) {
    jTo = signatureMatrix(certTo.positiveSignature, certTo.negativeSignature);
    jFrom =
        signatureMatrix(certFrom.positiveSignature, certFrom.negativeSignature);
    read.leakage = spectralNorm(m.adjoint() * jTo * m - jFrom);
  } else {
    read.leakage = spectralNorm(
        m.adjoint() * m -
        Eigen::MatrixXcd::Identity(m.cols(), m.cols()));
  }

  // ── threshold gates: reject BEFORE any polar/pseudo-unitary reduction ──
  const auto reject = [&](const std::string &reason) {
    read.accepted = false;
    read.rejectionReason = reason;
    read.certificate =
        Certificate::heuristicDiscovery(CertificateDomain::BandWindow,
                                        read.regime);
    if (read.numericalRank > 0 && m.rows() == m.cols() && m.size() > 0) {
      const cd det = m.determinant();
      if (std::abs(det) > 0.0) read.determinantPhase = det / std::abs(det);
    }
    return read;
  };

  if (!ranksMatch) return reject("band rank mismatch");
  if (cfg_.requireCertifiedFibers && !(certTo.accepted && certFrom.accepted))
    return reject("uncertified endpoint band (gap closed or residuals failed)");
  if (cfg_.minEndpointGap > 0.0 &&
      !(read.toGap >= cfg_.minEndpointGap &&
        read.fromGap >= cfg_.minEndpointGap))
    return reject("endpoint band gap below the configured floor");
  if (!(certTo.conditionNumber <= cfg_.conditionNumberCap) ||
      !(certFrom.conditionNumber <= cfg_.conditionNumberCap))
    return reject("endpoint frame conditioning above the cap");
  if (read.numericalRank < read.rank) return reject("rank-deficient overlap");
  if (!(read.overlapConditionNumber <= cfg_.conditionNumberCap))
    return reject("ill-conditioned overlap");

  if (nonNormal) {
    // Certified GL(r, C) transport: rawMap is the observable; no U(r) or
    // SU(3) value is emitted outside the positive-metric domain.
    read.accepted = true;
    const cd det = m.determinant();
    if (std::abs(det) > 0.0) read.determinantPhase = det / std::abs(det);
    const double gramResidual =
        fmaxAccumulate(certTo.gramDefect, certFrom.gramDefect);
    read.certificate = Certificate::certifiedNumerical(
        CertificateDomain::BandWindow, read.regime, gramResidual,
        read.overlapConditionNumber, cfg_.certificateTolerance);
    return read;
  }

  if (!toSignatureComplete || !fromSignatureComplete)
    return reject("neutral Krein directions (singular band Gram)");
  if (certTo.positiveSignature != certFrom.positiveSignature ||
      certTo.negativeSignature != certFrom.negativeSignature)
    return reject("Krein signature mismatch between the endpoint bands");
  if (!(read.leakage <= cfg_.leakageTolerance))
    return reject("leaking transport (isometry defect above tolerance)");

  const bool positivePair =
      read.regime == CertificateRegime::PositiveSemidefinite;
  if (positivePair) {
    // Polar factor V = M (M^dagger M)^{-1/2} = U V^dagger from the SVD —
    // exactly unitary-equivariant under local frame changes.
    read.unitaryMap = svd.matrixU() * svd.matrixV().adjoint();
    read.polarResidual = spectralNorm(
        read.unitaryMap.adjoint() * read.unitaryMap -
        Eigen::MatrixXcd::Identity(read.rank, read.rank));
  } else {
    // Pseudo-unitary reduction on MATCHING signatures:
    // V = M K^{-1/2}, K = J_B M^dagger J_A M (J_B-self-adjoint; principal
    // square root well defined for the near-J-isometric maps the leakage
    // gate admits), giving V^dagger J_A V = J_B.
    const Eigen::MatrixXcd k = jFrom * m.adjoint() * jTo * m;
    bool ok = false;
    const Eigen::MatrixXcd kSqrt = principalSqrt(k, &ok);
    if (!ok)
      return reject("pseudo-unitary square root undefined "
                    "(spectrum met the negative real axis)");
    read.unitaryMap = m * kSqrt.inverse();
    read.polarResidual =
        spectralNorm(read.unitaryMap.adjoint() * jTo * read.unitaryMap - jFrom);
  }

  const cd det = read.unitaryMap.determinant();
  read.determinantPhase = det;
  read.determinantResidual = std::abs(std::abs(det) - 1.0);
  read.projectiveOnly =
      read.rank == 3 && read.determinantResidual > cfg_.certificateTolerance;
  read.accepted = true;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, read.regime, read.polarResidual,
      read.overlapConditionNumber, cfg_.certificateTolerance);
  return read;
}

FiberTransportRead FiberConnection::transportOnSpacetime(
    const std::shared_ptr<Spacetime> &st, const SpectralFiber &to,
    const SpectralFiber &from,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (to.degree() != from.degree())
    throw std::invalid_argument(
        "FiberConnection::transportOnSpacetime: degree mismatch");
  const Eigen::MatrixXcd transfer = chainTransfer(
      st, to.degree(), to.cellVertices(), from.cellVertices(), weights);
  return deriveTransport(to, from, transfer);
}

FiberTransportRead FiberConnection::transportOnSpacetimeCached(
    cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
    const SpectralFiber &to, const SpectralFiber &from,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  const std::vector<std::uint64_t> ids = unionVertexIds({&to, &from});
  const std::int64_t parameter =
      chainedParameter(to.degree(), static_cast<int>(weights),
                       {fiberKey(to), fiberKey(from)});
  if (const auto payload = cache.fetch(ids, kTransportCacheKind, parameter))
    return *std::static_pointer_cast<FiberTransportRead>(payload);
  FiberTransportRead read = transportOnSpacetime(st, to, from, weights);
  cache.store(ids, kTransportCacheKind, parameter,
              std::make_shared<FiberTransportRead>(read), read.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// Wilson observables
// ---------------------------------------------------------------------------

WilsonHolonomyRead FiberConnection::holonomy(
    const std::vector<FiberTransportRead> &links) const {
  if (links.empty())
    throw std::invalid_argument("FiberConnection::holonomy: empty chain");
  const int rank = links.front().rank;
  bool unitary = true;
  for (std::size_t i = 0; i < links.size(); ++i) {
    if (!links[i].accepted)
      throw std::invalid_argument(
          "FiberConnection::holonomy: only ACCEPTED maps are multiplied "
          "(link " + std::to_string(i) + " was rejected: " +
          links[i].rejectionReason + ")");
    if (links[i].rank != rank)
      throw std::invalid_argument(
          "FiberConnection::holonomy: rank mismatch along the chain");
    if (links[i].unitaryMap.size() == 0) unitary = false;
  }

  WilsonHolonomyRead read;
  read.rank = rank;
  read.loopLength = links.size();
  read.baseKey = links.front().toKey;
  read.unitary = unitary;

  bool closed = true;
  for (std::size_t i = 0; i + 1 < links.size(); ++i)
    if (links[i].fromKey != links[i + 1].toKey) closed = false;
  if (links.back().fromKey != links.front().toKey) closed = false;
  read.closed = closed;

  Eigen::MatrixXcd h = Eigen::MatrixXcd::Identity(rank, rank);
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  double residual = kNaN;
  double conditioning = kNaN;
  for (const FiberTransportRead &link : links) {
    // Never a mixture: the unitary product uses every emitted factor, the
    // GL product the raw maps throughout.
    h = h * (unitary ? link.unitaryMap : link.rawMap);
    regime = pairedRegime(regime, link.regime);
    residual = fmaxAccumulate(residual, unitary ? link.polarResidual
                                                : link.certificate.residual());
    conditioning = fmaxAccumulate(conditioning, link.certificate.conditioning());
  }
  read.holonomy = h;
  read.normalizedTrace = h.trace() / static_cast<double>(rank);
  read.determinant = h.determinant();
  const double traceAbs = std::abs(h.trace());
  read.adjointTrace = cd(traceAbs * traceAbs - 1.0, 0.0);
  read.unitarityResidual = spectralNorm(
      h.adjoint() * h - Eigen::MatrixXcd::Identity(rank, rank));
  if (rank == 3) read.adjointMatrix = adjointRepresentation(h);
  if (unitary) residual = fmaxAccumulate(residual, read.unitarityResidual);
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime, residual, conditioning,
      cfg_.certificateTolerance);
  return read;
}

WilsonHolonomyRead FiberConnection::holonomyOnSpacetime(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<SpectralFiber> &fibers,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (fibers.size() < 2)
    throw std::invalid_argument(
        "FiberConnection::holonomyOnSpacetime: need at least two fibers");
  std::vector<FiberTransportRead> links;
  links.reserve(fibers.size());
  for (std::size_t i = 0; i < fibers.size(); ++i)
    links.push_back(transportOnSpacetime(
        st, fibers[i], fibers[(i + 1) % fibers.size()], weights));
  return holonomy(links);
}

WilsonHolonomyRead FiberConnection::holonomyOnSpacetimeCached(
    cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
    const std::vector<SpectralFiber> &fibers,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (fibers.size() < 2)
    throw std::invalid_argument(
        "FiberConnection::holonomyOnSpacetimeCached: need at least two fibers");
  std::vector<const SpectralFiber *> pointers;
  std::vector<std::uint64_t> orderedKeys;
  pointers.reserve(fibers.size());
  orderedKeys.reserve(fibers.size());
  for (const SpectralFiber &fiber : fibers) {
    pointers.push_back(&fiber);
    orderedKeys.push_back(fiberKey(fiber));
  }
  const std::vector<std::uint64_t> ids = unionVertexIds(pointers);
  const std::int64_t parameter = chainedParameter(
      fibers.front().degree(), static_cast<int>(weights), orderedKeys);
  if (const auto payload = cache.fetch(ids, kHolonomyCacheKind, parameter))
    return *std::static_pointer_cast<WilsonHolonomyRead>(payload);
  std::vector<FiberTransportRead> links;
  links.reserve(fibers.size());
  for (std::size_t i = 0; i < fibers.size(); ++i)
    links.push_back(transportOnSpacetimeCached(
        cache, st, fibers[i], fibers[(i + 1) % fibers.size()], weights));
  WilsonHolonomyRead read = holonomy(links);
  cache.store(ids, kHolonomyCacheKind, parameter,
              std::make_shared<WilsonHolonomyRead>(read), read.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// rank-three center structure
// ---------------------------------------------------------------------------

Eigen::MatrixXcd FiberConnection::projectiveRepresentative(
    const Eigen::MatrixXcd &unitary) {
  if (unitary.rows() != 3 || unitary.cols() != 3)
    throw std::invalid_argument(
        "FiberConnection::projectiveRepresentative: expected a 3x3 matrix");
  const cd det = unitary.determinant();
  if (!(std::abs(det) > 0.0))
    throw std::invalid_argument(
        "FiberConnection::projectiveRepresentative: singular matrix");
  // Principal cube root of the determinant PHASE (the modulus is left to
  // the caller's unitarity certificate — a representative, not a cleanup).
  const cd root = std::exp(cd(0.0, std::arg(det) / 3.0));
  return unitary / root;
}

Eigen::MatrixXcd FiberConnection::adjointRepresentation(
    const Eigen::MatrixXcd &unitary) {
  if (unitary.rows() != 3 || unitary.cols() != 3)
    throw std::invalid_argument(
        "FiberConnection::adjointRepresentation: expected a 3x3 matrix");
  // vec(U M U^dagger) = (conj(U) ⊗ U) vec(M), column-major vec index
  // i + 3j — the ColorFiber::adjointOctetProjector convention; the #767
  // projector restricts to the traceless octet (center-blind by
  // construction: Ad(zU) = Ad(U) for a central phase z).
  Eigen::MatrixXcd kron(9, 9);
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c)
      kron.block<3, 3>(3 * a, 3 * c) = std::conj(unitary(a, c)) * unitary;
  const Eigen::MatrixXcd p8 = ColorFiber::adjointOctetProjector();
  return p8 * kron * p8;
}

FundamentalLiftRead FiberConnection::fundamentalLift(
    const std::vector<FiberTransportRead> &links, int baseBranch) const {
  if (links.empty())
    throw std::invalid_argument("FiberConnection::fundamentalLift: empty path");
  if (baseBranch < 0 || baseBranch > 2)
    throw std::invalid_argument(
        "FiberConnection::fundamentalLift: base branch must be 0, 1, or 2");

  FundamentalLiftRead read;
  read.rank = links.front().rank;
  read.baseBranch = baseBranch;

  const auto invalid = [&](const std::string &reason) {
    read.valid = false;
    read.invalidReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::PositiveSemidefinite);
    return read;
  };

  if (read.rank != 3)
    return invalid("fundamental SU(3) lift requested at rank " +
                   std::to_string(read.rank) +
                   " (never hard-coded at generic rank)");
  Eigen::MatrixXcd h = Eigen::MatrixXcd::Identity(3, 3);
  double theta = 0.0;
  double maxStep = 0.0;
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (std::size_t i = 0; i < links.size(); ++i) {
    const FiberTransportRead &link = links[i];
    if (!link.accepted)
      return invalid("link " + std::to_string(i) + " rejected: " +
                     link.rejectionReason);
    if (link.rank != 3)
      return invalid("rank mismatch along the path");
    if (link.unitaryMap.size() == 0)
      return invalid("link " + std::to_string(i) +
                     " carries only a GL transport (no unitary factor)");
    const cd det = link.unitaryMap.determinant();
    if (!(std::abs(det) > 0.0))
      return invalid("vanishing link determinant");
    const double step = std::arg(det);  // principal, (-pi, pi]
    theta += step;
    maxStep = std::max(maxStep, std::abs(step));
    h = h * link.unitaryMap;
    regime = pairedRegime(regime, link.regime);
  }

  // Continued branch: lift = H e^{-i Theta / 3} omega^{-s0} with omega the
  // ALGEBRAIC #767 cube root (so the SU(3) determinant cancels exactly).
  const cd omega = ColorFiber::omega();
  const std::array<cd, 3> centerPower{cd(1.0, 0.0), omega, omega * omega};
  const cd omegaInverseS0 = centerPower[static_cast<std::size_t>(
      (3 - (baseBranch % 3)) % 3)];
  read.lift = h * std::exp(cd(0.0, -theta / 3.0)) * omegaInverseS0;
  read.liftTrace = read.lift.trace();
  read.accumulatedDeterminantPhase = theta;
  read.maxDeterminantPhaseStep = maxStep;

  // Accumulated center sector: the sheet count of Theta relative to its
  // principal value, mod 3 — branch-independent by construction.
  const double principal = std::arg(std::exp(cd(0.0, theta)));
  const long sheets = std::lround((theta - principal) / kTwoPi);
  read.centerSector = static_cast<int>(((sheets % 3) + 3) % 3);

  const cd liftDet = read.lift.determinant();
  const double unitarityResidual = spectralNorm(
      read.lift.adjoint() * read.lift - Eigen::MatrixXcd::Identity(3, 3));
  read.detResidual = std::abs(liftDet - cd(1.0, 0.0));
  read.valid = true;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime,
      std::max(read.detResidual, unitarityResidual), 1.0,
      cfg_.certificateTolerance);
  return read;
}

// ---------------------------------------------------------------------------
// determinant winding
// ---------------------------------------------------------------------------

DeterminantWindingRead FiberConnection::windingRead(
    const std::vector<FiberTransportRead> &family, bool cyclic,
    const WindingClosureSpec *closure) const {
  if (family.empty())
    throw std::invalid_argument("FiberConnection: empty transport family");

  DeterminantWindingRead read;
  if (cyclic) {
    read.windingClosure = "closed-family";
  } else {
    switch (closure->mode) {
      case WindingClosureSpec::Mode::None:
        read.windingClosure = "none";
        break;
      case WindingClosureSpec::Mode::MatchedReference:
        read.windingClosure = "matched-reference";
        break;
      case WindingClosureSpec::Mode::EndpointTrivialization:
        read.windingClosure = "endpoint-trivialization";
        break;
    }
    read.windingReferenceId = closure->referenceId;
  }
  const auto invalidate = [&](const std::string &reason) {
    read.winding.reset();
    read.invalidationReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::PositiveSemidefinite);
    return read;
  };

  // Family gates: an integer winding exists only for a continuous,
  // full-rank, gapped family of ACCEPTED transports of one rank.
  const int rank = family.front().rank;
  std::vector<cd> units;
  units.reserve(family.size());
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (std::size_t i = 0; i < family.size(); ++i) {
    const FiberTransportRead &sample = family[i];
    if (!sample.accepted)
      return invalidate("sample " + std::to_string(i) +
                        " is not an accepted transport (gap or rank closed): " +
                        sample.rejectionReason);
    if (sample.rank != rank)
      return invalidate("rank changed along the family");
    bool ok = false;
    const cd unit = unitDeterminant(sample, &ok);
    if (!ok) return invalidate("vanishing determinant along the family");
    units.push_back(unit);
    regime = pairedRegime(regime, sample.regime);
  }

  // The determinant phase path: principal legs of the declared composite.
  std::vector<double> legs;
  double closureDefect = 0.0;
  if (cyclic) {
    for (std::size_t k = 0; k + 1 < units.size(); ++k)
      legs.push_back(principalStep(units[k], units[k + 1]));
    legs.push_back(principalStep(units.back(), units.front()));
  } else {
    const WindingClosureSpec &spec = *closure;
    switch (spec.mode) {
      case WindingClosureSpec::Mode::None: {
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        double theta = 0.0;
        double maxStep = 0.0;
        for (const double leg : legs) {
          theta += leg;
          maxStep = std::max(maxStep, std::abs(leg));
        }
        read.accumulatedPhase = theta;  // the raw open-path phase —
        read.maxPhaseStep = maxStep;    // deliberately NOT an integer claim
        read.phaseStepMargin = maxStep / kPi;
        return invalidate(
            "no closure declared (a raw endpoint phase difference is not a "
            "winding certificate)");
      }
      case WindingClosureSpec::Mode::MatchedReference: {
        if (spec.referenceTransports.size() != family.size())
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: the matched reference "
              "must supply one transport per segment sample");
        std::vector<cd> refUnits;
        refUnits.reserve(spec.referenceTransports.size());
        for (const Eigen::MatrixXcd &r : spec.referenceTransports) {
          if (r.rows() != rank || r.cols() != rank)
            throw std::invalid_argument(
                "FiberConnection::openSegmentWinding: reference transport "
                "shape mismatch");
          const cd det = r.determinant();
          if (!(std::abs(det) > 0.0))
            throw std::invalid_argument(
                "FiberConnection::openSegmentWinding: singular reference "
                "transport");
          refUnits.push_back(det / std::abs(det));
        }
        // Segment forward, cross to the reference, reference backward,
        // cross back: a closed cycle of principal legs.
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        legs.push_back(principalStep(units.back(), refUnits.back()));
        for (std::size_t k = refUnits.size(); k-- > 1;)
          legs.push_back(principalStep(refUnits[k], refUnits[k - 1]));
        legs.push_back(principalStep(refUnits.front(), units.front()));
        const auto relativeMismatch = [](const Eigen::MatrixXcd &a,
                                         const Eigen::MatrixXcd &b) {
          const double scale = std::max(1.0, spectralNorm(b));
          return spectralNorm(a - b) / scale;
        };
        closureDefect = std::max(
            relativeMismatch(spec.referenceTransports.front(),
                             linkMatrix(family.front())),
            relativeMismatch(spec.referenceTransports.back(),
                             linkMatrix(family.back())));
        break;
      }
      case WindingClosureSpec::Mode::EndpointTrivialization: {
        if (spec.startTrivialization.rows() != rank ||
            spec.startTrivialization.cols() != rank ||
            spec.endTrivialization.rows() != rank ||
            spec.endTrivialization.cols() != rank)
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: trivialization shape "
              "mismatch");
        const cd det0 = spec.startTrivialization.determinant();
        const cd det1 = spec.endTrivialization.determinant();
        if (!(std::abs(det0) > 0.0) || !(std::abs(det1) > 0.0))
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: singular trivialization");
        const cd tau0 = det0 / std::abs(det0);
        const cd tau1 = det1 / std::abs(det1);
        // Four principal legs close the determinant path exactly:
        // tau0 -> V(0) -> ... -> V(n-1) -> tau1 -> tau0.
        legs.push_back(principalStep(tau0, units.front()));
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        legs.push_back(principalStep(units.back(), tau1));
        legs.push_back(principalStep(tau1, tau0));
        const auto unitarityDefect = [rank](const Eigen::MatrixXcd &t) {
          return spectralNorm(t.adjoint() * t -
                              Eigen::MatrixXcd::Identity(rank, rank));
        };
        closureDefect = std::max(unitarityDefect(spec.startTrivialization),
                                 unitarityDefect(spec.endTrivialization));
        break;
      }
    }
  }

  double theta = 0.0;
  double maxStep = 0.0;
  for (const double leg : legs) {
    theta += leg;
    maxStep = std::max(maxStep, std::abs(leg));
  }
  read.accumulatedPhase = theta;
  read.maxPhaseStep = maxStep;
  read.phaseStepMargin = maxStep / kPi;
  read.closureDefect = closureDefect;
  if (maxStep >= kPi * (1.0 - 1e-12))
    return invalidate("phase step aliasing (a leg reached pi)");

  const long nu = std::lround(theta / kTwoPi);
  read.winding = static_cast<int>(nu);
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime, closureDefect,
      read.phaseStepMargin < 1.0 ? 1.0 / (1.0 - read.phaseStepMargin) : kInf,
      cfg_.closureTolerance);
  return read;
}

DeterminantWindingRead FiberConnection::closedFamilyWinding(
    const std::vector<FiberTransportRead> &family) const {
  return windingRead(family, /*cyclic=*/true, nullptr);
}

DeterminantWindingRead FiberConnection::openSegmentWinding(
    const std::vector<FiberTransportRead> &segment,
    const WindingClosureSpec &closure) const {
  return windingRead(segment, /*cyclic=*/false, &closure);
}

}  // namespace tessera::observables
