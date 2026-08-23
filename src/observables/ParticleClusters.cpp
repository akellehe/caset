// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ParticleClusters.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "cobordism/AnalyticCache.h"
#include "cobordism/EigenstateSynthesis.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateGrade;
using cobordism::CertificateRegime;
using cd = std::complex<double>;

namespace {

constexpr std::int64_t kRecordSchemaVersion = 1;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// certificate <-> record helpers (the SpectralFiber/FiberConnection file-local
// convention).
// ---------------------------------------------------------------------------

std::string regimeName(CertificateRegime regime) {
  switch (regime) {
    case CertificateRegime::PositiveSemidefinite:
      return "positive-semidefinite";
    case CertificateRegime::HermitianIndefinite:
      return "hermitian-indefinite";
    case CertificateRegime::NonNormal:
      return "non-normal";
  }
  return "non-normal";
}

CertificateRegime regimeFromName(const std::string &name) {
  if (name == "positive-semidefinite")
    return CertificateRegime::PositiveSemidefinite;
  if (name == "hermitian-indefinite")
    return CertificateRegime::HermitianIndefinite;
  if (name == "non-normal") return CertificateRegime::NonNormal;
  throw std::invalid_argument("ParticleClusters: unknown regime '" + name +
                              "'");
}

std::string gradeName(CertificateGrade grade) {
  switch (grade) {
    case CertificateGrade::AlgebraicallyExact:
      return "algebraically-exact";
    case CertificateGrade::StructureExact:
      return "structure-exact";
    case CertificateGrade::CertifiedNumerical:
      return "certified-numerical";
    case CertificateGrade::HeuristicDiscovery:
      return "heuristic-discovery";
  }
  return "heuristic-discovery";
}

std::string domainName(CertificateDomain domain) {
  return domain == CertificateDomain::Static ? "static" : "band-window";
}

CertificateDomain domainFromName(const std::string &name) {
  if (name == "static") return CertificateDomain::Static;
  if (name == "band-window") return CertificateDomain::BandWindow;
  throw std::invalid_argument("ParticleClusters: unknown domain '" + name +
                              "'");
}

Record certificateToRecord(const Certificate &cert) {
  Record::Map m;
  m["grade"] = Record(gradeName(cert.grade()));
  m["domain"] = Record(domainName(cert.domain()));
  m["regime"] = Record(regimeName(cert.regime()));
  m["residual"] = Record(cert.residual());
  m["conditioning"] = Record(cert.conditioning());
  m["dense_reference_error"] = Record(cert.denseReferenceError());
  m["tolerance"] = Record(cert.tolerance());
  return Record(std::move(m));
}

Certificate certificateFromRecord(const Record &record) {
  const auto &m = record.asMap();
  const std::string grade = m.at("grade").asString();
  const CertificateDomain domain = domainFromName(m.at("domain").asString());
  const CertificateRegime regime = regimeFromName(m.at("regime").asString());
  const double residual = m.at("residual").asDouble();
  const double conditioning = m.at("conditioning").asDouble();
  const double tolerance = m.at("tolerance").asDouble();
  Certificate cert;
  if (grade == "algebraically-exact") {
    cert = Certificate::algebraicallyExact(domain, regime, residual, tolerance);
  } else if (grade == "structure-exact") {
    cert = Certificate::structureExact(domain, regime, residual, conditioning,
                                       tolerance);
  } else if (grade == "certified-numerical") {
    cert = Certificate::certifiedNumerical(domain, regime, residual,
                                           conditioning, tolerance);
  } else if (grade == "heuristic-discovery") {
    cert = Certificate::heuristicDiscovery(domain, regime);
  } else {
    throw std::invalid_argument(
        "ParticleClusters: unknown certificate grade '" + grade + "'");
  }
  cert.setDenseReferenceError(m.at("dense_reference_error").asDouble());
  return cert;
}

void requireSchema(const Record::Map &m, const char *type) {
  const auto version = m.find("schema_version");
  if (version == m.end() ||
      version->second.asInt() != kRecordSchemaVersion)
    throw std::invalid_argument(
        "ParticleClusters: unknown schema_version (reader rejects unknown "
        "checkpoint schemas)");
  const auto rt = m.find("record_type");
  if (rt == m.end() || rt->second.asString() != type)
    throw std::invalid_argument(std::string("ParticleClusters: expected a '") +
                                type + "' record");
}

Record optionalDouble(const std::optional<double> &value) {
  return value.has_value() ? Record(*value) : Record();
}

Record optionalInt(const std::optional<int> &value) {
  return value.has_value() ? Record(*value) : Record();
}

std::optional<double> optionalDoubleFrom(const Record &r) {
  if (r.isNull()) return std::nullopt;
  return r.asDouble();
}

std::optional<int> optionalIntFrom(const Record &r) {
  if (r.isNull()) return std::nullopt;
  return static_cast<int>(r.asInt());
}

// ---------------------------------------------------------------------------
// fingerprint helpers (the CovarianceState hashing idiom)
// ---------------------------------------------------------------------------

std::uint64_t chainHash(std::uint64_t seed, std::uint64_t value) {
  return mesh::Fingerprint::mix64(seed ^ value);
}

std::uint64_t hashDouble(std::uint64_t seed, double value) {
  return chainHash(seed, std::bit_cast<std::uint64_t>(value));
}

std::uint64_t hashComplex(std::uint64_t seed, cd value) {
  return hashDouble(hashDouble(seed, value.real()), value.imag());
}

std::uint64_t hashString(std::uint64_t seed, const std::string &s) {
  std::uint64_t h = chainHash(seed, s.size());
  for (const char c : s)
    h = chainHash(h, static_cast<std::uint64_t>(
                         static_cast<unsigned char>(c)));
  return h;
}

std::uint64_t hashCertificateHolds(std::uint64_t seed,
                                   const Certificate &cert) {
  std::uint64_t h = chainHash(seed, cert.holds() ? 1u : 0u);
  h = hashDouble(h, cert.residual());
  return hashDouble(h, cert.tolerance());
}

/// The deduplicated sorted cell-vertex-id set of a fiber (the
/// `AnalyticCache::componentKey` material — the FiberConnection convention).
std::vector<std::uint64_t> fiberVertexIds(const SpectralFiber &fiber) {
  std::set<std::uint64_t> ids;
  for (const auto &cell : fiber.cellVertices())
    ids.insert(cell.begin(), cell.end());
  return {ids.begin(), ids.end()};
}

/// max of `current` and `candidate` ignoring NaN candidates.
double maxFinite(double current, double candidate) {
  if (std::isnan(candidate)) return current;
  if (std::isnan(current)) return candidate;
  return std::max(current, candidate);
}

}  // namespace

// ---------------------------------------------------------------------------
// QuarkRead
// ---------------------------------------------------------------------------

std::string QuarkRead::describe() const {
  std::ostringstream out;
  out << "QuarkRead[" << classification;
  if (determinantWinding.has_value())
    out << ", nu=" << *determinantWinding << " (" << windingClosure << ")";
  else
    out << ", nu=unknown";
  if (baryonFlux.has_value())
    out << ", B=" << *baryonFlux;
  else
    out << ", B=unknown";
  out << ", parity=" << exteriorParity << ", color rank " << colorRank
      << ", confidence " << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

namespace {

Record thresholdsToRecord(const ParticleClustersConfig &cfg) {
  Record::Map m;
  m["parity_tolerance"] = Record(cfg.parityTolerance);
  m["occupation_tolerance"] = Record(cfg.occupationTolerance);
  m["min_anchor_score"] = Record(cfg.minAnchorScore);
  m["min_phase_coherence"] = Record(cfg.minPhaseCoherence);
  m["max_transport_leakage"] = Record(cfg.maxTransportLeakage);
  m["min_persistence_lifetime"] = Record(cfg.minPersistenceLifetime);
  m["min_persistence_overlap"] = Record(cfg.minPersistenceOverlap);
  m["min_localization"] = Record(cfg.minLocalization);
  m["min_refinement_overlap"] = Record(cfg.minRefinementOverlap);
  m["doublet_overlap_threshold"] = Record(cfg.doubletOverlapThreshold);
  m["min_doublet_frames"] =
      Record(static_cast<std::int64_t>(cfg.minDoubletFrames));
  m["isospin_tolerance"] = Record(cfg.isospinTolerance);
  m["gauss_tolerance"] = Record(cfg.gaussTolerance);
  m["min_enclosing_surfaces"] =
      Record(static_cast<std::int64_t>(cfg.minEnclosingSurfaces));
  m["ud_tolerance"] = Record(cfg.udTolerance);
  return Record(std::move(m));
}

ParticleClustersConfig thresholdsFromRecord(const Record &record) {
  const auto &m = record.asMap();
  ParticleClustersConfig cfg;
  cfg.parityTolerance = m.at("parity_tolerance").asDouble();
  cfg.occupationTolerance = m.at("occupation_tolerance").asDouble();
  cfg.minAnchorScore = m.at("min_anchor_score").asDouble();
  cfg.minPhaseCoherence = m.at("min_phase_coherence").asDouble();
  cfg.maxTransportLeakage = m.at("max_transport_leakage").asDouble();
  cfg.minPersistenceLifetime = m.at("min_persistence_lifetime").asDouble();
  cfg.minPersistenceOverlap = m.at("min_persistence_overlap").asDouble();
  cfg.minLocalization = m.at("min_localization").asDouble();
  cfg.minRefinementOverlap = m.at("min_refinement_overlap").asDouble();
  cfg.doubletOverlapThreshold = m.at("doublet_overlap_threshold").asDouble();
  cfg.minDoubletFrames =
      static_cast<std::size_t>(m.at("min_doublet_frames").asInt());
  cfg.isospinTolerance = m.at("isospin_tolerance").asDouble();
  cfg.gaussTolerance = m.at("gauss_tolerance").asDouble();
  cfg.minEnclosingSurfaces =
      static_cast<std::size_t>(m.at("min_enclosing_surfaces").asInt());
  cfg.udTolerance = m.at("ud_tolerance").asDouble();
  return cfg;
}

}  // namespace

Record QuarkRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("quark_read");
  m["component_hash"] = Record(component.canonicalHash());
  m["component_level"] =
      Record(static_cast<std::int64_t>(component.level()));
  m["exterior_parity"] = Record(exteriorParity);
  m["color_rank"] = Record(colorRank);
  m["triangle_anchor_score"] = Record(triangleAnchorScore);
  m["triangle_anchor_max_term"] = Record(triangleAnchorMaxTerm);
  m["triangle_anchor_participation"] = Record(triangleAnchorParticipation);
  m["anchor_phase_dispersion"] = Record(anchorPhaseDispersion);
  m["anchor_phase_coherence"] = Record(anchorPhaseCoherence);
  m["anchor_weighting_id"] = Record(anchorWeightingId);
  m["determinant_winding"] = optionalInt(determinantWinding);
  m["winding_closure"] = Record(windingClosure);
  m["winding_reference_id"] = Record(windingReferenceId);
  m["baryon_flux"] = optionalDouble(baryonFlux);
  m["isospin"] = optionalDouble(isospin);
  m["electric_flux"] = optionalDouble(electricFlux);
  m["confidence"] = Record(confidence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["classification"] = Record(classification);
  m["occupation_total"] = Record(occupationTotal);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["persistence_lifetime"] = Record(persistenceLifetime);
  m["persistence_min_overlap"] = Record(persistenceMinOverlap);
  m["localization"] = Record(localization);
  m["refinement_overlap"] = Record(refinementOverlap);
  m["ud_identification_proposed"] = Record(udIdentificationProposed);
  m["doublet_orientation"] = Record(doubletOrientation);
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

QuarkRead QuarkRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "quark_read");
  QuarkRead read;
  read.component = ComponentId(
      m.at("component_hash").asString(),
      static_cast<std::size_t>(m.at("component_level").asInt()));
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.colorRank = static_cast<int>(m.at("color_rank").asInt());
  read.triangleAnchorScore = m.at("triangle_anchor_score").asDouble();
  read.triangleAnchorMaxTerm = m.at("triangle_anchor_max_term").asDouble();
  read.triangleAnchorParticipation =
      m.at("triangle_anchor_participation").asDouble();
  read.anchorPhaseDispersion = m.at("anchor_phase_dispersion").asDouble();
  read.anchorPhaseCoherence = m.at("anchor_phase_coherence").asDouble();
  read.anchorWeightingId = m.at("anchor_weighting_id").asString();
  read.determinantWinding = optionalIntFrom(m.at("determinant_winding"));
  read.windingClosure = m.at("winding_closure").asString();
  read.windingReferenceId = m.at("winding_reference_id").asString();
  read.baryonFlux = optionalDoubleFrom(m.at("baryon_flux"));
  read.isospin = optionalDoubleFrom(m.at("isospin"));
  read.electricFlux = optionalDoubleFrom(m.at("electric_flux"));
  read.confidence = m.at("confidence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.classification = m.at("classification").asString();
  read.occupationTotal = m.at("occupation_total").asDouble();
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.persistenceLifetime = m.at("persistence_lifetime").asDouble();
  read.persistenceMinOverlap = m.at("persistence_min_overlap").asDouble();
  read.localization = m.at("localization").asDouble();
  read.refinementOverlap = m.at("refinement_overlap").asDouble();
  read.udIdentificationProposed =
      m.at("ud_identification_proposed").asBool();
  read.doubletOrientation =
      static_cast<int>(m.at("doublet_orientation").asInt());
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

// ---------------------------------------------------------------------------
// ParticleClusters
// ---------------------------------------------------------------------------

ParticleClusters::ParticleClusters(ParticleClustersConfig cfg) : cfg_(cfg) {}

bool ParticleClusters::gate(bool passed, const char *name,
                            std::vector<std::string> &failed) {
  if (!passed) failed.emplace_back(name);
  return passed;
}

QuarkRead ParticleClusters::classifyQuark(
    const QuarkCandidateEvidence &evidence) const {
  QuarkRead read;
  read.component = evidence.component;
  read.thresholds = cfg_;

  std::vector<std::string> failed;
  int passedCore = 0;
  constexpr int kCoreCertificates = 10;

  // 1. persistence (#765 track diagnostics; NaN = missing evidence).
  read.persistenceLifetime = evidence.persistenceLifetime;
  read.persistenceMinOverlap = evidence.persistenceMinOverlap;
  const bool persistenceOk =
      std::isfinite(evidence.persistenceLifetime) &&
      evidence.persistenceLifetime >= cfg_.minPersistenceLifetime &&
      std::isfinite(evidence.persistenceMinOverlap) &&
      evidence.persistenceMinOverlap >= cfg_.minPersistenceOverlap;
  passedCore += gate(persistenceOk, "persistence", failed);

  // 2. localization (from the color band's own certificate).
  const SpectralBandCertificate &band = evidence.colorBand.certificate();
  read.localization = band.localization;
  const bool localizationOk = std::isfinite(band.localization) &&
                              band.localization >= cfg_.minLocalization;
  passedCore += gate(localizationOk, "localization", failed);

  // 3. odd exterior parity (#780 Wick parity; an uncertified read never
  //    emits a sign — the #772 characterSign convention).
  const auto &parity = evidence.parityRead;
  const bool parityCertified = parity.certificate.holds();
  if (parityCertified &&
      std::abs(parity.value - cd(-1.0, 0.0)) <= cfg_.parityTolerance) {
    read.exteriorParity = -1;
  } else if (parityCertified &&
             std::abs(parity.value - cd(1.0, 0.0)) <= cfg_.parityTolerance) {
    read.exteriorParity = +1;
  } else {
    read.exteriorParity = 0;
  }
  passedCore += gate(read.exteriorParity == -1, "parity-odd", failed);

  // 4. single-fermion occupation (#780 Wick total number) — the
  //    total-occupation channel excluding the two-quark anti-triplet.
  const auto &occupation = evidence.occupationRead;
  const bool occupationCertified = occupation.certificate.holds();
  read.occupationTotal =
      occupationCertified ? occupation.value.real() : kNaN;
  const bool occupationOneOk =
      occupationCertified &&
      std::abs(occupation.value - cd(1.0, 0.0)) <= cfg_.occupationTolerance;
  passedCore += gate(occupationOneOk, "occupation-one", failed);

  // 5. accepted rank-three color band (#769 — rank is read, never
  //    requested from the detector).
  read.colorRank = static_cast<int>(evidence.colorBand.rank());
  const bool rankThreeOk =
      evidence.colorBand.accepted() && evidence.colorBand.rank() == 3;
  passedCore += gate(rankThreeOk, "color-rank-three", failed);

  // 6. calibrated oriented-triangle anchor (#767).  A default-constructed
  //    profile (no weighting declared) is MISSING evidence: the anchor
  //    fields stay NaN/unknown.
  const AnchorProfile &anchor = evidence.anchor;
  const bool anchorSupplied = !anchor.weightingId.empty();
  if (anchorSupplied) {
    read.triangleAnchorScore = anchor.score;
    read.triangleAnchorMaxTerm = anchor.maxTerm;
    read.triangleAnchorParticipation = anchor.participationRatio;
    read.anchorPhaseDispersion = anchor.phaseDispersion;
    read.anchorPhaseCoherence = anchor.phaseCoherence;
    read.anchorWeightingId = anchor.weightingId;
  }
  const bool anchorOk = anchorSupplied && anchor.certificate.holds() &&
                        anchor.score >= cfg_.minAnchorScore &&
                        anchor.phaseCoherence >= cfg_.minPhaseCoherence;
  passedCore += gate(anchorOk, "anchor", failed);

  // 7. bounded transport leakage over the lifetime (#770).
  read.transportCount = evidence.lifetimeTransports.size();
  bool allTransportsAccepted = !evidence.lifetimeTransports.empty();
  double maxLeakage = kNaN;
  for (const FiberTransportRead &transport : evidence.lifetimeTransports) {
    allTransportsAccepted = allTransportsAccepted && transport.accepted;
    maxLeakage = maxFinite(maxLeakage, transport.leakage);
  }
  read.transportLeakageMax = maxLeakage;
  const bool leakageOk = allTransportsAccepted &&
                         std::isfinite(maxLeakage) &&
                         maxLeakage <= cfg_.maxTransportLeakage;
  passedCore += gate(leakageOk, "transport-leakage", failed);

  // 8/9. certified determinant-line winding and unit magnitude (#770).
  //      The closure SPECIFICATION travels with the read; B = nu/3 exists
  //      exactly when the winding certificate does (a certified nu = 0 is
  //      a certified zero flux), and quark-ness additionally needs
  //      |nu| = 1.
  const DeterminantWindingRead &winding = evidence.winding;
  read.windingClosure = winding.windingClosure;
  read.windingReferenceId = winding.windingReferenceId;
  const bool windingOk =
      winding.winding.has_value() && winding.certificate.holds();
  if (windingOk) {
    read.determinantWinding = winding.winding;
    read.baryonFlux = static_cast<double>(*winding.winding) / 3.0;
  }
  passedCore += gate(windingOk, "winding", failed);
  const bool windingUnitOk =
      windingOk && std::abs(*winding.winding) == 1;
  passedCore += gate(windingUnitOk, "winding-unit", failed);

  // 10. refinement stability (band subspace overlap across a refinement).
  read.refinementOverlap = evidence.refinementOverlap;
  const bool refinementOk =
      std::isfinite(evidence.refinementOverlap) &&
      evidence.refinementOverlap >= cfg_.minRefinementOverlap;
  passedCore += gate(refinementOk, "refinement-stability", failed);

  read.confidence =
      static_cast<double>(passedCore) / static_cast<double>(kCoreCertificates);

  // Verdict: quark vs antiquark is the determinant-line ORIENTATION of the
  // certified winding — never the color representation alone.
  if (passedCore == kCoreCertificates) {
    read.classification = (*winding.winding == +1) ? "quark" : "antiquark";
  } else {
    read.classification = "none";
  }

  // Flavor: only an emergent certified two-state subclass reports isospin.
  const bool flavorOk = evidence.flavor.has_value() &&
                        evidence.flavor->found &&
                        evidence.flavor->certificate.holds();
  if (!flavorOk) {
    failed.emplace_back("flavor-doublet");
  } else {
    bool isospinDefinite = false;
    if (evidence.doubletOccupancy.has_value() &&
        (evidence.doubletOrientation == +1 ||
         evidence.doubletOrientation == -1)) {
      const Eigen::Vector2cd &f = *evidence.doubletOccupancy;
      const double norm2 = std::norm(f(0)) + std::norm(f(1));
      if (norm2 > 0.0) {
        const double i3Raw =
            (std::norm(f(0)) - std::norm(f(1))) / (2.0 * norm2);
        if (std::abs(i3Raw - 0.5) <= cfg_.isospinTolerance) {
          read.isospin = 0.5 * evidence.doubletOrientation;
          isospinDefinite = true;
        } else if (std::abs(i3Raw + 0.5) <= cfg_.isospinTolerance) {
          read.isospin = -0.5 * evidence.doubletOrientation;
          isospinDefinite = true;
        }
      }
    }
    if (isospinDefinite) {
      read.doubletOrientation = evidence.doubletOrientation;
    } else {
      failed.emplace_back("isospin");
    }
  }

  // Charge: the reused Gauss read must be consistent across nested
  // enclosing surfaces AND the doublet must be certified (#773 acceptance:
  // a missing/unstable flavor doublet yields unknown flavor AND charge).
  const bool gaussOk = evidence.charge.has_value() &&
                       evidence.charge->consistent &&
                       evidence.charge->certificate.holds() &&
                       evidence.charge->electricFlux.has_value();
  if (!gaussOk) failed.emplace_back("gauss-consistency");
  if (gaussOk && flavorOk) read.electricFlux = evidence.charge->electricFlux;

  // The proposed u/d identification: Q = I3 + B/2 is TESTED only when
  // baryon flux, isospin, and the Gauss-consistent charge all exist.
  if (read.baryonFlux.has_value() && read.isospin.has_value() &&
      read.electricFlux.has_value()) {
    const double predicted = *read.isospin + *read.baryonFlux / 2.0;
    if (std::abs(*read.electricFlux - predicted) <= cfg_.udTolerance) {
      read.udIdentificationProposed = true;
    } else {
      failed.emplace_back("ud-identification");
    }
  }

  read.failedCertificates = std::move(failed);

  // The graded claim: an accepted verdict is an exact boolean combination
  // GIVEN the consumed held certificates (StructureExact); residual and
  // tolerance are their maxima, so holds() follows from theirs.
  if (read.classification != "none") {
    double residual = kNaN;
    double tolerance = 0.0;
    const auto consume = [&](const Certificate &cert) {
      residual = maxFinite(residual, cert.residual());
      tolerance = std::max(tolerance, cert.tolerance());
    };
    consume(band.certificate);
    consume(anchor.certificate);
    for (const FiberTransportRead &transport : evidence.lifetimeTransports)
      consume(transport.certificate);
    consume(winding.certificate);
    consume(parity.certificate);
    consume(occupation.certificate);
    read.certificate = Certificate::structureExact(
        CertificateDomain::Static, band.certificate.regime(), residual,
        band.certificate.conditioning(), tolerance);
  } else {
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, band.certificate.regime());
  }
  return read;
}

std::vector<QuarkRead> ParticleClusters::classifyQuarks(
    const std::vector<QuarkCandidateEvidence> &candidates) const {
  std::vector<QuarkRead> reads;
  reads.reserve(candidates.size());
  for (const QuarkCandidateEvidence &evidence : candidates)
    reads.push_back(classifyQuark(evidence));
  return reads;
}

QuarkRead ParticleClusters::classifyQuarkCached(
    cobordism::AnalyticCache &cache,
    const QuarkCandidateEvidence &evidence) const {
  const std::vector<std::uint64_t> ids = fiberVertexIds(evidence.colorBand);
  const auto parameter =
      static_cast<std::int64_t>(evidenceFingerprint(evidence));
  if (const auto payload = cache.fetch(ids, kCacheKind, parameter))
    return *std::static_pointer_cast<const QuarkRead>(payload);
  QuarkRead read = classifyQuark(evidence);
  cache.store(ids, kCacheKind, parameter, std::make_shared<QuarkRead>(read),
              read.certificate);
  return read;
}

std::uint64_t ParticleClusters::evidenceFingerprint(
    const QuarkCandidateEvidence &evidence) const {
  std::uint64_t h = 0x9e3779b97f4a7c15ull;
  // The thresholds are part of the verdict: a different configuration must
  // never serve another configuration's cached read.
  h = hashDouble(h, cfg_.parityTolerance);
  h = hashDouble(h, cfg_.occupationTolerance);
  h = hashDouble(h, cfg_.minAnchorScore);
  h = hashDouble(h, cfg_.minPhaseCoherence);
  h = hashDouble(h, cfg_.maxTransportLeakage);
  h = hashDouble(h, cfg_.minPersistenceLifetime);
  h = hashDouble(h, cfg_.minPersistenceOverlap);
  h = hashDouble(h, cfg_.minLocalization);
  h = hashDouble(h, cfg_.minRefinementOverlap);
  h = hashDouble(h, cfg_.doubletOverlapThreshold);
  h = chainHash(h, cfg_.minDoubletFrames);
  h = hashDouble(h, cfg_.isospinTolerance);
  h = hashDouble(h, cfg_.gaussTolerance);
  h = chainHash(h, cfg_.minEnclosingSurfaces);
  h = hashDouble(h, cfg_.udTolerance);

  h = hashString(h, evidence.component.canonicalHash());
  h = chainHash(h, evidence.component.level());

  // Color band: cells, eigenvalues, rank, acceptance, localization.
  for (const auto &cell : evidence.colorBand.cellVertices()) {
    h = chainHash(h, cell.size());
    for (const std::uint64_t id : cell) h = chainHash(h, id);
  }
  for (const cd &lambda : evidence.colorBand.eigenvalues())
    h = hashComplex(h, lambda);
  h = chainHash(h, evidence.colorBand.rank());
  h = chainHash(h, evidence.colorBand.accepted() ? 1u : 0u);
  h = hashDouble(h, evidence.colorBand.certificate().localization);

  // Anchor profile (decision channels + declared weighting).
  h = hashDouble(h, evidence.anchor.score);
  h = hashDouble(h, evidence.anchor.maxTerm);
  h = hashDouble(h, evidence.anchor.participationRatio);
  h = hashDouble(h, evidence.anchor.phaseCoherence);
  h = hashDouble(h, evidence.anchor.phaseDispersion);
  h = hashString(h, evidence.anchor.weightingId);
  h = hashCertificateHolds(h, evidence.anchor.certificate);

  // Lifetime transports.
  h = chainHash(h, evidence.lifetimeTransports.size());
  for (const FiberTransportRead &transport : evidence.lifetimeTransports) {
    h = chainHash(h, transport.accepted ? 1u : 0u);
    h = hashDouble(h, transport.leakage);
    h = hashComplex(h, transport.determinantPhase);
  }

  // Winding read with its recorded closure specification.
  h = chainHash(h, evidence.winding.winding.has_value() ? 1u : 0u);
  if (evidence.winding.winding.has_value())
    h = chainHash(h, static_cast<std::uint64_t>(
                         static_cast<std::int64_t>(*evidence.winding.winding)));
  h = hashString(h, evidence.winding.windingClosure);
  h = hashString(h, evidence.winding.windingReferenceId);
  h = hashDouble(h, evidence.winding.accumulatedPhase);
  h = hashCertificateHolds(h, evidence.winding.certificate);

  // Quasi-free parity/occupation reads (covariance hashes included).
  h = hashComplex(h, evidence.parityRead.value);
  h = hashString(h, evidence.parityRead.covarianceHash);
  h = hashCertificateHolds(h, evidence.parityRead.certificate);
  h = hashComplex(h, evidence.occupationRead.value);
  h = hashString(h, evidence.occupationRead.covarianceHash);
  h = hashCertificateHolds(h, evidence.occupationRead.certificate);

  h = hashDouble(h, evidence.persistenceLifetime);
  h = hashDouble(h, evidence.persistenceMinOverlap);
  h = hashDouble(h, evidence.refinementOverlap);

  h = chainHash(h, evidence.flavor.has_value() ? 1u : 0u);
  if (evidence.flavor.has_value()) {
    h = chainHash(h, evidence.flavor->found ? 1u : 0u);
    h = chainHash(h, evidence.flavor->rank);
    h = hashDouble(h, evidence.flavor->minContinuationOverlap);
    h = hashCertificateHolds(h, evidence.flavor->certificate);
  }
  h = chainHash(h, evidence.doubletOccupancy.has_value() ? 1u : 0u);
  if (evidence.doubletOccupancy.has_value()) {
    h = hashComplex(h, (*evidence.doubletOccupancy)(0));
    h = hashComplex(h, (*evidence.doubletOccupancy)(1));
  }
  h = chainHash(h, static_cast<std::uint64_t>(
                       static_cast<std::int64_t>(evidence.doubletOrientation)));
  h = chainHash(h, evidence.charge.has_value() ? 1u : 0u);
  if (evidence.charge.has_value()) {
    h = chainHash(h, evidence.charge->consistent ? 1u : 0u);
    h = chainHash(h, evidence.charge->electricFlux.has_value() ? 1u : 0u);
    if (evidence.charge->electricFlux.has_value())
      h = hashDouble(h, *evidence.charge->electricFlux);
    for (const cd &flux : evidence.charge->fluxes) h = hashComplex(h, flux);
  }
  return h;
}

// ---------------------------------------------------------------------------
// conjugate-pair conservation
// ---------------------------------------------------------------------------

ConjugatePairRead ParticleClusters::conjugatePair(
    const QuarkRead &first, const QuarkRead &second) const {
  ConjugatePairRead out;

  const bool windingFirst = first.determinantWinding.has_value();
  const bool windingSecond = second.determinantWinding.has_value();
  gate(windingFirst, "winding-first", out.failedCertificates);
  gate(windingSecond, "winding-second", out.failedCertificates);
  if (windingFirst && windingSecond)
    out.totalWinding = *first.determinantWinding + *second.determinantWinding;
  if (first.baryonFlux.has_value() && second.baryonFlux.has_value())
    out.totalBaryonFlux = *first.baryonFlux + *second.baryonFlux;

  const bool parityFirst = first.exteriorParity != 0;
  const bool paritySecond = second.exteriorParity != 0;
  gate(parityFirst, "parity-first", out.failedCertificates);
  gate(paritySecond, "parity-second", out.failedCertificates);
  if (parityFirst && paritySecond) {
    out.totalParity = first.exteriorParity * second.exteriorParity;
    out.parityEven = (out.totalParity == +1);
  }

  const bool windingConserved =
      out.totalWinding.has_value() && *out.totalWinding == 0;
  gate(windingConserved, "winding-conservation", out.failedCertificates);
  gate(out.parityEven, "parity-even", out.failedCertificates);

  out.conserved = windingConserved && out.parityEven;
  if (out.conserved) {
    // Integer sums of certified integers: exact given the premises.
    out.certificate = Certificate::structureExact(
        CertificateDomain::Static, CertificateRegime::NonNormal,
        /*residual=*/0.0, /*conditioning=*/kNaN, /*tolerance=*/1e-15);
  } else {
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, CertificateRegime::NonNormal);
  }
  return out;
}

// ---------------------------------------------------------------------------
// the emergent flavor doublet
// ---------------------------------------------------------------------------

FlavorDoubletRead ParticleClusters::flavorDoubletSearch(
    const std::vector<ComponentBandRead> &frames) const {
  FlavorDoubletRead out;
  const auto fail = [&](const char *reason) {
    out.found = false;
    out.invalidationReason = reason;
    out.failedCertificates.emplace_back("flavor-doublet");
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::NonNormal);
    return out;
  };

  if (frames.size() < cfg_.minDoubletFrames || frames.size() < 2)
    return fail("insufficient-frames");

  // One chain per frame-0 fiber; extend across consecutive frames through
  // certified continuations only (SpectralFiberTracker::matchFibers — the
  // #769 tracking, consumed rather than reimplemented).
  struct Chain {
    std::vector<std::size_t> positions;  // fiber index per covered frame
    double minOverlap = 1.0;
    bool alive = true;
  };
  std::vector<Chain> chains(frames[0].fibers.size());
  for (std::size_t i = 0; i < chains.size(); ++i)
    chains[i].positions = {i};

  for (std::size_t t = 0; t + 1 < frames.size(); ++t) {
    const std::vector<FiberMatchRead> matches =
        SpectralFiberTracker::matchFibers(frames[t].fibers,
                                          frames[t + 1].fibers,
                                          cfg_.doubletOverlapThreshold);
    std::unordered_map<std::size_t, const FiberMatchRead *> bestByFrom;
    for (const FiberMatchRead &match : matches)
      bestByFrom[match.fromIndex] = &match;

    // Tentative targets; two chains merging onto one band invalidate each
    // other (an ambiguous continuation is no continuation).
    std::unordered_map<std::size_t, std::size_t> claims;
    for (const Chain &chain : chains) {
      if (!chain.alive) continue;
      const auto it = bestByFrom.find(chain.positions.back());
      if (it != bestByFrom.end() && it->second->certifiedContinuation)
        ++claims[it->second->toIndex];
    }
    for (Chain &chain : chains) {
      if (!chain.alive) continue;
      const auto it = bestByFrom.find(chain.positions.back());
      if (it == bestByFrom.end() || !it->second->certifiedContinuation ||
          claims[it->second->toIndex] > 1) {
        chain.alive = false;
        continue;
      }
      chain.positions.push_back(it->second->toIndex);
      chain.minOverlap =
          std::min(chain.minOverlap, it->second->overlap.subspaceOverlap);
    }
  }

  // Stable subclasses: full-length chains.  Ranks are an OUTCOME.
  std::vector<std::size_t> stableChains;
  for (std::size_t c = 0; c < chains.size(); ++c)
    if (chains[c].alive && chains[c].positions.size() == frames.size())
      stableChains.push_back(c);
  for (const std::size_t c : stableChains)
    out.stableSubclassRanks.push_back(frames[0].fibers[c].rank());

  std::vector<std::size_t> twoState;
  for (const std::size_t c : stableChains)
    if (frames[0].fibers[c].rank() == 2) twoState.push_back(c);
  out.twoStateCount = twoState.size();

  if (twoState.empty()) return fail("no-stable-two-state-subclass");
  if (twoState.size() > 1) return fail("ambiguous-two-state-subclasses");

  const Chain &winner = chains[twoState.front()];
  const SpectralFiber &doublet = frames[0].fibers[twoState.front()];
  out.found = true;
  out.degree = doublet.degree();
  out.rank = doublet.rank();
  out.framesTracked = frames.size();
  out.minContinuationOverlap = winner.minOverlap;
  double minIsolation = std::numeric_limits<double>::infinity();
  for (std::size_t t = 0; t < frames.size(); ++t) {
    const SpectralBandCertificate &cert =
        frames[t].fibers[winner.positions[t]].certificate();
    minIsolation = std::min(minIsolation,
                            std::min(cert.lowerGap, cert.upperGap));
  }
  out.minIsolation = minIsolation;
  out.doublet = doublet;
  out.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, doublet.certificate().certificate.regime(),
      /*residual=*/1.0 - winner.minOverlap,
      /*conditioning=*/doublet.certificate().conditionNumber,
      /*tolerance=*/1.0 - cfg_.doubletOverlapThreshold);
  return out;
}

// ---------------------------------------------------------------------------
// the reused Gauss-flux electric read
// ---------------------------------------------------------------------------

GaussFluxRead ParticleClusters::gaussFluxOnSurfaces(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::complex<double>> &fieldStrength,
    const std::vector<std::vector<std::uint64_t>> &enclosedVertexSets,
    bool electricOnly) const {
  if (enclosedVertexSets.empty())
    throw std::invalid_argument(
        "ParticleClusters::gaussFluxOnSurfaces: at least one enclosing "
        "surface is required");
  // The EXISTING degree-2 Gauss read is consumed verbatim; constructing the
  // reader mutates nothing (a documented read-only entry point).
  cobordism::EigenstateSynthesis reader(st, 2);
  std::vector<cd> fluxes;
  std::vector<std::size_t> counts;
  fluxes.reserve(enclosedVertexSets.size());
  counts.reserve(enclosedVertexSets.size());
  for (const auto &enclosed : enclosedVertexSets) {
    fluxes.push_back(
        reader.gaussLawCharge(fieldStrength, enclosed, electricOnly));
    const std::set<std::uint64_t> unique(enclosed.begin(), enclosed.end());
    counts.push_back(unique.size());
  }
  return gaussFluxConsistency(fluxes, counts, electricOnly);
}

GaussFluxRead ParticleClusters::gaussFluxConsistency(
    const std::vector<std::complex<double>> &fluxes,
    const std::vector<std::size_t> &surfaceVertexCounts,
    bool electricOnly) const {
  if (fluxes.empty())
    throw std::invalid_argument(
        "ParticleClusters::gaussFluxConsistency: at least one flux is "
        "required");
  GaussFluxRead out;
  out.fluxes = fluxes;
  out.surfaceVertexCounts = surfaceVertexCounts;
  out.electricOnly = electricOnly;

  double maxDeviation = 0.0;
  double imagLeakage = 0.0;
  for (std::size_t i = 0; i < fluxes.size(); ++i) {
    imagLeakage = std::max(imagLeakage, std::abs(fluxes[i].imag()));
    for (std::size_t j = i + 1; j < fluxes.size(); ++j)
      maxDeviation = std::max(maxDeviation, std::abs(fluxes[i] - fluxes[j]));
  }
  out.maxDeviation = maxDeviation;
  out.imagLeakage = imagLeakage;

  out.consistent = fluxes.size() >= cfg_.minEnclosingSurfaces &&
                   maxDeviation <= cfg_.gaussTolerance &&
                   imagLeakage <= cfg_.gaussTolerance;
  if (out.consistent) {
    double mean = 0.0;
    for (const cd &flux : fluxes) mean += flux.real();
    out.electricFlux = mean / static_cast<double>(fluxes.size());
    // The per-surface sums are exact signed sums of the supplied cochain;
    // the consistency residual is what is graded.
    out.certificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, CertificateRegime::NonNormal,
        std::max(maxDeviation, imagLeakage), cfg_.gaussTolerance);
  } else {
    out.failedCertificates.emplace_back("gauss-consistency");
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, CertificateRegime::NonNormal);
  }
  return out;
}

std::vector<std::vector<std::uint64_t>> ParticleClusters::nestedEnclosures(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::uint64_t> &seedVertexIds, std::size_t shells) {
  if (shells < 1)
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: shells must be >= 1");
  if (seedVertexIds.empty())
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: empty seed");

  std::unordered_set<std::uint64_t> present;
  for (const auto v : st->getVertexList()->toVector())
    if (v != nullptr) present.insert(v->getId());

  std::set<std::uint64_t> current;
  for (const std::uint64_t id : seedVertexIds)
    if (present.count(id)) current.insert(id);
  if (current.empty())
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: no seed vertex exists in the "
        "complex");

  // Undirected one-skeleton adjacency (read-only edge sweep).
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adjacency;
  for (const auto e : st->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    adjacency[s->getId()].push_back(t->getId());
    adjacency[t->getId()].push_back(s->getId());
  }

  std::vector<std::vector<std::uint64_t>> sets;
  sets.reserve(shells);
  sets.emplace_back(current.begin(), current.end());
  for (std::size_t k = 1; k < shells; ++k) {
    std::set<std::uint64_t> next = current;
    for (const std::uint64_t id : current) {
      const auto it = adjacency.find(id);
      if (it == adjacency.end()) continue;
      next.insert(it->second.begin(), it->second.end());
    }
    current = std::move(next);
    sets.emplace_back(current.begin(), current.end());
  }
  return sets;
}

// ---------------------------------------------------------------------------
// candidate tracking across scale/time
// ---------------------------------------------------------------------------

std::vector<FiberMatchRead> ParticleClusters::trackCandidates(
    const std::vector<QuarkCandidateEvidence> &from,
    const std::vector<QuarkCandidateEvidence> &to, double overlapThreshold) {
  std::vector<SpectralFiber> fromBands;
  std::vector<SpectralFiber> toBands;
  fromBands.reserve(from.size());
  toBands.reserve(to.size());
  for (const QuarkCandidateEvidence &evidence : from)
    fromBands.push_back(evidence.colorBand);
  for (const QuarkCandidateEvidence &evidence : to)
    toBands.push_back(evidence.colorBand);
  return SpectralFiberTracker::matchFibers(fromBands, toBands,
                                           overlapThreshold);
}

}  // namespace tessera::observables
