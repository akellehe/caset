// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Certificate.h"

#include <sstream>

namespace tessera::cobordism {

namespace {

const char *gradeName(CertificateGrade grade) {
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
  return "unknown";
}

const char *domainName(CertificateDomain domain) {
  switch (domain) {
  case CertificateDomain::Static:
    return "static";
  case CertificateDomain::BandWindow:
    return "band-window";
  }
  return "unknown";
}

const char *regimeName(CertificateRegime regime) {
  switch (regime) {
  case CertificateRegime::PositiveSemidefinite:
    return "positive-semidefinite";
  case CertificateRegime::HermitianIndefinite:
    return "hermitian-indefinite";
  case CertificateRegime::ComplexSymmetricPencil:
    return "complex-symmetric-pencil";
  case CertificateRegime::NonNormal:
    return "non-normal";
  }
  return "unknown";
}

} // namespace

Certificate Certificate::algebraicallyExact(CertificateDomain domain,
                                            CertificateRegime regime,
                                            double residual, double tolerance) {
  return {CertificateGrade::AlgebraicallyExact, domain,     regime,
          residual,                             kUnmeasured, tolerance};
}

Certificate Certificate::structureExact(CertificateDomain domain,
                                        CertificateRegime regime,
                                        double residual, double conditioning,
                                        double tolerance) {
  return {CertificateGrade::StructureExact, domain,       regime,
          residual,                         conditioning, tolerance};
}

Certificate Certificate::certifiedNumerical(CertificateDomain domain,
                                            CertificateRegime regime,
                                            double residual, double conditioning,
                                            double tolerance) {
  return {CertificateGrade::CertifiedNumerical, domain,       regime,
          residual,                             conditioning, tolerance};
}

Certificate Certificate::heuristicDiscovery(CertificateDomain domain,
                                            CertificateRegime regime) {
  return {CertificateGrade::HeuristicDiscovery, domain,      regime,
          kUnmeasured,                          kUnmeasured, 0.0};
}

std::string Certificate::describe() const {
  std::ostringstream out;
  out << gradeName(grade_) << " (" << domainName(domain_) << ", "
      << regimeName(regime_) << "): residual=" << residual_
      << " conditioning=" << conditioning_
      << " denseReferenceError=" << denseReferenceError_
      << " tolerance=" << tolerance_ << " holds=" << (holds() ? "yes" : "no");
  return out.str();
}

} // namespace tessera::cobordism
