// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// The register carried by a certified cluster (ticket #860, whitepaper
// section "Recursive spectral fibers").
//
// The specification states the construction directly:
//
//   "Within component C, choose an isolated localized spectral band and, in
//    the positive self-adjoint regime, a weighted orthonormal frame
//    Phi_C = (phi_1, ..., phi_r), Phi_C^dagger W_C Phi_C = I_r.
//    The derived fiber is E_C = Ran Phi_C."
//
// and its acceptance:
//
//   "It need not be a harmonic space and therefore need not be supported by
//    a hole.  What it does require is a spectral gap, localization, and
//    persistence.  A candidate component is accepted only if all of the
//    following remain stable across a stated range of scales:
//      - a persistent connected cluster support, however proposed;
//      - a localized spectral projector with stable rank;
//      - a nonzero band gap separating it from discarded modes;
//      - overlap with its predecessor and successor components;
//      - lifetime across multiple cobordism frames; and
//      - small external transport leakage."
//
// This file assembles exactly that from the existing observables — it
// derives no spectrum, no transport and no clustering of its own.  Bands and
// their gap/localization/regime certificates come from `SpectralFiber`
// (#769); the support and its frame lifetime/overlap come from
// `PersistentModularity` (#765); the leakage comes from `FiberConnection`
// (#770).
//
// NOTHING HERE IS TARGET-CONDITIONED.  There is no target vector, no
// residual, and no objective term: the register is READ from a relaxed
// geometry after the fact.  Nothing in this file may be reached from an
// emergence objective, and nothing in it tells the geometry what to become.
//
// NO HOLE IS REQUIRED OR CONSULTED.  A hole names the boundary cycle of a
// REMOVED top cell, and the removal is what makes that cycle non-bounding; a
// cluster inside a filled complex supplies only cycles that bound, and a
// harmonic form has vanishing period over a bounding cycle.  A period
// readout would therefore report zero on a cluster regardless of the
// geometry, which is why the fiber above is a FRAME RANGE and not a period.
// The existing hole machinery is untouched by this file and is not consulted
// as a fallback.

#ifndef TESSERA_OBSERVABLES_CLUSTERREGISTER_H
#define TESSERA_OBSERVABLES_CLUSTERREGISTER_H

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"
#include "observables/FiberConnection.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"
#include "observables/SpectralFiber.h"

namespace tessera::spacetime {
  class Spacetime;
}

namespace tessera::observables {

/// The whitepaper's six fiber-acceptance conjuncts, named.
///
/// Constants rather than literals at each site: a conjunct name is written
/// where it is decided and compared where it is consumed, and a mis-spelling
/// in either place would not fail to compile — it would silently produce a
/// name no consumer matches.  Bound to Python so a caller references the
/// constant instead of retyping the string.
struct RegisterConjunct {
  /// "a persistent connected cluster support, however proposed" — the
  /// support is non-empty and its induced one-skeleton is connected.  HOW
  /// the support was proposed is never consulted: any proposer is
  /// admissible and none may veto (the specification's "however proposed").
  static constexpr const char *kClusterSupport = "cluster-support";
  /// "a localized spectral projector with stable rank" — the band's
  /// localization excess is within the detector's declared cap.  Decided by
  /// `SpectralFiber`'s own certificate, re-read here rather than re-derived.
  static constexpr const char *kLocalizedProjector = "localized-projector";
  /// "a nonzero band gap separating it from discarded modes" — the band's
  /// separation from the nearest DISCARDED eigenvalue in the complex plane.
  static constexpr const char *kBandGap = "band-gap";
  /// "overlap with its predecessor and successor components" — the smallest
  /// adjacent-FRAME support overlap along the component's frame track.
  static constexpr const char *kNeighbourOverlap = "neighbour-overlap";
  /// "lifetime across multiple cobordism frames" — the number of
  /// consecutive frames the support was tracked through.
  static constexpr const char *kFrameLifetime = "frame-lifetime";
  /// "small external transport leakage" — the largest leakage over the
  /// supplied external transports.
  static constexpr const char *kTransportLeakage = "transport-leakage";
};

/// Why a conjunct could not be DECIDED, as distinct from being decided
/// against.
///
/// A conjunct that was measured and fell short is a FAILURE; a conjunct with
/// no evidence behind it is UNMEASURED.  Both block acceptance, and the two
/// are never merged: an unmeasured quantity is not a failed one, and neither
/// is ever encoded as a zero that would claim a measurement nobody made.
struct RegisterUnmeasured {
  /// No band was supplied, so nothing spectral could be decided.
  static constexpr const char *kNoBand = "no-band";
  /// The band carries no localization measurement (a NaN excess).
  static constexpr const char *kLocalizationUnmeasured = "localization-unmeasured";
  /// The band's separation from the discarded modes is UNKNOWN — the
  /// truncated sparse top leaves one side uncovered.
  static constexpr const char *kBandGapUnknown = "band-gap-unknown";
  /// No frame track was supplied, so the lifetime was never measured.  The
  /// certificate fails BY NAME rather than falling back to a single-frame
  /// test that would pass vacuously.
  static constexpr const char *kNoFrameTrack = "no-frame-track";
  /// No external transport was supplied, so leakage was never measured.
  /// Absence of transports is NOT evidence of small leakage.
  static constexpr const char *kNoTransport = "no-transport";
  /// The complex could not be read to decide support connectivity.
  static constexpr const char *kSupportUnreadable = "support-unreadable";
};

/// Thresholds the register is accepted under.  Every one is an analysis
/// parameter recorded on the read; none of them selects which bands or
/// clusters EXIST.
struct ClusterRegisterConfig {
  /// Floor on the smallest adjacent-frame support overlap
  /// (`FrameTrack::minAdjacentOverlap`).
  double minNeighbourOverlap = 0.5;
  /// Floor on the cobordism-frame lifetime (`FrameTrack::frames`).  The
  /// specification says "lifetime across MULTIPLE cobordism frames", so the
  /// default is two: one frame is not a lifetime.
  std::size_t minFrameLifetime = 2;
  /// Cap on the largest external transport leakage.
  double maxTransportLeakage = 1e-6;
};

/// What the specification requires be REPORTED of the band's metric regime.
///
/// "In a Hermitian indefinite regime record the inertia of
/// Phi_C^dagger W_C Phi_C and normalize it to a signature matrix
/// J_C = diag(I_p, -I_q). ... In a non-normal regime use matched right and
/// left frames Phi_C, Psi_C with Psi_C^dagger W_C Phi_C = I and report both
/// residuals and the frame condition number."
///
/// Quantities the regime does not define are NaN, never zero.
struct RegisterRegimeReport {
  /// The VERIFIED regime of the band's solve — never assumed.
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// The weighted Gram / signature defect ||Phi^dagger W Phi - J||.  In the
  /// positive regime J = I and this is the specification's orthonormality
  /// condition Phi_C^dagger W_C Phi_C = I_r.
  double gramDefect = std::numeric_limits<double>::quiet_NaN();
  /// Krein inertia (p, q) of Phi^dagger W Phi, so the normalized signature
  /// is J_C = diag(I_p, -I_q).  Reported in EVERY regime (p = rank, q = 0 in
  /// the positive one); the specification requires it in the Hermitian
  /// indefinite regime.  A NEGATIVE SIGNATURE IS A CERTIFICATE, never an
  /// automatic identification with an antiparticle.
  int positiveSignature = 0;
  int negativeSignature = 0;
  /// Neutral directions rank - p - q, nonzero only when the W-Gram is
  /// singular.
  int neutralSignature = 0;
  /// Whether the inertia is the specification's normalizable signature
  /// (no neutral directions), so J_C = diag(I_p, -I_q) exists.
  bool signatureNormalizable = false;
  /// Right-frame residual ||L Phi - Phi Lambda|| / ||L||.  The first of the
  /// "both residuals" the non-normal regime must report.
  double eigenResidual = std::numeric_limits<double>::quiet_NaN();
  /// Left-frame residual.  The second.  Equal to `eigenResidual` on the
  /// self-adjoint path, where the frames coincide.
  double leftResidual = std::numeric_limits<double>::quiet_NaN();
  /// The frame condition number the non-normal regime must report.
  double frameConditionNumber = std::numeric_limits<double>::quiet_NaN();
};

/// # ClusterRegisterRead
///
/// One register read: the cluster it is carried by, the fiber it is, the six
/// conjuncts as measured, and the verdict.
///
/// The fiber itself is `E_C = Ran Phi_C`, carried here as the band whose
/// right frame is `Phi_C` (`band.rightFrame()`); the range is represented by
/// the band PROJECTOR rather than by any individual eigenvector, since a
/// choice of in-band basis is a gauge choice and never determines an
/// identity.
///
/// Unmeasured values are NaN and unmeasured conjuncts are named; nothing is
/// zero-filled and no worst-case value ever stands in for a measurement.
struct ClusterRegisterRead {
  /// The cluster's label-free identity, when the caller supplied one.
  ComponentId component{};
  /// The cluster's level-0 vertex support, as supplied.  How it was proposed
  /// is deliberately not recorded here: it plays no part in acceptance.
  std::vector<std::uint64_t> support{};
  /// Form degree of the band.
  int degree = 0;
  /// Band rank r — the fiber's dimension.  Reported as measured; no rank is
  /// ever requested or required by this read.
  std::size_t rank = 0;
  /// The band carrying the fiber (its right frame is `Phi_C`).
  SpectralFiber band{};

  // ── the six conjuncts, as measured ────────────────────────────────
  /// Whether the support is non-empty and its induced one-skeleton is
  /// connected.
  bool supportConnected = false;
  /// Number of connected pieces the support induces (1 when connected).
  std::size_t supportPieces = 0;
  /// The band's localization excess (0 = as concentrated as the rank
  /// permits, 1 = perfectly delocalized).
  double localizationExcess = std::numeric_limits<double>::quiet_NaN();
  /// The band's separation from the nearest discarded eigenvalue.
  double bandGap = std::numeric_limits<double>::quiet_NaN();
  /// Smallest adjacent-frame support overlap along the frame track.
  double neighbourOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Consecutive cobordism frames the support was tracked through.
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Largest leakage over the supplied external transports.
  double transportLeakage = std::numeric_limits<double>::quiet_NaN();

  /// What the regime requires be reported.
  RegisterRegimeReport regime{};

  /// Conjuncts that were MEASURED and fell short, by name.
  std::vector<std::string> failedConjuncts{};
  /// Conjuncts that could not be measured at all, by name (see
  /// :class:`RegisterUnmeasured`).  Distinct from a failure: no evidence is
  /// not evidence of failure, and neither is a zero.
  std::vector<std::string> unmeasured{};

  /// Accepted exactly when all six conjuncts were measured and met, and the
  /// band's own certificate holds.
  bool accepted = false;

  /// The graded claim: `CertifiedNumerical` when accepted, otherwise the
  /// uncertified `HeuristicDiscovery`, which never `holds()`.
  cobordism::Certificate certificate{};

  /// The thresholds this read was decided under.
  ClusterRegisterConfig thresholds{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static ClusterRegisterRead fromRecord(const Record &record);
};

/// # ClusterRegister
///
/// Reads the register the whitepaper's "Recursive spectral fibers" section
/// defines: the fiber `E_C = Ran Phi_C` of an isolated localized band on a
/// persistent cluster, accepted under the six-conjunct list.
///
/// **Assembles, never re-derives.**  Bands, their gap/localization measures
/// and their regime certificates come from `SpectralFiber`; the frame track
/// comes from `PersistentModularity::trackAcrossFrames`; leakage comes from
/// `FiberConnection`.  This class decides the conjunction and reports it.
///
/// **However proposed.**  The support arrives as a plain vertex-id list and
/// its provenance is never consulted.  A modularity read cannot veto a fiber
/// that meets the six conjuncts, because no proposer is an input to the
/// decision at all.
///
/// **No color claim.**  The specification requires the anchoring certificate
/// "whenever a color interpretation is claimed"; a register is not such a
/// claim, so no anchor is required or consulted here.  A caller that wants a
/// color reading takes it from the quark classifier, which gates on the
/// anchor as the specification says it must.
///
/// **Read-only.**  Never solves on, and never mutates, the spacetime.
/// Nothing here enters any emergence objective.
class ClusterRegister {
  public:
    /// Bind the acceptance thresholds.
    explicit ClusterRegister(ClusterRegisterConfig cfg = {});

    [[nodiscard]] const ClusterRegisterConfig &config() const noexcept {
      return cfg_;
    }

    /// Read the register of one cluster.
    ///
    /// `support` is the cluster's vertex-id set, however proposed.  `band`
    /// is the isolated localized band whose right frame is `Phi_C`.  `track`
    /// supplies the cobordism-frame lifetime and the adjacent-frame overlap;
    /// absent means those two conjuncts are UNMEASURED, never satisfied.
    /// `externalTransports` are the transports leaving the cluster; an empty
    /// list likewise leaves leakage unmeasured rather than small.
    ///
    /// `st` is read only to decide support connectivity.
    [[nodiscard]] ClusterRegisterRead read(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::uint64_t> &support, const SpectralFiber &band,
        const std::optional<FrameTrack> &track,
        const std::vector<FiberTransportRead> &externalTransports,
        ComponentId component = {}) const;

    /// Whether the induced one-skeleton on `support` is connected, and in
    /// how many pieces.  Returns `{false, 0}` when the support is empty or
    /// the complex cannot be read.
    [[nodiscard]] static std::pair<bool, std::size_t> supportConnectivity(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::uint64_t> &support);

  private:
    ClusterRegisterConfig cfg_{};
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_CLUSTERREGISTER_H
