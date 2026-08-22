// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/AnalyticCache.h"

#include <functional>
#include <stdexcept>

#include "mesh/Fingerprint.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

// --------------------------------------------------------------------------
// TouchedStar
// --------------------------------------------------------------------------

void TouchedStar::addTouchedSimplex(const std::vector<std::uint64_t> &vertexIds) {
  vertices_.insert(vertexIds.begin(), vertexIds.end());
}

void TouchedStar::addChangedEdge(std::uint64_t vertexA, std::uint64_t vertexB) {
  vertices_.insert(vertexA);
  vertices_.insert(vertexB);
}

void TouchedStar::addCreatedCell(const std::vector<std::uint64_t> &vertexIds) {
  vertices_.insert(vertexIds.begin(), vertexIds.end());
  structural_ = true;
}

void TouchedStar::addDeletedCell(const std::vector<std::uint64_t> &vertexIds) {
  vertices_.insert(vertexIds.begin(), vertexIds.end());
  structural_ = true;
}

// --------------------------------------------------------------------------
// AnalyticCache
// --------------------------------------------------------------------------

AnalyticCache::AnalyticCache(std::shared_ptr<Spacetime> st)
    : st_(std::move(st)) {
  if (!st_)
    throw std::invalid_argument("AnalyticCache requires a spacetime");
  syncedRevision_ = st_->metricRevisionKey();
}

std::uint64_t AnalyticCache::componentKey(
    const std::vector<std::uint64_t> &vertexIds) {
  return ::tessera::mesh::Fingerprint::fingerprintOf(vertexIds);
}

std::uint64_t AnalyticCache::geometryRevision() const {
  return st_->metricRevisionKey();
}

std::uint64_t AnalyticCache::structuralRevision() const {
  return st_->structuralRevision();
}

std::size_t AnalyticCache::KeyHash::operator()(const Key &key) const noexcept {
  // Distinct-role mixing: rotate the kind and parameter hashes so equal
  // component/kind/parameter values in different roles do not cancel.
  const std::uint64_t kind = key.kind;
  const auto parameter =
      static_cast<std::uint64_t>(std::hash<std::int64_t>{}(key.parameter));
  std::uint64_t h = key.component;
  h ^= (kind << 17) | (kind >> 47);
  h ^= (parameter << 31) | (parameter >> 33);
  return static_cast<std::size_t>(h);
}

AnalyticCache::Key AnalyticCache::makeKey(
    const std::vector<std::uint64_t> &componentVertexIds,
    const std::string &kind, std::int64_t parameter) {
  return {componentKey(componentVertexIds),
          static_cast<std::uint64_t>(std::hash<std::string>{}(kind)), parameter};
}

bool AnalyticCache::fresh(const Entry &entry) const {
  const std::uint64_t current = st_->metricRevisionKey();
  // Served when nothing changed since the stamp, or when every change since
  // was published and this entry survived the intersection tests.
  return entry.revision == current || syncedRevision_ == current;
}

void AnalyticCache::store(const std::vector<std::uint64_t> &componentVertexIds,
                          const std::string &kind, std::int64_t parameter,
                          std::shared_ptr<void> payload, Certificate certificate) {
  Entry entry;
  entry.vertexIds = componentVertexIds;
  entry.payload = std::move(payload);
  entry.certificate = certificate;
  entry.revision = st_->metricRevisionKey();
  entries_[makeKey(componentVertexIds, kind, parameter)] = std::move(entry);
}

std::shared_ptr<void> AnalyticCache::fetch(
    const std::vector<std::uint64_t> &componentVertexIds,
    const std::string &kind, std::int64_t parameter) const {
  if (!enabled_) {
    ++misses_;
    return nullptr;
  }
  const auto found = entries_.find(makeKey(componentVertexIds, kind, parameter));
  if (found == entries_.end() || !fresh(found->second)) {
    ++misses_;
    return nullptr;
  }
  ++hits_;
  return found->second.payload;
}

const Certificate *AnalyticCache::fetchCertificate(
    const std::vector<std::uint64_t> &componentVertexIds,
    const std::string &kind, std::int64_t parameter) const {
  if (!enabled_)
    return nullptr;
  const auto found = entries_.find(makeKey(componentVertexIds, kind, parameter));
  if (found == entries_.end() || !fresh(found->second))
    return nullptr;
  return &found->second.certificate;
}

void AnalyticCache::publish(const TouchedStar &star) {
  for (auto it = entries_.begin(); it != entries_.end();) {
    bool touched = false;
    for (const std::uint64_t id : it->second.vertexIds) {
      if (star.vertices().count(id) > 0) {
        touched = true;
        break;
      }
    }
    if (touched) {
      it = entries_.erase(it);
      ++invalidations_;
    } else {
      ++it;
    }
  }
  syncedRevision_ = st_->metricRevisionKey();
}

void AnalyticCache::clear() {
  entries_.clear();
  syncedRevision_ = st_->metricRevisionKey();
}

} // namespace tessera::cobordism
