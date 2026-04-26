// MIT License -- Copyright (c) 2025 Andrew Kelleher
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"

namespace tessera {

std::uint32_t EdgeList::allocSlot(const VertexPtr &source, const VertexPtr &target, double squaredLength) {
  std::uint32_t slot;
  if (!freeSlots_.empty()) {
    slot = freeSlots_.back();
    freeSlots_.pop_back();
    pool_[slot] = Edge(source, target, squaredLength);
  } else {
    slot = static_cast<std::uint32_t>(pool_.size());
    pool_.emplace_back(source, target, squaredLength);
  }
  return slot;
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target) {
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];
  return getOrInsert(source, target, 0.0);
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target, double squaredLength) noexcept {
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];
  return getOrInsert(source, target, squaredLength);
}

EdgePtr EdgeList::getOrInsert(const VertexPtr &source, const VertexPtr &target, double squaredLength) {
  if (source->getId() == target->getId()) {
    throw std::runtime_error("You cannot create an edge from a vertex to itself.");
  }
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];

  auto slot = allocSlot(source, target, squaredLength);
  fpToSlot_.emplace(fp, slot);
  EdgePtr raw = &pool_[slot];
  raw->liveIdx_ = static_cast<std::uint32_t>(liveVec_.size());
  liveVec_.push_back(raw);
  return raw;
}

void EdgeList::remove(const EdgePtr &edge) noexcept {
  auto fp = edge->fingerprint.fingerprint();
  auto it = fpToSlot_.find(fp);
  if (it == fpToSlot_.end()) return;
  freeSlots_.push_back(it->second);
  fpToSlot_.erase(it);

  // Swap-and-pop from liveVec_ using the index stored on the Edge
  auto idx = edge->liveIdx_;
  if (idx < liveVec_.size()) {
    auto lastIdx = static_cast<std::uint32_t>(liveVec_.size() - 1);
    if (idx != lastIdx) {
      liveVec_[idx] = liveVec_[lastIdx];
      liveVec_[idx]->liveIdx_ = idx;
    }
    liveVec_.pop_back();
  }
  edge->liveIdx_ = UINT32_MAX;
}

void EdgeList::replace(const EdgePtr &toRemove, const EdgePtr &toAdd) noexcept {
  remove(toRemove);
  add(toAdd->getSource(), toAdd->getTarget(), toAdd->getSquaredLength());
}

void EdgeList::rekeyEdge(std::uint64_t oldFp, std::uint64_t newFp) {
  if (oldFp == newFp) return;
  auto it = fpToSlot_.find(oldFp);
  if (it == fpToSlot_.end()) return;
  auto slot = it->second;
  fpToSlot_.erase(it);
  fpToSlot_.emplace(newFp, slot);
  // liveIdx_ on the Edge object doesn't change — only the fingerprint key does
}

const Edges &EdgeList::toVector() const noexcept {
  return liveVec_;
}

std::size_t EdgeList::size() const {
  return fpToSlot_.size();
}

EdgePtr EdgeList::get(const std::uint64_t &fingerprint) {
  auto it = fpToSlot_.find(fingerprint);
  if (it == fpToSlot_.end()) throw std::out_of_range("Edge not found");
  return &pool_[it->second];
}

void EdgeList::reserve(std::size_t nSimplices) {
  fpToSlot_.reserve(nSimplices);
}

} // tessera
