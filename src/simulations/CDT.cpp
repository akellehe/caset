// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "simulations/CDT.h"
#include "Logger.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace { // anonymous — local to this TU
template<typename T, std::size_t Cap = 8>
struct StackVec {
  std::array<T, Cap> data_{};
  std::uint8_t len_ = 0;
  void push_back(T v) noexcept { if (len_ < Cap) data_[len_++] = v; }
  T  operator[](std::size_t i) const noexcept { return data_[i]; }
  T &operator[](std::size_t i)       noexcept { return data_[i]; }
  std::size_t size()  const noexcept { return len_; }
  const T *begin() const noexcept { return data_.data(); }
  const T *end()   const noexcept { return data_.data() + len_; }
};
} // anon

namespace tessera {

CDT::CDT(std::shared_ptr<Spacetime> spacetime_, double k0_, double k4_, double delta_,
         double epsilon_, std::size_t targetN41_, bool quadraticVolumeFix_)
    : spacetime(std::move(spacetime_)), k0(k0_), k4(k4_), delta(delta_),
      epsilon(epsilon_), targetN41(targetN41_), quadraticVolumeFix(quadraticVolumeFix_) {}

static int getDim(const std::shared_ptr<Spacetime> &st) {
  return st->getMetric()->getSignature()->getDimensions();
}

/// Check that a proposed simplex vertex set has a valid CDT orientation:
/// (d,1), (1,d), (d-1,2), or (2,d-1), AND spans exactly 2 time slices.
static bool isValidCDTOrientation(const VertexPtrs &verts, int d) {
  // Must span exactly 2 distinct times (CDT causality constraint)
  std::unordered_set<std::uint64_t> times;
  for (const auto &v : verts) {
    // Use floor cast (consistent with volume profile time binning)
    times.insert(static_cast<std::uint64_t>(v->getTime()));
  }
  if (times.size() != 2) return false;

  auto orient = SimplexOrientation::orientationOf(verts);
  auto [ti, tf] = orient.numeric();
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) return true;
  if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) return true;
  return false;
}

static bool isN41Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}


/// Select a uniformly random N41-type top simplex.
/// Uses rejection sampling with a fallback linear scan.
SimplexPtr CDT::getRandomN41Simplex(int d) {
  int dPlus1 = d + 1;
  // Fast path: rejection sampling (effective when N41/N4 is not too small)
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto s = spacetime->getRandomTopSimplex();
    if (s && static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d)) return s;
  }
  // Fallback: linear scan
  std::vector<SimplexPtr> matches;
  for (const auto &s : spacetime->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d))
      matches.push_back(s);
  }
  if (matches.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, matches.size() - 1);
  return matches[dist(rng)];
}

// ========================================
// Action Computation
// ========================================

double CDT::computeAction() const {
  auto n0 = static_cast<double>(spacetime->getVertexCount());
  auto n41 = static_cast<double>(spacetime->getN41());
  auto n32 = static_cast<double>(spacetime->getN32());

  double regge = -(k0 + 6.0 * delta) * n0
               + (k4 + 2.0 * delta) * n41
               + (k4 + delta) * n32;
  double target = static_cast<double>(targetN41);
  double volumeFix;
  if (quadraticVolumeFix) {
    volumeFix = epsilon * (n41 - target) * (n41 - target);
  } else {
    volumeFix = epsilon * std::abs(n41 - target);
  }
  return regge + volumeFix;
}

// computeDeltaAction and accept are inlined in CDT.h

// ========================================
// (2, 2d) Add Move: vertex insertion at spatial face
// ========================================
bool CDT::add() {
  addAttempts++;
  AddMove move(spacetime.get(), &rng, relabelVertices_);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  addAccepted++;
  return true;
}

// ========================================
// (2d, 2) Remove Move: vertex deletion (blind guessing)
// ========================================
bool CDT::remove() {
  removeAttempts++;
  RemoveMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  removeAccepted++;
  return true;
}

// ========================================
// (2, d) Flip Move
// ========================================
bool CDT::flip() {
  flipAttempts++;
  FlipMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  flipAccepted++;
  return true;
}

// ========================================
// (d, 2) Inverse Flip Move
// ========================================
bool CDT::iflip() {
  iflipAttempts++;
  IFlipMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  iflipAccepted++;
  return true;
}

// ========================================
// (3, 3) Shift (self-inverse)
// ========================================
bool CDT::shift() {
  shiftAttempts++;
  if (shiftImpl()) {
    shiftAccepted++;
    return true;
  }
  return false;
}

bool CDT::ishift() {
  ishiftAttempts++;
  if (shiftImpl()) {
    ishiftAccepted++;
    return true;
  }
  return false;
}

bool CDT::shiftImpl() {
  ShiftMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  return move.apply();
}

// ========================================
// Transactional move factories
// ========================================
//
// These hand the caller a fresh PachnerMove already bound to this
// simulation's spacetime and Markov-chain RNG.  Useful for the
// modularity-sweep optimizer (observables/ModularityOptimizer.h),
// which needs to layer custom acceptance (Q-direction filter) on top
// of the bare move mechanics.  Each factory calls ``propose()``;
// returns nullptr if no eligible target.

std::unique_ptr<PachnerMove> CDT::proposeAdd() {
  auto m = std::make_unique<AddMove>(spacetime.get(), &rng, relabelVertices_);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeRemove() {
  auto m = std::make_unique<RemoveMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeFlip() {
  auto m = std::make_unique<FlipMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeIflip() {
  auto m = std::make_unique<IFlipMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeShift() {
  auto m = std::make_unique<ShiftMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

// ========================================
// Metropolis Sweep
// ========================================

int CDT::sweep() {
  int n4 = static_cast<int>(spacetime->getSimplexCount());
  if (n4 <= 0) n4 = 1;
  int accepted = 0;

  std::uniform_int_distribution<int> moveDist(0, 4);
  for (int i = 0; i < n4; ++i) {
    int moveType = moveDist(rng);
    bool result = false;
    switch (moveType) {
      case 0: result = add(); break;
      case 1: result = remove(); break;
      case 2: result = flip(); break;
      case 3: result = iflip(); break;
      case 4: result = shift(); break;
    }
    if (result) accepted++;
  }
  return accepted;
}

void CDT::tune(std::function<void(int,int)> progress) {
  // Tune k4 to its pseudo-critical value using proportional feedback.
  // For the (2,2d) add move: dS_Regge ≈ -(k0+6Δ) + (2d-2)(k4+2Δ)
  // Setting this near 0 for d=4: k4_crit ≈ (k0+6Δ)/(2d-2) - 2Δ
  int d = getDim(spacetime);
  if (d <= 1) return;  // CDT requires d >= 2
  k4 = (k0 + 6.0 * delta) / (2.0 * d - 2.0) - 2.0 * delta;

  // Fine-tune with short feedback sweeps
  constexpr int nTuneSteps = 20;
  double target = static_cast<double>(targetN41);
  for (int i = 0; i < nTuneSteps; ++i) {
    sweep();
    double n41 = static_cast<double>(spacetime->getN41());
    double error = (n41 - target) / target;  // normalized error
    k4 += 0.01 * error;
    if (progress) progress(i + 1, nTuneSteps);
  }
}

void CDT::thermalize() {
  double prevAction = computeAction();
  for (int i = 0; i < 200; ++i) {
    sweep();
    double action = computeAction();
    if (std::abs(action - prevAction) / (std::abs(prevAction) + 1e-10) < 0.01) {
      if (i > 20) return;
    }
    prevAction = action;
  }
}

// ========================================
// Observables
// ========================================

std::vector<int> CDT::getVolumeProfile() const {
  int d = getDim(spacetime);
  std::size_t dPlus1 = static_cast<std::size_t>(d + 1);
  std::map<int, int> profile;
  for (const auto &s : spacetime->getSimplices()) {
    if (s->size() != dPlus1) continue;
    int tMin = static_cast<int>(s->getTi());
    profile[tMin]++;
  }
  if (profile.empty()) return {};
  int tMax = profile.rbegin()->first;
  int tMin = profile.begin()->first;
  std::vector<int> result(tMax - tMin + 1, 0);
  for (const auto &[t, count] : profile) {
    result[t - tMin] = count;
  }
  return result;
}

std::map<std::string, double> CDT::getAcceptanceRates() const {
  auto rate = [](std::int64_t accepted, std::int64_t attempted) -> double {
    return attempted > 0 ? static_cast<double>(accepted) / static_cast<double>(attempted) : 0.0;
  };
  return {
    {"add", rate(addAccepted, addAttempts)},
    {"remove", rate(removeAccepted, removeAttempts)},
    {"flip", rate(flipAccepted, flipAttempts)},
    {"iflip", rate(iflipAccepted, iflipAttempts)},
    {"shift", rate(shiftAccepted, shiftAttempts)},
    {"ishift", rate(ishiftAccepted, ishiftAttempts)},
  };
}

const std::shared_ptr<Spacetime> &CDT::getSpacetime() const noexcept { return spacetime; }
double CDT::getK0() const noexcept { return k0; }
double CDT::getK4() const noexcept { return k4; }
double CDT::getDelta() const noexcept { return delta; }

} // tessera
