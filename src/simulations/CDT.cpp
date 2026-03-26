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
#include <algorithm>
#include <cmath>
#include <map>

namespace caset {

CDT::CDT(std::shared_ptr<Spacetime> spacetime_, double k0_, double k4_, double delta_,
         double epsilon_, std::size_t targetN4_)
    : spacetime(std::move(spacetime_)), k0(k0_), k4(k4_), delta(delta_),
      epsilon(epsilon_), targetN4(targetN4_) {}

static int getDim(const std::shared_ptr<Spacetime> &st) {
  return st->getMetric()->getSignature()->getDimensions();
}

// ========================================
// Action Computation
// ========================================

double CDT::computeAction() const {
  auto n0 = static_cast<double>(spacetime->getVertexCount());
  auto n41 = static_cast<double>(spacetime->getN41());
  auto n32 = static_cast<double>(spacetime->getN32());
  auto n4 = n41 + n32;

  double regge = -(k0 + 6.0 * delta) * n0
               + (k4 + 2.0 * delta) * n41
               + (k4 + delta) * n32;
  double volumeFix = epsilon * (n4 - static_cast<double>(targetN4))
                             * (n4 - static_cast<double>(targetN4));
  return regge + volumeFix;
}

double CDT::computeDeltaAction(int dN0, int dN41, int dN32) const {
  int dN4 = dN41 + dN32;
  double n4 = static_cast<double>(spacetime->getN41() + spacetime->getN32());
  double target = static_cast<double>(targetN4);

  double dRegge = -(k0 + 6.0 * delta) * dN0
               + (k4 + 2.0 * delta) * dN41
               + (k4 + delta) * dN32;
  double oldFix = epsilon * (n4 - target) * (n4 - target);
  double newFix = epsilon * (n4 + dN4 - target) * (n4 + dN4 - target);
  return dRegge + (newFix - oldFix);
}

bool CDT::accept(double deltaS) {
  if (deltaS <= 0.0) return true;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < std::exp(-deltaS);
}

// ========================================
// Add Move: grow the complex by coning an external facet
// ========================================
bool CDT::add() {
  addAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  // Pick a random d-simplex
  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  // Find a causally available facet (non-timelike, < 2 cofaces)
  const auto &facets = sigma->getFacets();
  std::vector<SimplexPtr> available;
  for (const auto &f : facets) {
    if (!f->isTimelike() && f->isCausallyAvailable()) {
      available.push_back(f);
    }
  }
  if (available.empty()) return false;

  std::uniform_int_distribution<std::size_t> fDist(0, available.size() - 1);
  SimplexPtr facet = available[fDist(rng)];

  // Pre-compute the action change: +1 vertex, +1 d-simplex
  // The new simplex orientation depends on the cone direction. Estimate conservatively.
  auto [fti, ftf] = facet->getOrientation().numeric();
  auto coface = sigma;
  auto [cti, ctf] = coface->getOrientation().numeric();

  // Determine what orientation the new simplex will have
  int newTi, newTf;
  if (ctf > ftf) {
    // Cone adds vertex at ti → increases ti count
    newTi = fti + 1; newTf = ftf;
  } else {
    newTi = fti; newTf = ftf + 1;
  }

  int dN0 = 1;
  int dN41 = 0, dN32 = 0;
  if ((newTi == d && newTf == 1) || (newTi == 1 && newTf == d)) dN41 = 1;
  else if ((newTi == d - 1 && newTf == 2) || (newTi == 2 && newTf == d - 1)) dN32 = 1;

  double deltaS = computeDeltaAction(dN0, dN41, dN32);
  if (!accept(deltaS)) return false;

  // Execute: create vertex and cone
  double coneTime = (ctf > ftf) ? facet->getTi() : facet->getTf();
  std::vector<double> coords{coneTime};
  VertexPtr newVert = spacetime->createVertex(coords);
  facet->cone(newVert);

  addAccepted++;
  return true;
}

// ========================================
// Remove Move: shrink by removing a d-simplex with an external facet
// ========================================
bool CDT::remove() {
  removeAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  // Pick a random top simplex and check if it has a vertex belonging to only
  // this one top simplex (the "cone vertex" from a previous add move).
  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  VertexPtr uniqueVert = nullptr;
  for (const auto &v : sigma->getVertices()) {
    int dSimplexCount = 0;
    for (const auto &s : v->getSimplices()) {
      if (static_cast<int>(s->size()) == dPlus1) dSimplexCount++;
    }
    if (dSimplexCount == 1) {
      uniqueVert = v;
      break;
    }
  }
  if (!uniqueVert) return false;

  // Compute action change: -1 vertex, -1 d-simplex
  auto [sti, stf] = sigma->getOrientation().numeric();
  int dN0 = -1;
  int dN41 = 0, dN32 = 0;
  if ((sti == d && stf == 1) || (sti == 1 && stf == d)) dN41 = -1;
  else if ((sti == d - 1 && stf == 2) || (sti == 2 && stf == d - 1)) dN32 = -1;

  double deltaS = computeDeltaAction(dN0, dN41, dN32);
  if (!accept(deltaS)) return false;

  // Execute: remove simplex, clean up vertex
  spacetime->removeSimplex(sigma);

  // Remove edges connecting uniqueVert to the facet vertices
  // getEdges() returns a vector snapshot; safe to iterate while modifying
  Edges edgesToRemove = uniqueVert->getEdges();
  for (const auto &e : edgesToRemove) {
    VertexPtr other = (e->getSource()->getId() == uniqueVert->getId())
                      ? e->getTarget() : e->getSource();
    other->removeOutEdge(e);
    other->removeInEdge(e);
    spacetime->getEdgeList()->remove(e);
  }
  (void)spacetime->removeIfIsolated(uniqueVert);

  removeAccepted++;
  return true;
}

// ========================================
// (2, d) Flip Move
// ========================================
bool CDT::flip() {
  flipAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  // Get facets and pick a random one
  const auto &facets = sigma->getFacets();
  if (facets.empty()) return false;

  std::uniform_int_distribution<std::size_t> facetDist(0, facets.size() - 1);
  SimplexPtr facet = facets[facetDist(rng)];

  // Need exactly 2 d-simplex cofaces
  Simplices topCofaces;
  for (const auto &cf : facet->getCofaces()) {
    if (static_cast<int>(cf->size()) == dPlus1) topCofaces.push_back(cf);
  }
  if (topCofaces.size() != 2) return false;

  SimplexPtr s1 = topCofaces[0];
  SimplexPtr s2 = topCofaces[1];

  // Collect unique vertices: should be d+2 total (d shared + 2 unique)
  VertexPtrs allVerts;
  allVerts.reserve(d + 2);
  for (const auto &v : s1->getVertices()) allVerts.push_back(v);
  for (const auto &v : s2->getVertices()) {
    bool dup = false;
    for (const auto &av : allVerts) {
      if (av->getId() == v->getId()) { dup = true; break; }
    }
    if (!dup) allVerts.push_back(v);
  }
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (s1->hasVertex(v) && s2->hasVertex(v)) shared.push_back(v);
    else unique.push_back(v);
  }
  if (static_cast<int>(shared.size()) != d || unique.size() != 2) return false;

  // Count old orientations
  int old_n41 = 0, old_n32 = 0;
  for (const auto &s : {s1, s2}) {
    auto [sti, stf] = s->getOrientation().numeric();
    if ((sti == d && stf == 1) || (sti == 1 && stf == d)) old_n41++;
    else if ((sti == d - 1 && stf == 2) || (sti == 2 && stf == d - 1)) old_n32++;
  }

  // Create d new simplices: each has both unique + (d-1) of d shared
  std::vector<VertexPtrs> newSimplexVerts;
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs nv;
    for (int i = 0; i < d; ++i) {
      if (i != skip) nv.push_back(shared[i]);
    }
    nv.push_back(unique[0]);
    nv.push_back(unique[1]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    newSimplexVerts.push_back(nv);
  }

  // Count new orientations
  int new_n41 = 0, new_n32 = 0;
  for (const auto &nv : newSimplexVerts) {
    auto orient = SimplexOrientation::orientationOf(nv);
    auto [oti, otf] = orient.numeric();
    if ((oti == d && otf == 1) || (oti == 1 && otf == d)) new_n41++;
    else if ((oti == d - 1 && otf == 2) || (oti == 2 && otf == d - 1)) new_n32++;
  }

  int dN0 = 0;
  int dN41 = new_n41 - old_n41;
  int dN32 = new_n32 - old_n32;
  double deltaS = computeDeltaAction(dN0, dN41, dN32);
  if (!accept(deltaS)) return false;

  spacetime->removeSimplex(s1);
  spacetime->removeSimplex(s2);
  for (const auto &nv : newSimplexVerts) {
    spacetime->createSimplex(nv);
  }

  flipAccepted++;
  return true;
}

// ========================================
// (3, 3) Shift / Inverse Shift
// ========================================
bool CDT::shift() {
  shiftAttempts++;
  if (shiftImpl()) { shiftAccepted++; return true; }
  return false;
}

bool CDT::ishift() {
  ishiftAttempts++;
  if (shiftImpl()) { ishiftAccepted++; return true; }
  return false;
}

bool CDT::shiftImpl() {
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  // Pick 3 random vertices from sigma to form a candidate (d-2)-face
  const auto &sigmaVertsRef = sigma->getVertices();
  if (static_cast<int>(sigmaVertsRef.size()) < 3) return false;
  VertexPtrs sigmaVerts(sigmaVertsRef.begin(), sigmaVertsRef.end());
  std::shuffle(sigmaVerts.begin(), sigmaVerts.end(), rng);
  VertexPtrs triVerts(sigmaVerts.begin(), sigmaVerts.begin() + 3);

  // Find all d-simplices containing all 3 vertices
  std::vector<SimplexPtr> sharing;
  for (const auto &s : triVerts[0]->getSimplices()) {
    if (static_cast<int>(s->size()) != dPlus1) continue;
    if (s->hasVertex(triVerts[1]) && s->hasVertex(triVerts[2])) {
      sharing.push_back(s);
    }
  }
  if (sharing.size() != 3) return false;

  // Collect unique vertices (should be d+2)
  VertexPtrs allVerts;
  allVerts.reserve(d + 2);
  for (const auto &s : sharing) {
    for (const auto &v : s->getVertices()) {
      bool dup = false;
      for (const auto &av : allVerts) {
        if (av->getId() == v->getId()) { dup = true; break; }
      }
      if (!dup) allVerts.push_back(v);
    }
  }
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  // Separate shared (in all 3) and unique vertices
  VertexPtrs sharedVerts, uniqueVerts;
  for (const auto &v : allVerts) {
    bool inAll = true;
    for (const auto &s : sharing) {
      if (!s->hasVertex(v)) { inAll = false; break; }
    }
    if (inAll) sharedVerts.push_back(v);
    else uniqueVerts.push_back(v);
  }
  if (sharedVerts.size() != 3 || uniqueVerts.size() != 3) return false;

  // Count old orientations
  int old_n41 = 0, old_n32 = 0;
  for (const auto &s : sharing) {
    auto [sti, stf] = s->getOrientation().numeric();
    if ((sti == d && stf == 1) || (sti == 1 && stf == d)) old_n41++;
    else if ((sti == d - 1 && stf == 2) || (sti == 2 && stf == d - 1)) old_n32++;
  }

  // (3,3) move: each new simplex has all 3 unique + 2 of 3 shared
  std::vector<VertexPtrs> newSimplexVerts;
  for (int skip = 0; skip < 3; ++skip) {
    VertexPtrs nv;
    for (int i = 0; i < 3; ++i) {
      if (i != skip) nv.push_back(sharedVerts[i]);
    }
    for (const auto &u : uniqueVerts) nv.push_back(u);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    newSimplexVerts.push_back(nv);
  }

  int new_n41 = 0, new_n32 = 0;
  for (const auto &nv : newSimplexVerts) {
    auto orient = SimplexOrientation::orientationOf(nv);
    auto [oti, otf] = orient.numeric();
    if ((oti == d && otf == 1) || (oti == 1 && otf == d)) new_n41++;
    else if ((oti == d - 1 && otf == 2) || (oti == 2 && otf == d - 1)) new_n32++;
  }

  int dN0 = 0;
  int dN41 = new_n41 - old_n41;
  int dN32 = new_n32 - old_n32;
  double deltaS = computeDeltaAction(dN0, dN41, dN32);
  if (!accept(deltaS)) return false;

  for (const auto &s : sharing) spacetime->removeSimplex(s);
  for (const auto &nv : newSimplexVerts) spacetime->createSimplex(nv);

  return true;
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
      case 3: result = shift(); break;
      case 4: result = ishift(); break;
    }
    if (result) accepted++;
  }
  return accepted;
}

void CDT::tune() {
  // Tune k4 to its pseudo-critical value using proportional feedback.
  // The critical k4 makes the Regge action change ~0 for add/remove moves,
  // so that the volume constraint alone controls the volume.
  //
  // For an add move: dS_Regge = -(k0 + 6Δ)*1 + (k4 + ~1.5Δ)*1
  // Setting this near 0: k4_crit ≈ k0 + 6Δ - 1.5Δ = k0 + 4.5Δ
  k4 = k0 + 4.5 * delta;

  // Fine-tune with short feedback sweeps
  double target = static_cast<double>(targetN4);
  for (int i = 0; i < 200; ++i) {
    sweep();
    double n4 = static_cast<double>(spacetime->getSimplexCount());
    double error = (n4 - target) / target;  // normalized error
    k4 += 0.01 * error;
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
  auto rate = [](int accepted, int attempted) -> double {
    return attempted > 0 ? static_cast<double>(accepted) / attempted : 0.0;
  };
  return {
    {"add", rate(addAccepted, addAttempts)},
    {"remove", rate(removeAccepted, removeAttempts)},
    {"flip", rate(flipAccepted, flipAttempts)},
    {"shift", rate(shiftAccepted, shiftAttempts)},
    {"ishift", rate(ishiftAccepted, ishiftAttempts)},
  };
}

std::shared_ptr<Spacetime> CDT::getSpacetime() const noexcept { return spacetime; }
double CDT::getK0() const noexcept { return k0; }
double CDT::getK4() const noexcept { return k4; }
double CDT::getDelta() const noexcept { return delta; }

} // caset
