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

  double regge = -(k0 + 6.0 * delta) * n0 + (k4 + delta) * n41 + k4 * n32;
  double volumeFix = epsilon * (n4 - static_cast<double>(targetN4))
                             * (n4 - static_cast<double>(targetN4));
  return regge + volumeFix;
}

double CDT::computeDeltaAction(int dN0, int dN41, int dN32) const {
  int dN4 = dN41 + dN32;
  double n4 = static_cast<double>(spacetime->getN41() + spacetime->getN32());
  double target = static_cast<double>(targetN4);

  double dRegge = -(k0 + 6.0 * delta) * dN0 + (k4 + delta) * dN41 + k4 * dN32;
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
  SimplexPtr sigma = spacetime->getRandomSimplex();
  if (!sigma || static_cast<int>(sigma->getVertices().size()) != dPlus1) return false;

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

  SimplexPtr sigma = spacetime->getRandomSimplex();
  if (!sigma || static_cast<int>(sigma->getVertices().size()) != dPlus1) return false;

  // Check that sigma has a facet with only 1 coface (this simplex itself)
  // This means removing sigma "exposes" that facet as external.
  // Also check that the vertex unique to sigma (not in the shared facet) has
  // no other d-simplices — so we can cleanly remove it.
  const auto &facets = sigma->getFacets();
  SimplexPtr removableFacet = nullptr;
  VertexPtr uniqueVert = nullptr;

  for (const auto &f : facets) {
    // This facet should only have sigma as coface
    SimplexPtrSet topCofaces;
    for (const auto &cf : f->getCofaces()) {
      if (static_cast<int>(cf->getVertices().size()) == dPlus1) topCofaces.insert(cf);
    }
    if (topCofaces.size() != 1) continue;

    // Find the vertex in sigma that's NOT in this facet
    VertexPtr candidate = nullptr;
    for (const auto &v : sigma->getVertices()) {
      if (!f->hasVertex(v)) { candidate = v; break; }
    }
    if (!candidate) continue;

    // The vertex should belong to only this d-simplex (so removing is clean)
    int dSimplexCount = 0;
    for (const auto &s : candidate->getSimplices()) {
      if (static_cast<int>(s->getVertices().size()) == dPlus1) dSimplexCount++;
    }
    if (dSimplexCount != 1) continue;

    removableFacet = f;
    uniqueVert = candidate;
    break;
  }
  if (!removableFacet || !uniqueVert) return false;

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
  // Copy needed: we're modifying the edge sets during iteration
  EdgePtrSet edgesToRemove = uniqueVert->getEdges();
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

  SimplexPtr sigma = spacetime->getRandomSimplex();
  if (!sigma || static_cast<int>(sigma->getVertices().size()) != dPlus1) return false;

  // Get facets and pick a random one
  const auto &facets = sigma->getFacets();
  if (facets.empty()) return false;

  std::uniform_int_distribution<std::size_t> facetDist(0, facets.size() - 1);
  SimplexPtr facet = facets[facetDist(rng)];

  // Need exactly 2 d-simplex cofaces
  SimplexPtrSet topCofaces;
  for (const auto &cf : facet->getCofaces()) {
    if (static_cast<int>(cf->getVertices().size()) == dPlus1) topCofaces.insert(cf);
  }
  if (topCofaces.size() != 2) return false;

  auto it = topCofaces.begin();
  SimplexPtr s1 = *it++;
  SimplexPtr s2 = *it;

  // Collect vertices: should be d+2 total (d shared + 2 unique)
  VertexPtrSet allVerts;
  for (const auto &v : s1->getVertices()) allVerts.insert(v);
  for (const auto &v : s2->getVertices()) allVerts.insert(v);
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
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  SimplexPtr sigma = spacetime->getRandomSimplex();
  if (!sigma || static_cast<int>(sigma->getVertices().size()) != dPlus1) return false;

  // Pick 3 random vertices from sigma to form a candidate (d-2)-face
  const auto &sigmaVertsRef = sigma->getVertices();
  if (static_cast<int>(sigmaVertsRef.size()) < 3) return false;
  VertexPtrs sigmaVerts(sigmaVertsRef.begin(), sigmaVertsRef.end());
  std::shuffle(sigmaVerts.begin(), sigmaVerts.end(), rng);
  VertexPtrs triVerts(sigmaVerts.begin(), sigmaVerts.begin() + 3);

  // Find all d-simplices containing all 3 vertices
  std::vector<SimplexPtr> sharing;
  for (const auto &s : triVerts[0]->getSimplices()) {
    if (static_cast<int>(s->getVertices().size()) != dPlus1) continue;
    if (s->hasVertex(triVerts[1]) && s->hasVertex(triVerts[2])) {
      sharing.push_back(s);
    }
  }
  if (sharing.size() != 3) return false;

  // Collect all vertices (should be d+2)
  VertexPtrSet allVerts;
  for (const auto &s : sharing) {
    for (const auto &v : s->getVertices()) allVerts.insert(v);
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

  shiftAccepted++;
  return true;
}

bool CDT::ishift() {
  ishiftAttempts++;
  return shift();
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
  for (int i = 0; i < 100; ++i) {
    sweep();
    auto n4 = static_cast<double>(spacetime->getSimplexCount());
    double error = n4 - static_cast<double>(targetN4);
    k4 += 0.0001 * error;
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
