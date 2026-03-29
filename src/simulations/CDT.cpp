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

double CDT::computeDeltaAction(int dN0, int dN41, int dN32) const {
  double n41 = static_cast<double>(spacetime->getN41());
  double target = static_cast<double>(targetN41);

  double dRegge = -(k0 + 6.0 * delta) * dN0
               + (k4 + 2.0 * delta) * dN41
               + (k4 + delta) * dN32;
  double oldFix, newFix;
  if (quadraticVolumeFix) {
    oldFix = epsilon * (n41 - target) * (n41 - target);
    newFix = epsilon * (n41 + dN41 - target) * (n41 + dN41 - target);
  } else {
    oldFix = epsilon * std::abs(n41 - target);
    newFix = epsilon * std::abs(n41 + dN41 - target);
  }
  return dRegge + (newFix - oldFix);
}

bool CDT::accept(double deltaS, double logPrefactor) {
  double exponent = -deltaS + logPrefactor;
  if (exponent >= 0.0) return true;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < std::exp(exponent);
}

// ========================================
// (2, 2d) Add Move: vertex insertion at spatial face
// ========================================
bool CDT::add() {
  addAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  // Select a random N41-type top simplex. The prefactor N41/(N0+1) requires
  // the selection probability to be 1/N41 (Brunekreef Sec. 2.3.1).
  // We must NOT select from all N4 top simplices, as that would give
  // selection probability 1/N4, violating detailed balance.
  SimplexPtr sigma = getRandomN41Simplex(d);
  if (!sigma) return false;

  auto [sti, stf] = sigma->getOrientation().numeric();

  // Find the spatial facet: the one where all vertices are at the same time.
  // For (d,1): skip the 1 vertex at tf → spatial face has d vertices at ti.
  // For (1,d): skip the 1 vertex at ti → spatial face has d vertices at tf.
  SimplexPtr spatialFacet = nullptr;
  for (const auto &f : sigma->getFacets()) {
    if (f->isSpatial()) {
      spatialFacet = f;
      break;
    }
  }
  if (!spatialFacet) return false;

  // Find the adjacent simplex sharing this spatial facet (opposite orientation)
  SimplexPtr sigmaAdj = nullptr;
  for (const auto &cf : spatialFacet->getCofaces()) {
    if (static_cast<int>(cf->size()) == dPlus1 && cf != sigma) {
      sigmaAdj = cf;
      break;
    }
  }
  if (!sigmaAdj) return false;

  // The adjacent simplex should be N41-type (opposite orientation)
  auto [ati, atf] = sigmaAdj->getOrientation().numeric();
  bool adjIsN41 = (ati == d && atf == 1) || (ati == 1 && atf == d);
  if (!adjIsN41) return false;

  // Identify the non-spatial ("top" and "bottom") vertices
  VertexPtr vertA = nullptr, vertB = nullptr;
  for (const auto &v : sigma->getVertices()) {
    if (!spatialFacet->hasVertex(v)) { vertA = v; break; }
  }
  for (const auto &v : sigmaAdj->getVertices()) {
    if (!spatialFacet->hasVertex(v)) { vertB = v; break; }
  }
  if (!vertA || !vertB) return false;

  // Action change: dN0=+1, dN41=+(2d-2), dN32=0
  // We remove 2 N41 simplices and create 2d N41 simplices → net +(2d-2)
  int dN0 = 1;
  int dN41 = 2 * d - 2;
  int dN32 = 0;

  double deltaS = computeDeltaAction(dN0, dN41, dN32);

  // Combinatorial prefactor: N41/(N0+1)
  // From Brunekreef eq 26 pattern: selection 1/N41, relabeling 1/(N0+1),
  // reverse selection 1/(N0+1)
  double N41 = static_cast<double>(spacetime->getN41());
  double N0 = static_cast<double>(spacetime->getVertexCount());
  double logPrefactor = std::log(N41) - std::log(N0 + 1.0);

  if (!accept(deltaS, logPrefactor)) return false;

  // Execute: create new vertex at the spatial time slice
  double spatialTime = spatialFacet->getTi();  // all verts at same time
  VertexPtr newVert = spacetime->createVertex(std::vector<double>{spatialTime});

  // Save spatial vertices before removing simplices
  VertexPtrs spatialVerts(spatialFacet->getVertices().begin(),
                          spatialFacet->getVertices().end());

  // Remove the 2 old simplices
  spacetime->removeSimplex(sigma);
  spacetime->removeSimplex(sigmaAdj);

  // Create 2d new simplices: for each of d sub-faces (drop one spatial vertex, add newVert)
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs verts1, verts2;
    for (int i = 0; i < d; ++i) {
      if (i != skip) {
        verts1.push_back(spatialVerts[i]);
        verts2.push_back(spatialVerts[i]);
      }
    }
    verts1.push_back(newVert);
    verts1.push_back(vertA);
    verts2.push_back(newVert);
    verts2.push_back(vertB);

    spacetime->createSimplex(verts1);
    spacetime->createSimplex(verts2);
  }

  // Vertex relabeling ([BGL] Sec. 2.2.1/2.3.1): swap the new vertex's label
  // with a uniformly chosen existing vertex. This implements the 1/(N0+1)
  // factor in the acceptance prefactor. Observables are label-invariant, so
  // this is physically neutral but ensures correct detailed balance.
  if (relabelVertices_) {
    VertexPtr randomVert = spacetime->getRandomVertex();
    if (randomVert && randomVert->getId() != newVert->getId()) {
      spacetime->swapVertexLabels(newVert, randomVert);
    }
  }

  addAccepted++;
  return true;
}

// ========================================
// (2d, 2) Remove Move: vertex deletion (blind guessing)
// ========================================
bool CDT::remove() {
  removeAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;
  int requiredOrder = 2 * d;

  // Pick a random vertex (blind guessing, Brunekreef Sec 2.3.1)
  VertexPtr v = spacetime->getRandomVertex();
  if (!v) return false;

  // Count top simplices incident to this vertex; require exactly 2d
  std::vector<SimplexPtr> incident;
  for (const auto &s : v->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1) incident.push_back(s);
  }
  if (static_cast<int>(incident.size()) != requiredOrder) return false;

  // Verify structure: all incident simplices must be N41-type,
  // and they must share exactly d spatial vertices (besides v)
  // plus 2 non-spatial vertices (one "top", one "bottom")
  VertexPtr vertA = nullptr, vertB = nullptr;
  VertexPtrs spatialVerts;

  // Collect all vertices across incident simplices (besides v)
  std::map<std::uint64_t, VertexPtr> otherVerts;
  std::map<std::uint64_t, int> vertCounts;
  for (const auto &s : incident) {
    // All must be N41-type
    auto [sti, stf] = s->getOrientation().numeric();
    bool isN41 = (sti == d && stf == 1) || (sti == 1 && stf == d);
    if (!isN41) return false;

    for (const auto &vert : s->getVertices()) {
      if (vert->getId() == v->getId()) continue;
      otherVerts[vert->getId()] = vert;
      vertCounts[vert->getId()]++;
    }
  }

  // Should have d+2 other vertices: d spatial (each in 2(d-1) simplices) + 2 non-spatial (each in d)
  if (static_cast<int>(otherVerts.size()) != d + 2) return false;

  // Spatial vertices appear in all 2d incident simplices minus those that skip them.
  // Each spatial vertex appears in 2(d-1) of the 2d simplices.
  // The "top" and "bottom" vertices each appear in exactly d simplices.
  for (const auto &[vid, count] : vertCounts) {
    if (count == d) {
      if (!vertA) vertA = otherVerts[vid];
      else if (!vertB) vertB = otherVerts[vid];
      else return false;  // more than 2 non-spatial vertices
    } else if (count == 2 * (d - 1)) {
      spatialVerts.push_back(otherVerts[vid]);
    } else {
      return false;  // unexpected structure
    }
  }
  if (!vertA || !vertB || static_cast<int>(spatialVerts.size()) != d) return false;

  // Action change: dN0=-1, dN41=-(2d-2), dN32=0
  int dN0 = -1;
  int dN41_change = -(2 * d - 2);
  int dN32 = 0;

  double deltaS = computeDeltaAction(dN0, dN41_change, dN32);

  // Combinatorial prefactor: N0 / (N41 - (2d-2))
  // Inverse of add: reverse add from T' would select from N41'=N41-(2d-2)
  double N41 = static_cast<double>(spacetime->getN41());
  double N0 = static_cast<double>(spacetime->getVertexCount());
  double N41after = N41 + dN41_change;
  if (N41after <= 0) return false;
  double logPrefactor = std::log(N0) - std::log(N41after);

  if (!accept(deltaS, logPrefactor)) return false;

  // Execute: remove 2d simplices
  for (const auto &s : incident) {
    spacetime->removeSimplex(s);
  }

  // Remove all edges incident to v from both endpoints and the global edge list
  Edges edgesToRemove = v->getEdges();
  for (const auto &e : edgesToRemove) {
    VertexPtr other = (e->getSource()->getId() == v->getId())
                      ? e->getTarget() : e->getSource();
    other->removeOutEdge(e);
    other->removeInEdge(e);
    v->removeOutEdge(e);
    v->removeInEdge(e);
    spacetime->getEdgeList()->remove(e);
  }
  (void)spacetime->removeIfIsolated(v);

  // Create 2 replacement simplices: {spatialVerts + vertA} and {spatialVerts + vertB}
  VertexPtrs verts1(spatialVerts.begin(), spatialVerts.end());
  verts1.push_back(vertA);
  VertexPtrs verts2(spatialVerts.begin(), spatialVerts.end());
  verts2.push_back(vertB);

  spacetime->createSimplex(verts1);
  spacetime->createSimplex(verts2);

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

  // Reject if any new simplex would have a non-CDT orientation
  for (const auto &nv : newSimplexVerts) {
    if (!isValidCDTOrientation(nv, d)) return false;
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

  // Combinatorial prefactor: q(T'→T)/q(T→T') = N4/N4'.
  // Both flip and iflip proposals have total probability 2/(N×(d+1))
  // (flip: 2 simplices × 1 facet; iflip: d simplices × 1 edge ×
  // 2/(d(d+1)) = 2/(N(d+1))), so the ratio is simply N4/N4'.
  double N4 = static_cast<double>(spacetime->getSimplexCount());
  double logPrefactor = std::log(N4) - std::log(N4 + d - 2);

  if (!accept(deltaS, logPrefactor)) return false;

  spacetime->removeSimplex(s1);
  spacetime->removeSimplex(s2);
  for (const auto &nv : newSimplexVerts) {
    spacetime->createSimplex(nv);
  }

  flipAccepted++;
  return true;
}

// ========================================
// (d, 2) Inverse Flip Move
// ========================================
bool CDT::iflip() {
  iflipAttempts++;
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  // Pick a random edge of sigma
  const auto &edges = sigma->getEdges();
  if (edges.empty()) return false;
  std::uniform_int_distribution<std::size_t> edgeDist(0, edges.size() - 1);
  EdgePtr edge = edges[edgeDist(rng)];

  VertexPtr v1 = edge->getSource();
  VertexPtr v2 = edge->getTarget();

  // Find all top simplices containing both endpoints
  std::vector<SimplexPtr> sharing;
  for (const auto &s : v1->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1 && s->hasVertex(v2)) {
      sharing.push_back(s);
    }
  }
  if (static_cast<int>(sharing.size()) != d) return false;

  // Collect all vertices across the d simplices: should be d+2 total
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

  // Separate shared (the 2 edge endpoints) and unique (d vertices)
  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (v->getId() == v1->getId() || v->getId() == v2->getId()) shared.push_back(v);
    else unique.push_back(v);
  }
  if (shared.size() != 2 || static_cast<int>(unique.size()) != d) return false;

  // Manifold condition: the iflip creates two new simplices, each containing
  // all d unique vertices plus one shared vertex. Check that neither proposed
  // simplex already exists by verifying no top-simplex currently contains
  // {unique[0..d-1], shared[i]}. We check this by looking for a top-simplex
  // incident to unique[0] that contains all other unique vertices plus a shared
  // vertex but is NOT one of the d simplices we're about to remove.
  for (int i = 0; i < 2; ++i) {
    for (const auto &s : unique[0]->getSimplices()) {
      if (static_cast<int>(s->size()) != dPlus1) continue;
      // Skip the d simplices we're removing
      bool isSharing = false;
      for (const auto &sh : sharing) {
        if (s == sh) { isSharing = true; break; }
      }
      if (isSharing) continue;
      // Check if s contains all unique vertices and shared[i]
      if (!s->hasVertex(shared[i])) continue;
      bool hasAll = true;
      for (const auto &u : unique) {
        if (!s->hasVertex(u)) { hasAll = false; break; }
      }
      if (hasAll) return false; // would create duplicate simplex
    }
  }

  // Count old orientations
  int old_n41 = 0, old_n32 = 0;
  for (const auto &s : sharing) {
    auto [sti, stf] = s->getOrientation().numeric();
    if ((sti == d && stf == 1) || (sti == 1 && stf == d)) old_n41++;
    else if ((sti == d - 1 && stf == 2) || (sti == 2 && stf == d - 1)) old_n32++;
  }

  // Create 2 new simplices: each has all d unique + 1 of 2 shared
  std::vector<VertexPtrs> newSimplexVerts;
  for (int i = 0; i < 2; ++i) {
    VertexPtrs nv(unique.begin(), unique.end());
    nv.push_back(shared[i]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    newSimplexVerts.push_back(nv);
  }

  // Reject if any new simplex would have a non-CDT orientation
  for (const auto &nv : newSimplexVerts) {
    if (!isValidCDTOrientation(nv, d)) return false;
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

  // Combinatorial prefactor: q(T'→T)/q(T→T') = N4/N4'.
  // See flip() comment for derivation.
  double N4 = static_cast<double>(spacetime->getSimplexCount());
  double logPrefactor = std::log(N4) - std::log(N4 - d + 2);

  if (!accept(deltaS, logPrefactor)) return false;

  for (const auto &s : sharing) spacetime->removeSimplex(s);
  for (const auto &nv : newSimplexVerts) {
    spacetime->createSimplex(nv);
  }

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
  int d = getDim(spacetime);
  int dPlus1 = d + 1;

  SimplexPtr sigma = spacetime->getRandomTopSimplex();
  if (!sigma) return false;

  // Pick (d-1) random vertices from sigma to form a candidate (d-2)-face
  int hingeSize = d - 1;  // number of vertices in a (d-2)-simplex
  const auto &sigmaVertsRef = sigma->getVertices();
  if (static_cast<int>(sigmaVertsRef.size()) < hingeSize) return false;
  VertexPtrs sigmaVerts(sigmaVertsRef.begin(), sigmaVertsRef.end());
  std::shuffle(sigmaVerts.begin(), sigmaVerts.end(), rng);
  VertexPtrs faceVerts(sigmaVerts.begin(), sigmaVerts.begin() + hingeSize);

  // Find all d-simplices containing all (d-1) face vertices
  std::vector<SimplexPtr> sharing;
  for (const auto &s : faceVerts[0]->getSimplices()) {
    if (static_cast<int>(s->size()) != dPlus1) continue;
    bool containsAll = true;
    for (int i = 1; i < hingeSize; ++i) {
      if (!s->hasVertex(faceVerts[i])) { containsAll = false; break; }
    }
    if (containsAll) sharing.push_back(s);
  }
  if (static_cast<int>(sharing.size()) != hingeSize) return false;

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
  if (static_cast<int>(sharedVerts.size()) != hingeSize ||
      static_cast<int>(uniqueVerts.size()) != hingeSize) return false;

  // Count old orientations
  int old_n41 = 0, old_n32 = 0;
  for (const auto &s : sharing) {
    auto [sti, stf] = s->getOrientation().numeric();
    if ((sti == d && stf == 1) || (sti == 1 && stf == d)) old_n41++;
    else if ((sti == d - 1 && stf == 2) || (sti == 2 && stf == d - 1)) old_n32++;
  }

  // (d-1,d-1) move: each new simplex has all unique + (hingeSize-1) shared
  std::vector<VertexPtrs> newSimplexVerts;
  for (int skip = 0; skip < hingeSize; ++skip) {
    VertexPtrs nv;
    for (int i = 0; i < hingeSize; ++i) {
      if (i != skip) nv.push_back(sharedVerts[i]);
    }
    for (const auto &u : uniqueVerts) nv.push_back(u);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    newSimplexVerts.push_back(nv);
  }

  // Reject if any new simplex would have a non-CDT orientation
  for (const auto &nv : newSimplexVerts) {
    if (!isValidCDTOrientation(nv, d)) return false;
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

  // (3,3) is self-inverse with symmetric selection: no prefactor
  if (!accept(deltaS, 0.0)) return false;

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
      case 3: result = iflip(); break;
      case 4: result = shift(); break;
    }
    if (result) accepted++;
  }
  return accepted;
}

void CDT::tune() {
  // Tune k4 to its pseudo-critical value using proportional feedback.
  // For the (2,2d) add move: dS_Regge ≈ -(k0+6Δ) + (2d-2)(k4+2Δ)
  // Setting this near 0 for d=4: k4_crit ≈ (k0+6Δ)/(2d-2) - 2Δ
  int d = getDim(spacetime);
  if (d <= 1) return;  // CDT requires d >= 2
  k4 = (k0 + 6.0 * delta) / (2.0 * d - 2.0) - 2.0 * delta;

  // Fine-tune with short feedback sweeps
  double target = static_cast<double>(targetN41);
  for (int i = 0; i < 20; ++i) {
    sweep();
    double n41 = static_cast<double>(spacetime->getN41());
    double error = (n41 - target) / target;  // normalized error
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

} // caset
