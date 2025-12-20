//
// Created by andrew on 12/16/25.
//

#include "../include/SimplexGluer.h"
#include "Simplex.h"
#include "spacetime/Spacetime.h"

#include <algorithm>

namespace caset {
SimplexGluer::SimplexGluer(
  Spacetime *spacetime_,
  const VertexPtrs &unattachedVertices_
  ) : spacetime(spacetime_), unattachedVertices(unattachedVertices_) {
  for (const auto &unattached : unattachedVertices_) {
    stageVertex(unattached);
  }
  breakReferences();
}

void SimplexGluer::stageVertex(const VertexPtr &unattached) {
  const auto &unattachedSimplices = unattached->getSimplices();
  simplicesToProcessAsSet.insert(unattachedSimplices.begin(), unattachedSimplices.end());
}

void SimplexGluer::breakReferences() {
  // TODO: The unattached vertex belongs to a facet on a coface. We need to ensure those facets/cofaces are all replaced
  //  by those on the spacetime rather than accidentally duplicating them by replacing vertices such that they collide
  //  with existing simplices.
  simplicesToProcess.assign(simplicesToProcessAsSet.begin(), simplicesToProcessAsSet.end());

  // Sort simplices to Process by dimension(k), ascending.
  std::sort(simplicesToProcess.begin(),
            simplicesToProcess.end(),
            [](const SimplexPtr &a, const SimplexPtr &b) noexcept {
              return a->getOrientation().getK() < b->getOrientation().getK();
            });

  // Need to dereference everything, and store what was dereferenced.
  brokenReferences.reserve(simplicesToProcess.size());

  for (const auto &simplex : simplicesToProcess) {
    brokenReferences.push_back(Simplex::breakReferences(simplex));
  }
}

SimplexGluer::~SimplexGluer() {
  for (const auto &[simplex, brokenCofaces, brokenFacets] : brokenReferences) {
    // TODO: May need to check here whether or not the simplex is internal or external. I'm pretty sure this will always
    //  be internal as long as attach() is only used to attach previously unattached simplexes.

    auto registeredSimplex = spacetime->registerSimplex(simplex, !simplex->isCausallyAvailableFace());
    CLOG(DEBUG_LEVEL, "RE-Registering ", simplex->toString(), "to vertices and facets...");

    Simplices registeredBrokenCofaces{};
    registeredBrokenCofaces.reserve(brokenCofaces.size());
    for (const auto &bcf : brokenCofaces) {
      auto registeredCoface = spacetime->registerSimplex(bcf, bcf->isCausallyAvailableFace());
      registeredBrokenCofaces.push_back(registeredCoface);
    }

    Simplices registeredBrokenFacets{};
    registeredBrokenFacets.reserve(brokenFacets.size());
    for (const auto &bf : brokenFacets) {
      auto registeredFacet = spacetime->registerSimplex(bf, bf->isCausallyAvailableFace());
      registeredBrokenFacets.push_back(registeredFacet);
    }

    Simplex::restoreReferences(registeredSimplex, registeredBrokenCofaces, registeredBrokenFacets);
  }
}


const Simplices &SimplexGluer::getSimplicesToProcess() const noexcept {
  return simplicesToProcess;
}

} // caset