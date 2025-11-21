#include "Simplex.h"

namespace caset {
std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> Simplex::facetRegistry = {};

std::vector<std::shared_ptr<Simplex> > Simplex::getFacets() noexcept {
  if (!facets.empty()) return facets;
  auto verts = getVertices();
  facets.reserve(verts.size());
  for (int skip = 0; skip < verts.size(); skip++) {
    std::vector<std::shared_ptr<Vertex> > faceVertices;
    faceVertices.reserve(verts.size());
    faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
    faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
    std::shared_ptr<Simplex> facet = std::make_shared<Simplex>(faceVertices);
    if (facetRegistry.contains(facet->fingerprint.fingerprint())) {
      auto found = *facetRegistry.find(facet->fingerprint.fingerprint());
      found->addCoface(shared_from_this());
      facets.push_back(found);
    } else {
      facet->addCoface(shared_from_this());
      facetRegistry.insert(facet);
      facets.push_back(facet);
    }
  }
  return facets;
}
}
